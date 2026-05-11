"""
run_extract_signals.py — Collect hidden states, attention, and input gradients in one pass.

For each prompt, extracts:
  1. Last-layer mean-pooled hidden state (geometry signal, same as run_extract.py)
  2. Attention entropy per layer (mean entropy of last-token attention over input tokens)
  3. Full attention pattern of last token at each layer (n_layers × n_heads × n_input_tokens)
  4. Input token gradient norms (∂own_dist / ∂input_embeddings, requires backward pass)
  5. [--all-layers] All-layer hidden states + per-layer own_dist (for mismatch signal)

Two-phase design:
  Phase 1: forward-only pass over all A samples → compute A-centroid per layer
  Phase 2: forward+backward pass over all prompts → collect all signals

Output (in --run-dir):
  signals/
    reps_last_raw.npy          (N, D)                         last-layer hidden states
    attn_entropy.npy           (N, n_layers)                  attention entropy per layer
    attn_pattern_last.npy      (N, n_layers, n_heads, T_max)  last-token attention (zero-padded)
    input_grad_norms.npy       (N, T_max)                     gradient norm per input token
    token_lengths.npy          (N,)                           actual sequence length per prompt
    meta.jsonl                                                 prompt metadata
    [--all-layers only]
    reps_all_layers.npy        (N, n_layers, D)               all-layer hidden states
    own_dist_all_layers.npy    (N, n_layers)                  per-layer own_dist

Usage:
  cd /path/to/repo
  python scripts/run_extract_signals.py --model llama --prompts data/math_raw_pilot.jsonl \\
      --run-dir experiments/signals/math50_llama --forms MATH

  # With all-layer hidden states (for mismatch signal):
  python scripts/run_extract_signals.py --model llama --prompts data/workshop/math50_fact10.jsonl \\
      --run-dir experiments/signals/math50_llama_allL --forms MATH --all-layers --no-gradients
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── model paths ──────────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "llama": os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-HF"),
    "qwen":  os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral_small": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi3":    "microsoft/Phi-3-mini-4k-instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "gemma2":  "google/gemma-2-2b-it",
    # Mistral-Small-3.x successor pair: 3.2-Instruct is a minor update of
    # 3.1-Instruct; HF model tree lists 3.1-24B-Base-2503 as their shared base.
    "mistral_small_3_2":      "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistral_small_3_1_base": "mistralai/Mistral-Small-3.1-24B-Base-2503",
    # Gemma-3 multimodal-capable (text-only path used here).
    "gemma3_4b":  "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
}

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B-Instruct",
    "qwen":    "Qwen2.5-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
    "mistral_small":  "Mistral-Small-24B-Instruct",
    "qwen14b":  "Qwen2.5-14B-Instruct",
    "phi3":    "Phi-3-mini-4k-instruct",
    "smollm2": "SmolLM2-1.7B-Instruct",
    "gemma2":  "Gemma-2-2b-it",
    "mistral_small_3_2":      "Mistral-Small-3.2-24B-Instruct",
    "mistral_small_3_1_base": "Mistral-Small-3.1-24B-Base (base)",
    "gemma3_4b":  "Gemma-3-4b-it",
    "gemma3_12b": "Gemma-3-12b-it",
}


# ── helpers ──────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts(path, forms=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if forms is None or row.get("form") in forms:
                    rows.append(row)
    return rows


def normalise(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def compute_centroid(vecs):
    return np.mean(vecs, axis=0)


def own_dist(vec, centroid):
    """Cosine distance from vec to centroid."""
    return float(1.0 - np.dot(normalise(vec), normalise(centroid)))


def attn_entropy_last_token(attn_weights):
    """
    attn_weights: list of (1, n_heads, T, T) tensors, one per layer.
    Returns array of shape (n_layers,): mean entropy of last-token attention distribution.
    """
    entropies = []
    for layer_attn in attn_weights:
        # layer_attn: (1, n_heads, T, T)
        last_row = layer_attn[0, :, -1, :]       # (n_heads, T)
        last_row = last_row.float()
        # Clamp for numerical stability
        last_row = last_row.clamp(min=1e-12)
        last_row = last_row / last_row.sum(dim=-1, keepdim=True)
        ent = -(last_row * last_row.log()).sum(dim=-1)  # (n_heads,)
        entropies.append(ent.mean().item())
    return np.array(entropies, dtype=np.float32)


def attn_pattern_last_token(attn_weights, max_len):
    """
    Returns array (n_layers, n_heads, max_len): last-token attention over input tokens,
    zero-padded to max_len.
    """
    result = []
    for layer_attn in attn_weights:
        last_row = layer_attn[0, :, -1, :].float().detach().cpu().numpy()  # (n_heads, T)
        T = last_row.shape[-1]
        padded = np.zeros((last_row.shape[0], max_len), dtype=np.float32)
        padded[:, :T] = last_row
        result.append(padded)
    return np.stack(result, axis=0)  # (n_layers, n_heads, max_len)


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect hidden states, attention, and input gradients in one pass."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL.")
    parser.add_argument("--run-dir", required=True, help="Output directory.")
    parser.add_argument("--forms", nargs="+", default=None,
                        help="Filter to these forms (e.g. MATH CODE). Default: all.")
    parser.add_argument("--no-gradients", action="store_true",
                        help="Skip gradient computation (faster, attention only).")
    parser.add_argument("--all-layers", action="store_true",
                        help="Save all-layer hidden states and per-layer own_dist "
                             "(needed for attn/geometry mismatch signal). Implies --no-gradients.")
    args = parser.parse_args()
    if args.all_layers:
        args.no_gradients = True  # all-layers mode skips backward pass

    out_dir = os.path.join(args.run_dir, "signals")
    os.makedirs(out_dir, exist_ok=True)

    device     = get_device()
    model_id   = MODEL_PATHS[args.model]
    model_name = MODEL_LABELS[args.model]

    print(f"Model   : {model_name}")
    print(f"Device  : {device}")
    print(f"Run dir : {args.run_dir}")
    print(f"Gradients: {'disabled' if args.no_gradients else 'enabled'}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen/Gemma2 trained in bfloat16; float16 causes NaN on MPS.
    # Mistral-Small (2501 + 3.x) is bf16-native — keep in bf16 to match other
    # Mistral-Small handling in run_extract_minimal.py / compare_*.py.
    dtype = torch.bfloat16 if args.model in (
        "qwen", "smollm2", "gemma2", "qwen14b",
        "mistral_small", "mistral_small_3_2", "mistral_small_3_1_base",
        "gemma3_4b", "gemma3_12b",
    ) else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager"
        )
    except ValueError as e:
        err = str(e)
        if "Mistral3" in err:
            from transformers import Mistral3ForConditionalGeneration
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager"
            )
        elif "Gemma3" in err:
            from transformers import Gemma3ForConditionalGeneration
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager"
            )
        elif "Unrecognized configuration class" in err:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager"
            )
        else:
            raise
    model = model.to(device)
    model.eval()

    prompts = load_prompts(args.prompts, forms=set(args.forms) if args.forms else None)
    print(f"Loaded  : {len(prompts)} prompts "
          f"(A={sum(1 for r in prompts if r['answerable']=='A')}, "
          f"U={sum(1 for r in prompts if r['answerable']=='U')})\n")

    # ── Phase 1: compute A-centroid (last layer, always; all layers if --all-layers) ──
    print("Phase 1: computing A-centroid(s)...")
    a_vecs      = []   # last-layer vecs
    a_vecs_all  = []   # all-layer vecs, only populated when --all-layers
    for row in prompts:
        if row["answerable"] != "A":
            continue
        inputs = tokenizer(row["prompt"], return_tensors="pt", padding=False).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        vec = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu().numpy()
        a_vecs.append(vec)
        if args.all_layers:
            # hidden_states: tuple of (1, T, D) per layer (including embedding layer at index 0)
            # We use layers 1.. (transformer layers only, skip embedding layer)
            layer_vecs = np.stack([
                hs.mean(dim=1).squeeze(0).float().cpu().numpy()
                for hs in out.hidden_states[1:]
            ], axis=0)  # (n_layers, D)
            a_vecs_all.append(layer_vecs)

    centroid_A = compute_centroid(np.stack(a_vecs, axis=0))
    if args.all_layers:
        # centroids_A_all: (n_layers, D)
        centroids_A_all = np.stack(a_vecs_all, axis=0).mean(axis=0)  # mean over A samples
        n_layers = centroids_A_all.shape[0]
        print(f"  A-centroid computed from {len(a_vecs)} samples, {n_layers} layers.\n")
    else:
        centroids_A_all = None
        print(f"  A-centroid computed from {len(a_vecs)} samples.\n")

    # ── Phase 2: forward + (optional) backward for all prompts ───────────────────
    print("Phase 2: extracting signals...")

    all_reps          = []
    all_attn_ent      = []
    all_attn_pat      = []
    all_grad_norms    = []
    all_tok_lengths   = []
    all_reps_all_L    = []   # (n_layers, D) per prompt, only if --all-layers (mean-pooled)
    all_reps_lt_all_L = []   # (n_layers, D) per prompt, only if --all-layers (last-token)
    all_od_all_L      = []   # (n_layers,)   per prompt, only if --all-layers
    meta_rows         = []

    # Determine max token length for padding
    max_len = 0
    for row in prompts:
        ids = tokenizer(row["prompt"], return_tensors="pt").input_ids
        max_len = max(max_len, ids.shape[-1])

    for idx, row in enumerate(prompts):
        pid, form, answerable, prompt = (
            row["id"], row["form"], row["answerable"], row["prompt"]
        )

        if args.no_gradients:
            # Forward only
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
            with torch.no_grad():
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                )
            rep = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu().numpy()
            ent = attn_entropy_last_token(out.attentions)
            pat = attn_pattern_last_token(out.attentions, max_len)
            grad_norms = np.zeros(max_len, dtype=np.float32)
            T = inputs["input_ids"].shape[-1]

            if args.all_layers:
                # All transformer layers (skip index 0 = embedding layer)
                # Mean-pooled reps
                layer_vecs = np.stack([
                    hs.mean(dim=1).squeeze(0).float().cpu().numpy()
                    for hs in out.hidden_states[1:]
                ], axis=0)  # (n_layers, D)
                # Last-token reps (for matched-pooling orthogonality comparison)
                layer_vecs_lt = np.stack([
                    hs[0, -1, :].float().cpu().numpy()
                    for hs in out.hidden_states[1:]
                ], axis=0)  # (n_layers, D)
                od_per_layer = np.array([
                    own_dist(layer_vecs[l], centroids_A_all[l])
                    for l in range(layer_vecs.shape[0])
                ], dtype=np.float32)
                all_reps_all_L.append(layer_vecs)
                all_reps_lt_all_L.append(layer_vecs_lt)
                all_od_all_L.append(od_per_layer)

        else:
            # Forward + backward to get input gradients
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
            T = inputs["input_ids"].shape[-1]

            # Get input embeddings and enable grad
            embed_layer = model.get_input_embeddings()
            input_ids = inputs["input_ids"]
            embeds = embed_layer(input_ids).detach().requires_grad_(True)

            # Forward pass through model using inputs_embeds
            out = model(
                inputs_embeds=embeds,
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )

            rep = out.hidden_states[-1].mean(dim=1).squeeze(0).float().detach().cpu().numpy()
            ent = attn_entropy_last_token(out.attentions)
            pat = attn_pattern_last_token(out.attentions, max_len)

            # Backward: scalar = cosine distance from rep to A-centroid
            centroid_t = torch.tensor(centroid_A, dtype=torch.float32, device=device)
            rep_t      = out.hidden_states[-1].mean(dim=1).squeeze(0).float()
            scalar     = 1.0 - torch.nn.functional.cosine_similarity(
                rep_t.unsqueeze(0), centroid_t.unsqueeze(0)
            )
            scalar.backward()

            # Gradient norm per input token
            grad = embeds.grad  # (1, T, D)
            grad_norms_raw = grad[0].float().norm(dim=-1).cpu().numpy()  # (T,)
            grad_norms = np.zeros(max_len, dtype=np.float32)
            grad_norms[:T] = grad_norms_raw

            model.zero_grad()

        all_reps.append(rep)
        all_attn_ent.append(ent)
        all_attn_pat.append(pat)
        all_grad_norms.append(grad_norms)
        all_tok_lengths.append(T)
        meta_rows.append({
            "row_idx": idx, "id": pid, "form": form,
            "answerable": answerable, "prompt": prompt,
        })

        print(f"  [{idx+1:>3}/{len(prompts)}] {pid} ({form}/{answerable}) "
              f"T={T} own_dist={own_dist(rep, centroid_A):.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, "reps_last_raw.npy"),
            np.stack(all_reps, axis=0).astype(np.float32))
    np.save(os.path.join(out_dir, "attn_entropy.npy"),
            np.stack(all_attn_ent, axis=0).astype(np.float32))
    np.save(os.path.join(out_dir, "attn_pattern_last.npy"),
            np.stack(all_attn_pat, axis=0).astype(np.float32))
    np.save(os.path.join(out_dir, "input_grad_norms.npy"),
            np.stack(all_grad_norms, axis=0).astype(np.float32))
    np.save(os.path.join(out_dir, "token_lengths.npy"),
            np.array(all_tok_lengths, dtype=np.int32))

    if args.all_layers and all_reps_all_L:
        np.save(os.path.join(out_dir, "reps_all_layers.npy"),
                np.stack(all_reps_all_L, axis=0).astype(np.float32))
        if all_reps_lt_all_L:
            np.save(os.path.join(out_dir, "reps_last_token_all_layers.npy"),
                    np.stack(all_reps_lt_all_L, axis=0).astype(np.float32))
        np.save(os.path.join(out_dir, "own_dist_all_layers.npy"),
                np.stack(all_od_all_L, axis=0).astype(np.float32))

    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved to {out_dir}/")
    print(f"  reps_last_raw    : {np.stack(all_reps).shape}")
    print(f"  attn_entropy     : {np.stack(all_attn_ent).shape}")
    print(f"  attn_pattern_last: {np.stack(all_attn_pat).shape}")
    print(f"  input_grad_norms : {np.stack(all_grad_norms).shape}")
    if args.all_layers and all_reps_all_L:
        print(f"  reps_all_layers  : {np.stack(all_reps_all_L).shape}")
        if all_reps_lt_all_L:
            print(f"  reps_lt_all_L    : {np.stack(all_reps_lt_all_L).shape}")
        print(f"  own_dist_all_L   : {np.stack(all_od_all_L).shape}")


if __name__ == "__main__":
    main()
