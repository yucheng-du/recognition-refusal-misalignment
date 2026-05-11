"""
run_extract_minimal.py — Minimal hidden-state extraction for EMNLP 2026 main pipeline.

⚠️ SCOPE: This script is the MINIMAL variant. It only produces the three files
that the main-line eval scripts consume:
   - reps_all_layers.npy              (mean-pooled, all transformer layers)
   - reps_last_token_all_layers.npy   (last-token, all transformer layers)
   - meta.jsonl

It INTENTIONALLY does NOT produce:
   - attn_entropy.npy / attn_pattern_last.npy / input_grad_norms.npy /
     token_lengths.npy / reps_last_raw.npy / own_dist_all_layers.npy

If you need any of the above (e.g., to run
`experiments/token_attribution/run_attribution.py`,
`experiments/analyze_new_models.py`, or
`src/baselines/semantic_entropy.py`), use the full `run_extract_signals.py`
instead — those scripts depend on attention / gradient / own_dist outputs that
this minimal variant skips.

Compared to run_extract_signals.py, this saves ~40-50% of extraction time by:
  • skipping Phase 1 (A-centroid pre-scan — only needed for own_dist)
  • setting output_attentions=False (attention save is the largest cost)
  • skipping all unused np.save() calls

Layer indexing convention:
  hidden_states[0]   = embedding layer (skipped)
  hidden_states[k+1] = output of model.layers[k]
  Saved reps[:, k, :] = output of model.layers[k]  (k = 0 .. n_transformer_layers - 1)

Flags kept for backwards compatibility with extract_noncore_datasets.sh:
  --all-layers, --no-gradients : now implicit / no-op

Usage:
  python scripts/run_extract_minimal.py --model mistral \\
      --prompts data/abstentionbench_gsm8k.jsonl \\
      --run-dir experiments/signals/abstentionbench_gsm8k_mistral_allL \\
      --forms MATH --all-layers --no-gradients
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
    "olmo13b": "allenai/OLMo-2-1124-13B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    # ── 2025-dense-successor upgrades (Phase 1: Mac-tier; see drafts/mac_run_2026-04-29.md) ──
    "phi4mini":   "microsoft/Phi-4-mini-instruct",
    "gemma3_4b":  "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "qwen3_8b":   "Qwen/Qwen3-8B",
    "qwen3_14b":  "Qwen/Qwen3-14B",
    # ── Phase 2 / 3: RunPod (32B on Pod A; 70B on Pod B) ──
    "qwen3_32b":         "Qwen/Qwen3-32B",
    "llama70b":          "meta-llama/Llama-3.3-70B-Instruct",   # gated; HF agreement required
    # ── Base (pretraining-only, no RLHF) models for F-Base pilot ─────────────
    # Used to test whether orthogonality is RLHF-created or pretraining-level.
    "llama_base":    "meta-llama/Llama-3.1-8B",
    "mistral_base":  "mistralai/Mistral-7B-v0.3",
    "qwen_base":     "Qwen/Qwen2.5-7B",
    "qwen14b_base":  "Qwen/Qwen2.5-14B",
    "gemma2_base":   "google/gemma-2-2b",
    "smollm2_base":  "HuggingFaceTB/SmolLM2-1.7B",
    "mistral_small_base": "mistralai/Mistral-Small-24B-Base-2501",
    "olmo13b_base": "allenai/OLMo-2-1124-13B",
    "qwen32b_base": "Qwen/Qwen2.5-32B",
    # ── Phase 1 base variants (Phi-4-mini has no base release — Microsoft policy) ──
    "gemma3_4b_base": "google/gemma-3-4b-pt",
    "gemma3_12b_base": "google/gemma-3-12b-pt",
    "qwen3_8b_base":  "Qwen/Qwen3-8B-Base",
    "qwen3_14b_base": "Qwen/Qwen3-14B-Base",
    # ── Phase 2 base variants ──
    # Note: Qwen3 dense base lineup ends at 14B; Alibaba did not release
    # Qwen3-32B-Base. The 32B base/instruct verified pair role in §3.5 is
    # filled by Qwen2.5-32B (qwen32b / qwen32b_base above). No qwen3_32b_base
    # registry entry — see drafts/qwen3_32b_postrun_assessment.md §4.
    # Llama-3.3 has no base release; use Llama-3.1-70B (Meta confirmed 3.3 reuses 3.1
    # pretraining checkpoint, post-training-only update). §7 footnote.
    "llama70b_base":  "meta-llama/Llama-3.1-70B",   # gated; HF agreement required
    # Mistral-Small-3.x successor pair: 3.2-Instruct is a minor update of
    # 3.1-Instruct; HF model tree lists 3.1-24B-Base-2503 as their shared base.
    "mistral_small_3_2":      "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistral_small_3_1_base": "mistralai/Mistral-Small-3.1-24B-Base-2503",
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
    "olmo13b": "OLMo-2-13B-Instruct",
    "qwen32b": "Qwen2.5-32B-Instruct",
    "phi4mini":   "Phi-4-mini-instruct",
    "gemma3_4b":  "Gemma-3-4b-it",
    "gemma3_12b": "Gemma-3-12b-it",
    "qwen3_8b":   "Qwen3-8B",
    "qwen3_14b":  "Qwen3-14B",
    "qwen3_32b":         "Qwen3-32B",
    "llama70b":          "Llama-3.3-70B-Instruct",
    "llama_base":   "Llama-3.1-8B (base)",
    "mistral_base": "Mistral-7B-v0.3 (base)",
    "qwen_base":    "Qwen2.5-7B (base)",
    "qwen14b_base": "Qwen2.5-14B (base)",
    "gemma2_base":  "Gemma-2-2b (base)",
    "smollm2_base": "SmolLM2-1.7B (base)",
    "mistral_small_base": "Mistral-Small-24B-Base (base)",
    "olmo13b_base": "OLMo-2-13B (base)",
    "qwen32b_base": "Qwen2.5-32B (base)",
    "gemma3_4b_base": "Gemma-3-4b (base)",
    "gemma3_12b_base": "Gemma-3-12b (base)",
    "qwen3_8b_base":  "Qwen3-8B (base)",
    "qwen3_14b_base": "Qwen3-14B (base)",
    "llama70b_base":  "Llama-3.1-70B (base; pretraining-paired with Llama-3.3-70B-Instruct)",
    "mistral_small_3_2":      "Mistral-Small-3.2-24B-Instruct",
    "mistral_small_3_1_base": "Mistral-Small-3.1-24B-Base (base)",
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


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Minimal hidden-state extraction: mean-pooled + last-token reps."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL.")
    parser.add_argument("--run-dir", required=True, help="Output directory.")
    parser.add_argument("--forms", nargs="+", default=None,
                        help="Filter to these forms (e.g. MATH CODE). Default: all.")
    # Kept for backwards compatibility (scripts already pass these); now no-ops.
    parser.add_argument("--no-gradients", action="store_true",
                        help="[deprecated/no-op] kept for backwards compat; "
                             "gradients are never computed.")
    parser.add_argument("--all-layers", action="store_true",
                        help="[deprecated/no-op] kept for backwards compat; "
                             "all transformer-layer reps are always saved.")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "signals")
    os.makedirs(out_dir, exist_ok=True)

    device     = get_device()
    model_id   = MODEL_PATHS[args.model]
    model_name = MODEL_LABELS[args.model]

    print(f"Model   : {model_name}")
    print(f"Device  : {device}")
    print(f"Run dir : {args.run_dir}")
    print(f"Output  : reps_all_layers.npy, reps_last_token_all_layers.npy, meta.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Models trained in bfloat16; float16 causes NaN on MPS for them.
    # Mistral-Small-24B is natively bf16, keep it consistent with compare_*.py
    # Base versions share the same dtype as their Instruct siblings.
    _bf16_models = (
        "qwen", "smollm2", "gemma2", "qwen14b", "mistral_small",
        "qwen_base", "qwen14b_base", "smollm2_base", "gemma2_base",
        "mistral_small_base", "olmo13b", "olmo13b_base",
        "qwen32b", "qwen32b_base",
        # Phase 1 upgrades — all bf16-native. phi4mini stays fp16 (mirrors
        # phi3 default; revisit if NaN observed during pre-flight).
        "gemma3_4b", "gemma3_4b_base",
        "gemma3_12b", "gemma3_12b_base",
        "qwen3_8b", "qwen3_8b_base",
        "qwen3_14b", "qwen3_14b_base",
        # Phase 2 upgrades — bf16-native (Qwen3 32B; instruct-only, no base release).
        "qwen3_32b",
        # Phase 3: 70B (bf16-native; Llama family).
        "llama70b", "llama70b_base",
        # Mistral-Small-3.x successor pair (Phase 0 patch). bf16-native, same
        # as Mistral-Small-2501 24B handling above.
        "mistral_small_3_2", "mistral_small_3_1_base",
    )

    # ── Models that require multi-GPU sharding (device_map="auto") ───────────
    # 70B in bf16 needs ~141GB VRAM; doesn't fit single A100-80GB. Pod B is
    # 2× A100 80GB SXM = 160GB total, sharded via device_map="auto".
    # If a model is in this tuple AND we're on CUDA, load with device_map="auto"
    # and skip the explicit `.to(device)` step (sharding pre-places weights).
    _shard_models = (
        "llama70b", "llama70b_base",
    )
    dtype = torch.bfloat16 if args.model in _bf16_models else torch.float16
    use_shard = (args.model in _shard_models) and (device.type == "cuda")
    if use_shard:
        # Multi-GPU sharding for 70B-class. device_map="auto" pre-places weights;
        # do NOT call .to(device) afterwards (would error / move shards).
        print(f"  Loading {args.model} with device_map='auto' (multi-GPU sharding)")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager", device_map="auto"
            )
        except ValueError as e:
            err = str(e)
            if "Mistral3" in err:
                from transformers import Mistral3ForConditionalGeneration
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_id, dtype=dtype, attn_implementation="eager", device_map="auto"
                )
            elif "Gemma3" in err:
                from transformers import Gemma3ForConditionalGeneration
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id, dtype=dtype, attn_implementation="eager", device_map="auto"
                )
            elif "Unrecognized configuration class" in err:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id, dtype=dtype, attn_implementation="eager", device_map="auto"
                )
            else:
                raise
    else:
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
    n_a = sum(1 for r in prompts if r["answerable"] == "A")
    n_u = sum(1 for r in prompts if r["answerable"] == "U")
    print(f"Loaded  : {len(prompts)} prompts (A={n_a}, U={n_u})\n")

    # ── Extract hidden states ────────────────────────────────────────────────────
    print("Extracting hidden states (single forward pass per prompt)...")

    all_reps_all_L    = []   # (n_layers, D) per prompt — mean-pooled
    all_reps_lt_all_L = []   # (n_layers, D) per prompt — last-token
    meta_rows         = []

    for idx, row in enumerate(prompts):
        pid, form, answerable, prompt = (
            row["id"], row["form"], row["answerable"], row["prompt"]
        )

        inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        T = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,  # attention not used downstream → skip
                use_cache=False,
            )

        # hidden_states: tuple of (1, T, D) per layer (index 0 = embedding, skip it)
        # layer k output ↔ hidden_states[k+1]
        hs_layers = out.hidden_states[1:]  # all transformer layers

        layer_vecs_mp = np.stack([
            hs.mean(dim=1).squeeze(0).float().cpu().numpy() for hs in hs_layers
        ], axis=0)  # (n_layers, D)

        layer_vecs_lt = np.stack([
            hs[0, -1, :].float().cpu().numpy() for hs in hs_layers
        ], axis=0)  # (n_layers, D)

        all_reps_all_L.append(layer_vecs_mp)
        all_reps_lt_all_L.append(layer_vecs_lt)
        meta_rows.append({
            "row_idx": idx, "id": pid, "form": form,
            "answerable": answerable, "prompt": prompt,
        })

        print(f"  [{idx+1:>4}/{len(prompts)}] {pid} ({form}/{answerable}) T={T}")

    # ── Save ─────────────────────────────────────────────────────────────────────
    # Save reps_last_token_all_layers.npy FIRST so a crash mid-save still leaves
    # the resume-safety marker (meta.jsonl) in its pre-completion state.
    mp_arr = np.stack(all_reps_all_L, axis=0).astype(np.float32)
    lt_arr = np.stack(all_reps_lt_all_L, axis=0).astype(np.float32)

    np.save(os.path.join(out_dir, "reps_all_layers.npy"), mp_arr)
    np.save(os.path.join(out_dir, "reps_last_token_all_layers.npy"), lt_arr)

    # meta.jsonl must be written LAST — the extraction-resume logic treats
    # (reps_last_token_all_layers.npy + meta.jsonl) both-present as "done".
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Saved to {out_dir}/")
    print(f"  reps_all_layers            : {mp_arr.shape}")
    print(f"  reps_last_token_all_layers : {lt_arr.shape}")
    print(f"  meta.jsonl                 : {len(meta_rows)} rows")


if __name__ == "__main__":
    main()
