"""
Directional Intervention in Null-Space: Causal Evidence for Answerability Signal.

Instead of swapping entire hidden states (activation patching, which produces
degenerate outputs), we add/subtract the *null-space signal direction* at a
specific layer — a targeted, low-rank perturbation.

⚠ Protocol nature: generation-time activation steering, NOT a one-shot patch.
    Because the forward-hook is registered on model.layers[L] during the
    *entire* model.generate() call, the perturbation α·d̂ is applied on every
    forward pass — including every autoregressive step — at the last-token
    position of that pass. So the hidden state is nudged at (a) the prompt's
    final token on the first forward, and (b) every newly generated token
    thereafter. This is the same protocol used by activation-steering work
    (Turner et al. 2023, Li et al. 2023) and should be described as such.

Protocol:
  1. Precompute: PCA on A-class reps at layer L → null-space direction d̂
     (mean-diff of A vs U residuals, same as CosNSRT)
  2. For each U sample at layer L:
       x_patched = x − α · d̂   (remove U-signal → should flip to "attempt")
  3. For each A sample at layer L:
       x_patched = x + α · d̂   (inject U-signal → should flip to "refusal")
  4. Measure: flip rate (refusal ↔ attempt) at various α strengths.

Flip-rate reporting: we report BOTH
  - rate_signal          = n_flipped / n_valid          (unconditional)
  - rate_signal_gated    = n_flipped / n_clean_gate     (conditional on the
                           clean baseline already being in the expected class
                           for this condition; this is the rate on the subset
                           that can *in principle* be flipped by steering)
  - clean_refusal_rate   = fraction of samples where clean output is a refusal
Gated rates matter for models whose clean baseline is not saturated in the
expected class (e.g. Llama, which does not reliably refuse U-class math
prompts), since the unconditional rate is then ceiling-limited.

Control: intervene along a RANDOM direction of the same norm → should NOT flip.

This directly tests the causal role of our discovered null-space direction,
not just correlational detectability.

Requirements: transformers, torch, numpy, scipy
Runs on Apple M1 Max 64GB (MPS backend) for 7B models.

Usage:
  python scripts/intervention_nullspace.py --model mistral --dataset math800 \
      --layer 15 --n-samples 50 --alphas 0,5,10,20,40
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from numpy.linalg import norm

# ── Model paths ──────────────────────────────────────────────────────
MODEL_PATHS = {
    "llama":   os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-HF"),
    "qwen":    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    # §4.4 intervention re-run (Option 2 / 2026-04-29) — replaces qwen + qwen14b.
    "qwen3_8b":  "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    # Phase 3: 70B (RunPod 2× A100 80GB SXM, device_map="auto").
    "llama70b":  "meta-llama/Llama-3.3-70B-Instruct",
    # Mistral-Small-3.x successor pair: 3.2-Instruct is a minor update of
    # 3.1-Instruct; HF model tree lists 3.1-24B-Base-2503 as their shared base.
    "mistral_small_3_2":      "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistral_small_3_1_base": "mistralai/Mistral-Small-3.1-24B-Base-2503",
    # Gemma-3-4B (multimodal-capable; we use text-only path).
    "gemma3_4b": "google/gemma-3-4b-it",
    # Gemma-3-12B (multimodal-capable; text-only path; 48 transformer layers).
    "gemma3_12b": "google/gemma-3-12b-it",
}
BEST_LAYERS = {  # 0-indexed: reps[:, k, :] = output of model.layers[k]
    ("math800", "mistral"): 15, ("math800", "llama"): 15, ("math800", "qwen"): 18,
    ("code800", "mistral"): 15, ("code800", "llama"): 14, ("code800", "qwen"): 18,
    ("fact800", "mistral"): 17, ("fact800", "llama"): 15, ("fact800", "qwen"): 19,
    ("math800", "qwen14b"): 34, ("code800", "qwen14b"): 32, ("fact800", "qwen14b"): 34,
    ("abstentionbench_gsm8k", "mistral"): 12, ("abstentionbench_gsm8k", "llama"): 10, ("abstentionbench_gsm8k", "qwen"): 18,
    ("mathtrap", "mistral"): 11, ("mathtrap", "llama"): 11,
    ("falseqa", "mistral"): 18, ("falseqa", "llama"): 14,
    # Layers inherited from Mistral-Small-2501 24B layout; re-validate by
    # layer scan if the first runs look off.
    ("math800", "mistral_small_3_2"): 28,
    ("code800", "mistral_small_3_2"): 20,
    ("fact800", "mistral_small_3_2"): 28,
    ("math800", "mistral_small_3_1_base"): 28,
    ("code800", "mistral_small_3_1_base"): 20,
    ("fact800", "mistral_small_3_1_base"): 28,
    # Gemma-3-4B layers from existing main-grid derived data
    # (steering / direction_comparison filenames).
    ("math800", "gemma3_4b"): 16,
    ("code800", "gemma3_4b"): 15,
    ("fact800", "gemma3_4b"): 16,
    # Gemma-3-12B PLACEHOLDER layers (48-layer model, ~48% depth analog of 4B's
    # L16/34). Update after running analyze_layer_emergence on this model.
    ("math800", "gemma3_12b"): 24,
    ("code800", "gemma3_12b"): 22,
    ("fact800", "gemma3_12b"): 24,
}

# ── Refusal detection ────────────────────────────────────────────────
REFUSAL_KW = [
    "cannot", "does not", "undefined", "not defined", "no real", "doesn't",
    "invalid", "impossible", "not exist", "no solution", "indeterminate",
    "not possible", "no answer", "doesn't have", "can't", "unable",
    "not a valid", "not a real", "no meaning", "meaningless",
    "does not have a solution", "there is no",
]

def is_refusal(text):
    t = text.lower()
    return any(kw in t for kw in REFUSAL_KW)


# ── Direction computation ────────────────────────────────────────────
def compute_nullspace_direction(reps_path, meta_path, layer, k_pca=100, seed=42):
    """
    Compute mean-diff direction in null-space of A-class PCA.
    Returns d_hat (D,) unit vector and scale (mean separation norm).
    Uses 50% train split for direction, 50% for testing.
    """
    reps = np.load(reps_path, mmap_mode='r')
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m['answerable'] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]

    # Train/test split
    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[:len(A_idx)//2]]
    trU = U_idx[pU[:len(U_idx)//2]]

    # PCA on train-A
    nc = min(k_pca, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc).fit(X[trA])
    V = pca.components_

    # Null-space residuals
    R = X - X @ V.T @ V

    # Mean-diff direction
    mu_diff = R[trU].mean(0) - R[trA].mean(0)
    scale = norm(mu_diff)
    d_hat = mu_diff / (scale + 1e-15)

    # Also compute std of projections for scaling α
    R_all = np.concatenate([R[trA], R[trU]])
    proj_std = np.std(R_all @ d_hat)

    print(f"  Null-space direction: ||mu_diff||={scale:.4f}, proj_std={proj_std:.4f}")
    print(f"  PCA components: {nc}, explained var ratio sum: {pca.explained_variance_ratio_.sum():.3f}")

    return d_hat, scale, proj_std


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--dataset",  required=True,
                        choices=["math800","code800","fact800","abstentionbench_gsm8k","mathtrap","falseqa"])
    parser.add_argument("--layer",    type=int, default=None,
                        help="Layer to intervene at (0-indexed, matches reps[:, k, :]). Default: best layer.")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of A and U samples to test")
    parser.add_argument("--alphas",   type=str, default="0,5,10,20,40",
                        help="Comma-separated intervention strengths (multiples of proj_std)")
    parser.add_argument("--out-dir",  default="experiments/intervention")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--save-samples", action="store_true",
                        help="Save per-sample outputs for human validation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

    # Layer selection — use `is not None` (not `or`) so --layer 0 is respected
    layer = args.layer if args.layer is not None else BEST_LAYERS.get(
        (args.dataset, args.model), 15
    )
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"=== Directional Intervention ===")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Layer: {layer}")
    print(f"Alphas (× proj_std): {alphas}")

    # ── Step 1: Compute direction from saved representations ──
    # Prefer last-token reps (matches detection/orthogonality main pipeline);
    # fall back to mean-pooled if last-token not available.
    sig_dir = os.path.join(BASE, f"experiments/signals/{args.dataset}_{args.model}_allL/signals")
    lt_reps_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_reps_path = os.path.join(sig_dir, "reps_all_layers.npy")
    if os.path.exists(lt_reps_path):
        reps_path = lt_reps_path
        print("  Using last-token reps (aligned with main pipeline)")
    else:
        reps_path = mp_reps_path
        print("  WARNING: last-token reps not found, falling back to mean-pooled")
    meta_path = os.path.join(sig_dir, "meta.jsonl")

    reps_type = "last_token" if reps_path == lt_reps_path else "mean_pooled"

    print("\n[1/4] Computing null-space direction...")
    d_hat, mu_norm, proj_std = compute_nullspace_direction(
        reps_path, meta_path, layer, seed=args.seed
    )
    d_hat_torch = torch.tensor(d_hat, dtype=torch.bfloat16)

    # ── Step 2: Load model ──
    print(f"\n[2/4] Loading model {args.model}...")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[args.model], dtype=torch.bfloat16
        ).to(device)
    except ValueError as e:
        err = str(e)
        if "Mistral3" in err:
            from transformers import Mistral3ForConditionalGeneration
            model = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_PATHS[args.model], dtype=torch.bfloat16
            ).to(device)
        elif "Gemma3" in err:
            from transformers import Gemma3ForConditionalGeneration
            model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_PATHS[args.model], dtype=torch.bfloat16
            ).to(device)
        elif "Unrecognized configuration class" in err:
            # Generic fallback for other multimodal-capable wrappers
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATHS[args.model], dtype=torch.bfloat16
            ).to(device)
        else:
            raise
    model.eval()

    # Multimodal-safe decoder-layer locator. Mistral-3.x / Gemma-3 wrappers may
    # nest decoder layers under model.model.language_model.layers (or similar).
    def _get_decoder_layers(m):
        """Locate the decoder layer ModuleList. Works for plain LLM and multimodal wrappers
        (Gemma-3, Mistral-3.x). Tries fast hardcoded paths first, falls back to named_modules
        search for any ModuleList named '*.layers' with >= 20 entries (heuristic for decoder)."""
        import torch.nn as nn
        fast_paths = [
            lambda: m.model.layers,                              # plain LM: Llama, Qwen, Mistral, Phi
            lambda: m.model.language_model.layers,               # Gemma-3 multimodal (transformers ≥ 4.50)
            lambda: m.model.language_model.model.layers,         # alternate nesting
            lambda: m.language_model.model.layers,               # older multimodal API
            lambda: m.language_model.layers,                     # text-only sub-model exposed at top
        ]
        for fn in fast_paths:
            try:
                layers = fn()
                if isinstance(layers, nn.ModuleList) and len(layers) >= 10:
                    return layers
            except AttributeError:
                continue
        candidates = []
        for name, mod in m.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 20 and name.split('.')[-1] == 'layers':
                candidates.append((name, mod))
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            lm_paths = [c for c in candidates if 'language_model' in c[0]]
            if lm_paths:
                return lm_paths[0][1]
            return min(candidates, key=lambda c: len(c[0]))[1]
        raise AttributeError(
            f"Cannot locate decoder layers in {type(m).__name__}; "
            f"top children: {[n for n,_ in m.named_children()]}"
        )
    decoder_layers = _get_decoder_layers(model)
    n_layers = (
        getattr(model.config, 'num_hidden_layers', None)
        or getattr(getattr(model.config, 'text_config', None), 'num_hidden_layers', None)
        or len(decoder_layers)
    )
    print(f"  Loaded. n_layers={n_layers}")

    # ── Step 3: Select test samples ──
    print(f"\n[3/4] Selecting {args.n_samples} A + {args.n_samples} U samples...")
    meta = [json.loads(l) for l in open(meta_path)]
    # Use second half as test (first half used for direction)
    rng = np.random.RandomState(args.seed)
    A_all = [m for m in meta if m['answerable'] == 'A']
    U_all = [m for m in meta if m['answerable'] == 'U']
    rng.shuffle(A_all); rng.shuffle(U_all)
    A_test = A_all[len(A_all)//2 : len(A_all)//2 + args.n_samples]
    U_test = U_all[len(U_all)//2 : len(U_all)//2 + args.n_samples]
    print(f"  A test: {len(A_test)}, U test: {len(U_test)}")

    # ── Step 4: Run intervention ──
    print(f"\n[4/4] Running interventions...")

    # Also prepare random control direction (same norm)
    rng_ctrl = np.random.RandomState(args.seed + 999)
    d_random = rng_ctrl.randn(len(d_hat)).astype(np.float32)
    d_random /= (norm(d_random) + 1e-15)
    d_random_torch = torch.tensor(d_random, dtype=torch.bfloat16)

    def generate_with_direction_hook(prompt, direction_torch, alpha_scaled, target_layer):
        """Generate from prompt with intervention: x += alpha_scaled * direction at target_layer."""
        inputs = tok(prompt, return_tensors="pt").to(device)
        T_in = inputs["input_ids"].shape[1]

        if abs(alpha_scaled) < 1e-8:
            # No intervention — clean generation
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            return tok.decode(out[0][T_in:], skip_special_tokens=True).strip()

        direction_dev = direction_torch.to(device)

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                # Intervene on last token position (where model predicts next token)
                hs[0, -1, :] += alpha_scaled * direction_dev
                return (hs,) + output[1:]
            else:
                out = output.clone()
                out[0, -1, :] += alpha_scaled * direction_dev
                return out

        # 0-indexed: reps[:, k, :] = output of model.layers[k], so hook layers[k].
        # Use decoder_layers (resolved above) to support multimodal wrappers
        # (Gemma-3, Mistral-3.x) that nest layers under language_model.
        layer_module = decoder_layers[target_layer]
        handle = layer_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        handle.remove()
        return tok.decode(out[0][T_in:], skip_special_tokens=True).strip()

    # Results storage
    all_results = []
    all_samples = []  # per-sample outputs (only if --save-samples)

    for condition, samples, sign in [("U→A (remove signal)", U_test, -1.0),
                                      ("A→U (inject signal)", A_test, +1.0)]:
        print(f"\n--- {condition} ---")

        for alpha_mult in alphas:
            alpha_val = sign * alpha_mult * proj_std

            n_flipped_signal = 0
            n_flipped_random = 0
            n_valid = 0
            # Gated-denominator counters: only samples whose clean baseline is
            # already in the *expected* class for this condition can be flipped
            # by steering — so flip_rate_gated = flipped / n_clean_gate is the
            # meaningful causal success rate. n_clean_refusal is reported for
            # transparency (ceiling info for models like Llama whose clean
            # baseline is not saturated in the expected class).
            n_clean_gate    = 0
            n_clean_refusal = 0

            for idx, sample in enumerate(samples):
                prompt = sample['prompt']

                # Clean baseline
                clean_out = generate_with_direction_hook(prompt, d_hat_torch, 0.0, layer)
                clean_is_refusal = is_refusal(clean_out)

                # Signal direction intervention
                signal_out = generate_with_direction_hook(prompt, d_hat_torch, alpha_val, layer)
                signal_is_refusal = is_refusal(signal_out)

                # Random direction control
                random_out = generate_with_direction_hook(prompt, d_random_torch, alpha_val, layer)
                random_is_refusal = is_refusal(random_out)

                n_valid += 1
                if clean_is_refusal:
                    n_clean_refusal += 1

                if condition.startswith("U"):
                    # U→A: expect refusal→attempt (clean=refusal, patched=not refusal)
                    if clean_is_refusal:
                        n_clean_gate += 1
                    keyword_flip_signal = clean_is_refusal and not signal_is_refusal
                    keyword_flip_random = clean_is_refusal and not random_is_refusal
                    if keyword_flip_signal:
                        n_flipped_signal += 1
                    if keyword_flip_random:
                        n_flipped_random += 1
                else:
                    # A→U: expect attempt→refusal (clean=not refusal, patched=refusal)
                    if not clean_is_refusal:
                        n_clean_gate += 1
                    keyword_flip_signal = (not clean_is_refusal) and signal_is_refusal
                    keyword_flip_random = (not clean_is_refusal) and random_is_refusal
                    if keyword_flip_signal:
                        n_flipped_signal += 1
                    if keyword_flip_random:
                        n_flipped_random += 1

                if args.save_samples:
                    all_samples.append({
                        "condition": condition,
                        "alpha_mult": alpha_mult,
                        "sample_id": sample.get('id', f'{idx}'),
                        "prompt": prompt[:300],
                        "clean_output": clean_out,
                        "signal_output": signal_out,
                        "random_output": random_out,
                        "clean_is_refusal": clean_is_refusal,
                        "signal_is_refusal": signal_is_refusal,
                        "random_is_refusal": random_is_refusal,
                        "keyword_flip_signal": keyword_flip_signal,
                        "keyword_flip_random": keyword_flip_random,
                    })

                if idx < 3 or (idx < 10 and alpha_mult == alphas[-1]):
                    print(f"  [{idx}] α={alpha_mult:.0f}×σ: "
                          f"clean={'REF' if clean_is_refusal else 'ATT'} → "
                          f"signal={'REF' if signal_is_refusal else 'ATT'} | "
                          f"random={'REF' if random_is_refusal else 'ATT'}")
                    if alpha_mult == alphas[-1] and idx < 5:
                        print(f"       clean:  {clean_out[:80]}")
                        print(f"       signal: {signal_out[:80]}")

            rate_signal         = n_flipped_signal / max(n_valid, 1)
            rate_random         = n_flipped_random / max(n_valid, 1)
            clean_refusal_rate  = n_clean_refusal / max(n_valid, 1)
            # Gated rates are UNDEFINED when no clean sample matched the
            # expected baseline class. Report as None/null, NOT 0.0 — reading
            # 0.0 as "no causal effect" would be incorrect here.
            if n_clean_gate > 0:
                rate_signal_gated = n_flipped_signal / n_clean_gate
                rate_random_gated = n_flipped_random / n_clean_gate
                gated_fmt = (f"[gated: sig={rate_signal_gated:.3f} "
                             f"rand={rate_random_gated:.3f} denom={n_clean_gate}]")
            else:
                rate_signal_gated = None
                rate_random_gated = None
                gated_fmt = "[gated: N/A — no clean sample in expected class]"
            print(f"  α={alpha_mult:5.1f}×σ: "
                  f"flip_signal={rate_signal:.3f} ({n_flipped_signal}/{n_valid}), "
                  f"flip_random={rate_random:.3f} ({n_flipped_random}/{n_valid})  "
                  f"{gated_fmt}")

            all_results.append({
                "condition": condition,
                "alpha_mult": alpha_mult,
                "alpha_val": float(alpha_val),
                "n_valid": n_valid,
                "flip_signal": n_flipped_signal,
                "flip_random": n_flipped_random,
                "rate_signal": rate_signal,
                "rate_random": rate_random,
                # --- gated metrics (denominator = clean baseline in expected class) ---
                "n_clean_gate": n_clean_gate,
                "n_clean_refusal": n_clean_refusal,
                "clean_refusal_rate": clean_refusal_rate,
                "rate_signal_gated": rate_signal_gated,
                "rate_random_gated": rate_random_gated,
            })

    # ── Save results ──
    tag = f"{args.model}_{args.dataset}_L{layer}"
    out_path = os.path.join(args.out_dir, f"intervention_{tag}.json")
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": layer,
        "reps_type": reps_type,
        "n_samples_per_class": args.n_samples,
        "proj_std": float(proj_std),
        "mu_norm": float(mu_norm),
        "alphas": alphas,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*96}")
    print(f"SUMMARY: {tag}")
    print(f"{'='*96}")
    print(f"{'Condition':<25} {'α':>5} "
          f"{'Sig':>6} {'Rnd':>6} {'Δ':>7}  "
          f"{'SigG':>6} {'RndG':>6} {'ΔG':>7}  "
          f"{'clean_ref':>9} {'gateN':>5}")
    print("-" * 96)
    for r in all_results:
        delta = r['rate_signal'] - r['rate_random']
        if r['rate_signal_gated'] is None:
            # gate empty → gated metrics undefined, print as dashes
            gated_sig_str = "  N/A "
            gated_rnd_str = "  N/A "
            gated_dlt_str = "  N/A  "
        else:
            delta_gated = r['rate_signal_gated'] - r['rate_random_gated']
            gated_sig_str = f"{r['rate_signal_gated']:>6.3f}"
            gated_rnd_str = f"{r['rate_random_gated']:>6.3f}"
            gated_dlt_str = f"{delta_gated:>+7.3f}"
        print(f"{r['condition']:<25} {r['alpha_mult']:>4.0f}σ "
              f"{r['rate_signal']:>6.3f} {r['rate_random']:>6.3f} {delta:>+7.3f}  "
              f"{gated_sig_str} {gated_rnd_str} {gated_dlt_str}  "
              f"{r['clean_refusal_rate']:>9.3f} {r['n_clean_gate']:>5d}")

    print(f"\nSaved to {out_path}")

    # Save per-sample outputs for human validation
    if args.save_samples and all_samples:
        import csv
        samples_path = os.path.join(args.out_dir, f"samples_{tag}.tsv")
        fields = [
            "condition", "alpha_mult", "sample_id", "prompt",
            "clean_output", "signal_output", "random_output",
            "clean_is_refusal", "signal_is_refusal", "random_is_refusal",
            "keyword_flip_signal", "keyword_flip_random",
            "human_label",
        ]
        with open(samples_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for s in all_samples:
                s["human_label"] = ""
                w.writerow(s)
        print(f"Per-sample outputs: {samples_path} ({len(all_samples)} rows)")
        print(f"  → Open TSV, review, fill 'human_label': 1=keyword correct, 0=wrong")


if __name__ == "__main__":
    main()
