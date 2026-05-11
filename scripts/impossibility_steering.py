"""
Impossibility-Aware Decoding: Steering LLMs to Refuse Impossible Questions.

Experiment: For impossible questions where the model currently HALLUCINATES
(gives a numeric/wrong answer instead of refusing), apply the impossibility
direction at inference time to steer it toward appropriate refusal.

This directly tests the practical application of the impossibility direction:
can we reduce hallucination on structurally unanswerable questions?

Evaluation metrics (keyword-based, no ground-truth verification):
  1. Hallucination rate: % of impossible Qs where model gives a non-refusal answer
  2. Refusal rate: % of impossible Qs where model output contains refusal keywords
  3. Non-refusal rate (A): % of answerable Qs where model does NOT refuse
     NOTE: this measures absence of refusal, NOT correctness of the answer
  4. Overall proxy: (correct refusals + non-refusals on A) / total

Compares:
  - Baseline (no steering)
  - Impossibility direction steering (various α)
  - Random direction control
  - Refusal direction steering (Arditi-style, if available)

Usage:
  python scripts/impossibility_steering.py --model mistral --dataset math800 \
      --n-samples 100 --alphas 0,5,10,20,40
"""
import argparse, json, os, re, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from numpy.linalg import norm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

# ── Model paths ──────────────────────────────────────────────────────
MODEL_PATHS = {
    "llama":   os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B-Instruct-HF"),
    "qwen":    os.path.expanduser(
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
    # Phase 1 upgrades (Mac-tier; 2026-04-29).
    "phi4mini":   "microsoft/Phi-4-mini-instruct",
    "gemma3_4b":  "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "qwen3_8b":   "Qwen/Qwen3-8B",
    "qwen3_14b":  "Qwen/Qwen3-14B",
    # Phase 2 / 3 upgrades (RunPod 2026-04-30).
    "qwen3_32b":         "Qwen/Qwen3-32B",
    "llama70b":          "meta-llama/Llama-3.3-70B-Instruct",
    # Mistral-Small-3.x successor pair: 3.2-Instruct is a minor update of
    # 3.1-Instruct; HF model tree lists 3.1-24B-Base-2503 as their shared base.
    "mistral_small_3_2":      "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistral_small_3_1_base": "mistralai/Mistral-Small-3.1-24B-Base-2503",
}

BEST_LAYERS = {
    ("math800", "mistral"): 15, ("math800", "llama"): 15, ("math800", "qwen"): 18,
    ("code800", "mistral"): 15, ("code800", "llama"): 14, ("code800", "qwen"): 18,
    ("math800", "mistral_small"): 28, ("code800", "mistral_small"): 20,
    ("fact800", "mistral_small"): 28,
    ("math800", "qwen14b"): 34, ("code800", "qwen14b"): 32,
    # Small models — best layers TBD after extraction, using middle layer as default
    ("math800", "phi3"): 15, ("code800", "phi3"): 16,
    ("math800", "smollm2"): 11, ("code800", "smollm2"): 14,
    ("math800", "gemma2"): 16, ("code800", "gemma2"): 14,
    # FACT-800 — llama/mistral/qwen layers from old intervention experiments;
    # small models use same layer as math800 (reasonable default, adjust after layerwise AUC check)
    ("fact800", "llama"): 15, ("fact800", "mistral"): 17, ("fact800", "qwen"): 19,
    ("fact800", "qwen14b"): 34, ("fact800", "phi3"): 15,
    ("fact800", "smollm2"): 11, ("fact800", "gemma2"): 16,
    ("math800", "olmo13b"): 20, ("code800", "olmo13b"): 20,
    ("fact800", "olmo13b"): 20,
    # Layers inherited from Mistral-Small-2501 24B layout; re-validate by
    # layer scan if the first runs look off.
    ("math800", "mistral_small_3_2"): 28,
    ("code800", "mistral_small_3_2"): 20,
    ("fact800", "mistral_small_3_2"): 28,
    ("math800", "mistral_small_3_1_base"): 28,
    ("code800", "mistral_small_3_1_base"): 20,
    ("fact800", "mistral_small_3_1_base"): 28,
    # Gemma-3-4B layers from main-grid derived data.
    ("math800", "gemma3_4b"): 16,
    ("code800", "gemma3_4b"): 15,
    ("fact800", "gemma3_4b"): 16,
    # Gemma-3-12B PLACEHOLDER (48-layer model). Update after layer_emergence.
    ("math800", "gemma3_12b"): 24,
    ("code800", "gemma3_12b"): 22,
    ("fact800", "gemma3_12b"): 24,
}

# ── Classification of model outputs ──────────────────────────────────
REFUSAL_KW = [
    "cannot", "does not", "undefined", "not defined", "no real", "doesn't",
    "invalid", "impossible", "not exist", "no solution", "indeterminate",
    "not possible", "no answer", "doesn't have", "can't", "unable",
    "not a valid", "not a real", "no meaning", "meaningless",
    "does not have a solution", "there is no", "not meaningful",
    "division by zero", "denominator is zero", "divide by zero",
    # fact800 (SQuAD-style unanswerable) — models often refuse differently
    "not mentioned", "not stated", "not provided", "not specified",
    "no information", "not enough information", "insufficient information",
    "the passage does not", "the context does not", "the text does not",
    "unanswerable", "not answerable", "cannot be determined",
    "cannot be answered", "not clear from",
]

def classify_output(text, is_unanswerable):
    """
    Classify model output into categories based on refusal keyword matching.

    For unanswerable (U) samples: refusal keywords → 'correct_refusal', else → 'hallucination'
    For answerable (A) samples: refusal keywords → 'wrong_refusal', else → 'non_refusal'

    NOTE: 'non_refusal' means the model did NOT refuse, but does NOT verify answer correctness.
    Therefore non_refusal_rate_A measures "non-refusal rate", not "accuracy".
    """
    t = text.lower().strip()
    has_refusal = any(kw in t for kw in REFUSAL_KW)

    if is_unanswerable:
        return 'correct_refusal' if has_refusal else 'hallucination'
    else:
        return 'wrong_refusal' if has_refusal else 'non_refusal'


# ── Direction computation ────────────────────────────────────────────
def compute_impossibility_direction(reps_path, meta_path, layer, k_pca=100, seed=42):
    """Compute null-space mean-diff direction (same as CosNSRT/intervention)."""
    reps = np.load(reps_path, mmap_mode='r')
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m['answerable'] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]

    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[:len(A_idx)//2]]
    trU = U_idx[pU[:len(U_idx)//2]]

    nc = min(k_pca, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc).fit(X[trA])
    V = pca.components_
    R = X - X @ V.T @ V

    mu_diff = R[trU].mean(0) - R[trA].mean(0)
    scale = norm(mu_diff)
    d_hat = mu_diff / (scale + 1e-15)

    R_all = np.concatenate([R[trA], R[trU]])
    proj_std = np.std(R_all @ d_hat)

    print(f"  Impossibility direction: ||mu_diff||={scale:.4f}, proj_std={proj_std:.4f}")
    return d_hat, proj_std


def compute_cross_domain_direction(train_dataset, test_dataset, model_name, layer, k_pca=100, seed=42):
    """Compute direction from one domain, to apply on another."""
    sig_dir = os.path.join(BASE, f"experiments/signals/{train_dataset}_{model_name}_allL/signals")
    # Prefer last-token reps (aligned with main pipeline)
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_path = os.path.join(sig_dir, "reps_all_layers.npy")
    reps_path = lt_path if os.path.exists(lt_path) else mp_path
    meta_path = os.path.join(sig_dir, "meta.jsonl")
    d_hat, proj_std = compute_impossibility_direction(reps_path, meta_path, layer, k_pca, seed)
    print(f"  (Cross-domain: direction from {train_dataset}, applying to {test_dataset})")
    return d_hat, proj_std


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "llama","qwen","mistral","mistral_small","qwen14b","phi3","smollm2","gemma2","olmo13b","qwen32b",
        # Phase 1 upgrades — BEST_LAYERS not yet populated; pass --layer explicitly until then.
        "phi4mini","gemma3_4b","gemma3_12b","qwen3_8b","qwen3_14b",
        # Phase 2 upgrades — BEST_LAYERS not yet populated; pass --layer explicitly.
        "qwen3_32b",
        # Phase 3: 70B (multi-GPU sharded).
        "llama70b",
        # Mistral-Small-3.x successor pair (Phase 0 patch).
        "mistral_small_3_2", "mistral_small_3_1_base",
    ])
    parser.add_argument("--dataset", required=True, choices=["math800","code800","fact800"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of A and U samples each")
    parser.add_argument("--alphas", type=str, default="0,5,10,20,30,40",
                        help="Steering strengths (multiples of proj_std)")
    parser.add_argument("--cross-domain", action="store_true",
                        help="Also test cross-domain direction (e.g., CODE dir on MATH data)")
    parser.add_argument("--out-dir", default="experiments/steering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE, args.out_dir), exist_ok=True)
    layer = args.layer or BEST_LAYERS.get((args.dataset, args.model), 15)
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"{'='*80}")
    print(f"IMPOSSIBILITY-AWARE DECODING")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Layer: {layer}")
    print(f"Samples: {args.n_samples} A + {args.n_samples} U")
    print(f"Alphas (× proj_std): {alphas}")
    print(f"{'='*80}")

    # ── Step 1: Compute directions ──
    print("\n[1/4] Computing directions...")
    sig_dir = os.path.join(BASE, f"experiments/signals/{args.dataset}_{args.model}_allL/signals")
    # Prefer last-token reps (aligned with detection/orthogonality main pipeline);
    # fall back to mean-pooled if last-token not available.
    lt_reps_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_reps_path = os.path.join(sig_dir, "reps_all_layers.npy")
    if os.path.exists(lt_reps_path):
        reps_path = lt_reps_path
        print("  Using last-token reps (aligned with main pipeline)")
    else:
        reps_path = mp_reps_path
        print("  WARNING: last-token reps not found, falling back to mean-pooled")
    meta_path = os.path.join(sig_dir, "meta.jsonl")

    d_impos, proj_std = compute_impossibility_direction(reps_path, meta_path, layer, seed=args.seed)
    d_impos_torch = torch.tensor(d_impos, dtype=torch.bfloat16)

    # Random control direction
    rng = np.random.RandomState(args.seed + 999)
    d_random = rng.randn(len(d_impos)).astype(np.float32)
    d_random /= (norm(d_random) + 1e-15)
    d_random_torch = torch.tensor(d_random, dtype=torch.bfloat16)

    # Cross-domain direction (optional)
    d_cross_torch = None
    cross_proj_std = None
    if args.cross_domain:
        other = "code800" if args.dataset == "math800" else "math800"
        other_layer = BEST_LAYERS.get((other, args.model), layer)
        d_cross, cross_proj_std = compute_cross_domain_direction(
            other, args.dataset, args.model, other_layer, seed=args.seed)
        d_cross_torch = torch.tensor(d_cross, dtype=torch.bfloat16)

    # ── Step 2: Load model ──
    print(f"\n[2/4] Loading model {args.model}...")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Multi-GPU sharding for 70B-class (doesn't fit single A100-80GB at bf16).
    _shard_models = ("llama70b", "llama70b_base")
    use_shard = (args.model in _shard_models) and (device.type == "cuda")
    if use_shard:
        print(f"  Loading {args.model} with device_map='auto' (multi-GPU sharding)")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[args.model], dtype=torch.bfloat16, device_map="auto"
            )
        except ValueError as e:
            err = str(e)
            if "Mistral3" in err:
                from transformers import Mistral3ForConditionalGeneration
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    MODEL_PATHS[args.model], dtype=torch.bfloat16, device_map="auto"
                )
            elif "Gemma3" in err:
                from transformers import Gemma3ForConditionalGeneration
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    MODEL_PATHS[args.model], dtype=torch.bfloat16, device_map="auto"
                )
            elif "Unrecognized configuration class" in err:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_PATHS[args.model], dtype=torch.bfloat16, device_map="auto"
                )
            else:
                raise
    else:
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
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_PATHS[args.model], dtype=torch.bfloat16
                ).to(device)
            else:
                raise
    model.eval()
    # Multimodal-safe layer count: Gemma-3 / Mistral-3.x wrappers nest num_hidden_layers
    # under config.text_config. Try top-level first, then text_config, then count modules.
    def _get_decoder_layers(m):
        """Locate the decoder layer ModuleList. Works for plain LLM and multimodal wrappers
        (Gemma-3, Mistral-3.x). Tries fast hardcoded paths first, falls back to named_modules
        search for any ModuleList named '*.layers' with >= 20 entries (heuristic for decoder)."""
        import torch.nn as nn
        # Fast hardcoded paths (covers the common cases)
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
        # Fallback: walk named_modules
        candidates = []
        for name, mod in m.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 20 and name.split('.')[-1] == 'layers':
                candidates.append((name, mod))
        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            # Prefer the path containing 'language_model' or shortest path
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
    print(f"  Loaded. {n_layers} layers")

    # ── Step 3: Select test samples ──
    print(f"\n[3/4] Selecting test samples...")
    meta = [json.loads(l) for l in open(meta_path)]
    rng_split = np.random.RandomState(args.seed)
    A_all = [m for m in meta if m['answerable'] == 'A']
    U_all = [m for m in meta if m['answerable'] == 'U']
    rng_split.shuffle(A_all)
    rng_split.shuffle(U_all)

    # Use second half (not used for direction computation)
    A_test = A_all[len(A_all)//2 : len(A_all)//2 + args.n_samples]
    U_test = U_all[len(U_all)//2 : len(U_all)//2 + args.n_samples]
    print(f"  A test: {len(A_test)}, U test: {len(U_test)}")

    # ── Step 4: Generate with steering ──
    print(f"\n[4/4] Running steering experiments...")

    def generate_steered(prompt, direction_torch, alpha_scaled, target_layer):
        """Generate with direction added at target_layer."""
        # For sharded models, model.generate handles input device placement internally
        # via device_map; for single-device models, we need to .to(device) explicitly.
        if use_shard:
            # device_map="auto" puts the embedding layer on cuda:0; tokenizer outputs
            # default to CPU and HF .generate handles dispatch. But to be safe, send
            # explicitly to the embedding's device.
            embed_dev = next(model.get_input_embeddings().parameters()).device
            inputs = tok(prompt, return_tensors="pt").to(embed_dev)
        else:
            inputs = tok(prompt, return_tensors="pt").to(device)
        T_in = inputs["input_ids"].shape[1]

        if abs(alpha_scaled) < 1e-8:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            return tok.decode(out[0][T_in:], skip_special_tokens=True).strip()

        # target_layer is 0-indexed: reps[:, k, :] = hidden_states[k+1] = output of layers[k]
        # So hook on layers[target_layer] to match the layer the direction was fit from.
        # Use decoder_layers (resolved above) to support multimodal wrappers like Gemma-3.
        layer_module = decoder_layers[target_layer]
        # For sharded models, the target layer may live on cuda:0 or cuda:1 — get its
        # device dynamically. For single-device, use the global device.
        layer_dev = next(layer_module.parameters()).device

        def hook_fn(module, input, output):
            # Move direction to whichever device the layer's output landed on.
            # (For shard models: hook output is on layer_dev; for single-device: same as global.)
            if isinstance(output, tuple):
                hs = output[0]
                d_local = direction_torch.to(hs.device, dtype=hs.dtype)
                hs2 = hs.clone()
                hs2[0, -1, :] += alpha_scaled * d_local
                return (hs2,) + output[1:]
            else:
                d_local = direction_torch.to(output.device, dtype=output.dtype)
                out = output.clone()
                out[0, -1, :] += alpha_scaled * d_local
                return out

        handle = layer_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        handle.remove()
        return tok.decode(out[0][T_in:], skip_special_tokens=True).strip()

    # Run all conditions
    results_by_alpha = {}
    samples_log = []  # accumulate across ALL alphas (each sample has 'alpha' field)

    for alpha_mult in alphas:
        print(f"\n--- α = {alpha_mult}×σ ---")
        alpha_val_impos = alpha_mult * proj_std  # positive = inject impossibility signal

        counts = {
            'impossibility': {'correct_refusal': 0, 'hallucination': 0, 'total': 0},
            'random':        {'correct_refusal': 0, 'hallucination': 0, 'total': 0},
        }
        if args.cross_domain:
            counts['cross_domain'] = {'correct_refusal': 0, 'hallucination': 0, 'total': 0}

        answerable_counts = {
            'impossibility': {'non_refusal': 0, 'wrong_refusal': 0, 'total': 0},
            'random':        {'non_refusal': 0, 'wrong_refusal': 0, 'total': 0},
        }
        if args.cross_domain:
            answerable_counts['cross_domain'] = {'non_refusal': 0, 'wrong_refusal': 0, 'total': 0}

        # --- Test on Unanswerable questions (want: hallucination → refusal) ---
        for idx, sample in enumerate(U_test):
            prompt = sample['prompt']

            # Impossibility steering: ADD direction (push toward impossibility)
            out_impos = generate_steered(prompt, d_impos_torch, alpha_val_impos, layer)
            cat_impos = classify_output(out_impos, is_unanswerable=True)
            counts['impossibility'][cat_impos] += 1
            counts['impossibility']['total'] += 1

            # Random control
            out_rand = generate_steered(prompt, d_random_torch, alpha_val_impos, layer)
            cat_rand = classify_output(out_rand, is_unanswerable=True)
            counts['random'][cat_rand] += 1
            counts['random']['total'] += 1

            # Cross-domain
            if args.cross_domain and d_cross_torch is not None:
                alpha_val_cross = alpha_mult * (cross_proj_std or proj_std)
                out_cross = generate_steered(prompt, d_cross_torch, alpha_val_cross, layer)
                cat_cross = classify_output(out_cross, is_unanswerable=True)
                counts['cross_domain'][cat_cross] += 1
                counts['cross_domain']['total'] += 1

            if idx < 5 and alpha_mult == alphas[-1]:
                print(f"  U[{idx}] {cat_impos:>16s} | {out_impos[:60]}")

            samples_log.append({
                'type': 'U', 'idx': idx, 'prompt': prompt[:200],
                'impos_out': out_impos, 'impos_cat': cat_impos,
                'rand_out': out_rand, 'rand_cat': cat_rand,
                'alpha': alpha_mult,
            })

        # --- Test on Answerable questions (want: no spurious refusal) ---
        for idx, sample in enumerate(A_test):
            prompt = sample['prompt']

            out_impos = generate_steered(prompt, d_impos_torch, alpha_val_impos, layer)
            cat_impos = classify_output(out_impos, is_unanswerable=False)
            answerable_counts['impossibility'][cat_impos] += 1
            answerable_counts['impossibility']['total'] += 1

            out_rand = generate_steered(prompt, d_random_torch, alpha_val_impos, layer)
            cat_rand = classify_output(out_rand, is_unanswerable=False)
            answerable_counts['random'][cat_rand] += 1
            answerable_counts['random']['total'] += 1

            if args.cross_domain and d_cross_torch is not None:
                alpha_val_cross = alpha_mult * (cross_proj_std or proj_std)
                out_cross = generate_steered(prompt, d_cross_torch, alpha_val_cross, layer)
                cat_cross = classify_output(out_cross, is_unanswerable=False)
                answerable_counts['cross_domain'][cat_cross] += 1
                answerable_counts['cross_domain']['total'] += 1

            if idx < 5 and alpha_mult == alphas[-1]:
                print(f"  A[{idx}] {cat_impos:>16s} | {out_impos[:60]}")

            samples_log.append({
                'type': 'A', 'idx': idx, 'prompt': prompt[:200],
                'impos_out': out_impos, 'impos_cat': cat_impos,
                'rand_out': out_rand, 'rand_cat': cat_rand,
                'alpha': alpha_mult,
            })

        # Compute metrics
        metrics = {}
        for method in counts:
            u_total = counts[method]['total']
            a_total = answerable_counts[method]['total']
            refusal_rate = counts[method].get('correct_refusal', 0) / max(u_total, 1)
            halluc_rate = counts[method].get('hallucination', 0) / max(u_total, 1)
            preserve_rate = answerable_counts[method].get('non_refusal', 0) / max(a_total, 1)
            wrong_ref_rate = answerable_counts[method].get('wrong_refusal', 0) / max(a_total, 1)

            # NOTE: "overall" here is (correct_refusals + non_refusals) / total,
            # which is a keyword-based proxy, NOT ground-truth accuracy.
            proxy_correct = counts[method].get('correct_refusal', 0) + answerable_counts[method].get('non_refusal', 0)
            total = u_total + a_total
            overall_proxy = proxy_correct / max(total, 1)

            metrics[method] = {
                'refusal_rate_U': refusal_rate,
                'hallucination_rate_U': halluc_rate,
                'non_refusal_rate_A': preserve_rate,
                'wrong_refusal_rate_A': wrong_ref_rate,
                'overall_proxy': overall_proxy,
                'n_U': u_total, 'n_A': a_total,
            }

        results_by_alpha[str(alpha_mult)] = {
            'metrics': metrics,
            'raw_counts_U': {m: dict(counts[m]) for m in counts},
            'raw_counts_A': {m: dict(answerable_counts[m]) for m in answerable_counts},
        }

        # Print summary for this alpha
        print(f"\n  {'Method':<16} {'Refusal(U)':>10} {'Halluc(U)':>10} {'NonRef(A)':>10} {'WrongRef(A)':>12} {'Proxy':>8}")
        print(f"  {'-'*70}")
        for method in metrics:
            m = metrics[method]
            print(f"  {method:<16} {m['refusal_rate_U']:>10.3f} {m['hallucination_rate_U']:>10.3f} "
                  f"{m['non_refusal_rate_A']:>12.3f} {m['wrong_refusal_rate_A']:>12.3f} {m['overall_proxy']:>8.3f}")

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY: Impossibility-Aware Decoding")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Layer: {layer}")
    print(f"{'='*80}")

    print(f"\n{'α':>5} | {'Impos Refusal(U)':>16} {'Impos Halluc(U)':>16} {'Impos NonRef(A)':>16} {'Impos Proxy':>12} | {'Random Proxy':>13}")
    print(f"{'-'*95}")
    for alpha_mult in alphas:
        r = results_by_alpha[str(alpha_mult)]
        mi = r['metrics']['impossibility']
        mr = r['metrics']['random']
        print(f"{alpha_mult:>5.0f} | {mi['refusal_rate_U']:>16.3f} {mi['hallucination_rate_U']:>16.3f} "
              f"{mi['non_refusal_rate_A']:>18.3f} {mi['overall_proxy']:>14.3f} | {mr['overall_proxy']:>14.3f}")

    # Find best alpha (maximize overall proxy = correct_refusals + non_refusals)
    best_alpha = max(results_by_alpha.keys(),
                     key=lambda a: results_by_alpha[a]['metrics']['impossibility']['overall_proxy'])
    best_m = results_by_alpha[best_alpha]['metrics']['impossibility']
    baseline_m = results_by_alpha['0.0']['metrics']['impossibility']

    halluc_reduction = baseline_m['hallucination_rate_U'] - best_m['hallucination_rate_U']
    halluc_reduction_pct = halluc_reduction / max(baseline_m['hallucination_rate_U'], 1e-8) * 100
    preserve_cost = baseline_m['non_refusal_rate_A'] - best_m['non_refusal_rate_A']

    print(f"\n  Best α = {best_alpha}×σ:")
    print(f"    Hallucination reduction: {halluc_reduction:.3f} ({halluc_reduction_pct:.1f}% relative)")
    print(f"    Answerable preservation cost: {preserve_cost:.3f}")
    print(f"    Overall proxy improvement: {best_m['overall_proxy'] - baseline_m['overall_proxy']:+.3f}")

    # ── Save ──
    tag = f"{args.model}_{args.dataset}_L{layer}"
    out_path = os.path.join(BASE, args.out_dir, f"steering_{tag}.json")
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "layer": layer,
        "n_samples_per_class": args.n_samples,
        "proj_std": float(proj_std),
        "alphas": alphas,
        "results_by_alpha": results_by_alpha,
        "best_alpha": float(best_alpha),
        "hallucination_reduction": float(halluc_reduction),
        "hallucination_reduction_pct": float(halluc_reduction_pct),
        "non_refusal_cost": float(preserve_cost),
        "_note": "non_refusal_rate_A measures absence of refusal keywords, not answer correctness",
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Save per-sample log
    samples_path = os.path.join(BASE, args.out_dir, f"steering_samples_{tag}.jsonl")
    with open(samples_path, 'w') as f:
        for s in samples_log:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Per-sample log: {samples_path} ({len(samples_log)} rows)")


if __name__ == "__main__":
    main()
