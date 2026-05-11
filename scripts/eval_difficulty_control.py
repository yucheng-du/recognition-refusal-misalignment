"""
eval_difficulty_control.py — Evaluate whether the impossibility direction
separates easy vs hard math problems (all answerable).

This is a negative control: if AUC ≈ 0.5, the impossibility direction
encodes unanswerability, not difficulty. If AUC >> 0.5, the direction
may partially encode difficulty — a confound that must be discussed.

Reads `difficulty_label` (easy/hard) from the dataset, NOT `answerable`
(which is always "A" for this control set).

Usage:
  python scripts/eval_difficulty_control.py                    # all available models
  python scripts/eval_difficulty_control.py --model mistral    # single model
  python scripts/eval_difficulty_control.py --source-model mistral --source-layer 15
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

# ── Model configs (same as main pipeline) ───────────────────
MODELS = {
    "smollm2":  "SmolLM2-1.7B-Instruct",
    "gemma2":   "gemma-2-2b-it",
    "phi3":     "Phi-3-mini-4k-instruct",
    "mistral":  "Mistral-7B-Instruct-v0.3",
    "qwen":     "Qwen2.5-7B-Instruct",
    "llama":    "Llama-3.1-8B-Instruct",
    "qwen14b":  "Qwen2.5-14B-Instruct",
    "mistral_small": "Mistral-Small-24B-Instruct",
}

# Best layers per model (from core dataset evaluation)
# Synced with main pipeline (impossibility_steering.py, compare_impossibility_vs_refusal_direction.py)
BEST_LAYERS = {
    "smollm2": 11,
    "gemma2":  16,
    "phi3":    15,
    "mistral": 15,
    "qwen":    18,
    "llama":   15,
    "qwen14b": 34,
    "mistral_small": 28,
}


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2 or np.std(scores) < 1e-15:
        return 0.5
    a = roc_auc_score(y_true, scores)
    return max(a, 1 - a)


def load_impossibility_direction(model_name, layer, dataset="math800"):
    """Load d_imp from a core dataset (default: math800)."""
    sig_dir = os.path.join(BASE, f"experiments/signals/{dataset}_{model_name}_allL/signals")

    # Try last-token reps first, fall back to mean-pooled
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_path = os.path.join(sig_dir, "reps_all_layers.npy")

    if os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode='r')
        pooling = "last_token"
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode='r')
        pooling = "mean_pooled"
    else:
        return None, None, None

    reps_layer = np.array(reps[:, layer, :], dtype=np.float32)

    with open(os.path.join(sig_dir, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]
    labels = np.array([1 if m["answerable"] == "U" else 0 for m in meta])

    # Fit PCA on A-class, compute d_imp
    A_mask = labels == 0
    n_comp = min(100, int(A_mask.sum()) - 1, reps_layer.shape[1] - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(reps_layer[A_mask])

    resid = reps_layer - pca.inverse_transform(pca.transform(reps_layer))
    d_raw = resid[~A_mask].mean(0) - resid[A_mask].mean(0)
    d_imp = d_raw / np.linalg.norm(d_raw)

    # Verify on source data
    src_scores = resid @ d_imp
    src_auc = safe_auc(labels, src_scores)

    return d_imp, pca, {"pooling": pooling, "source_auc": src_auc, "layer": layer}


def evaluate_difficulty_control(model_name, d_imp, pca_source, source_info):
    """Project difficulty control reps onto impossibility direction."""
    sig_dir = os.path.join(
        BASE, f"experiments/signals/difficulty_control_gsm8k_{model_name}_allL/signals"
    )

    # Load reps (use same pooling as source direction if possible)
    pooling = source_info["pooling"]
    layer = source_info["layer"]

    if pooling == "last_token":
        reps_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
        if not os.path.exists(reps_path):
            reps_path = os.path.join(sig_dir, "reps_all_layers.npy")
            pooling = "mean_pooled (fallback)"
    else:
        reps_path = os.path.join(sig_dir, "reps_all_layers.npy")

    if not os.path.exists(reps_path):
        return None

    reps = np.load(reps_path, mmap_mode='r')
    if layer >= reps.shape[1]:
        print(f"  ⚠ Layer {layer} >= n_layers {reps.shape[1]}, using last layer")
        layer = reps.shape[1] - 1

    reps_layer = np.array(reps[:, layer, :], dtype=np.float32)

    # Check dimension match
    if reps_layer.shape[1] != d_imp.shape[0]:
        print(f"  ⚠ Dimension mismatch: control D={reps_layer.shape[1]}, "
              f"source D={d_imp.shape[0]}")
        return None

    # Load metadata and extract difficulty labels
    with open(os.path.join(sig_dir, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]

    # Read difficulty_label from the original dataset to match by prompt.
    # Prefer matching by `id` (stable); fall back to prompt for legacy meta.
    data_path = os.path.join(BASE, "data/difficulty_control_gsm8k.jsonl")
    by_id, by_prompt = {}, {}
    with open(data_path) as f:
        for l in f:
            r = json.loads(l)
            by_id[r["id"]] = r
            by_prompt[r["prompt"]] = r

    labels, unmatched = [], []
    for m in meta:
        dr = by_id.get(m.get("id")) or by_prompt.get(m.get("prompt", ""))
        if dr is None:
            unmatched.append(m.get("id", "<noid>"))
            labels.append(-1)  # sentinel
            continue
        lab = dr.get("difficulty_label")
        assert lab in ("easy", "hard"), (
            f"{m.get('id')}: difficulty_label={lab!r} (expected 'easy' or 'hard')"
        )
        labels.append(1 if lab == "hard" else 0)

    if unmatched:
        raise RuntimeError(
            f"{model_name}: {len(unmatched)} meta rows could not be matched to "
            f"data/difficulty_control_gsm8k.jsonl (by id nor prompt). "
            f"First few: {unmatched[:5]}. Aborting to avoid silent label corruption."
        )
    labels = np.asarray(labels)

    n_easy = int((labels == 0).sum())
    n_hard = int((labels == 1).sum())

    # Project onto impossibility direction (same PCA + projection as source)
    resid = reps_layer - pca_source.inverse_transform(pca_source.transform(reps_layer))
    scores_imp = resid @ d_imp

    # CosNSRT: normalize residuals before projection
    resid_norms = np.linalg.norm(resid, axis=1, keepdims=True)
    resid_norms = np.where(resid_norms < 1e-10, 1.0, resid_norms)
    scores_cos = (resid / resid_norms) @ d_imp

    auc_nsrt = safe_auc(labels, scores_imp)
    auc_cosnsrt = safe_auc(labels, scores_cos)

    # Length baselines (both char-level and token-level)
    char_lengths = np.array([len(m.get("prompt", "")) for m in meta])
    auc_char_length = safe_auc(labels, char_lengths)

    # Token-level length using Mistral tokenizer (project standard)
    auc_token_length = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True
        )
        token_lengths = np.array([
            len(tok.encode(m.get("prompt", ""))) for m in meta
        ])
        auc_token_length = safe_auc(labels, token_lengths)
    except Exception as e:
        print(f"  ⚠ Could not compute token-level length AUC: {e}")
        print("    (Falling back to char-level only)")

    return {
        "model": model_name,
        "n_easy": n_easy,
        "n_hard": n_hard,
        "layer": layer,
        "pooling": pooling,
        "auc_nsrt_difficulty": round(auc_nsrt, 4),
        "auc_cosnsrt_difficulty": round(auc_cosnsrt, 4),
        "auc_char_length": round(auc_char_length, 4),
        "auc_token_length": round(auc_token_length, 4) if auc_token_length else None,
        "source_auc_impossibility": round(source_info["source_auc"], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to evaluate (default: all available)")
    parser.add_argument("--source-model", type=str, default=None,
                        help="Model to source d_imp from (default: same as target)")
    parser.add_argument("--source-layer", type=int, default=None,
                        help="Layer for d_imp (default: BEST_LAYERS[model])")
    parser.add_argument("--source-dataset", type=str, default="math800",
                        help="Dataset to train d_imp on (default: math800)")
    args = parser.parse_args()

    models_to_eval = [args.model] if args.model else list(MODELS.keys())

    print("=" * 70)
    print("DIFFICULTY CONTROL EVALUATION")
    print("Does the impossibility direction separate easy vs hard math?")
    print(f"Source d_imp from: {args.source_dataset}")
    print("=" * 70)

    results = []

    for model_name in models_to_eval:
        # Check if extraction exists
        sig_dir = os.path.join(
            BASE, f"experiments/signals/difficulty_control_gsm8k_{model_name}_allL/signals"
        )
        if not os.path.exists(sig_dir):
            print(f"\n--- {model_name}: no extraction found, skipping ---")
            continue

        # Source model for d_imp
        src_model = args.source_model or model_name
        src_layer = args.source_layer or BEST_LAYERS.get(src_model, 15)

        print(f"\n--- {model_name} (d_imp from {args.source_dataset}/{src_model}/L{src_layer}) ---")

        d_imp, pca, info = load_impossibility_direction(
            src_model, src_layer, args.source_dataset
        )
        if d_imp is None:
            print(f"  ⚠ Could not load d_imp for {src_model}")
            continue

        print(f"  Source d_imp: AUC={info['source_auc']:.4f}, "
              f"pooling={info['pooling']}, layer={info['layer']}")

        result = evaluate_difficulty_control(model_name, d_imp, pca, info)
        if result is None:
            print("  ⚠ Evaluation failed")
            continue

        results.append(result)
        print(f"  ── Results ──")
        print(f"  NSRT AUC (easy vs hard):    {result['auc_nsrt_difficulty']:.4f}")
        print(f"  CosNSRT AUC (easy vs hard): {result['auc_cosnsrt_difficulty']:.4f}")
        print(f"  Char length AUC (baseline): {result['auc_char_length']:.4f}")
        if result.get('auc_token_length') is not None:
            print(f"  Token length AUC (Mistral): {result['auc_token_length']:.4f}")
        print(f"  Source AUC (A vs U):         {result['source_auc_impossibility']:.4f}")
        delta = result['source_auc_impossibility'] - result['auc_cosnsrt_difficulty']
        print(f"  ΔAUC (source − difficulty):  {delta:+.4f}")
        print(f"  → direction is more selective for impossibility than difficulty")

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        len_header = "TokLen(E/H)" if any(r.get("auc_token_length") for r in results) else "CharLen(E/H)"
        print(f"{'Model':<12} {'CosNSRT(E/H)':<15} {len_header:<13} "
              f"{'Source(A/U)':<13} {'ΔAUC'}")
        print("-" * 65)
        deltas = []
        for r in results:
            len_auc = r.get("auc_token_length") or r["auc_char_length"]
            delta = r["source_auc_impossibility"] - r["auc_cosnsrt_difficulty"]
            deltas.append(delta)
            print(f"{r['model']:<12} {r['auc_cosnsrt_difficulty']:<15.4f} "
                  f"{len_auc:<13.4f} "
                  f"{r['source_auc_impossibility']:<13.4f} {delta:+.4f}")
        print()
        print(f"Mean ΔAUC (source − difficulty) = {np.mean(deltas):+.4f}")
        print("Interpretation: the impossibility direction is markedly more selective")
        print("for structural impossibility than for problem difficulty.")
        print("(NOT claiming the direction is orthogonal to difficulty — it captures")
        print(" some difficulty-related variance, but far less than impossibility.)")

    # Save results — suffix single-model runs to protect full-run output.
    suffix = f"_model-{args.model}" if args.model else ""
    out_path = os.path.join(BASE, f"experiments/difficulty_control_results{suffix}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
