"""
Layer Emergence Analysis (F4): per-layer CosNSRT, SVM-NSRT, PC-space AUCs,
and NS_SNR. Reveals inverted-U shape (early build-up → mid peak → late decline)
and per-model peak layers.

Pipeline-aligned conventions (2026-04-15):
  - 7 core models × 2 core datasets (14 configs)
  - Auto-detect n_layers from reps shape (no hardcoded layer counts)
  - Prefer reps_last_token_all_layers.npy, fallback to reps_all_layers.npy
  - safe_auc = max(auc, 1-auc) (orientation-invariant)
  - 3 seeds (speed; emergence shape is robust to seed variance)
  - Output: experiments/layer_emergence_results{suffix}.json

Usage:
  python scripts/analyze_layer_emergence.py                    # all 7 models × 2 datasets
  python scripts/analyze_layer_emergence.py --model mistral   # single model
  python scripts/analyze_layer_emergence.py --dataset math800
"""
import argparse
import json
import os
import warnings

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(BASE))  # repo root (script lives in scripts/)

MODELS = [
    "smollm2", "gemma2", "phi3", "mistral", "qwen", "llama",
    "qwen14b", "olmo13b", "qwen32b",
    # Legacy Mistral-Small-24B-Instruct-2501 cell.
    "mistral_small",
    # Phase 1 upgrades (Mac-tier; 2026-04-29).
    "phi4mini", "gemma3_4b", "gemma3_12b", "qwen3_8b", "qwen3_14b",
    # Phase 2 / 3 upgrades (RunPod 2026-04-30).
    "qwen3_32b",
    "llama70b",
    # Mistral-Small-3.x successor (Phase 0 patch).
    "mistral_small_3_2",
]
CORE_DATASETS = ["math800", "code800"]
SEEDS = [42, 123, 456]  # 3 seeds for speed (emergence shape is robust)
K_PCA = 100


def load_all_layers(dataset, model):
    """Load all-layer reps preferring last-token over mean-pooled.

    Returns (reps, A_idx, U_idx, pooling, n_layers).
    """
    sig_dir = f"experiments/signals/{dataset}_{model}_allL/signals"
    lt_path = f"{sig_dir}/reps_last_token_all_layers.npy"
    mp_path = f"{sig_dir}/reps_all_layers.npy"

    if os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode="r")
        pooling = "last_token"
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode="r")
        pooling = "mean_pooled"
        print(f"  ⚠ {dataset}/{model}: last-token reps missing, using mean-pooled")
    else:
        return None, None, None, None, 0

    meta = [json.loads(l) for l in open(f"{sig_dir}/meta.jsonl")]
    labels = np.array([m["answerable"] for m in meta])
    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    return reps, A_idx, U_idx, pooling, reps.shape[1]


def split(A_idx, U_idx, seed):
    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[: len(A_idx) // 2]]
    teA = A_idx[pA[len(A_idx) // 2:]]
    trU = U_idx[pU[: len(U_idx) // 2]]
    teU = U_idx[pU[len(U_idx) // 2:]]
    te_idx = np.sort(np.concatenate([teA, teU]))
    te_labels = np.array([0 if i in set(teA) else 1 for i in te_idx])
    return trA, trU, te_idx, te_labels


def safe_auc(y, s):
    """Orientation-invariant AUC (max(auc, 1-auc))."""
    try:
        if len(np.unique(y)) < 2 or np.std(s) < 1e-15:
            return 0.5
        auc = roc_auc_score(y, s)
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def per_layer_metrics(X, A_idx, U_idx):
    """Compute null_md, null_svm, pc_md, full_md, ns_snr for one layer,
    averaged over SEEDS."""
    null_md_aucs, null_svm_aucs, pc_md_aucs, full_md_aucs = [], [], [], []
    snr_vals = []

    for seed in SEEDS:
        trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)
        tr_idx = np.concatenate([trA, trU])
        tr_labels = np.array([0] * len(trA) + [1] * len(trU))

        nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
        pca = PCA(n_components=nc).fit(X[trA])
        V = pca.components_

        R = X - X @ V.T @ V  # null-space residual
        P = X @ V.T @ V       # PC-space

        # NS_SNR: ||mu_u - mu_a||^2 / trace(Cov_within)
        mu_a_r = R[trA].mean(0)
        mu_u_r = R[trU].mean(0)
        d_md = mu_u_r - mu_a_r
        d_norm = norm(d_md)
        d_md_unit = d_md / (d_norm + 1e-15)

        cov_a = np.cov(R[trA].T) if len(trA) > 1 else np.zeros((R.shape[1], R.shape[1]))
        cov_u = np.cov(R[trU].T) if len(trU) > 1 else np.zeros((R.shape[1], R.shape[1]))
        tr_cov_w = (np.trace(cov_a) + np.trace(cov_u)) / 2
        snr = (d_norm ** 2) / (tr_cov_w + 1e-15)
        snr_vals.append(snr)

        # Null-space CosNSRT (mean-diff)
        norms_te = norm(R[te_idx], axis=1) + 1e-15
        scores = (R[te_idx] @ d_md_unit) / norms_te
        null_md_aucs.append(safe_auc(te_labels, scores))

        # Null-space SVM
        try:
            svm = LinearSVC(C=1.0, max_iter=3000, dual=True)
            svm.fit(R[tr_idx], tr_labels)
            d_svm = svm.coef_[0]
            d_svm /= (norm(d_svm) + 1e-15)
            scores_svm = (R[te_idx] @ d_svm) / norms_te
            null_svm_aucs.append(safe_auc(te_labels, scores_svm))
        except Exception:
            null_svm_aucs.append(0.5)

        # PC-space mean-diff
        mu_a_p = P[trA].mean(0)
        mu_u_p = P[trU].mean(0)
        d_md_p = mu_u_p - mu_a_p
        d_md_p = d_md_p / (norm(d_md_p) + 1e-15)
        norms_p = norm(P[te_idx], axis=1) + 1e-15
        scores_p = (P[te_idx] @ d_md_p) / norms_p
        pc_md_aucs.append(safe_auc(te_labels, scores_p))

        # Full-space mean-diff
        mu_a_f = X[trA].mean(0)
        mu_u_f = X[trU].mean(0)
        d_md_f = mu_u_f - mu_a_f
        d_md_f = d_md_f / (norm(d_md_f) + 1e-15)
        norms_f = norm(X[te_idx], axis=1) + 1e-15
        scores_f = (X[te_idx] @ d_md_f) / norms_f
        full_md_aucs.append(safe_auc(te_labels, scores_f))

    return {
        "null_md":  float(np.mean(null_md_aucs)),
        "null_svm": float(np.mean(null_svm_aucs)),
        "pc_md":    float(np.mean(pc_md_aucs)),
        "full_md":  float(np.mean(full_md_aucs)),
        "ns_snr":   float(np.mean(snr_vals)),
    }


def analyze_config(dataset, model):
    """Full layer sweep for one (dataset, model). Returns dict per-config."""
    out = load_all_layers(dataset, model)
    if out[0] is None:
        return None
    reps, A_idx, U_idx, pooling, n_layers = out

    print(f"\n{'='*90}")
    print(f"{dataset}/{model}  [n_layers={n_layers}, pooling={pooling}]")
    print(f"{'='*90}")
    print(f"{'Layer':>5}  {'NullMD':>8} {'NullSVM':>8} {'PC_MD':>8} {'FullMD':>8} {'NS_SNR':>8}")
    print("-" * 90)

    layer_rows = []
    for layer in range(n_layers):
        X = np.array(reps[:, layer, :], dtype=np.float32)
        m = per_layer_metrics(X, A_idx, U_idx)
        m["layer"] = layer
        layer_rows.append(m)
        print(f"{layer:>5}  {m['null_md']:>8.4f} {m['null_svm']:>8.4f} "
              f"{m['pc_md']:>8.4f} {m['full_md']:>8.4f} {m['ns_snr']:>8.2f}", flush=True)

    # Emergence: first layer where null_md > 0.7
    emergence = next((r["layer"] for r in layer_rows if r["null_md"] > 0.7), None)
    peak = max(layer_rows, key=lambda r: r["null_md"])
    gaps = [r["null_md"] - r["pc_md"] for r in layer_rows]
    max_gap_idx = int(np.argmax(gaps))
    mean_gap = float(np.mean(gaps))

    print(f"\n  Emergence (NullMD > 0.7): layer {emergence}")
    print(f"  Peak NullMD: layer {peak['layer']} (AUC={peak['null_md']:.4f})")
    print(f"  Avg Null-PC gap: {mean_gap:+.4f}")
    print(f"  Max Null-PC gap: {gaps[max_gap_idx]:+.4f} at layer {layer_rows[max_gap_idx]['layer']}")

    return {
        "dataset": dataset,
        "model": model,
        "n_layers": n_layers,
        "pooling": pooling,
        "emergence_layer": emergence,
        "peak_layer": peak["layer"],
        "peak_null_md": peak["null_md"],
        "mean_null_pc_gap": mean_gap,
        "max_null_pc_gap_layer": layer_rows[max_gap_idx]["layer"],
        "max_null_pc_gap": gaps[max_gap_idx],
        "layers": layer_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--dataset", choices=CORE_DATASETS, default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    datasets = [args.dataset] if args.dataset else CORE_DATASETS

    print("=" * 90)
    print("LAYER EMERGENCE ANALYSIS (F4, new pipeline)")
    print(f"Configs: {len(datasets)} datasets × {len(models)} models")
    print("=" * 90)

    results = []
    for dataset in datasets:
        for model in models:
            r = analyze_config(dataset, model)
            if r is not None:
                results.append(r)

    if not results:
        print("No configs evaluated.")
        return

    # Cross-config summary of peak layers
    print("\n" + "=" * 90)
    print("PEAK LAYER SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<10} {'Model':<10} {'Peak L':<8} {'Peak AUC':<10} "
          f"{'Emerge L':<10} {'Avg Null-PC gap':<16}")
    print("-" * 90)
    for r in results:
        print(f"{r['dataset']:<10} {r['model']:<10} {r['peak_layer']:<8} "
              f"{r['peak_null_md']:<10.4f} {str(r['emergence_layer']):<10} "
              f"{r['mean_null_pc_gap']:<+16.4f}")

    # Save JSON (subset-safe)
    suffix_parts = []
    if args.model:
        suffix_parts.append(f"model-{args.model}")
    if args.dataset:
        suffix_parts.append(f"ds-{args.dataset}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_path = f"experiments/layer_emergence_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_config": results,
            "models": MODELS,
            "datasets": CORE_DATASETS,
            "seeds": SEEDS,
            "k_pca": K_PCA,
            "cli_model_filter":   args.model,
            "cli_dataset_filter": args.dataset,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
