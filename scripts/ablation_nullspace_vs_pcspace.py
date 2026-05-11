"""
Ablation: Null-space vs PC-space vs Full-space detection (F3).

Hypothesis: Answerability signal concentrates in the null-space of the
answerable class, NOT in the principal component space.

Three conditions × 2 direction types:
  Representations:
    1. NULL-SPACE:  r = (I - V_k V_k^T) x  → residual after removing A-class PCs
    2. PC-SPACE:    p = V_k V_k^T x         → projection onto A-class PC subspace
    3. FULL-SPACE:  x raw
  Directions:
    A. SVM (LinearSVC on labels) — may have capacity confound (null-space is ~3900d)
    B. MeanDiff (mu_U - mu_A, 1-D) — no capacity confound; preferred for paper

Pipeline-aligned conventions (2026-04-15):
  - 7 core models × 2 core datasets (14 configs)
  - BEST_LAYERS = {smollm2:11, gemma2:16, phi3:15, mistral:15,
                   qwen:18, llama:15, qwen14b:34}
  - Prefer reps_last_token_all_layers.npy, fallback to reps_all_layers.npy
  - safe_auc = max(auc, 1-auc) (orientation-invariant)
  - 5 seeds, PCA(100 or smaller)
  - Output: experiments/ablation_nullpc_results{suffix}.json

Non-core datasets (abstentionbench/falseqa/mathtrap) intentionally excluded:
they don't belong to the main null-space-vs-PC-space argument.

Usage:
  python scripts/ablation_nullspace_vs_pcspace.py                  # full run (14 configs)
  python scripts/ablation_nullspace_vs_pcspace.py --model mistral  # smoke test single model
  python scripts/ablation_nullspace_vs_pcspace.py --dataset math800
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

BEST_LAYERS = {
    "smollm2": 11,
    "gemma2":  16,
    "phi3":    15,
    "mistral": 15,
    "qwen":    18,
    "llama":   15,
    "qwen14b": 34,
    "mistral_small": 28,
    "olmo13b": 20,
    "qwen32b": 0,  # PLACEHOLDER — set in STEP 3 after Phase 3 layer emergence (peak from layer_emergence_results_model-qwen32b_ds-math800.json). Suite always passes --layer override, so this 0 is never read in the suite path; needed only to satisfy argparse choices=list(BEST_LAYERS.keys()).
    # Phase 1 upgrades (2026-04-29). Same convention: 0 is a placeholder; suite
    # passes --layer explicitly via $MATH_LAYER / $CODE_LAYER from Phase 3.
    "phi4mini":  0,
    "gemma3_4b": 0,
    "gemma3_12b": 0,
    "qwen3_8b":  0,
    "qwen3_14b": 0,
    # Phase 2 upgrades (2026-04-30). Same placeholder convention.
    "qwen3_32b":         0,
    # Phase 3: 70B.
    "llama70b":          0,
    # Mistral-Small-3.x successor (Phase 0 patch). Layer inherited from
    # Mistral-Small-2501 24B layout; re-validate via layer scan if needed.
    "mistral_small_3_2": 28,
}
SEEDS = [42, 123, 456, 789, 2026]
K_PCA = 100
CORE_DATASETS = ["math800", "code800"]


def load(dataset, model, layer):
    """Load reps at given layer, preferring last-token over mean-pooled."""
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
        return None, None, None, None

    if layer >= reps.shape[1]:
        print(f"  ⚠ Layer {layer} >= n_layers {reps.shape[1]}, clipping to last")
        layer = reps.shape[1] - 1

    meta = [json.loads(l) for l in open(f"{sig_dir}/meta.jsonl")]
    labels = np.array([m["answerable"] for m in meta])
    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    X = np.array(reps[:, layer, :], dtype=np.float32)
    return X, A_idx, U_idx, pooling


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
    """Orientation-invariant AUC."""
    try:
        if len(np.unique(y)) < 2 or np.std(s) < 1e-15:
            return 0.5
        auc = roc_auc_score(y, s)
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def svm_cos_auc(X_train, y_train, X_test, y_test):
    """Train SVM, score test by cosine with SVM direction."""
    svm = LinearSVC(C=1.0, max_iter=5000, dual=True)
    svm.fit(X_train, y_train)
    d = svm.coef_[0]
    d = d / (norm(d) + 1e-15)
    norms = norm(X_test, axis=1) + 1e-15
    scores = (X_test @ d) / norms
    return safe_auc(y_test, scores)


def meandiff_cos_auc(X_train, y_train, X_test, y_test):
    """Mean-diff direction + cosine scoring (no capacity confound)."""
    mu_a = X_train[y_train == 0].mean(axis=0)
    mu_u = X_train[y_train == 1].mean(axis=0)
    d = mu_u - mu_a
    d = d / (norm(d) + 1e-15)
    norms = norm(X_test, axis=1) + 1e-15
    scores = (X_test @ d) / norms
    return safe_auc(y_test, scores)


def evaluate_config(dataset, model, layer, include_svm=True):
    """Return dict with null/pc/full AUCs for MeanDiff (+ optional SVM), averaged over seeds.

    include_svm=True:  compute both MeanDiff and SVM (slow, ~2 hours full run on CPU)
    include_svm=False: compute only MeanDiff (fast, ~2 minutes full run)
    """
    out = load(dataset, model, layer)
    if any(x is None for x in out):
        return None
    X, A_idx, U_idx, pooling = out

    null_svm, pc_svm, full_svm = [], [], []
    null_md, pc_md, full_md = [], [], []
    nc_last, null_dim_last = 0, 0

    for seed in SEEDS:
        trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)

        nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
        pca = PCA(n_components=nc).fit(X[trA])
        V = pca.components_  # (nc, D)

        # Three representations
        R = X - X @ V.T @ V   # null-space
        P = X @ V.T @ V       # PC-space
        # Full space: X raw

        tr_idx = np.concatenate([trA, trU])
        tr_labels = np.array([0] * len(trA) + [1] * len(trU))

        if include_svm:
            null_svm.append(svm_cos_auc(R[tr_idx], tr_labels, R[te_idx], te_labels))
            pc_svm.append(svm_cos_auc(P[tr_idx], tr_labels, P[te_idx], te_labels))
            full_svm.append(svm_cos_auc(X[tr_idx], tr_labels, X[te_idx], te_labels))

        null_md.append(meandiff_cos_auc(R[tr_idx], tr_labels, R[te_idx], te_labels))
        pc_md.append(meandiff_cos_auc(P[tr_idx], tr_labels, P[te_idx], te_labels))
        full_md.append(meandiff_cos_auc(X[tr_idx], tr_labels, X[te_idx], te_labels))

        nc_last = nc
        null_dim_last = X.shape[1] - nc

    result = {
        "dataset": dataset, "model": model, "layer": int(layer),
        "pooling": pooling, "D": int(X.shape[1]),
        "pc_dim": int(nc_last), "null_dim": int(null_dim_last),
        "null_md":   float(np.mean(null_md)),
        "pc_md":     float(np.mean(pc_md)),
        "full_md":   float(np.mean(full_md)),
        "include_svm": bool(include_svm),
    }
    if include_svm:
        result["null_svm"]  = float(np.mean(null_svm))
        result["pc_svm"]    = float(np.mean(pc_svm))
        result["full_svm"]  = float(np.mean(full_svm))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BEST_LAYERS.keys()), default=None)
    parser.add_argument("--dataset", choices=CORE_DATASETS, default=None)
    parser.add_argument("--layer", type=int, default=None,
                        help="Override BEST_LAYERS for a single --model run.")
    parser.add_argument("--include-svm", action="store_true",
                        help="Also compute SVM direction (slow, ~2h full run on CPU). "
                             "Default: only MeanDiff (fast, ~2min full run). "
                             "MeanDiff is the main paper claim; SVM is appendix-only.")
    args = parser.parse_args()

    models = [args.model] if args.model else list(BEST_LAYERS.keys())
    datasets = [args.dataset] if args.dataset else CORE_DATASETS

    if args.layer is not None and args.model is None:
        raise ValueError("--layer override requires --model")

    configs = [
        (ds, m, args.layer if (args.layer is not None and m == args.model) else BEST_LAYERS[m])
        for ds in datasets for m in models
    ]

    mode = "MeanDiff + SVM (slow)" if args.include_svm else "MeanDiff only (fast)"
    print("=" * 100)
    print(f"ABLATION: Null-Space vs PC-Space vs Full-Space (F3, new pipeline)")
    print(f"Mode: {mode}")
    print(f"Configs: {len(configs)} ({len(datasets)} datasets × {len(models)} models)")
    print("=" * 100)

    all_results = []

    import time
    for i, (dataset, model, layer) in enumerate(configs, 1):
        cfg = f"{dataset}/{model}/L{layer}"
        t0 = time.time()
        print(f"\n[{i}/{len(configs)}] {cfg}  (running...)", flush=True)
        r = evaluate_config(dataset, model, layer, include_svm=args.include_svm)
        if r is None:
            print(f"  ⚠ extraction missing, skipping")
            continue
        dt = time.time() - t0

        print(f"  [D={r['D']}, nc={r['pc_dim']}, null_dim={r['null_dim']}, "
              f"pool={r['pooling']}, elapsed={dt:.1f}s]")
        print(f"  {'':20} {'Null':>9} {'PC':>9} {'Full':>9} {'Null-PC':>9}")
        if args.include_svm:
            print(f"  {'SVM direction':20} {r['null_svm']:>9.4f} {r['pc_svm']:>9.4f} "
                  f"{r['full_svm']:>9.4f} {r['null_svm']-r['pc_svm']:>+9.4f}")
        print(f"  {'MeanDiff direction':20} {r['null_md']:>9.4f} {r['pc_md']:>9.4f} "
              f"{r['full_md']:>9.4f} {r['null_md']-r['pc_md']:>+9.4f}")

        all_results.append(r)

    if not all_results:
        print("No configs evaluated.")
        return

    # Summary tables
    avg = lambda k: float(np.mean([r[k] for r in all_results]))

    if args.include_svm:
        print("\n" + "=" * 100)
        print("SUMMARY — SVM Direction")
        print("=" * 100)
        print(f"{'Config':<35} {'Null':>9} {'PC':>9} {'Full':>9} {'Null-PC':>9}")
        print("-" * 100)
        for r in all_results:
            cfg = f"{r['dataset']}/{r['model']}/L{r['layer']}"
            print(f"{cfg:<35} {r['null_svm']:>9.4f} {r['pc_svm']:>9.4f} "
                  f"{r['full_svm']:>9.4f} {r['null_svm']-r['pc_svm']:>+9.4f}")
        print("-" * 100)
        print(f"{'AVERAGE':<35} {avg('null_svm'):>9.4f} {avg('pc_svm'):>9.4f} "
              f"{avg('full_svm'):>9.4f} {avg('null_svm')-avg('pc_svm'):>+9.4f}")

    print("\n" + "=" * 100)
    print("SUMMARY — MeanDiff Direction (no capacity confound, main paper claim)")
    print("=" * 100)
    print(f"{'Config':<35} {'Null':>9} {'PC':>9} {'Full':>9} {'Null-PC':>9}")
    print("-" * 100)
    for r in all_results:
        cfg = f"{r['dataset']}/{r['model']}/L{r['layer']}"
        print(f"{cfg:<35} {r['null_md']:>9.4f} {r['pc_md']:>9.4f} "
              f"{r['full_md']:>9.4f} {r['null_md']-r['pc_md']:>+9.4f}")
    print("-" * 100)
    print(f"{'AVERAGE':<35} {avg('null_md'):>9.4f} {avg('pc_md'):>9.4f} "
          f"{avg('full_md'):>9.4f} {avg('null_md')-avg('pc_md'):>+9.4f}")

    # Counts
    n = len(all_results)
    n_md_np    = sum(1 for r in all_results if r["null_md"]  > r["pc_md"])
    n_md_nf    = sum(1 for r in all_results if r["null_md"]  > r["full_md"])
    print(f"\nCounts across {n} configs:")
    print(f"  MeanDiff : Null > PC   {n_md_np}/{n}     Null > Full {n_md_nf}/{n}")
    if args.include_svm:
        n_svm_np = sum(1 for r in all_results if r["null_svm"] > r["pc_svm"])
        n_svm_nf = sum(1 for r in all_results if r["null_svm"] > r["full_svm"])
        print(f"  SVM      : Null > PC   {n_svm_np}/{n}     Null > Full {n_svm_nf}/{n}")

    # Save JSON (subset-safe output)
    suffix_parts = []
    if args.model:
        suffix_parts.append(f"model-{args.model}")
    if args.dataset:
        suffix_parts.append(f"ds-{args.dataset}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_path = f"experiments/ablation_nullpc_results{suffix}.json"

    summary = {
        "per_config": all_results,
        "averages": {
            "meandiff": {
                "null":  avg("null_md"),
                "pc":    avg("pc_md"),
                "full":  avg("full_md"),
            },
        },
        "counts": {
            "n_configs":        n,
            "md_null_gt_pc":    n_md_np,
            "md_null_gt_full":  n_md_nf,
        },
        "best_layers": BEST_LAYERS,
        "seeds": SEEDS,
        "k_pca": K_PCA,
        "cli_model_filter":   args.model,
        "cli_dataset_filter": args.dataset,
        "include_svm":        bool(args.include_svm),
    }
    if args.include_svm:
        summary["averages"]["svm"] = {
            "null":  avg("null_svm"),
            "pc":    avg("pc_svm"),
            "full":  avg("full_svm"),
        }
        summary["counts"]["svm_null_gt_pc"]   = n_svm_np
        summary["counts"]["svm_null_gt_full"] = n_svm_nf

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
