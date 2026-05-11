"""
eval_cross_domain_transfer.py — F8: Cross-domain transfer of the impossibility signal.

Train PCA + direction on math800 (or code800), evaluate on code800 (or math800).
If the signal is domain-general, cross-domain AUC should be substantially above
chance, though typically below within-domain.

Pipeline conventions (aligned with eval_natural_transfer / compare_impossibility):
  - 7 models: smollm2, gemma2, phi3, mistral, qwen, llama, qwen14b
  - Prefer reps_last_token_all_layers.npy; fallback to mean-pooled with warning
  - PCA(100 or less) fit on train-half of A-class only
  - d_imp = normalize(mean(U_resid) - mean(A_resid))
  - CosNSRT = (R / ||R||) @ d_imp  (equivalent to (R @ d_imp) / ||R||)
  - safe_auc = max(auc, 1-auc) (orientation-invariant)
  - HO-AU protocol: 50/50 split on train domain, full eval on test domain
  - 5 seeds: [42, 123, 456, 789, 2026]

Output:
  - Full run: experiments/cross_domain_transfer.json
  - Subset run (--model X or --dataset Y): experiments/cross_domain_transfer_{suffix}.json

Usage:
  python scripts/eval_cross_domain_transfer.py                        # all 7 models, both directions
  python scripts/eval_cross_domain_transfer.py --model mistral        # single model
  python scripts/eval_cross_domain_transfer.py --dataset math800      # only math→code
"""

import argparse
import json
import os
import warnings

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(BASE))  # repo root (script lives in scripts/)

# ── Pipeline constants (synced with compare_impossibility_vs_refusal_direction.py) ─
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
    "qwen32b": 0,  # PLACEHOLDER — set in STEP 3 after Phase 3 layer emergence. Suite always passes --layer, so this 0 is never read in the suite path; needed only to satisfy argparse choices=list(BEST_LAYERS.keys()).
    # Phase 1 upgrades (2026-04-29). Placeholder 0; suite passes --layer.
    "phi4mini":  0,
    "gemma3_4b": 0,
    "qwen3_8b":  0,
    "qwen3_14b": 0,
    # Phase 2 upgrades (2026-04-30).
    "qwen3_32b":         0,
    # Phase 3: 70B.
    "llama70b":          0,
}
SEEDS = [42, 123, 456, 789, 2026]
K_PCA = 100
DATASET_PAIRS = [("math800", "code800"), ("code800", "math800")]


def load(dataset, model, layer):
    """Load reps at given layer, preferring last-token over mean-pooled.

    Returns (X, A_idx, U_idx, pooling).
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
        print(f"  ⚠ {dataset}/{model}: last_token reps unavailable, falling back to mean-pooled")
    else:
        return None, None, None, None

    # Clip layer if extraction has fewer layers (e.g., partial or smaller model)
    if layer >= reps.shape[1]:
        print(f"  ⚠ Layer {layer} >= n_layers {reps.shape[1]}, clipping to {reps.shape[1] - 1}")
        layer = reps.shape[1] - 1

    meta = [json.loads(l) for l in open(f"{sig_dir}/meta.jsonl")]
    labels = np.array([m["answerable"] for m in meta])
    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    X = np.array(reps[:, layer, :], dtype=np.float32)
    del reps
    return X, A_idx, U_idx, pooling


def safe_auc(y, s):
    """Orientation-invariant AUC (robust to direction sign)."""
    try:
        if len(np.unique(y)) < 2 or np.std(s) < 1e-15:
            return 0.5
        auc = roc_auc_score(y, s)
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def evaluate_pair(model, train_name, test_name, X_tr, A_tr, U_tr, X_te, A_te, U_te):
    """For one (train, test) domain pair, run 5 seeds, return within+cross AUCs."""
    within_aucs, cross_aucs = [], []

    for seed in SEEDS:
        rng = np.random.RandomState(seed)

        # 50/50 split on train domain
        pA = rng.permutation(len(A_tr))
        pU = rng.permutation(len(U_tr))
        trA = A_tr[pA[: len(A_tr) // 2]]
        trU = U_tr[pU[: len(U_tr) // 2]]
        teA = A_tr[pA[len(A_tr) // 2:]]
        teU = U_tr[pU[len(U_tr) // 2:]]

        # Fit PCA on train-A only
        nc = min(K_PCA, len(trA) - 1, X_tr.shape[1] - 1)
        pca = PCA(n_components=nc).fit(X_tr[trA])

        # Residuals in train domain
        R_tr = X_tr - X_tr @ pca.components_.T @ pca.components_

        # Direction: normalize(mean(U_resid) - mean(A_resid))
        mu_diff = R_tr[trU].mean(0) - R_tr[trA].mean(0)
        d_mean = mu_diff / (norm(mu_diff) + 1e-15)

        # ── Within-domain AUC (test half of train domain) ──
        te_within_idx = np.concatenate([teA, teU])
        te_within_labels = np.concatenate([np.zeros(len(teA)), np.ones(len(teU))])
        R_within = R_tr[te_within_idx]
        norms_w = norm(R_within, axis=1) + 1e-15
        # CosNSRT: equivalent to (R/||R||) @ d_mean
        scores_w = (R_within @ d_mean) / norms_w
        within_aucs.append(safe_auc(te_within_labels, scores_w))

        # ── Cross-domain AUC (project test domain through SAME PCA) ──
        R_cross = X_te - X_te @ pca.components_.T @ pca.components_
        cross_idx = np.concatenate([A_te, U_te])
        cross_labels = np.concatenate([np.zeros(len(A_te)), np.ones(len(U_te))])
        R_cross_eval = R_cross[cross_idx]
        norms_c = norm(R_cross_eval, axis=1) + 1e-15
        scores_c = (R_cross_eval @ d_mean) / norms_c
        cross_aucs.append(safe_auc(cross_labels, scores_c))

    return {
        "model": model,
        "train": train_name,
        "test": test_name,
        "within_mean": float(np.mean(within_aucs)),
        "within_std":  float(np.std(within_aucs)),
        "cross_mean":  float(np.mean(cross_aucs)),
        "cross_std":   float(np.std(cross_aucs)),
        "drop":        float(np.mean(within_aucs) - np.mean(cross_aucs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BEST_LAYERS.keys()), default=None)
    parser.add_argument("--dataset", choices=["math800", "code800"], default=None,
                        help="If set, only use this as train domain (still evaluates on the other)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Override BEST_LAYERS for a single --model run.")
    args = parser.parse_args()
    if args.layer is not None and args.model is None:
        raise ValueError("--layer override requires --model")

    models = [args.model] if args.model else list(BEST_LAYERS.keys())
    dataset_pairs = DATASET_PAIRS
    if args.dataset:
        dataset_pairs = [p for p in DATASET_PAIRS if p[0] == args.dataset]

    print("=" * 80)
    print("CROSS-DOMAIN TRANSFER VERIFICATION (F8)")
    print("Train on domain A → Evaluate on domain B")
    print("Same-model source, PCA fit on train-A, CosNSRT scoring")
    print("=" * 80)

    all_results = []

    for model in models:
        layer = args.layer if (args.layer is not None and model == args.model) else BEST_LAYERS[model]
        print(f"\n--- {model} / L{layer} ---")

        # Preload both domains
        math_data = load("math800", model, layer)
        code_data = load("code800", model, layer)
        if any(x is None for x in (math_data[0], code_data[0])):
            print(f"  ⚠ Missing extraction, skipping {model}")
            continue

        X_math, A_math, U_math, pool_math = math_data
        X_code, A_code, U_code, pool_code = code_data
        if pool_math != pool_code:
            print(f"  ⚠ pooling mismatch between datasets: math={pool_math}, code={pool_code}")
        print(f"  pooling: {pool_math}")

        for train_name, test_name in dataset_pairs:
            if train_name == "math800":
                X_tr, A_tr, U_tr = X_math, A_math, U_math
                X_te, A_te, U_te = X_code, A_code, U_code
            else:
                X_tr, A_tr, U_tr = X_code, A_code, U_code
                X_te, A_te, U_te = X_math, A_math, U_math

            r = evaluate_pair(model, train_name, test_name,
                              X_tr, A_tr, U_tr, X_te, A_te, U_te)
            r["pooling"] = pool_math
            all_results.append(r)

            print(f"  Train={train_name} → Test={test_name}:  "
                  f"within={r['within_mean']:.4f}±{r['within_std']:.4f}  "
                  f"cross={r['cross_mean']:.4f}±{r['cross_std']:.4f}  "
                  f"drop={r['drop']:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_results:
        print(f"{'Model':<10} {'Train→Test':<25} {'Within AUC':>15} {'Cross AUC':>15} {'Drop':>10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['model']:<10} {r['train']+'→'+r['test']:<25} "
                  f"{r['within_mean']:>10.4f}±{r['within_std']:.4f} "
                  f"{r['cross_mean']:>10.4f}±{r['cross_std']:.4f} "
                  f"{r['drop']:>+10.4f}")

        avg_within = np.mean([r["within_mean"] for r in all_results])
        avg_cross  = np.mean([r["cross_mean"] for r in all_results])
        print("-" * 80)
        print(f"{'AVERAGE':<10} {'':<25} "
              f"{avg_within:>10.4f}{'':>6} "
              f"{avg_cross:>10.4f}{'':>6} "
              f"{avg_within - avg_cross:>+10.4f}")

    # ── Output (with subset-safe suffix) ──
    suffix_parts = []
    if args.model:
        suffix_parts.append(f"model-{args.model}")
    if args.dataset:
        suffix_parts.append(f"ds-{args.dataset}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_path = f"experiments/cross_domain_transfer{suffix}.json"

    # Fail-fast on empty results when a single --model was filtered in.
    # Without this guard the script wrote `[]` and the suite's run_if_missing
    # treated that empty marker as "done", silently skipping the cell on retry
    # once signals existed. (Same fix applied to analyze_form_conditionality.)
    if args.model and not all_results:
        raise SystemExit(
            f"--model {args.model!r}: cross-domain results empty. "
            f"Most common cause: extraction signals at "
            f"experiments/signals/{{math800,code800}}_{args.model}_allL/signals/ "
            f"are missing or incomplete. Refusing to write a stub marker file."
        )

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
