"""
Quick verification: do core conclusions hold across all models and datasets?
Uses EXISTING extracted signals — no model inference needed.

Core conclusions to verify:
  C1. Detection: CosNSRT AUC >> 0.5 (LLMs encode impossibility)
  C2. Orthogonality: impossibility direction ⊥ refusal direction (cos ≈ 0.09-0.17)
      — only 3 models have refusal direction data, but we can check cross-dataset
        direction cosines as a proxy for all models
  C3. Steering: hallucination reduction > 0 with moderate non-refusal cost

Run: python scripts/verify_core_conclusions.py
"""
import json, os, glob
import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import norm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

def compute_cosnsrt_auc(reps_path, meta_path, layer, k_pca=100, seed=42):
    """CosNSRT detection AUC at a given layer."""
    reps = np.load(reps_path, mmap_mode='r')
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m['answerable'] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    # Remove NaN rows
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X = X[~nan_mask]
        labels = labels[~nan_mask]

    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]

    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))

    # 50/50 train/test split
    trA = A_idx[pA[:len(A_idx)//2]]
    teA = A_idx[pA[len(A_idx)//2:]]
    trU = U_idx[pU[:len(U_idx)//2]]
    teU = U_idx[pU[len(U_idx)//2:]]

    # PCA on train-A
    nc = min(k_pca, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc).fit(X[trA])
    V = pca.components_
    R = X - X @ V.T @ V

    # Direction from train split
    mu_diff = R[trU].mean(0) - R[trA].mean(0)
    d_hat = mu_diff / (norm(mu_diff) + 1e-15)

    # Score test samples
    test_idx = np.concatenate([teA, teU])
    test_labels = np.concatenate([np.zeros(len(teA)), np.ones(len(teU))])
    scores = R[test_idx] @ d_hat

    # AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_labels, scores)
    return auc


def compute_direction(reps_path, meta_path, layer, k_pca=100, seed=42):
    """Compute impossibility direction (for cosine comparison)."""
    reps = np.load(reps_path, mmap_mode='r')
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m['answerable'] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X = X[~nan_mask]
        labels = labels[~nan_mask]

    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]

    rng = np.random.RandomState(seed)
    trA = A_idx[rng.permutation(len(A_idx))[:len(A_idx)//2]]
    trU = U_idx[rng.permutation(len(U_idx))[:len(U_idx)//2]]

    nc = min(k_pca, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc).fit(X[trA])
    V = pca.components_
    R = X - X @ V.T @ V

    mu_diff = R[trU].mean(0) - R[trA].mean(0)
    d_hat = mu_diff / (norm(mu_diff) + 1e-15)
    return d_hat


# ── BEST_LAYERS (same as steering script) ──
BEST_LAYERS = {
    ("math800", "mistral"): 15, ("math800", "llama"): 15, ("math800", "qwen"): 18,
    ("code800", "mistral"): 15, ("code800", "llama"): 14, ("code800", "qwen"): 18,
    ("math800", "mistral_small"): 28, ("code800", "mistral_small"): 20,
    ("math800", "qwen14b"): 34, ("code800", "qwen14b"): 32,
    ("math800", "phi3"): 15, ("code800", "phi3"): 16,
    ("math800", "smollm2"): 11, ("code800", "smollm2"): 14,
    ("math800", "gemma2"): 16, ("code800", "gemma2"): 14,
    ("fact800", "llama"): 15, ("fact800", "mistral"): 17, ("fact800", "qwen"): 19,
    ("fact800", "qwen14b"): 34, ("fact800", "phi3"): 15,
    ("fact800", "smollm2"): 11, ("fact800", "gemma2"): 16,
}

def main():
    print("=" * 90)
    print("QUICK VERIFICATION: Core Conclusions Across All Models")
    print("=" * 90)

    # ── C1: Detection AUC ──
    print("\n" + "─" * 90)
    print("C1. DETECTION: CosNSRT AUC (held-out test, PCA on train-A)")
    print("─" * 90)

    datasets = ["math800", "code800", "fact800"]
    models = ["smollm2", "gemma2", "phi3", "mistral", "qwen", "llama", "qwen14b", "mistral_small"]

    # Header
    print(f"\n{'Model':>14s}", end="")
    for ds in datasets:
        print(f"  {ds:>10s}", end="")
    print()
    print("-" * 50)

    auc_table = {}
    for model in models:
        row = {}
        print(f"{model:>14s}", end="")
        for ds in datasets:
            sig_dir = os.path.join(BASE, f"experiments/signals/{ds}_{model}_allL/signals")
            reps_path = os.path.join(sig_dir, "reps_all_layers.npy")
            meta_path = os.path.join(sig_dir, "meta.jsonl")

            if not os.path.exists(reps_path):
                print(f"  {'—':>10s}", end="")
                row[ds] = None
                continue

            layer = BEST_LAYERS.get((ds, model), 15)
            try:
                auc = compute_cosnsrt_auc(reps_path, meta_path, layer)
                print(f"  {auc:>10.3f}", end="")
                row[ds] = auc
            except Exception as e:
                print(f"  {'ERR':>10s}", end="")
                row[ds] = None
        auc_table[model] = row
        print()

    # Summary
    print(f"\n  All math800 AUCs > 0.85?", end="")
    math_aucs = [v["math800"] for v in auc_table.values() if v.get("math800")]
    print(f"  min={min(math_aucs):.3f}  {'YES ✓' if min(math_aucs) > 0.85 else 'NO ✗'}")

    print(f"  All code800 AUCs > 0.85?", end="")
    code_aucs = [v["code800"] for v in auc_table.values() if v.get("code800")]
    print(f"  min={min(code_aucs):.3f}  {'YES ✓' if min(code_aucs) > 0.85 else 'NO ✗'}")

    fact_aucs = [v["fact800"] for v in auc_table.values() if v.get("fact800")]
    if fact_aucs:
        print(f"  All fact800 AUCs > 0.80?", end="")
        print(f"  min={min(fact_aucs):.3f}  {'YES ✓' if min(fact_aucs) > 0.80 else 'NO ✗'}")

    # ── C2: Orthogonality ──
    print("\n" + "─" * 90)
    print("C2. ORTHOGONALITY: cos(impossibility_direction, refusal_direction)")
    print("─" * 90)

    # Load existing direction_comparison results
    print("\n  From pre-computed direction_comparison (imp vs Arditi refusal):")
    for model in ["mistral", "llama", "qwen"]:
        fpath = os.path.join(BASE, f"experiments/direction_comparison_{model}.json")
        if os.path.exists(fpath):
            d = json.load(open(fpath))
            cos_full = d.get("cos_imp_refusal_full", "N/A")
            cos_null = d.get("cos_imp_refusal_null", "N/A")
            print(f"    {model:>10s}: cos_full={cos_full:.4f}  cos_null={cos_null:.4f}")

    # Cross-dataset direction cosines as proxy — works for ALL models
    print("\n  Cross-dataset direction cosines (proxy for all models):")
    print(f"  {'Model':>14s}  {'cos(math,code)':>14s}  {'cos(math,fact)':>14s}  {'cos(code,fact)':>14s}")
    print("  " + "-" * 50)

    for model in models:
        dirs = {}
        for ds in datasets:
            sig_dir = os.path.join(BASE, f"experiments/signals/{ds}_{model}_allL/signals")
            reps_path = os.path.join(sig_dir, "reps_all_layers.npy")
            meta_path = os.path.join(sig_dir, "meta.jsonl")
            if not os.path.exists(reps_path):
                continue
            layer = BEST_LAYERS.get((ds, model), 15)
            try:
                dirs[ds] = compute_direction(reps_path, meta_path, layer)
            except:
                pass

        print(f"  {model:>14s}", end="")
        for pair in [("math800", "code800"), ("math800", "fact800"), ("code800", "fact800")]:
            if pair[0] in dirs and pair[1] in dirs:
                d1, d2 = dirs[pair[0]], dirs[pair[1]]
                # Ensure same dimensionality
                if len(d1) == len(d2):
                    cos = np.dot(d1, d2)
                    print(f"  {cos:>14.4f}", end="")
                else:
                    print(f"  {'dim≠':>14s}", end="")
            else:
                print(f"  {'—':>14s}", end="")
        print()

    # ── C3: Steering results ──
    print("\n" + "─" * 90)
    print("C3. STEERING: Hallucination reduction at best α")
    print("─" * 90)

    print(f"\n{'Model':>14s} {'Dataset':>10s} {'Base Halluc':>12s} {'Best Halluc':>12s} {'Reduction':>10s} {'NonRef Cost':>12s} {'Best α':>8s}")
    print("-" * 80)

    for ds in datasets:
        for model in models:
            pattern = os.path.join(BASE, f"experiments/steering/steering_{model}_{ds}_L*.json")
            files = glob.glob(pattern)
            if not files:
                continue
            f = files[0]
            try:
                d = json.load(open(f))
                ba = d.get("best_alpha", "?")
                # Handle both old and new metric names
                baseline = d["results_by_alpha"]["0.0"]["metrics"]["impossibility"]
                best_key = str(float(ba))
                best = d["results_by_alpha"][best_key]["metrics"]["impossibility"]

                base_h = baseline["hallucination_rate_U"]
                best_h = best["hallucination_rate_U"]
                reduction = base_h - best_h

                # Try new name first, fall back to old
                base_p = baseline.get("non_refusal_rate_A", baseline.get("preserve_rate_A", "?"))
                best_p = best.get("non_refusal_rate_A", best.get("preserve_rate_A", "?"))
                if isinstance(base_p, (int, float)) and isinstance(best_p, (int, float)):
                    cost = base_p - best_p
                else:
                    cost = "?"

                print(f"{model:>14s} {ds:>10s} {base_h:>12.3f} {best_h:>12.3f} {reduction:>10.3f} {cost:>12.3f} {ba:>8.0f}")
            except Exception as e:
                pass

    print("\n" + "=" * 90)
    print("END OF VERIFICATION")
    print("=" * 90)


if __name__ == "__main__":
    main()
