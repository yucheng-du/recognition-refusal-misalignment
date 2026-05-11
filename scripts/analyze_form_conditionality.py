"""
Form-Conditionality Deep Analysis (F5)

1. Per-category AUC breakdown (16 MATH categories, 8 CODE categories)
2. Attractor tightness τ per category (within-class mean cosine distance of A-class reps)
3. Additional predictors: intrinsic dimensionality, class separation (Cohen's d),
   spectral decay rate, null-space signal-to-noise ratio (NS_SNR — the main
   within-dataset predictor)
4. Cross-dataset comparison (math800/code800/fact800/abstentionbench_gsm8k/falseqa;
   mathtrap intentionally excluded — A-side duplicates).

Pipeline-aligned conventions:
  - 7 models × 2 core datasets, BEST_LAYERS[model] per config
  - reps_last_token_all_layers.npy preferred over reps_all_layers.npy
  - CosNSRT verified baseline, HO-AU protocol, 5 seeds

Output:
  - stdout: per-model per-category tables + correlations
  - experiments/form_conditionality_results.json: structured results for paper tables
"""
import argparse, numpy as np, json, os, re, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

# ── CLI (subset-safe: --model filters, adds suffix to output) ──
_cli = argparse.ArgumentParser()
_cli.add_argument("--model", default=None,
                  help="Optional: restrict PART 1 to one model (smoke test).")
_cli.add_argument("--dataset", default=None, choices=["math800", "code800"],
                  help="Optional: restrict PART 1 to one core dataset.")
_cli.add_argument("--layer", type=int, default=None,
                  help="Override BEST_LAYERS for a single --model run.")
_args = _cli.parse_args()
if _args.layer is not None and _args.model is None:
    raise ValueError("--layer override requires --model")


def extract_category(sample_id):
    """Extract category prefix from sample ID.
    Math IDs: mdiv_001a, msqrt_002u, mno_r003a, etc.
    Code IDs: ctype001a, czero002u, etc.
    """
    match = re.match(r'^([a-zA-Z_]+?)_?\d', sample_id)
    if match:
        return match.group(1)
    return 'unknown'

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(BASE))  # repo root (script lives in scripts/)

SEEDS = [42, 123, 456, 789, 2026]
K_PCA = 100

# Pipeline-aligned best-per-model layers (synced with eval_natural_transfer,
# compare_impossibility_vs_refusal_direction, etc.).
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
    "qwen32b": 0,  # PLACEHOLDER — set in STEP 3 after Phase 3 layer emergence. Suite always passes --layer override, so this 0 is bypassed by the conditional in the _all_configs list comprehension when --model qwen32b is set.
    # Phase 1 upgrades (2026-04-29). Placeholder 0; suite passes --layer.
    "phi4mini":  0,
    "gemma3_4b": 0,
    "gemma3_12b": 0,
    "qwen3_8b":  0,
    "qwen3_14b": 0,
    # Phase 2 upgrades (2026-04-30).
    "qwen3_32b":         0,
    # Phase 3: 70B.
    "llama70b":          0,
    # Mistral-Small-3.x successor (Phase 0 patch). Layer inherited from
    # Mistral-Small-2501 24B layout; re-validate via layer scan if needed.
    "mistral_small_3_2": 28,
}

# Fail-fast guard: if a model is requested but not registered, refuse to
# write a stub marker file (previous behavior produced an empty
# experiments/form_conditionality_results_model-<m>_ds-<ds>.json that the
# suite's run_if_missing treated as "done", silently skipping the cell).
if _args.model is not None and _args.model not in BEST_LAYERS:
    raise SystemExit(
        f"--model {_args.model!r} is not registered in BEST_LAYERS. "
        f"Add it (placeholder layer 0 is fine — the suite passes --layer) "
        f"or use one of: {sorted(BEST_LAYERS.keys())}"
    )

# Per-category analysis: all 7 models × 2 core datasets (filterable via CLI)
_all_configs = [
    (ds, m, _args.layer if (_args.layer is not None and m == _args.model) else BEST_LAYERS[m])
    for ds in ("math800", "code800")
    for m in BEST_LAYERS.keys()
]
CATEGORY_CONFIGS = [
    (ds, m, L) for (ds, m, L) in _all_configs
    if (_args.model is None or m == _args.model)
    and (_args.dataset is None or ds == _args.dataset)
]

# Cross-dataset predictors: one model per dataset (mistral for readability).
# Layer follows same-model convention (BEST_LAYERS["mistral"]=15).
# mathtrap intentionally excluded (data quality issues; see prepare_difficulty_control.py notes).
CROSS_DATASET_CONFIGS = [
    ("math800",               "mistral", BEST_LAYERS["mistral"]),
    ("code800",               "mistral", BEST_LAYERS["mistral"]),
    ("fact800",               "mistral", BEST_LAYERS["mistral"]),
    ("abstentionbench_gsm8k", "mistral", BEST_LAYERS["mistral"]),
    ("falseqa",               "mistral", BEST_LAYERS["mistral"]),
]


def load(dataset, model, layer):
    """Load reps at given layer, preferring last-token over mean-pooled."""
    sig_dir = f"experiments/signals/{dataset}_{model}_allL/signals"
    lt_path = f"{sig_dir}/reps_last_token_all_layers.npy"
    mp_path = f"{sig_dir}/reps_all_layers.npy"

    if os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode='r')
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode='r')
        print(f"  ⚠ {dataset}/{model}: last-token reps missing, using mean-pooled")
    else:
        return None, None, None, None

    if layer >= reps.shape[1]:
        print(f"  ⚠ Layer {layer} >= n_layers {reps.shape[1]}, clipping to last")
        layer = reps.shape[1] - 1

    meta = [json.loads(l) for l in open(f"{sig_dir}/meta.jsonl")]
    labels = np.array([m['answerable'] for m in meta])
    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]
    X = np.array(reps[:, layer, :], dtype=np.float32)
    return X, A_idx, U_idx, meta


def split(A_idx, U_idx, seed):
    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx)); pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[:len(A_idx)//2]]; teA = A_idx[pA[len(A_idx)//2:]]
    trU = U_idx[pU[:len(U_idx)//2]]; teU = U_idx[pU[len(U_idx)//2:]]
    te_idx = np.sort(np.concatenate([teA, teU]))
    te_labels = np.array([0 if i in set(teA) else 1 for i in te_idx])
    return trA, trU, te_idx, te_labels


def safe_auc(y, s):
    if len(np.unique(y)) < 2 or np.std(s) < 1e-15:
        return 0.5
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def attractor_tightness(X_A):
    """τ = mean pairwise cosine distance among A-class representations."""
    if len(X_A) < 2:
        return float('nan')
    # Normalize
    norms = norm(X_A, axis=1, keepdims=True) + 1e-15
    X_norm = X_A / norms
    # Mean cosine distance = 1 - mean cosine similarity
    cos_sim = X_norm @ X_norm.T
    n = len(X_A)
    # Upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_cos_sim = cos_sim[mask].mean()
    return 1.0 - mean_cos_sim


def spectral_decay_rate(X_A, k=20):
    """Rate at which eigenvalues decay — fast decay = low intrinsic dim = tight attractor."""
    if len(X_A) < k + 1:
        return float('nan')
    pca = PCA(n_components=k).fit(X_A)
    explained = pca.explained_variance_ratio_
    # Ratio of top-1 to top-k
    return explained[0] / (explained[-1] + 1e-15)


def effective_dimensionality(X_A, threshold=0.95):
    """Number of PCA components needed to explain threshold% variance."""
    if len(X_A) < 5:
        return float('nan')
    nc = min(len(X_A) - 1, X_A.shape[1] - 1, 200)
    pca = PCA(n_components=nc).fit(X_A)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d_eff = np.searchsorted(cumvar, threshold) + 1
    return d_eff


def class_separation_cohen_d(X_A, X_U):
    """Cohen's d along mean-difference direction in full space."""
    mu_diff = X_U.mean(0) - X_A.mean(0)
    d_hat = mu_diff / (norm(mu_diff) + 1e-15)
    proj_A = X_A @ d_hat
    proj_U = X_U @ d_hat
    pooled_std = np.sqrt((proj_A.var() + proj_U.var()) / 2)
    if pooled_std < 1e-15:
        return 0.0
    return (proj_U.mean() - proj_A.mean()) / pooled_std


def null_space_snr(R_A, R_U):
    """Signal-to-noise ratio in null-space along mean-diff direction."""
    mu_diff = R_U.mean(0) - R_A.mean(0)
    d_hat = mu_diff / (norm(mu_diff) + 1e-15)
    proj_A = R_A @ d_hat
    proj_U = R_U @ d_hat
    signal = (proj_U.mean() - proj_A.mean()) ** 2
    noise = (proj_A.var() + proj_U.var()) / 2
    if noise < 1e-15:
        return float('inf')
    return signal / noise


# ═══════════════════════════════════════════════════════════════
# PART 1: Per-category AUC breakdown
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("PART 1: Per-Category AUC Breakdown")
print("=" * 80)

# Structured results accumulator (written as JSON at end of script)
all_part1_results = []  # list of per-config dicts
all_part2_results = []  # cross-dataset predictor table

for dataset, model, layer in CATEGORY_CONFIGS:
    cfg = f"{dataset}/{model}"
    print(f"\n--- {cfg} (L{layer}) ---")

    sig_dir = f"experiments/signals/{dataset}_{model}_allL/signals"
    lt_path = f"{sig_dir}/reps_last_token_all_layers.npy"
    mp_path = f"{sig_dir}/reps_all_layers.npy"
    if os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode='r')
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode='r')
        print(f"  ⚠ last-token reps missing, using mean-pooled")
    else:
        print(f"  ⚠ no reps found for {dataset}/{model}, skipping")
        continue

    if layer >= reps.shape[1]:
        layer = reps.shape[1] - 1

    meta = [json.loads(l) for l in open(f"{sig_dir}/meta.jsonl")]
    labels = np.array([m['answerable'] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    # Get categories from ID prefix
    categories = {}
    for i, m in enumerate(meta):
        cat = extract_category(m['id'])
        if cat not in categories:
            categories[cat] = {'A': [], 'U': []}
        categories[cat][labels[i]].append(i)

    A_idx = np.where(labels == 'A')[0]
    U_idx = np.where(labels == 'U')[0]

    # Global CosNSRT for reference
    global_aucs = []
    for seed in SEEDS:
        trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)
        nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
        pca = PCA(n_components=nc).fit(X[trA])
        R = X - X @ pca.components_.T @ pca.components_
        mu_diff = R[trU].mean(0) - R[trA].mean(0)
        d_mean = mu_diff / (norm(mu_diff) + 1e-15)
        R_te = R[te_idx]
        norms_te = norm(R_te, axis=1) + 1e-15
        scores = (R_te @ d_mean) / norms_te
        global_aucs.append(safe_auc(te_labels, scores))
    global_auc = np.mean(global_aucs)
    print(f"  Global CosNSRT: {global_auc:.4f}")

    # Per-category: use global PCA and direction, but evaluate AUC per category
    print(f"  {'Category':<25} {'N_A':>4} {'N_U':>4} {'AUC':>7} {'τ':>7} {'d_eff':>6} {'Cohen_d':>8} {'NS_SNR':>7}")

    cat_results = []
    for cat in sorted(categories.keys()):
        cat_A = np.array(categories[cat]['A'])
        cat_U = np.array(categories[cat]['U'])
        if len(cat_A) < 5 or len(cat_U) < 5:
            continue

        # Per-category AUC (using global PCA + direction, evaluate on this category only)
        cat_aucs = []
        for seed in SEEDS:
            trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)
            nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
            pca = PCA(n_components=nc).fit(X[trA])
            R = X - X @ pca.components_.T @ pca.components_
            mu_diff = R[trU].mean(0) - R[trA].mean(0)
            d_mean = mu_diff / (norm(mu_diff) + 1e-15)

            # Test only on this category's samples that are in test set
            te_set = set(te_idx)
            cat_te_A = [i for i in cat_A if i in te_set]
            cat_te_U = [i for i in cat_U if i in te_set]
            if len(cat_te_A) < 2 or len(cat_te_U) < 2:
                continue
            cat_te = np.array(sorted(cat_te_A + cat_te_U))
            cat_te_labels = np.array([0 if i in set(cat_te_A) else 1 for i in cat_te])

            R_cat = R[cat_te]
            norms_cat = norm(R_cat, axis=1) + 1e-15
            scores_cat = (R_cat @ d_mean) / norms_cat
            cat_aucs.append(safe_auc(cat_te_labels, scores_cat))

        if not cat_aucs:
            continue

        cat_auc = np.mean(cat_aucs)

        # Compute predictors on TRAIN-ONLY category data (averaged across seeds)
        tau_seeds, deff_seeds, cohend_seeds, nssnr_seeds = [], [], [], []
        for seed in SEEDS:
            trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)
            tr_set_A = set(trA)
            tr_set_U = set(trU)
            cat_trA = np.array([i for i in cat_A if i in tr_set_A])
            cat_trU = np.array([i for i in cat_U if i in tr_set_U])
            if len(cat_trA) < 5 or len(cat_trU) < 5:
                continue
            tau_seeds.append(attractor_tightness(X[cat_trA]))
            deff_seeds.append(effective_dimensionality(X[cat_trA]))
            cohend_seeds.append(class_separation_cohen_d(X[cat_trA], X[cat_trU]))
            # Null-space SNR using train PCA
            nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
            pca_tr = PCA(n_components=nc).fit(X[trA])
            R_tr = X - X @ pca_tr.components_.T @ pca_tr.components_
            nssnr_seeds.append(null_space_snr(R_tr[cat_trA], R_tr[cat_trU]))

        if not tau_seeds:
            continue
        tau = np.nanmean(tau_seeds)
        d_eff = np.nanmean(deff_seeds)
        cohen_d = np.nanmean(cohend_seeds)
        ns_snr = np.nanmean(nssnr_seeds)

        print(f"  {cat:<25} {len(cat_A):>4} {len(cat_U):>4} {cat_auc:>7.4f} {tau:>7.4f} {d_eff:>6.0f} {cohen_d:>8.3f} {ns_snr:>7.2f}")
        cat_results.append({
            'category': cat, 'auc': cat_auc, 'tau': tau,
            'd_eff': d_eff, 'cohen_d': cohen_d, 'ns_snr': ns_snr,
            'n_A': len(cat_A), 'n_U': len(cat_U)
        })

    # Correlation analysis
    corr_results = {}
    if len(cat_results) >= 5:
        aucs = [r['auc'] for r in cat_results]
        taus = [r['tau'] for r in cat_results]
        d_effs = [r['d_eff'] for r in cat_results]
        cohen_ds = [r['cohen_d'] for r in cat_results]
        ns_snrs = [r['ns_snr'] for r in cat_results]

        print(f"\n  Correlations with per-category AUC (n={len(cat_results)}):")
        for name, vals in [('tau', taus), ('d_eff', d_effs), ('cohen_d', cohen_ds), ('ns_snr', ns_snrs)]:
            valid = [(a, v) for a, v in zip(aucs, vals) if np.isfinite(v)]
            if len(valid) >= 4:
                a_valid, v_valid = zip(*valid)
                r_p, p_p = pearsonr(a_valid, v_valid)
                r_s, p_s = spearmanr(a_valid, v_valid)
                print(f"    {name:<10} Pearson r={r_p:+.3f} (p={p_p:.3f}), "
                      f"Spearman ρ={r_s:+.3f} (p={p_s:.3f})")
                corr_results[name] = {
                    "pearson_r": float(r_p), "pearson_p": float(p_p),
                    "spearman_rho": float(r_s), "spearman_p": float(p_s),
                    "n": len(valid),
                }

    all_part1_results.append({
        "dataset": dataset,
        "model": model,
        "layer": int(layer),
        "global_cosnsrt_auc": float(global_auc),
        "per_category": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in r.items()}
            for r in cat_results
        ],
        "correlations": corr_results,
    })

    del reps


# ═══════════════════════════════════════════════════════════════
# PART 2: Cross-Dataset Comparison
# ═══════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 2: Cross-Dataset Predictors (Mistral)")
print("=" * 80)

print(f"\n  {'Dataset':<25} {'AUC':>7} {'τ':>7} {'d_eff':>6} {'Cohen_d':>8} {'NS_SNR':>7}")

ds_results = []
for dataset, model, layer in CROSS_DATASET_CONFIGS:
    out = load(dataset, model, layer)
    if any(x is None for x in out):
        print(f"  ⚠ {dataset}/{model}: reps missing, skipping")
        continue
    X, A_idx, U_idx, meta = out

    # CosNSRT AUC
    aucs = []
    for seed in SEEDS:
        trA, trU, te_idx, te_labels = split(A_idx, U_idx, seed)
        nc = min(K_PCA, len(trA) - 1, X.shape[1] - 1)
        pca = PCA(n_components=nc).fit(X[trA])
        R = X - X @ pca.components_.T @ pca.components_
        mu_diff = R[trU].mean(0) - R[trA].mean(0)
        d_mean = mu_diff / (norm(mu_diff) + 1e-15)
        R_te = R[te_idx]
        norms_te = norm(R_te, axis=1) + 1e-15
        scores = (R_te @ d_mean) / norms_te
        aucs.append(safe_auc(te_labels, scores))
    auc = np.mean(aucs)

    # Predictors
    tau = attractor_tightness(X[A_idx])
    d_eff = effective_dimensionality(X[A_idx])
    cohen_d = class_separation_cohen_d(X[A_idx], X[U_idx])

    pca_pred = PCA(n_components=min(K_PCA, len(A_idx)//2 - 1, X.shape[1] - 1)).fit(X[A_idx[:len(A_idx)//2]])
    R_pred = X - X @ pca_pred.components_.T @ pca_pred.components_
    ns_snr = null_space_snr(R_pred[A_idx], R_pred[U_idx])

    print(f"  {dataset:<25} {auc:>7.4f} {tau:>7.4f} {d_eff:>6.0f} {cohen_d:>8.3f} {ns_snr:>7.2f}")
    ds_results.append({
        'dataset': dataset, 'model': model, 'layer': int(layer),
        'auc': float(auc), 'tau': float(tau), 'd_eff': float(d_eff),
        'cohen_d': float(cohen_d), 'ns_snr': float(ns_snr),
    })

all_part2_results.extend(ds_results)

# Cross-dataset correlations
part2_corr = {}
if len(ds_results) >= 4:
    aucs = [r['auc'] for r in ds_results]
    print(f"\n  Cross-dataset correlations (n={len(ds_results)}):")
    for name in ['tau', 'd_eff', 'cohen_d', 'ns_snr']:
        vals = [r[name] for r in ds_results]
        valid = [(a, v) for a, v in zip(aucs, vals) if np.isfinite(v)]
        if len(valid) >= 4:
            a_valid, v_valid = zip(*valid)
            r_s, p_s = spearmanr(a_valid, v_valid)
            print(f"    {name:<10} Spearman ρ={r_s:+.3f} (p={p_s:.3f})")
            part2_corr[name] = {
                "spearman_rho": float(r_s), "spearman_p": float(p_s),
                "n": len(valid),
            }

# ── Write structured JSON output (subset-safe: suffix single-model/dataset runs) ──
suffix_parts = []
if _args.model:
    suffix_parts.append(f"model-{_args.model}")
if _args.dataset:
    suffix_parts.append(f"ds-{_args.dataset}")
suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
out_path = f"experiments/form_conditionality_results{suffix}.json"

# Fail-fast on empty per-category results when a single --model was filtered
# in. Previously this wrote a marker JSON whose `part1_per_category` was [] and
# `part2_cross_dataset` was the (always-mistral) cross-dataset block — the
# suite's run_if_missing then treated the marker as "done" and silently
# skipped the cell on a subsequent retry once signals existed.
if _args.model and not all_part1_results:
    raise SystemExit(
        f"--model {_args.model!r}: part1_per_category came out empty. "
        f"Most common cause: extraction signals at "
        f"experiments/signals/{{math800,code800}}_{_args.model}_allL/signals/ "
        f"are missing or incomplete. Refusing to write a stub marker file."
    )

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "part1_per_category": all_part1_results,
        "part2_cross_dataset": all_part2_results,
        "part2_cross_dataset_correlations": part2_corr,
        "best_layers": BEST_LAYERS,
        "cli_layer_override": _args.layer,
        "seeds": SEEDS,
        "k_pca": K_PCA,
        "cli_model_filter": _args.model,
        "cli_dataset_filter": _args.dataset,
    }, f, indent=2)
print(f"\nResults saved to {out_path}")

print("\nDone.")
