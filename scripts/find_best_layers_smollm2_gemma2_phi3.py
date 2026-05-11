"""
Find best layers for small models (SmolLM2, Gemma2, Phi3) using CosNSRT,
then update impossibility_steering.py with the correct values.

Also saves detection results (CosNSRT AUC etc.) to experiments/ for the paper.
"""
import numpy as np, json, os, re, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(BASE))  # repo root (script lives in scripts/)

MODELS = {
    "smollm2": {"n_layers": 24, "D": 2048},
    "gemma2":  {"n_layers": 26, "D": 2304},
    "phi3":    {"n_layers": 32, "D": 3072},
}

DATASETS = ["math800", "code800"]
K_PCA = 100
SEEDS = [42, 123, 456, 789, 1024]


def cosnsrt_auc(X, A_idx, U_idx, seed, k_pca=K_PCA):
    rng = np.random.RandomState(seed)
    pA = rng.permutation(len(A_idx)); pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[:len(A_idx)//2]]; teA = A_idx[pA[len(A_idx)//2:]]
    trU = U_idx[pU[:len(U_idx)//2]]; teU = U_idx[pU[len(U_idx)//2:]]

    nc = min(k_pca, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc).fit(X[trA])
    V = pca.components_
    R = X - X @ V.T @ V

    mu_diff = R[trU].mean(0) - R[trA].mean(0)
    d_hat = mu_diff / (norm(mu_diff) + 1e-15)

    te_idx = np.sort(np.concatenate([teA, teU]))
    te_labels = np.array([0 if i in set(teA) else 1 for i in te_idx])
    scores = R[te_idx] @ d_hat

    try:
        auc = roc_auc_score(te_labels, scores)
        return auc
    except:
        return 0.5


results = {}

for model_name, info in MODELS.items():
    results[model_name] = {}
    for dataset in DATASETS:
        sig_dir = f"experiments/signals/{dataset}_{model_name}_allL/signals"
        reps_path = f"{sig_dir}/reps_all_layers.npy"
        meta_path = f"{sig_dir}/meta.jsonl"

        if not os.path.exists(reps_path):
            print(f"  SKIP {dataset}/{model_name}: no reps file")
            continue

        print(f"\n{'='*70}")
        print(f"{dataset}/{model_name} ({info['n_layers']} layers)")
        print(f"{'='*70}")

        try:
            reps = np.load(reps_path, mmap_mode='r')
        except Exception as e:
            print(f"  SKIP {dataset}/{model_name}: failed to load reps: {e}")
            continue
        meta = [json.loads(l) for l in open(meta_path)]
        labels = np.array([m['answerable'] for m in meta])
        A_idx = np.where(labels == 'A')[0]
        U_idx = np.where(labels == 'U')[0]

        # Use actual layer count from data, not hardcoded
        actual_layers = reps.shape[1]
        if actual_layers != info['n_layers']:
            print(f"  WARNING: expected {info['n_layers']} layers, got {actual_layers}. Using actual.")
            info['n_layers'] = actual_layers

        layer_aucs = []
        for layer in range(actual_layers):
            X = np.array(reps[:, layer, :], dtype=np.float32)
            aucs = [cosnsrt_auc(X, A_idx, U_idx, s) for s in SEEDS]
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            layer_aucs.append((layer, mean_auc, std_auc))
            if layer % 4 == 0 or layer == info['n_layers'] - 1:
                print(f"  Layer {layer:>2}: CosNSRT AUC = {mean_auc:.4f} ± {std_auc:.4f}")

        best_layer, best_auc, best_std = max(layer_aucs, key=lambda x: x[1])
        print(f"\n  >>> BEST: layer {best_layer}, AUC = {best_auc:.4f} ± {best_std:.4f}")

        results[model_name][dataset] = {
            "best_layer": best_layer,
            "n_layers": info['n_layers'],
            "D": info['D'],
            "CosNSRT": {"mean": round(best_auc, 4), "std": round(best_std, 4)},
            "all_layers": [(l, round(a, 4), round(s, 4)) for l, a, s in layer_aucs],
        }

# ── Save results ──
for model_name in results:
    out_path = f"experiments/eval_{model_name}.json"
    # Merge with existing if present
    existing = {}
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
    for dataset in results[model_name]:
        key = f"{dataset}/{model_name}"
        existing[key] = {k: v for k, v in results[model_name][dataset].items() if k != "all_layers"}
    with open(out_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved: {out_path}")

# ── Update impossibility_steering.py with correct best layers ──
steering_path = "scripts/impossibility_steering.py"
code = open(steering_path).read()

for model_name in results:
    for dataset in results[model_name]:
        best_layer = results[model_name][dataset]["best_layer"]
        # Pattern: ("math800", "smollm2"): 12,
        old_pattern = rf'(\("{dataset}", "{model_name}"\):\s*)\d+'
        new_val = rf'\g<1>{best_layer}'
        code = re.sub(old_pattern, new_val, code)
        print(f"  Updated BEST_LAYERS[({dataset}, {model_name})] = {best_layer}")

with open(steering_path, 'w') as f:
    f.write(code)
print(f"\nUpdated {steering_path} with correct best layers.")

# ── Summary ──
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for model_name in results:
    for dataset in results[model_name]:
        r = results[model_name][dataset]
        print(f"  {dataset}/{model_name}: best_layer={r['best_layer']}, CosNSRT={r['CosNSRT']['mean']:.4f}")
