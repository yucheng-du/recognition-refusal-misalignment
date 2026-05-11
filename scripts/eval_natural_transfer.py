"""
eval_natural_transfer.py — Zero-shot CosNSRT transfer evaluation on non-core datasets.

PRIMARY REPORT: Zero-shot projection of math800-trained impossibility direction
onto target dataset representations, using SAME MODEL for source and target.

For each (model, dataset):
  1. Load d_imp from math800/{model}/{BEST_LAYER[model]} (same model, peak layer)
  2. Project target dataset reps onto d_imp → zero-shot AUC
  3. Also compute CosNSRT (length-normalized residual projection)
  4. Optionally report fresh within-dataset AUC:
       - By default: same peak layer (reference number, not a true upper bound)
       - With --sweep-layers: best-layer sweep (strict upper bound; slower)

Prefers reps_last_token_all_layers.npy (matches main pipeline).
Falls back to reps_all_layers.npy (mean-pooled) if last-token not available.

Usage:
  python scripts/eval_natural_transfer.py                          # all (model, dataset) with extraction
  python scripts/eval_natural_transfer.py --dataset abstentionbench_gsm8k
  python scripts/eval_natural_transfer.py --model mistral
  python scripts/eval_natural_transfer.py --no-fresh               # skip fresh AUC (faster)
"""

import argparse
import gc
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

# ── Same-model source layer mapping (from core steering + orthogonality results) ──
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

# Non-core datasets to evaluate
NON_CORE_DATASETS = ["abstentionbench_gsm8k", "falseqa"]
# NOTE: mathtrap intentionally excluded (A-side duplicates, data quality issues).
# NOTE: difficulty_control_gsm8k uses dedicated eval_difficulty_control.py
#       because its positive class is difficulty_label, not answerable.

ALL_MODELS = list(BEST_LAYERS.keys())


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2 or np.std(scores) < 1e-15:
        return 0.5
    a = roc_auc_score(y_true, scores)
    return max(a, 1 - a)


def load_reps_at_layer(sig_dir: str, layer: int, prefer_last_token: bool = True):
    """Load reps at a specific layer. Returns (reps, pooling_used)."""
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_path = os.path.join(sig_dir, "reps_all_layers.npy")

    if prefer_last_token and os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode="r")
        pooling = "last_token"
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode="r")
        pooling = "mean_pooled"
    else:
        return None, None

    if layer >= reps.shape[1]:
        print(f"  ⚠ Layer {layer} >= n_layers {reps.shape[1]}, clipping to last layer")
        layer = reps.shape[1] - 1

    reps_layer = np.array(reps[:, layer, :], dtype=np.float32)
    del reps
    gc.collect()
    return reps_layer, pooling


def compute_source_direction(model: str):
    """Train d_imp on math800/{model}/{BEST_LAYER[model]}."""
    src_sig = os.path.join(BASE, f"experiments/signals/math800_{model}_allL/signals")
    if not os.path.exists(src_sig):
        return None

    layer = BEST_LAYERS[model]
    reps_layer, pooling = load_reps_at_layer(src_sig, layer, prefer_last_token=True)
    if reps_layer is None:
        return None

    with open(os.path.join(src_sig, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]
    labels = np.array([1 if m["answerable"] == "U" else 0 for m in meta])

    A_mask = labels == 0
    n_comp = min(100, int(A_mask.sum()) - 1, reps_layer.shape[1] - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(reps_layer[A_mask])

    resid = reps_layer - pca.inverse_transform(pca.transform(reps_layer))
    d_raw = resid[~A_mask].mean(0) - resid[A_mask].mean(0)
    d_imp = d_raw / np.linalg.norm(d_raw)

    # Verify on source (A vs U)
    scores_nsrt = resid @ d_imp
    src_auc_nsrt = safe_auc(labels, scores_nsrt)

    resid_norms = np.linalg.norm(resid, axis=1, keepdims=True)
    resid_norms = np.where(resid_norms < 1e-10, 1.0, resid_norms)
    scores_cos = (resid / resid_norms) @ d_imp
    src_auc_cos = safe_auc(labels, scores_cos)

    return {
        "model": model,
        "layer": layer,
        "pooling": pooling,
        "d_imp": d_imp,
        "pca": pca,
        "src_auc_nsrt": src_auc_nsrt,
        "src_auc_cos": src_auc_cos,
        "dim": reps_layer.shape[1],
    }


def zero_shot_transfer(source: dict, target_dataset: str, model: str):
    """Apply source d_imp to target dataset at the SAME layer."""
    sig_dir = os.path.join(
        BASE, f"experiments/signals/{target_dataset}_{model}_allL/signals"
    )
    if not os.path.exists(sig_dir):
        return None

    layer = source["layer"]
    reps_layer, pooling = load_reps_at_layer(sig_dir, layer, prefer_last_token=True)
    if reps_layer is None:
        return None

    if reps_layer.shape[1] != source["dim"]:
        return {"error": f"dim mismatch: source {source['dim']} vs target {reps_layer.shape[1]}"}

    # Labels
    with open(os.path.join(sig_dir, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]
    labels = np.array([1 if m["answerable"] == "U" else 0 for m in meta])

    # Filter rows with NaN/Inf (rare but happens with bf16 on MPS for some prompts)
    bad_mask = np.isnan(reps_layer).any(axis=1) | np.isinf(reps_layer).any(axis=1)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        good = ~bad_mask
        reps_layer = reps_layer[good]
        labels = labels[good]
        print(f"  ⚠ {target_dataset}/{model}: dropped {n_bad} NaN/Inf rows")

    # Project onto source PCA residuals, then onto d_imp
    resid = reps_layer - source["pca"].inverse_transform(source["pca"].transform(reps_layer))
    scores_nsrt = resid @ source["d_imp"]

    # CosNSRT (normalize residual before projection)
    resid_norms = np.linalg.norm(resid, axis=1, keepdims=True)
    resid_norms = np.where(resid_norms < 1e-10, 1.0, resid_norms)
    scores_cos = (resid / resid_norms) @ source["d_imp"]

    auc_nsrt = safe_auc(labels, scores_nsrt)
    auc_cos = safe_auc(labels, scores_cos)

    n_a = int((labels == 0).sum())
    n_u = int((labels == 1).sum())

    return {
        "n_a": n_a,
        "n_u": n_u,
        "layer": layer,
        "pooling": pooling,
        "source_pooling": source["pooling"],
        "zero_shot_nsrt_auc": round(auc_nsrt, 4),
        "zero_shot_cosnsrt_auc": round(auc_cos, 4),
    }


def fresh_detection(target_dataset: str, model: str, sweep_layers: bool = False):
    """Train fresh d_imp on target dataset.

    If sweep_layers=True, sweeps all layers and returns the true best-layer AUC
    (strict upper bound on zero-shot transfer). If False, evaluates only at the
    same peak layer as the source — a reference number, not an upper bound.
    """
    sig_dir = os.path.join(
        BASE, f"experiments/signals/{target_dataset}_{model}_allL/signals"
    )
    if not os.path.exists(sig_dir):
        return None

    # Prefer last-token
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_path = os.path.join(sig_dir, "reps_all_layers.npy")
    reps_path = lt_path if os.path.exists(lt_path) else mp_path
    if not os.path.exists(reps_path):
        return None
    pooling = "last_token" if reps_path == lt_path else "mean_pooled"

    reps_all = np.load(reps_path, mmap_mode="r")
    n_layers = reps_all.shape[1]
    D = reps_all.shape[2]

    with open(os.path.join(sig_dir, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]
    labels = np.array([1 if m["answerable"] == "U" else 0 for m in meta])
    A_mask = labels == 0

    layers_to_try = range(n_layers) if sweep_layers else [BEST_LAYERS[model]]

    best_nsrt = 0.0
    best_cos = 0.0
    best_layer = layers_to_try.start if hasattr(layers_to_try, "start") else layers_to_try[0]

    for layer in layers_to_try:
        if layer >= n_layers:
            continue
        reps_l = np.array(reps_all[:, layer, :], dtype=np.float32)
        # Filter NaN/Inf (rare bf16 numerical issues on MPS)
        bad = np.isnan(reps_l).any(axis=1) | np.isinf(reps_l).any(axis=1)
        if bad.any():
            good = ~bad
            reps_l = reps_l[good]
            labels_l = labels[good]
            A_mask_l = A_mask[good]
        else:
            labels_l = labels
            A_mask_l = A_mask
        n_comp = min(100, int(A_mask_l.sum()) - 1, D - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(reps_l[A_mask_l])
        resid = reps_l - pca.inverse_transform(pca.transform(reps_l))
        d_raw = resid[~A_mask_l].mean(0) - resid[A_mask_l].mean(0)
        d_l = d_raw / np.linalg.norm(d_raw)

        auc_nsrt = safe_auc(labels_l, resid @ d_l)
        rn = np.linalg.norm(resid, axis=1, keepdims=True)
        rn = np.where(rn < 1e-10, 1.0, rn)
        auc_cos = safe_auc(labels_l, (resid / rn) @ d_l)

        if auc_cos > best_cos:
            best_cos = auc_cos
            best_nsrt = auc_nsrt
            best_layer = layer

        del reps_l, resid
        gc.collect()

    del reps_all
    gc.collect()

    return {
        "pooling": pooling,
        "best_layer": best_layer,
        "fresh_nsrt_auc": round(best_nsrt, 4),
        "fresh_cosnsrt_auc": round(best_cos, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=NON_CORE_DATASETS, default=None,
                        help="Single dataset (default: all non-core)")
    parser.add_argument("--model", choices=ALL_MODELS, default=None,
                        help="Single model (default: all)")
    parser.add_argument("--no-fresh", action="store_true",
                        help="Skip fresh within-dataset AUC (faster)")
    parser.add_argument("--sweep-layers", action="store_true",
                        help="For fresh AUC, sweep all layers (slower, finds true best)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else NON_CORE_DATASETS
    models = [args.model] if args.model else ALL_MODELS

    print("=" * 80)
    print("ZERO-SHOT CosNSRT TRANSFER: math800 direction → non-core datasets")
    print("Same-model source: math800/{model}/{BEST_LAYER[model]}")
    print("=" * 80)

    results = []

    for model in models:
        # Train source d_imp
        src_sig = os.path.join(BASE, f"experiments/signals/math800_{model}_allL/signals")
        if not os.path.exists(src_sig):
            print(f"\n[{model}] No math800 extraction, skipping.")
            continue

        source = compute_source_direction(model)
        if source is None:
            print(f"\n[{model}] Failed to compute source direction.")
            continue

        print(f"\n[{model}] Source: math800/L{source['layer']}, "
              f"pooling={source['pooling']}, "
              f"src_NSRT={source['src_auc_nsrt']:.4f}, "
              f"src_CosNSRT={source['src_auc_cos']:.4f}")

        for ds in datasets:
            target_sig = os.path.join(
                BASE, f"experiments/signals/{ds}_{model}_allL/signals"
            )
            if not os.path.exists(target_sig):
                print(f"  [{ds}] no extraction, skipping")
                continue

            zs = zero_shot_transfer(source, ds, model)
            if zs is None or zs.get("error"):
                err = zs.get("error") if zs else "load failed"
                print(f"  [{ds}] error: {err}")
                continue

            entry = {
                "model": model,
                "dataset": ds,
                "source_layer": source["layer"],
                "source_pooling": source["pooling"],
                "n_a": zs["n_a"],
                "n_u": zs["n_u"],
                "target_pooling": zs["pooling"],
                "zero_shot_nsrt_auc": zs["zero_shot_nsrt_auc"],
                "zero_shot_cosnsrt_auc": zs["zero_shot_cosnsrt_auc"],
            }

            pool_match = zs["pooling"] == zs["source_pooling"]
            mismatch_flag = "" if pool_match else f"  ⚠ pooling mismatch: src={zs['source_pooling']} tgt={zs['pooling']}"
            print(f"  [{ds}] N=({zs['n_a']}A+{zs['n_u']}U), "
                  f"zero-shot NSRT={zs['zero_shot_nsrt_auc']:.4f}, "
                  f"CosNSRT={zs['zero_shot_cosnsrt_auc']:.4f}{mismatch_flag}")

            if not args.no_fresh:
                fresh = fresh_detection(ds, model, sweep_layers=args.sweep_layers)
                if fresh:
                    entry["fresh_best_layer"] = fresh["best_layer"]
                    entry["fresh_nsrt_auc"] = fresh["fresh_nsrt_auc"]
                    entry["fresh_cosnsrt_auc"] = fresh["fresh_cosnsrt_auc"]
                    entry["fresh_swept_all_layers"] = bool(args.sweep_layers)
                    entry["gap_cosnsrt"] = round(
                        fresh["fresh_cosnsrt_auc"] - zs["zero_shot_cosnsrt_auc"], 4)
                    label = "fresh strict UB (swept)" if args.sweep_layers else "fresh @ same layer"
                    print(f"         ({label}: L{fresh['best_layer']}, "
                          f"CosNSRT={fresh['fresh_cosnsrt_auc']:.4f}, "
                          f"gap={entry['gap_cosnsrt']:+.4f})")

            results.append(entry)

        # Free memory
        del source
        gc.collect()

    # Summary table
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY — Zero-shot CosNSRT transfer from math800")
        print("=" * 80)

        show_fresh = any("fresh_cosnsrt_auc" in r for r in results)
        any_swept = any(r.get("fresh_swept_all_layers") for r in results)
        fresh_col = "Fresh_UB" if any_swept else "Fresh@L"
        header = f"{'Model':<10} {'Dataset':<25} {'N_A+N_U':<10} {'ZS_CosNSRT':<12}"
        if show_fresh:
            header += f" {fresh_col:<10} {'Gap':<8}"
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: (x["dataset"], x["model"])):
            line = (f"{r['model']:<10} {r['dataset']:<25} "
                    f"{r['n_a']}+{r['n_u']:<7} "
                    f"{r['zero_shot_cosnsrt_auc']:<12.4f}")
            if show_fresh and "fresh_cosnsrt_auc" in r:
                line += f" {r['fresh_cosnsrt_auc']:<10.4f} {r['gap_cosnsrt']:+.4f}"
            print(line)

        # Group averages
        print("\nPer-dataset mean CosNSRT:")
        for ds in datasets:
            ds_results = [r for r in results if r["dataset"] == ds]
            if ds_results:
                mean_zs = np.mean([r["zero_shot_cosnsrt_auc"] for r in ds_results])
                n_models = len(ds_results)
                extra = ""
                if show_fresh:
                    fresh_vals = [r.get("fresh_cosnsrt_auc") for r in ds_results
                                  if r.get("fresh_cosnsrt_auc") is not None]
                    if fresh_vals:
                        fresh_label = ("fresh UB mean"
                                       if any(r.get("fresh_swept_all_layers") for r in ds_results)
                                       else "fresh@L mean")
                        extra = f" | {fresh_label} = {np.mean(fresh_vals):.4f}"
                print(f"  {ds}: {mean_zs:.4f} (n={n_models} models){extra}")

    # Protect full-run outputs from being overwritten by smoke tests.
    # If a subset was requested (--model or --dataset), write to a suffixed file.
    suffix_parts = []
    if args.model:
        suffix_parts.append(f"model-{args.model}")
    if args.dataset:
        suffix_parts.append(f"ds-{args.dataset}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_path = os.path.join(BASE, f"experiments/natural_transfer_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
