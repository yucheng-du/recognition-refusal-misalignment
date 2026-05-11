"""
eval_abstentionbench_lengthcontrol.py — Length-controlled subset analysis for
abstentionbench_gsm8k.

Problem: abstentionbench has extreme length confound — U prompts are the
stripped question of A prompts (A ~3x longer than U). A naive length-based
classifier achieves near-perfect separation (token AUC ≈ 0.99).

This script tests whether the zero-shot impossibility direction (from math800)
still separates A/U when restricted to a length-matched subset.

Strategy:
  1. Tokenize all prompts with Mistral tokenizer.
  2. Find length-matched subsets by percentile bucketing:
     - Find the overlapping range between A and U token-length distributions.
     - Within that range, draw balanced A and U samples.
  3. Apply math800/{model}/{BEST_LAYER} d_imp to the subset.
  4. Compare zero-shot AUC on (a) full set vs (b) length-matched subset.

Expected outcome: the length-matched subset should still show strong AUC if
the impossibility direction truly captures answerability (not length).

Usage:
  python scripts/eval_abstentionbench_lengthcontrol.py
  python scripts/eval_abstentionbench_lengthcontrol.py --model mistral
"""

import argparse
import gc
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, NOT scripts/

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

DATASET = "abstentionbench_gsm8k"


def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2 or np.std(scores) < 1e-15:
        return 0.5
    a = roc_auc_score(y_true, scores)
    return max(a, 1 - a)


def tokenize_all(prompts, tokenizer):
    return np.array([len(tokenizer.encode(p)) for p in prompts])


def build_length_matched_subset(token_lengths, labels, seed=42):
    """
    Balance A/U within the overlapping length range.

    Returns indices of a length-matched subset where:
      - Min length = max(A_min, U_min)
      - Max length = min(A_max, U_max)
      - Within that range, match counts per length bucket.

    Returns (subset_indices, subset_stats).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    token_lengths = np.asarray(token_lengths)

    A_idx = np.where(labels == 0)[0]
    U_idx = np.where(labels == 1)[0]

    A_lens = token_lengths[A_idx]
    U_lens = token_lengths[U_idx]

    # Overlapping range
    lo = max(A_lens.min(), U_lens.min())
    hi = min(A_lens.max(), U_lens.max())

    # Bucketing: per-integer-length buckets (token length is integer)
    buckets = np.arange(lo, hi + 1)
    subset_A = []
    subset_U = []
    for b in buckets:
        a_in_bucket = A_idx[(A_lens >= b) & (A_lens < b + 1)]
        u_in_bucket = U_idx[(U_lens >= b) & (U_lens < b + 1)]
        k = min(len(a_in_bucket), len(u_in_bucket))
        if k == 0:
            continue
        # Random sample k from each
        subset_A.extend(rng.choice(a_in_bucket, size=k, replace=False))
        subset_U.extend(rng.choice(u_in_bucket, size=k, replace=False))

    subset = np.array(sorted(subset_A + subset_U))
    stats = {
        "overlap_range": [int(lo), int(hi)],
        "n_a_matched": len(subset_A),
        "n_u_matched": len(subset_U),
        "n_total_matched": len(subset),
        "full_n_a": int(len(A_idx)),
        "full_n_u": int(len(U_idx)),
        "full_a_len_mean": float(A_lens.mean()),
        "full_u_len_mean": float(U_lens.mean()),
        "full_a_len_range": [int(A_lens.min()), int(A_lens.max())],
        "full_u_len_range": [int(U_lens.min()), int(U_lens.max())],
    }
    if len(subset) > 0:
        matched_lens = token_lengths[subset]
        stats["matched_len_mean"] = float(matched_lens.mean())
        stats["matched_len_range"] = [int(matched_lens.min()), int(matched_lens.max())]
    return subset, stats


def resample_matched_auc(scores_cos, scores_nsrt, labels, token_lengths,
                         K=100, base_seed=42, min_subset=20):
    """Subset-resampling CI for matched-subset CosNSRT / NSRT AUC.

    NOT a strict full bootstrap over the data: varies only the RNG seed inside
    `build_length_matched_subset`, which reshuffles which A/U samples fall into
    each length bucket. Projections (scores_cos, scores_nsrt) are pre-computed
    once; each iteration is just re-indexing + AUC.

    This captures sensitivity to the matched-subset construction, not
    full sampling variability of the underlying dataset. Good enough to argue
    "the length-control conclusion is stable across matched-subset draws".

    Returns dict with mean + 95% percentile CI for CosNSRT and NSRT.
    Returns None if all K iterations produced under-sized subsets.
    """
    cos_aucs, nsrt_aucs, subset_sizes = [], [], []
    for k in range(K):
        sidx, _ = build_length_matched_subset(token_lengths, labels,
                                              seed=base_seed + k)
        if len(sidx) < min_subset:
            continue
        y = labels[sidx]
        cos_aucs.append(safe_auc(y, scores_cos[sidx]))
        nsrt_aucs.append(safe_auc(y, scores_nsrt[sidx]))
        subset_sizes.append(len(sidx))

    if not cos_aucs:
        return None

    def _stats(arr):
        a = np.asarray(arr)
        return {
            "mean":     float(a.mean()),
            "ci_lower": float(np.percentile(a, 2.5)),
            "ci_upper": float(np.percentile(a, 97.5)),
        }

    return {
        "n_bootstrap_iterations": len(cos_aucs),
        "subset_size_range": [int(min(subset_sizes)), int(max(subset_sizes))],
        "cosnsrt": _stats(cos_aucs),
        "nsrt":    _stats(nsrt_aucs),
    }


def load_layer_reps(sig_dir, layer):
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    mp_path = os.path.join(sig_dir, "reps_all_layers.npy")
    if os.path.exists(lt_path):
        reps = np.load(lt_path, mmap_mode="r")
        pooling = "last_token"
    elif os.path.exists(mp_path):
        reps = np.load(mp_path, mmap_mode="r")
        pooling = "mean_pooled"
    else:
        return None, None, None
    if layer >= reps.shape[1]:
        layer = reps.shape[1] - 1
    reps_layer = np.array(reps[:, layer, :], dtype=np.float32)
    del reps
    gc.collect()
    return reps_layer, pooling, layer


def compute_source_direction(model, layer):
    """Train d_imp on math800/{model}/{layer}."""
    src_sig = os.path.join(BASE, f"experiments/signals/math800_{model}_allL/signals")
    if not os.path.exists(src_sig):
        return None
    reps_layer, pooling, used_layer = load_layer_reps(src_sig, layer)
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

    return {"d_imp": d_imp, "pca": pca, "pooling": pooling, "layer": used_layer,
            "dim": reps_layer.shape[1]}


def evaluate(model: str, tokenizer, resample_k: int = 0):
    layer = BEST_LAYERS[model]

    # Train source direction
    source = compute_source_direction(model, layer)
    if source is None:
        print(f"[{model}] No math800 extraction, skipping.")
        return None

    # Load target (abstentionbench) reps and meta
    tgt_sig = os.path.join(BASE, f"experiments/signals/{DATASET}_{model}_allL/signals")
    if not os.path.exists(tgt_sig):
        print(f"[{model}] No {DATASET} extraction, skipping.")
        return None

    reps_layer, tgt_pooling, _ = load_layer_reps(tgt_sig, layer)
    if reps_layer is None:
        print(f"[{model}] Could not load target reps.")
        return None

    if reps_layer.shape[1] != source["dim"]:
        print(f"[{model}] Dim mismatch: source {source['dim']} vs target {reps_layer.shape[1]}")
        return None

    with open(os.path.join(tgt_sig, "meta.jsonl")) as f:
        meta = [json.loads(l) for l in f]
    labels = np.array([1 if m["answerable"] == "U" else 0 for m in meta])

    # Tokenize all prompts to get token lengths (using Mistral tokenizer for consistency)
    prompts = [m.get("prompt", "") for m in meta]
    token_lengths = tokenize_all(prompts, tokenizer)

    # Filter NaN/Inf rows (rare bf16 numerical issues on MPS)
    bad_mask = np.isnan(reps_layer).any(axis=1) | np.isinf(reps_layer).any(axis=1)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        good = ~bad_mask
        reps_layer = reps_layer[good]
        labels = labels[good]
        token_lengths = token_lengths[good]
        print(f"  ⚠ {model}: dropped {n_bad} NaN/Inf rows from abstentionbench reps")

    # Full-set projection
    resid_full = reps_layer - source["pca"].inverse_transform(
        source["pca"].transform(reps_layer))
    scores_nsrt = resid_full @ source["d_imp"]
    resid_norms = np.linalg.norm(resid_full, axis=1, keepdims=True)
    resid_norms = np.where(resid_norms < 1e-10, 1.0, resid_norms)
    scores_cos = (resid_full / resid_norms) @ source["d_imp"]

    full_nsrt = safe_auc(labels, scores_nsrt)
    full_cos = safe_auc(labels, scores_cos)
    full_length_auc = safe_auc(labels, token_lengths.astype(float))
    # Raw direction (without forcing >0.5) for length — tells us which side is shorter
    full_length_auc_raw = roc_auc_score(labels, token_lengths.astype(float))

    # Length-matched subset
    subset_idx, stats = build_length_matched_subset(token_lengths, labels)

    if len(subset_idx) < 20:
        print(f"[{model}] Length-matched subset too small ({len(subset_idx)}), skipping.")
        subset_result = None
    else:
        sub_labels = labels[subset_idx]
        sub_scores_nsrt = scores_nsrt[subset_idx]
        sub_scores_cos = scores_cos[subset_idx]
        sub_token_lengths = token_lengths[subset_idx].astype(float)

        sub_nsrt = safe_auc(sub_labels, sub_scores_nsrt)
        sub_cos = safe_auc(sub_labels, sub_scores_cos)
        sub_length_auc = safe_auc(sub_labels, sub_token_lengths)
        sub_length_auc_raw = roc_auc_score(sub_labels, sub_token_lengths)

        subset_result = {
            "n_a": stats["n_a_matched"],
            "n_u": stats["n_u_matched"],
            "overlap_range": stats["overlap_range"],
            "matched_len_mean": stats.get("matched_len_mean"),
            "matched_len_range": stats.get("matched_len_range"),
            "nsrt_auc": round(sub_nsrt, 4),
            "cosnsrt_auc": round(sub_cos, 4),
            "length_auc_maxed": round(sub_length_auc, 4),
            "length_auc_raw": round(sub_length_auc_raw, 4),
        }

        # Subset-resampling CI (optional) — NOT a full bootstrap over the data.
        # Varies the matched-subset construction seed across K draws.
        if resample_k > 0:
            rs = resample_matched_auc(
                scores_cos, scores_nsrt, labels, token_lengths,
                K=resample_k, base_seed=42, min_subset=20,
            )
            if rs is not None:
                subset_result["resample_ci"] = {
                    "method": "matched-subset resampling (seed varied); "
                              "NOT full bootstrap over data",
                    "n_iterations": rs["n_bootstrap_iterations"],
                    "subset_size_range": rs["subset_size_range"],
                    "cosnsrt_mean":     round(rs["cosnsrt"]["mean"], 4),
                    "cosnsrt_ci_lower": round(rs["cosnsrt"]["ci_lower"], 4),
                    "cosnsrt_ci_upper": round(rs["cosnsrt"]["ci_upper"], 4),
                    "nsrt_mean":        round(rs["nsrt"]["mean"], 4),
                    "nsrt_ci_lower":    round(rs["nsrt"]["ci_lower"], 4),
                    "nsrt_ci_upper":    round(rs["nsrt"]["ci_upper"], 4),
                }

    del reps_layer, resid_full
    gc.collect()

    return {
        "model": model,
        "layer": layer,
        "source_pooling": source["pooling"],
        "target_pooling": tgt_pooling,
        "full": {
            "n_a": stats["full_n_a"],
            "n_u": stats["full_n_u"],
            "a_len_mean": round(stats["full_a_len_mean"], 2),
            "u_len_mean": round(stats["full_u_len_mean"], 2),
            "a_len_range": stats["full_a_len_range"],
            "u_len_range": stats["full_u_len_range"],
            "nsrt_auc": round(full_nsrt, 4),
            "cosnsrt_auc": round(full_cos, 4),
            "length_auc_maxed": round(full_length_auc, 4),
            "length_auc_raw": round(full_length_auc_raw, 4),
        },
        "length_matched": subset_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BEST_LAYERS.keys()), default=None)
    parser.add_argument("--resample", "--bootstrap", type=int, default=0,
                        dest="resample_k",
                        help="K matched-subset resamples for CI "
                             "(0 = disabled; recommend 100). "
                             "Note: this is subset-construction resampling, "
                             "not a full bootstrap over the data.")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print("Loading Mistral tokenizer for length measurement...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True)

    models = [args.model] if args.model else list(BEST_LAYERS.keys())

    print("=" * 82)
    print("LENGTH-CONTROLLED EVAL: abstentionbench_gsm8k (zero-shot from math800)")
    if args.resample_k > 0:
        print(f"Subset-resampling CI: K={args.resample_k} matched-subset draws "
              f"(not a full data bootstrap)")
    print("=" * 82)
    print("full = all 2426 records (A 3x longer than U by construction)")
    print("length-matched = per-bucket balanced A/U subset within overlapping length range")
    print()

    results = []
    for model in models:
        r = evaluate(model, tokenizer, resample_k=args.resample_k)
        if r is None:
            continue

        pool_match = r["source_pooling"] == r["target_pooling"]
        pool_note = "" if pool_match else (
            f"  ⚠ POOLING MISMATCH: source={r['source_pooling']}, "
            f"target={r['target_pooling']} (re-extract target for apples-to-apples)")
        print(f"\n[{model} / L{r['layer']}]{pool_note}")
        f = r["full"]
        print(f"  FULL SET         N=({f['n_a']}A+{f['n_u']}U)")
        print(f"    A tok_len mean={f['a_len_mean']} range={f['a_len_range']}")
        print(f"    U tok_len mean={f['u_len_mean']} range={f['u_len_range']}")
        print(f"    zero-shot CosNSRT: {f['cosnsrt_auc']:.4f}")
        print(f"    length AUC (raw): {f['length_auc_raw']:.4f}  (maxed: {f['length_auc_maxed']:.4f})")

        if r["length_matched"]:
            s = r["length_matched"]
            print(f"  LENGTH-MATCHED   N=({s['n_a']}A+{s['n_u']}U) in len range {s['overlap_range']}")
            print(f"    matched mean={s['matched_len_mean']:.1f}")
            print(f"    zero-shot CosNSRT: {s['cosnsrt_auc']:.4f}")
            if "resample_ci" in s:
                b = s["resample_ci"]
                print(f"    Resample-CI CosNSRT μ={b['cosnsrt_mean']:.4f} "
                      f"95% CI=[{b['cosnsrt_ci_lower']:.4f}, {b['cosnsrt_ci_upper']:.4f}] "
                      f"(K={b['n_iterations']}, N∈{b['subset_size_range']})")
            print(f"    length AUC (raw): {s['length_auc_raw']:.4f}  (maxed: {s['length_auc_maxed']:.4f})")
            drop = f["cosnsrt_auc"] - s["cosnsrt_auc"]
            print(f"    ΔCosNSRT (full → matched) = {drop:+.4f}")

        results.append(r)

    # Summary
    if results:
        print("\n" + "=" * 82)
        print("SUMMARY")
        print("=" * 82)

        has_resample = any(
            r["length_matched"] and "resample_ci" in r["length_matched"]
            for r in results if r.get("length_matched")
        )
        if has_resample:
            print(f"{'Model':<10} {'Full_CosNSRT':<14} {'Matched_CosNSRT':<17} "
                  f"{'Resample_95%CI':<22} {'Drop':<9}")
        else:
            print(f"{'Model':<10} {'Full_CosNSRT':<14} {'Matched_CosNSRT':<17} "
                  f"{'Drop':<9} {'Full_len_raw':<14} {'Matched_len_raw':<16}")
        print("-" * 82)
        for r in results:
            if r["length_matched"] is None:
                continue
            f = r["full"]
            s = r["length_matched"]
            drop = f["cosnsrt_auc"] - s["cosnsrt_auc"]
            if has_resample and "resample_ci" in s:
                b = s["resample_ci"]
                ci_s = f"[{b['cosnsrt_ci_lower']:.3f}, {b['cosnsrt_ci_upper']:.3f}]"
                print(f"{r['model']:<10} {f['cosnsrt_auc']:<14.4f} {s['cosnsrt_auc']:<17.4f} "
                      f"{ci_s:<22} {drop:+.4f}")
            else:
                print(f"{r['model']:<10} {f['cosnsrt_auc']:<14.4f} {s['cosnsrt_auc']:<17.4f} "
                      f"{drop:+.4f}  {f['length_auc_raw']:<14.4f} {s['length_auc_raw']:<16.4f}")

        print()
        if has_resample:
            print("Interpretation (single seeded subset + K-draw resampling CI):")
            print("  - Matched length_auc_raw ≈ 0.5 confirms length is neutralized in the subset.")
            print("  - Resample CI reflects sensitivity to matched-subset draw, NOT full data bootstrap.")
            print("  - If CosNSRT + CI stay clearly > 0.5 → strong evidence against a length-based shortcut.")
            print("  - Large drop (full → matched) would indicate partial reliance on length.")
        else:
            print("Interpretation (single seeded matched subset; pass --resample K for CIs):")
            print("  - Matched length_auc_raw ≈ 0.5 confirms length is neutralized in the subset.")
            print("  - If matched CosNSRT stays high → strong evidence against a pure length-based")
            print("    shortcut (one seed only; use --resample 100 for a CI).")
            print("  - Large drop would indicate partial reliance on length.")

    # Suffix single-model runs to protect full-run output from smoke tests.
    suffix = f"_model-{args.model}" if args.model else ""
    out_path = os.path.join(BASE, f"experiments/abstentionbench_lengthcontrol_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
