"""
semantic_entropy.py — Farquhar-style semantic entropy baseline for EMNLP_2026.

This script adapts the core logic from the official `semantic_uncertainty`
repository to the local JSONL format produced by `run_semantic_entropy.py`.

Core steps:
  1. Load N sampled responses per prompt.
  2. Group semantically equivalent responses via bidirectional entailment.
  3. Compute:
       - semantic_entropy        (log-likelihood-aware, closest to the paper)
       - cluster_assignment_entropy (discrete cluster entropy)
  4. Evaluate ROC-AUC against answerability labels.

Example:
  python src/baselines/semantic_entropy.py \
      --samples-file experiments/semantic_entropy/llama_math50/samples/llama_math50.jsonl \
      --run-dirs experiments/runs/run_003 experiments/runs/run_003b experiments/runs/run_003c \
      --label llama_math50 \
      --forms MATH
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEBERTA_MODEL = "microsoft/deberta-v2-xlarge-mnli"


class EntailmentDeberta:
    """Official-style entailment checker based on DeBERTa MNLI."""

    def __init__(self, model_name=DEBERTA_MODEL, cache_path=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(DEVICE)
        self.model.eval()
        self.cache = {}
        self.cache_path = cache_path
        self.new_cache_entries = 0
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.cache[(row["text1"], row["text2"])] = row["label"]

    def flush_cache(self):
        if not self.cache_path or self.new_cache_entries == 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        existing = set()
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    existing.add((row["text1"], row["text2"]))
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for (text1, text2), label in self.cache.items():
                if (text1, text2) in existing:
                    continue
                f.write(json.dumps({
                    "text1": text1,
                    "text2": text2,
                    "label": label,
                }, ensure_ascii=False) + "\n")
        self.new_cache_entries = 0

    def check_implication(self, text1, text2):
        key = (text1, text2)
        if key in self.cache:
            return self.cache[key]

        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        prediction = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().item()
        self.cache[key] = prediction
        self.new_cache_entries += 1
        return prediction


def get_semantic_ids(strings_list, model, strict_entailment=False):
    """Group a list of responses into semantic clusters."""

    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2)
        implication_2 = model.check_implication(text2, text1)
        assert implication_1 in [0, 1, 2] and implication_2 in [0, 1, 2]
        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)

        implications = [implication_1, implication_2]
        return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] != -1:
            continue
        semantic_set_ids[i] = next_id
        for j in range(i + 1, len(strings_list)):
            if semantic_set_ids[j] == -1 and are_equivalent(string1, strings_list[j]):
                semantic_set_ids[j] = next_id
        next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods):
    """
    Aggregate response log-probabilities by semantic cluster.

    This mirrors the official code's `sum_normalized` aggregation:
    first normalize over all sampled responses, then log-sum-exp within each
    semantic cluster.
    """
    unique_ids = sorted(set(semantic_ids))
    assert unique_ids == list(range(len(unique_ids)))

    norm = np.log(np.sum(np.exp(log_likelihoods)))
    cluster_log_probs = []
    for uid in unique_ids:
        indices = [i for i, sid in enumerate(semantic_ids) if sid == uid]
        vals = np.array([log_likelihoods[i] for i in indices], dtype=np.float64)
        cluster_log_probs.append(float(np.log(np.sum(np.exp(vals - norm)))))
    return cluster_log_probs


def predictive_entropy_rao(log_probs):
    probs = np.exp(log_probs)
    return float(-np.sum(probs * log_probs))


def cluster_assignment_entropy(semantic_ids):
    counts = np.bincount(semantic_ids)
    probabilities = counts / len(semantic_ids)
    return float(-(probabilities * np.log(probabilities)).sum())


def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_to_vec(X, vec):
    return 1.0 - normalise(X) @ (vec / (np.linalg.norm(vec) + 1e-12))


def cosine_dist_to_loo_centroid(reps):
    """
    Leave-one-out cosine distance for each row in reps.

    Each answerable sample is scored against the centroid of the remaining
    answerable samples from the same form.
    """
    n = len(reps)
    if n < 2:
        raise ValueError("Need at least 2 answerable examples for LOO centroid.")

    sum_all = reps.sum(axis=0)
    dists = np.zeros(n, dtype=np.float64)
    for i in range(n):
        centroid_i = (sum_all - reps[i]) / (n - 1)
        dists[i] = cosine_dist_to_vec(reps[i : i + 1], centroid_i)[0]
    return dists


def load_raw_multi(run_dirs):
    seen = set()
    all_reps = []
    all_meta = []
    offset = 0
    for run_dir in run_dirs:
        reps = np.load(
            os.path.join(run_dir, "reps", "reps_last_raw.npy"), allow_pickle=False
        ).astype(np.float32)
        with open(os.path.join(run_dir, "reps", "meta.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for i, row in enumerate(rows):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            row["_idx"] = offset + i
            all_meta.append(row)
            all_reps.append(reps[i])
        offset += len(rows)
    raw = np.stack(all_reps)
    return raw - raw.mean(axis=0), all_meta


def compute_own_dist(reps, meta):
    label_arr = np.array([row["form"] for row in meta])
    ans_arr = np.array([row["answerable"] for row in meta])
    own_dists = {}
    for form in sorted(set(label_arr)):
        mask_A = (label_arr == form) & (ans_arr == "A")
        if mask_A.sum() == 0:
            continue
        mask_all = label_arr == form
        form_rows = np.array(meta, dtype=object)[mask_all]
        reps_A = reps[mask_A]
        loo_dists_A = cosine_dist_to_loo_centroid(reps_A)
        centroid_A_full = reps_A.mean(axis=0)

        idx_A_form = np.where(mask_A)[0]
        idx_all_form = np.where(mask_all)[0]
        a_pos = {idx: pos for pos, idx in enumerate(idx_A_form)}

        for idx, row in zip(idx_all_form, form_rows):
            if idx in a_pos:
                own_dists[row["id"]] = float(loo_dists_A[a_pos[idx]])
            else:
                own_dists[row["id"]] = float(
                    cosine_dist_to_vec(reps[idx : idx + 1], centroid_A_full)[0]
                )
    return own_dists


def compute_auc_best_f1(scores, labels):
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(labels, scores)
    best_f1 = 0.0
    for threshold in sorted(set(scores)):
        preds = [1 if s >= threshold else 0 for s in scores]
        best_f1 = max(best_f1, f1_score(labels, preds, zero_division=0))
    return float(auc), float(best_f1)


def load_samples(samples_file, expected_samples=None, forms=None):
    grouped = defaultdict(list)
    with open(samples_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if forms and row["form"] not in forms:
                continue
            grouped[row["id"]].append(row)

    examples = []
    for prompt_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["sample_idx"])
        if expected_samples is not None and len(rows) != expected_samples:
            continue
        examples.append({
            "id": prompt_id,
            "form": rows[0]["form"],
            "answerable": rows[0]["answerable"],
            "model": rows[0]["model"],
            "responses": [r["response"] for r in rows],
            "total_log_likelihoods": [
                float(r["total_log_likelihood"]) for r in rows
            ],
        })
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--run-dirs", nargs="+", default=None)
    parser.add_argument("--label", default="semantic_entropy")
    parser.add_argument("--forms", nargs="+", default=None)
    parser.add_argument("--expected-samples", type=int, default=10)
    parser.add_argument("--strict-entailment", action="store_true")
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    exp_root = project_root / "experiments" / "semantic_entropy" / args.label
    out_dir = exp_root / "results"
    out_dir_str = str(out_dir)
    os.makedirs(out_dir_str, exist_ok=True)

    examples = load_samples(
        args.samples_file,
        expected_samples=args.expected_samples,
        forms=set(args.forms) if args.forms else None,
    )
    if not examples:
        raise ValueError("No complete prompts found in samples file.")

    cache_path = args.cache_file or str(exp_root / "cache" / "entailment_cache.jsonl")
    print(f"Loading entailment model: {DEBERTA_MODEL}")
    print(f"Cache file: {cache_path}")
    entailment_model = EntailmentDeberta(cache_path=cache_path)
    print(f"Loaded entailment model on device={DEVICE}.")
    print(f"Prompts to process: {len(examples)}")

    rows_out = []
    for idx, ex in enumerate(examples, start=1):
        if idx == 1 or idx % args.progress_every == 0:
            print(
                f"[{idx}/{len(examples)}] processing {ex['id']} "
                f"({ex['form']}/{ex['answerable']})",
                flush=True,
            )
        semantic_ids = get_semantic_ids(
            ex["responses"],
            entailment_model,
            strict_entailment=args.strict_entailment,
        )
        cluster_log_probs = logsumexp_by_id(
            semantic_ids,
            ex["total_log_likelihoods"],
        )
        rows_out.append({
            "id": ex["id"],
            "form": ex["form"],
            "answerable": ex["answerable"],
            "model": ex["model"],
            "n_samples": len(ex["responses"]),
            "n_clusters": len(set(semantic_ids)),
            "semantic_ids": semantic_ids,
            "semantic_entropy": predictive_entropy_rao(cluster_log_probs),
            "cluster_assignment_entropy": cluster_assignment_entropy(semantic_ids),
        })
        if idx % args.progress_every == 0:
            entailment_model.flush_cache()

    entailment_model.flush_cache()

    labels = [1 if row["answerable"] == "U" else 0 for row in rows_out]
    se_scores = [row["semantic_entropy"] for row in rows_out]
    cae_scores = [row["cluster_assignment_entropy"] for row in rows_out]

    summary = {
        "label": args.label,
        "samples_file": args.samples_file,
        "forms": sorted(set(row["form"] for row in rows_out)),
        "n_prompts": len(rows_out),
        "expected_samples": args.expected_samples,
        "strict_entailment": bool(args.strict_entailment),
        "semantic_entropy_auc": compute_auc_best_f1(se_scores, labels)[0],
        "semantic_entropy_best_f1": compute_auc_best_f1(se_scores, labels)[1],
        "cluster_assignment_entropy_auc": compute_auc_best_f1(cae_scores, labels)[0],
        "cluster_assignment_entropy_best_f1": compute_auc_best_f1(cae_scores, labels)[1],
    }

    if args.run_dirs:
        reps, meta = load_raw_multi(args.run_dirs)
        own_dist = compute_own_dist(reps, meta)
        geom_scores = [own_dist[row["id"]] for row in rows_out]
        geom_auc, geom_best_f1 = compute_auc_best_f1(geom_scores, labels)
        summary["geometry_auc"] = geom_auc
        summary["geometry_best_f1"] = geom_best_f1

    rows_path = str(out_dir / "per_prompt_scores.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = str(out_dir / "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "=" * 64,
        f"SEMANTIC ENTROPY BASELINE  [{args.label}]",
        "=" * 64,
        f"prompts={summary['n_prompts']}  expected_samples={args.expected_samples}",
        f"forms={summary['forms']}",
        f"strict_entailment={summary['strict_entailment']}",
        "",
        "[AUC / BEST-F1]",
        f"semantic_entropy           auc={summary['semantic_entropy_auc']:.4f}  best_f1={summary['semantic_entropy_best_f1']:.4f}",
        f"cluster_assignment_entropy auc={summary['cluster_assignment_entropy_auc']:.4f}  best_f1={summary['cluster_assignment_entropy_best_f1']:.4f}",
    ]
    if "geometry_auc" in summary:
        lines.append(
            f"geometry                   auc={summary['geometry_auc']:.4f}  best_f1={summary['geometry_best_f1']:.4f}"
        )
    lines += [
        "",
        f"per-prompt scores: {rows_path}",
        f"summary json:      {summary_path}",
    ]
    summary_txt = "\n".join(lines) + "\n"

    summary_txt_path = str(out_dir / "summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)

    print(summary_txt, end="")


if __name__ == "__main__":
    main()
