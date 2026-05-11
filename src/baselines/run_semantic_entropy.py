"""
run_semantic_entropy.py — Generate N samples per prompt for semantic entropy.

This is initially a direct copy of an earlier self-consistency runner so we
can reuse the same multi-sample generation pipeline for the semantic entropy
baseline. At this stage, it only handles answer sampling and JSONL output; the
semantic entropy logic itself will be implemented in a separate analysis step.

Key design choices:
  - Uses num_return_sequences=N to generate all samples in a single forward pass
    (3-4x faster than looping N times per prompt).
  - Default max_new_tokens=50 (sufficient to capture the answer for short
    math/fact/code questions; saves ~50% decoding time vs. 100 tokens).
  - Tokenises each prompt once and reuses the tensor.
  - Supports resume: appends to existing output file, skipping already-done ids.

Usage:
  # MATH-50 for one model (recommended first semantic entropy validation)
  python src/baselines/run_semantic_entropy.py --model llama --n-samples 10 \\
      --run-dirs experiments/runs/run_003 experiments/runs/run_003b experiments/runs/run_003c \\
      --out-file experiments/semantic_entropy/llama_math50/samples/llama_math50.jsonl

  # CODE-30
  python src/baselines/run_semantic_entropy.py --model qwen --n-samples 10 \\
      --run-dirs experiments/runs/run_004_code_qwen \\
      --out-file experiments/semantic_entropy/qwen_code30/samples/qwen_code30.jsonl

Output JSONL (one line per sample):
  {"id": "m01a", "form": "MATH", "answerable": "A",
   "sample_idx": 0, "response": "...", "token_log_likelihoods": [...],
   "model": "Llama-3.1-8B-Instruct"}
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATHS = {
    "llama": os.path.expanduser(
        "~/.llama/checkpoints/Llama3.1-8B-Instruct-HF"
    ),
    "qwen": os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B-Instruct",
    "qwen":    "Qwen2.5-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts_from_dirs(run_dirs):
    """Load and deduplicate prompts from multiple run directories."""
    seen_ids = set()
    prompts = []
    for d in run_dirs:
        path = os.path.join(d, "prompts.jsonl")
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    prompts.append(row)
    return prompts


def extract_response_and_logprobs(tokenizer, full_sequence, prompt_len, transition_scores):
    """
    Convert one generated sequence into:
      - response text (special tokens removed)
      - per-token log-likelihoods for the generated continuation
      - summed log-likelihood over the continuation

    We trim at the first EOS token if present so the likelihoods and decoded
    text refer to the same continuation span.
    """
    gen_token_ids = full_sequence[prompt_len:]
    gen_scores = transition_scores[: len(gen_token_ids)]

    eos_id = tokenizer.eos_token_id
    cutoff = len(gen_token_ids)
    if eos_id is not None:
        for idx, tok_id in enumerate(gen_token_ids.tolist()):
            if tok_id == eos_id:
                cutoff = idx
                break

    trimmed_ids = gen_token_ids[:cutoff]
    trimmed_scores = gen_scores[:cutoff]

    response = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
    token_log_likelihoods = [float(x) for x in trimmed_scores.tolist()]
    total_log_likelihood = float(sum(token_log_likelihoods))
    return response, token_log_likelihoods, total_log_likelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "mistral"])
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples per prompt (default 10). All generated in one pass.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens per sample (default 50; sufficient for "
                             "short math/fact/code answers and ~2x faster than 100).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    done_ids = set()
    if os.path.isfile(args.out_file):
        id_counts: dict[str, int] = {}
        with open(args.out_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                id_counts[r["id"]] = id_counts.get(r["id"], 0) + 1
        done_ids = {pid for pid, cnt in id_counts.items() if cnt >= args.n_samples}
        print(f"Resume: {len(done_ids)} prompts already fully done.")

    device     = get_device()
    model_id   = MODEL_PATHS[args.model]
    model_name = MODEL_LABELS[args.model]

    print(f"Model        : {model_name}")
    print(f"Device       : {device}")
    print(f"Samples/prompt: {args.n_samples}  (generated in one pass per prompt)")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature  : {args.temperature}  top_p: {args.top_p}")
    print(f"Output       : {args.out_file}\n")

    prompts = load_prompts_from_dirs(args.run_dirs)
    todo = [p for p in prompts if p["id"] not in done_ids]
    print(f"Prompts total: {len(prompts)}  |  Remaining: {len(todo)}\n")

    if not todo:
        print("Nothing to do — all prompts already complete.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16
    ).to(device)
    model.eval()

    with open(args.out_file, "a", encoding="utf-8") as out_f:
        for idx, row in enumerate(todo):
            pid, form, answerable, prompt_text = (
                row["id"], row["form"], row["answerable"], row["prompt"]
            )

            inputs = tokenizer(
                prompt_text, return_tensors="pt", padding=False
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.n_samples,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            transition_scores = model.compute_transition_scores(
                gen_out.sequences,
                gen_out.scores,
                normalize_logits=True,
            )

            for s_idx, gen in enumerate(gen_out.sequences):
                response, token_log_likelihoods, total_log_likelihood = extract_response_and_logprobs(
                    tokenizer,
                    gen,
                    prompt_len,
                    transition_scores[s_idx],
                )
                out_f.write(json.dumps({
                    "id":         pid,
                    "form":       form,
                    "answerable": answerable,
                    "sample_idx": s_idx,
                    "response":   response,
                    "token_log_likelihoods": token_log_likelihoods,
                    "total_log_likelihood": total_log_likelihood,
                    "model":      model_name,
                }, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"  [{idx+1:>3}/{len(todo)}] {pid} ({form}/{answerable})")

    print(f"\nDone. Output: {args.out_file}")


if __name__ == "__main__":
    main()
