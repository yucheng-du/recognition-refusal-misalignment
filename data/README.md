# Data directory

Six matched-pair `.jsonl` prompt sets. Each row is a JSON object with
`{id, form, answerable, prompt}` (additional fields vary per file).
`answerable ∈ {A, U}`; an A-prompt and its U-pair share the same form /
topic / construction.

| File | A/U size | Built by | Role in paper | Attribution |
|---|---|---|---|---|
| `math800.jsonl` | 800 A + 800 U (12–15 categories × ~55 pairs) | `src/data/generate_math800.py` | Structural-impossibility benchmark, math domain | this repo's LICENSE |
| `code800.jsonl` | 800 A + 800 U (8 categories × 100 pairs) | `src/data/generate_code800.py` | Structural-impossibility benchmark, code domain | this repo's LICENSE |
| `fact800.jsonl` | 800 A + 800 U (SQuAD 2.0 sampled, seed=42) | `src/data/prepare_squad2.py` + `src/data/fix_fact800.py` | Epistemic-unanswerability scope-boundary benchmark | `LICENSE-SQUAD.md` |
| `falseqa.jsonl` | matched-pair cleaned subset | `src/data/clean_falseqa.py` (cleaning of Hu et al. 2023) | Zero-shot false-premise transfer scope boundary | `LICENSE-FALSEQA.md` |
| `abstentionbench_gsm8k.jsonl` | matched-pair GSM8K subset of AbstentionBench | `src/data/clean_abstentionbench_gsm8k.py` | Natural-distribution epistemic-style transfer + length-control analysis | `LICENSE-ABSTENTIONBENCH.md` |
| `difficulty_control_gsm8k.jsonl` | difficulty-controlled split | `scripts/prepare_difficulty_control.py` | Difficulty-axis null check (§5) | `LICENSE-GSM8K.md` |

---

## Row schema

```jsonc
{
  "id": "sq001a",           // unique id; convention: stem + 'a' (answerable) / 'u' (unanswerable)
  "form": "FACT",           // task type: MATH / CODE / FACT / FALSE-PREMISE / NATURAL
  "answerable": "A",        // "A" or "U"
  "prompt": "Context: ...\nQuestion: ...\nAnswer:"
}
```

For the self-built math/code sets, the matched-pair structure is preserved by an additional `pair_id` field linking each A to its U partner.

---

## Rebuilding `fact800.jsonl` from scratch

```bash
# Step 1: generate prompts (one-time; all models share the output)
# Run on a host that has HuggingFace access (the datasets library will fetch SQuAD 2.0).
cd /path/to/repo
python src/data/prepare_squad2.py \
    --split train \
    --n-pairs 800 \
    --out-file data/fact800.jsonl \
    --seed 42
# Output: data/fact800.jsonl  (800 A + 800 U = 1600 rows)
```

For the other transfer datasets (`falseqa.jsonl`, `abstentionbench_gsm8k.jsonl`, `difficulty_control_gsm8k.jsonl`), see the corresponding `src/data/clean_*.py` or `scripts/prepare_difficulty_control.py`.

---

## Notes

- The shipped self-built `math800.jsonl` / `code800.jsonl` are the post-verification frozen versions used in the paper (the generation scripts call an external LLM API to synthesize candidates; you do **not** need to re-run them for reproducing paper results).
- `fact800.jsonl` contains CJK character content in some questions/contexts — these are genuine SQuAD 2.0 prompts about Mandarin/Hokkien etymology, place names, and similar topics. The CJK is meaningful prompt data, not metadata.
- See the per-dataset `LICENSE-*.md` files for upstream attribution and redistribution terms.
