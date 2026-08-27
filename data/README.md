# Data directory

Five matched-pair JSONL prompt sets are shipped, with fetch-and-clean support
for FalseQA. Each row is a JSON object with
`{id, form, answerable, prompt}` (additional fields vary per file).
`answerable ∈ {A, U}`; an A-prompt and its U-pair share the same form /
topic / construction.

| File | A/U size | Built by | Role in paper | Attribution |
|---|---|---|---|---|
| `math800.jsonl` | 800 A + 800 U (16 categories × 50 pairs) | `src/data/generate_math800.py` + frozen verification | Structural-impossibility benchmark, math domain | this repo's LICENSE |
| `code800.jsonl` | 800 A + 800 U (8 categories × 100 pairs) | `src/data/generate_code800.py` + frozen verification | Structural-impossibility benchmark, code domain | this repo's LICENSE |
| `fact800.jsonl` | 800 A + 800 U (SQuAD 2.0 sampled, seed=42) | `src/data/prepare_squad2.py` + `src/data/fix_fact800.py` | Epistemic-unanswerability scope-boundary benchmark | `LICENSE-SQUAD.md` |
| `falseqa.jsonl` (**NOT SHIPPED**) | matched-pair cleaned subset | `src/data/fetch_falseqa.py` + `src/data/clean_falseqa.py` (Hu et al., 2023) | Zero-shot false-premise transfer scope boundary | `LICENSE-FALSEQA.md` |
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

## `math800` / `code800` construction

`math800.jsonl` and `code800.jsonl` are frozen release datasets. The repository ships the stochastic candidate-generation scripts, but the frozen post-verification JSONL files—not exact generator replay—are the reproducibility targets.

- **math800** — 16 structural-impossibility categories, 800 answerable (A) + 800 unanswerable (U) prompts (50 matched A/U pairs per category).
- **code800** — 8 Python runtime-failure categories, 800 A + 800 U prompts (100 matched A/U pairs per category).
- Within each A/U pair the two prompts share form, topic, and construction, differing only in answerability: A is well-posed, while U is structurally impossible (math) or raises a category-correct runtime failure (code).
- Prompts were produced by LLM-assisted candidate drafting and then filtered and verified before freezing — math A/U checked against each category's formal rule, code A/U checked by executing the expression in CPython, with duplicate prompts removed.

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

- The shipped `math800.jsonl` / `code800.jsonl` are the **frozen release datasets** used in the paper and are the reproducibility target. Their A/U status was checked by the category rules described above (and by CPython execution for code). The stochastic generator scripts record the original API configuration but do not guarantee byte-identical replay.
- `fact800.jsonl` contains CJK character content in some questions/contexts — these are genuine SQuAD 2.0 prompts about Mandarin/Hokkien etymology, place names, and similar topics. The CJK is meaningful prompt data, not metadata.
- See the per-dataset `LICENSE-*.md` files for upstream attribution and redistribution terms.
