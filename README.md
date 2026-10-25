# Recognition–Refusal Misalignment in LLMs

Code, datasets, and frozen experiment artifacts for **Recognition–Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions** (EMNLP 2026 Main Conference).

**[Proceedings](https://aclanthology.org/2026.emnlp-main.1078/) · [arXiv](https://arxiv.org/abs/2608.29109) · [Hugging Face datasets](https://huggingface.co/datasets/yucheng-du/recognition-refusal-misalignment) · [Pipeline guide](scripts/README.md) · [Figure regeneration](paper/README.md)**

Across 11 instruction-tuned models (1.7B–70B), a one-dimensional direction detects structurally unanswerable math/code prompts with mean AUC **0.939**, yet is nearly orthogonal to the safety-refusal direction (mean cosine **0.087**).

Yucheng Du (University of Southern California; [email](mailto:yuchengd@usc.edu)) and [Xiyang Hu](https://xiyanghu.github.io) (Arizona State University; [correspondence](mailto:xiyanghu@asu.edu)).

## Load the datasets

`math800` and `code800` each contain **800 matched answerable/unanswerable pairs** (1,600 rows). Both are self-built, frozen datasets with LLM-drafted candidates and deterministic verification.

```bash
pip install datasets
```

```python
from datasets import load_dataset

repo_id = "yucheng-du/recognition-refusal-misalignment"
math = load_dataset(repo_id, "math800", split="test")
code = load_dataset(repo_id, "code800", split="test")
assert len(math) == len(code) == 1600
```

The Hub `test` split contains each **complete frozen set**, not the paper's experimental holdout. Rows include `id`, `form`, `answerable` (`A`/`U`), and `prompt`; recover the pair key with `row["id"][:-1]` (there is no `pair_id` column).

The Hub JSONL files are byte-identical to this release; the dataset card records source commit and SHA-256 checksums. See the [data guide](data/README.md) for construction, schema, and the auxiliary datasets. The Hub release includes only `math800` and `code800`.

## Reproduce the released results

Run from the repository root:

```bash
pip install -r requirements.txt
python3 scripts/verify_core_conclusions.py
```

This lightweight check validates the **shipped aggregate JSONs and provenance metadata**. The [figure guide](paper/README.md) provides regeneration commands and inputs; the [pipeline guide](scripts/README.md) covers extraction, probing, intervention, and steering.

Raw reruns require separately downloaded model weights, approximately **151 GB** for the omitted activation tree, and a GPU for intervention/steering. Some label-dependent aggregates also require effective label TSVs that are **not included**. The released frozen aggregates are the reproducibility targets for those comparisons; they use LLM-assisted labels with provisional audit-subset fills or candidate-label passthrough, not completed human adjudication. See [rerun requirements and label provenance](scripts/README.md#raw-reruns-and-release-boundaries).

Additional resources:

- [Frozen experiment artifacts](experiments/) and [intervention factsheet](experiments/intervention/v2_final_4anchor_factsheet.md)
- [Multidimensional robustness checks](analysis/subspace_overlap/README.md)
- [Theoretical derivations](docs/theory/)

## Citation

The [published paper](https://aclanthology.org/2026.emnlp-main.1078/) and [arXiv preprint](https://arxiv.org/abs/2608.29109) are available. Manuscript source is maintained separately and is not included in this release.

```bibtex
@inproceedings{du-hu-2026-recognition,
    title = "Recognition{--}Refusal Misalignment in {LLM}s: Why Models Answer Structurally Unanswerable Questions",
    author = "Du, Yucheng  and
      Hu, Xiyang",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1078/",
    pages = "24766--24794",
    ISBN = "979-8-89176-459-0"
}
```

## License

The [MIT License](LICENSE) covers original code and self-created artifacts, including `math800` and `code800`. Third-party data and model weights retain their upstream terms:

- `fact800`: [SQuAD 2.0 / CC BY-SA 4.0](data/LICENSE-SQUAD.md).
- AbstentionBench-GSM8K: [CC BY-NC 4.0 and source attribution](data/LICENSE-ABSTENTIONBENCH.md).
- Difficulty-controlled GSM8K: [MIT and source attribution](data/LICENSE-GSM8K.md).
- FalseQA: [fetch-only; not redistributed](data/LICENSE-FALSEQA.md), because upstream had no explicit license at release time.
