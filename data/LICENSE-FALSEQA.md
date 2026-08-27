# Attribution — FalseQA (basis for `data/falseqa.jsonl`)

`data/falseqa.jsonl` is derived from the **FalseQA** dataset by Hu et al.
(2023). The cleaning pipeline `src/data/clean_falseqa.py` applies minimal
normalizations (preserving row count, A/U balance, and id structure) plus
one documented local dedup rewrite — see the paper's Appendix on data
construction for the full cleaning procedure.

- **Upstream**: <https://github.com/thunlp/FalseQA>
- **Citation**: Hu et al., "Won't Get Fooled Again: Answering Questions
  with False Premises by LLMs," 2023.
- **Upstream LICENSE file**: **No explicit LICENSE file was found in
  the upstream repository at the time of preparing this release** (the
  repo contains `dataset/`, `scripts/`, `src/`, `README.md`, but no
  `LICENSE` / `LICENSE.md` / `COPYING` file; the README does not state
  license terms either). Per GitHub's default policy, source without
  an explicit license is "all rights reserved" under copyright law and
  does not grant automatic redistribution permission.

**This repo does NOT redistribute `data/falseqa.jsonl`.** Because the
upstream license is not explicit, we provide a fetch-and-clean
pipeline instead, so the FalseQA data is obtained by the user
directly from the upstream repository under whatever terms its
authors apply:

```bash
python src/data/fetch_falseqa.py    # clones upstream, writes data/falseqa.jsonl (1374 rows)
python src/data/clean_falseqa.py    # applies the documented local cleanup (App. A.5)
```

The fetch script (`src/data/fetch_falseqa.py`) shallow-clones
`github.com/thunlp/FalseQA`, auto-detects the dataset file in
upstream `dataset/`, converts to the schema expected by this repo
(`{id, form, answerable, category, prompt, source_dataset}` with
ID format `fqa_NNNNa` / `fqa_NNNNu`), and writes `data/falseqa.jsonl`.

By running the fetch script you accept that you are downloading
FalseQA from the upstream repository under that repository's terms
(which, as noted above, are not explicit at the time of this release).
