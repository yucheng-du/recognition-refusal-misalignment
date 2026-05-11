# Attribution — GSM8K (basis for `data/difficulty_control_gsm8k.jsonl`)

`data/difficulty_control_gsm8k.jsonl` is a difficulty-controlled split
derived from the **GSM8K** dataset by Cobbe et al. (2021), built via
`scripts/prepare_difficulty_control.py`. It supports the §5 difficulty-
control analysis ($d_{\mathrm{imp}}$ AUC 0.61 as a hard-vs-easy classifier
vs. 0.96 on impossibility, $\Delta = +0.35$).

- **Upstream**: <https://github.com/openai/grade-school-math>
- **Citation**: Cobbe et al., "Training Verifiers to Solve Math Word
  Problems," 2021 (arXiv:2110.14168).
- **License**: MIT License (per the upstream repository).

The MIT license permits redistribution-with-attribution, so the shipped
`difficulty_control_gsm8k.jsonl` is freely redistributable as a derivative
of GSM8K. If you re-derive the split, the GSM8K download itself is the
sole upstream input.
