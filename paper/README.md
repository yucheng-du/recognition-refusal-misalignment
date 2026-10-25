# Figure regeneration

This directory contains figure assets and their regeneration pipeline. The
[paper](https://aclanthology.org/2026.emnlp-main.1078/) is published in the EMNLP 2026
proceedings and is also available on [arXiv](https://arxiv.org/abs/2608.29109).
Manuscript source (`main.tex`, `references.bib`, etc.) is
maintained separately and is not redistributed in this code/data release.

## Quickstart

After installing the root `requirements.txt`, run from the repository root:

```bash
python3 paper/generate_fig5_v2.py
python3 paper/generate_figures.py --figs fig2 fig3 fig4 fig6
```

The standalone TikZ teaser additionally requires LaTeX:

```bash
cd paper/figures
pdflatex -interaction=nonstopmode fig1_teaser.tex
```

## Figure inputs and outputs

Script figure numbers follow the asset names and can differ from the paper's
final figure numbers. Both matplotlib generators use `_figure_style.py`.

| Paper placement | Output under `paper/figures/` | Regeneration command (from repo root) | Inputs |
|---|---|---|---|
| Fig 1 (page-1 teaser) | `fig1_teaser.pdf` | TikZ command above | `paper/figures/fig1_teaser.tex` |
| Fig 2 (orthogonality) | `fig3_orthogonality.pdf` | `python3 paper/generate_figures.py --figs fig3` | `experiments/main_grid_facts_v2.json` + the 12-file BASE_PAIRS `direction_comparison_*.json` set |
| Fig 3 (causal control) | `fig5_causal_v2_compact.pdf` | `python3 paper/generate_fig5_v2.py` | The 12 anchor `experiments/intervention/intervention_*_full_v2.json` files |
| Appendix detection heatmap | `fig2_detection_heatmap.pdf` | `python3 paper/generate_figures.py --figs fig2` | `experiments/main_grid_facts_v2.json` |
| Appendix null-space ablation | `fig6_nullspace_ablation.pdf` | `python3 paper/generate_figures.py --figs fig6` | `experiments/ablation_nullpc_results_11model.json` |
| Appendix GSRS ablation | `fig4_gsrs_ablation.pdf` | `python3 paper/generate_figures.py --figs fig4` | Frozen constants in script; no file I/O |

These figures regenerate from shipped sources and aggregates without the raw
activation tree. The causal-control generator also produces the Mistral code
A→U dose-response panel. Rendered PDFs are included in `figures/`.

## Other released analysis targets

`experiments/d_struct_behav_matrix.json` is the frozen aggregate for the
behavior-defined invalidity-aware direction. Its builder needs omitted raw
activations and effective intervention-label TSVs; see the
[pipeline release boundaries](../scripts/README.md#raw-reruns-and-release-boundaries).

See the [multidimensional robustness guide](../analysis/subspace_overlap/README.md)
for the frozen JSON results and rerun requirements for the paper's
5–10-dimensional robustness check.
