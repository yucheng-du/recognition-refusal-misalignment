# `paper/` — figure regeneration only

The accepted manuscript is on the [OpenReview submission page](https://openreview.net/forum?id=ShHf3O62rH); the paper source
(`main.tex`, `references.bib`, etc.) is maintained separately and is not
redistributed in this code/data release.

This directory ships only the figure-regeneration pipeline:

| File | Purpose |
|---|---|
| `figures/fig1_teaser.tex` | Standalone TikZ source for the page-1 teaser. CLI: `cd paper/figures && pdflatex -interaction=nonstopmode fig1_teaser.tex`. |
| `generate_figures.py` | Regenerates the matplotlib figures from shipped aggregate JSONs under `experiments/`. CLI: `python paper/generate_figures.py --figs fig2 fig3 fig4 fig6`. |
| `generate_fig5_v2.py` | Regenerates Fig 5 (4-anchor causal-control grid + Mistral code A→U dose-response) from the shipped intervention JSONs under `experiments/intervention/`. |
| `_figure_style.py` | Shared style module used by both generators. |
| `figures/` | Rendered figure PDFs used in the paper, plus the Fig 1 TikZ source `fig1_teaser.tex`. |

See the root `README.md` "Figure reproducibility" section for the
per-figure regeneration matrix and the input-data dependencies.
