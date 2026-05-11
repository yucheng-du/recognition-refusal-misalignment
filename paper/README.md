# `paper/` — figure regeneration only

The paper PDF is on the OpenReview submission page; the paper source
(`main.tex`, `references.bib`, etc.) is not redistributed in this
anonymous code/data release.

This directory ships only the figure-regeneration pipeline:

| File | Purpose |
|---|---|
| `generate_figures.py` | Regenerates Figs 2, 3, 4, 6 from shipped aggregate JSONs under `experiments/`. CLI: `python paper/generate_figures.py --figs fig2 fig3 fig4 fig6`. Fig 1 is skipped with a clear message (requires the unshipped raw-activation tree). |
| `generate_fig5_v2.py` | Regenerates Fig 5 (4-anchor causal-control grid + Mistral code A→U dose-response) from the shipped intervention JSONs under `experiments/intervention/`. |
| `_figure_style.py` | Shared style module used by both generators. |
| `figures/` | The six rendered figure PDFs as embedded in the paper. |

See the root `README.md` "Figure reproducibility" section for the
per-figure regeneration matrix and the input-data dependencies.
