# The Model Knows but Doesn't Act — Anonymous Submission

This repository accompanies the EMNLP 2026 submission *"The Model Knows but Doesn't Act: Why LLMs Answer Structurally Impossible Questions."* It contains the matched-pair datasets, the analysis / extraction / steering pipeline, the figure-regeneration scripts, and the curated experiment artifacts needed to reproduce 5 of 6 figures end-to-end. The paper PDF itself is on the OpenReview submission page; the paper source (`main.tex`, `references.bib`, etc.) is not redistributed in this code/data release.

This release is anonymized for double-blind review. A de-anonymized version will be released on acceptance.

---

## TL;DR finding

Across an 11-model main grid spanning 1.7B–70B and two structural-impossibility domains (math, code):

1. **The model knows.** A one-dimensional null-space MeanDiff direction $d_{\mathrm{imp}}$ separates answerable (A) from unanswerable (U) prompts with mean AUC **0.939** across 22 cells.
2. **But it doesn't act.** $d_{\mathrm{imp}}$ is near-orthogonal to the safety refusal direction $d_{\mathrm{ref}}$ (Arditi et al., 2024): mean $\cos$ **0.087**, range $[0.020, 0.130]$, 20/22 behavior-verified.
3. **The misalignment largely predates RLHF.** Across 6 base/instruct pairs on math800, $\Delta\cos$ is $[-0.008, +0.110]$ (mean $+0.037$) — instruction tuning modulates the angle in a low-cosine regime rather than producing it.
4. **The direction is causal.** Generation-time activation steering on $d_{\mathrm{imp}}$ flips refusal bidirectionally on the 4-anchor intervention grid (Mistral-7B-Instruct, Gemma-3-4B-it, Qwen3-14B, Qwen3-8B); gated flip rates +33 to +52 pp.

The full paper PDF is on the OpenReview submission page (this anonymous code/data release is paired with that submission and does not redistribute the paper source).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Regenerate the 5-of-6 figures that are end-to-end reproducible from this release
python paper/generate_fig5_v2.py
python paper/generate_figures.py --figs fig2 fig3 fig4 fig6

# 3. Rebuild the paper PDF (requires a TeX Live installation)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

A reviewer-only smoke test takes < 5 minutes.

For the heavier pipeline (extracting hidden states from a HuggingFace model, fitting probes, running steering), see `scripts/README.md` and the per-script CLI help. The intervention pipeline (`scripts/intervention_nullspace.py`) requires GPU access and human verification; results in `experiments/intervention/intervention_*_full_v2.json` are the shipped post-verification artifacts.

---

## Figure reproducibility

| Figure | `\includegraphics` target | Regen status | Command + inputs |
|---|---|---|---|
| Fig 1 (conceptual) | `figures/fig1_conceptual_compact.pdf` | **No** — requires unshipped `experiments/signals/*.npy` (≈150 GB raw activations) | shipped PDF only; the regen script prints a clear skip message |
| Fig 2 (detection heatmap) | `figures/fig2_detection_heatmap.pdf` | **Yes** | `python paper/generate_figures.py --figs fig2` reads `experiments/main_grid_facts_v2.json` |
| Fig 3 (orthogonality) | `figures/fig3_orthogonality.pdf` | **Yes** | `python paper/generate_figures.py --figs fig3` reads `experiments/main_grid_facts_v2.json` + the 12-file BASE_PAIRS `direction_comparison_*.json` set |
| Fig 4 (GSRS ablation) | `figures/fig4_gsrs_ablation.pdf` | **Yes** | `python paper/generate_figures.py --figs fig4` — frozen constants in script, no file I/O |
| Fig 5 (causal control) | `figures/fig5_causal_v2_compact.pdf` | **Yes** | `python paper/generate_fig5_v2.py` reads the 12 anchor `experiments/intervention/intervention_*_full_v2.json` files |
| Fig 6 (null-space ablation) | `figures/fig6_nullspace_ablation.pdf` | **Yes** | `python paper/generate_figures.py --figs fig6` reads `experiments/ablation_nullpc_results_11model.json` |

Fig 1's raw inputs are the 150 GB activation-tensor tree; the rendered PDF in `paper/figures/` is the canonical Fig 1 used in the OpenReview paper. Reviewers running `python paper/generate_figures.py --figs fig1` will see a clear skip message rather than a crash.

---

## Datasets

| File | Size | Source | License |
|---|---|---|---|
| `data/math800.jsonl` | 260 KB | Self-built (12–15 categories × ~55 matched A/U pairs) | this repo's LICENSE |
| `data/code800.jsonl` | 263 KB | Self-built (8 categories × ~100 matched A/U pairs) | this repo's LICENSE |
| `data/fact800.jsonl` | 1.35 MB | Derived from SQuAD 2.0 validation split (seed=42, official `is_impossible` labels) | SQuAD 2.0 (CC-BY-SA-4.0) — see `data/LICENSE-SQUAD.md` |
| `data/falseqa.jsonl` | NOT SHIPPED | Reconstruct via `python src/data/fetch_falseqa.py && python src/data/clean_falseqa.py` (fetches from upstream `github.com/thunlp/FalseQA`, then applies cleanup). Upstream has no explicit LICENSE file at time of release; see `data/LICENSE-FALSEQA.md` for details. | upstream FalseQA — see `data/LICENSE-FALSEQA.md` |
| `data/abstentionbench_gsm8k.jsonl` | 841 KB | GSM8K subset of AbstentionBench; see `src/data/clean_abstentionbench_gsm8k.py` | upstream AbstentionBench — see `data/LICENSE-ABSTENTIONBENCH.md` |
| `data/difficulty_control_gsm8k.jsonl` | 194 KB | Difficulty-controlled split derived from GSM8K (Cobbe et al., 2021); see `scripts/prepare_difficulty_control.py` | upstream GSM8K — see `data/LICENSE-GSM8K.md` |

Each row is a JSON object with at least `{id, form, answerable, prompt}` where `answerable ∈ {A, U}`. The matched-pair structure (A vs. U sharing the same form/topic) is preserved across all six datasets.

---

## Required model weights (download separately from HuggingFace Hub)

The pipeline expects 11 instruct models and 6 base models, all fetched from the HuggingFace Hub at runtime. Set `HF_HOME` if you want them cached outside `~/.cache/huggingface`. The full list is registered in `scripts/run_model_suite.sh` (`MODEL_PATHS`); the main grid is:

- `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`
- `google/gemma-3-4b-it`, `google/gemma-3-12b-it`
- `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-4-mini-instruct`
- `allenai/OLMo-2-1124-13B-Instruct`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- Plus the matched base models for the 6 base/instruct pairs (Qwen2.5-7B/14B/32B base, Qwen3-8B/14B base, the Llama-3 70B base proxy).

Per-model peak layers are pre-resolved and registered in the relevant scripts; see `scripts/find_best_layers_smollm2_gemma2_phi3.py` for how to re-resolve.

---

## Repository layout

```
.
├── README.md                                 (this file)
├── LICENSE                                   (placeholder for anonymous review)
├── requirements.txt
├── paper/                                    (figure pipeline only — paper PDF is on OpenReview)
│   ├── README.md                             (what this directory ships and what it doesn't)
│   ├── _figure_style.py                      (shared matplotlib style helpers)
│   ├── generate_figures.py                   (Figs 2, 3, 4, 6 + Fig 1 skip)
│   ├── generate_fig5_v2.py                   (Fig 5 end-to-end)
│   └── figures/                              (6 rendered figure PDFs used in the OpenReview paper)
├── scripts/                                  (CLI entry points: extraction, detection, intervention, steering, evaluation)
├── src/
│   ├── baselines/                            (semantic-entropy baseline)
│   └── data/                                 (dataset construction)
├── analysis/                                 (post-hoc analyses for appendix tables)
├── docs/theory/                              (3 derivations cited by paper appendix)
├── data/                                     (6 matched-pair .jsonl datasets + 4 LICENSE-* stubs)
└── experiments/                              (curated subset; aggregate JSONs + the 4-anchor intervention factsheet)
    ├── main_grid_facts_v2.json
    ├── ablation_nullpc_results_11model.json
    ├── direction_comparison_*.json           (12 files: 6 instruct + 6 base, for Fig 3 base/instruct panel)
    ├── intervention/
    │   ├── v2_final_4anchor_factsheet.md
    │   └── intervention_*_full_v2.json       (13 cells = 4 anchors × 3 datasets + 1 supplementary)
    ├── steering/                             (per-cell aggregate steering JSONs across 16 models)
    └── analysis/d_ref_energy_decomp/         (11-model energy-decomposition table)
```

The 151 GB `experiments/signals/` raw activation tree is intentionally **not** in this release. Reviewers who want to recompute representations from scratch should re-run the extraction pipeline:

```bash
python scripts/run_extract_minimal.py --model <hf_id> --prompts data/math800.jsonl \
    --run-dir experiments/signals/math800_<key>_allL --all-layers --no-gradients
```

This populates the same `experiments/signals/` layout the analysis scripts read.

---

## Citation

```bibtex
@misc{anon2026,
  author = {Anonymous},
  title  = {The Model Knows but Doesn't Act: Why LLMs Answer Structurally Impossible Questions},
  year   = {2026},
  note   = {EMNLP 2026 submission, under review}
}
```

---

## Reproducibility note

All numerical claims in the paper PDF (on OpenReview) derive from the shipped artifacts under `experiments/` (curated subset). Re-running the extraction + analysis pipeline end-to-end from raw model weights is supported but requires ~150 GB of disk for the signals tree and a GPU for the intervention / steering experiments.
