# Recognition–Refusal Misalignment in LLMs — Anonymous Submission

This repository accompanies the EMNLP 2026 submission *"Recognition–Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions."* It contains the matched-pair datasets, the analysis / extraction / steering pipeline, the figure-regeneration scripts, and the curated experiment artifacts needed to reproduce the shipped figure assets. The paper PDF itself is on the OpenReview submission page; the paper source (`main.tex`, `references.bib`, etc.) is not redistributed in this code/data release.

This release is anonymized for double-blind review. A de-anonymized version will be released on acceptance.

---

## TL;DR finding

Across an 11-model main grid spanning 1.7B–70B and two structural-impossibility domains (math, code):

1. **The model knows.** A one-dimensional null-space MeanDiff direction $d_{\mathrm{imp}}$ separates answerable (A) from unanswerable (U) prompts with mean AUC **0.939** across 22 cells.
2. **But it doesn't act.** $d_{\mathrm{imp}}$ is near-orthogonal to the canonical safety-refusal direction $d_{\mathrm{ref,safety}}$ (Arditi et al., 2024): mean $\cos$ **0.087**, range $[0.020, 0.130]$. It is only partially aligned with an in-domain behavior-defined invalidity-aware direction ($\cos \approx 0.40$).
3. **The misalignment largely predates RLHF.** Across 6 base/instruct pairs on math800, $\Delta\cos$ is $[-0.008, +0.110]$ (mean $+0.037$) — instruction tuning modulates the angle in a low-cosine regime rather than producing it.
4. **The direction is causal.** Generation-time activation steering on $d_{\mathrm{imp}}$ changes invalidity-aware behavior on the 4-anchor math/code intervention grid; gated $\Delta$G improves by **+33 to +52 pp**, and a 16-model steering sweep confirms the math/code footprint.

The full paper PDF is on the OpenReview submission page (this anonymous code/data release is paired with that submission and does not redistribute the paper source).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Regenerate the standalone TikZ teaser and the matplotlib figures
cd paper/figures && pdflatex -interaction=nonstopmode fig1_teaser.tex && cd ../..
python paper/generate_fig5_v2.py
python paper/generate_figures.py --figs fig2 fig3 fig4 fig6

# 3. The manuscript PDF/source is not redistributed here
#    Use the OpenReview submission PDF for the paper text.
```

A reviewer-only smoke test takes < 5 minutes.

For the heavier pipeline (extracting hidden states from a HuggingFace model, fitting probes, running steering), see `scripts/README.md` and the per-script CLI help. The intervention pipeline (`scripts/intervention_nullspace.py`) requires GPU access and human verification; results in `experiments/intervention/intervention_*_full_v2.json` are the shipped post-verification artifacts.

---

## Figure reproducibility

| Paper placement | `\includegraphics` target | Regen status | Command + inputs |
|---|---|---|---|
| Fig 1 (page-1 teaser) | `figures/fig1_teaser.pdf` | **Yes** | `cd paper/figures && pdflatex -interaction=nonstopmode fig1_teaser.tex` |
| Fig 2 (orthogonality) | `figures/fig3_orthogonality.pdf` | **Yes** | `python paper/generate_figures.py --figs fig3` reads `experiments/main_grid_facts_v2.json` + the 12-file BASE_PAIRS `direction_comparison_*.json` set |
| Fig 3 (causal control) | `figures/fig5_causal_v2_compact.pdf` | **Yes** | `python paper/generate_fig5_v2.py` reads the 12 anchor `experiments/intervention/intervention_*_full_v2.json` files |
| Appendix detection heatmap | `figures/fig2_detection_heatmap.pdf` | **Yes** | `python paper/generate_figures.py --figs fig2` reads `experiments/main_grid_facts_v2.json` |
| Appendix null-space ablation | `figures/fig6_nullspace_ablation.pdf` | **Yes** | `python paper/generate_figures.py --figs fig6` reads `experiments/ablation_nullpc_results_11model.json` |
| Appendix GSRS ablation | `figures/fig4_gsrs_ablation.pdf` | **Yes** | `python paper/generate_figures.py --figs fig4` — frozen constants in script, no file I/O |

Fig 1 is a standalone TikZ overview figure and does not require the raw activation-tensor tree. The raw 151 GB `experiments/signals/` tree is still needed only if reviewers want to recompute representations from scratch rather than use the shipped aggregate artifacts.

`experiments/d_struct_behav_matrix.json` is a small derived aggregate for the behavior-defined invalidity-aware direction reported in the paper. Its local builder consumes the omitted raw `experiments/signals/*.npy` activations plus post-verification intervention labels; the JSON is therefore shipped as the reproducibility target for this comparison, rather than as a one-command-from-scratch regeneration step in the lightweight anonymous release.

---

## Datasets

| File | Size | Source | License |
|---|---|---|---|
| `data/math800.jsonl` | 260 KB | Self-built frozen set (16 categories × 50 matched A/U pairs; LLM-drafted, rule/manually verified) | this repo's LICENSE |
| `data/code800.jsonl` | 263 KB | Self-built frozen set (8 categories × 100 matched A/U pairs; LLM-drafted, rule/manually verified) | this repo's LICENSE |
| `data/fact800.jsonl` | 1.35 MB | Derived from SQuAD 2.0 train split (seed=42, official `is_impossible` labels) | SQuAD 2.0 (CC-BY-SA-4.0) — see `data/LICENSE-SQUAD.md` |
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
│   ├── generate_figures.py                   (matplotlib figures: detection, orthogonality, GSRS, null-space)
│   ├── generate_fig5_v2.py                   (Fig 5 end-to-end)
│   └── figures/                              (rendered figure PDFs + Fig 1 TikZ source fig1_teaser.tex)
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
    ├── d_struct_behav_matrix.json           (behavior-defined invalidity-aware direction matrix)
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

Some post-adjudication aggregates, including `experiments/d_struct_behav_matrix.json`, additionally require verified intervention-label TSVs. These are represented in the release by the frozen aggregate JSONs used by the paper figures/tables; full regeneration from raw activations follows the same extraction and verification pipeline but is outside the lightweight smoke-test path.

---

## Citation

```bibtex
@misc{anon2026,
  author = {Anonymous},
  title  = {Recognition--Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions},
  year   = {2026},
  note   = {EMNLP 2026 submission, under review}
}
```

---

## Reproducibility note

All numerical claims in the paper PDF (on OpenReview) derive from the shipped artifacts under `experiments/` (curated subset). Re-running the extraction + analysis pipeline end-to-end from raw model weights is supported but requires ~150 GB of disk for the signals tree and a GPU for the intervention / steering experiments.
