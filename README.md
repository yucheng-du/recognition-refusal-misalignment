# Recognition–Refusal Misalignment in LLMs

This repository accompanies the EMNLP 2026 Main Conference paper *"Recognition–Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions"* by Yucheng Du and Xiyang Hu. It contains the matched-pair datasets, the analysis / extraction / steering pipeline, the figure-regeneration scripts, and the curated experiment artifacts needed to reproduce the shipped figure assets. The paper source (`main.tex`, `references.bib`, etc.) is maintained separately and is not redistributed in this code/data release. Public manuscript links (arXiv and ACL Anthology) are pending.

Authors: Yucheng Du (University of Southern California; [yuchengd@usc.edu](mailto:yuchengd@usc.edu)) and Xiyang Hu (Arizona State University). Correspondence: Xiyang Hu ([xiyanghu@asu.edu](mailto:xiyanghu@asu.edu)).

---

## TL;DR finding

Across an 11-model main grid spanning 1.7B–70B and two structural-impossibility domains (math, code):

1. **The model knows.** A one-dimensional null-space MeanDiff direction $d_{\mathrm{imp}}$ separates answerable (A) from unanswerable (U) prompts with mean AUC **0.939** across 22 cells.
2. **But it doesn't act.** $d_{\mathrm{imp}}$ is near-orthogonal to the canonical safety-refusal direction $d_{\mathrm{ref,safety}}$ (Arditi et al., 2024): mean $\cos$ **0.087**, range $[0.020, 0.130]$. It is only partially aligned with an in-domain behavior-defined invalidity-aware direction ($\cos \approx 0.40$).
3. **The misalignment largely predates RLHF.** Across 6 base/instruct pairs on math800, $\Delta\cos$ is $[-0.008, +0.110]$ (mean $+0.037$) — instruction tuning modulates the angle in a low-cosine regime rather than producing it.
4. **The direction is causal.** Generation-time activation steering on $d_{\mathrm{imp}}$ changes invalidity-aware behavior on the 4-anchor math/code intervention grid; gated $\Delta$G improves by **+33 to +52 pp**, and a 16-model steering sweep confirms the math/code footprint.

Public manuscript links (arXiv and ACL Anthology) are pending. This code/data release does not redistribute the paper source.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check the shipped aggregate artifacts and provenance metadata
python3 scripts/verify_core_conclusions.py

# 3. Regenerate the standalone TikZ teaser and the matplotlib figures
cd paper/figures && pdflatex -interaction=nonstopmode fig1_teaser.tex && cd ../..
python3 paper/generate_fig5_v2.py
python3 paper/generate_figures.py --figs fig2 fig3 fig4 fig6

# 4. The manuscript source is not redistributed here
#    Public manuscript links (arXiv and ACL Anthology) are pending.
```

A lightweight smoke test takes < 5 minutes.

For the heavier pipeline (extracting hidden states from a HuggingFace model, fitting probes, running steering), see `scripts/README.md` and the per-script CLI help. The intervention pipeline (`scripts/intervention_nullspace.py`) requires GPU access. The shipped v2 aggregates use LLM-assisted candidate labels under a fixed invalidity-aware rubric: nine cells in the 4-anchor grid apply provisional audit-subset fills, while the three Qwen3-8B cells use candidate-label passthrough. The supplementary Gemma-3-12B/code cell also applies provisional audit-subset fills. These provenance labels describe the effective inputs to the released aggregates and do not claim completed human adjudication.

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

Fig 1 is a standalone TikZ overview figure and does not require the raw activation-tensor tree. The raw 151 GB `experiments/signals/` tree is needed only to recompute representations from scratch rather than use the shipped aggregate artifacts.

`experiments/d_struct_behav_matrix.json` is a small derived aggregate for the behavior-defined invalidity-aware direction reported in the paper. Its local builder consumes the omitted raw `experiments/signals/*.npy` activations plus the effective LLM-assisted labels used by the released intervention aggregates. In its eight math/code cells, the six non-Qwen3-8B cells use candidate labels plus provisional audit-subset fills and the two Qwen3-8B cells use candidate-label passthrough. The JSON is therefore shipped as the reproducibility target for this comparison, rather than as a one-command-from-scratch regeneration step in the lightweight release.

`analysis/subspace_overlap/` ships the verified v2 scripts and frozen JSON results for the paper's 5--10-dimensional robustness check. The frozen JSONs are directly checkable in the lightweight release; rerunning the scripts requires the omitted raw signal tensors and cached directions, and the principal-angle script also loads the relevant model weights.

---

## Datasets

| File | Size | Source | License |
|---|---|---|---|
| `data/math800.jsonl` | 260 KB | Self-built frozen set (16 categories × 50 matched A/U pairs; LLM-drafted, deterministically checked) | this repo's LICENSE |
| `data/code800.jsonl` | 263 KB | Self-built frozen set (8 categories × 100 matched A/U pairs; LLM-drafted, deterministically checked) | this repo's LICENSE |
| `data/fact800.jsonl` | 1.35 MB | Derived from SQuAD 2.0 train split (seed=42, official `is_impossible` labels) | SQuAD 2.0 (CC-BY-SA-4.0) — see `data/LICENSE-SQUAD.md` |
| `data/falseqa.jsonl` | NOT SHIPPED | Reconstruct via `python src/data/fetch_falseqa.py && python src/data/clean_falseqa.py` (fetches from upstream `github.com/thunlp/FalseQA`, then applies cleanup). Upstream has no explicit LICENSE file at time of release; see `data/LICENSE-FALSEQA.md` for details. | upstream FalseQA — see `data/LICENSE-FALSEQA.md` |
| `data/abstentionbench_gsm8k.jsonl` | 841 KB | GSM8K subset of AbstentionBench; see `src/data/clean_abstentionbench_gsm8k.py` | upstream AbstentionBench — see `data/LICENSE-ABSTENTIONBENCH.md` |
| `data/difficulty_control_gsm8k.jsonl` | 194 KB | Difficulty-controlled split derived from GSM8K (Cobbe et al., 2021); see `scripts/prepare_difficulty_control.py` | upstream GSM8K — see `data/LICENSE-GSM8K.md` |

Each row is a JSON object with at least `{id, form, answerable, prompt}` where `answerable ∈ {A, U}`. The matched-pair structure (A vs. U sharing the same form/topic) is preserved across the five shipped JSONL datasets and the fetch-and-clean FalseQA dataset.

---

## Required model weights (download separately from HuggingFace Hub)

The detection/orthogonality main grid uses 11 instruct checkpoints; the base/instruct comparison adds six base counterparts, and the deterministic steering-breadth check uses 16 model keys. Set `HF_HOME` if you want HuggingFace downloads cached outside `~/.cache/huggingface`. The frozen JSON artifacts are the authoritative record of experiment identity; the corresponding registries are in `scripts/run_extract_minimal.py` and `scripts/impossibility_steering.py`.

The 11-model main grid is:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `microsoft/Phi-4-mini-instruct`, `google/gemma-3-4b-it`
- `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-14B`, `allenai/OLMo-2-1124-13B-Instruct`, `mistralai/Mistral-Small-24B-Instruct-2501`
- `Qwen/Qwen3-32B`, `meta-llama/Llama-3.3-70B-Instruct`

The 16-model v2det steering breadth uses the keys `gemma2`, `gemma3_4b`, `llama`, `mistral`, `mistral_small`, `mistral_small_3_2`, `olmo13b`, `phi3`, `phi4mini`, `qwen`, `qwen14b`, `qwen32b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, and `smollm2`. The six base/instruct pairs use Qwen2.5-7B/14B/32B base, Qwen3-8B/14B base, and the Llama-3.1-70B base proxy.

Per-model peak layers are pre-resolved and registered in the relevant scripts; see `scripts/find_best_layers_smollm2_gemma2_phi3.py` for how to re-resolve.

---

## Repository layout

```
.
├── README.md                                 (this file)
├── LICENSE                                   (MIT license for original code and self-created artifacts)
├── requirements.txt
├── paper/                                    (figure pipeline only — manuscript source is not included)
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
│   └── subspace_overlap/                     (verified v2 scripts + frozen multidimensional results)
├── docs/theory/                              (3 derivations cited by paper appendix)
├── data/                                     (5 shipped JSONL datasets, FalseQA fetch support, license stubs)
└── experiments/                              (curated subset; aggregate JSONs + the 4-anchor intervention factsheet)
    ├── main_grid_facts_v2.json
    ├── ablation_nullpc_results_11model.json
    ├── d_struct_behav_matrix.json           (behavior-defined invalidity-aware direction matrix)
    ├── direction_comparison_*.json           (12 files: 6 instruct + 6 base, for Fig 3 base/instruct panel)
    ├── intervention/
    │   ├── v2_final_4anchor_factsheet.md
    │   └── intervention_*_full_v2.json       (13 cells = 4 anchors × 3 datasets + 1 supplementary)
    ├── steering/
    │   ├── steering_*.json                   (source/legacy aggregates)
    │   └── v2det/                            (48 current v2det JSONs + comparison report)
    └── analysis/d_ref_energy_decomp/         (11-model energy-decomposition table)
```

The 151 GB `experiments/signals/` raw activation tree is intentionally **not** in this release. To recompute representations from scratch, re-run the extraction pipeline:

```bash
python scripts/run_extract_minimal.py --model <hf_id> --prompts data/math800.jsonl \
    --run-dir experiments/signals/math800_<key>_allL --all-layers --no-gradients
```

This populates the same `experiments/signals/` layout the analysis scripts read.

Some label-dependent aggregates, including `experiments/d_struct_behav_matrix.json`, additionally require the effective intervention-label TSVs used during aggregation. Those TSVs are not included in this lightweight release; the frozen aggregate JSONs used by the paper figures/tables are the released reproducibility targets. Their provenance is LLM-assisted candidate labeling with provisional audit-subset fills for the six-cell non-Qwen3-8B behavior subset and candidate-label passthrough for the two Qwen3-8B behavior cells.

---

## Citation

```bibtex
@inproceedings{du2026recognition,
  author = {Yucheng Du and Xiyang Hu},
  title  = {Recognition--Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions},
  year   = {2026},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  note   = {Main Conference}
}
```

---

## License and third-party data

The root [MIT License](LICENSE) applies to the authors' original code and self-created artifacts, including `math800` and `code800`. It does **not** relicense third-party datasets, model weights, or other upstream material. In particular:

- `fact800.jsonl` remains subject to SQuAD 2.0's CC BY-SA 4.0 terms (`data/LICENSE-SQUAD.md`).
- `abstentionbench_gsm8k.jsonl` remains subject to AbstentionBench's CC BY-NC 4.0 terms, layered over its GSM8K source (`data/LICENSE-ABSTENTIONBENCH.md`).
- `difficulty_control_gsm8k.jsonl` is derived from MIT-licensed GSM8K; retain its upstream attribution (`data/LICENSE-GSM8K.md`).
- FalseQA is fetch-only because its upstream repository did not state a license when this release was prepared; this repository does not redistribute `data/falseqa.jsonl` (`data/LICENSE-FALSEQA.md`).
- Model weights are downloaded separately and remain governed by their respective upstream terms.

---

## Reproducibility note

Current paper claims derive from the shipped frozen artifacts under `experiments/` and `analysis/subspace_overlap/`. Re-running the extraction + analysis pipeline end-to-end from raw model weights is supported but requires ~150 GB of disk for the signals tree and a GPU for the intervention / steering experiments.
