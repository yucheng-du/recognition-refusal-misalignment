"""generate_figures.py — EMNLP 2026 "Recognition ⊥ Refusal" main figures.

Six PDFs into ``paper/figures/``:

    fig1_conceptual.pdf          d_imp ⊥ d_ref schematic + dose-response teaser
    fig2_detection_heatmap.pdf   22 cells (11 models × 2 datasets) CosNSRT AUC
    fig3_orthogonality.pdf       22 instruct cells + 6 base/instruct pairs
    fig4_gsrs_ablation.pdf       P + w + φ three-factor decomposition (legacy)
    fig5_causal.pdf              3×3 gated flip-rate grid + dose-response
    fig6_nullspace_ablation.pdf  Null vs PC vs Full MeanDiff ablation, 22 cells

Style is unified through ``paper/_figure_style.py``.

Usage
-----
    python paper/generate_figures.py                  # all 6
    python paper/generate_figures.py --figs fig2 fig3 # subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from _figure_style import (
    HEATMAP_CMAP,
    SEMANTIC,
    SIZE_ANNOT,
    SIZE_AXIS,
    SIZE_LEGEND,
    SIZE_TICK,
    SIZE_TITLE,
    WIDTH_DOUBLE,
    WIDTH_SINGLE,
    apply as apply_style,
    errorbar_style,
    light_grid,
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script's location.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
INT = EXP / "intervention"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model display order + labels
# ---------------------------------------------------------------------------
# 11-model main grid (instruct-only) — ordered by approximate scale.
MAIN_GRID = [
    ("smollm2",        "SmolLM2-1.7B"),
    ("phi4mini",       "Phi-4-mini-3.8B"),
    ("gemma3_4b",      "Gemma-3-4B"),
    ("mistral",        "Mistral-7B"),
    ("qwen3_8b",       "Qwen3-8B"),
    ("llama",          "Llama-3.1-8B"),
    ("qwen3_14b",      "Qwen3-14B"),
    ("olmo13b",        "OLMo-2-13B"),
    ("mistral_small",  "Mistral-Small-24B"),
    ("qwen3_32b",      "Qwen3-32B"),
    ("llama70b",       "Llama-3.3-70B"),
]
MAIN_GRID_KEYS = [k for k, _ in MAIN_GRID]
MAIN_GRID_LABELS = {k: v for k, v in MAIN_GRID}

# 6-pair base/instruct comparison set used in §4.3 (Fig 3 bottom panel).
# Same order as ``main_grid_facts_v2.json["base_pairs"]["pairs"]``.
BASE_PAIRS = [
    ("qwen",      "Qwen2.5-7B"),
    ("qwen14b",   "Qwen2.5-14B"),
    ("qwen32b",   "Qwen2.5-32B"),
    ("qwen3_8b",  "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"),
    ("llama70b",  "Llama-3.x-70B"),  # vendor-confirmed post-training-only pair
]


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------
def _load_main_grid_facts() -> dict:
    with open(EXP / "main_grid_facts_v2.json") as f:
        return json.load(f)


def _load_ablation_11model() -> dict:
    with open(EXP / "ablation_nullpc_results_11model.json") as f:
        return json.load(f)


def _read_dir_cmp(filename: str) -> dict:
    path = EXP / filename
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 1 — Two-panel teaser: (a) geometric schematic with real scatter,
#                           (b) dose-response preview of causal control.
#
# Data unchanged: illustrative Qwen-14B / math800 / L34 schematic, identical
# to the legacy Fig 1.  Only style (palette, fonts, embedded TrueType,
# semantic encoding) is updated.
# ---------------------------------------------------------------------------
def _fig1_load_panel_a_data():
    from sklearn.decomposition import PCA as _PCA

    base = EXP / "signals" / "math800_qwen14b_allL" / "signals"
    reps = np.load(base / "reps_last_token_all_layers.npy")
    meta = [json.loads(line) for line in open(base / "meta.jsonl")]
    labels = np.array([m["answerable"] for m in meta])
    X = reps[:, 34, :].astype(np.float64)

    rng = np.random.default_rng(7)
    idx_A = np.where(labels == "A")[0]; rng.shuffle(idx_A)
    idx_U = np.where(labels == "U")[0]; rng.shuffle(idx_U)
    trA = idx_A[: len(idx_A) // 2]
    trU = idx_U[: len(idx_U) // 2]

    mu = X[trA].mean(0)
    V = _PCA(n_components=100, random_state=0).fit(X[trA] - mu).components_

    def _proj_null(Z):
        Zc = Z - mu
        return Zc - (Zc @ V.T) @ V

    Xn = _proj_null(X)
    d_imp = Xn[trU].mean(0) - Xn[trA].mean(0)
    d_imp /= np.linalg.norm(d_imp)

    R = Xn - np.outer(Xn @ d_imp, d_imp)
    perp = _PCA(n_components=1, random_state=0).fit(R).components_[0]
    perp -= (perp @ d_imp) * d_imp
    perp /= np.linalg.norm(perp)

    x2 = Xn @ d_imp
    y2 = Xn @ perp
    x2 = x2 - x2.mean()
    y2 = (y2 - y2.mean()) / y2.std()
    return x2, y2, labels


def _fig1_load_panel_b_data():
    path = INT / "intervention_qwen14b_math800_L34.json"
    d = json.load(open(path))
    rows = d["results"]
    out = {}
    for cond in ("A→U (inject signal)", "U→A (remove signal)"):
        rs = sorted(
            (r for r in rows if r["condition"] == cond),
            key=lambda r: r["alpha_mult"],
        )
        out[cond] = {
            "alpha":  [r["alpha_mult"] for r in rs],
            "signal": [r["rate_signal_gated"] * 100 for r in rs],
            "random": [r["rate_random_gated"] * 100 for r in rs],
        }
    return out


def fig1_conceptual():
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(WIDTH_DOUBLE, 2.9),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.28},
    )

    # ----------------------------- Panel (a): geometry -----------------------
    x2, y2, labels = _fig1_load_panel_a_data()
    rng = np.random.default_rng(13)
    idx_A = np.where(labels == "A")[0]
    idx_U = np.where(labels == "U")[0]
    rng.shuffle(idx_A); rng.shuffle(idx_U)
    nk = 60
    pA, pU = idx_A[:nk], idx_U[:nk]

    def _norm_clip(xs, ys):
        x = xs.copy(); y = ys.copy()
        xl, xh = np.percentile(x, 2), np.percentile(x, 98)
        yl, yh = np.percentile(y, 2), np.percentile(y, 98)
        x = np.clip(x, xl, xh); y = np.clip(y, yl, yh)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        return x, y

    xA_n, yA_n = _norm_clip(x2[pA], y2[pA])
    xU_n, yU_n = _norm_clip(x2[pU], y2[pU])

    xA_disp = -0.70 + 0.18 * xA_n
    yA_disp = -0.55 + 0.10 * yA_n
    xU_disp = 0.75 + 0.18 * xU_n
    yU_disp = -0.55 + 0.10 * yU_n

    cos_meas = 0.085
    angle = np.arccos(cos_meas)
    d_ref_vec = np.array([-np.cos(angle), np.sin(angle)])
    L_imp, L_ref = 1.20, 1.10

    axA.annotate(
        "", xy=(L_imp, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SEMANTIC["d_imp"], lw=2.0),
    )
    axA.annotate(
        "", xy=(L_ref * d_ref_vec[0], L_ref * d_ref_vec[1]), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SEMANTIC["d_ref"], lw=2.0),
    )
    axA.text(L_imp + 0.05, -0.02, r"$\mathbf{d}_{\mathrm{imp}}$",
             color=SEMANTIC["d_imp"], fontsize=11.5, ha="left", va="center",
             fontweight="bold")
    axA.text(L_ref * d_ref_vec[0] - 0.05, L_ref * d_ref_vec[1] + 0.04,
             r"$\mathbf{d}_{\mathrm{ref}}$", color=SEMANTIC["d_ref"],
             fontsize=11.5, ha="right", va="bottom", fontweight="bold")
    axA.text(0.14, 0.20, r"$\cos \approx 0.09$", fontsize=SIZE_AXIS,
             color="#333", ha="left", va="center")

    axA.scatter(xA_disp, yA_disp, s=14, c=SEMANTIC["A"], alpha=0.8,
                edgecolors="none", zorder=3)
    axA.scatter(xU_disp, yU_disp, s=18, c=SEMANTIC["U"], alpha=0.8,
                marker="^", edgecolors="none", zorder=3)

    axA.text(-0.70, -0.86, "A", color=SEMANTIC["A"], fontsize=10,
             ha="center", va="top", fontweight="bold")
    axA.text(0.75, -0.86, "U", color=SEMANTIC["U"], fontsize=10,
             ha="center", va="top", fontweight="bold")

    axA.plot([0.03, 0.03], [-0.85, -0.30], linestyle="--",
             color="#888", lw=0.8, alpha=0.6)

    axA.set_xlim(-1.30, 1.70)
    axA.set_ylim(-1.00, 1.25)
    axA.set_aspect("equal")
    axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_visible(False)
    axA.set_title(r"(a)  Recognition $\perp$ Refusal", fontsize=SIZE_TITLE,
                  pad=6)

    # ----------------------------- Panel (b): causal ------------------------
    data_b = _fig1_load_panel_b_data()
    au = data_b["A→U (inject signal)"]

    axB.plot(au["alpha"], au["signal"], marker="o", lw=1.8, ms=5.5,
             color=SEMANTIC["signal"],
             label=r"signal $\mathbf{d}_{\mathrm{imp}}$")
    axB.plot(au["alpha"], au["random"], marker="s", lw=1.3, ms=4.5,
             color=SEMANTIC["random"], linestyle="--",
             label="random direction")

    axB.set_xlim(-2, 42)
    axB.set_ylim(-3, 105)
    axB.set_xticks([0, 5, 10, 20, 40])
    axB.set_yticks([0, 25, 50, 75, 100])
    axB.set_xlabel(r"Steering strength  $\alpha$")
    axB.set_ylabel(r"A$\to$U gated flip rate (%)")
    light_grid(axB, axis="y")
    axB.set_title(
        r"(b)  Steering on $\mathbf{d}_{\mathrm{imp}}$ induces refusal  "
        r"(A$\to$U, same cell)",
        fontsize=SIZE_TITLE, pad=4,
    )
    axB.legend(loc="upper left", frameon=False, fontsize=SIZE_LEGEND,
               handletextpad=0.5, borderpad=0.2, labelspacing=0.25)

    out = FIG_DIR / "fig1_conceptual.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Fig 1 (compact, single-column) — vertical stack of the same two panels.
# Writes fig1_conceptual_compact.pdf (the variant referenced by the paper).
# ---------------------------------------------------------------------------
def fig1_conceptual_compact():
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(WIDTH_SINGLE, 4.5),
        gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.55},
    )

    # ----------------------------- Panel (a): geometry -----------------------
    x2, y2, labels = _fig1_load_panel_a_data()
    rng = np.random.default_rng(13)
    idx_A = np.where(labels == "A")[0]
    idx_U = np.where(labels == "U")[0]
    rng.shuffle(idx_A); rng.shuffle(idx_U)
    nk = 60
    pA, pU = idx_A[:nk], idx_U[:nk]

    def _norm_clip(xs, ys):
        x = xs.copy(); y = ys.copy()
        xl, xh = np.percentile(x, 2), np.percentile(x, 98)
        yl, yh = np.percentile(y, 2), np.percentile(y, 98)
        x = np.clip(x, xl, xh); y = np.clip(y, yl, yh)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        return x, y

    xA_n, yA_n = _norm_clip(x2[pA], y2[pA])
    xU_n, yU_n = _norm_clip(x2[pU], y2[pU])

    xA_disp = -0.70 + 0.18 * xA_n
    yA_disp = -0.55 + 0.10 * yA_n
    xU_disp = 0.75 + 0.18 * xU_n
    yU_disp = -0.55 + 0.10 * yU_n

    cos_meas = 0.085
    angle = np.arccos(cos_meas)
    d_ref_vec = np.array([-np.cos(angle), np.sin(angle)])
    L_imp, L_ref = 1.20, 1.10

    axA.annotate(
        "", xy=(L_imp, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SEMANTIC["d_imp"], lw=1.8),
    )
    axA.annotate(
        "", xy=(L_ref * d_ref_vec[0], L_ref * d_ref_vec[1]), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=SEMANTIC["d_ref"], lw=1.8),
    )
    axA.text(L_imp + 0.05, -0.02, r"$\mathbf{d}_{\mathrm{imp}}$",
             color=SEMANTIC["d_imp"], fontsize=10, ha="left", va="center",
             fontweight="bold")
    axA.text(L_ref * d_ref_vec[0] - 0.05, L_ref * d_ref_vec[1] + 0.04,
             r"$\mathbf{d}_{\mathrm{ref}}$", color=SEMANTIC["d_ref"],
             fontsize=10, ha="right", va="bottom", fontweight="bold")
    axA.text(0.14, 0.20, r"$\cos \approx 0.09$", fontsize=SIZE_TICK,
             color="#333", ha="left", va="center")

    axA.scatter(xA_disp, yA_disp, s=10, c=SEMANTIC["A"], alpha=0.85,
                edgecolors="none", zorder=3)
    axA.scatter(xU_disp, yU_disp, s=14, c=SEMANTIC["U"], alpha=0.85,
                marker="^", edgecolors="none", zorder=3)

    axA.text(-0.70, -0.86, "A", color=SEMANTIC["A"], fontsize=9,
             ha="center", va="top", fontweight="bold")
    axA.text(0.75, -0.86, "U", color=SEMANTIC["U"], fontsize=9,
             ha="center", va="top", fontweight="bold")

    axA.plot([0.03, 0.03], [-0.85, -0.30], linestyle="--",
             color="#888", lw=0.7, alpha=0.6)

    axA.set_xlim(-1.30, 1.70)
    axA.set_ylim(-1.00, 1.25)
    axA.set_aspect("equal")
    axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_visible(False)
    axA.set_title(r"(a) Recognition $\perp$ Refusal", fontsize=SIZE_TITLE,
                  pad=4)

    # ----------------------------- Panel (b): causal ------------------------
    data_b = _fig1_load_panel_b_data()
    au = data_b["A→U (inject signal)"]

    axB.plot(au["alpha"], au["signal"], marker="o", lw=1.6, ms=4.8,
             color=SEMANTIC["signal"],
             label=r"signal $\mathbf{d}_{\mathrm{imp}}$")
    axB.plot(au["alpha"], au["random"], marker="s", lw=1.2, ms=3.8,
             color=SEMANTIC["random"], linestyle="--",
             label="random direction")

    axB.set_xlim(-2, 42)
    axB.set_ylim(-3, 105)
    axB.set_xticks([0, 5, 10, 20, 40])
    axB.set_yticks([0, 25, 50, 75, 100])
    axB.set_xlabel(r"Steering strength  $\alpha$", fontsize=SIZE_AXIS)
    axB.set_ylabel(r"A$\to$U gated flip rate (\%)", fontsize=SIZE_AXIS)
    axB.tick_params(labelsize=SIZE_TICK)
    light_grid(axB, axis="y")
    axB.set_title(
        r"(b) Steering on $\mathbf{d}_{\mathrm{imp}}$ induces refusal",
        fontsize=SIZE_TITLE, pad=4,
    )
    axB.legend(loc="upper left", frameon=False, fontsize=SIZE_LEGEND,
               handletextpad=0.4, borderpad=0.2, labelspacing=0.2)

    out = FIG_DIR / "fig1_conceptual_compact.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Fig 2 — Detection heatmap: 11 models × 2 datasets CosNSRT AUC (22 cells)
# ---------------------------------------------------------------------------
def fig2_detection_heatmap():
    facts = _load_main_grid_facts()
    cells = facts["form"]["cells"]
    auc_by = {(c["model"], c["dataset"]): c["global_cosnsrt_auc"] for c in cells}

    # Hard-fail if any cell is missing.
    expected = [(m, d) for m in MAIN_GRID_KEYS for d in ("math800", "code800")]
    missing = [k for k in expected if k not in auc_by]
    if missing:
        raise RuntimeError(
            f"fig2: missing {len(missing)}/22 cells in form facts: {missing}"
        )

    datasets = ["math800", "code800"]
    dataset_labels = ["Math800", "Code800"]
    grid = np.array(
        [[auc_by[(m, d)] for d in datasets] for m in MAIN_GRID_KEYS]
    )

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 4.5))
    im = ax.imshow(grid, cmap=HEATMAP_CMAP, vmin=0.82, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(dataset_labels)
    ax.set_yticks(range(len(MAIN_GRID_KEYS)))
    ax.set_yticklabels([MAIN_GRID_LABELS[k] for k in MAIN_GRID_KEYS])

    for i in range(len(MAIN_GRID_KEYS)):
        for j in range(len(datasets)):
            v = grid[i, j]
            color = "white" if v < 0.93 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=SIZE_TICK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("CosNSRT AUC", fontsize=SIZE_TICK)
    cbar.ax.tick_params(labelsize=SIZE_LEGEND)

    avg = float(np.mean(grid))
    auc_min, auc_max = float(np.min(grid)), float(np.max(grid))
    ax.annotate(
        f"mean {avg:.3f}  ·  range [{auc_min:.3f}, {auc_max:.3f}]",
        xy=(0.5, -0.07), xycoords="axes fraction",
        ha="center", va="top", fontsize=SIZE_ANNOT, color="#333",
    )

    ax.set_title("Impossibility detection (22 cells)",
                 fontsize=SIZE_TITLE, pad=6)
    ax.set_xlabel(""); ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    out = FIG_DIR / "fig2_detection_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  ({len(MAIN_GRID_KEYS) * 2} cells)")


# ---------------------------------------------------------------------------
# Fig 3 — Orthogonality: 22 instruct cells + 6 base/instruct pairs.
# ---------------------------------------------------------------------------
INSTRUCT_DC_FILES = {
    "smollm2":       ("direction_comparison_smollm2.json",
                      "direction_comparison_smollm2_code800_L11.json"),
    "phi4mini":      ("direction_comparison_phi4mini.json",
                      "direction_comparison_phi4mini_code800_L29.json"),
    "gemma3_4b":     ("direction_comparison_gemma3_4b.json",
                      "direction_comparison_gemma3_4b_code800_L15.json"),
    "mistral":       ("direction_comparison_mistral.json",
                      "direction_comparison_mistral_code800_L15.json"),
    "qwen3_8b":      ("direction_comparison_qwen3_8b.json",
                      "direction_comparison_qwen3_8b_code800_L19.json"),
    "llama":         ("direction_comparison_llama.json",
                      "direction_comparison_llama_code800_L15.json"),
    "qwen3_14b":     ("direction_comparison_qwen3_14b.json",
                      "direction_comparison_qwen3_14b_code800_L24.json"),
    "olmo13b":       ("direction_comparison_olmo13b.json",
                      "direction_comparison_olmo13b_code800_L15.json"),
    "mistral_small": ("direction_comparison_mistral_small.json",
                      "direction_comparison_mistral_small_code800_L28.json"),
    "qwen3_32b":     ("direction_comparison_qwen3_32b.json",
                      "direction_comparison_qwen3_32b_code800_L47.json"),
    "llama70b":      ("direction_comparison_llama70b.json",
                      "direction_comparison_llama70b_code800_L72.json"),
}


def _load_orth_top_panel(facts: dict) -> list[dict]:
    """22-cell top panel: read CIs from main_grid_facts_v2.json."""
    cells = facts["orthogonality"]["cells"]
    out = []
    for m in MAIN_GRID_KEYS:
        for d in ("math800", "code800"):
            cell = next(
                (c for c in cells if c["model"] == m and c["dataset"] == d),
                None,
            )
            if cell is None:
                raise RuntimeError(f"fig3 top panel: missing cell {m}/{d}")
            out.append({
                "model": m, "dataset": d,
                "cos": cell["cos_matched_full"],
                "lo": cell["ci95_lo"], "hi": cell["ci95_hi"],
                "verified": cell["behavior_verified"],
            })
    if len(out) != 22:
        raise RuntimeError(f"fig3 top panel: got {len(out)} cells, expected 22")
    return out


def _load_orth_base_panel(facts: dict) -> list[dict]:
    """6-pair base/instruct panel — CIs read directly from per-model JSONs.

    ``main_grid_facts_v2.json["base_pairs"]`` lacks ``bootstrap_cos_ci95_*``
    fields, so we always re-read the underlying ``direction_comparison_*``
    files at the matched layer recorded in the aggregate.
    """
    aggregate = {p["instruct"]: p for p in facts["base_pairs"]["pairs"]}
    out = []
    for instruct_key, _ in BASE_PAIRS:
        if instruct_key not in aggregate:
            raise RuntimeError(f"fig3 base panel: pair {instruct_key} not in facts")
        agg = aggregate[instruct_key]
        instruct_layer = agg["instruct_layer"]
        base_layer = agg["base_layer"]

        # Instruct file (math800 matched layer).
        inst_file = f"direction_comparison_{instruct_key}.json"
        inst = _read_dir_cmp(inst_file)
        if inst.get("layer") != instruct_layer:
            raise RuntimeError(
                f"fig3 base panel: instruct file {inst_file} layer "
                f"{inst.get('layer')} ≠ aggregate {instruct_layer}"
            )

        # Base file: matched layer only — never neighboring sweep cells.
        base_file = f"direction_comparison_{instruct_key}_base_math800_L{base_layer}.json"
        base = _read_dir_cmp(base_file)

        out.append({
            "instruct": instruct_key,
            "instruct_cos": inst["cos_matched_full"],
            "instruct_lo": inst["bootstrap_cos_ci95_lo"],
            "instruct_hi": inst["bootstrap_cos_ci95_hi"],
            "instruct_verified": inst["behavior_verified"],
            "base_cos": base["cos_matched_full"],
            "base_lo": base["bootstrap_cos_ci95_lo"],
            "base_hi": base["bootstrap_cos_ci95_hi"],
            "base_verified": base["behavior_verified"],
            "delta_cos": agg["delta_cos"],
        })
    if len(out) != 6:
        raise RuntimeError(f"fig3 base panel: got {len(out)} pairs, expected 6")
    return out


def fig3_orthogonality():
    facts = _load_main_grid_facts()
    top_rows = _load_orth_top_panel(facts)
    pair_rows = _load_orth_base_panel(facts)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(WIDTH_DOUBLE, 5.2),
        gridspec_kw={"height_ratios": [11, 6], "hspace": 0.45},
    )

    # ------------------- Top panel: 22 instruct cells (math + code) ----------
    n_models = len(MAIN_GRID_KEYS)
    y = np.arange(n_models)
    bar_h = 0.38

    cos_math = np.array([
        next(r["cos"] for r in top_rows if r["model"] == m and r["dataset"] == "math800")
        for m in MAIN_GRID_KEYS
    ])
    lo_math = np.array([
        cos_math[i] - next(r["lo"] for r in top_rows if r["model"] == MAIN_GRID_KEYS[i] and r["dataset"] == "math800")
        for i in range(n_models)
    ])
    hi_math = np.array([
        next(r["hi"] for r in top_rows if r["model"] == MAIN_GRID_KEYS[i] and r["dataset"] == "math800") - cos_math[i]
        for i in range(n_models)
    ])
    cos_code = np.array([
        next(r["cos"] for r in top_rows if r["model"] == m and r["dataset"] == "code800")
        for m in MAIN_GRID_KEYS
    ])
    lo_code = np.array([
        cos_code[i] - next(r["lo"] for r in top_rows if r["model"] == MAIN_GRID_KEYS[i] and r["dataset"] == "code800")
        for i in range(n_models)
    ])
    hi_code = np.array([
        next(r["hi"] for r in top_rows if r["model"] == MAIN_GRID_KEYS[i] and r["dataset"] == "code800") - cos_code[i]
        for i in range(n_models)
    ])

    ax_top.barh(y + bar_h / 2, cos_math, bar_h, color=SEMANTIC["math"],
                edgecolor="none", label="Math800")
    ax_top.errorbar(cos_math, y + bar_h / 2,
                    xerr=[lo_math, hi_math], **errorbar_style())
    ax_top.barh(y - bar_h / 2, cos_code, bar_h, color=SEMANTIC["code"],
                edgecolor="none", label="Code800")
    ax_top.errorbar(cos_code, y - bar_h / 2,
                    xerr=[lo_code, hi_code], **errorbar_style())

    # Asterisks on Llama-3.1-8B (proxy d_ref, both math800 and code800).
    for i, m in enumerate(MAIN_GRID_KEYS):
        for sign, dataset, cos_arr, hi_arr in [
            (+1, "math800", cos_math, hi_math),
            (-1, "code800", cos_code, hi_code),
        ]:
            verified = next(
                r["verified"] for r in top_rows
                if r["model"] == m and r["dataset"] == dataset
            )
            if not verified:
                ax_top.text(
                    cos_arr[i] + max(hi_arr[i], 0.002) + 0.004,
                    y[i] + sign * bar_h / 2,
                    "*", fontsize=10, color="#333", va="center",
                )

    ax_top.set_yticks(y)
    ax_top.set_yticklabels([MAIN_GRID_LABELS[m] for m in MAIN_GRID_KEYS])
    ax_top.invert_yaxis()
    ax_top.set_xlabel(r"$\cos(\mathbf{d}_{\mathrm{imp}},\,\mathbf{d}_{\mathrm{ref}})$  (22 instruct cells)")
    ax_top.set_xlim(0, 0.16)
    light_grid(ax_top, axis="x")

    cos_all = np.concatenate([cos_math, cos_code])
    ax_top.set_title(
        rf"(a) Recognition $\perp$ Refusal — 22 main-grid cells "
        rf"(mean cos {cos_all.mean():.3f}, range [{cos_all.min():.3f}, {cos_all.max():.3f}])",
        fontsize=SIZE_TITLE, pad=4, loc="left",
    )

    handles_top = [
        mpatches.Patch(color=SEMANTIC["math"], label="Math800"),
        mpatches.Patch(color=SEMANTIC["code"], label="Code800"),
        plt.Line2D([0], [0], marker="*", linestyle="none", color="#333",
                   label="proxy $\\mathbf{d}_{\\mathrm{ref}}$"),
    ]
    ax_top.legend(handles=handles_top, frameon=False, loc="lower right",
                  fontsize=SIZE_LEGEND, handletextpad=0.5, borderpad=0.3)

    # ------------------- Bottom panel: 6 base/instruct pairs (math800) -------
    n_pairs = len(BASE_PAIRS)
    yp = np.arange(n_pairs)
    bar_h_p = 0.38

    inst_cos = np.array([r["instruct_cos"] for r in pair_rows])
    inst_lo = np.array([r["instruct_cos"] - r["instruct_lo"] for r in pair_rows])
    inst_hi = np.array([r["instruct_hi"] - r["instruct_cos"] for r in pair_rows])
    base_cos = np.array([r["base_cos"] for r in pair_rows])
    base_lo = np.array([r["base_cos"] - r["base_lo"] for r in pair_rows])
    base_hi = np.array([r["base_hi"] - r["base_cos"] for r in pair_rows])

    ax_bot.barh(yp + bar_h_p / 2, inst_cos, bar_h_p,
                color=SEMANTIC["instruct"], edgecolor="none",
                label="Instruct")
    ax_bot.errorbar(inst_cos, yp + bar_h_p / 2,
                    xerr=[inst_lo, inst_hi], **errorbar_style())
    ax_bot.barh(yp - bar_h_p / 2, base_cos, bar_h_p,
                color=SEMANTIC["base"], edgecolor="none", label="Base")
    ax_bot.errorbar(base_cos, yp - bar_h_p / 2,
                    xerr=[base_lo, base_hi], **errorbar_style())

    # Asterisk: Llama-3.1-70B-Base (proxy base d_ref, last row).
    for i, r in enumerate(pair_rows):
        if not r["base_verified"]:
            ax_bot.text(
                r["base_cos"] + max(base_hi[i], 0.002) + 0.004,
                yp[i] - bar_h_p / 2,
                "*", fontsize=10, color="#333", va="center",
            )

    ax_bot.set_yticks(yp)
    ax_bot.set_yticklabels([lbl for _, lbl in BASE_PAIRS])
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel(r"$\cos(\mathbf{d}_{\mathrm{imp}},\,\mathbf{d}_{\mathrm{ref}})$  on Math800")
    ax_bot.set_xlim(min(0, base_cos.min() - 0.03), 0.16)
    ax_bot.axvline(0.0, color="#888", lw=0.6)
    light_grid(ax_bot, axis="x")

    deltas = np.array([r["delta_cos"] for r in pair_rows])
    ax_bot.set_title(
        rf"(b) Pretraining origin — 6 base/instruct pairs "
        rf"($\Delta\cos$ range [{deltas.min():+.3f}, {deltas.max():+.3f}], mean {deltas.mean():+.3f})",
        fontsize=SIZE_TITLE, pad=4, loc="left",
    )

    handles_bot = [
        mpatches.Patch(color=SEMANTIC["instruct"], label="Instruct"),
        mpatches.Patch(color=SEMANTIC["base"], label="Base"),
        plt.Line2D([0], [0], marker="*", linestyle="none", color="#333",
                   label="proxy base $\\mathbf{d}_{\\mathrm{ref}}$"),
    ]
    ax_bot.legend(handles=handles_bot, frameon=False, loc="lower right",
                  fontsize=SIZE_LEGEND, handletextpad=0.5, borderpad=0.3)

    out = FIG_DIR / "fig3_orthogonality.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  (top: 22 cells; base: 6 pairs)")


# ---------------------------------------------------------------------------
# Fig 4 — GSRS three-factor ablation.
#
# Numbers are FROZEN to the legacy 16-config GSRS ablation per factsheet B3
# ("not re-aggregated on the 11-model grid"). Constants below mirror the
# values cited in §4.1 / Fig 4 caption and are unchanged.
# ---------------------------------------------------------------------------
def fig4_gsrs_ablation():
    factors = ["P\n(Projection)", "w\n(Direction)", r"$\varphi$" + "\n(Scoring)"]
    avg_gain = np.array([22.5, 10.9, 1.1])
    hard_gain = np.array([np.nan, np.nan, 5.7])

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.7))
    x = np.arange(len(factors))
    bar_w = 0.55

    bar_colors = [SEMANTIC["A"], SEMANTIC["d_ref"], SEMANTIC["d_imp"]]
    ax.bar(x, avg_gain, bar_w, color=bar_colors, edgecolor="none",
           label="Avg (16-config legacy ablation)")

    ax.scatter([x[2]], [hard_gain[2]], marker="D", s=35, facecolor="white",
               edgecolor=SEMANTIC["d_imp"], linewidths=1.2, zorder=5,
               label=r"$\varphi$ on hard datasets")

    for i, v in enumerate(avg_gain):
        ax.text(x[i], v + 0.6, f"+{v:.1f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.text(x[2] + 0.03, hard_gain[2] + 0.6, f"+{hard_gain[2]:.1f}",
            ha="left", va="bottom", fontsize=SIZE_TICK, color=SEMANTIC["d_imp"])

    ax.set_ylabel("AUC gain over baseline (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylim(0, 27)
    ax.legend(loc="upper right", frameon=False, fontsize=SIZE_LEGEND,
              handletextpad=0.4, borderpad=0.3)
    ax.set_title("GSRS three-factor decomposition", fontsize=SIZE_TITLE, pad=4)
    light_grid(ax, axis="y")

    out = FIG_DIR / "fig4_gsrs_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Fig 5 — 3×3 gated flip-rate grid + dose-response.
#
# Data unchanged: Mistral-7B / Qwen-7B / Qwen-14B × {math, code, fact} from
# experiments/intervention/*.json (per factsheet B0 — the 3×3 grid is
# intentionally not extended to the 11-model main grid).
# ---------------------------------------------------------------------------
INT_FILES = {
    ("mistral", "math800"):  "intervention_mistral_math800_L15.json",
    ("mistral", "code800"):  "intervention_mistral_code800_L15.json",
    ("mistral", "fact800"):  "intervention_mistral_fact800_L17.json",
    ("qwen",    "math800"):  "intervention_qwen_math800_L18.json",
    ("qwen",    "code800"):  "intervention_qwen_code800_L18.json",
    ("qwen",    "fact800"):  "intervention_qwen_fact800_L19.json",
    ("qwen14b", "math800"):  "intervention_qwen14b_math800_L34.json",
    ("qwen14b", "code800"):  "intervention_qwen14b_code800_L32.json",
    ("qwen14b", "fact800"):  "intervention_qwen14b_fact800_L34.json",
}
DATASET_LBL = {"math800": "Math800", "code800": "Code800", "fact800": "Fact800"}
SHORT_MODEL_LBL = {
    "mistral": "Mistral\n-7B",
    "qwen":    "Qwen\n-7B",
    "qwen14b": "Qwen\n-14B",
}


def _best_gated(cond_rows):
    best = None
    for r in cond_rows:
        if r["alpha_mult"] == 0.0:
            continue
        delta = r["rate_signal_gated"] - r["rate_random_gated"]
        if best is None or delta > best["delta"]:
            best = {
                "alpha_mult": r["alpha_mult"],
                "signal": r["rate_signal_gated"],
                "random": r["rate_random_gated"],
                "delta": delta,
            }
    return best


def fig5_causal():
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    datasets = ["math800", "code800", "fact800"]
    models = ["mistral", "qwen", "qwen14b"]

    n_groups = len(datasets) * len(models)
    short_labels = []
    sig_remove, rnd_remove = [], []
    sig_inject, rnd_inject = [], []

    for d in datasets:
        for m in models:
            with open(INT / INT_FILES[(m, d)]) as f:
                data = json.load(f)
            rm = [r for r in data["results"] if r["condition"].startswith("U→A")]
            inj = [r for r in data["results"] if r["condition"].startswith("A→U")]
            b_rm = _best_gated(rm)
            b_inj = _best_gated(inj)
            short_labels.append(SHORT_MODEL_LBL[m])
            sig_remove.append(b_rm["signal"] * 100)
            rnd_remove.append(b_rm["random"] * 100)
            sig_inject.append(b_inj["signal"] * 100)
            rnd_inject.append(b_inj["random"] * 100)

    x = np.arange(n_groups)
    w = 0.2

    # U→A (signal removal): blue (signal control)
    ax_a.bar(x - 1.5 * w, sig_remove, w, color=SEMANTIC["signal"],
             label="U→A signal")
    ax_a.bar(x - 0.5 * w, rnd_remove, w, color=SEMANTIC["signal"],
             alpha=0.35, hatch="///", edgecolor="white", lw=0,
             label="U→A random")
    # A→U (signal inject): orange-red (the U-class color)
    ax_a.bar(x + 0.5 * w, sig_inject, w, color=SEMANTIC["U"],
             label="A→U signal")
    ax_a.bar(x + 1.5 * w, rnd_inject, w, color=SEMANTIC["U"],
             alpha=0.35, hatch="///", edgecolor="white", lw=0,
             label="A→U random")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_labels, fontsize=SIZE_LEGEND)

    for i in [2.5, 5.5]:
        ax_a.axvline(i, color="#BBB", lw=0.6, ls="--")
    for k, d in enumerate(datasets):
        cx = x[k * 3 + 1]
        ax_a.text(cx, 1.02, DATASET_LBL[d],
                  ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color="#333",
                  transform=ax_a.get_xaxis_transform())

    ax_a.set_ylabel("Gated flip rate (%)")
    ax_a.set_ylim(0, 100)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38),
                frameon=False, fontsize=SIZE_LEGEND, ncol=4,
                handletextpad=0.4, borderpad=0.3, columnspacing=1.2)
    ax_a.set_title("(a) Gated flip rate at best α",
                   fontsize=SIZE_TITLE, loc="left", pad=14)
    light_grid(ax_a, axis="y")

    # --- (b) Dose-response for Qwen14B / math800 A→U ---
    with open(INT / INT_FILES[("qwen14b", "math800")]) as f:
        data = json.load(f)
    inj_rows = [r for r in data["results"] if r["condition"].startswith("A→U")]
    inj_rows = sorted(inj_rows, key=lambda r: r["alpha_mult"])
    alphas = [r["alpha_mult"] for r in inj_rows]
    sig = [r["rate_signal_gated"] * 100 for r in inj_rows]
    rnd = [r["rate_random_gated"] * 100 for r in inj_rows]

    ax_b.plot(alphas, sig, "-o", color=SEMANTIC["signal"], lw=1.6, ms=5,
              label=r"signal $\hat{\mathbf{d}}_{\mathrm{imp}}$")
    ax_b.plot(alphas, rnd, "--s", color=SEMANTIC["random"], lw=1.2, ms=4,
              label="random direction")

    for a, s in zip(alphas, sig):
        ax_b.annotate(f"{s:.0f}%", (a, s), textcoords="offset points",
                      xytext=(4, -2), fontsize=SIZE_LEGEND,
                      color=SEMANTIC["signal"])

    ax_b.set_xlabel(r"Steering magnitude $\alpha / \mathrm{proj\_std}$")
    ax_b.set_ylabel("A→U flip rate (%)")
    ax_b.set_ylim(-5, 115)
    ax_b.set_xticks(alphas)
    ax_b.legend(loc="upper left", frameon=False, fontsize=SIZE_LEGEND,
                handletextpad=0.4)
    ax_b.set_title("(b) Qwen-14B / Math800 A→U dose–response",
                   fontsize=SIZE_TITLE, loc="left", pad=14)
    light_grid(ax_b, axis="both")

    out = FIG_DIR / "fig5_causal.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Fig 6 — Null vs PC vs Full ablation, 22 cells.
# ---------------------------------------------------------------------------
def fig6_nullspace_ablation():
    data = _load_ablation_11model()
    configs = data["per_config"]

    ordered = []
    for ds in ("math800", "code800"):
        for m in MAIN_GRID_KEYS:
            row = next(
                (c for c in configs if c["model"] == m and c["dataset"] == ds),
                None,
            )
            if row is None:
                raise RuntimeError(f"fig6: missing config {m}/{ds}")
            ordered.append(row)
    if len(ordered) != 22:
        raise RuntimeError(f"fig6: got {len(ordered)} configs, expected 22")

    n = len(ordered)
    x = np.arange(n)
    w = 0.27
    null_v = np.array([r["null_md"] for r in ordered])
    pc_v   = np.array([r["pc_md"]   for r in ordered])
    full_v = np.array([r["full_md"] for r in ordered])

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.4))
    ax.bar(x - w, null_v, w, color=SEMANTIC["null"], label="Null-space")
    ax.bar(x,     pc_v,   w, color=SEMANTIC["pc"],   label="Top-k PC")
    ax.bar(x + w, full_v, w, color=SEMANTIC["full"], label="Full-space")

    avg = data["averages"]["meandiff"]
    ax.axhline(avg["null"], color=SEMANTIC["null"], ls=":", lw=0.8, alpha=0.85)
    ax.axhline(avg["pc"],   color=SEMANTIC["pc"],   ls=":", lw=0.8, alpha=0.85)
    ax.axhline(avg["full"], color=SEMANTIC["full"], ls=":", lw=0.8, alpha=0.85)

    n_models = len(MAIN_GRID_KEYS)
    ax.axvline(n_models - 0.5, color="#AAA", lw=0.7, ls="--")
    ax.text((n_models - 1) / 2, 1.04, "Math800",
            ha="center", fontsize=SIZE_AXIS, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(n_models + (n_models - 1) / 2, 1.04, "Code800",
            ha="center", fontsize=SIZE_AXIS, fontweight="bold",
            transform=ax.get_xaxis_transform())

    labels = [MAIN_GRID_LABELS[m] for m in MAIN_GRID_KEYS] * 2
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=SIZE_LEGEND)
    ax.set_ylabel("MeanDiff AUC")
    ax.set_ylim(0.55, 1.05)

    handles = [
        mpatches.Patch(color=SEMANTIC["null"], label=f"Null-space  (avg {avg['null']:.3f})"),
        mpatches.Patch(color=SEMANTIC["pc"],   label=f"Top-k PC  (avg {avg['pc']:.3f})"),
        mpatches.Patch(color=SEMANTIC["full"], label=f"Full-space  (avg {avg['full']:.3f})"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=False,
              fontsize=SIZE_LEGEND, ncol=3, handletextpad=0.5,
              borderpad=0.3, columnspacing=1.4)
    light_grid(ax, axis="y")

    counts = data["counts"]
    ax.text(
        0.99, 0.04,
        f"Null > PC: {counts['md_null_gt_pc']}/{counts['n_configs']}  ·  "
        f"Null > Full: {counts['md_null_gt_full']}/{counts['n_configs']}",
        transform=ax.transAxes, fontsize=SIZE_LEGEND, color="#333",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F5F5F5",
                  ec="#CCC", lw=0.5),
    )

    ax.set_title("Projection ablation (MeanDiff AUC, 5-seed avg, 22 cells)",
                 fontsize=SIZE_TITLE, pad=14)

    out = FIG_DIR / "fig6_nullspace_ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  ({len(ordered)} cells)")


# ---------------------------------------------------------------------------
# Main + CLI
# ---------------------------------------------------------------------------
ALL_FIGS = {
    "fig1": fig1_conceptual,
    "fig1compact": fig1_conceptual_compact,
    "fig2": fig2_detection_heatmap,
    "fig3": fig3_orthogonality,
    "fig4": fig4_gsrs_ablation,
    "fig5": fig5_causal,
    "fig6": fig6_nullspace_ablation,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figs", nargs="+", default=None,
        choices=sorted(ALL_FIGS.keys()),
        help="Subset of figures to regenerate. Default: all six.",
    )
    args = parser.parse_args(argv)
    apply_style()

    todo = args.figs or list(ALL_FIGS.keys())
    print("Generating figures into", FIG_DIR)
    # Fig 1 reads raw activation tensors under experiments/signals/, which are
    # not included in the anonymous-review release (~150 GB). The shipped
    # fig1_conceptual_compact.pdf is the static figure embedded in the paper PDF.
    SIGNALS_DEPENDENT = {"fig1", "fig1compact"}
    SIGNALS_PATH = EXP / "signals"
    for name in todo:
        print(f"[{name}]")
        if name in SIGNALS_DEPENDENT and not SIGNALS_PATH.exists():
            print(
                f"[skip] {name}: requires the full experiments/signals/ tree "
                "(raw activation .npy files, ~150 GB) which is not in this "
                "release. The shipped paper/figures/fig1_conceptual_compact.pdf "
                "is the static figure embedded in the paper PDF (see OpenReview "
                "submission). See README 'Figure reproducibility' section "
                "for the per-figure regen matrix."
            )
            continue
        ALL_FIGS[name]()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
