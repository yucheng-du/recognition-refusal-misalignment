#!/usr/bin/env python3
"""Generate fig5_causal_v2 — the v2 4-anchor causal-control figure.

Outputs (written to paper/figures/ alongside the other shipped PDFs):
  figures/fig5_causal_v2.pdf          full-width, two-column-spanning
  figures/fig5_causal_v2_compact.pdf  single-column variant embedded in the paper PDF

Driven by per-cell v2 invalidity-aware intervention JSONs under
experiments/intervention/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent      # paper/
PAPER_DIR = HERE                            # paper/
REPO_ROOT = HERE.parent                     # repo root
FIG_DIR = PAPER_DIR / "figures"

# Reuse the shared style helpers from paper/_figure_style.py.
sys.path.insert(0, str(PAPER_DIR))
from _figure_style import (  # type: ignore  # noqa: E402
    GRAY_DARK,
    GRAY_LIGHT,
    GRAY_MID,
    SEMANTIC,
    SIZE_ANNOT,
    SIZE_LEGEND,
    SIZE_TICK,
    SIZE_TITLE,
    WIDTH_DOUBLE,
    apply,
    hide_top_right,
    light_grid,
)

apply()


# ---------------------------------------------------------------------------
# Authoritative input map (do not glob — extra files exist that are NOT
# part of the v2 4-anchor grid).
# ---------------------------------------------------------------------------
INT_DIR = REPO_ROOT / "experiments" / "intervention"

CELL_FILES = {
    ("mistral",   "math800"): "intervention_mistral_math800_L15_full_v2.json",
    ("mistral",   "code800"): "intervention_mistral_code800_L15_full_v2.json",
    ("mistral",   "fact800"): "intervention_mistral_fact800_L17_full_v2.json",
    ("gemma3_4b", "math800"): "intervention_gemma3_4b_math800_L16_full_v2.json",
    ("gemma3_4b", "code800"): "intervention_gemma3_4b_code800_L15_full_v2.json",
    ("gemma3_4b", "fact800"): "intervention_gemma3_4b_fact800_L16_full_v2.json",
    ("qwen3_14b", "math800"): "intervention_qwen3_14b_math800_L25_full_v2.json",
    ("qwen3_14b", "code800"): "intervention_qwen3_14b_code800_L24_full_v2.json",
    ("qwen3_14b", "fact800"): "intervention_qwen3_14b_fact800_L25_full_v2.json",
    ("qwen3_8b",  "math800"): "intervention_qwen3_8b_math800_L21_full_v2.json",
    ("qwen3_8b",  "code800"): "intervention_qwen3_8b_code800_L19_full_v2.json",
    ("qwen3_8b",  "fact800"): "intervention_qwen3_8b_fact800_L21_full_v2.json",
}

# Row order in the heatmap (top → bottom).
ANCHORS = [
    ("mistral",   "Mistral-7B-Instruct"),
    ("gemma3_4b", "Gemma-3-4B-it"),
    ("qwen3_14b", "Qwen3-14B"),
    ("qwen3_8b",  "Qwen3-8B"),
]
DATASETS   = ["math800", "code800", "fact800"]
DIRECTIONS = ["A→U", "U→A"]

# Mistral math/code keystone cells get the 1pt outline.
KEYSTONE_CELLS = {
    ("mistral", "math800", "A→U"),
    ("mistral", "math800", "U→A"),
    ("mistral", "code800", "A→U"),
    ("mistral", "code800", "U→A"),
}

# v2 factsheet expected ΔG (pp). Used only for ±2pp sanity check.
EXPECTED_DG = {
    ("mistral",   "math800", "A→U"): +33, ("mistral",   "math800", "U→A"): +38,
    ("mistral",   "code800", "A→U"): +35, ("mistral",   "code800", "U→A"): +44,
    ("mistral",   "fact800", "A→U"):  +4, ("mistral",   "fact800", "U→A"): None,
    ("gemma3_4b", "math800", "A→U"): +17, ("gemma3_4b", "math800", "U→A"): +48,
    ("gemma3_4b", "code800", "A→U"): +35, ("gemma3_4b", "code800", "U→A"): +40,
    ("gemma3_4b", "fact800", "A→U"): +24, ("gemma3_4b", "fact800", "U→A"): None,
    ("qwen3_14b", "math800", "A→U"):  +7, ("qwen3_14b", "math800", "U→A"): +21,
    ("qwen3_14b", "code800", "A→U"): +37, ("qwen3_14b", "code800", "U→A"): +52,
    ("qwen3_14b", "fact800", "A→U"): +20, ("qwen3_14b", "fact800", "U→A"): None,
    ("qwen3_8b",  "math800", "A→U"): +20, ("qwen3_8b",  "math800", "U→A"): +42,
    ("qwen3_8b",  "code800", "A→U"): +22, ("qwen3_8b",  "code800", "U→A"): +24,
    ("qwen3_8b",  "fact800", "A→U"): +10, ("qwen3_8b",  "fact800", "U→A"): None,
}

ANCHOR_THRESHOLD_PP = 30.0           # ≥ +30pp → anchor-quality
MIN_GATEN           = 5              # gateN > 4 → measurable
ALPHA_BUDGET        = {5, 10, 20, 40}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _alpha_in_budget(a) -> bool:
    try:
        return int(round(float(a))) in ALPHA_BUDGET
    except (TypeError, ValueError):
        return False


def _alpha_to_int(a) -> int:
    return int(round(float(a)))


def load_cell_rows(anchor: str, dataset: str, direction: str):
    """Return α-rows for (anchor, dataset, condition starting with `direction`)."""
    fname = CELL_FILES.get((anchor, dataset))
    if fname is None:
        raise RuntimeError(f"cell ({anchor},{dataset}) not in CELL_FILES")
    path = INT_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"missing v2 file: {path}")
    data = json.loads(path.read_text())
    rows = [
        r for r in data["results"]
        if r["condition"].startswith(direction) and _alpha_in_budget(r["alpha_mult"])
    ]
    if len(rows) != 4:
        raise RuntimeError(
            f"expected 4 α-rows for ({anchor},{dataset},{direction}); got {len(rows)}"
        )
    for r in rows:
        ia = r["criterion_invalidity_aware"]
        for branch in ("signal", "random"):
            for key in ("n_clean_gate", "rate_gated"):
                if key not in ia[branch]:
                    raise RuntimeError(
                        f"schema: missing criterion_invalidity_aware.{branch}.{key} in "
                        f"{fname} (α={r['alpha_mult']})"
                    )
        if "delta_gated" not in ia:
            raise RuntimeError(
                f"schema: missing criterion_invalidity_aware.delta_gated in {fname}"
            )
    return rows


def best_alpha_row(rows):
    """Pick the row with largest signed delta_gated subject to gateN>=5 in both
    signal AND random branches. Returns None if no row qualifies."""
    valid = [
        r for r in rows
        if r["criterion_invalidity_aware"]["signal"]["n_clean_gate"] >= MIN_GATEN
        and r["criterion_invalidity_aware"]["random"]["n_clean_gate"] >= MIN_GATEN
        and r["criterion_invalidity_aware"]["delta_gated"] is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda r: r["criterion_invalidity_aware"]["delta_gated"])


# ---------------------------------------------------------------------------
# Panel (a): 6-column × 4-row heatmap
# ---------------------------------------------------------------------------
HEATMAP_CMAP = plt.cm.RdBu_r
HEATMAP_NORM = Normalize(vmin=-55.0, vmax=+55.0)
HATCH_COLOR  = "#888888"


def _cell_color(dg_pp: float, gateN: int) -> tuple:
    """Color a cell by its signed delta_gated (pp)."""
    if gateN < MIN_GATEN:
        return (0.93, 0.93, 0.93)          # gray — anecdotal
    return HEATMAP_CMAP(HEATMAP_NORM(dg_pp))


def _scan_cells():
    """Compute every cell's render state. Returns a dict keyed by
    (anchor_key, dataset, direction) → dict."""
    out = {}
    for anchor_key, _ in ANCHORS:
        for dataset in DATASETS:
            for direction in DIRECTIONS:
                rows = load_cell_rows(anchor_key, dataset, direction)
                gateNs_sig = [r["criterion_invalidity_aware"]["signal"]["n_clean_gate"]
                              for r in rows]
                best = best_alpha_row(rows)
                if best is None:
                    out[(anchor_key, dataset, direction)] = dict(
                        anecdotal=True,
                        gateN=max(gateNs_sig),
                        best_alpha=None,
                        delta_pp=None,
                        signal_pp=None,
                        random_pp=None,
                    )
                    continue
                ia = best["criterion_invalidity_aware"]
                out[(anchor_key, dataset, direction)] = dict(
                    anecdotal=False,
                    gateN=ia["signal"]["n_clean_gate"],
                    best_alpha=_alpha_to_int(best["alpha_mult"]),
                    delta_pp=ia["delta_gated"] * 100.0,
                    signal_pp=ia["signal"]["rate_gated"] * 100.0,
                    random_pp=ia["random"]["rate_gated"] * 100.0,
                )
    return out


def _validate_against_factsheet(cells):
    """Hard-error if any non-anecdotal cell deviates by >2pp from the
    factsheet table, OR if anecdotal status appears outside the four
    fact U→A cells."""
    errs = []
    expected_anecdotal = {
        (a, "fact800", "U→A") for a, _ in ANCHORS
    }
    for key, info in cells.items():
        anchor_key, dataset, direction = key
        exp = EXPECTED_DG[(anchor_key, dataset, direction)]
        if info["anecdotal"]:
            if key not in expected_anecdotal:
                errs.append(f"UNEXPECTED anecdotal at {key} (gateN={info['gateN']})")
            elif exp is not None:
                errs.append(f"factsheet says {key} expected={exp} but cell is anecdotal")
            continue
        if exp is None:
            errs.append(f"{key}: factsheet expects N/A but cell rendered {info['delta_pp']:+.1f}pp")
            continue
        diff = abs(info["delta_pp"] - exp)
        if diff > 2.0:
            errs.append(f"{key}: rendered {info['delta_pp']:+.1f}pp vs factsheet {exp:+d}pp (Δ={diff:.1f}pp > 2)")
    if errs:
        raise RuntimeError("factsheet validation failed:\n  " + "\n  ".join(errs))


def draw_panel_a(ax, cells, *, font_label=SIZE_TICK, font_value=SIZE_LEGEND,
                 font_alpha=6.5, compact=False):
    n_rows = len(ANCHORS)
    n_cols = len(DATASETS) * len(DIRECTIONS)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # top row = first anchor

    # column header positions (column index → (dataset, direction))
    col_specs = []
    for ds in DATASETS:
        for dr in DIRECTIONS:
            col_specs.append((ds, dr))

    # draw cells
    for ri, (anchor_key, _) in enumerate(ANCHORS):
        for ci, (ds, dr) in enumerate(col_specs):
            info = cells[(anchor_key, ds, dr)]
            x0, y0 = ci, ri
            if info["anecdotal"]:
                # gray hatched
                rect = Rectangle((x0, y0), 1, 1,
                                 facecolor="#EEEEEE",
                                 edgecolor=HATCH_COLOR, linewidth=0.6,
                                 hatch="///")
                ax.add_patch(rect)
                ax.text(x0 + 0.5, y0 + 0.5, "N/A",
                        ha="center", va="center",
                        fontsize=font_value, color=GRAY_DARK,
                        fontweight="normal")
            else:
                color = _cell_color(info["delta_pp"], info["gateN"])
                rect = Rectangle((x0, y0), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=0.4)
                ax.add_patch(rect)
                # numeric ΔG (pp) — choose text color for contrast
                dg = info["delta_pp"]
                # Light text on saturated blue (|dg| big), dark text on pale
                if abs(dg) >= 38:
                    txt_color = "white"
                else:
                    txt_color = "#1A1A1A"
                ax.text(x0 + 0.5, y0 + 0.55,
                        f"{dg:+.0f}",
                        ha="center", va="center",
                        fontsize=font_value, color=txt_color,
                        fontweight="bold")
                if not compact:
                    ax.text(x0 + 0.5, y0 + 0.84,
                            f"α={info['best_alpha']}",
                            ha="center", va="center",
                            fontsize=font_alpha, color=txt_color,
                            alpha=0.85)

            # Keystone outline for the 4 Mistral math/code cells
            if (anchor_key, ds, dr) in KEYSTONE_CELLS:
                ax.add_patch(Rectangle(
                    (x0, y0), 1, 1, facecolor="none",
                    edgecolor="black", linewidth=1.0, zorder=5,
                ))

    # Column headers (direction per column, dataset centered over each pair)
    ds_label = {"math800": "math", "code800": "code", "fact800": "fact"}
    for ci, (_, dr) in enumerate(col_specs):
        ax.text(ci + 0.5, -0.12, dr,
                ha="center", va="bottom",
                fontsize=font_label, color="#222")

    # Bracket + dataset name once per pair (each dataset spans 2 columns).
    for k, ds in enumerate(DATASETS):
        x_lo = 2 * k
        x_hi = 2 * k + 2
        # bracket above the direction labels
        ax.plot([x_lo + 0.08, x_hi - 0.08], [-0.46, -0.46],
                color="#666", lw=0.7, clip_on=False)
        # small ticks at bracket ends so the grouping is visually clear
        for x_end in (x_lo + 0.08, x_hi - 0.08):
            ax.plot([x_end, x_end], [-0.46, -0.40],
                    color="#666", lw=0.7, clip_on=False)
        # centered dataset name above bracket
        ax.text(x_lo + 1.0, -0.74, ds_label[ds],
                ha="center", va="bottom",
                fontsize=font_label, color="#222", fontweight="bold")

    # Row labels — anchor names
    for ri, (_, label) in enumerate(ANCHORS):
        ax.text(-0.18, ri + 0.5, label,
                ha="right", va="center",
                fontsize=font_label, color="#222")

    # Hide spines/ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_panel_a_legend(ax, cells, *, compact=False):
    """Tiny inline legend explaining encodings — placed below panel a."""
    # We use empty proxy artists in a legend call.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=HEATMAP_CMAP(HEATMAP_NORM(50)), edgecolor="white",
              label=r"$\Delta\mathrm{G}\geq+30$pp (anchor)"),
        Patch(facecolor=HEATMAP_CMAP(HEATMAP_NORM(15)), edgecolor="white",
              label=r"$0<\Delta\mathrm{G}<+30$pp (sub-anchor)"),
        Patch(facecolor="#EEEEEE", edgecolor=HATCH_COLOR, hatch="///",
              label=r"$\mathrm{gateN}\leq 4$ (anecdotal, N/A)"),
        Patch(facecolor="none", edgecolor="black", linewidth=1.0,
              label="Mistral math/code keystone"),
    ]
    ncol = 2 if compact else 4
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.05 if compact else -0.18),
              frameon=False, fontsize=SIZE_LEGEND,
              ncol=ncol, handlelength=1.6, handletextpad=0.5, columnspacing=1.2)


# ---------------------------------------------------------------------------
# Panel (b): Mistral code800 A→U dose-response
# ---------------------------------------------------------------------------
def _load_panel_b():
    rows = load_cell_rows("mistral", "code800", "A→U")
    rows = sorted(rows, key=lambda r: _alpha_to_int(r["alpha_mult"]))
    alphas = [_alpha_to_int(r["alpha_mult"]) for r in rows]
    sig = [r["criterion_invalidity_aware"]["signal"]["rate_gated"] * 100.0 for r in rows]
    rnd = [r["criterion_invalidity_aware"]["random"]["rate_gated"] * 100.0 for r in rows]

    # ±2pp sanity check against panel-b constants table
    expected = {5: (2.1, 0.0), 10: (2.1, 0.0), 20: (12.5, 0.0), 40: (35.4, 0.0)}
    for a, s, r in zip(alphas, sig, rnd):
        es, er = expected[a]
        if abs(s - es) > 2.0 or abs(r - er) > 2.0:
            raise RuntimeError(
                f"panel-b: α={a} rendered ({s:.1f},{r:.1f}) vs expected ({es},{er})"
            )
    return alphas, sig, rnd


def draw_panel_b(ax, alphas, sig, rnd, *, compact=False):
    ax.plot(alphas, sig, "-o", color=SEMANTIC["signal"], lw=1.6, ms=5,
            label=r"signal $\hat{\mathbf{d}}_{\mathrm{imp}}$",
            markerfacecolor=SEMANTIC["signal"], markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    ax.plot(alphas, rnd, "--o", color=GRAY_MID, lw=1.2, ms=4,
            label="random direction",
            markerfacecolor="white", markeredgecolor=GRAY_MID,
            markeredgewidth=1.0, zorder=2)

    # annotate signal points
    for a, s in zip(alphas, sig):
        dy = 5 if s < 70 else -10
        ax.annotate(f"{s:.1f}%", (a, s),
                    textcoords="offset points",
                    xytext=(6, dy), fontsize=SIZE_LEGEND,
                    color=SEMANTIC["signal"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.minorticks_off()
    ax.set_xlabel(r"Steering magnitude $\alpha\,/\,\mathrm{proj\_std}$")
    ax.set_ylabel("Gated flip rate (%)")
    ax.set_ylim(-4, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    title_size = SIZE_TITLE - 0.5 if compact else SIZE_TITLE
    ax.set_title("(b) Mistral-7B / code800 A→U (L15)",
                 loc="left", fontsize=title_size, pad=4 if compact else 6)
    ax.legend(loc="upper left", frameon=False, fontsize=SIZE_LEGEND,
              handletextpad=0.4, borderpad=0.2)
    light_grid(ax, axis="y")
    hide_top_right(ax)


# ---------------------------------------------------------------------------
# Top-level renders
# ---------------------------------------------------------------------------
def render_full(cells, alphas, sig, rnd):
    # Full-width: two-column-spanning, ~6.5 × 3.0 in. Slightly taller than
    # the 2.6-in target so the (a) heatmap title clears the dataset header.
    fig = plt.figure(figsize=(6.5, 3.1))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2.4, 1.0],
        wspace=0.30, left=0.135, right=0.985, top=0.82, bottom=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_panel_a(ax_a, cells, font_label=SIZE_TICK, font_value=SIZE_LEGEND,
                 font_alpha=6.0, compact=False)
    ax_a.set_title("(a) Gated $\\Delta$G at best $\\alpha$ (invalidity-aware)",
                   loc="left", fontsize=SIZE_TITLE, pad=46)
    draw_panel_a_legend(ax_a, cells, compact=False)

    draw_panel_b(ax_b, alphas, sig, rnd, compact=False)

    out = FIG_DIR / "fig5_causal_v2.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def render_compact(cells, alphas, sig, rnd):
    # Single-column variant: vertical stack (heatmap on top, dose-response below).
    # 3.3-in wide column; 4.6-in tall so both panels stay legible.
    fig = plt.figure(figsize=(3.3, 4.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.45, 1.0],
        hspace=0.65, left=0.21, right=0.985, top=0.93, bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    draw_panel_a(ax_a, cells, font_label=6.5, font_value=6.5,
                 font_alpha=5.5, compact=True)
    ax_a.set_title("(a) Gated $\\Delta$G at best $\\alpha$",
                   loc="left", fontsize=SIZE_TITLE - 0.5, pad=32)
    draw_panel_a_legend(ax_a, cells, compact=True)

    draw_panel_b(ax_b, alphas, sig, rnd, compact=True)

    out = FIG_DIR / "fig5_causal_v2_compact.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# LaTeX caption files
# ---------------------------------------------------------------------------
CAPTION_FULL = r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig5_causal_v2.pdf}
  \caption{\textbf{$d_{\mathrm{imp}}$ causally controls refusal behavior;
    math and code admit anchor-quality control, fact is a structural
    boundary case.} (a) Gated $\Delta$G at best $\alpha$ across the
    4-anchor v2 intervention grid (Mistral-7B-Instruct, Gemma-3-4B-it,
    Qwen3-14B, Qwen3-8B) $\times$ \{math800, code800, fact800\} $\times$
    \{A$\to$U, U$\to$A\}, under the invalidity-aware verification
    rubric (Appendix~\ref{app:protocol-audit}). Mistral-7B is the only
    bidirectional anchor on both math and code (outlined cells). The
    four fact U$\to$A direction cells are rendered N/A because all
    sixteen underlying per-$\alpha$ fact U$\to$A rows have
    $\mathrm{gateN} \leq 4$. (b) Mistral-7B code800 A$\to$U
    dose-response: signal-direction gated rate reaches $35.4\%$ at
    $\alpha{=}40$ ($\Delta\mathrm{G} = +35.4$pp because the
    random-direction control is $0\%$ throughout) -- the cleanest
    near-zero random null in the v2 grid.}
  \label{fig:causal}
\end{figure*}
"""

CAPTION_COMPACT = r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig5_causal_v2_compact.pdf}
  \caption{\textbf{$d_{\mathrm{imp}}$ causally controls refusal
    behavior; fact is a structural boundary case.} (a) Gated $\Delta$G
    at best $\alpha$ across the v2 4-anchor $\times$ 3-dataset
    $\times$ 2-direction grid; Mistral-7B is the bidirectional
    keystone (outlined). The four fact U$\to$A direction cells are
    N/A because every underlying per-$\alpha$ row has
    $\mathrm{gateN} \leq 4$. (b) Mistral code A$\to$U dose-response:
    signal (solid) vs random (dashed); rubric in
    Appendix~\ref{app:protocol-audit}.}
  \label{fig:causal-compact-app}
\end{figure}
"""


def write_captions():
    (HERE / "fig5_causal_v2_caption_full.tex").write_text(CAPTION_FULL)
    (HERE / "fig5_causal_v2_caption_compact.tex").write_text(CAPTION_COMPACT)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_panel_a_table(cells):
    print()
    print("PANEL (a) per-cell renders:")
    print(
        f"  {'anchor':<22} {'dataset':<7} {'dir':<5} "
        f"{'best_α':>6} {'gateN':>6} {'sig%':>6} {'rnd%':>6} {'ΔG':>7}"
    )
    n_na = 0
    for anchor_key, anchor_lbl in ANCHORS:
        for ds in DATASETS:
            for dr in DIRECTIONS:
                info = cells[(anchor_key, ds, dr)]
                if info["anecdotal"]:
                    n_na += 1
                    print(f"  {anchor_lbl:<22} {ds:<7} {dr:<5} {'N/A':>6} {info['gateN']:>6} {'-':>6} {'-':>6} {'N/A':>7}")
                else:
                    print(
                        f"  {anchor_lbl:<22} {ds:<7} {dr:<5} "
                        f"{info['best_alpha']:>6d} {info['gateN']:>6d} "
                        f"{info['signal_pp']:>5.1f}% {info['random_pp']:>5.1f}% "
                        f"{info['delta_pp']:>+6.1f}pp"
                    )
    print(f"  → anecdotal cells = {n_na} (expected = 4, all on fact U→A)")


def _print_panel_b_table(alphas, sig, rnd):
    print()
    print("PANEL (b) per-α renders (Mistral-7B code800 A→U):")
    print(f"  {'α':>3}  {'signal%':>8}  {'random%':>8}")
    for a, s, r in zip(alphas, sig, rnd):
        print(f"  {a:>3d}  {s:>7.1f}%  {r:>7.1f}%")


def main():
    print(f"INT_DIR = {INT_DIR}")
    print(f"FIG_DIR = {FIG_DIR}")
    cells = _scan_cells()
    _validate_against_factsheet(cells)

    alphas, sig, rnd = _load_panel_b()

    out_full    = render_full(cells, alphas, sig, rnd)
    out_compact = render_compact(cells, alphas, sig, rnd)
    write_captions()

    _print_panel_a_table(cells)
    _print_panel_b_table(alphas, sig, rnd)
    print()
    print("FILES WRITTEN")
    for p in (out_full, out_compact,
              HERE / "fig5_causal_v2_caption_full.tex",
              HERE / "fig5_causal_v2_caption_compact.tex"):
        sz = p.stat().st_size if p.exists() else 0
        print(f"  {p}  ({sz} bytes)")


if __name__ == "__main__":
    main()
