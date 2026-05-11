"""d_ref / d_imp Energy Decomposition (v2, 11-model main grid).

Re-runs the energy-decomposition + cos_full_full attribution from
``analysis/d_ref_energy_decomp.py`` on the 22 (model, dataset) cells of the
11-model main grid defined in ``experiments/main_grid_facts_v2.json``
(``orthogonality.cells``).

Algorithm is identical to the v1 script (closed-form for d_ref,
saved-reps PCA for d_imp_full); only the cell list is swapped from the
legacy 8-model grid to the 11-model main grid. No model forward.

Hard-fails if the computation does not produce exactly 22 instruct rows.

Outputs (NEW; do NOT overwrite the v1 archival files):
    experiments/analysis/d_ref_energy_decomp/d_ref_energy_decomp_11model_instruct.csv
    experiments/analysis/d_ref_energy_decomp/d_ref_energy_decomp_11model_instruct.md
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent.parent
EXP_DIR = BASE / "experiments"
OUT_DIR = EXP_DIR / "analysis" / "d_ref_energy_decomp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_PC = 100
SEED = 42

# v1-archival paths must NOT be overwritten.
V1_INSTRUCT_CSV = OUT_DIR / "d_ref_energy_decomp_instruct.csv"
V1_BASE_CSV = OUT_DIR / "d_ref_energy_decomp_base.csv"
V1_COMBINED_CSV = OUT_DIR / "d_ref_energy_decomp.csv"
V1_MD = OUT_DIR / "d_ref_energy_decomp.md"


# ---------------------------------------------------------------------------
# Cell list (11-model / 22-cell instruct main grid)
# ---------------------------------------------------------------------------
def load_main_grid_cells() -> list[dict]:
    facts = json.loads((EXP_DIR / "main_grid_facts_v2.json").read_text())
    cells = facts["orthogonality"]["cells"]
    if len(cells) != 22:
        raise RuntimeError(
            f"main_grid_facts_v2.json orthogonality.cells: expected 22, got {len(cells)}"
        )
    n_math = sum(1 for c in cells if c["dataset"] == "math800")
    n_code = sum(1 for c in cells if c["dataset"] == "code800")
    if n_math != 11 or n_code != 11:
        raise RuntimeError(
            f"expected 11 math800 + 11 code800 cells, got {n_math} + {n_code}"
        )
    return cells


def cell_to_dc_path(cell: dict) -> Path:
    """Resolve a cell to its direction_comparison_*.json path."""
    m, d, L = cell["model"], cell["dataset"], cell["layer"]
    if d == "math800":
        return EXP_DIR / f"direction_comparison_{m}.json"
    return EXP_DIR / f"direction_comparison_{m}_code800_L{L}.json"


def cell_to_signals_dir(cell: dict) -> Path:
    return EXP_DIR / "signals" / f"{cell['dataset']}_{cell['model']}_allL" / "signals"


# ---------------------------------------------------------------------------
# Energy decomposition (mirror v1)
# ---------------------------------------------------------------------------
def compute_dimp_full_energy(cell: dict) -> tuple[float, float, int]:
    """E_PC(d_imp_full) and E_null(d_imp_full) from saved reps. Returns
    (E_PC, E_null, K_actual)."""
    sig_dir = cell_to_signals_dir(cell)
    lt_path = sig_dir / "reps_last_token_all_layers.npy"
    meta_path = sig_dir / "meta.jsonl"
    if not (lt_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"missing saved reps for {cell['model']}/{cell['dataset']}: {sig_dir}"
        )

    reps = np.load(lt_path, mmap_mode="r")
    meta = [json.loads(line) for line in open(meta_path)]
    labels = np.array([m["answerable"] for m in meta])
    L = cell["layer"]
    if L >= reps.shape[1]:
        raise RuntimeError(
            f"{cell['model']}/{cell['dataset']}: layer L{L} >= reps shape {reps.shape}"
        )
    X = np.array(reps[:, L, :], dtype=np.float32)

    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    rng = np.random.RandomState(SEED)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[: len(A_idx) // 2]]
    trU = U_idx[pU[: len(U_idx) // 2]]

    K_actual = min(K_PC, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=K_actual).fit(X[trA])
    V = pca.components_  # K x D

    d_imp_full = X[trU].mean(0) - X[trA].mean(0)
    d_imp_full = d_imp_full / (norm(d_imp_full) + 1e-15)

    pc_proj = V.T @ (V @ d_imp_full)
    E_PC = float(pc_proj @ pc_proj)
    null_resid = d_imp_full - pc_proj
    E_null = float(null_resid @ null_resid)
    return E_PC, E_null, K_actual


def decompose_one(cell: dict) -> dict:
    json_path = cell_to_dc_path(cell)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with open(json_path) as f:
        d = json.load(f)
    if d.get("layer") != cell["layer"]:
        raise RuntimeError(
            f"{cell['model']}/{cell['dataset']}: dc-json layer L{d.get('layer')} "
            f"!= cell layer L{cell['layer']}"
        )

    cm = float(d["cos_matched_full"])
    cn = float(d["cos_same_A_null"])
    cff = float(d.get("cos_full_full", float("nan")))
    D = int(d["hidden_dim"])
    verified = bool(d.get("behavior_verified", False))
    layer = int(d["layer"])
    nH = int(d.get("n_harmful_verified_raw", -1))
    nHL = int(d.get("n_harmless_verified_raw", -1))

    # d_ref energy via closed form (v1 identity).
    if abs(cn) < 1e-9:
        E_null_dref = float("nan")
        E_PC_dref = float("nan")
    else:
        ratio = cm / cn
        E_null_dref = min(max(ratio * ratio, 0.0), 1.0)
        E_PC_dref = 1.0 - E_null_dref

    # d_imp_full energy from saved reps.
    E_PC_dimp, E_null_dimp, K_actual = compute_dimp_full_energy(cell)

    null_null_contrib = float(np.sqrt(max(E_null_dimp, 0.0))) * cm
    pc_pc_contrib = cff - null_null_contrib
    abs_total = abs(pc_pc_contrib) + abs(null_null_contrib)
    if abs_total > 0:
        pc_pc_share = abs(pc_pc_contrib) / abs_total
        null_null_share = abs(null_null_contrib) / abs_total
    else:
        pc_pc_share = float("nan")
        null_null_share = float("nan")

    sign_agreement = (
        (pc_pc_contrib > 0 and null_null_contrib > 0 and cff > 0) or
        (pc_pc_contrib < 0 and null_null_contrib < 0 and cff < 0)
    )

    random_E_PC = K_PC / D
    conc_dref = (E_PC_dref / random_E_PC) if random_E_PC > 0 else float("nan")
    conc_dimp = (
        (E_PC_dimp / random_E_PC) if (E_PC_dimp is not None and random_E_PC > 0)
        else float("nan")
    )

    return {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "layer": layer,
        "verified": verified,
        "hidden_dim": D,
        "K_PC": K_actual or K_PC,
        "cos_matched_full": cm,
        "cos_same_A_null": cn,
        "cos_full_full": cff,
        "E_PC_dref": E_PC_dref,
        "E_null_dref": E_null_dref,
        "E_PC_dimp_full": E_PC_dimp,
        "E_null_dimp_full": E_null_dimp,
        "random_E_PC": random_E_PC,
        "conc_dref": conc_dref,
        "conc_dimp": conc_dimp,
        "cff_PC_PC": pc_pc_contrib,
        "cff_null_null": null_null_contrib,
        "cff_PC_PC_share": pc_pc_share,
        "cff_null_null_share": null_null_share,
        "sign_agreement_with_cff": sign_agreement,
        "n_harmful_verified": nH,
        "n_harmless_verified": nHL,
        "source_dc_json": str(json_path.relative_to(BASE)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "model", "dataset", "layer", "verified", "hidden_dim", "K_PC",
    "cos_matched_full", "cos_same_A_null", "cos_full_full",
    "E_PC_dref", "E_null_dref", "E_PC_dimp_full", "E_null_dimp_full",
    "random_E_PC", "conc_dref", "conc_dimp",
    "cff_PC_PC", "cff_null_null", "cff_PC_PC_share", "cff_null_null_share",
    "sign_agreement_with_cff",
    "n_harmful_verified", "n_harmless_verified",
    "source_dc_json",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def fmt_summary(rows: list[dict]) -> dict:
    def stats(key):
        vals = [r[key] for r in rows if isinstance(r[key], float) and r[key] == r[key]]
        if not vals:
            return None
        return {
            "n": len(vals),
            "min": min(vals), "max": max(vals),
            "mean": sum(vals) / len(vals),
        }

    return {
        "E_PC_dref":         stats("E_PC_dref"),
        "E_PC_dimp_full":    stats("E_PC_dimp_full"),
        "cff_PC_PC_share":   stats("cff_PC_PC_share"),
        "cff_null_null_share": stats("cff_null_null_share"),
        "cos_full_full_abs": {
            "n": len(rows),
            "min": min(abs(r["cos_full_full"]) for r in rows),
            "max": max(abs(r["cos_full_full"]) for r in rows),
            "mean": sum(abs(r["cos_full_full"]) for r in rows) / len(rows),
        },
        "n_sign_disagreement": sum(1 for r in rows if not r["sign_agreement_with_cff"]),
    }


def write_md(rows: list[dict], summary: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append("# d_ref / d_imp Energy Decomposition + cos_full_full Attribution (11-model main grid)")
    lines.append("")
    lines.append(
        "Re-run of the v1 closed-form / saved-reps decomposition (see "
        "`analysis/d_ref_energy_decomp.py`) on the 22 (model, dataset) cells "
        "of the 11-model main grid defined in "
        "`experiments/main_grid_facts_v2.json` (`orthogonality.cells`). Algorithm "
        "is unchanged; only the cell list is swapped from the legacy 8-model grid "
        "to the 11-model main grid. No model forward."
    )
    lines.append("")
    lines.append("## Per-cell decomposition (22 instruct cells)")
    lines.append("")
    lines.append(
        "| model | dataset | L | D | verified | "
        "$\\cos_\\text{matched,full}$ | $\\cos_\\text{full,full}$ | "
        "$E_{\\mathrm{PC}}(d_\\text{ref})$ | $E_\\text{null}(d_\\text{ref})$ | "
        "$E_{\\mathrm{PC}}(d_\\text{imp,full})$ | $E_\\text{null}(d_\\text{imp,full})$ | "
        "PC-PC | null-null | PC-PC share | null-null share |"
    )
    lines.append("|---|---|---:|---:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['layer']} | {r['hidden_dim']} | "
            f"{'✓' if r['verified'] else '—'} | "
            f"{r['cos_matched_full']:+.4f} | {r['cos_full_full']:+.4f} | "
            f"{r['E_PC_dref']:.4f} | {r['E_null_dref']:.4f} | "
            f"{r['E_PC_dimp_full']:.4f} | {r['E_null_dimp_full']:.4f} | "
            f"{r['cff_PC_PC']:+.4f} | {r['cff_null_null']:+.4f} | "
            f"{r['cff_PC_PC_share']:.3f} | {r['cff_null_null_share']:.3f} |"
        )
    lines.append("")
    lines.append("## Summary stats (22 instruct cells)")
    lines.append("")
    for name, key in [
        ("$E_{\\mathrm{PC}}(d_\\text{ref})$", "E_PC_dref"),
        ("$E_{\\mathrm{PC}}(d_\\text{imp,full})$", "E_PC_dimp_full"),
        ("$|\\cos_\\text{full,full}|$", "cos_full_full_abs"),
        ("PC-PC share", "cff_PC_PC_share"),
        ("null-null share", "cff_null_null_share"),
    ]:
        s = summary[key]
        if s is None:
            continue
        lines.append(
            f"- {name}: mean **{s['mean']:.3f}**, range "
            f"$[{s['min']:.3f}, {s['max']:.3f}]$, n={s['n']}"
        )
    lines.append(
        f"- Cells with sign-disagreement among (PC-PC, null-null, $\\cos_\\text{{full,full}}$): "
        f"**{summary['n_sign_disagreement']}/{len(rows)}**"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "PC-PC share := $|\\text{PC-PC contrib}| / (|\\text{PC-PC}| + |\\text{null-null}|)$ — "
        "the magnitude share of $\\cos_\\text{full,full}$ attributable to PC-PC overlap. "
        "The two shares sum to 1 by construction. When PC-PC and null-null have the same "
        "sign as $\\cos_\\text{full,full}$, magnitude share equals signed share. We track "
        "sign agreement explicitly to flag any cell where 'share' wording would be ambiguous."
    )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    cells = load_main_grid_cells()
    rows = [decompose_one(c) for c in cells]
    if len(rows) != 22:
        raise RuntimeError(f"expected 22 instruct rows, got {len(rows)}")

    out_csv = OUT_DIR / "d_ref_energy_decomp_11model_instruct.csv"
    out_md = OUT_DIR / "d_ref_energy_decomp_11model_instruct.md"
    if any(p.resolve() == out_csv.resolve() for p in [V1_INSTRUCT_CSV, V1_BASE_CSV, V1_COMBINED_CSV]):
        raise RuntimeError(f"refusing to overwrite v1 archival path: {out_csv}")
    if out_md.resolve() == V1_MD.resolve():
        raise RuntimeError(f"refusing to overwrite v1 archival path: {out_md}")
    write_csv(rows, out_csv)

    summary = fmt_summary(rows)
    write_md(rows, summary, out_md)

    # Console
    print(f"22 instruct cells, decomposition complete. Wrote:")
    print(f"  {out_csv.relative_to(BASE)}")
    print(f"  {out_md.relative_to(BASE)}")
    print()
    print("Summary stats:")
    for name, key in [
        ("E_PC(d_ref)", "E_PC_dref"),
        ("E_PC(d_imp_full)", "E_PC_dimp_full"),
        ("|cos_full_full|", "cos_full_full_abs"),
        ("PC-PC share", "cff_PC_PC_share"),
        ("null-null share", "cff_null_null_share"),
    ]:
        s = summary[key]
        if s is None:
            continue
        print(f"  {name:24s} mean {s['mean']:.4f}  range [{s['min']:.4f}, {s['max']:.4f}]  n={s['n']}")
    print(f"  Sign disagreement cells:  {summary['n_sign_disagreement']}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
