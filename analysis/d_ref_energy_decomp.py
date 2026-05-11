"""
d_ref / d_imp Energy Decomposition (A-PC vs A-null subspace) + cos_full_full
attribution.

For each (model, dataset) cell in the F7 direction-comparison grid, we
decompose **both** refusal direction d_ref and impossibility direction
d_imp_full into the two subspaces defined by PCA on train-A activations:

    E_PC(v)   = || V_A^T V_A · v ||^2 / || v ||^2
    E_null(v) = || (I - V_A^T V_A) · v ||^2 / || v ||^2

where V_A are the top-K (K=100) PCs fitted on train-A last-token reps.

Then we attribute the full-space cosine cos_full_full = <d_imp_full, d_ref>
to its PC-PC and null-null contributions (cross terms vanish because V^T V
and I-V^T V are orthogonal projectors):

    cos_full_full = <V^T V d_imp_full, V^T V d_ref>   ← PC-PC contribution
                  + <(I-V^T V) d_imp_full, (I-V^T V) d_ref>  ← null-null contribution

Using the identity d_imp_lt = (I-V^T V) d_imp_full / sqrt(E_null(d_imp_full))
and the fact that d_imp_lt lives in A-null (so <d_imp_lt, V^T V d_ref> = 0):

    null_null = sqrt(E_null(d_imp_full)) · cos_matched_full
    PC_PC    = cos_full_full − null_null

── What needs saved data ─────────────────────────────────────────────────
(1) E_PC(d_ref) / E_null(d_ref): from JSON scalars alone (cos_matched_full,
    cos_same_A_null). No rerun.
(2) E_PC(d_imp_full) / E_null(d_imp_full): from saved
    `reps_last_token_all_layers.npy` + `meta.jsonl` + PCA(K=100) on train-A
    (seed 42, matching compare_impossibility_vs_refusal_direction.py). No
    model reload — just numpy.

── Derivation of (1) ─────────────────────────────────────────────────────
d_imp_lt lies in the A-null subspace by construction, so <d_imp_lt, V^T V d_ref> = 0.
Therefore:
    cos_matched_full = <d_imp_lt, d_ref>
                     = <d_imp_lt, (I-V^T V) d_ref>
                     = cos_same_A_null · sqrt(E_null(d_ref))
Rearranging:
    E_null(d_ref) = (cos_matched_full / cos_same_A_null)^2
    E_PC(d_ref)   = 1 − E_null(d_ref)

Usage:
    python analysis/d_ref_energy_decomp.py

Outputs:
    experiments/analysis/d_ref_energy_decomp/d_ref_energy_decomp.csv
    experiments/analysis/d_ref_energy_decomp/d_ref_energy_decomp.md
"""
import json
import os
import csv
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
OUT_DIR = os.path.join(EXP_DIR, "analysis", "d_ref_energy_decomp")
os.makedirs(OUT_DIR, exist_ok=True)

# Best layers (mirror compare_impossibility_vs_refusal_direction.BEST_LAYERS)
BEST_LAYERS = {
    "mistral": 15, "llama": 15, "qwen": 18, "smollm2": 11,
    "gemma2": 16, "phi3": 15, "qwen14b": 34, "mistral_small": 28,
}
INSTRUCT_MODELS = list(BEST_LAYERS.keys())
BASE_MODELS = [m + "_base" for m in INSTRUCT_MODELS]
DATASETS = ["math800", "code800"]
K_PC = 100  # matches pca.n_components in compare_impossibility_vs_refusal_direction.py
SEED = 42   # matches train-A permutation seed


def find_json(model, dataset, layer):
    """Locate direction_comparison JSON for a (model, dataset, layer) cell."""
    suffix = f"direction_comparison_{model}_{dataset}_L{layer}.json"
    default = f"direction_comparison_{model}.json"
    if dataset == "math800" and model in INSTRUCT_MODELS:
        p = os.path.join(EXP_DIR, default)
        if os.path.exists(p):
            return p
    p = os.path.join(EXP_DIR, suffix)
    return p if os.path.exists(p) else None


def compute_dimp_full_energy(model, dataset, layer):
    """Compute E_PC(d_imp_full) and E_null(d_imp_full) from saved reps.

    Mirrors compute_impossibility_direction() in
    compare_impossibility_vs_refusal_direction.py: same train-A/U split
    (seed 42, first half), same PCA (K=100 on train-A).

    Returns:
        (E_PC_dimp, E_null_dimp, K_actual) or (None, None, None) if reps missing.
    """
    sig_dir = os.path.join(
        EXP_DIR, f"signals/{dataset}_{model}_allL/signals"
    )
    lt_path = os.path.join(sig_dir, "reps_last_token_all_layers.npy")
    meta_path = os.path.join(sig_dir, "meta.jsonl")
    if not (os.path.exists(lt_path) and os.path.exists(meta_path)):
        return None, None, None

    reps = np.load(lt_path, mmap_mode="r")
    meta = [json.loads(l) for l in open(meta_path)]
    labels = np.array([m["answerable"] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float32)

    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    rng = np.random.RandomState(SEED)
    pA = rng.permutation(len(A_idx))
    pU = rng.permutation(len(U_idx))
    trA = A_idx[pA[: len(A_idx) // 2]]
    trU = U_idx[pU[: len(U_idx) // 2]]

    K_actual = min(K_PC, len(trA) - 1, X.shape[1] - 1)
    pca = PCA(n_components=K_actual).fit(X[trA])
    V = pca.components_  # K × D

    # d_imp_full = unit(mean(X[trU]) - mean(X[trA]))
    d_imp_full = X[trU].mean(0) - X[trA].mean(0)
    d_imp_full = d_imp_full / (norm(d_imp_full) + 1e-15)

    # Project onto PC subspace
    pc_proj = V.T @ (V @ d_imp_full)  # V^T V d_imp_full, shape D
    E_PC = float(pc_proj @ pc_proj)
    null_resid = d_imp_full - pc_proj
    E_null = float(null_resid @ null_resid)

    return E_PC, E_null, K_actual


def decompose_one(json_path, model, dataset):
    """Extract cos scalars from JSON and compute both direction decompositions.

    For d_ref: closed-form from (cos_matched_full, cos_same_A_null).
    For d_imp_full: compute from saved reps.
    Then attribute cos_full_full into PC-PC + null-null contributions.
    """
    with open(json_path, "r") as f:
        d = json.load(f)
    cm = float(d["cos_matched_full"])
    cn = float(d["cos_same_A_null"])
    cff = float(d.get("cos_full_full", float("nan")))
    D = int(d["hidden_dim"])
    verified = bool(d.get("behavior_verified", False))
    layer = int(d["layer"])
    nH = int(d.get("n_harmful_verified_raw", -1))
    nHL = int(d.get("n_harmless_verified_raw", -1))

    # d_ref energy via closed form
    if abs(cn) < 1e-9:
        E_null_dref = float("nan")
        E_PC_dref = float("nan")
    else:
        ratio = cm / cn
        E_null_dref = min(max(ratio * ratio, 0.0), 1.0)
        E_PC_dref = 1.0 - E_null_dref

    # d_imp_full energy from saved reps
    E_PC_dimp, E_null_dimp, K_actual = compute_dimp_full_energy(model, dataset, layer)

    # Attribute cos_full_full
    # NOTE on share semantics: pc_pc_share / null_null_share are computed as
    # |contribution| / (|PC-PC| + |null-null|) — i.e. MAGNITUDE shares, not
    # signed fractions of cff. This makes the two shares sum to 1 even when
    # signs disagree, and gives a defensible "share-of-alignment-mass"
    # reading. When both contributions have the same sign as cff (true for
    # all 16 instruct cells), magnitude share equals signed share. For paper
    # writeup, phrase as "X% of the magnitude of cos_full_full" to stay
    # precise. See AUDIT_REPORT §9.6.3 for the wording history.
    if E_null_dimp is not None and E_null_dimp == E_null_dimp:  # not NaN
        null_null_contrib = float(np.sqrt(max(E_null_dimp, 0.0))) * cm
        pc_pc_contrib = cff - null_null_contrib
        abs_total = abs(pc_pc_contrib) + abs(null_null_contrib)
        pc_pc_share = abs(pc_pc_contrib) / abs_total if abs_total > 0 else float("nan")
        null_null_share = abs(null_null_contrib) / abs_total if abs_total > 0 else float("nan")
    else:
        null_null_contrib = float("nan")
        pc_pc_contrib = float("nan")
        pc_pc_share = float("nan")
        null_null_share = float("nan")

    random_E_PC = K_PC / D
    conc_dref = (E_PC_dref / random_E_PC) if random_E_PC > 0 else float("nan")
    conc_dimp = (
        (E_PC_dimp / random_E_PC) if (E_PC_dimp is not None and random_E_PC > 0)
        else float("nan")
    )

    return {
        "layer": layer,
        "verified": verified,
        "hidden_dim": D,
        "K_PC": K_actual if K_actual else K_PC,
        "cos_matched_full": cm,
        "cos_same_A_null": cn,
        "cos_full_full": cff,
        "E_PC_dref": E_PC_dref,
        "E_null_dref": E_null_dref,
        "E_PC_dimp_full": E_PC_dimp if E_PC_dimp is not None else float("nan"),
        "E_null_dimp_full": E_null_dimp if E_null_dimp is not None else float("nan"),
        "random_E_PC": random_E_PC,
        "conc_dref": conc_dref,
        "conc_dimp": conc_dimp,
        "cff_PC_PC": pc_pc_contrib,
        "cff_null_null": null_null_contrib,
        "cff_PC_PC_share": pc_pc_share,
        "cff_null_null_share": null_null_share,
        "n_harmful_verified": nH,
        "n_harmless_verified": nHL,
    }


def collect(models):
    rows = []
    missing = []
    for m in models:
        layer = BEST_LAYERS.get(m.replace("_base", ""))
        if layer is None:
            continue
        for ds in DATASETS:
            path = find_json(m, ds, layer)
            if path is None:
                missing.append((m, ds, layer))
                continue
            rec = decompose_one(path, m, ds)
            rec["model"] = m
            rec["dataset"] = ds
            rec["source"] = os.path.relpath(path, BASE)
            rows.append(rec)
    return rows, missing


def write_csv(rows, path):
    if not rows:
        return
    fields = [
        "model", "dataset", "layer", "verified", "hidden_dim", "K_PC",
        "cos_matched_full", "cos_same_A_null", "cos_full_full",
        "E_PC_dref", "E_null_dref", "E_PC_dimp_full", "E_null_dimp_full",
        "random_E_PC", "conc_dref", "conc_dimp",
        "cff_PC_PC", "cff_null_null", "cff_PC_PC_share", "cff_null_null_share",
        "n_harmful_verified", "n_harmless_verified", "source",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def format_energy_table(rows, title):
    lines = [f"### {title} — energy decomposition", ""]
    lines.append(
        "| model | dataset | L | D | verified | "
        "E_PC(d_ref) | E_null(d_ref) | ×rand | "
        "E_PC(d_imp_full) | E_null(d_imp_full) | ×rand |"
    )
    lines.append("|---|---|---:|---:|:-:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['layer']} | {r['hidden_dim']} | "
            f"{'✓' if r['verified'] else '—'} | "
            f"{r['E_PC_dref']:.4f} | {r['E_null_dref']:.4f} | {r['conc_dref']:.1f}× | "
            f"{r['E_PC_dimp_full']:.4f} | {r['E_null_dimp_full']:.4f} | {r['conc_dimp']:.1f}× |"
        )
    lines.append("")
    return "\n".join(lines)


def format_attribution_table(rows, title):
    lines = [f"### {title} — cos_full_full attribution", ""]
    lines.append(
        "| model | dataset | cos_full_full | cos_matched_full | "
        "null-null = √E_null(d_imp)·cos_matched | PC-PC = cff − null-null | "
        "PC-PC share | null-null share |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | "
            f"{r['cos_full_full']:+.4f} | {r['cos_matched_full']:+.4f} | "
            f"{r['cff_null_null']:+.4f} | {r['cff_PC_PC']:+.4f} | "
            f"{r['cff_PC_PC_share']:.3f} | {r['cff_null_null_share']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def summary_stats(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], float) and r[key] == r[key]]
    if not vals:
        return None
    return {
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def main():
    instruct_rows, instruct_missing = collect(INSTRUCT_MODELS)
    base_rows, base_missing = collect(BASE_MODELS)

    # CSVs
    write_csv(instruct_rows, os.path.join(OUT_DIR, "d_ref_energy_decomp_instruct.csv"))
    write_csv(base_rows, os.path.join(OUT_DIR, "d_ref_energy_decomp_base.csv"))
    write_csv(
        instruct_rows + base_rows,
        os.path.join(OUT_DIR, "d_ref_energy_decomp.csv"),
    )

    # Markdown
    md_parts = [
        "# d_ref / d_imp Energy Decomposition + cos_full_full Attribution",
        "",
        "For each (model, dataset) cell we decompose **both** the Arditi refusal "
        "direction d_ref and the full-space impossibility direction d_imp_full "
        "into the A-PC subspace (top-K=100 PCs on train-A) and its orthogonal "
        "complement (A-null). We then attribute the full-space cosine "
        "`cos_full_full = <d_imp_full, d_ref>` into its PC-PC and null-null "
        "contributions (cross terms vanish because V^T V and I-V^T V are "
        "orthogonal projectors).",
        "",
        "Identities used:",
        "- `E_null(d_ref) = (cos_matched_full / cos_same_A_null)²` — from JSON scalars.",
        "- `E_null(d_imp_full)` — from saved reps + PCA on train-A (seed 42).",
        "- `null_null_contrib = √E_null(d_imp_full) · cos_matched_full`",
        "- `PC_PC_contrib = cos_full_full − null_null_contrib`",
        "",
        format_energy_table(instruct_rows, "Instruct models (16 cells)"),
        "",
        format_attribution_table(instruct_rows, "Instruct models (16 cells)"),
        "",
        format_energy_table(base_rows, "Base models"),
        "",
        format_attribution_table(base_rows, "Base models"),
        "",
    ]
    if instruct_missing or base_missing:
        md_parts.append("**Missing cells:**")
        for m, ds, L in instruct_missing + base_missing:
            md_parts.append(f"- {m} / {ds} / L{L}")
        md_parts.append("")

    # Summary numbers for wording decisions
    s_pc_dref = summary_stats(instruct_rows, "E_PC_dref")
    s_pc_dimp = summary_stats(instruct_rows, "E_PC_dimp_full")
    s_pc_share = summary_stats(instruct_rows, "cff_PC_PC_share")
    s_nn_share = summary_stats(instruct_rows, "cff_null_null_share")
    md_parts.extend([
        "## Summary (Instruct models, 16 cells)",
        "",
        f"- E_PC(d_ref): mean {s_pc_dref['mean']:.3f}, range "
        f"[{s_pc_dref['min']:.3f}, {s_pc_dref['max']:.3f}]"
        if s_pc_dref else "",
        f"- E_PC(d_imp_full): mean {s_pc_dimp['mean']:.3f}, range "
        f"[{s_pc_dimp['min']:.3f}, {s_pc_dimp['max']:.3f}]"
        if s_pc_dimp else "",
        f"- cos_full_full PC-PC share: mean {s_pc_share['mean']:.3f}, range "
        f"[{s_pc_share['min']:.3f}, {s_pc_share['max']:.3f}]"
        if s_pc_share else "",
        f"- cos_full_full null-null share: mean {s_nn_share['mean']:.3f}, range "
        f"[{s_nn_share['min']:.3f}, {s_nn_share['max']:.3f}]"
        if s_nn_share else "",
        "",
        "## Interpretation (data-driven)",
        "",
        "PC-PC share := |PC-PC contribution| / (|PC-PC| + |null-null|) — it is "
        "the fraction of the *magnitude* of cos_full_full attributable to "
        "PC-PC overlap. null-null share is defined analogously. They sum to 1 "
        "by construction. When both contributions have the same sign as "
        "cos_full_full (true for all 16 instruct cells), the magnitude share "
        "equals the signed fraction; otherwise the magnitude form is the "
        "safe one. For paper writeup use phrases like *\"~85% of the magnitude "
        "of cos_full_full\"* rather than *\"85% of cos_full_full\"* to avoid "
        "implying a signed fraction claim.",
        "",
        "cos_matched_full is by construction a scaled version of the null-null "
        "contribution (scaled by √E_null(d_imp_full)): null_null_contrib = "
        "√E_null(d_imp_full) · cos_matched_full. The large PC-PC share is "
        "what separates cos_full_full from cos_matched_full.",
        "",
    ])
    out_md = os.path.join(OUT_DIR, "d_ref_energy_decomp.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md_parts))

    # Console print
    print(format_energy_table(instruct_rows, "Instruct models (16 cells)"))
    print(format_attribution_table(instruct_rows, "Instruct models (16 cells)"))
    print(format_energy_table(base_rows, "Base models"))
    print(format_attribution_table(base_rows, "Base models"))
    print("\nSummary stats (instruct):")
    for name, s in [
        ("E_PC(d_ref)", s_pc_dref), ("E_PC(d_imp_full)", s_pc_dimp),
        ("PC-PC share", s_pc_share), ("null-null share", s_nn_share),
    ]:
        if s is None:
            continue
        print(f"  {name}: mean {s['mean']:.3f}, range [{s['min']:.3f}, {s['max']:.3f}], n={s['n']}")
    if instruct_missing or base_missing:
        print("\nMissing cells:")
        for m, ds, L in instruct_missing + base_missing:
            print(f"  - {m} / {ds} / L{L}")
    print(f"\nWrote:\n  {os.path.join(OUT_DIR, 'd_ref_energy_decomp.csv')}\n  {out_md}")


if __name__ == "__main__":
    main()
