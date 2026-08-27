#!/usr/bin/env python3
"""Lightweight consistency check for the shipped aggregate artifacts.

This command performs no model inference and does not require the omitted
``experiments/signals`` tree. It reports the frozen detection, geometry, and
deterministic steering-breadth summaries, then checks that intervention-label
provenance metadata matches the released effective labels.

Run from the repository root:

    python3 scripts/verify_core_conclusions.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = ROOT / "experiments" / "main_grid_facts_v2.json"
INTERVENTION_DIR = ROOT / "experiments" / "intervention"
STRUCT_BEHAV_PATH = ROOT / "experiments" / "d_struct_behav_matrix.json"
V2DET_DIR = ROOT / "experiments" / "steering" / "v2det"
SUBSPACE_DIR = ROOT / "analysis" / "subspace_overlap"

PROVISIONAL_PROTOCOL = (
    "llm_assisted_candidate_plus_provisional_audit_fill_invalidity_aware_v2"
)
PASSTHROUGH_PROTOCOL = "llm_assisted_candidate_passthrough_invalidity_aware_v2"
V2DET_PROTOCOL = "v2_deterministic_invalidity_aware_with_mixed_output_guard"
V2DET_MODELS = {
    "gemma2", "gemma3_4b", "llama", "mistral", "mistral_small",
    "mistral_small_3_2", "olmo13b", "phi3", "phi4mini", "qwen",
    "qwen14b", "qwen32b", "qwen3_8b", "qwen3_14b", "qwen3_32b", "smollm2",
}
SUBSPACE_CELLS = {
    ("mistral", "math800", 15),
    ("mistral", "code800", 15),
    ("qwen3_14b", "math800", 25),
    ("qwen3_14b", "code800", 24),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check(condition: bool, message: str, failures: list[str]) -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {message}")
    if not condition:
        failures.append(message)


def main() -> int:
    facts = load_json(FACTS_PATH)
    failures: list[str] = []

    print("=" * 78)
    print("LIGHTWEIGHT ARTIFACT CONSISTENCY CHECK")
    print("=" * 78)

    form = facts["form"]
    print("\n1. Structural-impossibility detection (22 math/code cells)")
    print(
        "  CosNSRT AUC: "
        f"mean={form['global_auc_mean']:.3f}, "
        f"range=[{form['global_auc_min']:.3f}, {form['global_auc_max']:.3f}]"
    )
    check(form["n_cells"] == 22, "all 22 detection cells are present", failures)
    check(not form["missing_cells"], "no detection cells are marked missing", failures)

    orth = facts["orthogonality"]
    print("\n2. Recognition/refusal geometry (22 math/code cells)")
    print(
        "  matched-space cosine: "
        f"mean={orth['cos_mean']:.3f}, "
        f"range=[{orth['cos_min']:.3f}, {orth['cos_max']:.3f}]"
    )
    check(orth["n_cells"] == 22, "all 22 geometry cells are present", failures)
    check(not orth["missing_cells"], "no geometry cells are marked missing", failures)

    print("\n3. Current v2det steering breadth (48 cells)")
    v2det_files = sorted(V2DET_DIR.glob("steering_*_v2det.json"))
    v2det_cells: dict[tuple[str, str], dict] = {}
    v2det_protocols: Counter[str] = Counter()
    duplicate_v2det_cells: list[tuple[str, str]] = []
    for path in v2det_files:
        artifact = load_json(path)
        key = (artifact["model"], artifact["dataset"])
        if key in v2det_cells:
            duplicate_v2det_cells.append(key)
        v2det_cells[key] = artifact
        v2det_protocols[artifact.get("verification_protocol", "<missing>")] += 1

    check(len(v2det_files) == 48, "48 current v2det steering artifacts are present", failures)
    check(not duplicate_v2det_cells, "no duplicate v2det model--dataset cells", failures)
    check(
        {model for model, _ in v2det_cells} == V2DET_MODELS,
        "v2det artifacts cover the exact 16-model breadth grid",
        failures,
    )
    check(
        set(v2det_cells) == {
            (model, dataset)
            for model in V2DET_MODELS
            for dataset in ("math800", "code800", "fact800")
        },
        "every v2det model has exactly one math, code, and fact cell",
        failures,
    )
    check(
        v2det_protocols[V2DET_PROTOCOL] == 48 and len(v2det_protocols) == 1,
        "all v2det artifacts use the deterministic invalidity-aware protocol",
        failures,
    )

    expected_v2det = {
        "math800": (15, 0.1325),
        "code800": (12, 0.088125),
        "fact800": (4, 0.00375),
    }
    for dataset, (expected_positive, expected_mean) in expected_v2det.items():
        rows = [artifact for (_, ds), artifact in v2det_cells.items() if ds == dataset]
        reductions = [artifact["hallucination_reduction"] for artifact in rows]
        positive = sum(value > 0 for value in reductions)
        mean_reduction = sum(reductions) / len(reductions)
        print(
            f"  {dataset:8s}: positive-reduction={positive}/{len(rows)}, "
            f"mean reduction={mean_reduction:.6f}"
        )
        check(len(rows) == 16, f"{dataset} has 16 v2det cells", failures)
        check(positive == expected_positive, f"{dataset} positive-cell count matches the paper", failures)
        check(
            abs(mean_reduction - expected_mean) < 1e-12,
            f"{dataset} mean v2det reduction matches the frozen summary",
            failures,
        )

    print("\n4. Multidimensional subspace robustness (4 cells)")
    energies = load_json(SUBSPACE_DIR / "subspace_overlap_v2_energies.json")
    angles = []
    for name in (
        "subspace_overlap_v2_angles_mistral.json",
        "subspace_overlap_v2_angles_qwen3_14b.json",
    ):
        angles.extend(load_json(SUBSPACE_DIR / name))

    energy_cells = {(cell["model"], cell["dataset"], cell["layer"]) for cell in energies}
    angle_cells = {(cell["model"], cell["dataset"], cell["layer"]) for cell in angles}
    check(len(energies) == 4 and energy_cells == SUBSPACE_CELLS,
          "energy JSON covers the exact four reported cells", failures)
    check(len(angles) == 4 and angle_cells == SUBSPACE_CELLS,
          "principal-angle JSONs cover the exact four reported cells", failures)
    check(
        all("cached behavior-verified d_ref" in cell.get("provenance", "") for cell in energies),
        "energy results use cached behavior-verified refusal directions",
        failures,
    )
    check(
        all("behavior-verified prompt subsets" in cell.get("provenance", "") for cell in angles),
        "angle results use behavior-verified prompt subsets",
        failures,
    )

    full_k5 = [100 * cell["k5"]["dref_full"]["energy"] for cell in energies]
    full_k10 = [100 * cell["k10"]["dref_full"]["energy"] for cell in energies]
    anull_k5 = [100 * cell["k5"]["dref_Anull"]["energy"] for cell in energies]
    anull_k10 = [100 * cell["k10"]["dref_Anull"]["energy"] for cell in energies]
    first_cos = [cell[k]["first_cos"] for cell in angles for k in ("k5", "k10")]
    first_angles = [cell[k]["first_angle_deg"] for cell in angles for k in ("k5", "k10")]
    print(f"  d_ref full-space energy: k=5 {min(full_k5):.1f}--{max(full_k5):.1f}%, "
          f"k=10 {min(full_k10):.1f}--{max(full_k10):.1f}%")
    print(f"  d_ref A-null energy:    k=5 {min(anull_k5):.1f}--{max(anull_k5):.1f}%, "
          f"k=10 {min(anull_k10):.1f}--{max(anull_k10):.1f}%")
    print(f"  first cosine range={min(first_cos):.2f}--{max(first_cos):.2f}; "
          f"minimum angle={min(first_angles):.1f} degrees")
    check(
        (round(min(full_k5), 1), round(max(full_k5), 1),
         round(min(full_k10), 1), round(max(full_k10), 1)) == (1.2, 2.1, 2.0, 3.4),
        "full-space refusal-energy ranges match the paper",
        failures,
    )
    check(
        (round(min(anull_k5), 1), round(max(anull_k5), 1),
         round(min(anull_k10), 1), round(max(anull_k10), 1)) == (1.4, 2.4, 2.4, 3.9),
        "A-null-renormalized refusal-energy ranges match the paper",
        failures,
    )
    check(round(min(first_angles), 1) == 74.1,
          "minimum reported principal angle is 74.1 degrees", failures)
    check((round(min(first_cos), 2), round(max(first_cos), 2)) == (0.14, 0.27),
          "first-principal-cosine range is 0.14--0.27", failures)

    print("\n5. Intervention-label provenance metadata")
    intervention_files = sorted(INTERVENTION_DIR.glob("intervention_*_full_v2.json"))
    protocols: Counter[str] = Counter()
    qwen_passthrough_ok = True
    for path in intervention_files:
        artifact = load_json(path)
        protocol = artifact.get("verification_protocol", "<missing>")
        protocols[protocol] += 1
        if "intervention_qwen3_8b_" in path.name:
            stats = artifact.get("merge_stats", {})
            qwen_passthrough_ok &= (
                protocol == PASSTHROUGH_PROTOCOL
                and stats.get("audit_overrides_applied") == 0
                and stats.get("non_audit_passthrough") == 900
            )
    print(f"  provisional-audit-fill artifacts: {protocols[PROVISIONAL_PROTOCOL]}")
    print(f"  candidate-passthrough artifacts:  {protocols[PASSTHROUGH_PROTOCOL]}")
    check(len(intervention_files) == 13, "13 released v2 intervention artifacts are present", failures)
    check(protocols[PROVISIONAL_PROTOCOL] == 10, "10 artifacts record provisional audit-subset fills", failures)
    check(protocols[PASSTHROUGH_PROTOCOL] == 3, "3 Qwen3-8B artifacts record candidate passthrough", failures)
    check(qwen_passthrough_ok, "Qwen3-8B merge stats confirm zero applied overrides", failures)

    struct_behav = load_json(STRUCT_BEHAV_PATH)
    label_sources = Counter(cell["label_source"] for cell in struct_behav["cells"])
    check(
        label_sources["llm_assisted_candidate_plus_provisional_audit_fill"] == 6,
        "behavior-direction matrix records six provisional-fill cells",
        failures,
    )
    check(
        label_sources["llm_assisted_candidate_passthrough"] == 2,
        "behavior-direction matrix records two Qwen3-8B passthrough cells",
        failures,
    )

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED: {len(failures)} consistency check(s) failed")
        return 1
    print("PASS: shipped aggregate artifacts and provenance metadata are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
