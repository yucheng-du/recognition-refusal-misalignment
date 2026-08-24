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

PROVISIONAL_PROTOCOL = (
    "llm_assisted_candidate_plus_provisional_audit_fill_invalidity_aware_v2"
)
PASSTHROUGH_PROTOCOL = "llm_assisted_candidate_passthrough_invalidity_aware_v2"


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

    steering = facts["steering"]
    print("\n3. Deterministic steering-breadth summary (supporting evidence only)")
    for dataset in ("math800", "code800", "fact800"):
        block = steering["by_dataset"][dataset]
        print(
            f"  {dataset:8s}: positive-reduction={block['n_positive_reduction']}"
            f"/{block['n_cells']}, best-alpha-zero={block['n_best_alpha_zero']}, "
            f"mean active reduction={block['mean_halluc_reduction_active']:.3f}, "
            f"mean active preservation cost={block['mean_preservation_cost_active']:.3f}"
        )
    check(steering["n_cells_total"] == 33, "all 33 steering-breadth cells are present", failures)
    check(not steering["missing_cells"], "no steering-breadth cells are marked missing", failures)

    print("\n4. Intervention-label provenance metadata")
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
