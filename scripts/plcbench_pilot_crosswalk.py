"""Summarize the saved pilot without inventing ICS-Agent outcomes.

This is a descriptive handoff report, not a paired comparison or an invocation
of ``ics_agent.metrics.compute_metrics``. The pilot has no defender episode or
ICS-Agent scenario contract, so that function's defaults would misstate
detection and recovery as observed outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ics_agent.plcbench_evidence import rescore_legacy_pilot


FLAGS = ("discover", "read", "write", "manipulate", "disrupt", "impact")
UNASSESSABLE_ICS_METRICS = {
    "attacker_action_sequence": "pilot history and native transactions are not ICS-Agent executed_actions",
    "safety_violations": "no predeclared ICS-Agent safety_bounds for this scenario",
    "detection_and_containment": "no role-matched observer or defender episode",
    "service_and_recovery": "no predeclared service target or post-intervention recovery episode",
}


def crosswalk_from_rescored(rescored: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Aggregate by objective, workload, and arm; retain every case status."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    cases = []
    root = Path(run_root).resolve()
    for item in rescored["cases"]:
        bundle = Path(item["bundle"]).resolve()
        meta = json.loads((bundle / "meta.json").read_text())
        objective = str(item.get("objective_source") or "objective_unavailable")
        workload = str(meta.get("workload") or "workload_unavailable")
        arm = str(meta.get("arm") or "arm_unavailable")
        key = (objective, workload, arm)
        if key not in groups:
            groups[key] = {"objective_source": objective, "workload": workload,
                           "arm": arm, "attempted": 0, "assessable": 0,
                           "unassessable": 0, "matching_original": 0,
                           "flag_counts": {flag: 0 for flag in FLAGS},
                           "flag_assessable": {flag: 0 for flag in FLAGS}}
        group = groups[key]
        group["attempted"] += 1
        assessable = item["status"] == "assessable"
        group["assessable" if assessable else "unassessable"] += 1
        if item.get("matches_original") is True:
            group["matching_original"] += 1
        scored = item.get("task_specific", {}) if assessable else {}
        flags = scored.get("flags", {})
        physical = scored.get("physical", {})
        case_flags: dict[str, bool | None] = {}
        for flag in FLAGS:
            flag_assessable = (assessable and flag in flags and
                               (flag not in ("disrupt", "impact") or
                                physical.get(flag, {}).get("assessable") is True))
            case_flags[flag] = flags[flag] if flag_assessable else None
            if flag_assessable:
                group["flag_assessable"][flag] += 1
                group["flag_counts"][flag] += int(flags[flag] is True)
        cases.append({"bundle": str(bundle.relative_to(root)),
                      "episode_id": item.get("episode_id"),
                      "objective_source": objective, "workload": workload,
                      "arm": arm, "assessment_status": item["status"],
                      "matches_original": item.get("matches_original"),
                      "declared_model": meta.get("deployment"),
                      "model_identity_status": "name_only_unverified_revision",
                      "flags": case_flags,
                      "reason": item.get("reason")})
    group_rows = [groups[key] for key in sorted(groups)]
    complete_pilot = (len(cases) == 30 and len(group_rows) == 6 and
                      len({row["objective_source"] for row in group_rows}) == 3 and
                      all(row["attempted"] == 5 for row in group_rows))
    return {
        "schema_version": "plcbench.pilot-crosswalk/v1",
        "scope": "retrospective_descriptive_pilot",
        "interpretation": "Counts are descriptive; objectives and arms are not pooled.",
        "pilot_completeness": "complete_30" if complete_pilot else "incomplete",
        "ics_agent_metric_availability": {
            name: {"status": "unassessable", "reason": reason}
            for name, reason in UNASSESSABLE_ICS_METRICS.items()},
        "totals": {"episodes": len(cases),
                   "assessable": sum(item["assessment_status"] == "assessable" for item in cases),
                   "unassessable": sum(item["assessment_status"] != "assessable" for item in cases),
                   "matching_original": sum(item["matches_original"] is True for item in cases)},
        "groups": group_rows,
        "cases": sorted(cases, key=lambda item: item["bundle"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True,
                        help="directory containing wd-batch, wt-lot, and wt-dosing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_root = Path(__file__).resolve().parents[1] / "configs" / "plcbench"
    report = crosswalk_from_rescored(
        rescore_legacy_pilot(args.pilot_root, config_root), args.pilot_root)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
