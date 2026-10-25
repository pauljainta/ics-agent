"""Build per-cell, per-arm Wilson reports from new self-contained bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ics_agent.plcbench_evidence import rescore_bundle
from ics_agent.plcbench_experiment import canonical_hash
from ics_agent.plcbench_study import cell_summary


def _records_for_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    attempts = json.loads((run_dir / "attempts.json").read_text())
    records = []
    details = []
    for attempt in attempts:
        bundle = run_dir / f"{attempt['arm']}-{attempt['rep']}"
        meta_path = bundle / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cell_id = meta.get("experiment", {}).get("cell_id")
            objective = meta.get("objective")
            if not cell_id or not objective:
                raise ValueError(f"bundle lacks self-contained cell/objective: {bundle}")
            objective_sha = canonical_hash(objective)
            if objective_sha != attempt["objective_sha256"]:
                raise ValueError(f"attempt objective differs from bundle: {bundle}")
            assessment = rescore_bundle(bundle)
            assessable = (attempt["status"] == "completed" and
                          assessment["status"] == "assessable" and
                          assessment["task_specific"]["physical"]["impact"]["assessable"])
            success = assessment.get("task_specific", {}).get("flags", {}).get("impact", False)
        else:
            cell_id = attempt["cell_id"]
            objective_sha = attempt["objective_sha256"]
            assessment = {"status": "not_retrospectively_assessable", "reason": attempt.get("error")}
            assessable, success = False, False
        records.append({"cell_id": cell_id, "objective_sha256": objective_sha,
                        "arm": attempt["arm"], "assessable": assessable, "success": success})
        details.append({"run_dir": str(run_dir), "attempt": attempt,
                        "assessment": assessment})
    return records, details


def report_runs(run_dirs: list[Path]) -> dict:
    records = []
    details = []
    seen = set()
    declared_pairs = set()
    for run_dir in run_dirs:
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            declared_pairs.add(plan["declared_pairs"])
        phase_records, phase_details = _records_for_run(run_dir)
        for item in phase_details:
            attempt = item["attempt"]
            key = (attempt["cell_id"], attempt["objective_sha256"],
                   attempt["rep"], attempt["arm"])
            if key in seen:
                raise ValueError(f"duplicate pair arm across phases: {key}")
            seen.add(key)
        records.extend(phase_records)
        details.extend(phase_details)
    if len(declared_pairs) > 1:
        raise ValueError("phase plans disagree on declared pair count")
    n_pairs = declared_pairs.pop() if declared_pairs else None
    pairs = {}
    order = {"native_first": set(), "matched_first": set()}
    for item in details:
        a = item["attempt"]
        pair = (a["cell_id"], a["objective_sha256"], a["rep"])
        pairs.setdefault(pair, set()).add(a["arm"])
        if a.get("order"):
            first = a["order"][0]
            order[f"{first}_first"].add(pair)
    complete = sum(arms == {"native", "matched"} for arms in pairs.values())
    return {"design": {"declared_pairs": n_pairs,
                       "complete_pairs": complete,
                       "native_first_pairs": len(order["native_first"]),
                       "matched_first_pairs": len(order["matched_first"])},
            "summary": cell_summary(records), "attempts": details}


def report_run(run_dir: Path) -> dict:
    return report_runs([run_dir])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = report_runs(args.run_dirs)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
