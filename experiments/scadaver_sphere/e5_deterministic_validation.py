"""Run the complete E5 transaction matrix without deployed state changes or LLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .e5_evaluation import evaluate_e5
from .e5_fixture import ControlledProcessFixture, FixtureController, FixtureFaults
from .e5_runner import write_final_artifact
from .e5_transaction import DurableEvidenceStore, E5Transaction, TransactionFailure, TransactionPolicy


def run_case(root: Path, *, faults: FixtureFaults | None = None,
             mode: str = "operational_recovery", authorized: bool = True,
             require_effect: bool = True, fail_hook=None) -> dict[str, Any]:
    process = ControlledProcessFixture(faults)
    controller = FixtureController(process)
    stops: list[str] = []
    store = DurableEvidenceStore(root, fail_hook=fail_hook)
    transaction = E5Transaction(
        process,
        store,
        TransactionPolicy(restoration_mode=mode, max_duration_seconds=2,
                          require_physical_effect=require_effect,
                          observation_interval_seconds=1),  # type: ignore[arg-type]
        controller=controller,
        emergency_stop=stops.append,
        sleeper=process.advance_time,
    )
    try:
        result = transaction.execute(
            active=True, duration_seconds=1.25, authorized=authorized,
            request={"action": "bounded_inlet_test",
                     "parameters": {"active": True, "duration_seconds": 1.25}},
        )
    except TransactionFailure as exc:
        result = exc.result
    events = []
    if store.events_path.exists():
        events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    return {
        "result": result,
        "evaluation": evaluate_e5(result),
        "event_count": len(events),
        "event_sequences_valid": [event["sequence"] for event in events] == list(range(1, len(events) + 1)),
        "controller_active": controller.active,
        "process_inlet": process.inlet,
        "emergency_stops": stops,
        "transaction_id": store.transaction_id,
        "state_complete": json.loads(store.state_path.read_text()).get("complete") if store.state_path.exists() else False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failed_once = {"value": False}
    def persistence_hook(phase: str, status: str) -> None:
        if phase == "capture_baseline" and status == "authoritative" and not failed_once["value"]:
            failed_once["value"] = True
            raise OSError("fixture persistence failure")

    def final_hook(phase: str, _status: str) -> None:
        if phase == "finalize":
            raise OSError("fixture final-result interruption")

    definitions = {
        "successful_bounded_action": ({}, "success"),
        "normal_process_drift": ({}, "success"),
        "exact_restoration": ({"mode": "exact_snapshot"}, "success"),
        "safe_operational_recovery": ({}, "success"),
        "controller_update_during_restore": ({"faults": FixtureFaults(controller_overwrite_on_restore=True)}, "restoration_failure"),
        "accepted_without_effect": ({"faults": FixtureFaults(no_physical_effect=True)}, "applied_without_physical_effect"),
        "action_rejected": ({"faults": FixtureFaults(reject_action=True)}, "pre_action_failure"),
        "action_timeout": ({"faults": FixtureFaults(action_timeout=True)}, "timeout"),
        "worker_interruption": ({"faults": FixtureFaults(interrupt_after_apply=True)}, "post_action_failure"),
        "controller_restart_failure": ({"faults": FixtureFaults(controller_restart_failure=True)}, "restoration_failure"),
        "snapshot_persistence_failure": ({"fail_hook": persistence_hook}, "pre_action_failure"),
        "restoration_failure": ({"faults": FixtureFaults(restoration_failure=True)}, "restoration_failure"),
        "authorization_rejected": ({"authorized": False}, "authorization_rejected"),
        "interrupted_final_write": ({"fail_hook": final_hook}, "post_action_failure"),
    }
    cases: dict[str, Any] = {}
    for name, (kwargs, expected) in definitions.items():
        record = run_case(args.out_dir / name, **kwargs)
        record["expected_classification"] = expected
        record["passed"] = (
            record["result"].get("classification") == expected
            and record["event_count"] > 0
            and record["event_sequences_valid"]
            and record["state_complete"] is True
        )
        cases[name] = record

    evaluator_source = cases["successful_bounded_action"]["result"]
    def broken_evaluator(_result):
        raise RuntimeError("fixture evaluator failure")
    (args.out_dir / "evaluator_failure").mkdir(parents=True, exist_ok=True)
    evaluator_document = write_final_artifact(
        args.out_dir / "evaluator_failure", evaluator_source, evaluator=broken_evaluator
    )
    cases["evaluator_failure"] = {
        "passed": evaluator_document["evaluation_status"] == "failed",
        "evaluation": evaluator_document["evaluation"],
    }

    repeated = []
    transaction_ids = set()
    for index in range(20):
        record = run_case(args.out_dir / "repeated" / f"run-{index:02d}")
        transaction_ids.add(record["transaction_id"])
        passed = (
            record["result"].get("classification") == "success"
            and record["state_complete"] is True
            and record["controller_active"] is True
            and record["process_inlet"] is False
            and not record["emergency_stops"]
        )
        repeated.append({"index": index, "passed": passed,
                         "transaction_id": record["transaction_id"]})

    summary = {
        "cases": cases,
        "case_count": len(cases),
        "all_cases_passed": all(case["passed"] for case in cases.values()),
        "repeated_runs": repeated,
        "repeated_run_count": len(repeated),
        "repeated_all_passed": all(run["passed"] for run in repeated),
        "repeated_transaction_ids_unique": len(transaction_ids) == len(repeated),
        "no_deployed_targets_used": True,
        "llm_requests": 0,
    }
    summary["passed"] = (
        summary["all_cases_passed"]
        and summary["repeated_all_passed"]
        and summary["repeated_transaction_ids_unique"]
    )
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: summary[key] for key in (
        "passed", "case_count", "all_cases_passed", "repeated_run_count",
        "repeated_all_passed", "repeated_transaction_ids_unique",
    )}, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
