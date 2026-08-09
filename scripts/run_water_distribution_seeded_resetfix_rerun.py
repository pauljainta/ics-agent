#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.reporting import write_csv, write_json
from ics_agent.utils import dump_yaml
from ics_agent.water_distribution_seeded_backbone import (
    CANONICAL_RESULT_ROOT,
    FAMILY_BY_NAME,
    SEEDS,
    SeededBackboneRunner,
    TOTAL_RUNS,
    _read_json,
)


TARGET_FAMILIES = (
    "black_box_attacker__agentic_defender",
    "gray_box_attacker__agentic_defender",
)


def _materialize_metadata(runner: SeededBackboneRunner) -> None:
    runner.ensure_root_layout()
    expected_payloads = runner._expected_metadata_payloads()
    for actual_path, (kind, payload) in expected_payloads.items():
        if kind == "csv":
            write_csv(actual_path, list(payload))
        elif kind == "json":
            write_json(actual_path, payload)
        elif kind == "yaml":
            dump_yaml(actual_path, payload)
        else:
            actual_path.parent.mkdir(parents=True, exist_ok=True)
            actual_path.write_text(str(payload))
    if not runner.run_catalog_path.exists():
        write_csv(runner.run_catalog_path, [])
    if not runner.run_failures_path.exists():
        write_csv(runner.run_failures_path, [])
    runner._sync_all_ledgers()
    runner._set_last_step("manifest_frozen")
    runner._set_phase("manifest_frozen", "completed")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun only the corrected seeded black-box and gray-box families."
    )
    parser.add_argument(
        "--result-root",
        default=str(CANONICAL_RESULT_ROOT),
        help="Canonical seeded result root.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts per scenario worker run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = SeededBackboneRunner(Path(args.result_root))
    runner._ensure_env_defaults(SEEDS[0])
    _materialize_metadata(runner)
    runner.environment_validation_path.unlink(missing_ok=True)
    runner._validate_distribution_environment()
    runner._set_phase("running_grid", "in_progress")

    for family_name in TARGET_FAMILIES:
        family = FAMILY_BY_NAME[family_name]
        for seed in SEEDS:
            seed_status = _read_json(runner._seed_status_path(family.family_name, seed), {})
            if seed_status.get("completed_scenarios") == 11:
                continue
            code = runner.run_worker(
                family.family_name,
                seed,
                max_attempts=int(args.max_attempts),
            )
            if code != 0:
                return code

    runner._sync_all_ledgers()
    resume_state = _read_json(runner.resume_state_path, {})
    if resume_state.get("completed_runs") != TOTAL_RUNS:
        raise RuntimeError(
            f"expected {TOTAL_RUNS} completed runs, found {resume_state.get('completed_runs')}"
        )
    runner._set_phase("running_grid", "completed", completed_runs=TOTAL_RUNS)
    runner.write_family_aggregates()
    runner.write_paper_export()
    runner.write_repo_note()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
