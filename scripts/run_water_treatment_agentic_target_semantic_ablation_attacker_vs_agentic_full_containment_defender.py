#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from ics_agent.config import load_config, resolve_repo_path, resolve_runtime_config_path
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, portable_path, results_root
from ics_agent.reporting import write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import dump_yaml, load_yaml, utc_now


CONFIG_NAME = (
    "water_treatment_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
)

METRIC_KEYS_TO_EXPECT = [
    "wall_clock_time_to_first_useful_attacker_action_from_attack_start_sec",
    "wall_clock_time_to_detection_from_attack_start_sec",
    "wall_clock_time_to_first_defender_action_from_attack_start_sec",
    "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec",
    "wall_clock_time_to_first_safety_violation_from_attack_start_sec",
    "attacker_ineffective_action_count",
    "distinct_surfaces_probed_count",
    "distinct_surfaces_probed_before_first_useful_count",
    "physical_impact",
    "severity",
    "downtime_timestep_count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the water-treatment target-semantic ablation against the "
            "agentic full-containment defender."
        )
    )
    parser.add_argument(
        "--config",
        default=CONFIG_NAME,
        help="Experiment YAML path or configs/experiments filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override reproducibility.seed and write results under a seed-specific "
            "subfolder. Use --seed 101 for the canonical seed-101 run."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the scenario/condition grid without starting the backend.",
    )
    parser.add_argument(
        "--skip-mrg-preflight",
        action="store_true",
        help="Skip the MRG auth preflight. Use only if another wrapper checks auth.",
    )
    parser.add_argument(
        "--keep-failed-artifacts",
        action="store_true",
        help="Do not remove failed/incomplete trial folders before resuming.",
    )
    return parser


def _raw_results_root(config_path: Path, raw_config: dict[str, Any]) -> Path:
    return resolve_repo_path(
        raw_config.get("results_root", str(results_root())),
        base_dir=config_path.parent,
    )


def _seeded_config_path(config_arg: str | Path, seed: int | None) -> Path:
    config_path = resolve_runtime_config_path(config_arg)
    if seed is None:
        return config_path
    raw_config = load_yaml(config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"expected mapping at {config_path}")

    output_root = _raw_results_root(config_path, raw_config)
    generated_dir = output_root / "_generated_configs"
    generated_path = (
        generated_dir
        / f"{Path(config_path).stem}_seed_{int(seed)}.yaml"
    )

    seeded_config = dict(raw_config)
    reproducibility = dict(seeded_config.get("reproducibility") or {})
    reproducibility["seed"] = int(seed)
    seeded_config["reproducibility"] = reproducibility
    seeded_config["results_root"] = f"../seed_{int(seed)}"
    dump_yaml(generated_path, seeded_config)
    return generated_path


def _resolved_experiment_root(config: dict[str, object]) -> Path:
    results_base_dir = (
        Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
    )
    resolved_results_root = resolve_repo_path(
        config.get("results_root", str(results_root())),
        base_dir=results_base_dir,
    )
    if config.get("use_canonical_result_root"):
        return resolved_results_root / str(config["experiment_name"])
    return resolved_results_root / str(config["experiment_name"])


def _experiment_plan(config_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    config = load_config(config_path)
    testbed_id = str(config.get("testbed", "water_treatment"))
    catalog = ScenarioCatalog(testbed_id=testbed_id)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    scenarios = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
    conditions = orchestrator._enumerate_conditions(config)
    repeats = int(config.get("repeats", 1))
    planned_runs = len(scenarios) * len(conditions) * repeats
    experiment_root = _resolved_experiment_root(config)

    payload = {
        "validated": True,
        "config_path": portable_path(Path(config["_path"])),
        "experiment_name": config["experiment_name"],
        "testbed": testbed_id,
        "seed": config.get("reproducibility", {}).get("seed"),
        "resume": bool(config.get("resume")),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": repeats,
        "planned_runs": planned_runs,
        "results_root": config.get("results_root"),
        "resolved_experiment_root": portable_path(experiment_root),
        "conditions": [str(item.get("name")) for item in conditions],
        "scenarios": [scenario.catalog_id for scenario in scenarios],
        "metric_keys_recorded_by_shared_pipeline": METRIC_KEYS_TO_EXPECT,
    }
    plan = {
        "config": config,
        "testbed_id": testbed_id,
        "scenarios": scenarios,
        "conditions": conditions,
        "repeats": repeats,
        "experiment_root": experiment_root,
    }
    return payload, plan


def _preflight_mrg_auth(config: dict[str, object]) -> None:
    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        return
    if str(environment.get("backend", "")).strip() != "sphere_openplc":
        return
    merge_server = str(environment.get("merge_server", "")).strip()
    command = ["mrg"]
    if merge_server:
        command.extend(["-s", merge_server])
    command.append("whoami")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("MRG CLI is not installed or is not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MRG auth preflight timed out.") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "MRG auth preflight failed before launching experiments. "
            "Run `mrg login`, then rerun this script. "
            f"Details: {detail}"
        )


def _read_status(path: Path) -> str | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return "incomplete"
    return str(payload.get("status") or "incomplete")


def _remove_failed_or_incomplete_trials(plan: dict[str, object]) -> list[str]:
    removed: list[str] = []
    root = plan["experiment_root"]
    if not isinstance(root, Path) or not root.exists():
        return removed
    scenarios = plan["scenarios"]
    conditions = plan["conditions"]
    repeats = int(plan["repeats"])
    for condition_spec in conditions:
        condition_name = str(condition_spec.get("name"))
        for scenario in scenarios:
            for repeat in range(1, repeats + 1):
                run_dir = (
                    root
                    / condition_name
                    / scenario.suite
                    / scenario.scenario_id
                    / f"trial-{repeat:03d}"
                )
                if not run_dir.exists():
                    continue
                status = _read_status(run_dir)
                if status == "completed":
                    continue
                shutil.rmtree(run_dir)
                removed.append(portable_path(run_dir))
    return removed


def _progress_path(experiment_root: Path) -> Path:
    return experiment_root.parent / "_progress" / "target_semantic_ablation_run.json"


def main() -> int:
    args = build_parser().parse_args()
    config_path = _seeded_config_path(args.config, args.seed)
    planned_payload, plan = _experiment_plan(config_path)
    progress_path = _progress_path(plan["experiment_root"])

    if args.validate_only:
        payload = dict(planned_payload)
        payload["mode"] = "validate_only"
        print(json.dumps(payload, indent=2))
        return 0

    write_json(
        progress_path,
        {
            "status": "started",
            "started_utc": utc_now(),
            "runner": Path(__file__).name,
            "plan": planned_payload,
        },
    )
    if not args.skip_mrg_preflight:
        _preflight_mrg_auth(plan["config"])
    removed = []
    if not args.keep_failed_artifacts:
        removed = _remove_failed_or_incomplete_trials(plan)
    if removed:
        print(
            json.dumps(
                {
                    "pre_run_cleanup": "removed_failed_or_incomplete_trials",
                    "removed_count": len(removed),
                    "removed": removed,
                },
                indent=2,
            )
        )

    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    summary = orchestrator.run(config_path)
    summary["runner_plan"] = planned_payload
    write_json(
        progress_path,
        {
            "status": summary.get("status", "unknown"),
            "updated_utc": utc_now(),
            "runner": Path(__file__).name,
            "plan": planned_payload,
            "removed_failed_or_incomplete_trials": removed,
            "summary_path": portable_path(plan["experiment_root"] / "experiment_summary.json"),
        },
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
