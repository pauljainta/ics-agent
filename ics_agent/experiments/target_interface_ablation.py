from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from ics_agent.config import load_config, resolve_repo_path, resolve_runtime_config_path
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, portable_path, results_root
from ics_agent.reporting import write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import dump_yaml, load_yaml, utc_now


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

FIXED_PROFILE_KEYS = (
    "allowed_action_types",
    "llm",
    "action_cooldown_sec",
    "note_cooldown_sec",
    "reasoning_cooldown_sec",
    "reasoning_refresh_sec",
    "memory",
    "behavior_model",
    "knowledge_files",
    "abort_on_backend_error",
)


def build_parser(*, default_config: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
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
    parser.add_argument(
        "--only-condition",
        default=None,
        help=(
            "Execute only one named condition after validating the full "
            "two-condition interface-ablation manifest."
        ),
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
    generated_path = generated_dir / f"{Path(config_path).stem}_seed_{int(seed)}.yaml"

    seeded_config = dict(raw_config)
    reproducibility = dict(seeded_config.get("reproducibility") or {})
    reproducibility["seed"] = int(seed)
    seeded_config["reproducibility"] = reproducibility
    seeded_config["results_root"] = f"../seed_{int(seed)}"
    dump_yaml(generated_path, seeded_config)
    return generated_path


def _condition_filtered_config_path(
    config_path: Path,
    condition_name: str | None,
) -> Path:
    if not condition_name:
        return config_path
    raw_config = load_yaml(config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"expected mapping at {config_path}")
    conditions = raw_config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError(f"expected non-empty conditions list at {config_path}")
    selected = [
        condition
        for condition in conditions
        if isinstance(condition, dict) and str(condition.get("name")) == condition_name
    ]
    if len(selected) != 1:
        raise RuntimeError(f"condition `{condition_name}` not found exactly once")

    filtered_config = dict(raw_config)
    filtered_config["conditions"] = selected
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", condition_name).strip("_")
    generated_path = config_path.parent / f"{config_path.stem}_only_{slug}.yaml"
    dump_yaml(generated_path, filtered_config)
    return generated_path


def _resolved_experiment_root(config: dict[str, object]) -> Path:
    results_base_dir = (
        Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
    )
    resolved_results_root = resolve_repo_path(
        config.get("results_root", str(results_root())),
        base_dir=results_base_dir,
    )
    return resolved_results_root / str(config["experiment_name"])


def _target_semantics_without_interface(profile: dict[str, Any]) -> dict[str, Any]:
    comparable = deepcopy(profile)
    target_semantics = comparable.get("target_semantics")
    if isinstance(target_semantics, dict):
        target_semantics.pop("interface_variant", None)
        target_semantics.pop("interaction_interface", None)
    comparable.pop("target_interface_variant", None)
    return comparable


def _interface_variant(profile: dict[str, Any]) -> str:
    target_semantics = profile.get("target_semantics")
    value: Any = None
    if isinstance(target_semantics, dict):
        value = target_semantics.get("interface_variant")
        if value is None:
            value = target_semantics.get("interaction_interface")
    if value is None:
        value = profile.get("target_interface_variant")
    return str(value or "normalized_interface").strip().lower().replace("-", "_")


def _selected_fields(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(profile[key]) for key in FIXED_PROFILE_KEYS if key in profile}


def _all_equal(values: list[Any]) -> bool:
    return all(item == values[0] for item in values[1:]) if values else True


def _interface_capability_manifest(
    *,
    config: dict[str, object],
    conditions: list[dict[str, Any]],
    scenarios: list[Any],
    repeats: int,
    require_pair: bool = True,
) -> dict[str, Any]:
    if require_pair and len(conditions) != 2:
        raise RuntimeError("target-interface ablation must define exactly two conditions")

    attacker_profiles = [
        condition["effective_profiles"]["attacker"] for condition in conditions
    ]
    defender_profiles = [
        condition["effective_profiles"]["defender"] for condition in conditions
    ]
    observer_profiles = [
        condition["effective_profiles"]["observer"] for condition in conditions
    ]
    stripped_attackers = [
        _target_semantics_without_interface(profile) for profile in attacker_profiles
    ]
    condition_entries = []
    for condition, attacker in zip(conditions, attacker_profiles):
        target_semantics = dict(attacker.get("target_semantics") or {})
        condition_entries.append(
            {
                "condition": str(condition["name"]),
                "attacker_profile": str(condition["profiles"]["attacker"]),
                "knowledge_mode": str(attacker.get("knowledge_mode")),
                "target_semantics_variant": str(target_semantics.get("variant")),
                "interface_variant": _interface_variant(attacker),
                "allowed_action_types": list(attacker.get("allowed_action_types", [])),
                "fixed_policy_fields": _selected_fields(attacker),
            }
        )

    expected_interfaces = {"normalized_interface", "legacy_white_interface"}
    observed_interfaces = {
        str(entry["interface_variant"]) for entry in condition_entries
    }
    invariance_checks = {
        "attacker_profiles_match_after_removing_interface_variant": _all_equal(
            stripped_attackers
        ),
        "defender_profiles_identical": _all_equal(defender_profiles),
        "observer_profiles_identical": _all_equal(observer_profiles),
        "allowed_action_types_identical": _all_equal(
            [entry["allowed_action_types"] for entry in condition_entries]
        ),
        "llm_configs_identical": _all_equal(
            [dict(profile.get("llm") or {}) for profile in attacker_profiles]
        ),
        "timing_fields_identical": _all_equal(
            [
                {
                    key: profile.get(key)
                    for key in (
                        "action_cooldown_sec",
                        "note_cooldown_sec",
                        "reasoning_cooldown_sec",
                        "reasoning_refresh_sec",
                        "max_llm_decisions_per_episode",
                    )
                }
                for profile in attacker_profiles
            ]
        ),
        "memory_configs_identical": _all_equal(
            [dict(profile.get("memory") or {}) for profile in attacker_profiles]
        ),
        "target_semantics_fixed_at_full": all(
            str(entry["target_semantics_variant"]) == "full"
            for entry in condition_entries
        ),
        "expected_interface_variants_present": (
            observed_interfaces == expected_interfaces if require_pair else True
        ),
    }
    failed = [key for key, value in invariance_checks.items() if not value]
    if failed:
        raise RuntimeError(
            "target-interface capability manifest failed invariance checks: "
            + ", ".join(failed)
        )

    return {
        "experiment_name": str(config["experiment_name"]),
        "experiment_id": str(config.get("experiment_id", config["experiment_name"])),
        "testbed": str(config.get("testbed")),
        "seed": (config.get("reproducibility") or {}).get("seed"),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": repeats,
        "planned_runs": len(scenarios) * len(conditions) * repeats,
        "intended_comparison": {
            "normalized": "C_norm=<P,R,Q,O,Kg,Kt_full,I_norm,U,pi,Delta>",
            "rich_interface": "C_rich-I=<P,R,Q,O,Kg,Kt_full,I_legacy,U,pi,Delta>",
        },
        "fixed_dimensions": {
            "P_A": "post_access_operational_surface",
            "R_A": "same_reachable_interface_for_each_scenario",
            "Q_A": "same_within_episode_persistence",
            "O_A": "same_runtime_physical_evidence_and_feedback",
            "K_A_g": "same_generic_cyber_ics_context",
            "K_A_t": "full_target_semantics_in_both_conditions",
            "U_A": "same_underlying_executable_action_authority_and_final_validator",
            "pi_A": "same_agentic_policy_scaffold",
            "Delta_A": "same_timing_decision_budget_and_episode_duration",
        },
        "varied_dimension": {
            "I_A": [
                "normalized_interface",
                "legacy_white_interface",
            ]
        },
        "candidate_generation_constraints": [
            "The rich interface may expose or organize only actions already reachable under the shared U_A.",
            "The rich interface must not add evaluator labels, future defender behavior, hidden simulator state, or authored winning recipes.",
            "No white_box_commit_guardrail, faster cadence, larger candidate window, or model/backend change is enabled by this ablation.",
        ],
        "invariance_checks": invariance_checks,
        "conditions": condition_entries,
    }


def _experiment_plan(
    config_path: str | Path,
    *,
    require_pair_manifest: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    config = load_config(config_path)
    testbed_id = str(config.get("testbed", "water_treatment"))
    catalog = ScenarioCatalog(testbed_id=testbed_id)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    scenarios = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
    conditions = orchestrator._enumerate_conditions(config)
    repeats = int(config.get("repeats", 1))
    experiment_root = _resolved_experiment_root(config)
    manifest = _interface_capability_manifest(
        config=config,
        conditions=conditions,
        scenarios=scenarios,
        repeats=repeats,
        require_pair=require_pair_manifest,
    )

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
        "planned_runs": len(scenarios) * len(conditions) * repeats,
        "results_root": config.get("results_root"),
        "resolved_experiment_root": portable_path(experiment_root),
        "conditions": [str(item.get("name")) for item in conditions],
        "scenarios": [scenario.catalog_id for scenario in scenarios],
        "metric_keys_recorded_by_shared_pipeline": METRIC_KEYS_TO_EXPECT,
        "interface_capability_manifest": manifest,
    }
    plan = {
        "config": config,
        "testbed_id": testbed_id,
        "scenarios": scenarios,
        "conditions": conditions,
        "repeats": repeats,
        "experiment_root": experiment_root,
        "interface_capability_manifest": manifest,
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
    return experiment_root.parent / "_progress" / "target_interface_ablation_run.json"


def _manifest_path(experiment_root: Path) -> Path:
    return experiment_root.parent / "_manifests" / "interface_capability_manifest.json"


def main_for(*, default_config: str, testbed_id: str, description: str) -> int:
    args = build_parser(
        default_config=default_config,
        description=description,
    ).parse_args()
    base_config_path = _seeded_config_path(args.config, args.seed)
    comparison_payload, comparison_plan = _experiment_plan(
        base_config_path,
        require_pair_manifest=True,
    )
    manifest_path = _manifest_path(comparison_plan["experiment_root"])
    write_json(manifest_path, comparison_plan["interface_capability_manifest"])
    run_config_path = _condition_filtered_config_path(
        Path(base_config_path),
        args.only_condition,
    )
    planned_payload, plan = _experiment_plan(
        run_config_path,
        require_pair_manifest=False,
    )
    progress_path = _progress_path(plan["experiment_root"])
    planned_payload["interface_capability_manifest_path"] = portable_path(manifest_path)
    planned_payload["interface_capability_manifest"] = comparison_plan[
        "interface_capability_manifest"
    ]
    if args.only_condition:
        planned_payload["execution_filter"] = {
            "only_condition": args.only_condition,
            "full_comparison_config_path": portable_path(base_config_path),
            "filtered_config_path": portable_path(run_config_path),
            "full_comparison_planned_runs": comparison_payload["planned_runs"],
        }

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

    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    summary = orchestrator.run(run_config_path)
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
