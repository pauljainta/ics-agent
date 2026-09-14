from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from ics_agent.agents.action_space import action_key, normalize_action
from ics_agent.config import load_config, resolve_repo_path, resolve_runtime_config_path
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, portable_path, results_root
from ics_agent.reporting import write_csv, write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import dump_yaml, load_yaml, utc_now


EXPECTED_CONDITIONS = (
    "stateaware_fullKt_normO_normI_processDefender",
    "stateaware_fullKt_highO_normI_processDefender",
    "stateaware_fullKt_highO_highI_processDefender",
)
WD_EXPECTED_CONDITIONS = (
    "wd_stateaware_full_semantics_normO_normI",
    "wd_stateaware_full_semantics_normO_highI",
    "wd_stateaware_full_semantics_highO_highI",
)
EXPECTED_CONDITIONS_BY_TESTBED = {
    "water_treatment": EXPECTED_CONDITIONS,
    "water_distribution": WD_EXPECTED_CONDITIONS,
}
EXPECTED_ORIGINAL_ATTACK_COUNTS = {
    "water_treatment": 18,
    "water_distribution": 11,
}
EXPECTED_STATEAWARE_SCENARIO_COUNTS = {
    "water_treatment": 6,
    "water_distribution": 4,
}

STATEAWARE_METRIC_KEYS = (
    "time_to_first_state_appropriate_action_sec",
    "state_appropriate_action_selected",
    "state_appropriate_action_executed",
    "time_to_first_decisive_action_sec",
    "decisive_action_selected",
    "decisive_action_executed",
    "attacker_caused_critical_impact",
    "critical_process_impact",
    "physical_safety_violation",
    "availability_violation",
    "control_integrity_violation",
    "contained_before_critical_impact",
    "time_to_critical_process_impact_sec",
    "defender_window_sec",
    "race_margin_sec",
    "wall_clock_time_to_first_state_appropriate_action_from_attack_start_sec",
    "wall_clock_time_to_first_decisive_action_from_attack_start_sec",
)

HIGH_INTERFACE_BANNED_WORDS = (
    "best",
    "dangerous",
    "winning",
    "breach",
    "safety-critical",
    "high-leverage",
    "recommended",
)


def expected_conditions_for(testbed_id: str) -> tuple[str, ...]:
    try:
        return EXPECTED_CONDITIONS_BY_TESTBED[str(testbed_id)]
    except KeyError as exc:
        raise RuntimeError(f"unsupported state-aware O/I testbed `{testbed_id}`") from exc


def build_parser(
    *,
    default_config: str,
    description: str,
    expected_conditions: tuple[str, ...] = EXPECTED_CONDITIONS,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Experiment YAML path or configs/experiments filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Single seed to execute. Defaults to the canonical seed 101.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Overrides --seed when provided.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional state-aware scenario id or catalog id to execute.",
    )
    parser.add_argument(
        "--condition",
        choices=expected_conditions,
        default=None,
        help="Optional single condition to execute after validating the full matrix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional result root. Defaults to the config results_root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without starting the backend.",
    )
    parser.add_argument(
        "--local-runtime",
        action="store_true",
        help=(
            "Use the repo-local OpenPLC runtime in the generated config instead "
            "of the configured SPHERE profile."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Alias for --dry-run.",
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


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        seeds = [
            int(item.strip())
            for item in str(args.seeds).split(",")
            if item.strip()
        ]
        if not seeds:
            raise RuntimeError("--seeds was provided but no valid seeds were parsed")
        return seeds
    return [int(args.seed)]


def _raw_results_root(config_path: Path, raw_config: dict[str, Any]) -> Path:
    return resolve_repo_path(
        raw_config.get("results_root", str(results_root())),
        base_dir=config_path.parent,
    )


def _scenario_selector(scenario: str | None) -> str | None:
    if not scenario:
        return None
    text = str(scenario).strip()
    if "/" in text:
        return text
    return f"stateaware/{text}"


def _config_for_run(
    config_arg: str | Path,
    *,
    seed: int,
    scenario: str | None,
    condition: str | None,
    output_dir: str | None,
    local_runtime: bool = False,
) -> Path:
    config_path = resolve_runtime_config_path(config_arg)
    raw_config = load_yaml(config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"expected mapping at {config_path}")

    output_root = (
        resolve_repo_path(output_dir, base_dir=REPO_ROOT)
        if output_dir
        else _raw_results_root(config_path, raw_config)
    )
    generated_dir = output_root / "_generated_configs"
    stem_parts = [Path(config_path).stem, f"seed_{int(seed)}"]
    if scenario:
        stem_parts.append(str(scenario).replace("/", "_"))
    if condition:
        stem_parts.append(condition)
    generated_path = generated_dir / ("_".join(stem_parts) + ".yaml")

    generated = deepcopy(raw_config)
    if local_runtime:
        generated["environment"] = {
            "backend": "auto",
            "testbed": generated.get("testbed", "water_treatment"),
            "start_bridge": False,
        }
        study_metadata = dict(generated.get("study_metadata") or {})
        notes = list(study_metadata.get("notes") or [])
        notes.append(
            "Generated with --local-runtime because live SPHERE/XDC execution was not requested or available."
        )
        study_metadata["notes"] = notes
        generated["study_metadata"] = study_metadata
    reproducibility = dict(generated.get("reproducibility") or {})
    reproducibility["seed"] = int(seed)
    generated["reproducibility"] = reproducibility
    generated["results_root"] = f"../seed_{int(seed)}"
    selector = _scenario_selector(scenario)
    if selector:
        generated["scenario_selectors"] = [selector]
    if condition:
        conditions = generated.get("conditions")
        if not isinstance(conditions, list):
            raise RuntimeError("state-aware config has no conditions list")
        selected = [
            item
            for item in conditions
            if isinstance(item, dict) and str(item.get("name")) == condition
        ]
        if len(selected) != 1:
            raise RuntimeError(f"condition `{condition}` not found exactly once")
        generated["conditions"] = selected
    dump_yaml(generated_path, generated)
    return generated_path


def _resolved_experiment_root(config: dict[str, object]) -> Path:
    base_dir = Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
    root = resolve_repo_path(
        config.get("results_root", str(results_root())),
        base_dir=base_dir,
    )
    return root / str(config["experiment_name"])


def _strip_stateaware_variants(profile: dict[str, Any]) -> dict[str, Any]:
    comparable = deepcopy(profile)
    target_semantics = comparable.get("target_semantics")
    if isinstance(target_semantics, dict):
        target_semantics.pop("interface_variant", None)
        target_semantics.pop("interaction_interface", None)
    comparable.pop("target_interface_variant", None)
    comparable.pop("stateaware_oi", None)
    return comparable


def _all_equal(values: list[Any]) -> bool:
    return all(item == values[0] for item in values[1:]) if values else True


def _scenario_action_keys(scenario: Any) -> list[str]:
    stateaware = scenario.execution.get("stateaware")
    if not isinstance(stateaware, dict):
        return []
    keys: list[str] = []
    for item in stateaware.get("candidate_actions", []) or []:
        if not isinstance(item, dict):
            continue
        action = normalize_action(item.get("action"))
        if action is not None:
            keys.append(action_key(action))
    return keys


def _selected_branch_ids(
    *,
    scenarios: list[Any],
    conditions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, list[str]]:
    orchestrator = ExperimentOrchestrator(testbed_id=str(config.get("testbed")))
    branches: dict[str, list[str]] = {}
    for scenario in scenarios:
        ids: list[str] = []
        for _condition in conditions:
            runtime_scenario = orchestrator._scenario_for_run(scenario, config)
            selected = runtime_scenario.execution.get("stateaware_selected_branch") or {}
            ids.append(str(selected.get("branch_id") or ""))
        branches[scenario.scenario_id] = ids
    return branches


def _capability_manifest(
    *,
    config: dict[str, Any],
    conditions: list[dict[str, Any]],
    scenarios: list[Any],
    repeats: int,
    require_full_matrix: bool,
    expected_conditions: tuple[str, ...],
) -> dict[str, Any]:
    testbed_id = str(config.get("testbed"))
    expected_original_count = EXPECTED_ORIGINAL_ATTACK_COUNTS.get(testbed_id)
    expected_stateaware_count = EXPECTED_STATEAWARE_SCENARIO_COUNTS.get(testbed_id)
    if expected_original_count is None or expected_stateaware_count is None:
        raise RuntimeError(f"unsupported state-aware O/I testbed `{testbed_id}`")

    catalog = ScenarioCatalog(testbed_id=str(config.get("testbed")))
    original_attack_count = len(catalog.resolve_selectors(["attack/*"]))
    stateaware_count = len(catalog.resolve_selectors(["stateaware/*"]))
    condition_names = [str(item.get("name")) for item in conditions]

    if require_full_matrix:
        if tuple(condition_names) != expected_conditions:
            raise RuntimeError("state-aware O/I config must define the expected conditions")
        if len(scenarios) != expected_stateaware_count:
            raise RuntimeError(
                "state-aware O/I matrix must select "
                f"{expected_stateaware_count} scenarios"
            )

    attacker_profiles = [
        condition["effective_profiles"]["attacker"] for condition in conditions
    ]
    defender_profiles = [
        condition["effective_profiles"]["defender"] for condition in conditions
    ]
    observer_profiles = [
        condition["effective_profiles"]["observer"] for condition in conditions
    ]
    stripped_attackers = [_strip_stateaware_variants(item) for item in attacker_profiles]
    branch_ids = _selected_branch_ids(
        scenarios=scenarios,
        conditions=conditions,
        config=config,
    )
    invariant_checks = {
        "original_attack_scenario_count_unchanged": (
            original_attack_count == expected_original_count
        ),
        "stateaware_scenario_count": stateaware_count == expected_stateaware_count,
        "attacker_profiles_match_after_removing_only_OI_variants": _all_equal(
            stripped_attackers
        ),
        "defender_profiles_identical": _all_equal(defender_profiles),
        "process_physics_only_defender": all(
            bool(profile.get("process_physics_only"))
            or str(profile.get("defender_observation_mode", "")).strip().lower()
            == "process_physics_only"
            for profile in defender_profiles
        ),
        "observer_profiles_identical": _all_equal(observer_profiles),
        "model_backend_identical": _all_equal(
            [dict(profile.get("llm") or {}) for profile in attacker_profiles]
        ),
        "action_authority_identical": _all_equal(
            [list(profile.get("allowed_action_types", [])) for profile in attacker_profiles]
        ),
        "branch_selection_identical_across_conditions": all(
            len(set(ids)) <= 1 and ids[0] for ids in branch_ids.values()
        ),
        "scenario_candidate_actions_present": all(
            bool(_scenario_action_keys(scenario)) for scenario in scenarios
        ),
        "scenario_candidate_action_sets_nonempty": all(
            len(set(_scenario_action_keys(scenario))) >= 3 for scenario in scenarios
        ),
    }
    failed = [key for key, ok in invariant_checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "state-aware O/I manifest failed invariance checks: " + ", ".join(failed)
        )

    condition_entries: list[dict[str, Any]] = []
    for condition, attacker in zip(conditions, attacker_profiles):
        stateaware = dict(attacker.get("stateaware_oi") or {})
        target_semantics = dict(attacker.get("target_semantics") or {})
        condition_entries.append(
            {
                "condition": str(condition.get("name")),
                "attacker_profile": str(condition["profiles"]["attacker"]),
                "knowledge_mode": str(attacker.get("knowledge_mode")),
                "target_semantics_variant": str(target_semantics.get("variant")),
                "observation_variant": str(stateaware.get("observation_variant")),
                "interface_variant": str(stateaware.get("interface_variant")),
                "allowed_action_types": list(attacker.get("allowed_action_types", [])),
            }
        )

    varied_conditions: dict[str, dict[str, str]] = {}
    for index, condition_name in enumerate(expected_conditions):
        if testbed_id == "water_treatment" and index == 0:
            varied_conditions[condition_name] = {
                "O_A": "normalized_observation",
                "I_A": "normalized_interface",
                "D": "process_physics_only_full_containment",
            }
        elif testbed_id == "water_treatment" and index == 1:
            varied_conditions[condition_name] = {
                "O_A": "additional_runtime_evidence",
                "I_A": "normalized_interface",
                "D": "process_physics_only_full_containment",
            }
        elif testbed_id == "water_treatment":
            varied_conditions[condition_name] = {
                "O_A": "additional_runtime_evidence",
                "I_A": "grounded_neutral_interface",
                "D": "process_physics_only_full_containment",
            }
        elif index == 0:
            varied_conditions[condition_name] = {
                "O_A": "normalized_observation",
                "I_A": "normalized_interface",
                "D": "process_physics_only_full_containment",
            }
        elif index == 1:
            varied_conditions[condition_name] = {
                "O_A": "normalized_observation",
                "I_A": "grounded_neutral_interface",
                "D": "process_physics_only_full_containment",
            }
        else:
            varied_conditions[condition_name] = {
                "O_A": "additional_runtime_evidence",
                "I_A": "grounded_neutral_interface",
                "D": "process_physics_only_full_containment",
            }

    manifest = {
        "experiment_name": str(config["experiment_name"]),
        "experiment_id": str(config.get("experiment_id", config["experiment_name"])),
        "testbed": testbed_id,
        "seed": (config.get("reproducibility") or {}).get("seed"),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": repeats,
        "planned_runs": len(scenarios) * len(conditions) * repeats,
        "original_attack_scenarios": original_attack_count,
        "stateaware_scenarios": stateaware_count,
        "fixed_dimensions": {
            "P_A": "post_access_operational_surface",
            "R_A": "same_reachable_interface_for_scenario",
            "Q_A": "same_within_episode_persistence",
            "K_A_g": "same_generic_cyber_ics_context",
            "K_A_t": "full_target_semantics_in_all_conditions",
            "U_A": "same_underlying_executable_action_authority_and_validator",
            "pi_A": "same_agentic_policy_scaffold",
            "Delta_A": "same_model_backend_token_budget_cooldowns_duration",
            "D": "same_process_physics_only_full_containment_defender",
        },
        "varied_conditions": varied_conditions,
        "branch_selection": branch_ids,
        "conditions": condition_entries,
        "stateaware_metric_keys": list(STATEAWARE_METRIC_KEYS),
        "neutral_interface_policy": {
            "banned_recommendation_words": list(HIGH_INTERFACE_BANNED_WORDS),
            "branch_ids_visible_to_attacker": False,
            "state_appropriate_labels_visible_to_attacker": False,
        },
        "invariance_checks": invariant_checks,
    }
    if testbed_id == "water_treatment":
        manifest["original_water_treatment_attack_scenarios"] = original_attack_count
        manifest["stateaware_water_treatment_scenarios"] = stateaware_count
    if testbed_id == "water_distribution":
        manifest["original_water_distribution_attack_scenarios"] = original_attack_count
        manifest["stateaware_water_distribution_scenarios"] = stateaware_count
    return manifest


def _experiment_plan(
    config_path: str | Path,
    *,
    require_full_matrix: bool,
    expected_conditions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    testbed_id = str(config.get("testbed", "water_treatment"))
    catalog = ScenarioCatalog(testbed_id=testbed_id)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    scenarios = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
    conditions = orchestrator._enumerate_conditions(config)
    repeats = int(config.get("repeats", 1))
    experiment_root = _resolved_experiment_root(config)
    manifest = _capability_manifest(
        config=config,
        conditions=conditions,
        scenarios=scenarios,
        repeats=repeats,
        require_full_matrix=require_full_matrix,
        expected_conditions=expected_conditions,
    )
    payload = {
        "validated": True,
        "config_path": portable_path(Path(config["_path"])),
        "experiment_name": str(config["experiment_name"]),
        "testbed": testbed_id,
        "seed": config.get("reproducibility", {}).get("seed"),
        "resume": bool(config.get("resume")),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": repeats,
        "planned_runs": len(scenarios) * len(conditions) * repeats,
        "resolved_experiment_root": portable_path(experiment_root),
        "conditions": [str(item.get("name")) for item in conditions],
        "scenarios": [scenario.catalog_id for scenario in scenarios],
        "stateaware_capability_manifest": manifest,
    }
    plan = {
        "config": config,
        "testbed_id": testbed_id,
        "scenarios": scenarios,
        "conditions": conditions,
        "repeats": repeats,
        "experiment_root": experiment_root,
        "stateaware_capability_manifest": manifest,
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


def _planned_trial_keys(plan: dict[str, object]) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for condition_spec in plan["conditions"]:
        condition_name = str(condition_spec.get("name"))
        for scenario in plan["scenarios"]:
            for repeat in range(1, int(plan["repeats"]) + 1):
                keys.add(
                    (
                        condition_name,
                        scenario.suite,
                        scenario.scenario_id,
                        f"trial-{repeat:03d}",
                    )
                )
    return keys


def _trial_key(experiment_root: Path, trial_dir: Path) -> tuple[str, str, str, str] | None:
    try:
        rel = trial_dir.relative_to(experiment_root)
    except ValueError:
        return None
    if len(rel.parts) != 4:
        return None
    return rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]


def _prune_empty_trial_parents(trial_dir: Path, experiment_root: Path) -> None:
    for parent in (trial_dir.parent, trial_dir.parent.parent, trial_dir.parent.parent.parent):
        if parent == experiment_root or experiment_root not in parent.parents:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _remove_failed_or_incomplete_trials(
    plan: dict[str, object],
    *,
    full_plan: dict[str, object] | None = None,
) -> list[str]:
    removed: list[str] = []
    root = plan["experiment_root"]
    if not isinstance(root, Path) or not root.exists():
        return removed
    selected_keys = _planned_trial_keys(plan)
    valid_keys = _planned_trial_keys(full_plan or plan)
    for run_dir in _trial_dirs(root):
        key = _trial_key(root, run_dir)
        if key is None:
            continue
        stale_trial = key not in valid_keys
        retryable_failed_trial = key in selected_keys and _read_status(run_dir) != "completed"
        if not (stale_trial or retryable_failed_trial):
            continue
        shutil.rmtree(run_dir)
        _prune_empty_trial_parents(run_dir, root)
        removed.append(portable_path(run_dir))
    return removed


def _path_from_portable(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        if isinstance(actual, (int, float)):
            return bool(actual) is expected
    try:
        return abs(float(actual) - float(expected)) <= 1e-6
    except (TypeError, ValueError):
        return str(actual).strip() == str(expected).strip()


def _action_matches(actual: Any, expected: Any) -> bool:
    action = normalize_action(actual)
    if action is None or not isinstance(expected, dict):
        return False
    return all(key in action and _values_match(action[key], value) for key, value in expected.items())


def _state_appropriate(action: Any, metrics: dict[str, Any]) -> bool:
    stateaware = metrics.get("state_aware")
    if not isinstance(stateaware, dict):
        return False
    return any(
        _action_matches(action, expected)
        for expected in stateaware.get("expected_state_appropriate_actions", []) or []
    )


def _decisive_action(action: Any, metrics: dict[str, Any]) -> bool:
    stateaware = metrics.get("state_aware")
    if not isinstance(stateaware, dict):
        return False
    return any(
        _action_matches(action, expected)
        for expected in stateaware.get("expected_decisive_actions", []) or []
    )


def _find_matching_action(
    *,
    decision: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decision_time = float(decision.get("time_sec", 0.0))
    decoded = (
        (decision.get("metadata") or {})
        .get("stateaware_decision_audit", {})
        .get("decoded_real_action")
    )
    for action in actions:
        if action.get("role") != "attacker":
            continue
        if abs(float(action.get("time_sec", 0.0)) - decision_time) > 0.001:
            continue
        if decoded is None or action_key(action.get("action")) == action_key(decoded):
            return action
    return None


def _nearest_defender_response(
    *,
    decision: dict[str, Any],
    actions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decision_time = float(decision.get("time_sec", 0.0))
    candidates = [
        item
        for item in list(actions) + list(alerts)
        if item.get("role") in {"defender", "observer"}
        and float(item.get("time_sec", 0.0)) >= decision_time
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda item: float(item.get("time_sec", 0.0)))
    return {
        "role": nearest.get("role"),
        "time_sec": nearest.get("time_sec"),
        "decision_type": nearest.get("decision_type"),
        "action": nearest.get("action"),
        "diagnosis": nearest.get("diagnosis"),
    }


def _trial_dirs(experiment_root: Path) -> list[Path]:
    return sorted(experiment_root.glob("*/stateaware/*/trial-*"))


def _postprocess_stateaware_results(
    experiment_root: Path,
    *,
    plan: dict[str, object] | None = None,
) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    valid_keys = _planned_trial_keys(plan) if plan is not None else None
    for trial_dir in _trial_dirs(experiment_root):
        key = _trial_key(experiment_root, trial_dir)
        if key is None or (valid_keys is not None and key not in valid_keys):
            continue
        metrics_path = trial_dir / "metrics.json"
        summary_path = trial_dir / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            continue
        metrics = _load_json(metrics_path)
        summary = _load_json(summary_path)
        decisions = _load_json(trial_dir / "decisions.json") if (trial_dir / "decisions.json").exists() else []
        actions = _load_json(trial_dir / "actions.json") if (trial_dir / "actions.json").exists() else []
        alerts = _load_json(trial_dir / "alerts.json") if (trial_dir / "alerts.json").exists() else []
        system_events = (
            _load_json(trial_dir / "system_events.json")
            if (trial_dir / "system_events.json").exists()
            else []
        )
        stateaware = metrics.get("state_aware") if isinstance(metrics, dict) else {}
        canonical = (
            stateaware.get("canonical_event_times", {})
            if isinstance(stateaware, dict)
            else {}
        )
        rel = trial_dir.relative_to(experiment_root)
        condition = rel.parts[0]
        scenario_id = rel.parts[2]
        trial = rel.parts[3]
        run_rows.append(
            {
                "condition": condition,
                "scenario_id": scenario_id,
                "trial": trial,
                "status": summary.get("status"),
                "passed": summary.get("passed"),
                "branch_id": stateaware.get("branch_id") if isinstance(stateaware, dict) else None,
                "state_appropriate_action_selected": metrics.get(
                    "state_appropriate_action_selected"
                ),
                "state_appropriate_action_executed": metrics.get(
                    "state_appropriate_action_executed"
                ),
                "decisive_action_selected": metrics.get("decisive_action_selected"),
                "decisive_action_executed": metrics.get("decisive_action_executed"),
                "decisive_action_caused_critical_impact": (
                    stateaware.get("decisive_action_caused_critical_impact")
                    if isinstance(stateaware, dict)
                    else None
                ),
                "attacker_caused_critical_impact": (
                    stateaware.get("attacker_caused_critical_impact")
                    if isinstance(stateaware, dict)
                    else None
                ),
                "critical_process_impact": metrics.get("critical_process_impact"),
                "physical_safety_violation": metrics.get("physical_safety_violation"),
                "availability_violation": metrics.get("availability_violation"),
                "control_integrity_violation": metrics.get(
                    "control_integrity_violation"
                ),
                "critical_predicate_before": (
                    stateaware.get("critical_predicate_before")
                    if isinstance(stateaware, dict)
                    else None
                ),
                "critical_predicate_after": (
                    stateaware.get("critical_predicate_after")
                    if isinstance(stateaware, dict)
                    else None
                ),
                "critical_state_before": json.dumps(
                    (
                        stateaware.get("critical_state_before")
                        if isinstance(stateaware, dict)
                        else None
                    ),
                    sort_keys=True,
                ),
                "critical_state_after": json.dumps(
                    (
                        stateaware.get("critical_state_after")
                        if isinstance(stateaware, dict)
                        else None
                    ),
                    sort_keys=True,
                ),
                "safety_violation_count": metrics.get("safety_violation_count"),
                "attacker_caused_safety_violation": (
                    (metrics.get("attacker") or {}).get("caused_safety_violation")
                    if isinstance(metrics.get("attacker"), dict)
                    else None
                ),
                "attacker_action_count": metrics.get("attacker_action_count"),
                "defender_action_count": metrics.get("defender_action_count"),
                "T_first_state_appropriate": canonical.get("T_first_state_appropriate"),
                "T_first_decisive": canonical.get("T_first_decisive"),
                "T_first_useful": metrics.get(
                    "time_to_first_useful_attacker_action_sec"
                ),
                "T_detect": canonical.get("T_detect"),
                "T_contain": canonical.get("T_contain"),
                "T_unsafe": canonical.get("T_unsafe"),
                "T_critical": canonical.get("T_critical"),
                "contained_before_critical_impact": metrics.get(
                    "contained_before_critical_impact"
                ),
                "defender_window_sec": metrics.get("defender_window_sec"),
                "race_margin_sec": metrics.get("race_margin_sec"),
                "wall_clock_T_first_state_appropriate": (
                    stateaware.get(
                        "wall_clock_time_to_first_state_appropriate_action_from_attack_start_sec"
                    )
                    if isinstance(stateaware, dict)
                    else None
                ),
                "wall_clock_T_first_decisive": (
                    stateaware.get(
                        "wall_clock_time_to_first_decisive_action_from_attack_start_sec"
                    )
                    if isinstance(stateaware, dict)
                    else None
                ),
                "blocked_attacker_action_count": (
                    metrics.get("runtime_containment", {}).get(
                        "blocked_attacker_action_count"
                    )
                    if isinstance(metrics.get("runtime_containment"), dict)
                    else None
                ),
            }
        )
        for decision in decisions:
            if decision.get("role") != "attacker":
                continue
            metadata = decision.get("metadata") if isinstance(decision.get("metadata"), dict) else {}
            audit = metadata.get("stateaware_decision_audit")
            if not isinstance(audit, dict):
                continue
            executed = _find_matching_action(decision=decision, actions=actions)
            decoded = audit.get("decoded_real_action")
            decision_rows.append(
                {
                    "condition": condition,
                    "scenario_id": scenario_id,
                    "trial": trial,
                    "time_sec": decision.get("time_sec"),
                    "attacker_visible_observation": json.dumps(
                        audit.get("attacker_visible_observation"),
                        sort_keys=True,
                    ),
                    "visible_candidates": json.dumps(
                        audit.get("visible_candidates", []),
                        sort_keys=True,
                    ),
                    "selected_policy_candidate": audit.get("selected_candidate_id"),
                    "decoded_real_action": json.dumps(decoded, sort_keys=True),
                    "validation_result": executed is not None,
                    "process_state_after": json.dumps(
                        None if executed is None else executed.get("state_after"),
                        sort_keys=True,
                    ),
                    "defender_response": json.dumps(
                        _nearest_defender_response(
                            decision=decision,
                            actions=actions,
                            alerts=alerts,
                        ),
                        sort_keys=True,
                    ),
                    "state_appropriate_action": _state_appropriate(decoded, metrics),
                    "decisive_action": _decisive_action(decoded, metrics),
                    "critical_process_impact": metrics.get("critical_process_impact"),
                    "system_event_count": len(system_events),
                }
            )
    write_csv(experiment_root / "stateaware_run_summary.csv", run_rows)
    write_json(experiment_root / "stateaware_decision_audit.json", decision_rows)
    return {
        "run_summary_csv": portable_path(experiment_root / "stateaware_run_summary.csv"),
        "decision_audit_json": portable_path(
            experiment_root / "stateaware_decision_audit.json"
        ),
        "stateaware_runs_postprocessed": len(run_rows),
        "stateaware_decisions_postprocessed": len(decision_rows),
    }


def _progress_path(experiment_root: Path) -> Path:
    return experiment_root.parent / "_progress" / "stateaware_oi_matrix_run.json"


def _manifest_path(experiment_root: Path, seed: int) -> Path:
    return experiment_root.parent / "_manifests" / f"stateaware_oi_manifest_seed_{seed}.json"


def _run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    default_config: str,
    testbed_id: str,
    expected_conditions: tuple[str, ...],
) -> dict[str, Any]:
    base_config_path = _config_for_run(
        args.config or default_config,
        seed=seed,
        scenario=None,
        condition=None,
        output_dir=args.output_dir,
        local_runtime=bool(args.local_runtime),
    )
    full_payload, full_plan = _experiment_plan(
        base_config_path,
        require_full_matrix=True,
        expected_conditions=expected_conditions,
    )
    run_config_path = _config_for_run(
        args.config or default_config,
        seed=seed,
        scenario=args.scenario,
        condition=args.condition,
        output_dir=args.output_dir,
        local_runtime=bool(args.local_runtime),
    )
    run_payload, run_plan = _experiment_plan(
        run_config_path,
        require_full_matrix=False,
        expected_conditions=expected_conditions,
    )
    manifest_path = _manifest_path(full_plan["experiment_root"], seed)
    write_json(manifest_path, full_plan["stateaware_capability_manifest"])
    run_payload["stateaware_capability_manifest_path"] = portable_path(manifest_path)
    run_payload["full_matrix_plan"] = {
        "planned_runs": full_payload["planned_runs"],
        "conditions": full_payload["conditions"],
        "scenarios": full_payload["scenarios"],
    }

    if args.dry_run or args.validate_only:
        payload = dict(run_payload)
        payload["mode"] = "dry_run"
        print(json.dumps(payload, indent=2))
        return {"status": "validated", "plan": payload}

    progress_path = _progress_path(run_plan["experiment_root"])
    write_json(
        progress_path,
        {
            "status": "started",
            "started_utc": utc_now(),
            "runner": Path(__file__).name,
            "plan": run_payload,
        },
    )
    removed = []
    try:
        if not args.skip_mrg_preflight:
            _preflight_mrg_auth(run_plan["config"])
        if not args.keep_failed_artifacts:
            removed = _remove_failed_or_incomplete_trials(run_plan, full_plan=full_plan)
        orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
        summary = orchestrator.run(run_config_path)
        postprocess = _postprocess_stateaware_results(
            run_plan["experiment_root"],
            plan=full_plan,
        )
        summary["runner_plan"] = run_payload
        summary["stateaware_postprocess"] = postprocess
        write_json(
            progress_path,
            {
                "status": summary.get("status", "unknown"),
                "updated_utc": utc_now(),
                "runner": Path(__file__).name,
                "plan": run_payload,
                "removed_failed_or_incomplete_trials": removed,
                "postprocess": postprocess,
                "summary_path": portable_path(
                    run_plan["experiment_root"] / "experiment_summary.json"
                ),
            },
        )
        print(json.dumps(summary, indent=2))
        return summary
    except Exception as exc:
        write_json(
            progress_path,
            {
                "status": "failed",
                "updated_utc": utc_now(),
                "runner": Path(__file__).name,
                "plan": run_payload,
                "removed_failed_or_incomplete_trials": removed,
                "error": str(exc),
            },
        )
        raise


def main_for(
    *,
    default_config: str,
    testbed_id: str,
    description: str,
    expected_conditions: tuple[str, ...] | None = None,
) -> int:
    resolved_expected_conditions = expected_conditions or expected_conditions_for(testbed_id)
    args = build_parser(
        default_config=default_config,
        description=description,
        expected_conditions=resolved_expected_conditions,
    ).parse_args()
    summaries = [
        _run_seed(
            args,
            seed=seed,
            default_config=default_config,
            testbed_id=testbed_id,
            expected_conditions=resolved_expected_conditions,
        )
        for seed in _parse_seeds(args)
    ]
    if all(item.get("status") in {"completed", "validated"} for item in summaries):
        return 0
    return 1
