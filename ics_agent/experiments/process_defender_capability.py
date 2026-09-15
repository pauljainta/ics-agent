from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from ics_agent.agents.action_space import action_key, normalize_action
from ics_agent.config import load_config, resolve_repo_path, resolve_runtime_config_path
from ics_agent.experiments.stateaware_oi_matrix import _preflight_mrg_auth
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, portable_path, results_root
from ics_agent.reporting import write_csv, write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import dump_yaml, load_yaml, stable_hash, utc_now


TESTBED_ALIASES = {
    "wt": "water_treatment",
    "treatment": "water_treatment",
    "water_treatment": "water_treatment",
    "wd": "water_distribution",
    "distribution": "water_distribution",
    "water_distribution": "water_distribution",
}

SUITE_ALIASES = {
    "kt": "kt",
    "target-semantic": "kt",
    "target_semantic": "kt",
    "target_semantics": "kt",
    "semantics": "kt",
    "interface": "interface",
    "target-interface": "interface",
    "target_interface": "interface",
    "oi": "observation",
    "observation": "observation",
    "high-o": "observation",
    "high_o": "observation",
    "all": "all-capability",
    "all-capability": "all-capability",
    "all_capability": "all-capability",
}

SUITE_CONFIGS = {
    ("water_treatment", "kt"): (
        "water_treatment_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
    ),
    ("water_distribution", "kt"): (
        "water_distribution_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
    ),
    ("water_treatment", "interface"): (
        "water_treatment_target_interface_ablation_vs_agentic_full_containment_defender.yaml"
    ),
    ("water_distribution", "interface"): (
        "water_distribution_target_interface_ablation_vs_agentic_full_containment_defender.yaml"
    ),
}

SUITE_DIRS = {
    "kt": "target-semantic-ablation",
    "interface": "target-interface-ablation",
    "observation": "target-observation-ablation",
}

EXPECTED_ATTACK_SCENARIO_COUNTS = {
    "water_treatment": 18,
    "water_distribution": 11,
}

METRIC_KEYS_TO_EXPECT = [
    "wall_clock_time_to_first_useful_attacker_action_from_attack_start_sec",
    "wall_clock_time_to_detection_from_attack_start_sec",
    "wall_clock_time_to_first_defender_action_from_attack_start_sec",
    "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec",
    "wall_clock_time_to_first_safety_violation_from_attack_start_sec",
    "attacker_effect_capable_action_count",
    "attacker_ineffective_action_count",
    "distinct_surfaces_probed_count",
    "distinct_surfaces_probed_before_first_useful_count",
    "physical_impact",
    "severity",
    "downtime_timestep_count",
    "attacker_execution_cutoff_pre_attack_start",
    "invalid_precontained_run",
]

DEFENDER_PROCESS_ONLY_PATCH = {
    "process_physics_only": True,
    "defender_observation_mode": "process_physics_only",
    "defer_containment_until_attacker_effect": True,
}

def _canonical_testbed(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in TESTBED_ALIASES:
        valid = ", ".join(sorted(TESTBED_ALIASES))
        raise RuntimeError(f"unsupported testbed `{value}`; valid values: {valid}")
    return TESTBED_ALIASES[key]


def _canonical_suite(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    key = key.replace("target-semantics", "target-semantic")
    if key not in SUITE_ALIASES:
        valid = ", ".join(sorted(SUITE_ALIASES))
        raise RuntimeError(f"unsupported suite `{value}`; valid values: {valid}")
    return SUITE_ALIASES[key]


def _testbed_result_root(testbed_id: str) -> Path:
    if testbed_id == "water_treatment":
        return results_root() / "Water-treatment"
    if testbed_id == "water_distribution":
        return results_root() / "water_distribution_openplc"
    raise RuntimeError(f"unsupported testbed `{testbed_id}`")


def _default_config_for(testbed_id: str, suite: str) -> str | None:
    return SUITE_CONFIGS.get((testbed_id, suite))


def _suite_root(testbed_id: str, suite: str) -> Path:
    return (
        _testbed_result_root(testbed_id)
        / "process-defender-critical-impact-v2"
        / SUITE_DIRS[suite]
    )


def _target_semantics_without_variant(profile: dict[str, Any]) -> dict[str, Any]:
    comparable = deepcopy(profile)
    comparable.pop("target_semantics", None)
    return comparable


def _target_semantics_without_interface(profile: dict[str, Any]) -> dict[str, Any]:
    comparable = deepcopy(profile)
    target_semantics = comparable.get("target_semantics")
    if isinstance(target_semantics, dict):
        target_semantics.pop("interface_variant", None)
        target_semantics.pop("interaction_interface", None)
    comparable.pop("target_interface_variant", None)
    return comparable


def _all_equal(values: list[Any]) -> bool:
    return all(item == values[0] for item in values[1:]) if values else True


def _notes_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _patch_defender_profile(profile: dict[str, Any]) -> dict[str, Any]:
    patched = deepcopy(profile)
    patched.update(DEFENDER_PROCESS_ONLY_PATCH)
    behavior_model = dict(patched.get("behavior_model") or {})
    behavior_model["response_mode"] = "active_intervention"
    behavior_model["defender_observation_contract"] = "process_physics_only"
    notes = _notes_list(behavior_model.get("notes"))
    note = (
        "Harmonized process-aware defender: process/controller evidence only; "
        "containment may take effect only after attacker process effect is observed."
    )
    if note not in notes:
        notes.append(note)
    behavior_model["notes"] = notes
    patched["behavior_model"] = behavior_model
    return patched


def _patch_condition_defender(condition: dict[str, Any]) -> dict[str, Any]:
    patched = deepcopy(condition)
    overrides = dict(patched.get("profile_overrides") or {})
    defender = dict(overrides.get("defender") or {})
    overrides["defender"] = _patch_defender_profile(defender)
    patched["profile_overrides"] = overrides
    return patched


def _patch_config(
    *,
    raw_config: dict[str, Any],
    source_config_path: Path,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    only_condition: str | None,
) -> dict[str, Any]:
    generated = deepcopy(raw_config)
    base_name = str(generated.get("experiment_name") or source_config_path.stem)
    generated["experiment_name"] = f"{base_name}_process_defender_critical_impact_v2"
    generated["experiment_id"] = (
        f"{generated['experiment_name']}_seed_{int(seed)}_"
        f"{stable_hash({'source': portable_path(source_config_path), 'seed': int(seed)}, length=6)}"
    )
    generated["testbed"] = testbed_id
    generated["resume"] = True
    generated["use_canonical_result_root"] = True
    generated["results_root"] = f"../seed_{int(seed)}"

    reproducibility = dict(generated.get("reproducibility") or {})
    reproducibility["seed"] = int(seed)
    generated["reproducibility"] = reproducibility

    if scenario:
        selector = str(scenario).strip()
        generated["scenario_selectors"] = [
            selector if "/" in selector else f"attack/{selector}"
        ]

    conditions = generated.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError(f"expected non-empty conditions list in {source_config_path}")
    patched_conditions = [_patch_condition_defender(item) for item in conditions]
    if only_condition:
        selected = [
            item
            for item in patched_conditions
            if isinstance(item, dict) and str(item.get("name")) == only_condition
        ]
        if len(selected) != 1:
            raise RuntimeError(f"condition `{only_condition}` not found exactly once")
        patched_conditions = selected
    generated["conditions"] = patched_conditions

    if isinstance(generated.get("frozen_defender_override"), dict):
        generated["frozen_defender_override"] = _patch_defender_profile(
            generated["frozen_defender_override"]
        )

    study_metadata = dict(generated.get("study_metadata") or {})
    notes = _notes_list(study_metadata.get("notes"))
    process_note = (
        "Generated process-defender-critical-impact-v2 rerun. Attacker-side "
        "capability matrix is preserved from the legacy source config; only the "
        "defender observation/chronology contract is harmonized."
    )
    if process_note not in notes:
        notes.append(process_note)
    study_metadata.update(
        {
            "family": f"{study_metadata.get('family', base_name)}_process_defender_critical_impact_v2",
            "legacy_source_config": portable_path(source_config_path),
            "capability_suite": suite,
            "defender_observation_contract": "process_physics_only",
            "critical_impact_audit": "legacy_safety_bounds_proxy_unless_stateaware_contract_present",
            "notes": notes,
        }
    )
    generated["study_metadata"] = study_metadata
    return generated


def _raw_results_root(config_path: Path, raw_config: dict[str, Any]) -> Path:
    return resolve_repo_path(
        raw_config.get("results_root", str(results_root())),
        base_dir=config_path.parent,
    )


def _generated_config_path(
    *,
    source_config_path: Path,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    only_condition: str | None,
) -> Path:
    root = _suite_root(testbed_id, suite)
    parts = [source_config_path.stem, "process_defender", f"seed_{int(seed)}"]
    if scenario:
        parts.append(str(scenario).replace("/", "_").replace(" ", "_"))
    if only_condition:
        parts.append(str(only_condition).replace("/", "_").replace(" ", "_"))
    return root / "_generated_configs" / ("_".join(parts) + ".yaml")


def _write_generated_config(
    *,
    source_config: str | Path,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    only_condition: str | None,
) -> Path:
    source_config_path = resolve_runtime_config_path(source_config)
    raw_config = load_yaml(source_config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"expected mapping at {source_config_path}")
    expected_root = _raw_results_root(source_config_path, raw_config)
    generated = _patch_config(
        raw_config=raw_config,
        source_config_path=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        seed=seed,
        scenario=scenario,
        only_condition=only_condition,
    )
    generated.setdefault("study_metadata", {})["legacy_results_root"] = portable_path(
        expected_root
    )
    path = _generated_config_path(
        source_config_path=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        seed=seed,
        scenario=scenario,
        only_condition=only_condition,
    )
    dump_yaml(path, generated)
    return path


def _resolved_experiment_root(config: dict[str, object]) -> Path:
    base_dir = Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
    root = resolve_repo_path(
        config.get("results_root", str(results_root())),
        base_dir=base_dir,
    )
    return root / str(config["experiment_name"])


def _trial_dirs(experiment_root: Path) -> list[Path]:
    return sorted(experiment_root.glob("*/attack/*/trial-*"))


def _trial_key(experiment_root: Path, trial_dir: Path) -> tuple[str, str, str, str] | None:
    try:
        rel = trial_dir.relative_to(experiment_root)
    except ValueError:
        return None
    if len(rel.parts) != 4:
        return None
    return rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]


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


def _read_status(path: Path) -> str | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return "incomplete"
    return str(payload.get("status") or "incomplete")


def _prune_empty_trial_parents(trial_dir: Path, experiment_root: Path) -> None:
    parent = trial_dir.parent
    while parent != experiment_root and experiment_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _remove_failed_or_incomplete_trials(
    plan: dict[str, object],
    *,
    full_plan: dict[str, object] | None = None,
) -> list[str]:
    root = plan["experiment_root"]
    if not isinstance(root, Path) or not root.exists():
        return []
    selected_keys = _planned_trial_keys(plan)
    valid_keys = _planned_trial_keys(full_plan or plan)
    removed: list[str] = []
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


def _profile_variation_key(profile: dict[str, Any], suite: str) -> dict[str, Any]:
    if suite == "kt":
        return _target_semantics_without_variant(profile)
    if suite == "interface":
        return _target_semantics_without_interface(profile)
    return deepcopy(profile)


def _compare_attacker_profiles(
    *,
    base_conditions: list[dict[str, Any]],
    generated_conditions: list[dict[str, Any]],
) -> bool:
    if len(base_conditions) != len(generated_conditions):
        return False
    for base, generated in zip(base_conditions, generated_conditions):
        if str(base.get("name")) != str(generated.get("name")):
            return False
        base_profile = base["effective_profiles"]["attacker"]
        generated_profile = generated["effective_profiles"]["attacker"]
        if base_profile != generated_profile:
            return False
    return True


def _capability_manifest(
    *,
    base_config: dict[str, Any],
    generated_config: dict[str, Any],
    base_conditions: list[dict[str, Any]],
    generated_conditions: list[dict[str, Any]],
    scenarios: list[Any],
    repeats: int,
    testbed_id: str,
    suite: str,
    require_full_matrix: bool,
) -> dict[str, Any]:
    expected_count = EXPECTED_ATTACK_SCENARIO_COUNTS[testbed_id]
    attacker_profiles = [
        condition["effective_profiles"]["attacker"] for condition in generated_conditions
    ]
    defender_profiles = [
        condition["effective_profiles"]["defender"] for condition in generated_conditions
    ]
    observer_profiles = [
        condition["effective_profiles"]["observer"] for condition in generated_conditions
    ]
    attacker_variation_stripped = [
        _profile_variation_key(profile, suite) for profile in attacker_profiles
    ]
    conditions = [
        {
            "condition": str(condition.get("name")),
            "attacker_profile": str(condition.get("profiles", {}).get("attacker")),
            "defender_profile": str(condition.get("profiles", {}).get("defender")),
            "target_semantics": dict(
                condition["effective_profiles"]["attacker"].get("target_semantics")
                or {}
            ),
            "attacker_allowed_action_types": list(
                condition["effective_profiles"]["attacker"].get(
                    "allowed_action_types", []
                )
            ),
            "defender_observation_mode": str(
                condition["effective_profiles"]["defender"].get(
                    "defender_observation_mode", ""
                )
            ),
            "defender_defer_containment_until_attacker_effect": bool(
                condition["effective_profiles"]["defender"].get(
                    "defer_containment_until_attacker_effect"
                )
            ),
        }
        for condition in generated_conditions
    ]
    invariance_checks = {
        "attack_scenario_count_expected_for_full_suite": (
            not require_full_matrix or len(scenarios) == expected_count
        ),
        "only_attack_scenarios_selected": all(
            str(scenario.suite) == "attack" for scenario in scenarios
        ),
        "condition_names_preserved": [
            str(item.get("name")) for item in base_conditions
        ]
        == [str(item.get("name")) for item in generated_conditions],
        "attacker_profiles_identical_to_legacy_source": _compare_attacker_profiles(
            base_conditions=base_conditions,
            generated_conditions=generated_conditions,
        ),
        "attacker_profiles_match_except_suite_dimension": _all_equal(
            attacker_variation_stripped
        ),
        "observer_profiles_identical_across_conditions": _all_equal(observer_profiles),
        "defender_profiles_identical_across_conditions": _all_equal(defender_profiles),
        "defender_process_physics_only": all(
            bool(profile.get("process_physics_only"))
            and str(profile.get("defender_observation_mode")).strip().lower()
            == "process_physics_only"
            for profile in defender_profiles
        ),
        "defender_defer_containment_until_attacker_effect": all(
            bool(profile.get("defer_containment_until_attacker_effect"))
            for profile in defender_profiles
        ),
        "profile_bindings_preserved": [
            dict(item.get("profiles") or {}) for item in base_conditions
        ]
        == [dict(item.get("profiles") or {}) for item in generated_conditions],
        "repeats_preserved": int(base_config.get("repeats", 1)) == int(
            generated_config.get("repeats", 1)
        ),
        "scenario_selectors_preserved_for_full_suite": (
            not require_full_matrix
            or list(base_config.get("scenario_selectors", []))
            == list(generated_config.get("scenario_selectors", []))
        ),
    }
    failed = [key for key, ok in invariance_checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "process-defender capability manifest failed invariance checks: "
            + ", ".join(failed)
        )
    return {
        "experiment_name": str(generated_config["experiment_name"]),
        "source_experiment_name": str(base_config["experiment_name"]),
        "testbed": testbed_id,
        "suite": suite,
        "seed": (generated_config.get("reproducibility") or {}).get("seed"),
        "scenario_count": len(scenarios),
        "condition_count": len(generated_conditions),
        "repeats": repeats,
        "planned_runs": len(scenarios) * len(generated_conditions) * repeats,
        "scenario_ids": [scenario.catalog_id for scenario in scenarios],
        "conditions": conditions,
        "preserved_attacker_side_dimensions": {
            "P_A": "unchanged_from_source_config",
            "R_A": "unchanged_from_source_config",
            "Q_A": "unchanged_from_source_config",
            "K_A_g": "unchanged_from_source_config",
            "K_A_t": (
                "varied_exactly_as_source_config"
                if suite == "kt"
                else "unchanged_from_source_config"
            ),
            "O_A": "unchanged_from_source_config",
            "I_A": (
                "varied_exactly_as_source_config"
                if suite == "interface"
                else "unchanged_from_source_config"
            ),
            "U_A": "unchanged_from_source_config",
            "pi_A": "unchanged_from_source_config",
            "Delta_A": "unchanged_from_source_config",
            "model_backend": "unchanged_from_source_config",
            "seed": "set_to_requested_seed",
        },
        "harmonized_defender_dimensions": {
            "observation": "process/controller evidence only",
            "private_attacker_trace_visible": False,
            "startup_containment_allowed": False,
            "containment_chronology": (
                "attacker action executes and plant/controller state updates before "
                "defender observation, detection, and containment"
            ),
        },
        "critical_impact_audit": {
            "stateaware_contract_supported": False,
            "legacy_attack_suite_semantics": (
                "physical safety bounds proxy; branch-level state-aware critical "
                "predicates are not present in legacy attack scenarios"
            ),
            "invalid_precontained_required_absent": True,
        },
        "metric_keys_recorded_by_shared_pipeline": METRIC_KEYS_TO_EXPECT,
        "invariance_checks": invariance_checks,
    }


def _experiment_plan(
    config_path: str | Path,
    *,
    source_config_path: str | Path,
    testbed_id: str,
    suite: str,
    require_full_matrix: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    source_config = load_config(source_config_path)
    catalog = ScenarioCatalog(testbed_id=testbed_id)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    scenarios = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
    conditions = orchestrator._enumerate_conditions(config)
    source_conditions = orchestrator._enumerate_conditions(source_config)
    source_by_name = {str(item.get("name")): item for item in source_conditions}
    selected_source_conditions = []
    for condition in conditions:
        name = str(condition.get("name"))
        if name not in source_by_name:
            raise RuntimeError(
                f"condition `{name}` in generated config is absent from source config"
            )
        selected_source_conditions.append(source_by_name[name])
    repeats = int(config.get("repeats", 1))
    if require_full_matrix and len(conditions) != len(source_conditions):
        raise RuntimeError(
            "full process-defender rerun must preserve the source condition count"
        )
    experiment_root = _resolved_experiment_root(config)
    manifest = _capability_manifest(
        base_config=source_config,
        generated_config=config,
        base_conditions=(
            source_conditions if require_full_matrix else selected_source_conditions
        ),
        generated_conditions=conditions,
        scenarios=scenarios,
        repeats=repeats,
        testbed_id=testbed_id,
        suite=suite,
        require_full_matrix=require_full_matrix,
    )
    first_defender_profile = (
        conditions[0]["effective_profiles"]["defender"] if conditions else {}
    )
    effective_contract = {
        "defender_process_physics_only": True,
        "defender_observation_mode": "process_physics_only",
        "defer_containment_until_attacker_effect": True,
        "defender_allowed_action_types": list(
            first_defender_profile.get("allowed_action_types", [])
            if isinstance(first_defender_profile, dict)
            else []
        ),
        "containment_active_at_start": False,
        "defender_profile_source": portable_path(Path(config["_path"])),
        "defender_mode": "process-aware",
        "testbed": testbed_id,
        "suite": suite,
    }
    payload = {
        "validated": True,
        "config_path": portable_path(Path(config["_path"])),
        "source_config_path": portable_path(Path(source_config["_path"])),
        "experiment_name": str(config["experiment_name"]),
        "testbed": testbed_id,
        "suite": suite,
        "defender_mode": "process-aware",
        "seed": config.get("reproducibility", {}).get("seed"),
        "seeds": [config.get("reproducibility", {}).get("seed")],
        "resume": bool(config.get("resume")),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": repeats,
        "planned_runs": len(scenarios) * len(conditions) * repeats,
        "result_root": portable_path(experiment_root.parent),
        "resolved_experiment_root": portable_path(experiment_root),
        "wd_xdc_override_source": (
            "cli_or_env" if testbed_id == "water_distribution" else "not_applicable"
        ),
        "wd_xdc_override_configured": (
            bool(os.environ.get("ICS_AGENT_SPHERE_XDC_OVERRIDE"))
            if testbed_id == "water_distribution"
            else None
        ),
        "conditions": [str(item.get("name")) for item in conditions],
        "scenarios": [scenario.catalog_id for scenario in scenarios],
        "effective_defender_contract": effective_contract,
        "process_defender_capability_manifest": manifest,
    }
    plan = {
        "config": config,
        "source_config": source_config,
        "testbed_id": testbed_id,
        "suite": suite,
        "scenarios": scenarios,
        "conditions": conditions,
        "repeats": repeats,
        "experiment_root": experiment_root,
        "manifest": manifest,
    }
    return payload, plan


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _tri_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _tri_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n"}
    return False


def _tri_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if _tri_true(value):
        return True
    if _tri_false(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _any_tri_true(values: list[Any]) -> bool | None:
    if any(_tri_true(value) for value in values):
        return True
    if any(value is None for value in values):
        return None
    return False


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _coerce_csv_value(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return text


def _load_observations(trial_dir: Path) -> list[dict[str, Any]]:
    json_path = trial_dir / "observations.json"
    if json_path.exists():
        payload = _load_json(json_path)
        return payload if isinstance(payload, list) else []
    csv_path = trial_dir / "observations.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[Any]) -> float | None:
    numeric = [_safe_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _legacy_critical_audit_row(
    *,
    condition: str,
    scenario_id: str,
    trial: str,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    system_events: list[dict[str, Any]],
    scenario: Any | None,
    seed: int | None,
    testbed_id: str,
    suite: str,
) -> dict[str, Any]:
    attacker_metrics = metrics.get("attacker") if isinstance(metrics, dict) else {}
    if not isinstance(attacker_metrics, dict):
        attacker_metrics = {}
    runtime = metrics.get("runtime_containment") if isinstance(metrics, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}
    expected_actions = _scenario_attack_actions(scenario)
    first_decision = _first_role_decision(decisions, "attacker")
    first_action = _first_role_action(actions, "attacker")
    first_defender_decision = _first_role_decision(decisions, "defender")
    first_defender_action = _first_role_action(actions, "defender")
    decisive_decision = _first_matching_decision(decisions, expected_actions)
    decisive_action = _first_matching_action(actions, expected_actions)
    first_block = _first_block_event(system_events)
    before_state = _nearest_observation(
        observations,
        time_sec=(decisive_action or first_action or {}).get("time_sec"),
        phase="pre_action",
        before=True,
    )
    after_state = _nearest_observation(
        observations,
        time_sec=(decisive_action or first_action or {}).get("time_sec"),
        phase="post_action",
        before=False,
    )
    defender_observation = _nearest_observation(
        observations,
        time_sec=(first_defender_action or first_defender_decision or {}).get("time_sec"),
        before=True,
    )
    safety_bounds = (
        dict((getattr(scenario, "metadata", {}) or {}).get("safety_bounds") or {})
        if scenario is not None
        else {}
    )
    safety_count_raw = metrics.get("safety_violation_count")
    safety_count = None
    try:
        safety_count = None if safety_count_raw is None else int(safety_count_raw)
    except (TypeError, ValueError):
        safety_count = None
    attacker_safety_raw = attacker_metrics.get("caused_safety_violation")
    attacker_safety = _tri_bool(attacker_safety_raw)
    if attacker_safety is None and safety_count is not None:
        attacker_safety = safety_count > 0
    safety_evaluated = bool(safety_bounds) or safety_count is not None
    action_impact_evaluated = (decisive_action or first_action) is not None
    critical_impact_evaluated = (
        True if safety_evaluated or action_impact_evaluated else None
    )
    initial_critical = (
        bool(_safety_violated(_first_observation(observations), safety_bounds))
        if safety_evaluated
        else None
    )
    safety_before = (
        bool(_safety_violated(before_state, safety_bounds))
        if before_state and safety_evaluated
        else initial_critical
    )
    safety_after = (
        bool(_safety_violated(after_state, safety_bounds))
        if after_state and safety_evaluated
        else (attacker_safety if attacker_safety is not None else None)
    )
    impact_types = _legacy_impact_types(
        executed_action=(decisive_action or first_action or {}).get("action"),
        safety_impact=bool(attacker_safety),
        action_impact_evaluated=action_impact_evaluated,
    )
    critical = _any_tri_true(
        [
            safety_after,
            impact_types["availability_impact"],
            impact_types["control_integrity_impact"],
        ]
    )
    if critical_impact_evaluated is None:
        critical_before = None
        critical_after = None
        attacker_caused_critical = None
    else:
        critical_before = bool(safety_before) if safety_before is not None else False
        critical_after = bool(critical)
        attacker_caused_critical = bool(
            (not critical_before) and critical_after and bool(attacker_safety)
        ) or _tri_true(impact_types["availability_impact"]) or _tri_true(
            impact_types["control_integrity_impact"]
        )
    has_expected_decisive_action = bool(expected_actions)
    decisive_action_caused_critical = (
        None
        if not has_expected_decisive_action or critical_impact_evaluated is None
        else bool(decisive_action is not None and attacker_caused_critical)
    )
    blocked_count = runtime.get("blocked_attacker_action_count") or 0
    try:
        blocked_count_numeric = int(blocked_count)
    except (TypeError, ValueError):
        blocked_count_numeric = 0
    defender_cutoff = bool(runtime.get("attacker_execution_cutoff_activated"))
    invalid_precontained = bool(metrics.get("invalid_precontained_run"))
    decisive_selected = decisive_decision is not None
    decisive_executed = decisive_action is not None
    defender_prevention = bool(
        not attacker_caused_critical
        and decisive_selected
        and not decisive_executed
        and blocked_count_numeric > 0
        and not invalid_precontained
    )
    defender_mitigation = bool(
        attacker_caused_critical and defender_cutoff and not invalid_precontained
    )
    defender_recovery = bool(
        attacker_caused_critical and metrics.get("recovery_success_rate")
    )
    selected_operation = _action_operation((decisive_decision or first_decision or {}).get("action"))
    selected_args = _action_arguments((decisive_decision or first_decision or {}).get("action"))
    executed_operation = _action_operation((decisive_action or first_action or {}).get("action"))
    executed_args = _action_arguments((decisive_action or first_action or {}).get("action"))
    return {
        "condition": condition,
        "scenario_id": scenario_id,
        "trial": trial,
        "status": summary.get("status"),
        "testbed": testbed_id,
        "suite": suite,
        "run_id": summary.get("run_id"),
        "seed": seed,
        "scenario_fingerprint": stable_hash(
            getattr(scenario, "execution", {}) if scenario is not None else scenario_id,
            length=12,
        ),
        "initial_state_fingerprint": stable_hash(
            (getattr(scenario, "execution", {}) or {}).get("initial_conditions", {})
            if scenario is not None
            else {},
            length=12,
        ),
        "critical_impact_evaluated": critical_impact_evaluated,
        "critical_impact_contract_status": "legacy_derived_audit",
        "legacy_contract_basis": (
            "scenario timeline, safety_bounds, executed attacker actions, and "
            "shared runtime containment metrics"
        ),
        "stateaware_branch_id": metrics.get("stateaware_branch_id"),
        "critical_process_impact": critical,
        "attacker_caused_critical_impact": attacker_caused_critical,
        "decisive_action_caused_critical_impact": decisive_action_caused_critical,
        "physical_safety_impact": safety_after if safety_evaluated else None,
        "availability_impact": impact_types["availability_impact"],
        "control_integrity_impact": impact_types["control_integrity_impact"],
        "impact_subtype": impact_types["impact_subtype"],
        "critical_predicate_before": critical_before,
        "critical_predicate_after": critical_after or critical,
        "initial_critical": initial_critical,
        "true_state_before": json.dumps(before_state, sort_keys=True),
        "true_state_after": json.dumps(after_state, sort_keys=True),
        "relevant_scenario_variables_before": json.dumps(
            _bounded_fields(before_state, safety_bounds),
            sort_keys=True,
        ),
        "relevant_scenario_variables_after": json.dumps(
            _bounded_fields(after_state, safety_bounds),
            sort_keys=True,
        ),
        "expected_decisive_actions": json.dumps(expected_actions, sort_keys=True),
        "candidate_count": _candidate_count(first_decision),
        "selected_operation": selected_operation,
        "selected_target": _action_target((decisive_decision or first_decision or {}).get("action")),
        "selected_arguments": json.dumps(selected_args, sort_keys=True),
        "executed_operation": executed_operation,
        "executed_target": _action_target((decisive_action or first_action or {}).get("action")),
        "executed_arguments": json.dumps(executed_args, sort_keys=True),
        "state_appropriate_action_selected": (
            decisive_selected if has_expected_decisive_action else None
        ),
        "state_appropriate_action_executed": (
            decisive_executed if has_expected_decisive_action else None
        ),
        "decisive_action_selected": (
            decisive_selected if has_expected_decisive_action else None
        ),
        "decisive_action_issued": (
            decisive_selected and bool(decisive_decision.get("action"))
            if has_expected_decisive_action
            else None
        ),
        "decisive_action_blocked": first_block is not None,
        "decisive_action_executed": (
            decisive_executed if has_expected_decisive_action else None
        ),
        "block_reason": None if first_block is None else first_block.get("reason"),
        "defender_visible_observation_snapshot": json.dumps(
            defender_observation,
            sort_keys=True,
        ),
        "defender_detection_reason": (
            (first_defender_decision or {}).get("diagnosis")
            or (first_defender_decision or {}).get("summary")
        ),
        "defender_selected_response": json.dumps(
            (first_defender_action or first_defender_decision or {}).get("action"),
            sort_keys=True,
        ),
        "event_phase_order": (
            "pre_action -> attacker_issue/execute -> post_action/process_update "
            "-> observer -> defender_observe/act -> post_defense"
        ),
        "legacy_safety_violation_count": safety_count,
        "attacker_caused_safety_violation": attacker_safety,
        "defender_prevention": defender_prevention,
        "defender_mitigation": defender_mitigation,
        "defender_recovery": defender_recovery,
        "invalid_precontained_run": invalid_precontained,
        "attacker_execution_cutoff_pre_attack_start": bool(
            metrics.get("attacker_execution_cutoff_pre_attack_start")
        ),
        "defender_cutoff_activated": defender_cutoff,
        "blocked_attacker_action_count": runtime.get("blocked_attacker_action_count"),
        "contained_before_safety_violation": (
            metrics.get("defender_intervention", {}) or {}
        ).get("contained_before_safety_violation"),
        "time_to_detection_sec": metrics.get("time_to_detection_sec"),
        "time_to_first_defender_action_sec": (
            metrics.get("defender_intervention", {}) or {}
        ).get("time_to_first_action_sec"),
        "time_to_effective_containment_cutoff_sec": runtime.get(
            "attacker_execution_cutoff_activation_time_sec"
        ),
        "time_to_first_safety_violation_sec": attacker_metrics.get(
            "time_to_physical_impact_sec"
        ),
        "wall_clock_time_to_detection_sec": metrics.get(
            "wall_clock_time_to_detection_from_attack_start_sec"
        ),
        "wall_clock_time_to_first_defender_action_sec": metrics.get(
            "wall_clock_time_to_first_defender_action_from_attack_start_sec"
        ),
        "wall_clock_time_to_effective_containment_cutoff_sec": metrics.get(
            "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
        ),
        "wall_clock_time_to_first_safety_violation_sec": metrics.get(
            "wall_clock_time_to_first_safety_violation_from_attack_start_sec"
        ),
        "T_select": _event_time(decisive_decision),
        "T_issue": _event_time(decisive_decision),
        "T_execute": _event_time(decisive_action),
        "T_process_update": _observation_time(after_state),
        "T_critical": attacker_metrics.get("time_to_physical_impact_sec"),
        "T_detect": metrics.get("time_to_detection_sec"),
        "T_contain": runtime.get("attacker_execution_cutoff_activation_time_sec"),
        "T_recover": None,
        "attacker_action_count": metrics.get("attacker_action_count"),
        "attacker_executed_action_count": len(
            [item for item in actions if item.get("role") == "attacker"]
        ),
        "attacker_attempted_action_count": metrics.get(
            "attacker_attempted_action_count"
        ),
        "attacker_effect_capable_action_count": metrics.get(
            "attacker_effect_capable_action_count"
        ),
        "attacker_ineffective_action_count": metrics.get(
            "attacker_ineffective_action_count"
        ),
        "distinct_surfaces_probed_count": metrics.get(
            "distinct_surfaces_probed_count"
        ),
        "downtime_timestep_count": metrics.get("downtime_timestep_count"),
    }


def _event_time(item: dict[str, Any] | None) -> float | None:
    if not isinstance(item, dict):
        return None
    return _safe_float(item.get("time_sec"))


def _observation_time(item: dict[str, Any] | None) -> float | None:
    if not isinstance(item, dict):
        return None
    return _safe_float(item.get("elapsed_sec", item.get("time_sec")))


def _first_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    return observations[0] if observations else None


def _first_role_decision(
    decisions: list[dict[str, Any]],
    role: str,
) -> dict[str, Any] | None:
    for item in sorted(decisions, key=lambda row: float(row.get("time_sec", 0.0))):
        if item.get("role") == role:
            return item
    return None


def _first_role_action(
    actions: list[dict[str, Any]],
    role: str,
) -> dict[str, Any] | None:
    for item in sorted(actions, key=lambda row: float(row.get("time_sec", 0.0))):
        if item.get("role") == role:
            return item
    return None


def _first_block_event(system_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in sorted(system_events, key=lambda row: float(row.get("time_sec", 0.0))):
        if item.get("event_type") == "attacker_action_blocked":
            return item
    return None


def _scenario_attack_actions(scenario: Any | None) -> list[dict[str, Any]]:
    if scenario is None:
        return []
    execution = getattr(scenario, "execution", {}) or {}
    metadata = getattr(scenario, "metadata", {}) or {}
    bindings = (metadata.get("role_bindings") or {}).get("attacker") or []
    timeline = list(execution.get("timeline") or [])
    actions: list[dict[str, Any]] = []
    for index in bindings:
        try:
            item = timeline[int(index)]
        except (IndexError, TypeError, ValueError):
            continue
        action = normalize_action(item.get("action"))
        if action is not None and action.get("type") != "noop":
            actions.append(action)
    return actions


def _action_matches(actual: Any, expected: dict[str, Any]) -> bool:
    action = normalize_action(actual)
    if action is None:
        return False
    return action_key(action) == action_key(expected)


def _first_matching_decision(
    decisions: list[dict[str, Any]],
    expected_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not expected_actions:
        return None
    for item in sorted(decisions, key=lambda row: float(row.get("time_sec", 0.0))):
        if item.get("role") != "attacker":
            continue
        if any(_action_matches(item.get("action"), expected) for expected in expected_actions):
            return item
    return None


def _first_matching_action(
    actions: list[dict[str, Any]],
    expected_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not expected_actions:
        return None
    for item in sorted(actions, key=lambda row: float(row.get("time_sec", 0.0))):
        if item.get("role") != "attacker":
            continue
        if any(_action_matches(item.get("action"), expected) for expected in expected_actions):
            return item
    return None


def _candidate_count(decision: dict[str, Any] | None) -> int | None:
    if not isinstance(decision, dict):
        return None
    metadata = decision.get("metadata") if isinstance(decision.get("metadata"), dict) else {}
    audit = metadata.get("stateaware_decision_audit")
    if isinstance(audit, dict) and isinstance(audit.get("visible_candidates"), list):
        return len(audit["visible_candidates"])
    candidates = metadata.get("candidate_actions")
    if isinstance(candidates, list):
        return len(candidates)
    count = metadata.get("candidate_count")
    try:
        return None if count is None else int(count)
    except (TypeError, ValueError):
        return None


def _action_operation(action: Any) -> str | None:
    normalized = normalize_action(action)
    return None if normalized is None else str(normalized.get("type"))


def _action_target(action: Any) -> str | None:
    normalized = normalize_action(action)
    if normalized is None:
        return None
    for key in ("name", "process_id", "surface", "host", "target"):
        value = normalized.get(key)
        if value is not None:
            return str(value)
    return None


def _action_arguments(action: Any) -> dict[str, Any]:
    normalized = normalize_action(action)
    if normalized is None:
        return {}
    return {
        key: value
        for key, value in normalized.items()
        if key not in {"type", "name", "process_id", "surface", "host", "target"}
    }


def _safety_violated(
    observation: dict[str, Any] | None,
    safety_bounds: dict[str, Any],
) -> bool:
    if not isinstance(observation, dict):
        return False
    for field, bounds in safety_bounds.items():
        if field not in observation or not isinstance(bounds, dict):
            continue
        value = _safe_float(observation.get(field))
        if value is None:
            continue
        minimum = _safe_float(bounds.get("min"))
        maximum = _safe_float(bounds.get("max"))
        if minimum is not None and value < minimum:
            return True
        if maximum is not None and value > maximum:
            return True
    return False


def _bounded_fields(
    observation: dict[str, Any] | None,
    safety_bounds: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    return {
        field: observation.get(field)
        for field in safety_bounds
        if field in observation
    }


def _nearest_observation(
    observations: list[dict[str, Any]],
    *,
    time_sec: Any,
    phase: str | None = None,
    before: bool | None = None,
) -> dict[str, Any] | None:
    target = _safe_float(time_sec)
    if target is None:
        return None
    candidates: list[dict[str, Any]] = []
    for observation in observations:
        current_time = _observation_time(observation)
        if current_time is None:
            continue
        if phase and str(observation.get("sample_phase", "")) != phase:
            continue
        if before is True and current_time > target:
            continue
        if before is False and current_time < target:
            continue
        candidates.append(dict(observation))
    if not candidates:
        return None
    if before is True:
        return max(candidates, key=lambda item: _observation_time(item) or 0.0)
    if before is False:
        return min(candidates, key=lambda item: _observation_time(item) or 0.0)
    return min(candidates, key=lambda item: abs((_observation_time(item) or 0.0) - target))


def _legacy_impact_types(
    *,
    executed_action: Any,
    safety_impact: bool,
    action_impact_evaluated: bool = True,
) -> dict[str, Any]:
    action_type = _action_operation(executed_action)
    availability = (
        None
        if not action_impact_evaluated
        else action_type in {"hmi_stop", "hmi_estop"}
    )
    control_integrity = (
        None
        if not action_impact_evaluated
        else action_type
        in {
            "select_process",
            "set_manual_valve",
            "set_sensor",
            "set_actuator",
        }
    )
    subtypes: list[str] = []
    if safety_impact:
        subtypes.append("physical_safety")
    if availability:
        subtypes.append("availability")
    if control_integrity:
        subtypes.append("control_integrity")
    return {
        "availability_impact": availability,
        "control_integrity_impact": control_integrity,
        "impact_subtype": "+".join(subtypes) if subtypes else None,
    }


def _action_key_or_none(action: Any) -> str | None:
    normalized = normalize_action(action)
    if normalized is None:
        return None
    return action_key(normalized)


def _same_time(left: Any, right: Any) -> bool:
    left_time = _safe_float(left)
    right_time = _safe_float(right)
    return (
        left_time is not None
        and right_time is not None
        and abs(left_time - right_time) <= 1e-6
    )


def _matching_action_for_decision(
    *,
    decision: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decision_key = _action_key_or_none(decision.get("action"))
    if decision_key is None:
        return None
    for action in actions:
        if action.get("role") != decision.get("role"):
            continue
        if not _same_time(action.get("time_sec"), decision.get("time_sec")):
            continue
        if _action_key_or_none(action.get("action")) == decision_key:
            return action
    return None


def _matching_block_for_decision(
    *,
    decision: dict[str, Any],
    system_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decision_key = _action_key_or_none(decision.get("action"))
    if decision_key is None:
        return None
    for event in system_events:
        if event.get("event_type") != "attacker_action_blocked":
            continue
        if not _same_time(event.get("time_sec"), decision.get("time_sec")):
            continue
        if _action_key_or_none(event.get("attempted_action")) == decision_key:
            return event
    return None


def _guardrail_rejection_for_decision(
    *,
    decision: dict[str, Any],
    system_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for event in system_events:
        if event.get("event_type") != "guardrail_rejection":
            continue
        if event.get("role") != decision.get("role"):
            continue
        if _same_time(event.get("time_sec"), decision.get("time_sec")):
            return event
    return None


def _candidate_identifiers(decision: dict[str, Any]) -> list[Any]:
    metadata = decision.get("metadata") if isinstance(decision.get("metadata"), dict) else {}
    audit = metadata.get("stateaware_decision_audit")
    if isinstance(audit, dict) and isinstance(audit.get("visible_candidates"), list):
        return [
            item.get("id")
            for item in audit["visible_candidates"]
            if isinstance(item, dict) and item.get("id") is not None
        ]
    candidates = metadata.get("candidate_actions")
    if isinstance(candidates, list):
        return [
            item.get("id")
            for item in candidates
            if isinstance(item, dict) and item.get("id") is not None
        ]
    return []


def _selected_candidate_id(decision: dict[str, Any]) -> Any:
    metadata = decision.get("metadata") if isinstance(decision.get("metadata"), dict) else {}
    audit = metadata.get("stateaware_decision_audit")
    if isinstance(audit, dict):
        return audit.get("selected_candidate_id")
    for key in ("selected_candidate_id", "candidate_id"):
        if key in metadata:
            return metadata.get(key)
    return None


def _decoded_action_for_decision(decision: dict[str, Any]) -> Any:
    metadata = decision.get("metadata") if isinstance(decision.get("metadata"), dict) else {}
    audit = metadata.get("stateaware_decision_audit")
    if isinstance(audit, dict) and audit.get("decoded_real_action") is not None:
        return audit.get("decoded_real_action")
    return decision.get("action")


def _surface_for_action(action: Any) -> str | None:
    normalized = normalize_action(action)
    if normalized is None:
        return None
    return _action_target(normalized) or _action_operation(normalized)


def _is_recon_action(action: Any) -> bool | None:
    action_type_value = _action_operation(action)
    if action_type_value is None:
        return None
    return action_type_value in {"inspect_surface", "noop"}


def _is_effect_capable_action(action: Any) -> bool | None:
    is_recon = _is_recon_action(action)
    if is_recon is None:
        return None
    return not is_recon


def _attacker_trace_rows(
    *,
    condition: str,
    scenario_id: str,
    trial: str,
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    system_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attacker_decisions = [
        decision
        for decision in sorted(decisions, key=lambda item: float(item.get("time_sec", 0.0)))
        if decision.get("role") == "attacker"
    ]
    for index, decision in enumerate(attacker_decisions, start=1):
        decoded = _decoded_action_for_decision(decision)
        executed = _matching_action_for_decision(decision=decision, actions=actions)
        blocked = _matching_block_for_decision(
            decision=decision,
            system_events=system_events,
        )
        guardrail = _guardrail_rejection_for_decision(
            decision=decision,
            system_events=system_events,
        )
        action_time = decision.get("time_sec")
        before_state = _nearest_observation(
            observations,
            time_sec=action_time,
            phase="pre_action",
            before=True,
        )
        after_state = _nearest_observation(
            observations,
            time_sec=action_time,
            phase="post_action",
            before=False,
        )
        executed_flag = executed is not None
        blocked_flag = blocked is not None
        validation_succeeded = (
            decoded is not None and guardrail is None and bool(executed or blocked)
        )
        effect_capable = _is_effect_capable_action(decoded)
        rows.append(
            {
                "condition": condition,
                "scenario_id": scenario_id,
                "trial": trial,
                "event_index": index,
                "event_phase": "attacker_decision",
                "action_index": index,
                "T_select": action_time,
                "T_issue": action_time if decoded is not None else None,
                "T_validate": action_time if decoded is not None else None,
                "T_execute": executed.get("time_sec") if executed else None,
                "candidate_count": _candidate_count(decision),
                "candidate_identifiers": json.dumps(
                    _candidate_identifiers(decision),
                    sort_keys=True,
                ),
                "selected_candidate_id": _selected_candidate_id(decision),
                "selected_operation": _action_operation(decoded),
                "selected_target": _action_target(decoded),
                "selected_arguments": json.dumps(
                    _action_arguments(decoded),
                    sort_keys=True,
                ),
                "issued": decoded is not None,
                "validation_succeeded": validation_succeeded,
                "validation_failure_reason": (
                    None if guardrail is None else guardrail.get("detail")
                ),
                "blocked": blocked_flag,
                "block_reason": None if blocked is None else blocked.get("reason"),
                "executed": executed_flag,
                "execution_succeeded": executed_flag,
                "execution_error": None,
                "action_surface": _surface_for_action(decoded),
                "recon_action": _is_recon_action(decoded),
                "effect_capable_action": effect_capable,
                "ineffective_action": (
                    None
                    if effect_capable is None
                    else bool(not executed_flag or not effect_capable)
                ),
                "true_state_before": json.dumps(before_state, sort_keys=True),
                "true_state_after": json.dumps(after_state, sort_keys=True),
            }
        )
    return rows


def _nearest_containment_activation(
    *,
    decision: dict[str, Any],
    system_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decision_time = _safe_float(decision.get("time_sec"))
    candidates = [
        event
        for event in system_events
        if event.get("event_type") == "attacker_execution_cutoff_activated"
        and event.get("role") == "defender"
        and (
            decision_time is None
            or (_safe_float(event.get("time_sec")) is not None and _safe_float(event.get("time_sec")) >= decision_time)
        )
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: _safe_float(item.get("time_sec")) or 0.0)


def _later_blocked_count(
    *,
    decision: dict[str, Any],
    system_events: list[dict[str, Any]],
) -> int:
    decision_time = _safe_float(decision.get("time_sec"))
    return sum(
        1
        for event in system_events
        if event.get("event_type") == "attacker_action_blocked"
        and (
            decision_time is None
            or (_safe_float(event.get("time_sec")) is not None and _safe_float(event.get("time_sec")) >= decision_time)
        )
    )


def _defender_trace_rows(
    *,
    condition: str,
    scenario_id: str,
    trial: str,
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    system_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    defender_decisions = [
        decision
        for decision in sorted(decisions, key=lambda item: float(item.get("time_sec", 0.0)))
        if decision.get("role") == "defender"
    ]
    for index, decision in enumerate(defender_decisions, start=1):
        decoded = _decoded_action_for_decision(decision)
        executed = _matching_action_for_decision(decision=decision, actions=actions)
        guardrail = _guardrail_rejection_for_decision(
            decision=decision,
            system_events=system_events,
        )
        activation = _nearest_containment_activation(
            decision=decision,
            system_events=system_events,
        )
        observation = _nearest_observation(
            observations,
            time_sec=decision.get("time_sec"),
            before=True,
        )
        action_containment = (
            executed.get("containment")
            if executed and isinstance(executed.get("containment"), dict)
            else {}
        )
        containment_requested = bool(
            action_containment.get("is_containment_action")
            or (activation is not None and _same_time(activation.get("time_sec"), decision.get("time_sec")))
        )
        rows.append(
            {
                "condition": condition,
                "scenario_id": scenario_id,
                "trial": trial,
                "event_index": index,
                "event_phase": "defender_decision",
                "defender_event_id": index,
                "time_sec": decision.get("time_sec"),
                "defender_visible_observation_snapshot": json.dumps(
                    observation,
                    sort_keys=True,
                ),
                "private_attacker_fields_excluded_from_snapshot": True,
                "detection_flag": decision.get("decision_type") in {"alert", "action"}
                or bool(decision.get("diagnosis")),
                "detection_reason": decision.get("diagnosis") or decision.get("summary"),
                "recognition_flag": bool(decision.get("diagnosis")),
                "selected_defender_action": json.dumps(decoded, sort_keys=True),
                "issued": decoded is not None,
                "executed": executed is not None,
                "validation_succeeded": decoded is not None and guardrail is None,
                "validation_failure_reason": (
                    None if guardrail is None else guardrail.get("detail")
                ),
                "containment_requested": containment_requested,
                "containment_effective": activation is not None,
                "containment_effective_time_sec": (
                    None if activation is None else activation.get("time_sec")
                ),
                "blocked_attacker_actions_afterward": _later_blocked_count(
                    decision=decision,
                    system_events=system_events,
                ),
            }
        )
    return rows


def _event_trace_rows(
    *,
    condition: str,
    scenario_id: str,
    trial: str,
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    system_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phase_order = {
        "pre_action": 10,
        "attacker_decision": 20,
        "attacker_execute": 30,
        "post_action": 40,
        "defender_decision": 50,
        "defender_execute": 60,
        "post_defense": 70,
        "system_event": 80,
    }
    raw_rows: list[dict[str, Any]] = []
    for observation in observations:
        phase = str(observation.get("sample_phase") or "observation")
        raw_rows.append(
            {
                "time_sec": _observation_time(observation),
                "event_phase": phase,
                "event_type": "observation",
                "role": None,
                "payload": observation,
            }
        )
    for decision in decisions:
        role = str(decision.get("role") or "")
        raw_rows.append(
            {
                "time_sec": _safe_float(decision.get("time_sec")),
                "event_phase": f"{role}_decision",
                "event_type": "decision",
                "role": role,
                "payload": decision,
            }
        )
    for action in actions:
        role = str(action.get("role") or "")
        raw_rows.append(
            {
                "time_sec": _safe_float(action.get("time_sec")),
                "event_phase": f"{role}_execute",
                "event_type": "action",
                "role": role,
                "payload": action,
            }
        )
    for event in system_events:
        raw_rows.append(
            {
                "time_sec": _safe_float(event.get("time_sec")),
                "event_phase": str(event.get("event_type") or "system_event"),
                "event_type": str(event.get("event_type") or "system_event"),
                "role": event.get("role"),
                "payload": event,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
        time_value = _safe_float(row.get("time_sec"))
        phase = str(row.get("event_phase") or "")
        return (
            -1.0 if time_value is None else time_value,
            phase_order.get(phase, phase_order.get(row.get("event_type"), 90)),
            phase,
        )

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(raw_rows, key=sort_key), start=1):
        payload = row.get("payload")
        rows.append(
            {
                "condition": condition,
                "scenario_id": scenario_id,
                "trial": trial,
                "event_index": index,
                "time_sec": row.get("time_sec"),
                "event_phase": row.get("event_phase"),
                "event_type": row.get("event_type"),
                "role": row.get("role"),
                "payload": json.dumps(payload, sort_keys=True),
            }
        )
    return rows


def _condition_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition")), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for condition, condition_rows in sorted(grouped.items()):
        completed = [row for row in condition_rows if row.get("status") == "completed"]
        # Three-state semantics for null-safe critical impact evaluation.
        evaluated = [
            row for row in completed if row.get("critical_impact_evaluated") is True
        ]
        unknown = [
            row for row in completed if row.get("critical_impact_evaluated") is None
        ]
        false_evaluated = [
            row
            for row in evaluated
            if _tri_false(row.get("attacker_caused_critical_impact"))
        ]
        true_attacker_caused = [
            row
            for row in evaluated
            if _tri_true(row.get("attacker_caused_critical_impact"))
        ]
        evaluated_count = len(evaluated)
        unknown_count = len(unknown)
        true_count = len(true_attacker_caused)
        false_count = len(false_evaluated)
        attacker_caused_rate = None
        if evaluated_count:
            attacker_caused_rate = true_count / evaluated_count
        aggregates.append(
            {
                "condition": condition,
                "attack_run_count": len(condition_rows),
                "completed_run_count": len(completed),
                "failed_run_count": len(condition_rows) - len(completed),
                "critical_impact_evaluated_run_count": evaluated_count,
                "critical_impact_unknown_run_count": unknown_count,
                "attacker_caused_critical_impact_run_count": len(true_attacker_caused),
                "attacker_caused_critical_impact_false_run_count": false_count,
                "attacker_caused_critical_impact_rate": (
                    None if attacker_caused_rate is None else round(attacker_caused_rate, 4)
                ),
                "state_appropriate_selected_run_count": sum(
                    1
                    for row in completed
                    if _to_bool(row.get("state_appropriate_action_selected"))
                ),
                "state_appropriate_executed_run_count": sum(
                    1
                    for row in completed
                    if _to_bool(row.get("state_appropriate_action_executed"))
                ),
                "decisive_selected_run_count": sum(
                    1 for row in completed if _to_bool(row.get("decisive_action_selected"))
                ),
                "decisive_issued_run_count": sum(
                    1 for row in completed if _to_bool(row.get("decisive_action_issued"))
                ),
                "decisive_executed_run_count": sum(
                    1 for row in completed if _to_bool(row.get("decisive_action_executed"))
                ),
                "physical_safety_impact_run_count": sum(
                    1 for row in completed if _tri_true(row.get("physical_safety_impact"))
                ),
                "physical_safety_impact_evaluated_run_count": sum(
                    1
                    for row in completed
                    if row.get("physical_safety_impact") is not None
                ),
                "physical_safety_impact_unknown_run_count": sum(
                    1
                    for row in completed
                    if row.get("physical_safety_impact") is None
                ),
                "defender_cutoff_activated_run_count": sum(
                    1 for row in completed if _to_bool(row.get("defender_cutoff_activated"))
                ),
                "contained_before_safety_violation_run_count": sum(
                    1
                    for row in completed
                    if _to_bool(row.get("contained_before_safety_violation"))
                ),
                "defender_prevention_run_count": sum(
                    1 for row in completed if _to_bool(row.get("defender_prevention"))
                ),
                "defender_mitigation_run_count": sum(
                    1 for row in completed if _to_bool(row.get("defender_mitigation"))
                ),
                "defender_recovery_run_count": sum(
                    1 for row in completed if _to_bool(row.get("defender_recovery"))
                ),
                "invalid_precontained_run_count": sum(
                    1 for row in completed if _to_bool(row.get("invalid_precontained_run"))
                ),
                "avg_T_critical": _mean([row.get("T_critical") for row in completed]),
                "avg_T_detect": _mean([row.get("T_detect") for row in completed]),
                "avg_T_contain": _mean([row.get("T_contain") for row in completed]),
                "avg_T_recover": _mean([row.get("T_recover") for row in completed]),
                "avg_time_to_detection_sec": _mean(
                    [row.get("time_to_detection_sec") for row in completed]
                ),
                "avg_time_to_effective_containment_cutoff_sec": _mean(
                    [
                        row.get("time_to_effective_containment_cutoff_sec")
                        for row in completed
                    ]
                ),
                "avg_wall_clock_time_to_detection_sec": _mean(
                    [row.get("wall_clock_time_to_detection_sec") for row in completed]
                ),
                "avg_wall_clock_time_to_effective_containment_cutoff_sec": _mean(
                    [
                        row.get("wall_clock_time_to_effective_containment_cutoff_sec")
                        for row in completed
                    ]
                ),
                "avg_attacker_action_count": _mean(
                    [row.get("attacker_action_count") for row in completed]
                ),
                "avg_attempted_actions": _mean(
                    [row.get("attacker_attempted_action_count") for row in completed]
                ),
                "avg_executed_actions": _mean(
                    [row.get("attacker_executed_action_count") for row in completed]
                ),
                "avg_effect_capable_actions": _mean(
                    [row.get("attacker_effect_capable_action_count") for row in completed]
                ),
                "avg_attacker_ineffective_action_count": _mean(
                    [row.get("attacker_ineffective_action_count") for row in completed]
                ),
                "avg_ineffective_actions": _mean(
                    [row.get("attacker_ineffective_action_count") for row in completed]
                ),
                "avg_blocked_actions": _mean(
                    [row.get("blocked_attacker_action_count") for row in completed]
                ),
                "avg_distinct_surfaces_probed_count": _mean(
                    [row.get("distinct_surfaces_probed_count") for row in completed]
                ),
                "avg_downtime_timestep_count": _mean(
                    [row.get("downtime_timestep_count") for row in completed]
                ),
            }
        )
    return aggregates


def _postprocess_results(
    experiment_root: Path,
    *,
    plan: dict[str, object],
) -> dict[str, Any]:
    valid_keys = _planned_trial_keys(plan)
    scenario_by_id = {
        str(getattr(scenario, "scenario_id", "")): scenario
        for scenario in plan["scenarios"]
    }
    seed_value = (plan["config"].get("reproducibility") or {}).get("seed")
    try:
        seed = None if seed_value is None else int(seed_value)
    except (TypeError, ValueError):
        seed = None
    audit_rows: list[dict[str, Any]] = []
    attacker_trace_rows: list[dict[str, Any]] = []
    defender_trace_rows: list[dict[str, Any]] = []
    event_trace_rows: list[dict[str, Any]] = []
    for trial_dir in _trial_dirs(experiment_root):
        key = _trial_key(experiment_root, trial_dir)
        if key is None or key not in valid_keys:
            continue
        metrics_path = trial_dir / "metrics.json"
        summary_path = trial_dir / "summary.json"
        if not metrics_path.exists() or not summary_path.exists():
            continue
        condition, _suite_name, scenario_id, trial = key
        metrics = _load_json(metrics_path)
        summary = _load_json(summary_path)
        decisions = (
            _load_json(trial_dir / "decisions.json")
            if (trial_dir / "decisions.json").exists()
            else []
        )
        actions = (
            _load_json(trial_dir / "actions.json")
            if (trial_dir / "actions.json").exists()
            else []
        )
        system_events = (
            _load_json(trial_dir / "system_events.json")
            if (trial_dir / "system_events.json").exists()
            else []
        )
        decisions_list = decisions if isinstance(decisions, list) else []
        actions_list = actions if isinstance(actions, list) else []
        observations = _load_observations(trial_dir)
        system_events_list = system_events if isinstance(system_events, list) else []
        audit_rows.append(
            _legacy_critical_audit_row(
                condition=condition,
                scenario_id=scenario_id,
                trial=trial,
                summary=summary,
                metrics=metrics,
                decisions=decisions_list,
                actions=actions_list,
                observations=observations,
                system_events=system_events_list,
                scenario=scenario_by_id.get(scenario_id),
                seed=seed,
                testbed_id=str(plan["testbed_id"]),
                suite=str(plan["suite"]),
            )
        )
        attacker_trace_rows.extend(
            _attacker_trace_rows(
                condition=condition,
                scenario_id=scenario_id,
                trial=trial,
                decisions=decisions_list,
                actions=actions_list,
                observations=observations,
                system_events=system_events_list,
            )
        )
        defender_trace_rows.extend(
            _defender_trace_rows(
                condition=condition,
                scenario_id=scenario_id,
                trial=trial,
                decisions=decisions_list,
                actions=actions_list,
                observations=observations,
                system_events=system_events_list,
            )
        )
        event_trace_rows.extend(
            _event_trace_rows(
                condition=condition,
                scenario_id=scenario_id,
                trial=trial,
                decisions=decisions_list,
                actions=actions_list,
                observations=observations,
                system_events=system_events_list,
            )
        )
    aggregate_rows = _condition_aggregate_rows(audit_rows)
    write_csv(experiment_root / "process_defender_critical_impact_audit.csv", audit_rows)
    write_json(
        experiment_root / "process_defender_critical_impact_audit.json",
        audit_rows,
    )
    write_csv(
        experiment_root / "process_defender_condition_comparison.csv",
        aggregate_rows,
    )
    write_json(
        experiment_root / "process_defender_condition_comparison.json",
        aggregate_rows,
    )
    write_csv(
        experiment_root / "process_defender_attacker_trace.csv",
        attacker_trace_rows,
    )
    write_json(
        experiment_root / "process_defender_attacker_trace.json",
        attacker_trace_rows,
    )
    write_csv(
        experiment_root / "process_defender_defender_trace.csv",
        defender_trace_rows,
    )
    write_json(
        experiment_root / "process_defender_defender_trace.json",
        defender_trace_rows,
    )
    write_csv(
        experiment_root / "process_defender_event_trace.csv",
        event_trace_rows,
    )
    write_json(
        experiment_root / "process_defender_event_trace.json",
        event_trace_rows,
    )
    validation = {
        "validation_passed": bool(audit_rows)
        and all(row.get("status") == "completed" for row in audit_rows)
        and not any(_to_bool(row.get("invalid_precontained_run")) for row in audit_rows),
        "testbed": str(plan["testbed_id"]),
        "suite": str(plan["suite"]),
        "run_count": len(audit_rows),
        "attacker_trace_event_count": len(attacker_trace_rows),
        "defender_trace_event_count": len(defender_trace_rows),
        "event_trace_event_count": len(event_trace_rows),
        "invalid_precontained_run_count": sum(
            1 for row in audit_rows if _to_bool(row.get("invalid_precontained_run"))
        ),
        "critical_impact_contract_status": "legacy_safety_bounds_proxy",
    }
    write_json(experiment_root / "process_defender_validation_report.json", validation)
    return {
        "critical_impact_audit_csv": portable_path(
            experiment_root / "process_defender_critical_impact_audit.csv"
        ),
        "critical_impact_audit_json": portable_path(
            experiment_root / "process_defender_critical_impact_audit.json"
        ),
        "condition_comparison_csv": portable_path(
            experiment_root / "process_defender_condition_comparison.csv"
        ),
        "condition_comparison_json": portable_path(
            experiment_root / "process_defender_condition_comparison.json"
        ),
        "validation_report_json": portable_path(
            experiment_root / "process_defender_validation_report.json"
        ),
        "attacker_trace_csv": portable_path(
            experiment_root / "process_defender_attacker_trace.csv"
        ),
        "attacker_trace_json": portable_path(
            experiment_root / "process_defender_attacker_trace.json"
        ),
        "defender_trace_csv": portable_path(
            experiment_root / "process_defender_defender_trace.csv"
        ),
        "defender_trace_json": portable_path(
            experiment_root / "process_defender_defender_trace.json"
        ),
        "event_trace_csv": portable_path(
            experiment_root / "process_defender_event_trace.csv"
        ),
        "event_trace_json": portable_path(
            experiment_root / "process_defender_event_trace.json"
        ),
        "runs_postprocessed": len(audit_rows),
        "attacker_trace_events": len(attacker_trace_rows),
        "defender_trace_events": len(defender_trace_rows),
        "event_trace_events": len(event_trace_rows),
        "validation_passed": validation["validation_passed"],
    }


def _progress_path(experiment_root: Path) -> Path:
    return experiment_root.parent / "_progress" / "process_defender_capability_run.json"


def _manifest_path(experiment_root: Path, seed: int) -> Path:
    return (
        experiment_root.parent
        / "_manifests"
        / f"process_defender_capability_manifest_seed_{seed}.json"
    )


def _not_discovered_payload(testbed_id: str, seed: int) -> dict[str, Any]:
    root = _suite_root(testbed_id, "observation") / f"seed_{int(seed)}"
    return {
        "status": "not_discovered",
        "testbed": testbed_id,
        "suite": "observation",
        "seed": int(seed),
        "resolved_result_root": portable_path(root),
        "reason": (
            "No standalone legacy observation-only attack-suite ablation config was "
            "found. The active observation/interface ablation lives in the newer "
            "stateaware/* suite and is intentionally not part of --suite all-capability."
        ),
    }


def _set_wd_xdc_override(testbed_id: str, xdc: str | None) -> None:
    if testbed_id != "water_distribution":
        return
    requested = str(
        xdc
        or os.environ.get("ICS_AGENT_WD_PROCESS_DEFENDER_XDC")
        or os.environ.get("ICS_AGENT_SPHERE_XDC_OVERRIDE")
        or ""
    ).strip()
    if not requested:
        raise RuntimeError(
            "Water Distribution process-defender reruns require an explicit XDC "
            "via --xdc, ICS_AGENT_WD_PROCESS_DEFENDER_XDC, or "
            "ICS_AGENT_SPHERE_XDC_OVERRIDE."
        )
    os.environ["ICS_AGENT_SPHERE_XDC_OVERRIDE"] = requested


def _smoke_selection(
    *,
    source_config_path: Path,
    testbed_id: str,
    suite: str,
    requested_scenario: str | None,
    requested_condition: str | None,
) -> tuple[str | None, str | None]:
    scenario = requested_scenario
    condition = requested_condition
    if not requested_scenario:
        try:
            source_config = load_config(source_config_path)
            catalog = ScenarioCatalog(testbed_id=testbed_id)
            scenarios = catalog.resolve_selectors(
                list(source_config.get("scenario_selectors", []))
            )
            if scenarios:
                scenario = str(scenarios[0].catalog_id)
        except Exception:
            scenario = None
    if not requested_condition:
        try:
            source_config = load_config(source_config_path)
            orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
            conditions = orchestrator._enumerate_conditions(source_config)
            if conditions:
                condition = str(conditions[0].get("name"))
        except Exception:
            condition = None
    return scenario, condition


def _run_one_suite(
    args: argparse.Namespace,
    *,
    testbed_id: str,
    suite: str,
    emit: bool = True,
) -> dict[str, Any]:
    if suite == "observation":
        payload = _not_discovered_payload(testbed_id, int(args.seed))
        output = _suite_root(testbed_id, suite) / f"seed_{int(args.seed)}"
        write_json(output / "_manifests" / "observation_suite_not_discovered.json", payload)
        if emit:
            print(json.dumps(payload, indent=2))
        return payload

    source_config = args.config or _default_config_for(testbed_id, suite)
    if source_config is None:
        raise RuntimeError(f"no config registered for {testbed_id}/{suite}")
    source_config_path = resolve_runtime_config_path(source_config)
    selected_scenario = args.scenario
    selected_condition = args.only_condition
    if args.smoke:
        selected_scenario, selected_condition = _smoke_selection(
            source_config_path=source_config_path,
            testbed_id=testbed_id,
            suite=suite,
            requested_scenario=args.scenario,
            requested_condition=args.only_condition,
        )
    generated_config_path = _write_generated_config(
        source_config=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        seed=int(args.seed),
        scenario=selected_scenario,
        only_condition=selected_condition,
    )
    full_config_path = _write_generated_config(
        source_config=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        seed=int(args.seed),
        scenario=None,
        only_condition=None,
    )
    full_payload, full_plan = _experiment_plan(
        full_config_path,
        source_config_path=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        require_full_matrix=True,
    )
    run_payload, run_plan = _experiment_plan(
        generated_config_path,
        source_config_path=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
        require_full_matrix=False,
    )
    manifest_path = _manifest_path(full_plan["experiment_root"], int(args.seed))
    write_json(manifest_path, full_plan["manifest"])
    run_payload["process_defender_capability_manifest_path"] = portable_path(manifest_path)
    run_payload["full_suite_plan"] = {
        "planned_runs": full_payload["planned_runs"],
        "conditions": full_payload["conditions"],
        "scenarios": full_payload["scenarios"],
    }

    if args.validate_only or args.dry_run:
        payload = dict(run_payload)
        payload["mode"] = "dry_run" if args.dry_run else "validate_only"
        result = {"status": "validated", "plan": payload}
        if emit:
            print(json.dumps(result, indent=2))
        return result

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
    removed: list[str] = []
    try:
        if not args.skip_mrg_preflight:
            _preflight_mrg_auth(run_plan["config"])
        if not args.keep_failed_artifacts:
            removed = _remove_failed_or_incomplete_trials(
                run_plan,
                full_plan=full_plan,
            )
        print(
            json.dumps({"mode": "run_manifest", "plan": run_payload}, indent=2),
            file=sys.stderr,
            flush=True,
        )
        summary = ExperimentOrchestrator(testbed_id=testbed_id).run(
            generated_config_path
        )
        postprocess = _postprocess_results(
            run_plan["experiment_root"],
            plan=full_plan,
        )
        summary["runner_plan"] = run_payload
        summary["process_defender_postprocess"] = postprocess
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
        if emit:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun legacy attacker-capability ablations with the harmonized "
            "process-aware full-containment defender and critical-impact audit."
        )
    )
    parser.add_argument("--testbed", required=True, help="wt or wd")
    parser.add_argument(
        "--suite",
        default="all-capability",
        help="kt, interface, observation, or all-capability",
    )
    parser.add_argument(
        "--defender",
        default="process-aware",
        choices=["process-aware"],
        help="Only the harmonized process-aware defender is supported.",
    )
    parser.add_argument(
        "--critical-impact-audit",
        action="store_true",
        help="Write per-run critical-impact audit CSV/JSON outputs.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional source YAML. Only valid when --suite is not all-capability.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional attack scenario id or catalog id for focused reruns/tests.",
    )
    parser.add_argument(
        "--only-condition",
        default=None,
        help="Optional exact condition name for focused reruns/tests.",
    )
    parser.add_argument(
        "--xdc",
        default=None,
        help=(
            "Water Distribution XDC. Prefer a local ignored env value; do not "
            "commit concrete infrastructure identifiers."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the generated plan without starting the backend.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Alias for validation/inspection mode. Prints scenarios, conditions, "
            "defender mode, seed, XDC source, and planned run counts without "
            "starting experiments."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Inspect or run a single representative scenario/condition in the "
            "generated legacy plan. If --scenario or --only-condition is not "
            "provided, the first discovered scenario and the first enumerated "
            "condition are selected."
        ),
    )
    parser.add_argument(
        "--skip-mrg-preflight",
        action="store_true",
        help="Skip MRG/XDC auth preflight. Use only for local validation/tests.",
    )
    parser.add_argument(
        "--keep-failed-artifacts",
        action="store_true",
        help="Do not remove failed/incomplete selected trials before resuming.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    testbed_id = _canonical_testbed(args.testbed)
    suite = _canonical_suite(args.suite)
    _set_wd_xdc_override(testbed_id, args.xdc)
    if args.config and suite == "all-capability":
        raise RuntimeError("--config cannot be combined with --suite all-capability")
    suites = ["kt", "interface", "observation"] if suite == "all-capability" else [suite]
    summaries = [
        _run_one_suite(
            args,
            testbed_id=testbed_id,
            suite=current_suite,
            emit=suite != "all-capability",
        )
        for current_suite in suites
    ]
    if suite == "all-capability":
        print(
            json.dumps(
                {
                    "status": (
                        "completed"
                        if all(
                            item.get("status")
                            in {"completed", "validated", "not_discovered"}
                            for item in summaries
                        )
                        else "failed"
                    ),
                    "testbed": testbed_id,
                    "suite": suite,
                    "seed": int(args.seed),
                    "summaries": summaries,
                },
                indent=2,
            )
        )
    if all(item.get("status") in {"completed", "validated", "not_discovered"} for item in summaries):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
