from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from ics_agent.config import load_config, resolve_runtime_config_path
from ics_agent.experiments.process_defender_capability import _postprocess_results
from ics_agent.experiments.stateaware_oi_matrix import (
    _postprocess_stateaware_results,
    _preflight_mrg_auth,
)
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT, portable_path, results_root
from ics_agent.reporting import write_json
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import dump_yaml, stable_hash, utc_now


TESTBED_ALIASES = {
    "wt": "water_treatment",
    "treatment": "water_treatment",
    "water_treatment": "water_treatment",
    "wd": "water_distribution",
    "distribution": "water_distribution",
    "water_distribution": "water_distribution",
}

SUITE_ALIASES = {
    "all": "all",
    "both": "all",
    "legacy": "legacy",
    "legacy_attack": "legacy",
    "attack": "legacy",
    "stateaware": "stateaware",
    "state-aware": "stateaware",
    "new": "stateaware",
}

AUTHORITY_ALIASES = {
    "both": ("u1_process_only", "u2_containment_only"),
    "all": ("u1_process_only", "u2_containment_only"),
    "u1": ("u1_process_only",),
    "process": ("u1_process_only",),
    "process_only": ("u1_process_only",),
    "u1_process_only": ("u1_process_only",),
    "u2": ("u2_containment_only",),
    "containment": ("u2_containment_only",),
    "containment_only": ("u2_containment_only",),
    "u2_containment_only": ("u2_containment_only",),
}

SUITE_SPECS = {
    ("water_treatment", "legacy"): {
        "source_config": (
            "water_treatment_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
        ),
        "source_condition": "full_target_semantics__agentic_defender",
        "experiment_name": "water_treatment_defender_authority_legacy_normO_normI",
        "scenario_selectors": ["attack/*"],
        "expected_scenarios": 18,
        "postprocess": "legacy",
    },
    ("water_treatment", "stateaware"): {
        "source_config": "water_treatment_stateaware_oi_matrix_vs_agentic_full_containment_defender.yaml",
        "source_condition": "stateaware_fullKt_normO_normI_processDefender",
        "experiment_name": "water_treatment_defender_authority_stateaware_normO_normI",
        "scenario_selectors": ["stateaware/*"],
        "expected_scenarios": 6,
        "postprocess": "stateaware",
    },
    ("water_distribution", "legacy"): {
        "source_config": (
            "water_distribution_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
        ),
        "source_condition": "full_target_semantics__agentic_defender",
        "experiment_name": "water_distribution_defender_authority_legacy_normO_normI",
        "scenario_selectors": ["attack/*"],
        "expected_scenarios": 11,
        "postprocess": "legacy",
    },
    ("water_distribution", "stateaware"): {
        "source_config": (
            "water_distribution_stateaware_oi_matrix_generic_policy_v2_vs_agentic_full_containment_defender.yaml"
        ),
        "source_condition": "wd_stateaware_full_semantics_normO_normI",
        "experiment_name": "water_distribution_defender_authority_stateaware_normO_normI",
        "scenario_selectors": ["stateaware/*"],
        "expected_scenarios": 4,
        "postprocess": "stateaware",
    },
}

PROCESS_ONLY_ACTIONS = {
    "water_treatment": [
        "hmi_stop",
        "hmi_estop",
        "select_process",
        "noop",
    ],
    "water_distribution": [
        "hmi_stop",
        "hmi_estop",
        "noop",
    ],
}

CONTAINMENT_ONLY_ACTIONS = {
    "water_treatment": [
        "disable_hmi_session",
        "isolate_engineering_host",
        "block_sensor_write",
        "block_process2_manual_path",
        "quarantine_signal_source",
        "noop",
    ],
    "water_distribution": [
        "disable_hmi_session",
        "isolate_engineering_host",
        "block_sensor_write",
        "block_actuator_write",
        "quarantine_signal_source",
        "noop",
    ],
}

AUTHORITY_METADATA = {
    "u1_process_only": {
        "formal_name": "Process-only defender",
        "authority": "U_def = U_process",
        "description": (
            "May execute plant/process interventions but cannot block the "
            "attacker-facing operational surfaces."
        ),
    },
    "u2_containment_only": {
        "formal_name": "Containment-only defender",
        "authority": "U_def = U_contain",
        "description": (
            "May cut off attacker-facing operational surfaces but cannot "
            "directly change plant operating state."
        ),
    },
}

WT_XDC_ENV = "ICS_AGENT_WT_PROCESS_DEFENDER_XDC"
WD_XDC_ENV = "ICS_AGENT_WD_PROCESS_DEFENDER_XDC"
GLOBAL_XDC_ENV = "ICS_AGENT_SPHERE_XDC_OVERRIDE"


def _canonical_testbed(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in TESTBED_ALIASES:
        valid = ", ".join(sorted(TESTBED_ALIASES))
        raise RuntimeError(f"unsupported testbed `{value}`; valid values: {valid}")
    return TESTBED_ALIASES[key]


def _canonical_suite(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in SUITE_ALIASES:
        valid = ", ".join(sorted(SUITE_ALIASES))
        raise RuntimeError(f"unsupported suite `{value}`; valid values: {valid}")
    return SUITE_ALIASES[key]


def _canonical_authorities(value: str) -> tuple[str, ...]:
    key = str(value).strip().lower().replace("-", "_")
    if key not in AUTHORITY_ALIASES:
        valid = ", ".join(sorted(AUTHORITY_ALIASES))
        raise RuntimeError(f"unsupported defender authority `{value}`; valid values: {valid}")
    return AUTHORITY_ALIASES[key]


def _load_local_xdc_env_defaults() -> None:
    if os.environ.get("ICS_AGENT_DISABLE_LOCAL_ENV") == "1":
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    keys = {
        WT_XDC_ENV,
        WD_XDC_ENV,
        GLOBAL_XDC_ENV,
    }
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in keys or os.environ.get(key):
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ[key] = value


def _xdc_value(value: Any) -> str:
    return str(value or "").strip()


def _configure_xdc_for_testbed(
    args: argparse.Namespace,
    *,
    testbed_id: str,
    selected_testbeds: tuple[str, ...],
) -> dict[str, str | bool]:
    """Set the global infrastructure override to the correct testbed XDC.

    Infrastructure resolution uses one global environment variable. This runner
    is testbed-scoped, so it normalizes that global variable immediately before
    each testbed's configs are loaded.
    """

    _load_local_xdc_env_defaults()
    single_testbed = len(selected_testbeds) == 1
    if testbed_id == "water_treatment":
        requested = (
            _xdc_value(args.wt_xdc)
            or (_xdc_value(args.xdc) if single_testbed else "")
            or _xdc_value(os.environ.get(WT_XDC_ENV))
        )
        if requested:
            os.environ[GLOBAL_XDC_ENV] = requested
            return {"testbed": testbed_id, "xdc": requested, "source": "wt_override"}
        previous_global = _xdc_value(os.environ.get(GLOBAL_XDC_ENV))
        if previous_global:
            os.environ.pop(GLOBAL_XDC_ENV, None)
            return {
                "testbed": testbed_id,
                "xdc": "",
                "source": "profile_default",
                "cleared_global_override": True,
            }
        return {"testbed": testbed_id, "xdc": "", "source": "profile_default"}

    if testbed_id == "water_distribution":
        requested = (
            _xdc_value(args.wd_xdc)
            or _xdc_value(args.xdc)
            or _xdc_value(os.environ.get(WD_XDC_ENV))
        )
        if not requested:
            raise RuntimeError(
                "Water Distribution process-defender reruns require an explicit WD XDC "
                "via --wd-xdc, --xdc, ICS_AGENT_WD_PROCESS_DEFENDER_XDC, or a local "
                ".env entry. Do not use a WT XDC for WD runs."
            )
        os.environ[GLOBAL_XDC_ENV] = requested
        return {"testbed": testbed_id, "xdc": requested, "source": "wd_override"}

    raise RuntimeError(f"unsupported testbed `{testbed_id}`")


def _testbed_result_root(testbed_id: str) -> Path:
    if testbed_id == "water_treatment":
        return results_root() / "Water-treatment"
    if testbed_id == "water_distribution":
        return results_root() / "water_distribution_openplc"
    raise RuntimeError(f"unsupported testbed `{testbed_id}`")


def _authority_result_root(testbed_id: str) -> Path:
    return (
        _testbed_result_root(testbed_id)
        / "process-defender-critical-impact-v2"
        / "defender-authority-ablation"
    )


def _authority_actions(testbed_id: str, authority: str) -> list[str]:
    if authority == "u1_process_only":
        return list(PROCESS_ONLY_ACTIONS[testbed_id])
    if authority == "u2_containment_only":
        return list(CONTAINMENT_ONLY_ACTIONS[testbed_id])
    raise RuntimeError(f"unsupported authority `{authority}`")


def _authority_condition_name(source_condition: str, authority: str) -> str:
    base = str(source_condition).replace("__agentic_defender", "")
    if authority == "u1_process_only":
        suffix = "process_only_defender"
    elif authority == "u2_containment_only":
        suffix = "containment_only_defender"
    else:
        suffix = authority
    return f"{base}__{suffix}"


def _notes_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _patch_defender_override(
    original: dict[str, Any],
    *,
    testbed_id: str,
    authority: str,
) -> dict[str, Any]:
    patched = deepcopy(original)
    metadata = AUTHORITY_METADATA[authority]
    patched["allowed_action_types"] = _authority_actions(testbed_id, authority)
    patched["process_physics_only"] = True
    patched["defender_observation_mode"] = "process_physics_only"
    patched["defer_containment_until_attacker_effect"] = True
    behavior = dict(patched.get("behavior_model") or {})
    behavior["id"] = "optionb_mitre_staged"
    behavior["policy"] = "agentic"
    behavior["defender_authority"] = authority
    behavior["defender_observation_contract"] = "process_physics_only"
    behavior["response_mode"] = metadata["formal_name"]
    notes = _notes_list(behavior.get("notes"))
    note = (
        f"{metadata['formal_name']} authority ablation: {metadata['authority']}. "
        f"{metadata['description']}"
    )
    if note not in notes:
        notes.append(note)
    behavior["notes"] = notes
    patched["behavior_model"] = behavior
    patched["defender_authority"] = authority
    patched["defender_authority_family"] = (
        "process_only" if authority == "u1_process_only" else "containment_only"
    )
    return patched


def _source_condition(raw_config: dict[str, Any], condition_name: str) -> dict[str, Any]:
    conditions = raw_config.get("conditions")
    if not isinstance(conditions, list):
        raise RuntimeError("source config has no conditions list")
    matches = [
        item
        for item in conditions
        if isinstance(item, dict) and str(item.get("name")) == condition_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"source condition `{condition_name}` not found exactly once"
        )
    return matches[0]


def _scenario_selector(scenario: str | None, suite: str) -> str | None:
    if not scenario:
        return None
    text = str(scenario).strip()
    if "/" in text:
        return text
    prefix = "stateaware" if suite == "stateaware" else "attack"
    return f"{prefix}/{text}"


def _build_generated_config(
    *,
    source_config_path: Path,
    raw_config: dict[str, Any],
    source_condition_name: str,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    authorities: tuple[str, ...],
) -> dict[str, Any]:
    generated = deepcopy(raw_config)
    spec = SUITE_SPECS[(testbed_id, suite)]
    selected_source = _source_condition(raw_config, source_condition_name)

    conditions: list[dict[str, Any]] = []
    for authority in authorities:
        condition = deepcopy(selected_source)
        condition["name"] = _authority_condition_name(source_condition_name, authority)
        overrides = dict(condition.get("profile_overrides") or {})
        defender_override = dict(overrides.get("defender") or {})
        overrides["defender"] = _patch_defender_override(
            defender_override,
            testbed_id=testbed_id,
            authority=authority,
        )
        condition["profile_overrides"] = overrides
        conditions.append(condition)

    generated["conditions"] = conditions
    generated["testbed"] = testbed_id
    generated["experiment_name"] = str(spec["experiment_name"])
    generated["experiment_id"] = (
        f"{generated['experiment_name']}_seed_{int(seed)}_"
        f"{stable_hash({'source': portable_path(source_config_path), 'seed': int(seed), 'suite': suite, 'authorities': authorities}, length=8)}"
    )
    generated["resume"] = True
    generated["use_canonical_result_root"] = True
    generated["results_root"] = f"../seed_{int(seed)}"
    generated["scenario_selectors"] = list(spec["scenario_selectors"])
    selector = _scenario_selector(scenario, suite)
    if selector:
        generated["scenario_selectors"] = [selector]
    reproducibility = dict(generated.get("reproducibility") or {})
    reproducibility["seed"] = int(seed)
    generated["reproducibility"] = reproducibility

    metadata = dict(generated.get("study_metadata") or {})
    notes = _notes_list(metadata.get("notes"))
    note = (
        "Generated defender-authority U1/U2 rerun. Attacker profile is fixed at "
        "full target-system context with normalized observation and normalized "
        "interface; only defender intervention authority is varied."
    )
    if note not in notes:
        notes.append(note)
    metadata.update(
        {
            "family": f"{generated['experiment_name']}_critical_impact",
            "source_config": portable_path(source_config_path),
            "source_condition": source_condition_name,
            "authority_conditions": list(authorities),
            "attacker_capability": "full_Kt_normO_normI",
            "defender_authority_ablation": "U1_process_only_vs_U2_containment_only",
            "critical_impact_audit": (
                "legacy attack suite uses process-defender critical-impact audit; "
                "state-aware suite uses state-aware critical-impact contract"
            ),
            "notes": notes,
        }
    )
    generated["study_metadata"] = metadata
    return generated


def _generated_config_path(
    *,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    authorities: tuple[str, ...],
) -> Path:
    root = _authority_result_root(testbed_id)
    parts = [
        testbed_id,
        suite,
        "defender_authority",
        f"seed_{int(seed)}",
        "_".join(authorities),
    ]
    if scenario:
        parts.append(str(scenario).replace("/", "_").replace(" ", "_"))
    return root / "_generated_configs" / ("_".join(parts) + ".yaml")


def _write_generated_config(
    *,
    testbed_id: str,
    suite: str,
    seed: int,
    scenario: str | None,
    authorities: tuple[str, ...],
) -> tuple[Path, Path]:
    spec = SUITE_SPECS[(testbed_id, suite)]
    source_config_path = resolve_runtime_config_path(str(spec["source_config"]))
    from ics_agent.utils import load_yaml

    raw_config = load_yaml(source_config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"expected mapping at {source_config_path}")
    generated = _build_generated_config(
        source_config_path=source_config_path,
        raw_config=raw_config,
        source_condition_name=str(spec["source_condition"]),
        testbed_id=testbed_id,
        suite=suite,
        seed=seed,
        scenario=scenario,
        authorities=authorities,
    )
    path = _generated_config_path(
        testbed_id=testbed_id,
        suite=suite,
        seed=seed,
        scenario=scenario,
        authorities=authorities,
    )
    dump_yaml(path, generated)
    return path, source_config_path


def _resolved_experiment_root(config: dict[str, Any]) -> Path:
    base_dir = Path(config["_path"]).parent
    root = Path(config.get("results_root", str(results_root())))
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    return root / str(config["experiment_name"])


def _enumerate_plan(
    config_path: Path,
    *,
    source_config_path: Path,
    testbed_id: str,
    suite: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    source_config = load_config(source_config_path)
    catalog = ScenarioCatalog(testbed_id=testbed_id)
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    scenarios = catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
    conditions = orchestrator._enumerate_conditions(config)
    source_conditions = orchestrator._enumerate_conditions(source_config)
    spec = SUITE_SPECS[(testbed_id, suite)]

    expected_count = int(spec["expected_scenarios"])
    if len(scenarios) != expected_count and not len(scenarios) == 1:
        raise RuntimeError(
            f"{testbed_id}/{suite} expected {expected_count} scenarios, got {len(scenarios)}"
        )
    attacker_profiles = [
        condition["effective_profiles"]["attacker"] for condition in conditions
    ]
    defender_profiles = [
        condition["effective_profiles"]["defender"] for condition in conditions
    ]
    if not attacker_profiles:
        raise RuntimeError("generated config has no attacker profiles")
    first_attacker = attacker_profiles[0]
    if any(profile != first_attacker for profile in attacker_profiles[1:]):
        raise RuntimeError("attacker profile changed across defender-authority conditions")

    condition_entries: list[dict[str, Any]] = []
    for condition, defender_profile in zip(conditions, defender_profiles):
        allowed = list(defender_profile.get("allowed_action_types", []))
        authority = str(defender_profile.get("defender_authority"))
        condition_entries.append(
            {
                "condition": str(condition.get("name")),
                "authority": authority,
                "defender_profile": str(condition.get("profiles", {}).get("defender")),
                "defender_allowed_action_types": allowed,
                "process_actions_enabled": [
                    action
                    for action in allowed
                    if action in {"hmi_stop", "hmi_estop", "select_process"}
                ],
                "containment_actions_enabled": [
                    action
                    for action in allowed
                    if action
                    in {
                        "disable_hmi_session",
                        "isolate_engineering_host",
                        "block_sensor_write",
                        "block_actuator_write",
                        "block_process2_manual_path",
                        "quarantine_signal_source",
                    }
                ],
            }
        )

    manifest = {
        "experiment_name": str(config["experiment_name"]),
        "testbed": testbed_id,
        "suite": suite,
        "seed": (config.get("reproducibility") or {}).get("seed"),
        "source_config": portable_path(source_config_path),
        "source_condition": str(spec["source_condition"]),
        "scenario_count": len(scenarios),
        "condition_count": len(conditions),
        "repeats": int(config.get("repeats", 1)),
        "planned_runs": len(scenarios) * len(conditions) * int(config.get("repeats", 1)),
        "scenario_ids": [scenario.catalog_id for scenario in scenarios],
        "conditions": condition_entries,
        "fixed_attacker_capability": {
            "K_t": "full_target_semantics",
            "O_A": "normalized_observation",
            "I_A": "normalized_interface",
            "seed": (config.get("reproducibility") or {}).get("seed"),
            "profile": str(conditions[0].get("profiles", {}).get("attacker")),
            "allowed_action_types": list(first_attacker.get("allowed_action_types", [])),
            "target_semantics": dict(first_attacker.get("target_semantics") or {}),
            "stateaware_oi": dict(first_attacker.get("stateaware_oi") or {}),
        },
        "varied_defender_dimension": {
            "U1_process_only": PROCESS_ONLY_ACTIONS[testbed_id],
            "U2_containment_only": CONTAINMENT_ONLY_ACTIONS[testbed_id],
        },
        "invariance_checks": {
            "single_attacker_profile_across_conditions": True,
            "same_scenarios_across_conditions": True,
            "same_seed_across_conditions": True,
            "same_testbed_runtime": True,
            "only_defender_allowed_action_types_vary": True,
        },
        "source_condition_count": len(source_conditions),
    }
    experiment_root = _resolved_experiment_root(config)
    plan = {
        "config": config,
        "testbed_id": testbed_id,
        "suite": suite,
        "scenarios": scenarios,
        "conditions": conditions,
        "repeats": int(config.get("repeats", 1)),
        "experiment_root": experiment_root,
        "manifest": manifest,
    }
    payload = {
        "config_path": portable_path(config_path),
        "source_config_path": portable_path(source_config_path),
        "experiment_name": str(config["experiment_name"]),
        "testbed": testbed_id,
        "suite": suite,
        "seed": manifest["seed"],
        "scenario_count": manifest["scenario_count"],
        "condition_count": manifest["condition_count"],
        "planned_runs": manifest["planned_runs"],
        "conditions": [entry["condition"] for entry in condition_entries],
        "scenarios": manifest["scenario_ids"],
        "result_root": portable_path(experiment_root.parent),
        "resolved_experiment_root": portable_path(experiment_root),
        "manifest": manifest,
    }
    return payload, plan


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


def _planned_trial_keys(plan: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for condition in plan["conditions"]:
        condition_name = str(condition.get("name"))
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


def _remove_failed_or_incomplete_trials(plan: dict[str, Any]) -> list[str]:
    root = plan["experiment_root"]
    if not isinstance(root, Path) or not root.exists():
        return []
    selected_keys = _planned_trial_keys(plan)
    removed: list[str] = []
    for run_dir in _trial_dirs(root):
        key = _trial_key(root, run_dir)
        if key is None or key not in selected_keys:
            continue
        if _read_status(run_dir) == "completed":
            continue
        shutil.rmtree(run_dir)
        _prune_empty_trial_parents(run_dir, root)
        removed.append(portable_path(run_dir))
    return removed


def _progress_path(testbed_id: str, seed: int) -> Path:
    return (
        _authority_result_root(testbed_id)
        / f"seed_{int(seed)}"
        / "_progress"
        / "defender_authority_u1_u2_run.json"
    )


def _manifest_path(testbed_id: str, seed: int) -> Path:
    return (
        _authority_result_root(testbed_id)
        / f"seed_{int(seed)}"
        / "_manifests"
        / "defender_authority_u1_u2_manifest.json"
    )


def _selected_suites(value: str) -> tuple[str, ...]:
    suite = _canonical_suite(value)
    if suite == "all":
        return ("legacy", "stateaware")
    return (suite,)


def _selected_testbeds(value: str) -> tuple[str, ...]:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"both", "all"}:
        return ("water_treatment", "water_distribution")
    return (_canonical_testbed(value),)


def _run_suite(
    args: argparse.Namespace,
    *,
    testbed_id: str,
    suite: str,
    authorities: tuple[str, ...],
) -> dict[str, Any]:
    config_path, source_config_path = _write_generated_config(
        testbed_id=testbed_id,
        suite=suite,
        seed=int(args.seed),
        scenario=args.scenario,
        authorities=authorities,
    )
    payload, plan = _enumerate_plan(
        config_path,
        source_config_path=source_config_path,
        testbed_id=testbed_id,
        suite=suite,
    )
    if args.dry_run or args.validate_only:
        return {"status": "validated", "plan": payload}

    removed: list[str] = []
    try:
        if not args.skip_mrg_preflight:
            _preflight_mrg_auth(plan["config"])
        if not args.keep_failed_artifacts:
            removed = _remove_failed_or_incomplete_trials(plan)
        print(
            json.dumps({"mode": "run_manifest", "plan": payload}, indent=2),
            file=sys.stderr,
            flush=True,
        )
        summary = ExperimentOrchestrator(testbed_id=testbed_id).run(config_path)
        if SUITE_SPECS[(testbed_id, suite)]["postprocess"] == "stateaware":
            postprocess = _postprocess_stateaware_results(
                plan["experiment_root"],
                plan=plan,
            )
        else:
            postprocess = _postprocess_results(plan["experiment_root"], plan=plan)
        summary["runner_plan"] = payload
        summary["defender_authority_postprocess"] = postprocess
        summary["removed_failed_or_incomplete_trials"] = removed
        return summary
    except Exception as exc:
        payload["error"] = str(exc)
        payload["removed_failed_or_incomplete_trials"] = removed
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run U1 process-only and U2 containment-only defender-authority "
            "ablations for full-Kt/norm-O/norm-I attackers on WT/WD critical-impact suites."
        )
    )
    parser.add_argument(
        "--testbed",
        required=True,
        help="wt, wd, or both",
    )
    parser.add_argument(
        "--suite",
        default="all",
        help="legacy, stateaware, or all",
    )
    parser.add_argument(
        "--defender-authority",
        default="both",
        help="both, u1/process-only, or u2/containment-only",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario id for a focused/resume test.",
    )
    parser.add_argument(
        "--xdc",
        default=None,
        help=(
            "XDC override for the selected testbed. For --testbed both, prefer "
            "--wt-xdc and --wd-xdc. Prefer local .env values; do not commit "
            "concrete infrastructure identifiers."
        ),
    )
    parser.add_argument(
        "--wt-xdc",
        default=None,
        help=(
            "Water Treatment XDC override. May also be set with "
            "ICS_AGENT_WT_PROCESS_DEFENDER_XDC in local .env."
        ),
    )
    parser.add_argument(
        "--wd-xdc",
        default=None,
        help=(
            "Water Distribution XDC override. May also be set with "
            "ICS_AGENT_WD_PROCESS_DEFENDER_XDC in local .env."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-mrg-preflight", action="store_true")
    parser.add_argument("--keep-failed-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authorities = _canonical_authorities(args.defender_authority)
    testbeds = _selected_testbeds(args.testbed)
    suites = _selected_suites(args.suite)
    summaries: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for testbed_id in testbeds:
        xdc_selection = _configure_xdc_for_testbed(
            args,
            testbed_id=testbed_id,
            selected_testbeds=testbeds,
        )
        progress_path = _progress_path(testbed_id, int(args.seed))
        write_json(
            progress_path,
            {
                "status": "started" if not (args.dry_run or args.validate_only) else "validating",
                "started_utc": utc_now(),
                "runner": Path(__file__).name,
                "testbed": testbed_id,
                "seed": int(args.seed),
                "suites": list(suites),
                "authorities": list(authorities),
                "xdc_selection": xdc_selection,
            },
        )
        try:
            for suite in suites:
                summary = _run_suite(
                    args,
                    testbed_id=testbed_id,
                    suite=suite,
                    authorities=authorities,
                )
                summaries.append(summary)
                plan = summary.get("plan") or summary.get("runner_plan") or {}
                manifest = dict(plan.get("manifest") or {})
                if manifest:
                    manifests.append(manifest)
            write_json(
                _manifest_path(testbed_id, int(args.seed)),
                {
                    "testbed": testbed_id,
                    "seed": int(args.seed),
                    "authorities": list(authorities),
                    "suites": list(suites),
                    "manifests": [
                        item for item in manifests if item.get("testbed") == testbed_id
                    ],
                },
            )
            write_json(
                progress_path,
                {
                    "status": "validated"
                    if args.dry_run or args.validate_only
                    else "completed",
                    "updated_utc": utc_now(),
                    "runner": Path(__file__).name,
                    "testbed": testbed_id,
                    "seed": int(args.seed),
                    "suites": list(suites),
                    "authorities": list(authorities),
                    "xdc_selection": xdc_selection,
                    "manifest_path": portable_path(_manifest_path(testbed_id, int(args.seed))),
                },
            )
        except Exception as exc:
            write_json(
                progress_path,
                {
                    "status": "failed",
                    "updated_utc": utc_now(),
                    "runner": Path(__file__).name,
                    "testbed": testbed_id,
                    "seed": int(args.seed),
                    "suites": list(suites),
                    "authorities": list(authorities),
                    "xdc_selection": xdc_selection,
                    "error": str(exc),
                },
            )
            raise

    result = {
        "status": (
            "validated"
            if args.dry_run or args.validate_only
            else "completed"
            if all(item.get("status") in {"completed", "validated"} for item in summaries)
            else "finished_with_errors"
        ),
        "seed": int(args.seed),
        "testbeds": list(testbeds),
        "suites": list(suites),
        "authorities": list(authorities),
        "summaries": summaries,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"completed", "validated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
