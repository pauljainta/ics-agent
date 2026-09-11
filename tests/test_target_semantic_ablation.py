from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ics_agent.agents.agentic import (
    AgenticRoleAgent,
    TARGET_INTERFACE_VARIANTS,
    TARGET_SEMANTIC_RESOURCE_KEYS,
)
from ics_agent.agents.action_space import normalize_action
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


CONFIGS = {
    "water_treatment": (
        "water_treatment_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
    ),
    "water_distribution": (
        "water_distribution_target_semantic_ablation_vs_agentic_full_containment_defender.yaml"
    ),
}

INTERFACE_CONFIGS = {
    "water_treatment": (
        "water_treatment_target_interface_ablation_vs_agentic_full_containment_defender.yaml"
    ),
    "water_distribution": (
        "water_distribution_target_interface_ablation_vs_agentic_full_containment_defender.yaml"
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
WD_RUNNER = (
    "scripts/"
    "run_water_distribution_agentic_target_semantic_ablation_attacker_vs_"
    "agentic_full_containment_defender.py"
)
WT_RUNNER = (
    "scripts/"
    "run_water_treatment_agentic_target_semantic_ablation_attacker_vs_"
    "agentic_full_containment_defender.py"
)
WD_INTERFACE_RUNNER = (
    "scripts/"
    "run_water_distribution_agentic_target_interface_ablation_attacker_vs_"
    "agentic_full_containment_defender.py"
)
WT_INTERFACE_RUNNER = (
    "scripts/"
    "run_water_treatment_agentic_target_interface_ablation_attacker_vs_"
    "agentic_full_containment_defender.py"
)

EXPECTED_VARIANTS = [
    "no_target_semantics",
    "topology_only",
    "asset_only",
    "control_effect_only",
    "safety_only",
    "surface_only",
    "full",
    "irrelevant_context",
]


def _condition_map(testbed_id: str) -> dict[str, dict[str, Any]]:
    config = load_config(CONFIGS[testbed_id])
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    return {
        str(condition["name"]): condition
        for condition in orchestrator._enumerate_conditions(config)
    }


def _attacker_without_variant(profile: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(profile)
    fixed.pop("target_semantics", None)
    return fixed


def _interface_condition_map(testbed_id: str) -> dict[str, dict[str, Any]]:
    config = load_config(INTERFACE_CONFIGS[testbed_id])
    orchestrator = ExperimentOrchestrator(testbed_id=testbed_id)
    return {
        str(condition["name"]): condition
        for condition in orchestrator._enumerate_conditions(config)
    }


def _attacker_without_interface_variant(profile: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(profile)
    target_semantics = fixed.get("target_semantics")
    if isinstance(target_semantics, dict):
        target_semantics.pop("interface_variant", None)
        target_semantics.pop("interaction_interface", None)
    fixed.pop("target_interface_variant", None)
    return fixed


@pytest.mark.parametrize("testbed_id", sorted(CONFIGS))
def test_target_semantic_ablation_configs_vary_only_attacker_semantic_payload(
    testbed_id: str,
) -> None:
    conditions = _condition_map(testbed_id)
    assert sorted(conditions) == sorted(
        f"{variant}__agentic_defender"
        if variant != "full"
        else "full_target_semantics__agentic_defender"
        for variant in EXPECTED_VARIANTS
    )

    first_condition = next(iter(conditions.values()))
    baseline_attacker = _attacker_without_variant(
        first_condition["effective_profiles"]["attacker"]
    )
    baseline_defender = first_condition["effective_profiles"]["defender"]
    baseline_observer = first_condition["effective_profiles"]["observer"]

    observed_variants = []
    for condition in conditions.values():
        profiles = condition["effective_profiles"]
        attacker = profiles["attacker"]
        observed_variants.append(attacker["target_semantics"]["variant"])
        assert attacker["knowledge_mode"] == "target_semantic_ablation"
        assert _attacker_without_variant(attacker) == baseline_attacker
        assert profiles["defender"] == baseline_defender
        assert profiles["observer"] == baseline_observer

    assert sorted(observed_variants) == sorted(EXPECTED_VARIANTS)


def _observation(testbed_id: str) -> dict[str, Any]:
    if testbed_id == "water_distribution":
        return {
            "STATE_NAME": "RUNNING",
            "elapsed_sec": 18.0,
            "Supply_Tank_Level": 940.0,
            "Supply_Pump_Speed": 35.0,
            "Supply_Pump_Sts": 1.0,
            "Grid_Elev_Tank_Level": 1550.0,
            "Grid_Consum_Tank_Level": 820.0,
            "Grid_Elev_Valve_Cmd": 0.0,
            "Grid_Elev_Valve_Sts": 0.0,
            "RWS_Tank_Level": 500.0,
            "RWS_Pump_Speed": 0.0,
            "RWS_Pump_Sts": 0.0,
            "Alarm_Supply_Tank_HH": 0.0,
        }
    return {
        "STATE_NAME": "RUNNING",
        "elapsed_sec": 18.0,
        "RW_Tank_Level": 600.0,
        "UF_UFFT_Tank_Level": 400.0,
        "ChemTreat_NaCl_Level": 800.0,
        "ChemTreat_NaOCl_Level": 800.0,
        "ChemTreat_HCl_Level": 800.0,
        "RW_Pump_Sts": 1.0,
        "RW_Pump_Flow": 100.0,
        "RW_Tank_PR_Valve_Sts": 0.0,
        "P2_Selected": 0.0,
        "ChemTreat_NaCl_Valve_Sts": 0.0,
        "ChemTreat_NaOCl_Valve_Sts": 0.0,
        "ChemTreat_HCl_Valve_Sts": 0.0,
        "Alarm_RW_Tank_LL": 0.0,
        "Alarm_RW_Tank_HH": 0.0,
    }


def _scenario(testbed_id: str) -> Any:
    scenario_id = (
        "attack/wd-attack-grid-valve-force-open"
        if testbed_id == "water_distribution"
        else "attack/wt-attack-sensor-rw-refill-spoof"
    )
    return ScenarioCatalog(testbed_id=testbed_id).get(scenario_id)


def _runtime_context(testbed_id: str, *, manipulation_ready: bool = False) -> dict[str, Any]:
    observation = _observation(testbed_id)
    context = {
        "scenario": _scenario(testbed_id),
        "observation": observation,
        "observations": [observation],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "remaining_playbook": [],
        "playbooks": {},
        "time_sec": 18.0,
    }
    if manipulation_ready:
        phases = [
            "staged_entry_context",
            "discovery",
            "process_understanding",
            "control_path_qualification",
        ]
        context["decisions"] = [
            {
                "role": "attacker",
                "time_sec": float(index),
                "metadata": {
                    "phase_model": "optionb_mitre_staged",
                    "phase_family": "attacker",
                    "mitre_phase": phase,
                    "mitre_phase_order": index,
                },
            }
            for index, phase in enumerate(phases, start=1)
        ]
    return context


def _semantic_context(testbed_id: str, variant: str) -> dict[str, Any]:
    condition_name = (
        "full_target_semantics__agentic_defender"
        if variant == "full"
        else f"{variant}__agentic_defender"
    )
    profile = _condition_map(testbed_id)[condition_name]["effective_profiles"][
        "attacker"
    ]
    agent = AgenticRoleAgent("target_semantic_ablation_audit", profile)
    scenario = _scenario(testbed_id)
    agent.reset(scenario)
    return agent._build_online_context(_runtime_context(testbed_id))


def _fixed_actor_context(context: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(context)
    fixed.pop("system_knowledge", None)
    fixed.pop("knowledge_ablation", None)
    return fixed


@pytest.mark.parametrize("testbed_id", sorted(CONFIGS))
def test_target_semantic_ablation_context_keeps_observations_and_candidates_fixed(
    testbed_id: str,
) -> None:
    contexts = {
        variant: _semantic_context(testbed_id, variant)
        for variant in EXPECTED_VARIANTS
    }
    baseline = _fixed_actor_context(contexts["no_target_semantics"])

    for variant, context in contexts.items():
        assert context["scenario_id"] == "target-semantic-ablation-live"
        assert _fixed_actor_context(context) == baseline, variant

    serialized_candidates = {
        json.dumps(context["candidate_actions"], sort_keys=True)
        for context in contexts.values()
    }
    serialized_observations = {
        json.dumps(context["observation"], sort_keys=True)
        for context in contexts.values()
    }
    assert len(serialized_candidates) == 1
    assert len(serialized_observations) == 1


@pytest.mark.parametrize("testbed_id", sorted(CONFIGS))
def test_target_semantic_ablation_payload_switches_only_requested_resources(
    testbed_id: str,
) -> None:
    none_context = _semantic_context(testbed_id, "no_target_semantics")
    full_context = _semantic_context(testbed_id, "full")
    safety_context = _semantic_context(testbed_id, "safety_only")
    irrelevant_context = _semantic_context(testbed_id, "irrelevant_context")

    assert none_context["system_knowledge"]["target_semantics"] == {}
    assert set(full_context["system_knowledge"]["target_semantics"]) == {
        "topology",
        "asset",
        "control_effect",
        "safety",
        "surface",
    }
    assert set(safety_context["system_knowledge"]["target_semantics"]) == {
        "safety"
    }
    assert "irrelevant_context_control" in irrelevant_context["system_knowledge"]
    assert irrelevant_context["system_knowledge"]["target_semantics"] == {}

    manifest = full_context["system_knowledge"]["capability_manifest"]
    target_flags = manifest["varied"]["K_A_t"]
    assert set(target_flags) == {
        "K_asset",
        "K_effect",
        "K_safety",
        "K_surface",
        "K_topo",
    }
    assert all(target_flags.values())
    assert tuple(TARGET_SEMANTIC_RESOURCE_KEYS) == (
        "topology",
        "asset",
        "effect",
        "safety",
        "surface",
    )


@pytest.mark.parametrize("testbed_id", sorted(CONFIGS))
def test_target_semantic_ablation_prompt_uses_single_shared_prompt(
    testbed_id: str,
) -> None:
    conditions = _condition_map(testbed_id)
    prompt_files = {
        condition["effective_profiles"]["attacker"]["llm"]["prompt_file"]
        for condition in conditions.values()
    }
    assert prompt_files == {"configs/prompts/target_semantic_ablation_attacker.md"}
    prompt_path = Path("configs/prompts/target_semantic_ablation_attacker.md")
    assert prompt_path.exists()


@pytest.mark.parametrize("testbed_id", sorted(CONFIGS))
def test_target_semantic_ablation_resolves_opaque_candidate_id_to_real_action(
    testbed_id: str,
) -> None:
    condition_name = "full_target_semantics__agentic_defender"
    profile = _condition_map(testbed_id)[condition_name]["effective_profiles"][
        "attacker"
    ]
    agent = AgenticRoleAgent("target_semantic_ablation_resolver_audit", profile)
    scenario = _scenario(testbed_id)
    agent.reset(scenario)
    context = agent._build_online_context(
        _runtime_context(testbed_id, manipulation_ready=True)
    )

    candidate = next(
        item
        for item in context["candidate_actions"]
        if item.get("id") != "bb_noop"
    )
    decision = agent.engine._finalize_payload(
        {
            "summary": "Choose a listed opaque candidate.",
            "candidate_id": candidate["id"],
            "confidence": 0.7,
            "diagnosis": "bounded_probe",
            "labels": ["agentic", "attacker"],
            "decision_type": "action",
        },
        context,
    )

    assert decision["candidate_id"] == candidate["id"]
    assert decision["action"] != candidate["action"]
    assert decision["action"]["type"] in {
        "inspect_surface",
        "hmi_start",
        "hmi_stop",
        "hmi_estop",
        "select_process",
        "set_manual_valve",
        "set_sensor",
        "set_actuator",
    }


def test_water_distribution_target_semantic_ablation_runner_validate_only() -> None:
    result = subprocess.run(
        [sys.executable, WD_RUNNER, "--validate-only"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["testbed"] == "water_distribution"
    assert payload["seed"] == 101
    assert payload["resume"] is True
    assert payload["scenario_count"] == 11
    assert payload["condition_count"] == len(EXPECTED_VARIANTS)
    assert payload["planned_runs"] == 88
    assert (
        "wall_clock_time_to_detection_from_attack_start_sec"
        in payload["metric_keys_recorded_by_shared_pipeline"]
    )


def test_water_treatment_target_semantic_ablation_runner_validate_only_with_seed() -> None:
    result = subprocess.run(
        [sys.executable, WT_RUNNER, "--validate-only", "--seed", "101"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["mode"] == "validate_only"
    assert payload["testbed"] == "water_treatment"
    assert payload["seed"] == 101
    assert payload["resume"] is True
    assert payload["scenario_count"] == 18
    assert payload["condition_count"] == len(EXPECTED_VARIANTS)
    assert payload["planned_runs"] == 144
    assert "seed_101" in payload["resolved_experiment_root"]
    assert (
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
        in payload["metric_keys_recorded_by_shared_pipeline"]
    )


@pytest.mark.parametrize("testbed_id", sorted(INTERFACE_CONFIGS))
def test_target_interface_ablation_configs_vary_only_attacker_interface(
    testbed_id: str,
) -> None:
    conditions = _interface_condition_map(testbed_id)
    assert sorted(conditions) == [
        "full_target_semantics_legacy_white_interface__agentic_defender",
        "full_target_semantics_normalized_interface__agentic_defender",
    ]

    first_condition = next(iter(conditions.values()))
    baseline_attacker = _attacker_without_interface_variant(
        first_condition["effective_profiles"]["attacker"]
    )
    baseline_defender = first_condition["effective_profiles"]["defender"]
    baseline_observer = first_condition["effective_profiles"]["observer"]

    observed_interfaces = []
    for condition in conditions.values():
        profiles = condition["effective_profiles"]
        attacker = profiles["attacker"]
        target_semantics = attacker["target_semantics"]
        observed_interfaces.append(target_semantics["interface_variant"])
        assert attacker["knowledge_mode"] == "target_semantic_ablation"
        assert target_semantics["variant"] == "full"
        assert _attacker_without_interface_variant(attacker) == baseline_attacker
        assert profiles["defender"] == baseline_defender
        assert profiles["observer"] == baseline_observer

    assert set(observed_interfaces) == TARGET_INTERFACE_VARIANTS


def _interface_context(testbed_id: str, interface_variant: str) -> tuple[Any, dict[str, Any]]:
    condition_name = (
        f"full_target_semantics_{interface_variant}__agentic_defender"
    )
    profile = _interface_condition_map(testbed_id)[condition_name]["effective_profiles"][
        "attacker"
    ]
    agent = AgenticRoleAgent("target_interface_ablation_audit", profile)
    scenario = _scenario(testbed_id)
    agent.reset(scenario)
    context = agent._build_online_context(
        _runtime_context(testbed_id, manipulation_ready=True)
    )
    return agent, context


@pytest.mark.parametrize("testbed_id", sorted(INTERFACE_CONFIGS))
def test_target_interface_ablation_changes_candidate_presentation_only(
    testbed_id: str,
) -> None:
    normalized_agent, normalized_context = _interface_context(
        testbed_id,
        "normalized_interface",
    )
    legacy_agent, legacy_context = _interface_context(
        testbed_id,
        "legacy_white_interface",
    )

    assert normalized_context["observation"] == legacy_context["observation"]
    assert normalized_context["recent_observations"] == legacy_context[
        "recent_observations"
    ]
    assert normalized_context["system_knowledge"]["target_semantics"] == (
        legacy_context["system_knowledge"]["target_semantics"]
    )
    assert normalized_context["system_knowledge"]["variant"] == "full"
    assert legacy_context["system_knowledge"]["interface_variant"] == (
        "legacy_white_interface"
    )
    assert normalized_context["candidate_actions"] != legacy_context[
        "candidate_actions"
    ]

    normalized_action_types = {
        normalize_action(item["action"])["type"]
        for item in normalized_agent._state["black_box_candidate_map"].values()
    }
    legacy_action_types = {
        normalize_action(item["action"])["type"]
        for item in legacy_agent._state["black_box_candidate_map"].values()
    }
    allowed_action_types = set(
        _interface_condition_map(testbed_id)[
            "full_target_semantics_normalized_interface__agentic_defender"
        ]["effective_profiles"]["attacker"]["allowed_action_types"]
    )
    assert normalized_action_types <= allowed_action_types
    assert legacy_action_types <= allowed_action_types
    assert legacy_action_types

    candidate = next(
        item for item in legacy_context["candidate_actions"] if item.get("id") != "noop"
    )
    decision = legacy_agent.engine._finalize_payload(
        {
            "summary": "Choose a listed rich-interface candidate.",
            "candidate_id": candidate["id"],
            "confidence": 0.7,
            "diagnosis": "bounded_probe",
            "labels": ["agentic", "attacker"],
            "decision_type": "action",
        },
        legacy_context,
    )
    assert decision["candidate_id"] == candidate["id"]
    assert decision["action"] == candidate["action"]


@pytest.mark.parametrize(
    "testbed_id,runner,expected_scenarios,expected_runs",
    [
        ("water_distribution", WD_INTERFACE_RUNNER, 11, 22),
        ("water_treatment", WT_INTERFACE_RUNNER, 18, 36),
    ],
)
def test_target_interface_ablation_runner_validate_only(
    testbed_id: str,
    runner: str,
    expected_scenarios: int,
    expected_runs: int,
) -> None:
    result = subprocess.run(
        [sys.executable, runner, "--validate-only", "--seed", "101"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    manifest = payload["interface_capability_manifest"]

    assert payload["mode"] == "validate_only"
    assert payload["testbed"] == testbed_id
    assert payload["seed"] == 101
    assert payload["resume"] is True
    assert payload["scenario_count"] == expected_scenarios
    assert payload["condition_count"] == 2
    assert payload["planned_runs"] == expected_runs
    assert manifest["varied_dimension"] == {
        "I_A": ["normalized_interface", "legacy_white_interface"]
    }
    assert all(manifest["invariance_checks"].values())
    assert "interface_capability_manifest_path" in payload
    assert (
        "wall_clock_time_to_effective_containment_cutoff_from_attack_start_sec"
        in payload["metric_keys_recorded_by_shared_pipeline"]
    )


def test_target_interface_ablation_runner_can_validate_only_one_condition() -> None:
    condition = "full_target_semantics_legacy_white_interface__agentic_defender"
    result = subprocess.run(
        [
            sys.executable,
            WT_INTERFACE_RUNNER,
            "--validate-only",
            "--seed",
            "101",
            "--only-condition",
            condition,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["mode"] == "validate_only"
    assert payload["testbed"] == "water_treatment"
    assert payload["condition_count"] == 1
    assert payload["planned_runs"] == 18
    assert payload["conditions"] == [condition]
    assert payload["execution_filter"]["only_condition"] == condition
    assert payload["execution_filter"]["full_comparison_planned_runs"] == 36
    assert payload["interface_capability_manifest"]["condition_count"] == 2
    assert payload["interface_capability_manifest"]["planned_runs"] == 36
