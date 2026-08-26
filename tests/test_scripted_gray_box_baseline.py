from __future__ import annotations

import json
from pathlib import Path

import ics_agent.water_treatment_scripted_gray_box_baseline as scripted_baseline
from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.water_treatment_scripted_gray_box_baseline import (
    CANONICAL_RESULT_ROOT,
    CONFIG_PATH,
    FAMILY_SPEC,
    RESUME_COMPATIBLE_CONFIG_HASH_PATHS,
    RUNS_PER_SEED,
    ScriptedGrayBoxBaselineRunner,
)
from test_config import TEST_MERGE_SERVER, TEST_TREATMENT_REALIZATION


REPO_ROOT = Path(__file__).resolve().parents[1]


def _condition_by_name(
    orchestrator: ExperimentOrchestrator,
    config_path: Path,
    condition_name: str,
) -> dict[str, object]:
    config = load_config(str(config_path))
    for condition in orchestrator._enumerate_conditions(config):
        if condition.get("name") == condition_name:
            return condition
    raise AssertionError(f"missing condition {condition_name}")


def _treatment_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "RW_Tank_Level": 640.0,
        "UF_UFFT_Tank_Level": 480.0,
        "ChemTreat_NaCl_Level": 780.0,
        "ChemTreat_NaOCl_Level": 760.0,
        "ChemTreat_HCl_Level": 750.0,
        "RW_Pump_Flow": 118.0,
        "RW_Pump_Speed_Cmd": 1.0,
        "RW_Tank_PR_Valve_Cmd": 1.0,
        "RW_Tank_P_Valve_Cmd": 0.0,
        "RW_Tank_PR_Valve_Sts": 1.0,
        "RW_Tank_P_Valve_Sts": 0.0,
        "RW_Pump_Sts": 1.0,
        "ChemTreat_NaCl_Valve_Cmd": 0.0,
        "ChemTreat_NaOCl_Valve_Cmd": 0.0,
        "ChemTreat_HCl_Valve_Cmd": 0.0,
        "ChemTreat_NaCl_Valve_Sts": 0.0,
        "ChemTreat_NaOCl_Valve_Sts": 0.0,
        "ChemTreat_HCl_Valve_Sts": 0.0,
        "SYS_IDLE": 0.0,
        "SYS_START": 0.0,
        "SYS_RUNNING": 1.0,
        "SYS_SHUTDOWN": 0.0,
        "P2_Selected": 0.0,
        "P2_NaCl_Cmd": 0.0,
        "P2_NaOCl_Cmd": 0.0,
        "P2_HCl_Cmd": 0.0,
        "Alarm_RW_Tank_LL": 0.0,
        "Alarm_RW_Tank_L": 0.0,
        "Alarm_RW_Tank_H": 0.0,
        "Alarm_RW_Tank_HH": 0.0,
        "HMI_Start_Active": 0.0,
        "HMI_Stop_Active": 0.0,
    }
    observation.update(overrides)
    return observation


def _raw_live_context() -> dict[str, object]:
    catalog = ScenarioCatalog()
    scenario = catalog.get("attack/wt-attack-sensor-rw-refill-spoof")
    observation = _treatment_observation()
    return {
        "scenario": scenario,
        "observation": observation,
        "observations": [observation],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "playbooks": {},
        "planned_events": {},
        "time_sec": 7.0,
    }


def test_scripted_gray_box_profile_and_condition_resolve_distinct_from_llm_gray_box() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/sphere_optionb_mitre_gray_box_vs_agentic_defender_fast_containment.yaml",
        "gray_box_attacker__agentic_defender",
    )
    scripted = _condition_by_name(orchestrator, CONFIG_PATH, FAMILY_SPEC.family_name)
    scripted_profile = orchestrator.agent_profiles[FAMILY_SPEC.attacker_profile_id]

    assert scripted_profile["role"] == "attacker"
    assert scripted_profile["family"] == "agentic"
    assert scripted_profile["knowledge_mode"] == "gray_box"
    assert scripted_profile["behavior_model"]["policy"] == "scripted_gray_box"
    assert scripted["profiles"]["attacker"] == FAMILY_SPEC.attacker_profile_id
    assert scripted["profiles"]["attacker"] != "gray_box_attacker"
    assert scripted["effective_profiles"]["attacker"]["allowed_action_types"] == baseline[
        "effective_profiles"
    ]["attacker"]["allowed_action_types"]
    assert str(CANONICAL_RESULT_ROOT).endswith(
        "Artifact/Results/Water-treatment/scripted_baselines/gray_box_attacker_vs_full_containment_defender"
    )


def test_scripted_gray_box_config_keeps_full_defender_resolution_unchanged() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/sphere_optionb_mitre_gray_box_vs_agentic_defender_fast_containment.yaml",
        "gray_box_attacker__agentic_defender",
    )
    scripted = _condition_by_name(orchestrator, CONFIG_PATH, FAMILY_SPEC.family_name)

    assert baseline["profiles"]["defender"] == "agentic_defender"
    assert scripted["profiles"]["defender"] == "agentic_defender"
    assert scripted["effective_profiles"]["defender"] == baseline["effective_profiles"][
        "defender"
    ]


def test_scripted_gray_box_runner_tracks_18_pending_runs(tmp_path: Path) -> None:
    runner = ScriptedGrayBoxBaselineRunner(tmp_path / "scripted-gray-box")
    runner.ensure_root_layout()
    rows = runner._load_scenario_rows()
    runner._write_queue_and_resume(rows)
    resume_state = json.loads(runner.resume_state_path.read_text())

    assert len(rows) == RUNS_PER_SEED
    assert resume_state["completed_runs"] == 0
    assert resume_state["pending_runs"] == RUNS_PER_SEED
    assert resume_state["next_run"]["scenario_id"] == rows[0]["scenario_id"]


def test_scripted_gray_box_runner_derives_required_live_preflight_refs() -> None:
    runner = ScriptedGrayBoxBaselineRunner(CANONICAL_RESULT_ROOT)
    refs = runner._required_preflight_refs(runner._runtime_config())

    assert set(refs) == {"merge_server", "realization", "materialization"}
    assert all(refs.values())


def test_scripted_gray_box_policy_is_deterministic_across_fresh_agents() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profile = orchestrator.agent_profiles[FAMILY_SPEC.attacker_profile_id]
    context = _raw_live_context()

    first_agent = AgenticRoleAgent(FAMILY_SPEC.attacker_profile_id, profile)
    first_agent.reset(context["scenario"])
    first_agent._state["mitre_transition_notes"] = {
        "staged_entry_context",
        "process_understanding",
    }
    first = first_agent.step(context)

    second_agent = AgenticRoleAgent(FAMILY_SPEC.attacker_profile_id, profile)
    second_agent.reset(context["scenario"])
    second_agent._state["mitre_transition_notes"] = {
        "staged_entry_context",
        "process_understanding",
    }
    second = second_agent.step(context)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].action == second[0].action
    assert first[0].summary == second[0].summary
    assert "deterministic" in first[0].labels
    assert "gray_box" in first[0].labels


def test_resume_compatible_config_hash_update_allows_runner_only_drift() -> None:
    runner_path = next(iter(RESUME_COMPATIBLE_CONFIG_HASH_PATHS))
    actual_payload = {
        "experiment_version": "water_treatment_scripted_gray_box_baseline_v1",
        "file_hashes": {
            runner_path: "old-runner-hash",
            "configs/agent_profiles.yaml": "shared-hash",
        },
    }
    expected_payload = {
        "experiment_version": "water_treatment_scripted_gray_box_baseline_v1",
        "file_hashes": {
            runner_path: "new-runner-hash",
            "configs/agent_profiles.yaml": "shared-hash",
        },
    }

    assert ScriptedGrayBoxBaselineRunner._is_resume_compatible_config_hash_update(
        actual_payload,
        expected_payload,
    )
