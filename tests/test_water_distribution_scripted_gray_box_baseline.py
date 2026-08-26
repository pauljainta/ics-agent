from __future__ import annotations

import json
from pathlib import Path

import ics_agent.experiments.water_distribution.scripted_gray_box_baseline as scripted_baseline
from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.experiments.water_distribution.scripted_gray_box_baseline import (
    CANONICAL_RESULT_ROOT,
    CONFIG_PATH,
    FAMILY_SPEC,
    RUNS_PER_SEED,
    ScriptedGrayBoxBaselineRunner,
)
from test_config import TEST_DISTRIBUTION_REALIZATION, TEST_MERGE_SERVER


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


def _distribution_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 860.0,
        "Supply_Pump_Flow": 112.0,
        "Supply_NaOCl_Level": 800.0,
        "Supply_NH4Cl_Level": 800.0,
        "Grid_Elev_Tank_Level": 940.0,
        "Grid_Consum_Tank_Level": 860.0,
        "RWS_Tank_Level": 500.0,
        "Supply_Pump_Sts": 1.0,
        "Supply_Mixer_Sts": 1.0,
        "Supply_Tank_Valve_Sts": 0.0,
        "Grid_Elev_Valve_Sts": 0.0,
        "RWS_Pump_Sts": 1.0,
        "RWS_Tank_Valve_Sts": 0.0,
        "Supply_Tank_Valve_Cmd": 0.0,
        "Grid_Elev_Valve_Cmd": 0.0,
        "RWS_Tank_Valve_Cmd": 0.0,
        "Supply_Pump_Speed_Cmd": 45.0,
        "RWS_Pump_Speed_Cmd": 35.0,
        "Alarm_Supply_Tank_HH": 0.0,
        "SYS_IDLE": 0.0,
        "SYS_START": 0.0,
        "SYS_RUNNING": 1.0,
        "SYS_SHUTDOWN": 0.0,
        "HMI_Start_PB": 0.0,
        "HMI_Stop_PB": 0.0,
    }
    observation.update(overrides)
    return observation


def _raw_live_context() -> dict[str, object]:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-low-spoof")
    observation = _distribution_observation()
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


def test_distribution_scripted_gray_box_profile_and_condition_resolve_distinct_from_llm_gray_box() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/water_distribution_gray_box_vs_agentic_defender.yaml",
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
        "Artifact/Results/water_distribution_openplc/scripted_baselines/gray_box_attacker_vs_full_containment_defender"
    )


def test_distribution_scripted_gray_box_config_keeps_full_defender_resolution_unchanged() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/water_distribution_gray_box_vs_agentic_defender.yaml",
        "gray_box_attacker__agentic_defender",
    )
    scripted = _condition_by_name(orchestrator, CONFIG_PATH, FAMILY_SPEC.family_name)

    assert baseline["profiles"]["defender"] == "agentic_defender"
    assert scripted["profiles"]["defender"] == "agentic_defender"
    assert scripted["effective_profiles"]["defender"] == baseline["effective_profiles"][
        "defender"
    ]


def test_distribution_scripted_gray_box_runner_tracks_11_pending_runs(
    tmp_path: Path,
) -> None:
    runner = ScriptedGrayBoxBaselineRunner(tmp_path / "scripted-gray-box")
    runner.ensure_root_layout()
    rows = runner._load_scenario_rows()
    runner._write_queue_and_resume(rows)
    resume_state = json.loads(runner.resume_state_path.read_text())

    assert len(rows) == RUNS_PER_SEED
    assert resume_state["completed_runs"] == 0
    assert resume_state["pending_runs"] == RUNS_PER_SEED
    assert resume_state["next_run"]["scenario_id"] == rows[0]["scenario_id"]


def test_distribution_scripted_gray_box_runner_derives_required_live_preflight_refs() -> None:
    runner = ScriptedGrayBoxBaselineRunner(CANONICAL_RESULT_ROOT)
    refs = runner._required_preflight_refs(runner._runtime_config())

    assert set(refs) == {"merge_server", "realization", "materialization"}
    assert all(refs.values())


def test_distribution_scripted_gray_box_policy_is_deterministic_across_fresh_agents() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
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


def test_validate_optionb_environment_reruns_live_preflight_with_cached_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = ScriptedGrayBoxBaselineRunner(tmp_path / "scripted-gray-box")
    runner.ensure_root_layout()
    runner.optionb_validation_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "preflight": {
                    "merge_user": "cached-user",
                    "realization": {"id": "cached-realization"},
                    "materialization": {"id": "cached-materialization"},
                },
            }
        )
    )

    calls = {"whoami": 0, "show": 0}

    class FakeBackend:
        experiment_ref = "replace-with-local-experiment"
        _realization_ref = TEST_DISTRIBUTION_REALIZATION
        revision = "replace-with-local-revision"
        _xdc_ref = "distribution-xdc-b"

    class FakeEnvironment:
        def __init__(self, _config):
            self._backend_impl = FakeBackend()

        def start(self) -> dict[str, str]:
            return {"status": "started"}

        def reset(self, _initial_conditions) -> dict[str, object]:
            return {
                "STATE_NAME": "IDLE",
                "initial_state_validation": {"status": "passed"},
                "post_reset_observation": _distribution_observation(
                    STATE_NAME="IDLE",
                    Supply_Tank_Level=930.0,
                    Grid_Elev_Tank_Level=920.0,
                    RWS_Tank_Level=320.0,
                ),
            }

        def observe_flat(self) -> dict[str, float | str]:
            return _distribution_observation(
                STATE_NAME="IDLE",
                Supply_Tank_Level=930.0,
                Grid_Elev_Tank_Level=920.0,
                RWS_Tank_Level=320.0,
            )

        def inspect_surface(self, _action) -> dict[str, object]:
            return {"reconnaissance": {"surface": "tags"}}

        def stop(self) -> None:
            return None

    def fake_transport(*, merge_server: str | None = None, timeout_sec: float = 20.0) -> str:
        calls["whoami"] += 1
        assert merge_server
        assert timeout_sec == 20.0
        return "live-user"

    def fake_show(kind: str, ref: str, *, merge_server: str, timeout_sec: float = 20.0):
        calls["show"] += 1
        assert merge_server
        assert timeout_sec == 20.0
        return {"kind": kind, "ref": ref}

    monkeypatch.setattr(scripted_baseline, "OpenPLCEnvironment", FakeEnvironment)
    monkeypatch.setattr(scripted_baseline, "_SphereOpenPLCBackend", FakeBackend)
    monkeypatch.setattr(runner, "_check_mrg_transport", fake_transport)
    monkeypatch.setattr(runner, "_check_mrg_show_json", fake_show)

    runner._validate_optionb_environment()

    payload = json.loads(runner.optionb_validation_path.read_text())
    assert payload["status"] == "passed"
    assert payload["preflight"]["mrg_whoami_ok"] is True
    assert payload["preflight"]["realization_check_ok"] is True
    assert calls == {"whoami": 1, "show": 2}
