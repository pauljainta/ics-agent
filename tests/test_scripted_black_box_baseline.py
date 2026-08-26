from __future__ import annotations

import json
from pathlib import Path

import ics_agent.experiments.water_treatment.scripted_black_box_baseline as scripted_baseline
from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.experiments.water_treatment.scripted_black_box_baseline import (
    CANONICAL_RESULT_ROOT,
    CONFIG_PATH,
    FAMILY_SPEC,
    RESUME_COMPATIBLE_CONFIG_HASH_PATHS,
    RUNS_PER_SEED,
    SEED,
    ScriptedBlackBoxBaselineRunner,
)
from test_config import (
    TEST_MERGE_SERVER,
    TEST_REVISION,
    TEST_TREATMENT_EXPERIMENT,
    TEST_TREATMENT_REALIZATION,
)


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


def test_scripted_black_box_profile_and_condition_resolve_distinct_from_llm_black_box() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")

    baseline = orchestrator.agent_profiles["black_box_attacker"]
    scripted = orchestrator.agent_profiles[FAMILY_SPEC.attacker_profile_id]
    condition = _condition_by_name(orchestrator, CONFIG_PATH, FAMILY_SPEC.family_name)

    assert scripted["role"] == "attacker"
    assert scripted["family"] == "agentic"
    assert scripted["knowledge_mode"] == "black_box"
    assert scripted["allowed_action_types"] == baseline["allowed_action_types"]
    assert scripted["behavior_model"]["policy"] == "scripted_black_box"
    assert scripted["behavior_model"]["id"] == "optionb_mitre_staged"
    assert condition["profiles"]["attacker"] == FAMILY_SPEC.attacker_profile_id
    assert condition["profiles"]["attacker"] != "black_box_attacker"
    assert str(CANONICAL_RESULT_ROOT).endswith(
        "Artifact/Results/Water-treatment/scripted_baselines/black_box_attacker_vs_full_containment_defender"
    )


def test_scripted_black_box_config_keeps_full_defender_resolution_unchanged() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    baseline = _condition_by_name(
        orchestrator,
        REPO_ROOT
        / "configs/experiments/water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
        "black_box_attacker__agentic_defender",
    )
    scripted = _condition_by_name(orchestrator, CONFIG_PATH, FAMILY_SPEC.family_name)

    assert baseline["profiles"]["defender"] == "agentic_defender"
    assert scripted["profiles"]["defender"] == "agentic_defender"
    assert scripted["effective_profiles"]["defender"] == baseline["effective_profiles"][
        "defender"
    ]


def test_scripted_black_box_runner_tracks_18_pending_runs(tmp_path: Path) -> None:
    runner = ScriptedBlackBoxBaselineRunner(tmp_path / "scripted-black-box")
    runner.ensure_root_layout()
    rows = runner._load_scenario_rows()
    runner._write_queue_and_resume(rows)
    resume_state = json.loads(runner.resume_state_path.read_text())

    assert len(rows) == RUNS_PER_SEED
    assert resume_state["completed_runs"] == 0
    assert resume_state["pending_runs"] == RUNS_PER_SEED
    assert resume_state["next_run"]["scenario_id"] == rows[0]["scenario_id"]


def test_scripted_black_box_runner_derives_required_live_preflight_refs() -> None:
    runner = ScriptedBlackBoxBaselineRunner(CANONICAL_RESULT_ROOT)
    refs = runner._required_preflight_refs(runner._runtime_config())

    assert set(refs) == {"merge_server", "realization", "materialization"}
    assert all(refs.values())


def test_scripted_black_box_policy_is_deterministic_across_fresh_agents() -> None:
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


def test_set_phase_completed_clears_stale_error_message(tmp_path: Path) -> None:
    runner = ScriptedBlackBoxBaselineRunner(tmp_path / "scripted-black-box")
    runner.ensure_root_layout()

    runner._set_phase("validated_optionb_environment", "failed", error_message="stale")
    runner._set_phase("validated_optionb_environment", "completed")

    payload = json.loads(runner.global_phase_status_path.read_text())
    phase = payload["phases"]["validated_optionb_environment"]
    assert phase["status"] == "completed"
    assert "error_message" not in phase


def test_validate_optionb_environment_reruns_live_preflight_with_cached_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = ScriptedBlackBoxBaselineRunner(tmp_path / "scripted-black-box")
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
        experiment_ref = TEST_TREATMENT_EXPERIMENT
        _realization_ref = TEST_TREATMENT_REALIZATION
        revision = TEST_REVISION
        _xdc_ref = "treatment-xdc-b"

    class FakeEnvironment:
        def __init__(self, _config):
            self._backend_impl = FakeBackend()

        def start(self) -> dict[str, str]:
            return {"status": "started"}

        def reset(self, _initial_conditions) -> dict[str, object]:
            return {
                "STATE_NAME": "IDLE",
                "post_reset_observation": _treatment_observation(
                    STATE_NAME="IDLE",
                    RW_Tank_Level=600.0,
                    UF_UFFT_Tank_Level=400.0,
                    ChemTreat_NaCl_Level=800.0,
                    ChemTreat_NaOCl_Level=800.0,
                    ChemTreat_HCl_Level=800.0,
                ),
            }

        def observe_flat(self) -> dict[str, float | str]:
            return _treatment_observation(
                STATE_NAME="IDLE",
                RW_Tank_Level=600.0,
                UF_UFFT_Tank_Level=400.0,
                ChemTreat_NaCl_Level=800.0,
                ChemTreat_NaOCl_Level=800.0,
                ChemTreat_HCl_Level=800.0,
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
    assert payload["validated_target"]["xdc_configured"] is True
    assert calls == {"whoami": 1, "show": 2}


def test_validate_optionb_environment_with_xdc_discovery_prefers_attached_wt1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = ScriptedBlackBoxBaselineRunner(tmp_path / "scripted-black-box")
    runner.ensure_root_layout()
    runner.preferred_xdc_short_names = ("treatment-xdc-a", "treatment-xdc-b")

    monkeypatch.setenv("ICS_AGENT_ALLOW_XDC_DISCOVERY", "1")
    monkeypatch.setattr(
        runner,
        "_check_mrg_transport",
        lambda merge_server=None, timeout_sec=20.0: "live-user",
    )
    monkeypatch.setattr(
        runner,
        "_listed_xdc_refs",
        lambda merge_server: (
            "XDC: treatment-xdc-b.live-user\nXDC: treatment-xdc-a.live-user\n",
            ["treatment-xdc-b.live-user", "treatment-xdc-a.live-user"],
        ),
    )

    def fake_inspect(
        *,
        candidate_short_names: list[str],
        merge_server: str,
        merge_user: str,
        realization_ref: str,
    ) -> list[dict[str, object]]:
        assert candidate_short_names[:2] == ["treatment-xdc-a", "treatment-xdc-b"]
        assert merge_server
        assert merge_user == "live-user"
        assert realization_ref
        return [
            {
                "short_name": "treatment-xdc-a",
                "ref": "treatment-xdc-a.live-user",
                "available": True,
                "healthy": True,
                "attached": True,
                "materialization": realization_ref,
                "status": "Ready",
            },
            {
                "short_name": "treatment-xdc-b",
                "ref": "treatment-xdc-b.live-user",
                "available": True,
                "healthy": False,
                "attached": False,
                "status": "Error",
            },
        ]

    def fake_attempt(
        *,
        xdc_short_name: str,
        validation_path: Path,
        timeout_sec: float,
    ) -> dict[str, object]:
        assert xdc_short_name == "treatment-xdc-a"
        assert timeout_sec == runner.validation_attempt_timeout_sec
        validation_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "validated_target": {
                        "xdc": "treatment-xdc-a.live-user",
                    },
                }
            )
        )
        return {
            "status": "passed",
            "short_name": xdc_short_name,
            "validation_path": str(validation_path),
        }

    monkeypatch.setattr(runner, "_inspect_xdc_candidates", fake_inspect)
    monkeypatch.setattr(runner, "_run_optionb_validation_attempt", fake_attempt)

    runner._validate_optionb_environment_with_xdc_discovery()

    payload = json.loads(runner.optionb_validation_path.read_text())
    phase = json.loads(runner.global_phase_status_path.read_text())["phases"][
        "validated_optionb_environment"
    ]
    assert payload["selected_xdc"]["short_name"] == "treatment-xdc-a"
    assert phase["status"] == "completed"
    assert phase["chosen_xdc"] == "treatment-xdc-a"


def test_resume_compatible_config_hash_update_allows_runner_only_drift() -> None:
    runner_path = next(iter(RESUME_COMPATIBLE_CONFIG_HASH_PATHS))
    actual_payload = {
        "experiment_version": "water_treatment_scripted_black_box_baseline_v1",
        "file_hashes": {
            runner_path: "old-runner-hash",
            "configs/profiles/shared.yaml": "shared-hash",
        },
    }
    expected_payload = {
        "experiment_version": "water_treatment_scripted_black_box_baseline_v1",
        "file_hashes": {
            runner_path: "new-runner-hash",
            "configs/profiles/shared.yaml": "shared-hash",
        },
    }

    assert ScriptedBlackBoxBaselineRunner._is_resume_compatible_config_hash_update(
        actual_payload,
        expected_payload,
    )


def test_resume_compatible_config_hash_update_rejects_behavioral_drift() -> None:
    runner_path = next(iter(RESUME_COMPATIBLE_CONFIG_HASH_PATHS))
    actual_payload = {
        "experiment_version": "water_treatment_scripted_black_box_baseline_v1",
        "file_hashes": {
            runner_path: "old-runner-hash",
            "ics_agent/agents/agentic.py": "old-agentic-hash",
        },
    }
    expected_payload = {
        "experiment_version": "water_treatment_scripted_black_box_baseline_v1",
        "file_hashes": {
            runner_path: "new-runner-hash",
            "ics_agent/agents/agentic.py": "new-agentic-hash",
        },
    }

    assert not ScriptedBlackBoxBaselineRunner._is_resume_compatible_config_hash_update(
        actual_payload,
        expected_payload,
    )
