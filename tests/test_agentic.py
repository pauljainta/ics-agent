from __future__ import annotations

from pathlib import Path

from ics_agent.mitre import (
    ATTACKER_PHASES,
    DEFENDER_PHASES,
    OPTIONB_MITRE_MODEL_ID,
    OPTIONB_STAGE_PATH,
    optionb_mitre_context,
)
from ics_agent.agents.agentic import AgenticRoleAgent, CachedQwenDecisionEngine
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT
from ics_agent.types import ScenarioDefinition


def test_optionb_mitre_context_preserves_staged_runtime_model() -> None:
    empty_context = {"decisions": [], "executed_actions": [], "alerts": []}

    attacker_context = optionb_mitre_context(role="attacker", context=empty_context)
    defender_context = optionb_mitre_context(role="defender", context=empty_context)

    assert attacker_context["phase_model"] == OPTIONB_MITRE_MODEL_ID
    assert attacker_context["staged_path"] == OPTIONB_STAGE_PATH
    assert attacker_context["phase_order"] == ATTACKER_PHASES
    assert attacker_context["current_phase"] == ATTACKER_PHASES[0]
    assert attacker_context["direct_plant_shortcuts_forbidden"] is True

    assert defender_context["phase_model"] == OPTIONB_MITRE_MODEL_ID
    assert defender_context["staged_path"] == OPTIONB_STAGE_PATH
    assert defender_context["phase_order"] == DEFENDER_PHASES
    assert defender_context["current_phase"] == DEFENDER_PHASES[0]


def test_configure_reproducibility_keeps_seeded_run_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "20260623")
    monkeypatch.setenv(
        "QWEN_MODEL_PATH",
        str((REPO_ROOT / "models" / "local_models" / "qwen3.5-9b").resolve()),
    )
    orchestrator = ExperimentOrchestrator()
    payload = orchestrator._configure_reproducibility(
        config={
            "reproducibility": {"seed": 20260623},
            "_path": Path("concrete-containment.yaml"),
        },
        conditions=[
            {
                "profiles": {
                    "attacker": "gray_box_attacker",
                    "defender": "agentic_defender",
                    "observer": "agentic_observer",
                    "planner": "scripted_planner",
                    "operator": "scripted_operator",
                    "offline_analysis": "scripted_offline_analyst",
                }
            }
        ],
    )

    assert payload["seed"] == 20260623
    assert payload["pythonhashseed"] == "20260623"
    assert payload["suite_decoding"]
    assert payload["experiment_config_path"] == "concrete-containment.yaml"
    attacker_decoding = payload["suite_decoding"]["attacker-gray_box_attacker__defender-agentic_defender__observer-agentic_observer__offline_analysis-scripted_offline_analyst__operator-scripted_operator__planner-scripted_planner"]["attacker"]
    assert attacker_decoding["context_limit"] == 262144
    assert attacker_decoding["top_p"] is None


def test_parse_output_recovers_truncated_json_value() -> None:
    raw = (
        '{"summary":"Stop process","candidate_id":"{\\"type\\":\\"hmi_stop\\"}",'
        '"confidence":0.85,"diagnosis":"supply_overflow_risk'
    )

    payload = CachedQwenDecisionEngine._parse_output(raw)

    assert payload is not None
    assert payload["summary"] == "Stop process"
    assert payload["candidate_id"] == '{"type":"hmi_stop"}'
    assert payload["diagnosis"] == "supply_overflow_risk"


def test_parse_output_recovers_after_truncated_trailing_field() -> None:
    raw = (
        '{"summary":"Observe jump","candidate_id":null,"confidence":0.52,'
        '"diagnosis":"sensor_or_process_jump","labels'
    )

    payload = CachedQwenDecisionEngine._parse_output(raw)

    assert payload is not None
    assert payload["summary"] == "Observe jump"
    assert payload["candidate_id"] is None
    assert payload["diagnosis"] == "sensor_or_process_jump"


def test_agentic_payload_null_confidence_does_not_abort_decision_bridge() -> None:
    agent = AgenticRoleAgent(
        "gray_box_attacker",
        {
            "role": "attacker",
            "knowledge_mode": "gray_box",
            "allowed_action_types": ["inspect_surface", "noop"],
            "llm": {
                "backend": "qwen_langchain",
                "fatal_on_parse_error": False,
            },
        },
    )
    agent.reset(
        ScenarioDefinition(
            catalog_id="attack/example",
            suite="attack",
            scenario_id="example",
            execution_path=Path("attack/example.yaml"),
            execution={},
            metadata={"testbed_id": "water_distribution"},
            testbed_id="water_distribution",
        )
    )

    decisions = agent._payload_to_decisions(
        {
            "summary": "Hold while preserving the current probe state.",
            "candidate_id": None,
            "confidence": None,
            "diagnosis": None,
            "labels": None,
            "decision_type": "note",
        },
        {
            "time_sec": 5.5,
            "candidate_actions": [{"id": "noop", "action": {"type": "noop"}}],
            "observation": {"STATE_NAME": "RUNNING"},
            "executed_actions": [],
            "recent_actions": [],
            "decisions": [],
            "alerts": [],
        },
    )

    assert len(decisions) == 1
    assert decisions[0].confidence == 0.65
    assert decisions[0].labels == ["agentic", "attacker"]


def test_white_box_commit_guardrail_forces_action_after_note_only_model() -> None:
    class NoteOnlyEngine:
        last_mode = "unit-test"
        last_error = None
        black_box_candidate_map: dict[str, dict] = {}

        def decide(self, context):
            self.context = context
            return {
                "summary": "Continue staged reasoning before manipulating the process.",
                "candidate_id": None,
                "confidence": 0.55,
                "diagnosis": None,
                "labels": ["agentic", "attacker"],
                "decision_type": "note",
            }

    agent = AgenticRoleAgent(
        "white_box_attacker",
        {
            "role": "attacker",
            "knowledge_mode": "white_box",
            "white_box_commit_guardrail": True,
            "allowed_action_types": [
                "inspect_surface",
                "hmi_start",
                "hmi_stop",
                "hmi_estop",
                "set_sensor",
                "set_actuator",
                "noop",
            ],
            "behavior_model": {
                "id": "optionb_mitre_staged",
                "policy": "agentic",
                "white_box_commit_guardrail": True,
            },
            "llm": {
                "backend": "qwen_langchain",
                "fatal_on_parse_error": False,
            },
            "memory": {"attacker_candidate_window": 8},
            "action_cooldown_sec": 0.5,
            "reasoning_cooldown_sec": 0.0,
            "reasoning_refresh_sec": 0.0,
        },
    )
    agent.engine = NoteOnlyEngine()
    scenario = ScenarioDefinition(
        catalog_id="attack/wd-grid-low-spoof",
        suite="attack",
        scenario_id="wd-grid-low-spoof",
        execution_path=Path("attack/wd-grid-low-spoof.yaml"),
        execution={"process_id": "WD-Grid"},
        metadata={
            "testbed_id": "water_distribution",
            "safety_bounds": {
                "Supply_Tank_Level": {"min": 0, "max": 1500},
                "Grid_Elev_Tank_Level": {"min": 0, "max": 2000},
                "RWS_Tank_Level": {"min": 0, "max": 1500},
            },
        },
        testbed_id="water_distribution",
    )
    agent.reset(scenario)

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 0.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "Supply_Tank_Level": 920,
                "Grid_Elev_Tank_Level": 940,
                "RWS_Tank_Level": 700,
                "Alarm_Supply_Tank_HH": False,
            },
            "observations": [],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "planned_events": {},
        }
    )

    assert decisions
    assert decisions[0].decision_type == "action"
    assert decisions[0].action is not None
    assert decisions[0].action["type"] in {"set_sensor", "set_actuator", "hmi_stop", "hmi_estop"}
    assert "commitment_guardrail" in decisions[0].labels


def test_cached_qwen_engine_finalizes_confidence_and_labels_without_backend() -> None:
    engine = CachedQwenDecisionEngine(
        {
            "role": "defender",
            "knowledge_mode": "white_box",
            "llm": {
                "backend": "qwen_langchain",
                "fatal_on_parse_error": False,
            },
        },
        "defender",
    )

    payload = engine._finalize_payload(
        {
            "summary": "Raise containment.",
            "candidate_id": None,
            "confidence": None,
            "diagnosis": "Sensor or process jump",
            "labels": None,
            "decision_type": "alert",
        },
        {"candidate_actions": []},
    )

    assert payload["confidence"] == 0.55
    assert payload["labels"] == ["agentic", "defender"]
    assert payload["diagnosis"] == "sensor_or_process_jump"
