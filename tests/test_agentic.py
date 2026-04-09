from __future__ import annotations

import pytest

from ics_agent.agents.action_space import action_key
from ics_agent.agents.agentic import AgenticRoleAgent, CachedQwenDecisionEngine
from ics_agent.scenarios import ScenarioCatalog


def _context(scenario):
    return {
        "scenario": scenario,
        "time_sec": 0.0,
        "observation": {
            "STATE_NAME": "IDLE",
            "P2_Selected": 0,
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
        },
        "observations": [],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "playbooks": {
            "operator": [
                {
                    "time_sec": 0.0,
                    "action": {"type": "hmi_start", "process_id": "WT-P1"},
                }
            ]
        },
    }


def test_agentic_resolves_valid_candidate_id_to_action(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-p1-short-nominal")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "_decide_with_langgraph",
        lambda context: agent.engine._finalize_payload(
            {
                "summary": "Start the plant now.",
                "candidate_id": action_key(
                    {"type": "hmi_start", "process_id": "WT-P1"}
                ),
                "decision_type": "action",
                "diagnosis": "nominal_operation",
            },
            context,
        ),
    )

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["backend"] == "qwen_langgraph"


def test_agentic_action_role_falls_back_to_due_playbook_when_backend_unavailable(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-p1-short-nominal")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "_decide_with_langgraph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'langgraph'")
        ),
    )
    monkeypatch.setattr(
        agent.engine,
        "_decide_with_langchain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'langchain'")
        ),
    )

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["fallback"] == "backend_error"


def test_agentic_matches_explicit_action_back_to_candidate(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-p1-short-nominal")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "_decide_with_langgraph",
        lambda context: agent.engine._finalize_payload(
            {
                "summary": "Start the plant now.",
                "action": {"type": "hmi_start", "process_id": "WT-P1"},
                "decision_type": "action",
                "diagnosis": "nominal_operation",
            },
            context,
        ),
    )

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["backend"] == "qwen_langgraph"


def test_agentic_does_not_backfill_ground_truth_diagnosis(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-rw-refill-spoof")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "_decide_with_langgraph",
        lambda context: agent.engine._finalize_payload(
            {
                "summary": "Pause and monitor.",
                "candidate_id": action_key({"type": "noop"}),
                "decision_type": "note",
            },
            context,
        ),
    )

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].diagnosis is None


def test_agentic_parser_handles_think_tags_and_fenced_json() -> None:
    payload = CachedQwenDecisionEngine._parse_output(
        """
        <think>
        reason silently
        </think>
        ```json
        {"summary":"ok","candidate_id":"{\\"type\\":\\"noop\\"}"}
        ```
        """
    )

    assert payload is not None
    assert payload["summary"] == "ok"


def test_action_roles_only_expose_due_playbook_actions() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-hmi-forced-stop")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    context = {
        "scenario": scenario,
        "time_sec": 0.0,
        "observation": {
            "STATE_NAME": "IDLE",
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
        },
        "observations": [],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "playbooks": {
            "operator": [
                {
                    "time_sec": 5.0,
                    "action": {"type": "hmi_start", "process_id": "WT-P1"},
                }
            ]
        },
    }

    early = agent._build_online_context(context)
    assert early["candidate_actions"] == [
        {
            "id": action_key({"type": "noop"}),
            "summary": "Wait until the next scheduled playbook action is due.",
            "action": {"type": "noop"},
            "source": "action_space",
        }
    ]

    context["time_sec"] = 5.0
    due = agent._build_online_context(context)
    assert due["candidate_actions"][0]["action"] == {
        "type": "hmi_start",
        "process_id": "WT-P1",
    }
    assert due["candidate_actions"][1]["action"] == {"type": "noop"}


def test_due_playbook_action_falls_back_when_model_only_notes(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-hmi-forced-stop")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Wait and monitor.",
            "candidate_id": action_key({"type": "noop"}),
            "decision_type": "note",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 5.0,
            "observation": {
                "STATE_NAME": "IDLE",
                "RW_Tank_Level": 600,
                "UF_UFFT_Tank_Level": 400,
            },
            "observations": [],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {
                "operator": [
                    {
                        "time_sec": 5.0,
                        "action": {"type": "hmi_start", "process_id": "WT-P1"},
                    }
                ]
            },
        }
    )

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["fallback"] == "due_playbook_action"


def test_due_redundant_playbook_action_is_skipped_when_state_already_satisfied() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-hh-alarm")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "allowed_action_types": ["hmi_stop", "set_sensor", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_attacker", profile)
    agent.reset(scenario)

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 30.0,
            "observation": {
                "STATE_NAME": "SHUTDOWN",
                "RW_Tank_Level": 1231,
                "UF_UFFT_Tank_Level": 426,
            },
            "observations": [],
            "executed_actions": [
                {
                    "role": "attacker",
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 1250,
                    },
                },
                {
                    "role": "defender",
                    "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                },
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {
                "attacker": [
                    {
                        "time_sec": 15.0,
                        "action": {
                            "type": "set_sensor",
                            "name": "RW_Tank_Level",
                            "value": 1250,
                        },
                    },
                    {
                        "time_sec": 30.0,
                        "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                    },
                ]
            },
        }
    )

    assert decisions == []


def test_defender_does_not_offer_process2_selection_in_p1_scenarios() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-hmi-forced-stop")
    profile = {
        "role": "defender",
        "family": "agentic",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_defender", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 12.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 520,
                "UF_UFFT_Tank_Level": 460,
                "P2_Selected": 0,
            },
            "observations": [],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert not any(
        item["action"].get("type") == "select_process"
        for item in context["candidate_actions"]
        if item.get("action")
    )


def test_offline_analysis_falls_back_when_backend_errors() -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-rw-refill-spoof")
    profile = {
        "role": "offline_analysis",
        "family": "agentic",
        "mode": "offline",
        "allowed_action_types": ["noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_offline", profile)
    agent.reset(scenario)
    agent.engine.last_error = "backend unavailable"
    agent.engine.decide = lambda context: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("backend unavailable")
    )

    decisions = agent.finalize(
        {
            "scenario": scenario,
            "duration_sec": 12.0,
            "metrics": {"time_to_detection_sec": 1.0},
            "passed": True,
            "observations": [],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
        }
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "analysis"
    assert decisions[0].labels == ["offline", "agentic", "backend-error"]
    assert decisions[0].metadata["error"] == "backend unavailable"


def test_agentic_parser_handles_yaml_like_payload() -> None:
    payload = CachedQwenDecisionEngine._parse_output(
        """
        summary: hold
        candidate_id: '{"type":"noop"}'
        confidence: 0.4
        diagnosis: null
        labels:
          - observer
        decision_type: note
        """
    )

    assert payload is not None
    assert payload["summary"] == "hold"
    assert payload["decision_type"] == "note"


def test_live_context_is_sanitized_and_enriched() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "defender",
        "family": "agentic",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_defender", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 16.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 400,
                "UF_UFFT_Tank_Level": 420,
                "RW_Tank_PR_Valve_Cmd": 1,
                "RW_Tank_PR_Valve_Sts": 1,
                "RW_Pump_Sts": 1,
                "Alarm_RW_Tank_LL": 1,
                "P2_Selected": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 600},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 400},
            ],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["scenario_id"] != scenario.scenario_id
    assert context["intent"] != scenario.metadata["intent"]
    assert "spoof" not in context["intent"].lower()
    assert "Alarm_RW_Tank_LL" in context["active_alarms"]
    assert "raw_water_low_low_alarm_active" in context["relationship_notes"]
    assert "raw_water_protection_risk" in context["risk_flags"]
    assert "RW_Tank_Level" in context["short_horizon_trends"]


def test_prompt_includes_operational_manual(tmp_path) -> None:
    manual_path = tmp_path / "manual.md"
    manual_path.write_text("Shared manual line.")
    engine = CachedQwenDecisionEngine(
        {
            "role": "observer",
            "knowledge_files": [str(manual_path)],
            "llm": {
                "backend": "qwen_langgraph",
                "strict_backend": True,
            },
        },
        "observer",
    )

    prompt = engine._format_prompt(
        {
            "scenario_id": "attack-wt-p1-live",
            "suite": "attack",
            "intent": "Monitor the plant safely.",
            "time_sec": 0.0,
            "process_id": "WT-P1",
            "candidate_actions": [],
            "remaining_playbook": [],
            "recent_observations": [],
            "recent_actions": [],
            "recent_decisions": [],
            "recent_alerts": [],
            "role_goal": [],
            "role_constraints": [],
            "expected_outcomes": [],
            "safety_bounds": {},
            "observation": {"STATE_NAME": "IDLE"},
        }
    )

    assert "Shared manual line." in prompt


def test_finalize_payload_drops_sentence_like_diagnosis() -> None:
    engine = CachedQwenDecisionEngine(
        {
            "role": "defender",
            "llm": {
                "backend": "qwen_langgraph",
                "strict_backend": True,
            },
        },
        "defender",
    )

    payload = engine._finalize_payload(
        {
            "summary": "Monitor.",
            "candidate_id": None,
            "diagnosis": "Plant is in RUNNING state with no active alarms.",
            "decision_type": "note",
        },
        {"candidate_actions": []},
    )

    assert payload["diagnosis"] is None
