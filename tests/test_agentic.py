from __future__ import annotations

import pytest

from ics_agent.agents.agentic import AgenticRoleAgent
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


def test_agentic_non_strict_backend_resolves_fallback_candidate_to_action(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-p1-short-nominal")
    profile = {
        "role": "operator",
        "family": "agentic",
        "allowed_action_types": ["hmi_start", "hmi_stop", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": False,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_operator", profile)
    agent.reset(scenario)
    monkeypatch.setattr(agent.engine, "_decide_with_langgraph", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'langgraph'")))
    monkeypatch.setattr(agent.engine, "_decide_with_langchain", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'langchain'")))

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["backend"] == "heuristic_fallback"
    assert "langgraph" in decisions[0].metadata["error"]


def test_agentic_strict_backend_raises_when_llm_stack_is_unavailable(monkeypatch) -> None:
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
    monkeypatch.setattr(agent.engine, "_decide_with_langgraph", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'langgraph'")))
    monkeypatch.setattr(agent.engine, "_decide_with_langchain", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'langchain'")))

    with pytest.raises(RuntimeError, match="backend `qwen_langgraph` unavailable"):
        agent.step(_context(scenario))


def test_agentic_uses_fallback_action_when_model_omits_candidate_id(monkeypatch) -> None:
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
        lambda context, fallback: agent.engine._finalize_payload(
            {
                "summary": "Start the plant now.",
                "decision_type": "action",
                "diagnosis": "nominal_operation",
            },
            context,
            fallback,
        ),
    )

    decisions = agent.step(_context(scenario))

    assert len(decisions) == 1
    assert decisions[0].action == {"type": "hmi_start", "process_id": "WT-P1"}
    assert decisions[0].metadata["backend"] == "qwen_langgraph"
