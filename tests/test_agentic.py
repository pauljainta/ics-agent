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


def test_due_redundant_playbook_action_is_skipped_when_state_already_satisfied(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Force the stop anyway.",
            "action": {"type": "hmi_stop", "process_id": "WT-P1"},
            "decision_type": "action",
        },
    )

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


def test_attacker_context_exposes_structured_memory_and_recovery_candidates() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "allowed_action_types": [
            "select_process",
            "set_manual_valve",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 12.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 0,
                "ChemTreat_HCl_Level": 790,
                "RW_Tank_Level": 610,
                "RW_Pump_Sts": 1,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "P2_Selected": 1, "ChemTreat_HCl_Level": 800},
                {"STATE_NAME": "RUNNING", "P2_Selected": 0, "ChemTreat_HCl_Level": 790},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 11.5,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": False,
                    },
                },
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 11.5,
                    "decision_type": "alert",
                    "summary": "Observer saw Process 2 toggled off after suspicious use.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {
                "attacker": [
                    {
                        "time_sec": 8.0,
                        "action": {
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": True,
                        },
                    },
                    {
                        "time_sec": 10.0,
                        "action": {
                            "type": "set_manual_valve",
                            "process_id": "WT-P2",
                            "chemical": "HCl",
                            "open": True,
                        },
                    },
                ]
            },
        }
    )

    assert context["structured_memory"]["leverage_hypothesis"] == "process2_manual_path"
    assert context["structured_memory"]["defender_attention_estimate"] == "high"
    assert any(
        item["action"]
        == {
            "type": "select_process",
            "process_id": "WT-P2",
            "active": True,
        }
        for item in context["candidate_actions"]
    )
    assert any(
        item["action"]
        == {
            "type": "set_manual_valve",
            "process_id": "WT-P2",
            "chemical": "HCl",
            "open": True,
        }
        for item in context["candidate_actions"]
    )


def test_observer_memory_tracks_evidence_and_attention_target() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-naocl-drain")
    profile = {
        "role": "observer",
        "family": "agentic",
        "allowed_action_types": ["noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_observer", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 12.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "ChemTreat_NaOCl_Level": 760,
                "ChemTreat_NaOCl_Valve_Sts": 1,
                "RW_Tank_Level": 610,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_NaOCl_Level": 800},
                {"STATE_NAME": "RUNNING", "ChemTreat_NaOCl_Level": 780},
                {"STATE_NAME": "RUNNING", "ChemTreat_NaOCl_Level": 760},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    memory = context["structured_memory"]
    assert memory["anomaly_hypotheses"][0]["diagnosis"] == "unauthorized_chemical_dosing"
    assert memory["recommended_attention_target"] == "WT-P2_manual_path"
    assert memory["evidence_ledger"]


def test_defender_memory_tracks_prior_mitigations_and_escalation() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
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
                "P2_Selected": 1,
                "ChemTreat_HCl_Level": 730,
                "ChemTreat_HCl_Valve_Sts": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_Level": 620,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 780, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 755, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 730, "P2_Selected": 1},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 12.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": False,
                    },
                },
            ],
            "decisions": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Repeated HCl drain evidence remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Repeated HCl drain evidence remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    memory = context["structured_memory"]
    assert memory["prior_mitigations_attempted"] == ["select_process:off"]
    assert memory["containment_effective"] is False
    assert memory["escalation_required"] is True


def test_prior_knowledge_defender_exposes_doctrine_and_escalates_failed_process2_containment() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "defender",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_defender", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 16.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "ChemTreat_HCl_Level": 730,
                "ChemTreat_HCl_Valve_Sts": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_Level": 620,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 780, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 755, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 730, "P2_Selected": 1},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 12.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": False,
                    },
                },
            ],
            "decisions": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Repeated HCl drain evidence remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Repeated HCl drain evidence remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    assert context["structured_memory"]["escalation_required"] is True
    assert context["structured_prior_knowledge"]["matched_patterns"][0]["id"] == "process2_dosing_abuse"
    assert (
        context["structured_prior_knowledge"]["attack_surface_assessment"]["attack_surface"]
        == "process2_manual_path"
    )
    assert (
        context["structured_prior_knowledge"]["attack_surface_assessment"][
            "containment_scope"
        ]
        == "broader_containment"
    )
    assert context["structured_prior_knowledge"]["current_doctrine"]["primary_action"] == "hmi_stop"
    assert context["incident"]["recommended_action"] == "hmi_stop"
    assert context["candidate_actions"][0]["action"] == {
        "type": "hmi_stop",
        "process_id": "WT-P1",
    }


def test_prior_knowledge_defender_prefers_targeted_process2_isolation_after_detection() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "defender",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_defender_targeted", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 11.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "ChemTreat_HCl_Level": 760,
                "ChemTreat_HCl_Valve_Sts": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_Level": 620,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 790, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 775, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 760, "P2_Selected": 1},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                }
            ],
            "decisions": [
                {
                    "role": "observer",
                    "time_sec": 10.5,
                    "decision_type": "alert",
                    "summary": "Observer confirmed unauthorized Process 2 manual activity.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 10.5,
                    "decision_type": "alert",
                    "summary": "Observer confirmed unauthorized Process 2 manual activity.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    assert context["incident"]["status"] == "confirmed"
    assert context["structured_prior_knowledge"]["post_detection_guidance_active"] is True
    assert (
        context["structured_prior_knowledge"]["attack_surface_assessment"]["attack_surface"]
        == "process2_manual_path"
    )
    assert (
        context["structured_prior_knowledge"]["attack_surface_assessment"][
            "containment_scope"
        ]
        == "targeted_isolation"
    )
    assert context["incident"]["recommended_action"] == "select_process"
    assert context["candidate_actions"][0]["action"] == {
        "type": "select_process",
        "process_id": "WT-P2",
        "active": False,
    }


def test_prior_knowledge_defender_keeps_live_doctrine_dormant_before_detection() -> None:
    scenario = ScenarioCatalog().get("smoke/wt-smoke-p1-short-nominal")
    profile = {
        "role": "defender",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_defender_idle", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 4.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 0,
                "RW_Pump_Sts": 1,
                "RW_Tank_Level": 610,
                "UF_UFFT_Tank_Level": 420,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Pump_Sts": 1,
                    "RW_Tank_Level": 612,
                    "UF_UFFT_Tank_Level": 419,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Pump_Sts": 1,
                    "RW_Tank_Level": 610,
                    "UF_UFFT_Tank_Level": 420,
                },
            ],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["incident"]["type"] == "none"
    assert context["structured_prior_knowledge"]["post_detection_guidance_active"] is False
    assert context["structured_prior_knowledge"]["matched_patterns"] == []


def test_prior_knowledge_defender_suppresses_actions_before_confirmation(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-select-chatter")
    profile = {
        "role": "defender",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_defender_gated", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Deselect Process 2 immediately.",
            "action": {
                "type": "select_process",
                "process_id": "WT-P2",
                "active": False,
            },
            "decision_type": "action",
            "diagnosis": "unauthorized_chemical_dosing",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 4.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "RW_Tank_Level": 620,
                "RW_Pump_Sts": 1,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "P2_Selected": 1, "RW_Tank_Level": 620},
                {"STATE_NAME": "RUNNING", "P2_Selected": 1, "RW_Tank_Level": 620},
            ],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "note"
    assert decisions[0].action is None


def test_prior_knowledge_guardrail_escalates_failed_process2_containment_when_model_holds(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "defender",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_defender", profile)
    agent.reset(scenario)
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Process 2 still looks suspicious; hold one more sample.",
            "candidate_id": action_key({"type": "noop"}),
            "decision_type": "note",
            "diagnosis": "unauthorized_chemical_dosing",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 16.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "ChemTreat_HCl_Level": 730,
                "ChemTreat_HCl_Valve_Sts": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_Level": 620,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 780, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 755, "P2_Selected": 1},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 730, "P2_Selected": 1},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 12.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": False,
                    },
                },
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Repeated HCl drain evidence remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    assert any(
        decision.action == {"type": "hmi_stop", "process_id": "WT-P1"}
        for decision in decisions
    )
    assert any("safety-guardrail" in decision.labels for decision in decisions)


def test_prior_knowledge_observer_marks_high_high_pattern_as_immediate() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-hh-alarm")
    profile = {
        "role": "observer",
        "family": "agentic",
        "knowledge_mode": "prior_knowledge",
        "allowed_action_types": ["noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_prior_knowledge_observer", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 15.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 1245,
                "Alarm_RW_Tank_HH": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_PR_Valve_Cmd": 1,
                "RW_Tank_PR_Valve_Sts": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 615},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1245},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 15.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 1250,
                    },
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["structured_prior_knowledge"]["matched_patterns"][0]["id"] == "raw_water_high_high_or_overflow"
    assert context["structured_prior_knowledge"]["matched_patterns"][0]["urgency"] == "immediate"
    assert context["structured_memory"]["suspicion_level"] == "high"


def test_black_box_attacker_context_strips_privileged_scenario_fields() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_manual_valve",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 7.0,
            "observation": {
                "STATE_NAME": "IDLE",
                "P2_Selected": 0,
                "RW_Tank_Level": 620,
                "UF_UFFT_Tank_Level": 420,
                "ChemTreat_HCl_Level": 810,
            },
            "observations": [],
            "executed_actions": [],
            "decisions": [
                {
                    "role": "observer",
                    "time_sec": 6.5,
                    "decision_type": "alert",
                    "summary": "Observer saw suspicious Process 2 behavior.",
                    "diagnosis": "unauthorized_chemical_dosing",
                },
                {
                    "role": "attacker",
                    "time_sec": 6.7,
                    "decision_type": "action",
                    "summary": "Probe Process 2.",
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
            ],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 6.5,
                    "decision_type": "alert",
                    "summary": "Observer saw suspicious Process 2 behavior.",
                    "diagnosis": "unauthorized_chemical_dosing",
                },
                {
                    "role": "defender",
                    "time_sec": 6.8,
                    "decision_type": "alert",
                    "summary": "Defender is preparing containment.",
                    "diagnosis": "unauthorized_chemical_dosing",
                },
            ],
            "playbooks": {
                "attacker": [
                    {
                        "time_sec": 8.0,
                        "action": {
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": True,
                        },
                    }
                ]
            },
        }
    )

    assert context["scenario_id"] == "attack-live"
    assert context["process_id"] is None
    assert context["remaining_playbook"] == []
    assert context["expected_outcomes"] == []
    assert context["safety_bounds"] == {}
    assert context["intent"].startswith("Discover controllable surfaces")
    assert context["role_goal"][0].startswith("Infer what each candidate action")
    assert context["structured_memory"]["defender_intervention_awareness"] == {
        "recent_blue_alert_count": 2,
        "recent_defender_action_count": 0,
        "pressure": "high",
    }
    assert context["recent_decisions"] == [
        {
            "role": "attacker",
            "decision_type": "action",
            "action": {
                "type": "select_process",
                "process_id": "WT-P2",
                "active": True,
            },
            "time_sec": 6.7,
            "age_sec": 0.3,
        }
    ]
    assert context["recent_alerts"] == [
        {
            "role": "observer",
            "signal": "attention_notice",
            "time_sec": 6.5,
            "age_sec": 0.5,
        },
        {
            "role": "defender",
            "signal": "attention_notice",
            "time_sec": 6.8,
            "age_sec": 0.2,
        },
    ]
    assert "summary" not in context["recent_alerts"][0]
    assert "diagnosis" not in context["recent_alerts"][0]
    assert "summary" not in context["recent_decisions"][0]
    assert "diagnosis" not in context["recent_decisions"][0]


def test_black_box_attacker_context_hides_defender_action_details() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_manual_valve",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 8.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "RW_Tank_Level": 430,
                "Alarm_RW_Tank_LL": 1,
                "ChemTreat_HCl_Level": 760,
                "ChemTreat_HCl_Valve_Sts": 1,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 1,
                    "RW_Tank_Level": 520,
                    "ChemTreat_HCl_Level": 800,
                    "ChemTreat_HCl_Valve_Sts": 0,
                }
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 6.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 7.2,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": False,
                    },
                },
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 7.0,
                    "decision_type": "alert",
                    "summary": "Process 2 behavior remains suspicious.",
                    "diagnosis": "unauthorized_chemical_dosing",
                },
                {
                    "role": "defender",
                    "time_sec": 7.4,
                    "decision_type": "alert",
                    "summary": "Containment remains active.",
                    "diagnosis": "unauthorized_chemical_dosing",
                },
            ],
            "playbooks": {},
        }
    )

    assert context["active_alarms"] == ["Alarm_RW_Tank_LL"]
    assert context["relationship_notes"] == []
    assert context["risk_flags"] == []
    assert [item["role"] for item in context["recent_actions"]] == ["attacker"]
    assert [item["role"] for item in context["recent_action_effects"]] == ["attacker"]
    assert context["structured_memory"]["defender_intervention_awareness"] == {
        "recent_blue_alert_count": 2,
        "recent_defender_action_count": 1,
        "pressure": "high",
    }


def test_optionb_mitre_black_box_attacker_context_exposes_phase_model_without_leakage() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_manual_valve",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "hmi_estop",
            "noop",
        ],
        "behavior_model": {
            "id": "optionb_mitre_staged",
            "policy": "agentic",
        },
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker_optionb_mitre", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 0.0,
            "observation": {
                "STATE_NAME": "IDLE",
                "RW_Tank_Level": 600,
                "UF_UFFT_Tank_Level": 400,
                "P2_Selected": 0,
            },
            "observations": [],
            "executed_actions": [],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["scenario_id"] == "attack-live"
    assert context["process_id"] is None
    assert context["remaining_playbook"] == []
    assert context["mitre_context"]["phase_model"] == "optionb_mitre_staged"
    assert context["mitre_context"]["phase_family"] == "attacker"
    assert context["mitre_context"]["next_phase"] == "staged_entry_context"
    assert context["mitre_context"]["direct_plant_shortcuts_forbidden"] is True
    assert context["staged_path"] == [
        "local_orchestrator",
        "access",
        "engineering",
        "plant_runtime",
    ]
    assert "target_tags" not in context["mitre_context"]


def test_optionb_mitre_defender_context_exposes_phase_model_without_hidden_plan_leakage() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "defender",
        "family": "agentic",
        "allowed_action_types": ["hmi_stop", "hmi_estop", "select_process", "noop"],
        "behavior_model": {
            "id": "optionb_mitre_staged",
            "policy": "agentic",
        },
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_defender_optionb_mitre", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 6.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 620,
                "UF_UFFT_Tank_Level": 430,
                "P2_Selected": 0,
            },
            "observations": [{"STATE_NAME": "RUNNING", "RW_Tank_Level": 620}],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 5.0,
                    "action": {"type": "inspect_surface", "surface": "process2_control_plane"},
                }
            ],
            "decisions": [
                {
                    "role": "attacker",
                    "time_sec": 4.5,
                    "decision_type": "note",
                    "summary": "Staged entry context established.",
                    "metadata": {
                        "phase_model": "optionb_mitre_staged",
                        "phase_family": "attacker",
                        "mitre_phase": "staged_entry_context",
                        "mitre_phase_order": 1,
                    },
                }
            ],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["scenario_id"] != scenario.scenario_id
    assert "hijack" not in context["intent"].lower()
    assert context["mitre_context"]["phase_model"] == "optionb_mitre_staged"
    assert context["mitre_context"]["phase_family"] == "defender"
    assert context["mitre_context"]["next_phase"] in {
        "discovery_recognition",
        "manipulation_detection",
    }
    assert context["mitre_context"]["direct_plant_shortcuts_forbidden"] is True
    assert context["mitre_context"]["attacker_highest_phase"] in {
        "staged_entry_context",
        "discovery",
    }
    assert "target_tags" not in context["mitre_context"]


def test_black_box_attacker_offers_exploratory_candidates() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
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
            "playbooks": {},
        }
    )

    assert any(
        item["action"] == {"type": "hmi_start", "process_id": "WT-P1"}
        for item in context["candidate_actions"]
    )
    assert any(
        item["action"]
        == {"process_id": "WT-P2", "type": "select_process", "active": True}
        for item in context["candidate_actions"]
    )
    assert any(
        item["action"].get("type") == "set_sensor"
        and item["action"].get("name") == "RW_Tank_Level"
        for item in context["candidate_actions"]
    )
    assert any(
        item["action"] == {"type": "inspect_surface", "surface": "process2_control_plane"}
        for item in context["candidate_actions"]
    )


def test_black_box_attacker_memory_tracks_discovered_surface() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_manual_valve",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 10.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "RW_Tank_Level": 610,
                "ChemTreat_HCl_Level": 790,
            },
            "observations": [],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 7.5,
                    "action": {
                        "type": "inspect_surface",
                        "surface": "process2_control_plane",
                    },
                    "result": {
                        "category": "reconnaissance",
                        "surface": "process2_control_plane",
                        "summary": "process2_control_plane: state=RUNNING; process2_selected; P2_HCl_Cmd=1",
                        "findings": [
                            "state=RUNNING",
                            "process2_selected",
                            "P2_HCl_Cmd=1",
                        ],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                }
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 9.5,
                    "decision_type": "alert",
                    "summary": "Process 2 was toggled unexpectedly.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    memory = context["structured_memory"]
    assert memory["best_known_attack_surface"] == "process2_manual_path"
    assert memory["detection_pressure"] in {"medium", "high"}
    assert memory["current_phase"] in {
        "exploit_commitment",
        "exploit_escalation",
        "evasion_or_exploitation",
    }
    assert memory["exploit_commitment"]["surface"] == "process2_manual_path"
    assert memory["simple_precondition_state"]["process2_enabled"] is True
    assert memory["reconnaissance_findings"][0]["surface"] == "process2_control_plane"
    assert memory["candidate_leverage_hypotheses"][0]["surface"] == "process2_manual_path"


def test_black_box_attacker_recon_budget_forces_process2_commitment() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_manual_valve",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 12.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "RW_Tank_Level": 610,
                "ChemTreat_HCl_Level": 790,
                "ChemTreat_HCl_Valve_Sts": 1,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 1,
                    "ChemTreat_HCl_Level": 820,
                    "ChemTreat_HCl_Valve_Sts": 0,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 1,
                    "ChemTreat_HCl_Level": 805,
                    "ChemTreat_HCl_Valve_Sts": 1,
                },
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 6.0,
                    "action": {"type": "inspect_surface", "surface": "tags"},
                    "result": {
                        "category": "reconnaissance",
                        "surface": "tags",
                        "summary": "tags: state=RUNNING; RW_Tank_Level=610",
                        "findings": ["state=RUNNING", "RW_Tank_Level=610"],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 7.0,
                    "action": {
                        "type": "inspect_surface",
                        "surface": "process2_control_plane",
                    },
                    "result": {
                        "category": "reconnaissance",
                        "surface": "process2_control_plane",
                        "summary": "process2_control_plane: state=RUNNING; process2_selected; P2_HCl_Cmd=1",
                        "findings": [
                            "state=RUNNING",
                            "process2_selected",
                            "P2_HCl_Cmd=1",
                        ],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 7.5,
                    "action": {"type": "inspect_surface", "surface": "alarm_bits"},
                    "result": {
                        "category": "reconnaissance",
                        "surface": "alarm_bits",
                        "summary": "alarm_bits: state=RUNNING",
                        "findings": ["state=RUNNING"],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 8.4,
                    "action": {
                        "type": "set_manual_valve",
                        "process_id": "WT-P2",
                        "chemical": "HCl",
                        "open": True,
                    },
                },
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    memory = context["structured_memory"]
    assert memory["recon_budget_remaining"] == 0
    assert memory["exploit_commitment"]["surface"] == "process2_manual_path"
    assert memory["current_phase"] == "exploit_escalation"
    assert context["candidate_actions"][0]["action"]["type"] == "set_manual_valve"
    assert context["candidate_actions"][0]["action"]["chemical"] in {"NaCl", "NaOCl"}
    assert not any(
        item["action"]["type"] == "inspect_surface"
        for item in context["candidate_actions"][:3]
    )


def test_black_box_attacker_sensor_commitment_escalates_after_effect() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "select_process",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 8.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 0,
                "RW_Tank_Level": 540,
                "UF_UFFT_Tank_Level": 410,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Tank_Level": 600,
                    "UF_UFFT_Tank_Level": 410,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Tank_Level": 560,
                    "UF_UFFT_Tank_Level": 410,
                },
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 6.0,
                    "action": {"type": "inspect_surface", "surface": "tags"},
                    "result": {
                        "category": "reconnaissance",
                        "surface": "tags",
                        "summary": "tags: state=RUNNING; RW_Tank_Level=560",
                        "findings": ["state=RUNNING", "RW_Tank_Level=560"],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 7.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 540,
                    },
                },
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    memory = context["structured_memory"]
    assert memory["best_known_attack_surface"] == "sensor_manipulation"
    assert memory["exploit_commitment"]["surface"] == "sensor_manipulation"
    assert memory["last_successful_effect"].startswith("RW_Tank_Level")
    assert any(
        item["action"]["type"] == "set_sensor"
        and item["action"]["name"] == "RW_Tank_Level"
        and item["action"]["value"] < 540
        for item in context["candidate_actions"][:3]
    )


def test_black_box_attacker_uses_prior_live_sample_for_effect_learning() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "set_sensor",
            "hmi_start",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    current_observation = {
        "STATE_NAME": "RUNNING",
        "P2_Selected": 0,
        "RW_Tank_Level": 520,
        "UF_UFFT_Tank_Level": 410,
        "elapsed_sec": 5.0,
        "sample_phase": "post_action",
    }
    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 5.0,
            "observation": current_observation,
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Tank_Level": 600,
                    "UF_UFFT_Tank_Level": 410,
                    "elapsed_sec": 4.5,
                    "sample_phase": "pre_action",
                },
                current_observation,
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 5.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 470,
                    },
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["structured_memory"]["last_successful_effect"].startswith(
        "RW_Tank_Level: telemetry_shift_observed"
    )


def test_black_box_attacker_suppresses_availability_chatter_after_stall() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-refill-spoof")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "knowledge_mode": "black_box",
        "allowed_action_types": [
            "inspect_surface",
            "set_sensor",
            "hmi_start",
            "hmi_stop",
            "hmi_estop",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_black_box_attacker", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 8.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 0,
                "RW_Tank_Level": 520,
                "UF_UFFT_Tank_Level": 410,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Tank_Level": 600,
                    "UF_UFFT_Tank_Level": 410,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "P2_Selected": 0,
                    "RW_Tank_Level": 560,
                    "UF_UFFT_Tank_Level": 410,
                },
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 2.0,
                    "action": {"type": "inspect_surface", "surface": "tags"},
                    "result": {
                        "category": "reconnaissance",
                        "surface": "tags",
                        "summary": "tags: state=RUNNING; RW_Tank_Level=560",
                        "findings": ["state=RUNNING", "RW_Tank_Level=560"],
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 3.0,
                    "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                },
                {
                    "role": "attacker",
                    "time_sec": 4.0,
                    "action": {"type": "hmi_start", "process_id": "WT-P1"},
                },
                {
                    "role": "attacker",
                    "time_sec": 5.0,
                    "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                },
                {
                    "role": "attacker",
                    "time_sec": 6.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 540,
                    },
                },
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert context["structured_memory"]["best_known_attack_surface"] == "sensor_manipulation"
    assert context["candidate_actions"][0]["action"]["type"] == "set_sensor"
    assert not any(
        item["action"]["type"] in {"hmi_stop", "hmi_estop"}
        for item in context["candidate_actions"][:3]
    )


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


def test_agentic_blue_team_ignores_planned_nominal_process2_operation() -> None:
    scenario = ScenarioCatalog().get("normal/wt-p2-hcl-dosing")
    profile = {
        "role": "observer",
        "family": "agentic",
        "allowed_action_types": ["noop"],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_observer", profile)
    agent.reset(scenario)

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 9.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "P2_Selected": 1,
                "ChemTreat_HCl_Valve_Sts": 1,
                "ChemTreat_HCl_Level": 790,
                "RW_Tank_Level": 620,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 800},
                {"STATE_NAME": "RUNNING", "ChemTreat_HCl_Level": 790},
            ],
            "executed_actions": [
                {
                    "role": "operator",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {
                "operator": [
                    {
                        "time_sec": 5.0,
                        "action": {"type": "hmi_start", "process_id": "WT-P1"},
                    },
                    {
                        "time_sec": 8.0,
                        "action": {
                            "type": "select_process",
                            "process_id": "WT-P2",
                            "active": True,
                        },
                    },
                    {
                        "time_sec": 9.0,
                        "action": {
                            "type": "set_manual_valve",
                            "process_id": "WT-P2",
                            "chemical": "HCl",
                            "open": True,
                        },
                    },
                ]
            },
        }
    )

    assert "planned_process2_operation" in context["relationship_notes"]
    assert "process2_manual_path_active" not in context["risk_flags"]
    assert context["structured_memory"]["anomaly_hypotheses"] == []


def test_agentic_blue_team_ignores_planned_nominal_refill_response() -> None:
    scenario = ScenarioCatalog().get("normal/wt-p1-refill-response")
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
            "time_sec": 23.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 451,
                "UF_UFFT_Tank_Level": 430,
                "Alarm_RW_Tank_L": 1,
                "Alarm_RW_Tank_LL": 0,
                "RW_Pump_Sts": 1,
                "RW_Tank_PR_Valve_Cmd": 1,
                "RW_Tank_PR_Valve_Sts": 1,
                "P2_Selected": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 475},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 471},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 460},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 451},
            ],
            "executed_actions": [
                {
                    "role": "operator",
                    "time_sec": 5.0,
                    "action": {"type": "hmi_start", "process_id": "WT-P1"},
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {
                "operator": [
                    {
                        "time_sec": 5.0,
                        "action": {"type": "hmi_start", "process_id": "WT-P1"},
                    },
                    {
                        "time_sec": 55.0,
                        "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                    },
                ]
            },
        }
    )

    assert "planned_refill_response" in context["relationship_notes"]
    assert context["incident"]["type"] == "telemetry_anomaly"
    assert not any(
        item.get("action", {}).get("type") == "hmi_stop"
        for item in context["candidate_actions"]
        if item.get("action")
    )

    decisions = agent._payload_to_decisions(
        {
            "summary": "Escalate to controlled stop.",
            "decision_type": "action",
            "action": {"type": "hmi_stop", "process_id": "WT-P1"},
            "diagnosis": "raw_water_low_or_spoof",
        },
        context,
    )

    assert len(decisions) == 1
    assert decisions[0].decision_type == "note"
    assert decisions[0].action is None


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
    assert context["incident"]["type"] == "raw_water_low_protection"
    assert context["incident"]["status"] in {"suspected", "confirmed"}


def test_live_context_compacts_history_and_hides_transient_memory() -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-hmi-forced-stop")
    profile = {
        "role": "attacker",
        "family": "agentic",
        "allowed_action_types": [
            "hmi_start",
            "hmi_stop",
            "hmi_estop",
            "select_process",
            "set_manual_valve",
            "set_sensor",
            "noop",
        ],
        "llm": {
            "backend": "qwen_langgraph",
            "strict_backend": True,
        },
    }
    agent = AgenticRoleAgent("pytest_agentic_attacker", profile)
    agent.reset(scenario)
    agent._state["role_memory"] = {
        "mission_objective": "Disrupt WT-P1 availability.",
        "current_tactic_phase": "availability_pressure",
        "last_rationale": "previous hold note",
        "last_decision_time_sec": 4.0,
    }

    context = agent._build_online_context(
        {
            "scenario": scenario,
            "time_sec": 12.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 640,
                "UF_UFFT_Tank_Level": 430,
                "P2_Selected": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 650},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 640},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 10.0,
                    "action": {"type": "hmi_stop"},
                    "state_after": "SHUTDOWN",
                    "agent_name": "attacker",
                }
            ],
            "decisions": [
                {
                    "role": "observer",
                    "time_sec": 11.0,
                    "decision_type": "alert",
                    "summary": "Observer flagged abrupt shutdown.",
                    "diagnosis": "availability_pressure",
                    "labels": ["observer"],
                    "metadata": {"backend": "qwen_langgraph"},
                }
            ],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 11.0,
                    "decision_type": "alert",
                    "summary": "Observer flagged abrupt shutdown.",
                    "diagnosis": "availability_pressure",
                    "metadata": {"backend": "qwen_langgraph"},
                }
            ],
            "playbooks": {},
        }
    )

    assert "last_rationale" not in context["structured_memory"]
    assert "last_decision_time_sec" not in context["structured_memory"]
    assert context["recent_actions"][0]["age_sec"] == 2.0
    assert context["recent_decisions"][0]["age_sec"] == 1.0
    assert "labels" not in context["recent_decisions"][0]
    assert "metadata" not in context["recent_decisions"][0]


def test_reasoning_signature_ignores_transient_incident_counters() -> None:
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
    mutated = dict(context)
    mutated["incident"] = {
        **(context["incident"] or {}),
        "persistence_sec": 99.0,
        "evidence_count": 9,
        "last_seen_time_sec": 99.0,
        "new_evidence": ["extra"],
        "evidence_tokens": ["extra"],
    }

    assert agent._reasoning_signature(
        context,
        channel="monitor",
    ) == agent._reasoning_signature(mutated, channel="monitor")


def test_defender_guardrail_contains_process2_drain_when_model_only_recognizes(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-p2-hcl-drain")
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
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Process 2 dosing looks suspicious; monitor and hold.",
            "candidate_id": action_key({"type": "noop"}),
            "decision_type": "note",
            "diagnosis": "unauthorized_chemical_dosing",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 10.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 610,
                "UF_UFFT_Tank_Level": 430,
                "ChemTreat_NaCl_Level": 800,
                "ChemTreat_NaOCl_Level": 800,
                "ChemTreat_HCl_Level": 760,
                "ChemTreat_HCl_Valve_Sts": 1,
                "P2_Selected": 1,
                "RW_Pump_Sts": 1,
            },
            "observations": [
                {
                    "STATE_NAME": "RUNNING",
                    "ChemTreat_HCl_Level": 800,
                    "P2_Selected": 1,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "ChemTreat_HCl_Level": 790,
                    "P2_Selected": 1,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "ChemTreat_HCl_Level": 775,
                    "P2_Selected": 1,
                },
                {
                    "STATE_NAME": "RUNNING",
                    "ChemTreat_HCl_Level": 760,
                    "P2_Selected": 1,
                },
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 8.0,
                    "action": {
                        "type": "select_process",
                        "process_id": "WT-P2",
                        "active": True,
                    },
                },
                {
                    "role": "attacker",
                    "time_sec": 10.0,
                    "action": {
                        "type": "set_manual_valve",
                        "process_id": "WT-P2",
                        "chemical": "HCl",
                        "open": True,
                    },
                },
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 10.0,
                    "decision_type": "alert",
                    "summary": "Observer saw Process 2 HCl drain activity.",
                    "diagnosis": "unauthorized_chemical_dosing",
                }
            ],
            "playbooks": {},
        }
    )

    assert any(
        decision.action == {
            "type": "select_process",
            "process_id": "WT-P2",
            "active": False,
        }
        for decision in decisions
    )
    assert any("safety-guardrail" in decision.labels for decision in decisions)


def test_defender_guardrail_escalates_raw_water_high_high_when_model_holds(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-hh-alarm")
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
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "High-high condition observed; monitor one more sample.",
            "candidate_id": action_key({"type": "noop"}),
            "decision_type": "note",
            "diagnosis": "raw_water_alarm_condition",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 15.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 1245,
                "UF_UFFT_Tank_Level": 430,
                "Alarm_RW_Tank_HH": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_PR_Valve_Cmd": 1,
                "RW_Tank_PR_Valve_Sts": 0,
                "P2_Selected": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 615},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1245},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 15.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 1250,
                    },
                }
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Observer saw abrupt RW high-high alarm.",
                    "diagnosis": "raw_water_alarm_condition",
                }
            ],
            "playbooks": {},
        }
    )

    assert any(
        decision.action == {"type": "hmi_estop", "process_id": "WT-P1"}
        for decision in decisions
    )
    assert any("safety-guardrail" in decision.labels for decision in decisions)


def test_defender_guardrail_does_not_repeat_same_action_after_containment(
    monkeypatch,
) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-hh-alarm")
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
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Stay cautious and monitor.",
            "candidate_id": action_key({"type": "noop"}),
            "decision_type": "note",
            "diagnosis": "raw_water_alarm_condition",
        },
    )

    first_decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 15.0,
            "observation": {
                "STATE_NAME": "RUNNING",
                "RW_Tank_Level": 1245,
                "Alarm_RW_Tank_HH": 1,
                "RW_Pump_Sts": 1,
                "RW_Tank_PR_Valve_Cmd": 1,
                "RW_Tank_PR_Valve_Sts": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 615},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1245},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 15.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 1250,
                    },
                }
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Observer saw abrupt RW high-high alarm.",
                    "diagnosis": "raw_water_alarm_condition",
                }
            ],
            "playbooks": {},
        }
    )

    assert any(decision.action is not None for decision in first_decisions)

    second_decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 18.0,
            "observation": {
                "STATE_NAME": "SHUTDOWN",
                "RW_Tank_Level": 1238,
                "Alarm_RW_Tank_HH": 1,
                "RW_Pump_Sts": 0,
                "RW_Tank_PR_Valve_Cmd": 0,
                "RW_Tank_PR_Valve_Sts": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 615},
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 1245},
                {"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 1238},
            ],
            "executed_actions": [
                {
                    "role": "attacker",
                    "time_sec": 15.0,
                    "action": {
                        "type": "set_sensor",
                        "name": "RW_Tank_Level",
                        "value": 1250,
                    },
                },
                {
                    "role": "defender",
                    "time_sec": 15.0,
                    "action": {"type": "hmi_estop", "process_id": "WT-P1"},
                },
            ],
            "decisions": [],
            "alerts": [
                {
                    "role": "observer",
                    "time_sec": 15.0,
                    "decision_type": "alert",
                    "summary": "Observer saw abrupt RW high-high alarm.",
                    "diagnosis": "raw_water_alarm_condition",
                }
            ],
            "playbooks": {},
        }
    )

    assert not any(decision.action is not None for decision in second_decisions)


def test_defender_model_action_is_suppressed_when_already_satisfied(monkeypatch) -> None:
    scenario = ScenarioCatalog().get("attack/wt-attack-sensor-rw-hh-alarm")
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
    monkeypatch.setattr(
        agent.engine,
        "decide",
        lambda context: {
            "summary": "Keep plant stopped.",
            "action": {"type": "hmi_stop", "process_id": "WT-P1"},
            "decision_type": "action",
            "diagnosis": "raw_water_alarm_condition",
        },
    )

    decisions = agent.step(
        {
            "scenario": scenario,
            "time_sec": 18.0,
            "observation": {
                "STATE_NAME": "SHUTDOWN",
                "RW_Tank_Level": 1238,
                "Alarm_RW_Tank_HH": 1,
                "RW_Pump_Sts": 0,
            },
            "observations": [
                {"STATE_NAME": "RUNNING", "RW_Tank_Level": 615},
                {"STATE_NAME": "SHUTDOWN", "RW_Tank_Level": 1238},
            ],
            "executed_actions": [
                {
                    "role": "defender",
                    "time_sec": 15.0,
                    "action": {"type": "hmi_stop", "process_id": "WT-P1"},
                }
            ],
            "decisions": [],
            "alerts": [],
            "playbooks": {},
        }
    )

    assert decisions == []


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
