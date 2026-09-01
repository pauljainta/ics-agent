from __future__ import annotations

from pathlib import Path

import yaml

from ics_agent.agents.agentic import AgenticRoleAgent, CachedQwenDecisionEngine
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.types import ScenarioDefinition


SMOKE_CONFIG = "water_treatment_white_box_pure_react_attacker_vs_silent_defender_smoke.yaml"
AUTONOMOUS_SMOKE_CONFIG = (
    "water_treatment_white_box_autonomous_react_attacker_vs_silent_defender_smoke.yaml"
)
SMOKE_SCENARIOS = [
    "wt-attack-p2-multi-chemical-drain",
    "wt-attack-p2-dual-hijack",
    "wt-attack-sensor-rw-cascade-low-hh",
    "wt-attack-sensor-uf-high-disrupt",
]


def _minimal_decision_context() -> dict:
    return {
        "scenario_id": "white-box-attack-live",
        "suite": "attack",
        "intent": "Drive a bounded water-treatment scenario toward unsafe impact.",
        "process_id": None,
        "time_sec": 1.0,
        "state_name": "RUNNING",
        "observation": {"STATE_NAME": "RUNNING", "RW_Tank_Level": 600},
        "candidate_actions": [
            {
                "id": "attack:set_sensor:RW_Tank_Level:1210",
                "summary": "Raise raw-water level past the high-high threshold.",
                "action": {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1210},
            },
            {"id": "noop", "summary": "Hold position.", "action": {"type": "noop"}},
        ],
        "remaining_playbook": [],
        "recent_observations": [],
        "recent_actions": [],
        "recent_alerts": [],
        "recent_action_effects": [],
        "active_alarms": [],
        "active_states": [],
        "active_commands": [],
        "short_horizon_trends": {},
        "relationship_notes": [],
        "risk_flags": [],
        "incident": None,
        "structured_memory": {},
        "structured_prior_knowledge": None,
        "system_knowledge": {},
        "knowledge_ablation": None,
        "mitre_context": {},
        "staged_path": [],
        "decision_epoch": {},
        "role_goal": [],
        "role_constraints": [],
        "expected_outcomes": [],
        "safety_bounds": {"RW_Tank_Level": {"min": 450, "max": 1200}},
    }


def test_pure_react_prompt_uses_bounded_react_schema() -> None:
    engine = CachedQwenDecisionEngine(
        {
            "role": "attacker",
            "knowledge_mode": "white_box",
            "scaffold_mode": "pure_react",
            "llm": {
                "backend": "qwen_langgraph",
                "prompt_file": "testbeds/water_treatment/prompts/white_box_attacker.md",
                "local_files_only": True,
                "require_cuda": True,
            },
        },
        "attacker",
    )

    prompt = engine._format_prompt(_minimal_decision_context())

    assert "Use a bounded ReAct scaffold" in prompt
    assert "observation_summary" in prompt
    assert "reasoning_summary" in prompt
    assert "expected_effect" in prompt
    assert "Never invent a new action or a new candidate id" in prompt
    assert "attack:set_sensor:RW_Tank_Level:1210" in prompt


def test_pure_react_payload_records_reasoning_metadata() -> None:
    agent = AgenticRoleAgent(
        "white_box_pure_react_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "knowledge_mode": "white_box",
            "scaffold_mode": "pure_react",
            "allowed_action_types": ["inspect_surface", "set_sensor", "noop"],
            "llm": {
                "backend": "qwen_langgraph",
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
            metadata={"testbed_id": "water_treatment"},
            testbed_id="water_treatment",
        )
    )
    context = _minimal_decision_context()

    decisions = agent._payload_to_decisions(
        {
            "observation_summary": "The process is running and raw-water level is nominal.",
            "reasoning_summary": "The selected candidate stresses the high-high boundary directly.",
            "summary": "Spoof raw-water high.",
            "candidate_id": "attack:set_sensor:RW_Tank_Level:1210",
            "expected_effect": "The high-high alarm or related state change should become visible.",
            "confidence": 0.83,
            "diagnosis": "high_high_boundary_probe",
            "labels": ["agentic", "react", "attacker"],
            "decision_type": "action",
        },
        context,
    )

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action == {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1210}
    assert decision.metadata["scaffold_mode"] == "pure_react"
    assert "high-high boundary" in decision.metadata["reasoning_summary"]
    assert "state change" in decision.metadata["expected_effect"]


def test_pure_react_smoke_config_selects_four_silent_defender_runs() -> None:
    config = load_config(SMOKE_CONFIG)
    catalog = ScenarioCatalog(testbed_id="water_treatment")
    selected = catalog.resolve_selectors(list(config["scenario_selectors"]))
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    conditions = orchestrator._enumerate_conditions(config)
    condition = conditions[0]
    profiles = condition["profiles"]
    effective = condition["effective_profiles"]
    decoding = orchestrator._collect_decoding_settings(
        profiles,
        effective_profiles=effective,
    )

    assert [scenario.scenario_id for scenario in selected] == SMOKE_SCENARIOS
    assert profiles["attacker"] == "white_box_pure_react_attacker"
    assert profiles["defender"] == "silent_defender"
    assert profiles["observer"] == "silent_observer"
    assert effective["attacker"]["scaffold_mode"] == "pure_react"
    assert effective["attacker"]["react_context_mode"] == "compact"
    assert effective["attacker"]["max_llm_decisions_per_episode"] == 4
    assert effective["attacker"]["llm"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert effective["attacker"]["llm"].get("trust_remote_code", True) is True
    assert effective["attacker"]["llm"].get("fatal_on_parse_error", True) is True
    assert effective["attacker"]["llm"]["quantization"] == "4bit"
    assert effective["attacker"]["llm"]["max_new_tokens"] == 96
    assert effective["attacker"]["reasoning_cooldown_sec"] == 1.0
    assert effective["attacker"]["reasoning_refresh_sec"] == 2.0
    assert effective["defender"]["allowed_action_types"] == ["noop"]
    assert effective["observer"]["allowed_action_types"] == ["noop"]
    assert decoding["attacker"]["scaffold_mode"] == "pure_react"
    assert config["environment"]["backend"] == "sphere_openplc"
    assert config["environment"]["infrastructure_profile"] == "water_treatment"
    assert config["environment_runtime"]["io_timeout_sec"] == 8
    assert orchestrator._environment_config(config)["io_timeout_sec"] == 8


def test_qwen35_checkpoint_selects_conditional_generation_loader() -> None:
    from transformers import (  # type: ignore
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    config = AutoConfig.from_pretrained(
        Path("models/local_models/qwen3.5-9b"),
        local_files_only=True,
        trust_remote_code=True,
    )

    selected = CachedQwenDecisionEngine._select_transformers_auto_model_class(
        config,
        causal_cls=AutoModelForCausalLM,
        image_text_cls=AutoModelForImageTextToText,
    )

    assert type(config).__name__ == "Qwen3_5Config"
    assert selected is AutoModelForImageTextToText


def test_pure_react_parse_failure_can_be_wrapped_into_bounded_commit_action() -> None:
    agent = AgenticRoleAgent(
        "white_box_pure_react_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "knowledge_mode": "white_box",
            "scaffold_mode": "pure_react",
            "white_box_commit_guardrail": True,
            "allowed_action_types": ["inspect_surface", "set_sensor", "noop"],
            "behavior_model": {
                "id": "optionb_mitre_staged",
                "policy": "agentic",
                "white_box_commit_guardrail": True,
            },
            "llm": {
                "backend": "qwen_langgraph",
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
            metadata={"testbed_id": "water_treatment"},
            testbed_id="water_treatment",
        )
    )
    context = _minimal_decision_context()
    context["structured_memory"] = {"white_box_direct_commit": True}
    context["candidate_actions"][0].update(
        {
            "category": "exploit",
            "priority": 180,
            "surface": "raw_water_sensor_path",
        }
    )

    fallback_payload = agent.engine._parse_failure_payload(context)
    wrapped_payload = agent._enforce_gray_box_payload(fallback_payload, context)
    decisions = agent._payload_to_decisions(wrapped_payload, context)

    assert len(decisions) == 1
    assert decisions[0].decision_type == "action"
    assert decisions[0].action == {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1210}
    assert "commitment_guardrail" in decisions[0].labels
    assert decisions[0].metadata["parse_failure_fallback"] is True


def test_pure_react_four_attack_smoke_executes_with_silent_defender(
    fake_openplc,
    monkeypatch,
    tmp_path: Path,
) -> None:
    del fake_openplc

    def fake_react_decide(
        self: CachedQwenDecisionEngine,
        context: dict,
    ) -> dict:
        self.last_mode = "unit-test-react"
        self.last_error = None
        selected = next(
            (
                item
                for item in context.get("candidate_actions", [])
                if (item.get("action") or {}).get("type") != "noop"
            ),
            None,
        )
        if selected is None:
            return {
                "observation_summary": "No executable candidate is currently exposed.",
                "reasoning_summary": "Hold because the bounded candidate set contains no action.",
                "summary": "Hold.",
                "candidate_id": None,
                "expected_effect": "No plant change is expected.",
                "confidence": 0.4,
                "diagnosis": None,
                "labels": ["agentic", "react", "unit-test"],
                "decision_type": "note",
            }
        return {
            "observation_summary": "A bounded candidate action is available.",
            "reasoning_summary": "Select the first non-noop bounded candidate to exercise execution.",
            "summary": "Execute bounded candidate.",
            "candidate_id": selected["id"],
            "expected_effect": "The fake runtime should record the selected action.",
            "confidence": 0.91,
            "diagnosis": "react_smoke_candidate",
            "labels": ["agentic", "react", "unit-test"],
            "decision_type": "action",
        }

    monkeypatch.setattr(CachedQwenDecisionEngine, "decide", fake_react_decide)
    config_path = tmp_path / "pure_react_four_attack_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "pure_react_four_attack_smoke",
                "experiment_id": "pure_react_four_attack_smoke",
                "testbed": "water_treatment",
                "scenario_selectors": [
                    f"attack/{scenario_id}" for scenario_id in SMOKE_SCENARIOS
                ],
                "repeats": 1,
                "resume": False,
                "use_canonical_result_root": True,
                "reproducibility": {"seed": 101},
                "results_root": str(tmp_path / "results"),
                "execution_mode": "live",
                "environment": {
                    "backend": "auto",
                    "testbed": "water_treatment",
                    "keep_running": True,
                    "start_bridge": False,
                    "reset_wait_sec": 0,
                    "loop_sleep_sec": 0,
                },
                "conditions": [
                    {
                        "name": "white_box_pure_react_attacker__silent_defender",
                        "profiles": {
                            "attacker": "white_box_pure_react_attacker",
                            "defender": "silent_defender",
                            "observer": "silent_observer",
                            "planner": "scripted_planner",
                            "operator": "scripted_operator",
                            "offline_analysis": "scripted_offline_analyst",
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    summary = ExperimentOrchestrator(testbed_id="water_treatment").run(config_path)
    runs = summary["conditions"][0]["runs"]

    assert summary["status"] == "completed"
    assert [run["scenario_id"] for run in runs] == SMOKE_SCENARIOS
    assert all(run["status"] == "completed" for run in runs)
    assert all((run.get("metrics") or {}).get("attacker_action_count", 0) > 0 for run in runs)
    assert all((run.get("metrics") or {}).get("defender_action_count", 0) == 0 for run in runs)


def _autonomous_agent() -> AgenticRoleAgent:
    config = load_config(AUTONOMOUS_SMOKE_CONFIG)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profile = orchestrator._enumerate_conditions(config)[0]["effective_profiles"][
        "attacker"
    ]
    agent = AgenticRoleAgent("white_box_autonomous_react_attacker", profile)
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(
        "attack/wt-attack-sensor-rw-cascade-low-hh"
    )
    agent.reset(scenario)
    return agent


def _autonomous_context(
    scenario_id: str = "attack/wt-attack-sensor-rw-cascade-low-hh",
    *,
    executed_actions: list[dict] | None = None,
) -> tuple[ScenarioDefinition, dict, dict]:
    scenario = ScenarioCatalog(testbed_id="water_treatment").get(scenario_id)
    observation = {
        "STATE_NAME": "RUNNING",
        "RW_Tank_Level": 600,
        "UF_UFFT_Tank_Level": 400,
        "P2_Selected": 0,
        "ChemTreat_NaCl_Level": 800,
        "ChemTreat_NaOCl_Level": 800,
        "ChemTreat_HCl_Level": 800,
    }
    context = {
        "scenario": scenario,
        "time_sec": 1.0,
        "observation": observation,
        "observations": [observation],
        "executed_actions": list(executed_actions or []),
        "decisions": [],
        "alerts": [],
    }
    telemetry_summary = {
        "risk_flags": [],
        "active_alarms": [],
        "relationship_notes": [],
        "active_states": ["RUNNING"],
        "active_commands": [],
        "short_horizon_trends": {},
    }
    return scenario, context, telemetry_summary


def _action_types(candidates: list[dict]) -> set[str]:
    return {
        str((item.get("action") or {}).get("type"))
        for item in candidates
        if item.get("action")
    }


def test_autonomous_react_profile_keeps_old_white_box_profile_intact() -> None:
    config = load_config(AUTONOMOUS_SMOKE_CONFIG)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profiles = orchestrator.agent_profiles

    autonomous = orchestrator._enumerate_conditions(config)[0]["effective_profiles"][
        "attacker"
    ]
    old_white_box = profiles["white_box_attacker"]
    old_pure_react = profiles["white_box_pure_react_attacker"]

    assert autonomous["autonomous_react_tools"] is True
    assert autonomous["white_box_commit_guardrail"] is False
    assert old_white_box["white_box_commit_guardrail"] is True
    assert old_pure_react["white_box_commit_guardrail"] is True


def test_autonomous_react_starts_with_discovery_tools_only() -> None:
    agent = _autonomous_agent()
    _, context, telemetry_summary = _autonomous_context()
    memory = {"autonomous_react_tool_mode": True}

    candidates = agent._white_box_autonomous_react_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory=memory,
        mitre_context=None,
    )

    assert candidates
    assert _action_types(candidates) <= {"inspect_surface", "noop"}
    assert candidates[0]["category"] == "recon"
    assert candidates[0]["source"] == "autonomous_react_tool"
    assert "tool" in candidates[0]


def test_autonomous_react_offers_bounded_write_tools_after_recon() -> None:
    agent = _autonomous_agent()
    executed_actions = [
        {"role": "attacker", "time_sec": 0.5, "action": {"type": "inspect_surface", "surface": "tags"}},
        {"role": "attacker", "time_sec": 1.0, "action": {"type": "inspect_surface", "surface": "alarm_bits"}},
    ]
    _, context, telemetry_summary = _autonomous_context(executed_actions=executed_actions)
    memory = {"autonomous_react_tool_mode": True}

    candidates = agent._white_box_autonomous_react_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory=memory,
        mitre_context=None,
    )

    actions = [item["action"] for item in candidates]
    assert any(action.get("type") == "set_sensor" for action in actions)
    assert any(action.get("type") == "hmi_estop" for action in actions)
    assert any(item.get("tool") == "write_allowed_tag" for item in candidates)
    assert all(item["source"] == "autonomous_react_tool" for item in candidates)


def test_autonomous_react_build_context_removes_surface_priors() -> None:
    config = load_config(AUTONOMOUS_SMOKE_CONFIG)
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profile = orchestrator._enumerate_conditions(config)[0]["effective_profiles"][
        "attacker"
    ]
    agent = AgenticRoleAgent("white_box_autonomous_react_attacker", profile)
    scenario, context, _ = _autonomous_context()
    agent.reset(scenario)

    online_context = agent._build_online_context(context)
    system_knowledge = online_context["system_knowledge"]
    process_context = system_knowledge["process_context"]

    assert online_context["structured_memory"]["autonomous_react_tool_mode"] is True
    assert process_context["preferred_surface_order"] == []
    assert process_context["surface_leverage_hints"] == []
    assert system_knowledge["autonomous_react_context"]["starting_assumption"] == (
        "bounded_post_access_presence"
    )


def test_autonomous_react_four_attack_smoke_executes_with_silent_defender(
    fake_openplc,
    monkeypatch,
    tmp_path: Path,
) -> None:
    del fake_openplc

    def fake_react_decide(
        self: CachedQwenDecisionEngine,
        context: dict,
    ) -> dict:
        self.last_mode = "unit-test-autonomous-react"
        self.last_error = None
        selected = next(
            (
                item
                for item in context.get("candidate_actions", [])
                if (item.get("action") or {}).get("type") != "noop"
            ),
            None,
        )
        if selected is None:
            return {
                "observation_summary": "No validated tool is currently exposed.",
                "reasoning_summary": "Hold because the tool menu contains no executable action.",
                "summary": "Hold.",
                "candidate_id": None,
                "expected_effect": "No plant change is expected.",
                "confidence": 0.4,
                "diagnosis": None,
                "labels": ["agentic", "autonomous-react", "unit-test"],
                "decision_type": "note",
            }
        return {
            "observation_summary": "A bounded tool is available from the current access point.",
            "reasoning_summary": "Select a validated tool, then use later observations to revise.",
            "summary": "Use validated ReAct tool.",
            "candidate_id": selected["id"],
            "expected_effect": selected.get("expected_effect", "The fake runtime should record the action."),
            "confidence": 0.88,
            "diagnosis": "autonomous_react_tool_step",
            "labels": ["agentic", "autonomous-react", "unit-test"],
            "decision_type": "action",
        }

    monkeypatch.setattr(CachedQwenDecisionEngine, "decide", fake_react_decide)
    config_path = tmp_path / "autonomous_react_four_attack_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "autonomous_react_four_attack_smoke",
                "experiment_id": "autonomous_react_four_attack_smoke",
                "testbed": "water_treatment",
                "scenario_selectors": [
                    f"attack/{scenario_id}" for scenario_id in SMOKE_SCENARIOS
                ],
                "repeats": 1,
                "resume": False,
                "use_canonical_result_root": True,
                "reproducibility": {"seed": 101},
                "results_root": str(tmp_path / "results"),
                "execution_mode": "live",
                "environment": {
                    "backend": "auto",
                    "testbed": "water_treatment",
                    "keep_running": True,
                    "start_bridge": False,
                    "reset_wait_sec": 0,
                    "loop_sleep_sec": 0,
                },
                "conditions": [
                    {
                        "name": "white_box_autonomous_react_attacker__silent_defender",
                        "profiles": {
                            "attacker": "white_box_autonomous_react_attacker",
                            "defender": "silent_defender",
                            "observer": "silent_observer",
                            "planner": "scripted_planner",
                            "operator": "scripted_operator",
                            "offline_analysis": "scripted_offline_analyst",
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )

    summary = ExperimentOrchestrator(testbed_id="water_treatment").run(config_path)
    runs = summary["conditions"][0]["runs"]

    assert summary["status"] == "completed"
    assert [run["scenario_id"] for run in runs] == SMOKE_SCENARIOS
    assert all(run["status"] == "completed" for run in runs)
    assert all((run.get("metrics") or {}).get("attacker_action_count", 0) > 0 for run in runs)
    assert all((run.get("metrics") or {}).get("defender_action_count", 0) == 0 for run in runs)
