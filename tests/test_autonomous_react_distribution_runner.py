from __future__ import annotations

from pathlib import Path

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


CONFIG_NAME = (
    "water_distribution_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.yaml"
)
SCRIPT_NAME = (
    "run_water_distribution_autonomous_react_white_box_attacker_vs_agentic_full_containment_defender.py"
)


def _distribution_agent() -> tuple[AgenticRoleAgent, dict[str, object]]:
    config = load_config(CONFIG_NAME)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    condition = orchestrator._enumerate_conditions(config)[0]
    profile = condition["effective_profiles"]["attacker"]
    return AgenticRoleAgent("white_box_autonomous_react_attacker", profile), profile


def _distribution_context(scenario_id: str, *, recon_done: bool) -> dict[str, object]:
    scenario = ScenarioCatalog(testbed_id="water_distribution").get(scenario_id)
    executed_actions = []
    if recon_done:
        executed_actions = [
            {
                "role": "attacker",
                "action": {"type": "inspect_surface", "surface": "grid_control_plane"},
            },
            {
                "role": "attacker",
                "action": {"type": "inspect_surface", "surface": "actuator_write_path"},
            },
        ]
    return {
        "scenario": scenario,
        "observation": {
            "STATE_NAME": "RUNNING",
            "Supply_Tank_Level": 930,
            "Grid_Elev_Tank_Level": 920,
            "Grid_Consum_Tank_Level": 840,
            "RWS_Tank_Level": 320,
            "Supply_Pump_Speed_Cmd": 0,
            "Grid_Elev_Valve_Cmd": 0,
            "RWS_Pump_Speed_Cmd": 0,
            "RWS_Tank_Valve_Cmd": 0,
        },
        "observations": [],
        "executed_actions": executed_actions,
        "decisions": [],
        "alerts": [],
        "playbooks": {},
        "time_sec": 0.0,
    }


def _action_types(candidates: list[dict[str, object]]) -> set[str]:
    return {
        str((candidate.get("action") or {}).get("type"))
        for candidate in candidates
        if isinstance(candidate.get("action"), dict)
    }


def test_distribution_autonomous_react_runner_config_is_retained_mitre11() -> None:
    config = load_config(CONFIG_NAME)
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    selected = catalog.resolve_selectors(list(config["scenario_selectors"]))

    assert Path(config["_path"]).name == CONFIG_NAME
    assert config["testbed"] == "water_distribution"
    assert config["reproducibility"]["seed"] == 101
    assert config["resume"] is True
    assert config["use_canonical_result_root"] is True
    assert config["results_root"].endswith(
        "Artifact/Results/water_distribution_openplc/Autonomous-React/White-Box"
    )
    assert len(selected) == 11
    assert all(scenario.catalog_id.startswith("attack/") for scenario in selected)


def test_distribution_autonomous_react_condition_profiles_are_expected() -> None:
    config = load_config(CONFIG_NAME)
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    condition = orchestrator._enumerate_conditions(config)[0]
    effective_profiles = condition["effective_profiles"]

    assert condition["name"] == "white_box_autonomous_react_attacker__agentic_defender"
    assert condition["profiles"]["attacker"] == "white_box_autonomous_react_attacker"
    assert condition["profiles"]["defender"] == "agentic_defender"

    attacker = effective_profiles["attacker"]
    defender = effective_profiles["defender"]

    assert attacker["knowledge_mode"] == "white_box"
    assert attacker["autonomous_react_tools"] is True
    assert attacker["white_box_commit_guardrail"] is False
    assert attacker["reasoning_cooldown_sec"] == 0.0
    assert attacker["reasoning_refresh_sec"] == 0.0
    assert "set_actuator" in attacker["allowed_action_types"]
    assert attacker["llm"]["model_id"] == "Qwen/Qwen3.5-9B"
    assert attacker["llm"]["quantization"] == "4bit"

    assert "block_sensor_write" in defender["allowed_action_types"]
    assert "block_actuator_write" in defender["allowed_action_types"]
    assert "quarantine_signal_source" in defender["allowed_action_types"]
    assert defender["llm"]["model_id"] == "Qwen/Qwen3.5-9B"


def test_distribution_autonomous_react_starts_with_recon_tools_only() -> None:
    agent, _profile = _distribution_agent()
    context = _distribution_context("attack/wd-attack-grid-valve-force-open", recon_done=False)
    agent.reset(context["scenario"])

    candidates = agent._white_box_autonomous_react_candidates(
        context=context,
        telemetry_summary={"active_alarms": [], "risk_flags": [], "relationship_notes": []},
        structured_memory={"autonomous_react_tool_mode": True},
        mitre_context={},
    )

    types = _action_types(candidates)
    assert "inspect_surface" in types
    assert "set_sensor" not in types
    assert "set_actuator" not in types
    assert "hmi_stop" not in types


def test_distribution_autonomous_react_offers_sensor_and_actuator_tools_after_recon() -> None:
    agent, _profile = _distribution_agent()
    context = _distribution_context("attack/wd-attack-grid-valve-force-open", recon_done=True)
    agent.reset(context["scenario"])

    candidates = agent._white_box_autonomous_react_candidates(
        context=context,
        telemetry_summary={"active_alarms": [], "risk_flags": [], "relationship_notes": []},
        structured_memory={"autonomous_react_tool_mode": True},
        mitre_context={},
    )

    types = _action_types(candidates)
    assert "set_sensor" in types
    assert "set_actuator" in types
    assert any(
        (candidate.get("action") or {}).get("name") == "Grid_Elev_Valve"
        for candidate in candidates
        if isinstance(candidate.get("action"), dict)
    )
    assert all(candidate.get("source") == "autonomous_react_tool" for candidate in candidates)


def test_distribution_autonomous_react_runner_script_exists() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT_NAME

    assert script_path.exists()
    assert "CONFIG_PATH" in script_path.read_text()
