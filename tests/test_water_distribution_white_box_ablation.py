from __future__ import annotations

import json

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.paths import REPO_ROOT
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.utils import load_yaml
from ics_agent.water_distribution_white_box_ablation import FAMILY_SPECS


def _distribution_observation() -> dict[str, float | str]:
    return {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 940.0,
        "Supply_Pump_Flow": 150.0,
        "Supply_NaOCl_Level": 800.0,
        "Supply_NH4Cl_Level": 800.0,
        "Grid_Elev_Tank_Level": 1200.0,
        "Grid_Consum_Tank_Level": 820.0,
        "RWS_Tank_Level": 500.0,
        "Supply_Pump_Sts": 1.0,
        "Supply_Tank_Valve_Sts": 1.0,
        "Grid_Elev_Valve_Sts": 0.0,
        "RWS_Pump_Sts": 0.0,
        "RWS_Tank_Valve_Sts": 0.0,
        "Alarm_Supply_Tank_HH": 0.0,
    }


def _ablation_context(scenario):
    observation = _distribution_observation()
    return {
        "scenario": scenario,
        "observation": observation,
        "observations": [observation],
        "executed_actions": [],
        "decisions": [],
        "alerts": [],
        "remaining_playbook": [],
        "time_sec": 16.0,
    }


def _ablation_agent(ablation: str) -> AgenticRoleAgent:
    return AgenticRoleAgent(
        f"white_box_{ablation}_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "mode": "online",
            "knowledge_mode": "white_box",
            "white_box_knowledge_ablation": ablation,
            "allowed_action_types": [
                "inspect_surface",
                "hmi_start",
                "hmi_stop",
                "hmi_estop",
                "set_sensor",
                "set_actuator",
                "noop",
            ],
        },
    )


def test_distribution_white_box_ablation_configs_preserve_attacker_action_space() -> None:
    orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")
    baseline_config = load_yaml(
        REPO_ROOT
        / "configs/experiments/water_distribution_white_box_vs_agentic_defender.yaml"
    )
    baseline_condition = orchestrator._enumerate_conditions(baseline_config)[0]
    baseline_attacker = baseline_condition["effective_profiles"]["attacker"]
    baseline_allowed = list(baseline_attacker["allowed_action_types"])

    expected_manuals = {
        "white_box_no_safety_bounds__agentic_defender": "testbeds/water_distribution/knowledge/white_box_no_safety_bounds_attacker_manual.md",
        "white_box_no_surface_priors__agentic_defender": "testbeds/water_distribution/knowledge/white_box_no_surface_priors_attacker_manual.md",
        "white_box_no_control_semantics__agentic_defender": "testbeds/water_distribution/knowledge/white_box_no_control_semantics_attacker_manual.md",
    }
    expected_ablations = {
        "white_box_no_safety_bounds__agentic_defender": "no_safety_bounds",
        "white_box_no_surface_priors__agentic_defender": "no_surface_priors",
        "white_box_no_control_semantics__agentic_defender": "no_control_semantics",
    }

    for family in FAMILY_SPECS:
        config = load_yaml(family.config_path)
        condition = orchestrator._enumerate_conditions(config)[0]
        attacker = condition["effective_profiles"]["attacker"]

        assert attacker["knowledge_mode"] == "white_box"
        assert list(attacker["allowed_action_types"]) == baseline_allowed
        assert attacker["white_box_knowledge_ablation"] == expected_ablations[
            family.family_name
        ]
        assert attacker["llm"]["prompt_file"] == (
            "testbeds/water_distribution/prompts/white_box_attacker_ablation.md"
        )
        assert attacker["knowledge_files"] == [expected_manuals[family.family_name]]


def test_distribution_white_box_no_safety_bounds_hides_explicit_thresholds() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-high-high-spoof")
    agent = _ablation_agent("no_safety_bounds")
    agent.reset(scenario)

    online_context = agent._build_online_context(_ablation_context(scenario))

    serialized = json.dumps(
        {
            "knowledge_ablation": online_context["knowledge_ablation"],
            "safety_bounds": online_context["safety_bounds"],
            "system_knowledge": online_context["system_knowledge"],
        },
        sort_keys=True,
    )
    sensor_summaries = [
        item["summary"]
        for item in online_context["candidate_actions"]
        if str((item.get("action") or {}).get("type")) == "set_sensor"
    ]
    leverage_hints = online_context["system_knowledge"]["process_context"][
        "surface_leverage_hints"
    ]

    assert online_context["knowledge_ablation"]["variant"] == "no_safety_bounds"
    assert online_context["safety_bounds"] == {}
    assert online_context["system_knowledge"]["safety_envelope"] == {}
    assert "1500" not in serialized
    assert "2000" not in serialized
    assert "1200" not in serialized
    assert sensor_summaries
    assert all(" known " not in summary for summary in sensor_summaries)
    assert all("published" not in summary for summary in sensor_summaries)
    assert leverage_hints
    assert all("preferred_value" not in item for item in leverage_hints)
    assert all("alternate_value" not in item for item in leverage_hints)
    assert all("breach_capable" not in item for item in leverage_hints)


def test_distribution_white_box_no_surface_priors_removes_leverage_hints() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-grid-low-spoof")
    agent = _ablation_agent("no_surface_priors")
    agent.reset(scenario)

    online_context = agent._build_online_context(_ablation_context(scenario))
    process_context = online_context["system_knowledge"]["process_context"]
    summaries = [item["summary"] for item in online_context["candidate_actions"]]

    assert online_context["knowledge_ablation"]["variant"] == "no_surface_priors"
    assert "white_box_surface_priors" not in online_context["structured_memory"]
    assert process_context["preferred_surface_order"] == []
    assert process_context["surface_leverage_hints"] == []
    assert all("fastest bounded distribution leverage path" not in summary for summary in summaries)
    assert all("best" not in summary.lower() for summary in summaries)


def test_distribution_white_box_no_control_semantics_withholds_operational_meaning() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-grid-valve-force-open")
    agent = _ablation_agent("no_control_semantics")
    agent.reset(scenario)

    online_context = agent._build_online_context(_ablation_context(scenario))
    system_knowledge = online_context["system_knowledge"]
    summaries = [item["summary"] for item in online_context["candidate_actions"]]

    assert online_context["knowledge_ablation"]["variant"] == "no_control_semantics"
    assert system_knowledge["control_semantics"] == {"withheld": True}
    assert system_knowledge["process_context"]["active_alarms"] == []
    assert system_knowledge["process_context"]["risk_flags"] == []
    assert system_knowledge["process_context"]["relationship_notes"] == []
    assert all(
        "role" not in item for item in system_knowledge["topology"]["process_paths"]
    )
    assert all(
        "maps_to" not in item
        for item in system_knowledge["topology"]["surface_relationships"]
    )
    assert all("controller" not in summary for summary in summaries)
    assert all("real process effect" not in summary for summary in summaries)
    assert all("fill path" not in summary for summary in summaries)
    assert all("recirculation" not in summary for summary in summaries)
