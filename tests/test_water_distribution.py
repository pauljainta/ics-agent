from __future__ import annotations

from pathlib import Path

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.environment import OpenPLCEnvironment
from ics_agent.scenarios import ScenarioCatalog
from ics_agent.testbeds.water_distribution import OpenPLCWaterDistributionTestbed


def test_water_distribution_catalog_loads_local_attack_suite() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")

    scenarios = catalog.resolve_selectors(["attack/*"])

    assert len(scenarios) == 12
    assert all(item.testbed_id == "water_distribution" for item in scenarios)
    assert {
        item.catalog_id for item in scenarios
    } >= {
        "attack/wd-attack-supply-low-spoof",
        "attack/wd-attack-grid-low-spoof",
        "attack/wd-attack-coordinated-grid-rws-delay",
    }

    grid_low = catalog.get("attack/wd-attack-grid-low-spoof")
    assert grid_low.metadata["name"] == "Elevated Tank Low Spoof"
    assert (
        grid_low.metadata["ground_truth"]["attack_class"] == "single_point_deception"
    )
    assert Path(grid_low.execution_path).is_file()


def test_water_distribution_environment_uses_local_testbed_and_actuator_containment() -> None:
    env = OpenPLCEnvironment(
        {
            "testbed": "water_distribution",
            "backend": "auto",
            "start_bridge": False,
        }
    )

    start = env.start()
    reset = env.reset()
    surface = env.inspect_surface({"type": "inspect_surface", "surface": "tags"})
    containment = env.act({"type": "block_actuator_write"}, role="defender")
    blocked = env.act(
        {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
        role="attacker",
    )

    assert start["started"] is True
    assert start["testbed"] == "water_distribution"
    assert Path(start["support_root"]).name == "openplc"
    assert reset["state_name"] == "IDLE"
    assert surface["reconnaissance"]["surface"] == "tags"
    assert "commands" in surface["reconnaissance"]["groups"]
    assert containment["containment"]["execution_mode"] == "runtime_enforced"
    assert containment["runtime_containment"]["target_surfaces"] == [
        "actuator_write_path"
    ]
    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "actuator_write_path"


def test_supply_pump_max_scenario_constrains_agentic_attacker_to_actuator_path() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-pump-max")

    assert scenario.metadata["role_bindings"] == {"operator": [0, 2], "attacker": [1]}
    assert scenario.metadata["attacker_constraints"]["allowed_actuator_names"] == [
        "Supply_Pump_Speed"
    ]
    assert scenario.metadata["attacker_constraints"]["allowed_actuator_values"] == {
        "Supply_Pump_Speed": [100]
    }
    assert scenario.metadata["ground_truth"]["incident_start_time_sec"] == 12

    agent = AgenticRoleAgent(
        "gray_box_attacker",
        {
            "role": "attacker",
            "mode": "online",
            "policy": "agentic",
            "knowledge_mode": "gray_box",
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
    agent.reset(scenario)

    context = {
        "scenario": scenario,
        "observation": {
            "STATE_NAME": "RUNNING",
            "Supply_Tank_Level": 480.0,
            "Supply_Pump_Flow": 150.0,
            "Supply_NaOCl_Level": 800.0,
            "Supply_NH4Cl_Level": 800.0,
            "Grid_Elev_Tank_Level": 580.0,
            "Grid_Consum_Tank_Level": 800.0,
            "RWS_Tank_Level": 280.0,
            "Supply_Pump_Sts": 1.0,
            "Supply_Tank_Valve_Sts": 1.0,
            "Grid_Elev_Valve_Sts": 1.0,
            "RWS_Pump_Sts": 0.0,
            "RWS_Tank_Valve_Sts": 0.0,
            "Alarm_Supply_Tank_HH": 0.0,
        },
        "executed_actions": [],
        "alerts": [],
        "remaining_playbook": [],
        "time_sec": 0.0,
    }
    telemetry_summary = {
        "risk_flags": [],
        "active_alarms": [],
        "relationship_notes": [],
    }

    pre_attack = agent._gray_box_attacker_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory={},
    )
    pre_attack_types = {
        (item.get("action") or {}).get("type")
        for item in pre_attack
    }
    assert "set_sensor" not in pre_attack_types
    assert "set_actuator" not in pre_attack_types

    context["time_sec"] = 12.0
    active_attack = agent._gray_box_attacker_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory={},
    )
    actuator_candidates = [
        item.get("action") or {}
        for item in active_attack
        if (item.get("action") or {}).get("type") == "set_actuator"
    ]

    assert actuator_candidates
    assert {
        str(item.get("name"))
        for item in actuator_candidates
    } == {"Supply_Pump_Speed"}
    assert {float(item.get("value", 0.0)) for item in actuator_candidates} == {100.0}


def test_supply_pump_max_scenario_starts_immediately_and_stays_nominal_pre_attack() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-pump-max")

    assert scenario.execution["timeline"][0]["time_sec"] == 0

    testbed = OpenPLCWaterDistributionTestbed(
        base_dir=Path("."),
        controller_host="localhost",
        controller_port=None,
        simulator_host="localhost",
        simulator_port=None,
        unit_id=1,
        use_remote_modbus=False,
    )
    testbed.reset(scenario.execution["initial_conditions"])

    for event in scenario.execution["timeline"]:
        if event["time_sec"] == 0:
            testbed.apply_action(event["action"])
            break

    pre_attack_alarm = False
    for _ in range(24):
        testbed._step_once(0.5)
        observation = testbed.observe_flat()
        if int(observation["Alarm_Supply_Tank_HH"]) == 1:
            pre_attack_alarm = True
            break

    assert pre_attack_alarm is False
