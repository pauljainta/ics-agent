from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import yaml

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.environment import OpenPLCEnvironment
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog
from testbeds.runtime.water_distribution_runtime import (
    ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP,
    ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP,
    OpenPLCWaterDistributionTestbed,
    SENSOR_OVERRIDE_ENABLE_REGISTER_MAP,
    SENSOR_OVERRIDE_VALUE_REGISTER_MAP,
    SIMULATOR_COMMAND_COIL_MAP,
    SIMULATOR_PROCESS_CONTROL_REGISTER_MAP,
    SIMULATOR_RUNTIME_HOLD_COIL,
)


def _distribution_defender_agent() -> AgenticRoleAgent:
    return AgenticRoleAgent(
        "agentic_defender",
        {
            "role": "defender",
            "family": "agentic",
            "mode": "online",
            "policy": "agentic",
            "allowed_action_types": [
                "hmi_stop",
                "hmi_estop",
                "disable_hmi_session",
                "isolate_engineering_host",
                "block_sensor_write",
                "block_actuator_write",
                "quarantine_signal_source",
                "noop",
            ],
        },
    )


def _distribution_observation(**overrides: float | str) -> dict[str, float | str]:
    observation: dict[str, float | str] = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 920.0,
        "Grid_Elev_Tank_Level": 1200.0,
        "Grid_Consum_Tank_Level": 780.0,
        "RWS_Tank_Level": 540.0,
        "Supply_Pump_Sts": 1.0,
        "Supply_Tank_Valve_Cmd": 1.0,
        "Supply_Tank_Valve_Sts": 1.0,
        "Grid_Elev_Valve_Cmd": 0.0,
        "Grid_Elev_Valve_Sts": 0.0,
        "RWS_Tank_Valve_Cmd": 0.0,
        "RWS_Tank_Valve_Sts": 0.0,
        "Alarm_Supply_Tank_HH": 0.0,
    }
    observation.update(overrides)
    return observation


def test_water_distribution_catalog_loads_local_attack_suite() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")

    scenarios = catalog.resolve_selectors(["attack/*"])

    assert len(scenarios) == 11
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


def test_water_distribution_defender_hides_passive_recon_history() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-low-spoof")
    agent = _distribution_defender_agent()
    agent.reset(scenario)

    observation = _distribution_observation()
    context = {
        "scenario": scenario,
        "observation": observation,
        "observations": [observation],
        "executed_actions": [
            {
                "role": "attacker",
                "time_sec": 4.0,
                "action": {"type": "inspect_surface", "surface": "alarm_bits"},
                "result": {
                    "category": "reconnaissance",
                    "surface": "alarm_bits",
                    "summary": "alarm_bits readback: bounded_readback_captured",
                    "findings": ["bounded_readback_captured"],
                },
            }
        ],
        "decisions": [
            {
                "role": "attacker",
                "decision_type": "action",
                "summary": "Inspect alarm bits to gauge defender-visible pressure.",
                "time_sec": 4.0,
                "action": {"type": "inspect_surface", "surface": "alarm_bits"},
                "metadata": {"mitre_phase": "discovery"},
            },
            {
                "role": "observer",
                "decision_type": "alert",
                "summary": "Observer recognized staged qualification of a process-facing control path.",
                "time_sec": 5.0,
                "diagnosis": "availability_disruption",
                "metadata": {"mitre_phase": "discovery_recognition"},
            },
        ],
        "alerts": [
            {
                "role": "observer",
                "summary": "Observer recognized staged qualification of a process-facing control path.",
                "time_sec": 5.0,
                "diagnosis": "availability_disruption",
                "metadata": {"mitre_phase": "discovery_recognition"},
            }
        ],
        "playbooks": {},
        "time_sec": 6.0,
    }

    built = agent._build_online_context(context)

    assert built["recent_actions"] == []
    assert built["recent_decisions"] == []
    assert built["recent_alerts"] == []
    assert built["recent_action_effects"] == []
    assert built["incident"]["type"] == "none"
    assert built["decision_epoch"]["action_count"] == 0
    assert {str((item.get("action") or {}).get("type")) for item in built["candidate_actions"]} == {
        "noop"
    }


def test_water_distribution_defender_keeps_visible_modifying_attack_evidence() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-grid-valve-force-open")
    agent = _distribution_defender_agent()
    agent.reset(scenario)

    previous_observation = _distribution_observation()
    observation = _distribution_observation(Grid_Elev_Valve_Sts=1.0)
    context = {
        "scenario": scenario,
        "observation": observation,
        "observations": [previous_observation, observation],
        "executed_actions": [
            {
                "role": "attacker",
                "time_sec": 4.0,
                "action": {"type": "set_actuator", "name": "Grid_Elev_Valve", "value": True},
                "result": {"status": "applied"},
            }
        ],
        "decisions": [],
        "alerts": [],
        "playbooks": {},
        "time_sec": 6.0,
    }

    built = agent._build_online_context(context)

    assert [item["action"]["type"] for item in built["recent_actions"]] == ["set_actuator"]
    assert [item["action_type"] for item in built["recent_action_effects"]] == ["set_actuator"]
    assert "command_status_divergence" in built["risk_flags"]
    assert "block_actuator_write" in {
        str((item.get("action") or {}).get("type")) for item in built["candidate_actions"]
    }


def test_water_distribution_white_box_candidates_include_grid_actuator_override() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-grid-valve-force-open")

    agent = AgenticRoleAgent(
        "white_box_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "mode": "online",
            "policy": "agentic",
            "knowledge_mode": "white_box",
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
            "Supply_Tank_Level": 940.0,
            "Supply_Pump_Flow": 150.0,
            "Supply_NaOCl_Level": 800.0,
            "Supply_NH4Cl_Level": 800.0,
            "Grid_Elev_Tank_Level": 1550.0,
            "Grid_Consum_Tank_Level": 820.0,
            "RWS_Tank_Level": 500.0,
            "Supply_Pump_Sts": 1.0,
            "Supply_Tank_Valve_Sts": 1.0,
            "Grid_Elev_Valve_Sts": 0.0,
            "RWS_Pump_Sts": 0.0,
            "RWS_Tank_Valve_Sts": 0.0,
            "Alarm_Supply_Tank_HH": 0.0,
        },
        "executed_actions": [],
        "alerts": [],
        "remaining_playbook": [],
        "time_sec": 18.0,
    }
    telemetry_summary = {
        "risk_flags": [],
        "active_alarms": [],
        "relationship_notes": [],
    }
    structured_memory = agent._white_box_attacker_memory(
        context=context,
        observation=context["observation"],
        telemetry_summary=telemetry_summary,
        recent_actions=[],
        recent_alerts=[],
        recent_action_effects=[],
    )

    candidates = agent._white_box_attacker_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory=structured_memory,
    )

    actuator_candidates = [
        item.get("action") or {}
        for item in candidates
        if (item.get("action") or {}).get("type") == "set_actuator"
    ]

    assert actuator_candidates
    assert {
        (str(item.get("name")), item.get("value"))
        for item in actuator_candidates
    } >= {("Grid_Elev_Valve", True)}


def test_water_distribution_white_box_candidates_expose_breach_capable_sensor_values() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-high-high-spoof")

    agent = AgenticRoleAgent(
        "white_box_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "mode": "online",
            "policy": "agentic",
            "knowledge_mode": "white_box",
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

    observation = {
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
    telemetry_summary = {
        "risk_flags": [],
        "active_alarms": [],
        "relationship_notes": [],
    }
    context = {
        "scenario": scenario,
        "observation": observation,
        "executed_actions": [],
        "alerts": [],
        "remaining_playbook": [],
        "time_sec": 16.0,
    }
    structured_memory = agent._white_box_attacker_memory(
        context=context,
        observation=observation,
        telemetry_summary=telemetry_summary,
        recent_actions=[],
        recent_alerts=[],
        recent_action_effects=[],
    )

    candidates = agent._white_box_attacker_candidates(
        context=context,
        telemetry_summary=telemetry_summary,
        structured_memory=structured_memory,
    )

    sensor_candidates = [
        item.get("action") or {}
        for item in candidates
        if (item.get("action") or {}).get("type") == "set_sensor"
    ]

    assert structured_memory["white_box_direct_commit"] is True
    assert any(
        str(item.get("name")) == "Supply_Tank_Level"
        and float(item.get("value", 0.0)) > 1500.0
        for item in sensor_candidates
    )


def test_water_distribution_white_box_system_knowledge_uses_distribution_topology() -> None:
    catalog = ScenarioCatalog(testbed_id="water_distribution")
    scenario = catalog.get("attack/wd-attack-supply-low-spoof")

    agent = AgenticRoleAgent(
        "white_box_attacker",
        {
            "role": "attacker",
            "family": "agentic",
            "mode": "online",
            "policy": "agentic",
            "knowledge_mode": "white_box",
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

    observation = {
        "STATE_NAME": "RUNNING",
        "Supply_Tank_Level": 860.0,
        "Supply_Pump_Flow": 150.0,
        "Supply_NaOCl_Level": 800.0,
        "Supply_NH4Cl_Level": 800.0,
        "Grid_Elev_Tank_Level": 940.0,
        "Grid_Consum_Tank_Level": 860.0,
        "RWS_Tank_Level": 500.0,
        "Supply_Pump_Sts": 1.0,
        "Supply_Tank_Valve_Sts": 1.0,
        "Grid_Elev_Valve_Sts": 1.0,
        "RWS_Pump_Sts": 0.0,
        "RWS_Tank_Valve_Sts": 0.0,
        "Alarm_Supply_Tank_HH": 0.0,
    }
    telemetry_summary = {
        "risk_flags": [],
        "active_alarms": [],
        "relationship_notes": [],
    }
    structured_memory = agent._white_box_attacker_memory(
        context={
            "scenario": scenario,
            "observation": observation,
            "executed_actions": [],
            "alerts": [],
            "remaining_playbook": [],
            "time_sec": 16.0,
        },
        observation=observation,
        telemetry_summary=telemetry_summary,
        recent_actions=[],
        recent_alerts=[],
        recent_action_effects=[],
    )

    system_knowledge = agent._white_box_attacker_system_knowledge(
        scenario=scenario,
        observation=observation,
        telemetry_summary=telemetry_summary,
        structured_memory=structured_memory,
        defender_reasoning=None,
        mitre_context=None,
    )

    process_assets = {
        item["asset"] for item in system_knowledge["topology"]["process_paths"]
    }
    surfaces = {
        item["surface"] for item in system_knowledge["topology"]["surface_relationships"]
    }

    assert {"WD-Supply", "WD-Grid", "WD-RWS"} <= process_assets
    assert {"sensor_write_path", "actuator_write_path"} <= surfaces
    assert "breach-capable values above a published safety max" in system_knowledge[
        "control_semantics"
    ]["set_sensor"]
    sensor_hint = next(
        item
        for item in system_knowledge["process_context"]["surface_leverage_hints"]
        if item["surface"] == "sensor_write_path"
    )
    assert sensor_hint["breach_capable"] is True
    assert float(sensor_hint["preferred_value"]) > 1500.0


def test_water_distribution_preserves_breach_capable_sensor_overrides_and_clamps_actuators() -> None:
    testbed = OpenPLCWaterDistributionTestbed(
        base_dir=Path("."),
        controller_host="localhost",
        controller_port=None,
        simulator_host="localhost",
        simulator_port=None,
        unit_id=1,
        use_remote_modbus=False,
    )
    testbed.reset()

    testbed.apply_action(
        {"type": "set_sensor", "name": "Supply_Tank_Level", "value": 1540}
    )
    testbed.apply_action(
        {"type": "set_sensor", "name": "Supply_Tank_Level", "value": 70575}
    )
    testbed.apply_action(
        {"type": "set_actuator", "name": "Supply_Pump_Speed", "value": 500}
    )

    assert testbed._sensor_overrides["Supply_Tank_Level"] == 65535.0
    assert testbed._actuator_overrides["Supply_Pump_Speed"] == 100.0


def test_water_distribution_sensor_override_accepts_above_bound_breach_value() -> None:
    testbed = OpenPLCWaterDistributionTestbed(
        base_dir=Path("."),
        controller_host="localhost",
        controller_port=None,
        simulator_host="localhost",
        simulator_port=None,
        unit_id=1,
        use_remote_modbus=False,
    )
    testbed.reset()

    testbed.apply_action(
        {"type": "set_sensor", "name": "Supply_Tank_Level", "value": 1540}
    )

    assert testbed._sensor_overrides["Supply_Tank_Level"] == 1540.0


def test_water_distribution_remote_clear_external_controls_clears_remote_overrides(
    monkeypatch,
) -> None:
    testbed = OpenPLCWaterDistributionTestbed(
        base_dir=Path("."),
        controller_host="controller",
        controller_port=1502,
        simulator_host="simulator",
        simulator_port=1503,
        unit_id=1,
        use_remote_modbus=True,
    )
    writes: list[tuple[str, int, int]] = []

    @contextmanager
    def fake_clients():
        yield "ctrl", "sim"

    monkeypatch.setattr(testbed, "clients", fake_clients)
    monkeypatch.setattr(
        testbed,
        "_write_remote_register",
        lambda client, address, value: writes.append((client, address, value)),
    )

    testbed._clear_external_controls()

    expected_writes = []
    for field in SENSOR_OVERRIDE_VALUE_REGISTER_MAP:
        expected_writes.append(("sim", SENSOR_OVERRIDE_ENABLE_REGISTER_MAP[field], 0))
        expected_writes.append(("sim", SENSOR_OVERRIDE_VALUE_REGISTER_MAP[field], 0))
    for field in ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP:
        expected_writes.append(("ctrl", ACTUATOR_OVERRIDE_ENABLE_REGISTER_MAP[field], 0))
        expected_writes.append(("ctrl", ACTUATOR_OVERRIDE_VALUE_REGISTER_MAP[field], 0))

    assert writes == expected_writes


def test_water_distribution_remote_reset_primes_hold_and_zeroes_runtime_inputs(
    monkeypatch,
) -> None:
    testbed = OpenPLCWaterDistributionTestbed(
        base_dir=Path("."),
        controller_host="controller",
        controller_port=1502,
        simulator_host="simulator",
        simulator_port=1503,
        unit_id=1,
        use_remote_modbus=True,
    )
    coil_writes: list[tuple[str, int, bool]] = []
    register_writes: list[tuple[str, int, int]] = []

    @contextmanager
    def fake_clients():
        yield "ctrl", "sim"

    monkeypatch.setattr(testbed, "clients", fake_clients)
    monkeypatch.setattr(
        testbed,
        "_write_remote_coil",
        lambda client, address, value: coil_writes.append((client, address, bool(value))),
    )
    monkeypatch.setattr(
        testbed,
        "_write_remote_register",
        lambda client, address, value: register_writes.append((client, address, int(value))),
    )
    monkeypatch.setattr(
        testbed,
        "observe_flat",
        lambda: {
            "STATE_NAME": "IDLE",
            "Supply_Tank_Level": 920,
            "Grid_Elev_Tank_Level": 940,
            "RWS_Tank_Level": 500,
        },
    )

    snapshot = testbed.reset(
        {
            "Supply_Tank_Level": 920,
            "Grid_Elev_Tank_Level": 940,
            "RWS_Tank_Level": 500,
        },
        wait_sec=0.0,
    )

    assert snapshot["STATE_NAME"] == "IDLE"
    assert ("ctrl", 2, True) in coil_writes
    assert ("ctrl", 2, False) in coil_writes
    assert ("sim", SIMULATOR_RUNTIME_HOLD_COIL, True) in coil_writes
    for address in SIMULATOR_COMMAND_COIL_MAP.values():
        assert ("sim", address, False) in coil_writes
    for address in SIMULATOR_PROCESS_CONTROL_REGISTER_MAP.values():
        assert ("sim", address, 0) in register_writes


def test_distribution_white_box_config_matches_treatment_strength_knobs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    treatment_config = yaml.safe_load(
        (
            repo_root
            / "configs/experiments/water_treatment_agentic_white_box_attacker_vs_agentic_full_containment_defender.yaml"
        ).read_text()
    )
    distribution_config = yaml.safe_load(
        (
            repo_root
            / "configs/experiments/water_distribution_white_box_vs_agentic_defender.yaml"
        ).read_text()
    )
    attack_families_config = yaml.safe_load(
        (
            repo_root
            / "tests/fixtures/water_distribution_attack_families.yaml"
        ).read_text()
    )
    treatment_orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    distribution_orchestrator = ExperimentOrchestrator(testbed_id="water_distribution")

    treatment_attacker = treatment_orchestrator._effective_condition_profiles(
        profiles=treatment_config["conditions"][0]["profiles"],
        profile_overrides=treatment_config["conditions"][0]["profile_overrides"],
    )["attacker"]
    distribution_attacker = distribution_orchestrator._effective_condition_profiles(
        profiles=distribution_config["conditions"][0]["profiles"],
        profile_overrides=distribution_config["conditions"][0]["profile_overrides"],
    )["attacker"]
    attack_families_attacker = distribution_orchestrator._effective_condition_profiles(
        profiles=attack_families_config["conditions"][2]["profiles"],
        profile_overrides=attack_families_config["conditions"][2]["profile_overrides"],
    )["attacker"]

    assert distribution_config["reproducibility"]["seed"] == treatment_config["reproducibility"][
        "seed"
    ]
    assert distribution_attacker["behavior_model"]["id"] == treatment_attacker[
        "behavior_model"
    ]["id"]
    assert distribution_attacker["behavior_model"]["policy"] == treatment_attacker[
        "behavior_model"
    ]["policy"]
    assert distribution_attacker["action_cooldown_sec"] == treatment_attacker[
        "action_cooldown_sec"
    ]
    assert distribution_attacker["note_cooldown_sec"] == treatment_attacker[
        "note_cooldown_sec"
    ]
    assert distribution_attacker["reasoning_cooldown_sec"] == treatment_attacker[
        "reasoning_cooldown_sec"
    ]
    assert distribution_attacker["reasoning_refresh_sec"] == treatment_attacker[
        "reasoning_refresh_sec"
    ]
    assert distribution_attacker["memory"]["attacker_candidate_window"] == treatment_attacker[
        "memory"
    ]["attacker_candidate_window"]
    for key in (
        "backend",
        "use_langgraph",
        "strict_backend",
        "allow_langchain_fallback",
        "max_new_tokens",
    ):
        assert distribution_attacker["llm"][key] == treatment_attacker["llm"][key]
        assert attack_families_attacker["llm"][key] == distribution_attacker["llm"][key]
    for key in (
        "action_cooldown_sec",
        "note_cooldown_sec",
        "reasoning_cooldown_sec",
        "reasoning_refresh_sec",
        "memory",
    ):
        assert attack_families_attacker[key] == distribution_attacker[key]
    assert attack_families_attacker["behavior_model"]["id"] == distribution_attacker[
        "behavior_model"
    ]["id"]
    assert attack_families_attacker["behavior_model"]["policy"] == distribution_attacker[
        "behavior_model"
    ]["policy"]

    assert {
        "testbeds/water_distribution/docs/process-models/wd-supply-line.md",
        "testbeds/water_distribution/docs/process-models/wd-distribution-grid.md",
        "testbeds/water_distribution/docs/process-models/wd-return-water.md",
    } <= set(distribution_attacker["knowledge_files"])


def test_distribution_white_box_prompt_uses_treatment_staged_wording() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prompt_text = (
        repo_root / "testbeds/water_distribution/prompts/white_box_attacker.md"
    ).read_text()

    assert "abstract MITRE ATT&CK for ICS inspired staged-entry model" in prompt_text
    assert (
        "The staged path is mandatory context: local orchestrator -> access -> engineering -> plant runtime."
        in prompt_text
    )
    assert (
        "Follow the staged chain in order: staged entry context, discovery, process understanding, control-path qualification, then manipulation and impact."
        in prompt_text
    )
    assert (
        "When the candidate list exposes a sensor value above a published safety bound"
        in prompt_text
    )
