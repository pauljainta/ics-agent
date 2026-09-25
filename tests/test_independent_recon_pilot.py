from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from ics_agent.agents.agentic import AgenticRoleAgent
from ics_agent.config import load_config
from ics_agent.environment import EnvironmentError, OpenPLCEnvironment
from ics_agent.orchestrator import ExperimentOrchestrator
from ics_agent.scenarios import ScenarioCatalog


CONFIG_NAME = (
    "water_treatment_independent_recon_attacker_vs_silent_defender_pilot.yaml"
)
SCRIPT_NAME = (
    "run_water_treatment_independent_recon_attacker_vs_silent_defender_pilot.py"
)


def _agent_and_scenario():
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    profile = orchestrator.agent_profiles["independent_recon_attacker"]
    scenario = ScenarioCatalog(testbed_id="water_treatment").resolve_selectors(
        ["attack/wt-attack-sensor-rw-hh-alarm"]
    )[0]
    agent = AgenticRoleAgent("independent_recon_attacker", profile)
    agent.reset(scenario)
    return agent, scenario


def _context(scenario, actions=None):
    return {
        "scenario": scenario,
        "time_sec": 2.0,
        "observation": {
            "STATE_NAME": "RUNNING",
            "RW_Tank_Level": 1250,
            "ChemTreat_NaOCl_Level": 800,
        },
        "observations": [],
        "executed_actions": list(actions or []),
        "decisions": [],
        "alerts": [],
    }


def _record(action, result, *, time_sec=0.5):
    return {
        "role": "attacker",
        "time_sec": time_sec,
        "action": action,
        "result": result,
    }


def test_independent_recon_config_is_two_scenario_silent_pilot() -> None:
    config = load_config(CONFIG_NAME)
    selected = ScenarioCatalog(testbed_id="water_treatment").resolve_selectors(
        list(config["scenario_selectors"])
    )
    orchestrator = ExperimentOrchestrator(testbed_id="water_treatment")
    condition = orchestrator._enumerate_conditions(config)[0]
    effective = condition["effective_profiles"]

    assert [scenario.scenario_id for scenario in selected] == [
        "wt-attack-sensor-rw-hh-alarm",
        "wt-attack-p2-naocl-drain",
    ]
    assert config["reproducibility"]["seed"] == 101
    assert condition["profiles"]["attacker"] == "independent_recon_attacker"
    assert condition["profiles"]["defender"] == "silent_defender"
    assert effective["defender"]["allowed_action_types"] == ["noop"]
    assert effective["attacker"]["independent_recon_tools"] is True
    assert effective["attacker"]["allowed_action_types"] == [
        "discover_hosts",
        "enumerate_services",
        "probe_protocol",
        "enumerate_protocol_objects",
        "write_protocol_object",
    ]


def test_independent_recon_context_does_not_leak_target_semantics() -> None:
    agent, scenario = _agent_and_scenario()

    actor_context = agent._build_online_context(_context(scenario))
    rendered = json.dumps(actor_context, sort_keys=True)

    assert actor_context["scenario_id"] == "independent-recon-live"
    assert actor_context["system_knowledge"] is None
    assert actor_context["safety_bounds"] == {}
    assert actor_context["active_alarms"] == []
    assert "wt-attack-sensor-rw-hh-alarm" not in rendered
    assert "RW_Tank_Level" not in rendered
    assert "ChemTreat_NaOCl" not in rendered
    assert "1250" not in rendered
    assert "high-high" not in rendered.lower()


def test_independent_recon_candidates_progress_from_discovery_to_raw_write() -> None:
    agent, scenario = _agent_and_scenario()
    context = _context(scenario)

    initial = agent._independent_recon_candidates(context)
    assert [item["action"]["type"] for item in initial] == ["discover_hosts"]
    assert [item["id"] for item in initial] == ["tool_01"]

    actions = [
        _record(
            {"type": "discover_hosts"},
            {"hosts": [{"id": "host_01"}, {"id": "host_02"}]},
        ),
        _record(
            {"type": "enumerate_services", "host_id": "host_01"},
            {"services": [{"id": "host_01:svc_01"}]},
        ),
        _record(
            {"type": "enumerate_services", "host_id": "host_02"},
            {"services": [{"id": "host_02:svc_01"}]},
        ),
        _record(
            {"type": "probe_protocol", "service_id": "host_01:svc_01"},
            {"service_id": "host_01:svc_01", "protocol": "modbus_tcp"},
        ),
        _record(
            {"type": "probe_protocol", "service_id": "host_02:svc_01"},
            {"service_id": "host_02:svc_01", "protocol": "modbus_tcp"},
        ),
    ]
    for host_id in ("host_01", "host_02"):
        for object_space in ("coils", "holding_registers"):
            actions.append(
                _record(
                    {
                        "type": "enumerate_protocol_objects",
                        "host_id": host_id,
                        "object_space": object_space,
                        "start": 0,
                        "count": 128 if object_space == "coils" else 400,
                    },
                    {
                        "host_id": host_id,
                        "object_space": object_space,
                        "start": 0,
                        "count": 128 if object_space == "coils" else 400,
                        "nonzero_objects": (
                            [{"id": f"{host_id}:{object_space}:300", "value": 600}]
                            if object_space == "holding_registers"
                            else []
                        ),
                        "zero_object_ids": [f"{host_id}:{object_space}:0"],
                    },
                )
            )

    write_candidates = agent._independent_recon_candidates(
        _context(scenario, actions)
    )
    writes = [
        item for item in write_candidates
        if item["action"]["type"] == "write_protocol_object"
    ]
    assert writes
    assert all(item["id"].startswith("tool_") for item in write_candidates)
    assert all("{" not in item["id"] for item in write_candidates)
    discovered_ids = {
        "host_01:holding_registers:300",
        "host_02:holding_registers:300",
        "host_01:holding_registers:0",
        "host_02:holding_registers:0",
        "host_01:coils:0",
        "host_02:coils:0",
    }
    assert all(item["action"]["object_id"] in discovered_ids for item in writes)
    rendered = json.dumps(write_candidates)
    assert "RW_Tank_Level" not in rendered
    assert "NaOCl" not in rendered


class _Response:
    def __init__(self, *, bits=None, registers=None):
        self.bits = bits
        self.registers = registers

    @staticmethod
    def isError() -> bool:
        return False


class _Client:
    def __init__(self):
        self.coils = {0: 0, 1: 1}
        self.registers = {0: 0, 1: 600}

    def read_coils(self, *, address, count):
        return _Response(bits=[self.coils.get(index, 0) for index in range(address, address + count)])

    def read_holding_registers(self, *, address, count):
        return _Response(
            registers=[self.registers.get(index, 0) for index in range(address, address + count)]
        )

    def write_coil(self, *, address, value):
        self.coils[address] = int(bool(value))
        return _Response(bits=[])

    def write_register(self, *, address, value):
        self.registers[address] = int(value)
        return _Response(registers=[])


class _Testbed:
    def __init__(self):
        self.controller = _Client()
        self.simulator = _Client()

    @contextmanager
    def clients(self):
        yield self.controller, self.simulator

    def observe_flat(self):
        return {"STATE_NAME": "RUNNING"}


def test_raw_write_requires_prior_object_enumeration() -> None:
    env = object.__new__(OpenPLCEnvironment)
    env.testbed = _Testbed()
    env._backend_impl = None
    env.config = {}
    env._testbed_error = RuntimeError
    env._independent_recon_discovered_objects = set()

    with pytest.raises(EnvironmentError, match="not enumerated"):
        env._independent_recon_action(
            {
                "type": "write_protocol_object",
                "object_id": "host_02:holding_registers:1",
                "value": 1200,
            }
        )

    scan = env._independent_recon_action(
        {
            "type": "enumerate_protocol_objects",
            "host_id": "host_02",
            "object_space": "holding_registers",
            "start": 0,
            "count": 2,
        }
    )
    assert "host_02:holding_registers:1" in {
        item["id"] for item in scan["reconnaissance"]["nonzero_objects"]
    }

    outcome = env._independent_recon_action(
        {
            "type": "write_protocol_object",
            "object_id": "host_02:holding_registers:1",
            "value": 1200,
        }
    )
    assert outcome["reconnaissance"]["before"] == 600
    assert outcome["reconnaissance"]["after"] == 1200


def test_independent_recon_runner_script_exists() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / SCRIPT_NAME
    assert script_path.exists()
    assert "CONFIG_PATH" in script_path.read_text()
