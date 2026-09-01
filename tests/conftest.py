from __future__ import annotations

import types
from pathlib import Path

import pytest

from ics_agent import environment as environment_module


@pytest.fixture(autouse=True)
def synthetic_sphere_infrastructure(monkeypatch, tmp_path: Path) -> None:
    """Resolve tracked SPHERE experiment configs through synthetic test profiles."""

    monkeypatch.delenv("ICS_AGENT_SPHERE_XDC_OVERRIDE", raising=False)
    monkeypatch.delenv("ICS_AGENT_MERGE_SERVER_OVERRIDE", raising=False)
    monkeypatch.delenv("MRG_SERVER", raising=False)
    monkeypatch.delenv("MERGE_SERVER", raising=False)
    monkeypatch.delenv("ICS_AGENT_ALLOW_XDC_DISCOVERY", raising=False)

    config_path = tmp_path / "sphere.local.yaml"
    config_path.write_text(
        """
profiles:
  water_treatment:
    merge_server: merge-test-host
    experiment: treatment-experiment.test-project
    realization: treatment-realization
    revision: treatment-revision
    xdc: treatment-xdc
    keep_running: true
    start_bridge: true
    remote_bind_host: 0.0.0.0
    controller_port: 2502
    simulator_port: 2503
    remote_controller_port: 1502
    remote_simulator_port: 1503
    controller_process_ip: controller-test-host
    simulator_process_ip: simulator-test-host
    jump_host: jump-test-host
    jump_port: 2022
    access_host: access
    engineering_host: engineering
    remote_roles: {}
    remote_role_port: 18600
    remote_role_local_ports:
      attacker: 28601
      defender: 28602
    reset_wait_sec: 1.5
    loop_sleep_sec: 0.05
    bridge_host: controller
    bridge_cycle_ms: 100
    bridge_package_targets:
      - controller
    bridge_start_timeout_sec: 20.0
    tunnel_start_timeout_sec: 20.0
    allow_xdc_discovery: false
  water_distribution:
    merge_server: merge-test-host
    experiment: distribution-experiment.test-project
    realization: distribution-realization
    revision: distribution-revision
    xdc: distribution-xdc
    keep_running: true
    start_bridge: true
    remote_bind_host: 0.0.0.0
    controller_port: 2502
    simulator_port: 2503
    remote_controller_port: 1502
    remote_simulator_port: 1503
    controller_process_ip: controller-test-host
    simulator_process_ip: simulator-test-host
    jump_host: jump-test-host
    jump_port: 2022
    access_host: access
    engineering_host: engineering
    remote_roles: {}
    remote_role_port: 18600
    remote_role_local_ports:
      attacker: 28601
      defender: 28602
    reset_wait_sec: 1.5
    loop_sleep_sec: 0.05
    bridge_host: controller
    bridge_cycle_ms: 100
    bridge_package_targets:
      - controller
    bridge_start_timeout_sec: 20.0
    tunnel_start_timeout_sec: 20.0
    allow_xdc_discovery: false
""".lstrip()
    )
    monkeypatch.setenv("ICS_AGENT_SPHERE_CONFIG", str(config_path))


class FakeTestbedError(RuntimeError):
    pass


class FakeOpenPLCWaterTreatmentTestbed:
    def __init__(
        self,
        *,
        base_dir,
        controller_host,
        controller_port,
        simulator_host,
        simulator_port,
        unit_id,
    ) -> None:
        self.base_dir = base_dir
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.simulator_host = simulator_host
        self.simulator_port = simulator_port
        self.unit_id = unit_id
        self.actions: list[dict[str, object]] = []
        self.state_name = "RUNNING"
        self.p2_selected = 1
        self.levels = {
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
            "ChemTreat_NaCl_Level": 800,
            "ChemTreat_NaOCl_Level": 800,
            "ChemTreat_HCl_Level": 800,
        }

    def ensure_stack(self, *, build_if_missing, timeout_sec, start_bridge):
        return {"started": True}

    def reset(self, initial_conditions=None, *, wait_sec):
        self.actions.clear()
        self.state_name = "RUNNING"
        self.p2_selected = 1
        if initial_conditions:
            self.levels.update(initial_conditions)
        return self.observe_flat()

    def observe_flat(self):
        return {
            "STATE_NAME": self.state_name,
            "P2_Selected": self.p2_selected,
            **self.levels,
        }

    def observe(self):
        return self.observe_flat()

    def inspect_surface(self, surface):
        flat = self.observe_flat()
        return {
            "state_name": flat["STATE_NAME"],
            "reconnaissance": {
                "category": "reconnaissance",
                "surface": str(surface),
                "summary": f"fake surface `{surface}` exposes bounded water-treatment readback",
                "findings": [
                    {"name": "RW_Tank_Level", "value": flat["RW_Tank_Level"]},
                    {"name": "UF_UFFT_Tank_Level", "value": flat["UF_UFFT_Tank_Level"]},
                    {"name": "P2_Selected", "value": flat["P2_Selected"]},
                ],
            },
        }

    def apply_action(self, action):
        descriptor = dict(action)
        self.actions.append(descriptor)
        action_type = str(descriptor.get("type", ""))
        if action_type == "hmi_stop":
            self.state_name = "STOPPED"
        elif action_type == "hmi_estop":
            self.state_name = "ESTOPPED"
        elif action_type == "select_process":
            self.p2_selected = int(bool(descriptor.get("active", True)))
        elif action_type == "set_sensor":
            self.levels[str(descriptor.get("name"))] = int(descriptor.get("value", 0))
        return {"state_name": self.state_name}


@pytest.fixture
def fake_openplc(monkeypatch):
    openplc_module = types.SimpleNamespace(
        OpenPLCWaterTreatmentTestbed=FakeOpenPLCWaterTreatmentTestbed,
        TestbedError=FakeTestbedError,
        FIELD_GROUPS={},
        state_name=lambda sample: "RUNNING",
    )

    def fake_import(module_name: str):
        if module_name in {"modbus_raw", "native_plc_runtime"}:
            return types.SimpleNamespace()
        if module_name == "openplc_testbed":
            return openplc_module
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(environment_module, "_import_openplc_module", fake_import)
    return openplc_module
