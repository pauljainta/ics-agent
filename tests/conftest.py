from __future__ import annotations

import types

import pytest

from ics_agent import environment as environment_module


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
