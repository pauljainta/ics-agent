from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import pytest

from ics_agent import environment as environment_module
from ics_agent.environment import EnvironmentError, OpenPLCEnvironment


def _fresh_env(fake_openplc) -> OpenPLCEnvironment:
    env = OpenPLCEnvironment({"backend": "auto", "start_bridge": False})
    env.start()
    env.reset()
    return env


def test_block_sensor_write_blocks_later_attacker_sensor_writes(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    containment = env.act({"type": "block_sensor_write"}, role="defender")
    blocked = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 123},
        role="attacker",
    )

    assert containment["containment"]["execution_mode"] == "runtime_enforced"
    assert containment["runtime_containment"]["target_surfaces"] == ["sensor_write_path"]
    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "sensor_write_path"
    assert env.testbed.actions == []


def test_quarantine_signal_source_strengthens_same_sensor_surface(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_sensor_write"}, role="defender")
    containment = env.act({"type": "quarantine_signal_source"}, role="defender")
    blocked = env.act(
        {"type": "set_sensor", "name": "UF_UFFT_Tank_Level", "value": 999},
        role="attacker",
    )

    rule = containment["runtime_containment"]["rules"]["sensor_write_path"]
    assert rule["source_action_type"] == "quarantine_signal_source"
    assert rule["strength"] == 2
    assert blocked["blocked_by_action_type"] == "quarantine_signal_source"


def test_disable_hmi_session_blocks_later_attacker_hmi_commands(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "disable_hmi_session"}, role="defender")
    blocked = env.act({"type": "hmi_stop", "process_id": "WT-P1"}, role="attacker")

    assert blocked["blocked_by_runtime_containment"] is True
    assert blocked["blocked_surface"] == "hmi_session"
    assert env.testbed.actions == []


def test_block_process2_manual_path_blocks_process2_select_and_valve_actions(
    fake_openplc,
) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_process2_manual_path"}, role="defender")
    blocked_select = env.act(
        {"type": "select_process", "process_id": "WT-P2", "active": True},
        role="attacker",
    )
    blocked_valve = env.act(
        {
            "type": "set_manual_valve",
            "process_id": "WT-P2",
            "chemical": "NaCl",
            "open": True,
        },
        role="attacker",
    )

    assert blocked_select["blocked_surface"] == "process2_manual_path"
    assert blocked_valve["blocked_surface"] == "process2_manual_path"
    assert env.testbed.actions == []


def test_isolate_engineering_host_blocks_all_later_attacker_actions(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "isolate_engineering_host"}, role="defender")
    blocked_recon = env.act(
        {"type": "inspect_surface", "surface": "tags"},
        role="attacker",
    )
    blocked_control = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 101},
        role="attacker",
    )

    assert blocked_recon["blocked_surface"] == "all"
    assert blocked_control["blocked_surface"] == "all"


def test_shutdown_actions_still_flow_to_concrete_plant_path(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    stop_result = env.act({"type": "hmi_stop", "process_id": "WT-P1"}, role="defender")
    estop_result = env.act({"type": "hmi_estop", "process_id": "WT-P1"}, role="defender")

    assert stop_result["state_name"] == "STOPPED"
    assert estop_result["state_name"] == "ESTOPPED"
    assert [item["type"] for item in env.testbed.actions] == ["hmi_stop", "hmi_estop"]


def test_runtime_containment_state_resets_between_episodes(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)

    env.act({"type": "block_sensor_write"}, role="defender")
    env.reset()
    allowed = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 456},
        role="attacker",
    )

    assert allowed.get("blocked_by_runtime_containment") is not True
    assert env.runtime_containment_snapshot()["active"] is False
    assert env.testbed.actions[-1]["type"] == "set_sensor"


def test_delayed_runtime_containment_activates_after_two_steps(fake_openplc) -> None:
    env = _fresh_env(fake_openplc)
    env.set_runtime_containment_delay_steps(2)

    scheduled = env.act({"type": "block_sensor_write"}, role="defender")
    assert scheduled["runtime_containment"]["active"] is False
    assert scheduled["runtime_containment"]["pending"] is True
    assert scheduled["runtime_containment"]["pending_target_surfaces"] == [
        "sensor_write_path"
    ]

    env.advance_runtime_containment_step(1)
    allowed_during_delay = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 777},
        role="attacker",
    )
    assert allowed_during_delay.get("blocked_by_runtime_containment") is not True

    activations = env.advance_runtime_containment_step(2)
    assert len(activations) == 1
    assert activations[0]["activation_delay_steps"] == 2
    assert activations[0]["activated_at_step"] == 2
    blocked_after_activation = env.act(
        {"type": "set_sensor", "name": "RW_Tank_Level", "value": 778},
        role="attacker",
    )
    assert blocked_after_activation["blocked_by_runtime_containment"] is True


def test_sphere_backend_delay_configuration_does_not_force_transport_start() -> None:
    backend = environment_module._SphereOpenPLCBackend(
        {
            "testbed": "water_treatment",
            "backend": "sphere_openplc",
            "start_bridge": False,
            "experiment": "openplcwater.lgarcia",
            "realization": "example-realization-03",
            "revision": "test-revision",
            "xdc": "example-xdc-07",
        }
    )
    start_calls = 0

    def _unexpected_start(self):
        nonlocal start_calls
        start_calls += 1
        raise AssertionError("delay configuration should not start the sphere transport")

    backend.start = MethodType(_unexpected_start, backend)

    backend.set_runtime_containment_delay_steps(2)

    assert start_calls == 0
    assert backend._runtime_containment_delay_steps == 2


def test_vendor_openplc_payload_imports_without_workspace_repos(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(environment_module, "WORKSPACE_ROOT", tmp_path)
    vendor_scripts = environment_module.OPENPLC_SUPPORT_SCRIPTS.resolve()

    for module_name in ("modbus_raw", "native_plc_runtime", "openplc_testbed"):
        sys.modules.pop(module_name, None)

    imported = {
        module_name: environment_module._import_openplc_module(module_name)
        for module_name in ("modbus_raw", "native_plc_runtime", "openplc_testbed")
    }

    for module in imported.values():
        module_path = Path(module.__file__).resolve()
        assert vendor_scripts in module_path.parents


class _FakeDistributionTestbed:
    def __init__(self, reset_snapshot: dict[str, object]) -> None:
        self.reset_snapshot = dict(reset_snapshot)
        self.clear_calls = 0
        self.reset_calls: list[dict[str, object]] = []

    def _clear_external_controls(self) -> None:
        self.clear_calls += 1

    def reset(self, initial_conditions=None, *, wait_sec: float) -> dict[str, object]:
        self.reset_calls.append(
            {
                "initial_conditions": dict(initial_conditions or {}),
                "wait_sec": wait_sec,
            }
        )
        return dict(self.reset_snapshot)


class _FakeTreatmentTestbed:
    def __init__(self, reset_snapshot: dict[str, object]) -> None:
        self.reset_snapshot = dict(reset_snapshot)
        self.clear_calls = 0
        self.reset_calls: list[dict[str, object]] = []
        self.set_levels_calls: list[dict[str, object]] = []

    def _backend_mode(self) -> str:
        return "native"

    def _clear_external_controls(self) -> None:
        self.clear_calls += 1

    def reset(self, initial_conditions=None, *, wait_sec: float) -> dict[str, object]:
        self.reset_calls.append(
            {
                "initial_conditions": dict(initial_conditions or {}),
                "wait_sec": wait_sec,
            }
        )
        return dict(self.reset_snapshot)

    def set_levels(self, updates, *, wait_sec: float) -> dict[str, object]:
        self.set_levels_calls.append(
            {
                "initial_conditions": dict(updates or {}),
                "wait_sec": wait_sec,
            }
        )
        return dict(self.reset_snapshot)


class _FakeDistributionDirectEnv:
    def __init__(
        self,
        *,
        testbed: _FakeDistributionTestbed,
        observations: list[dict[str, object]],
    ) -> None:
        self.testbed = testbed
        self._observations = [dict(item) for item in observations]
        self.clear_runtime_containment_calls = 0

    def clear_runtime_containment(self) -> None:
        self.clear_runtime_containment_calls += 1

    def observe_flat(self) -> dict[str, object]:
        if len(self._observations) > 1:
            return self._observations.pop(0)
        return dict(self._observations[0])


def _distribution_backend(
    fake_direct_env: _FakeDistributionDirectEnv,
) -> environment_module._SphereOpenPLCBackend:
    backend = environment_module._SphereOpenPLCBackend(
        {
            "testbed": "water_distribution",
            "backend": "sphere_openplc",
            "start_bridge": False,
            "remote_reset_mode": "soft_reseed",
            "soft_reset_level_wait_sec": 0.0,
            "reset_wait_sec": 0.0,
            "initial_state_validation_timeout_sec": 0.0,
        }
    )
    backend._require_direct_env = MethodType(lambda self: fake_direct_env, backend)
    return backend


def _treatment_backend(
    fake_direct_env: _FakeDistributionDirectEnv,
) -> environment_module._SphereOpenPLCBackend:
    backend = environment_module._SphereOpenPLCBackend(
        {
            "backend": "sphere_openplc",
            "start_bridge": False,
            "remote_reset_mode": "soft_reseed",
            "soft_reset_level_wait_sec": 0.25,
            "reset_wait_sec": 0.0,
        }
    )
    backend._require_direct_env = MethodType(lambda self: fake_direct_env, backend)
    backend._local_port_listening = MethodType(lambda self, port: True, backend)
    return backend


def test_distribution_backend_reset_uses_testbed_reset_and_records_validation() -> None:
    expected = {
        "Supply_Tank_Level": 920.0,
        "Grid_Elev_Tank_Level": 940.0,
        "RWS_Tank_Level": 500.0,
    }
    reset_snapshot = {
        "STATE_NAME": "IDLE",
        "Supply_Tank_Level": 920.0,
        "Grid_Elev_Tank_Level": 940.0,
        "Grid_Consum_Tank_Level": 820.0,
        "RWS_Tank_Level": 500.0,
    }
    fake_testbed = _FakeDistributionTestbed(reset_snapshot)
    fake_direct_env = _FakeDistributionDirectEnv(
        testbed=fake_testbed,
        observations=[reset_snapshot],
    )
    backend = _distribution_backend(fake_direct_env)

    result = backend.reset({"Supply_Tank_Level": 920, "Grid_Elev_Tank_Level": 940})

    assert fake_direct_env.clear_runtime_containment_calls == 1
    assert fake_testbed.clear_calls == 2
    assert len(fake_testbed.reset_calls) == 1
    assert (
        fake_testbed.reset_calls[0]["initial_conditions"]["Supply_Tank_Level"] == 920
    )
    assert (
        fake_testbed.reset_calls[0]["initial_conditions"]["Grid_Elev_Tank_Level"] == 940
    )
    assert fake_testbed.reset_calls[0]["wait_sec"] == 0.0
    assert result["post_reset_observation"]["STATE_NAME"] == "IDLE"
    assert result["initial_state_validation"]["status"] == "passed"
    assert result["initial_state_validation"]["expected"] == expected
    assert result["initial_state_validation"]["observed"] == expected


def test_treatment_backend_soft_reset_uses_remote_level_reseed_when_native_mode() -> None:
    reset_snapshot = {
        "STATE_NAME": "RUNNING",
        "RW_Tank_Level": 600.0,
        "UF_UFFT_Tank_Level": 400.0,
        "ChemTreat_NaCl_Level": 800.0,
    }
    fake_testbed = _FakeTreatmentTestbed(reset_snapshot)
    fake_direct_env = _FakeDistributionDirectEnv(
        testbed=fake_testbed,
        observations=[reset_snapshot],
    )
    backend = _treatment_backend(fake_direct_env)

    result = backend.reset({"RW_Tank_Level": 600, "UF_UFFT_Tank_Level": 400})

    assert fake_direct_env.clear_runtime_containment_calls == 1
    assert fake_testbed.clear_calls == 2
    assert fake_testbed.reset_calls == []
    assert len(fake_testbed.set_levels_calls) == 1
    assert fake_testbed.set_levels_calls[0]["initial_conditions"] == {
        "RW_Tank_Level": 600,
        "UF_UFFT_Tank_Level": 400,
        "ChemTreat_NaCl_Level": 800,
        "ChemTreat_NaOCl_Level": 800,
        "ChemTreat_HCl_Level": 800,
    }
    assert fake_testbed.set_levels_calls[0]["wait_sec"] == 0.25
    assert result["post_reset_observation"]["STATE_NAME"] == "RUNNING"
    assert "initial_state_validation" not in result


def test_distribution_backend_reset_fails_loudly_on_initial_state_mismatch() -> None:
    reset_snapshot = {
        "STATE_NAME": "IDLE",
        "Supply_Tank_Level": 8180.0,
        "Grid_Elev_Tank_Level": 2200.0,
        "Grid_Consum_Tank_Level": 920.0,
        "RWS_Tank_Level": 1650.0,
    }
    fake_testbed = _FakeDistributionTestbed(reset_snapshot)
    fake_direct_env = _FakeDistributionDirectEnv(
        testbed=fake_testbed,
        observations=[reset_snapshot],
    )
    backend = _distribution_backend(fake_direct_env)

    with pytest.raises(EnvironmentError) as exc_info:
        backend.reset(
            {
                "Supply_Tank_Level": 920,
                "Grid_Elev_Tank_Level": 940,
                "RWS_Tank_Level": 500,
            }
        )

    message = str(exc_info.value)
    assert "initial-state validation failed" in message
    assert "Supply_Tank_Level: expected 920.0, observed 8180.0" in message
    assert "Grid_Elev_Tank_Level: expected 940.0, observed 2200.0" in message
    assert "RWS_Tank_Level: expected 500.0, observed 1650.0" in message
