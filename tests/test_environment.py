from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ics_agent.environment import EnvironmentError, OpenPLCEnvironment, _SphereOpenPLCBackend


def test_local_backend_mode_controls_openplc_backend(monkeypatch) -> None:
    class FakeTestbed:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def ensure_stack(self, **kwargs):
            return {
                "requested_backend": os.environ.get("OPENPLC_TESTBED_BACKEND"),
                "kwargs": kwargs,
            }

        def down(self):
            return None

    fake_module = SimpleNamespace(
        OpenPLCWaterTreatmentTestbed=FakeTestbed,
        TestbedError=RuntimeError,
    )

    def fake_import(module_name: str):
        if module_name == "openplc_testbed":
            return fake_module
        return object()

    monkeypatch.setattr("ics_agent.environment._import_openplc_module", fake_import)
    monkeypatch.setenv("OPENPLC_TESTBED_BACKEND", "native")

    env = OpenPLCEnvironment({"backend": "docker"})
    result = env.start()

    assert result["requested_backend"] == "docker"
    assert os.environ["OPENPLC_TESTBED_BACKEND"] == "native"


def test_sphere_backend_is_delegated(monkeypatch) -> None:
    class FakeSphereBackend:
        def __init__(self, config) -> None:
            self.config = config
            self.started = False

        def start(self):
            self.started = True
            return {"backend": "sphere_openplc"}

        def reset(self, initial_conditions=None):
            return {"reset": initial_conditions}

        def observe_flat(self):
            return {"STATE_NAME": "IDLE"}

        def observe(self):
            return {"state_name": "IDLE"}

        def act(self, action):
            return {"action": action}

        def inspect_surface(self, action):
            return {"surface": action["surface"]}

        def stop(self):
            self.started = False

    monkeypatch.setattr("ics_agent.environment._SphereOpenPLCBackend", FakeSphereBackend)

    env = OpenPLCEnvironment({"backend": "sphere_openplc"})

    assert env.start() == {"backend": "sphere_openplc"}
    assert env.reset({"RW_Tank_Level": 500}) == {"reset": {"RW_Tank_Level": 500}}
    assert env.observe_flat() == {"STATE_NAME": "IDLE"}
    assert env.observe() == {"state_name": "IDLE"}
    assert env.act({"type": "hmi_stop"}) == {"action": {"type": "hmi_stop"}}
    assert env.inspect_surface({"surface": "tags"}) == {"surface": "tags"}
    env.stop()
    assert env.started is False


def test_sphere_backend_reference_helpers() -> None:
    assert (
        _SphereOpenPLCBackend._qualify_realization_ref(
            "wargamestable", "openplcwater", "lgarcia"
        )
        == "wargamestable.openplcwater.lgarcia"
    )
    assert _SphereOpenPLCBackend._qualify_xdc_ref("x0", "researcher") == "x0.researcher"

    with pytest.raises(EnvironmentError):
        _SphereOpenPLCBackend._split_ref("badref")


def test_sphere_backend_ssh_checked_retries_transient_failures() -> None:
    backend = _SphereOpenPLCBackend(
        {
            "experiment": "openplcwater.lgarcia",
            "ssh_attempts": 3,
            "ssh_retry_backoff_sec": 0.0,
        }
    )
    backend.ssh_config_path = backend.helper_root / "test_ssh_config"

    calls: list[tuple[list[str], float | None]] = []
    responses = [
        EnvironmentError("transient ssh failure"),
        "ready\n",
    ]

    def fake_run_capture(command, *, timeout_sec=None):
        calls.append((command, timeout_sec))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    with patch.object(backend, "_run_capture", side_effect=fake_run_capture):
        result = backend._ssh_checked("simulator", "echo ready", retries=2)

    assert result == "ready\n"
    assert len(calls) == 2
    assert calls[0][0][-2:] == ["simulator", "echo ready"]


def test_sphere_backend_start_cleans_partial_local_transport(monkeypatch) -> None:
    backend = _SphereOpenPLCBackend({"experiment": "openplcwater.lgarcia"})
    stopped: list[str] = []

    monkeypatch.setattr(
        backend,
        "_prepare_transport",
        lambda: (_ for _ in ()).throw(EnvironmentError("boom")),
    )
    monkeypatch.setattr(backend, "_stop_local_process", lambda name: stopped.append(name))

    with pytest.raises(EnvironmentError, match="boom"):
        backend.start()

    assert stopped == [
        "bridge",
        "controller_tunnel",
        "simulator_tunnel",
        "attacker_tunnel",
        "defender_tunnel",
    ]
    assert backend.started is False
