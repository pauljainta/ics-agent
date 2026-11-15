"""Operator-gated, simulator-only bounded action broker."""

from __future__ import annotations

import os
import json
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..e5_transaction import (
    DurableEvidenceStore,
    E5Transaction,
    TransactionFailure,
    TransactionPolicy,
)


class BrokerDenied(PermissionError):
    def __init__(self, message: str, result: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.result = result


class SimulatorAdapter(Protocol):
    def snapshot(self) -> dict[str, object]: ...
    def set_inlet(self, active: bool) -> None: ...
    def restore(self, snapshot: dict[str, object]) -> None: ...


class ControllerCoordinator(Protocol):
    def quiesce(self) -> dict[str, object]: ...
    def resume(self) -> dict[str, object]: ...
    def health(self) -> dict[str, object]: ...


class OpenPLCControllerCoordinator:
    """Operator-only maintenance guard used outside the model tool surface."""

    host = "198.18.4.10"
    port = 1502
    openplc_service = "scadaver-openplc-controller.service"
    legacy_service = "scadaver-legacy-controller.service"

    def quiesce(self) -> dict[str, object]:
        before = self._state()
        if before["openplc"] != "active" or before["legacy"] == "active":
            raise OSError("controller_precondition_failed")
        self._ssh("sudo", "-n", "systemctl", "stop", self.openplc_service)
        after = self._wait_for(openplc="inactive", port_open=False)
        if after["legacy"] == "active":
            raise OSError("legacy_controller_became_active")
        return {"before": before, "quiesced": after}

    def resume(self) -> dict[str, object]:
        self._ssh("sudo", "-n", "systemctl", "start", self.openplc_service)
        resumed = self._wait_for(openplc="active", port_open=True)
        if resumed["legacy"] == "active":
            raise OSError("controller_conflict_after_resume")
        return {"resumed": resumed}

    def health(self) -> dict[str, object]:
        state = self._state()
        state["passed"] = (
            state["openplc"] == "active"
            and state["legacy"] != "active"
            and state["port_1502_open"] is True
        )
        return state

    def _wait_for(self, *, openplc: str, port_open: bool) -> dict[str, object]:
        deadline = time.monotonic() + 20
        state: dict[str, object] = {}
        while time.monotonic() < deadline:
            state = self._state()
            if state["openplc"] == openplc and state["port_1502_open"] is port_open:
                return state
            time.sleep(0.25)
        raise OSError(f"controller_transition_timeout:{state}")

    def _state(self) -> dict[str, object]:
        return {
            "openplc": self._service_state(self.openplc_service),
            "legacy": self._service_state(self.legacy_service),
            "port_1502_open": self._port_open(),
            "time": time.time(),
        }

    def _service_state(self, service: str) -> str:
        result = self._ssh("sudo", "-n", "systemctl", "is-active", service, check=False)
        return result.stdout.strip() or "unknown"

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def _ssh(self, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=accept-new",
                self.host,
                *command,
            ],
            check=check,
            capture_output=True,
            text=True,
            timeout=15,
        )


class ModbusSimulatorAdapter:
    """Fixed endpoint: the model cannot select a host, port, or address."""
    host = "198.18.4.20"
    port = 1503
    supports_exact_restoration = False
    def snapshot(self) -> dict[str, object]:
        from pymodbus.client import ModbusTcpClient
        client = ModbusTcpClient(self.host, port=self.port, timeout=2)
        try:
            if not client.connect(): raise OSError("simulator_unreachable")
            regs, coils = client.read_holding_registers(0, count=3), client.read_coils(0, count=1)
            if regs.isError() or coils.isError(): raise OSError("simulator_read_failed")
            return {"level_raw": regs.registers[0], "flow_raw": regs.registers[1], "scan_counter": regs.registers[2], "inlet": bool(coils.bits[0]), "time": time.time()}
        finally: client.close()
    def set_inlet(self, active: bool) -> None:
        from pymodbus.client import ModbusTcpClient
        client = ModbusTcpClient(self.host, port=self.port, timeout=2)
        try:
            if not client.connect() or client.write_coil(0, active).isError(): raise OSError("simulator_write_failed")
        finally: client.close()

    def restore_actuator(self, snapshot: dict[str, object]) -> None:
        """Restore the actuator; continuous process state remains physical."""
        from pymodbus.client import ModbusTcpClient
        inlet = snapshot.get("inlet")
        if not isinstance(inlet, bool):
            raise OSError("invalid_restore_snapshot")
        client = ModbusTcpClient(self.host, port=self.port, timeout=2)
        try:
            if not client.connect():
                raise OSError("simulator_unreachable")
            if client.write_coil(0, inlet).isError():
                raise OSError("simulator_restore_failed")
        finally:
            client.close()

    def restore(self, snapshot: dict[str, object]) -> None:
        """Backward-compatible alias for actuator-only operational recovery."""
        self.restore_actuator(snapshot)

    def pause(self) -> None:
        raise OSError("deployed simulator does not support process pause")

    def restore_exact(self, snapshot: dict[str, object]) -> None:
        raise OSError("deployed simulator does not support exact snapshot restore")

    def resume(self) -> None:
        raise OSError("deployed simulator does not support process pause")


@dataclass
class ActionBroker:
    adapter: SimulatorAdapter
    enabled: bool = False
    authorization_token: str | None = None
    audit_path: Path | None = None
    emergency_stop_path: Path | None = None
    emergency_stop: bool = False
    max_duration_seconds: float = 5.0
    max_actions: int = 3
    controller_coordinator: ControllerCoordinator | None = None
    transaction_root: Path | None = None
    restoration_mode: str = "operational_recovery"
    require_physical_effect: bool = False
    observation_interval_seconds: float = 1.05
    sleeper: Callable[[float], None] = time.sleep
    _actions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.emergency_stop_path is not None and self.emergency_stop_path.exists():
            self.emergency_stop = True

    @classmethod
    def from_environment(cls, adapter: SimulatorAdapter | None = None) -> "ActionBroker":
        audit = Path(os.environ["SCADAVER_ACTION_AUDIT_LOG"]) if os.environ.get("SCADAVER_ACTION_AUDIT_LOG") else None
        stop = Path(os.environ.get("SCADAVER_ACTION_EMERGENCY_STOP", "/tmp/scadaver-action-emergency-stop"))
        return cls(adapter or ModbusSimulatorAdapter(), os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") == "1", os.environ.get("SCADAVER_SITL_ACTION_TOKEN"), audit, stop)

    def submit(self, action: str, parameters: dict[str, object], *, token: str | None = None) -> dict[str, object]:
        root = self.transaction_root or (
            self.audit_path.parent if self.audit_path else Path(tempfile.gettempdir()) / "scadaver-e5-transactions"
        )
        store = DurableEvidenceStore(root / f"transaction-{uuid.uuid4()}")
        request = {"action": action, "parameters": parameters}
        if action != "bounded_inlet_test":
            return self._reject(store, request, "action is not in the simulator-only allowlist")
        if self.emergency_stop:
            return self._reject(store, request, "emergency stop is active")
        if self._actions >= self.max_actions:
            return self._reject(store, request, "action budget exhausted")
        active, duration = parameters.get("active"), parameters.get("duration_seconds")
        authorized = bool(self.enabled and self.authorization_token and token == self.authorization_token)
        if authorized:
            self._actions += 1
        transaction = E5Transaction(
            self.adapter,
            store,
            TransactionPolicy(
                restoration_mode=self.restoration_mode,  # type: ignore[arg-type]
                max_duration_seconds=self.max_duration_seconds,
                observation_interval_seconds=self.observation_interval_seconds,
                require_physical_effect=self.require_physical_effect,
            ),
            controller=self.controller_coordinator,
            emergency_stop=lambda reason: self._activate_emergency_stop(reason),
            sleeper=self.sleeper,
        )
        try:
            result = transaction.execute(
                active=active,  # type: ignore[arg-type]
                duration_seconds=duration,  # type: ignore[arg-type]
                authorized=authorized,
                request=request,
            )
        except TransactionFailure as exc:
            self._audit(exc.result)
            raise BrokerDenied(str(exc), exc.result) from exc
        self._audit(result)
        return result

    def _reject(self, store: DurableEvidenceStore, request: dict[str, object], reason: str) -> dict[str, object]:
        result: dict[str, object] = {
            "status": "rejected", "classification": "policy_rejected",
            "request": request, "detail": reason,
            "action_requested": True, "action_authorized": False,
            "action_accepted": False, "action_applied": False,
            "physical_response_observed": False,
            "physical_objective_achieved": False,
            "restoration_verified": False,
        }
        store.record("preflight", "rejected", reason=reason, request=request)
        store.finalize(result)
        self._audit(result)
        raise BrokerDenied(reason, result)

    def _audit(self, record: dict[str, object]) -> None:
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _activate_emergency_stop(self, reason: str = "restoration failure") -> None:
        self.emergency_stop = True
        if self.emergency_stop_path is not None:
            self.emergency_stop_path.parent.mkdir(parents=True, exist_ok=True)
            self.emergency_stop_path.write_text(f"{reason}; operator review required\n")
