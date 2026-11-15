"""Controlled evolving process and controller fixtures for full E5 transactions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FixtureFaults:
    reject_action: bool = False
    action_timeout: bool = False
    interrupt_after_apply: bool = False
    no_physical_effect: bool = False
    restoration_failure: bool = False
    controller_overwrite_on_restore: bool = False
    controller_restart_failure: bool = False


class ControlledProcessFixture:
    supports_exact_restoration = True

    def __init__(self, faults: FixtureFaults | None = None, *, scan_seconds: float = 1.0) -> None:
        self.faults = faults or FixtureFaults()
        self.scan_seconds = scan_seconds
        self.level = 500
        self.inlet = False
        self.flow = 0
        self.scan_counter = 0
        self.paused = False
        self.action_calls = 0
        self.restore_calls = 0

    def snapshot(self) -> dict[str, object]:
        if self.faults.interrupt_after_apply and self.action_calls and not self.restore_calls:
            self.faults.interrupt_after_apply = False
            raise ConnectionError("fixture worker interrupted")
        return {
            "inlet": self.inlet,
            "level_raw": self.level,
            "flow_raw": self.flow,
            "scan_counter": self.scan_counter,
            "time": self.scan_counter * self.scan_seconds,
        }

    def set_inlet(self, active: bool) -> None:
        self.action_calls += 1
        if self.faults.reject_action:
            raise PermissionError("fixture rejected action")
        if self.faults.action_timeout:
            raise TimeoutError("fixture action timeout")
        self.inlet = active
        if not self.faults.no_physical_effect:
            self.flow = 120 if active else 0

    def restore_actuator(self, snapshot: dict[str, object]) -> None:
        self.restore_calls += 1
        if self.faults.restoration_failure:
            raise OSError("fixture actuator restoration failed")
        self.inlet = bool(snapshot["inlet"])
        self.flow = int(snapshot["flow_raw"])
        if self.faults.controller_overwrite_on_restore:
            self.inlet = not self.inlet
            self.flow = 120 if self.inlet else 0

    def pause(self) -> None:
        self.paused = True

    def restore_exact(self, snapshot: dict[str, object]) -> None:
        self.restore_calls += 1
        if self.faults.restoration_failure:
            raise OSError("fixture exact restoration failed")
        self.inlet = bool(snapshot["inlet"])
        self.level = int(snapshot["level_raw"])
        self.flow = int(snapshot["flow_raw"])
        self.scan_counter = int(snapshot["scan_counter"])

    def resume(self) -> None:
        self.paused = False

    def advance_time(self, seconds: float) -> None:
        scans = max(1, int(math.ceil(seconds / self.scan_seconds))) if seconds else 0
        for _ in range(scans):
            self.advance_scan()

    def advance_scan(self) -> None:
        if self.paused:
            return
        self.scan_counter += 1
        if not self.faults.no_physical_effect:
            self.level = max(0, min(1000, self.level + (12 if self.inlet else -7)))
            self.flow = 120 if self.inlet else 0


class FixtureController:
    def __init__(self, process: ControlledProcessFixture) -> None:
        self.process = process
        self.active = True
        self.legacy_active = False
        self.quiesce_calls = 0
        self.resume_calls = 0

    def quiesce(self) -> dict[str, object]:
        if not self.active or self.legacy_active:
            raise OSError("fixture controller precondition failed")
        self.quiesce_calls += 1
        self.active = False
        return {"before": {"active": True}, "quiesced": {"active": False}}

    def resume(self) -> dict[str, object]:
        self.resume_calls += 1
        if self.process.faults.controller_restart_failure:
            raise OSError("fixture controller restart failed")
        self.active = True
        return {"resumed": {"active": True}}

    def health(self) -> dict[str, object]:
        return {"passed": self.active and not self.legacy_active,
                "active": self.active, "legacy_active": self.legacy_active}
