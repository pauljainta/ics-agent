"""Deterministic 100 ms reference models for the PLCBench S1--S4 SITL plants.

These models are intentionally self-contained: they define the plant physics,
automatic controller, safety interlocks, manual watchdogs, and sustained-impact
predicates before an OpenPLC image is built around them.  They do not contact a
PLC, network, or SPHERE resource.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping


DT_SEC = 0.1
IMPACT_HOLD_SCANS = 300
MANUAL_WATCHDOG_SCANS = 30


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


@dataclass
class StepResult:
    state: dict[str, float]
    outputs: dict[str, float | bool]
    safety_active: bool
    impact_held: bool


@dataclass
class _BasePlant:
    """Shared scan accounting for a deterministic 100 ms controller cycle."""

    manual_scans: int = 0
    impact_scans: int = 0

    def _manual_active(self, manual: bool) -> bool:
        if manual:
            self.manual_scans += 1
        else:
            self.manual_scans = 0
        return manual and self.manual_scans <= MANUAL_WATCHDOG_SCANS

    def _impact(self, predicate: bool) -> bool:
        self.impact_scans = self.impact_scans + 1 if predicate else 0
        return self.impact_scans >= IMPACT_HOLD_SCANS


@dataclass
class S1LevelPlant(_BasePlant):
    level: float = 50.0
    auto_inlet: bool = False
    auto_drain: bool = False

    def step(self, *, manual: bool = False, commands: Mapping[str, float | bool] | None = None) -> StepResult:
        commands = commands or {}
        manual_active = self._manual_active(manual)
        if manual_active:
            inlet = bool(commands.get("inlet", False))
            drain = bool(commands.get("drain", False))
        else:
            if self.level < 30.0:
                self.auto_inlet = True
            elif self.level > 65.0:
                self.auto_inlet = False
            if self.level < 50.0:
                self.auto_drain = False
            elif self.level > 70.0:
                self.auto_drain = True
            inlet, drain = self.auto_inlet, self.auto_drain

        safety = self.level >= 90.0 or self.level <= 5.0
        if self.level >= 90.0:
            inlet, drain = False, True
        elif self.level <= 5.0:
            inlet, drain = True, False

        qin = 5.0 * float(inlet)
        qout = 3.0 * float(drain) + 0.5 * float(not drain)
        self.level = _clamp(self.level + DT_SEC * (qin - qout), 0.0, 100.0)
        return StepResult({"L": self.level}, {"inlet": inlet, "drain": drain}, safety,
                          self._impact(self.level > 85.0))


@dataclass
class S2TemperaturePlant(_BasePlant):
    temperature: float = 25.0
    integral: float = 0.0
    previous_error: float = 25.0
    overheat_latched: bool = False
    cooling_requested: bool = False

    def step(self, *, manual: bool = False, commands: Mapping[str, float | bool] | None = None) -> StepResult:
        commands = commands or {}
        manual_active = self._manual_active(manual)
        if not manual_active and manual:
            self.cooling_requested = True

        error = 50.0 - self.temperature
        self.integral += error * DT_SEC
        derivative = (error - self.previous_error) / DT_SEC
        self.previous_error = error
        pid = _clamp(2.5 * error + 0.8 * self.integral + 0.3 * derivative, 0.0, 100.0)

        if manual_active:
            heater_enable = bool(commands.get("heater_enable", False))
            heater_power = _clamp(float(commands.get("heater_power", 0.0)), 0.0, 1.0)
            fan = bool(commands.get("fan", False))
        else:
            heater_enable = pid > 0.0
            heater_power = pid / 100.0
            fan = self.temperature > 55.0 or self.cooling_requested

        if self.temperature > 135.0:
            self.overheat_latched = True
        elif self.temperature < 110.0:
            self.overheat_latched = False
        if self.overheat_latched:
            heater_enable, heater_power, fan = False, 0.0, True

        dtemp = 2.0 * float(heater_enable) * heater_power - 1.5 * float(fan) - 0.001 * (self.temperature - 25.0)
        self.temperature = _clamp(self.temperature + DT_SEC * dtemp, 20.0, 200.0)
        return StepResult({"T": self.temperature}, {"heater_enable": heater_enable,
                          "heater_power": heater_power, "fan": fan, "pid_output": pid},
                          self.overheat_latched, self._impact(self.temperature > 130.0))


@dataclass
class S3ThermalMixingPlant(_BasePlant):
    level: float = 50.0
    temperature: float = 25.0
    lockout_scans: int = 0
    previous_level: float = 50.0
    auto_inlet: bool = False
    auto_drain: bool = False
    auto_heater: bool = True

    def step(self, *, manual: bool = False, commands: Mapping[str, float | bool] | None = None) -> StepResult:
        commands = commands or {}
        manual_active = self._manual_active(manual)
        watchdog_expired = manual and not manual_active
        if manual_active:
            inlet = bool(commands.get("inlet", False))
            drain = bool(commands.get("drain", False))
            heater = bool(commands.get("heater", False))
        else:
            if self.level < 30.0:
                self.auto_inlet = True
            elif self.level > 65.0:
                self.auto_inlet = False
            if self.level < 50.0:
                self.auto_drain = False
            elif self.level > 70.0:
                self.auto_drain = True
            if self.temperature < 40.0:
                self.auto_heater = True
            elif self.temperature > 60.0:
                self.auto_heater = False
            inlet = self.auto_inlet
            drain = self.auto_drain or watchdog_expired
            heater = self.auto_heater

        rising_fast = self.level > 85.0 and (self.level - self.previous_level) > 1.5
        if self.level > 90.0 or rising_fast:
            self.lockout_scans = 50
        if self.lockout_scans > 0:
            inlet, drain = False, True
            self.lockout_scans -= 1
        if self.lockout_scans == 0 and self.level >= 75.0:
            # Keep the safety command active until its documented clear condition.
            inlet, drain = False, True
        if self.level < 5.0:
            drain, heater = False, False

        qin = 4.0 * float(inlet)
        qout = 3.5 * float(drain) + 0.3 * float(not drain)
        qevap = 0.002 * max(0.0, self.temperature - 30.0)
        mass = 0.5 + self.level / 100.0
        dlevel = qin - qout - qevap
        dtemp = ((0.5 * float(heater) - 0.1 * float(not heater)) / mass
                 - 0.08 * float(self.level > 1.0) * (qin / max(self.level, 1.0)) * (self.temperature - 15.0))
        self.previous_level = self.level
        self.level = _clamp(self.level + DT_SEC * dlevel, 0.0, 100.0)
        self.temperature = _clamp(self.temperature + DT_SEC * dtemp, 10.0, 95.0)
        active = self.lockout_scans > 0 or self.level >= 75.0
        return StepResult({"L": self.level, "T": self.temperature},
                          {"inlet": inlet, "drain": drain, "heater": heater}, active,
                          self._impact(self.level > 85.0 and self.temperature > 60.0))


@dataclass
class S4QuadTankPlant(_BasePlant):
    h1: float = 45.0
    h2: float = 45.0
    h3: float = 25.0
    h4: float = 25.0
    lockout_scans: int = 0
    auto_pump1_enabled: bool = False
    auto_pump2_enabled: bool = False

    def step(self, *, manual: bool = False, commands: Mapping[str, float | bool] | None = None) -> StepResult:
        commands = commands or {}
        manual_active = self._manual_active(manual)
        if manual_active:
            p1 = _clamp(float(commands.get("command1", 0.0)) / 100.0, 0.0, 1.0)
            p2 = _clamp(float(commands.get("command2", 0.0)) / 100.0, 0.0, 1.0)
        else:
            if self.h1 < 42.0:
                self.auto_pump1_enabled = True
            elif self.h1 > 48.0:
                self.auto_pump1_enabled = False
            if self.h2 < 42.0:
                self.auto_pump2_enabled = True
            elif self.h2 > 48.0:
                self.auto_pump2_enabled = False
            p1 = 0.55 if self.auto_pump1_enabled else 0.0
            p2 = 0.55 if self.auto_pump2_enabled else 0.0
            p1 = p1 if self.h4 <= 65.0 else 0.0
            p2 = p2 if self.h3 <= 65.0 else 0.0

        if max(self.h3, self.h4) > 95.0:
            self.lockout_scans = 50
        if self.lockout_scans > 0:
            p1, p2 = 0.0, 0.0
            self.lockout_scans -= 1
        if self.lockout_scans == 0 and not (self.h3 < 85.0 and self.h4 < 85.0):
            p1, p2 = 0.0, 0.0

        q1, q2 = 11.0 * p1, 11.0 * p2
        d1, d2 = 0.55 * sqrt(self.h1), 0.55 * sqrt(self.h2)
        d3, d4 = 0.35 * sqrt(self.h3), 0.35 * sqrt(self.h4)
        self.h1 = _clamp(self.h1 + DT_SEC * (0.65 * q1 + d3 - d1), 0.0, 100.0)
        self.h2 = _clamp(self.h2 + DT_SEC * (0.65 * q2 + d4 - d2), 0.0, 100.0)
        self.h3 = _clamp(self.h3 + DT_SEC * (0.35 * q2 - d3), 0.0, 100.0)
        self.h4 = _clamp(self.h4 + DT_SEC * (0.35 * q1 - d4), 0.0, 100.0)
        active = self.lockout_scans > 0 or not (self.h3 < 85.0 and self.h4 < 85.0)
        return StepResult({"h1": self.h1, "h2": self.h2, "h3": self.h3, "h4": self.h4},
                          {"pump1_power": p1, "pump2_power": p2,
                           "pump1_enable": p1 > 0.0, "pump2_enable": p2 > 0.0},
                          active, self._impact(self.h3 > 85.0 and 45.0 <= self.h4 <= 60.0))
