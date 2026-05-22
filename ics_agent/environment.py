from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .paths import SPHERE_WT_OPENPLC_DIR, SPHERE_WT_OPENPLC_SCRIPTS


class EnvironmentError(RuntimeError):
    """Raised when an environment backend cannot satisfy a request."""


def _import_openplc_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pyc_path = (
            SPHERE_WT_OPENPLC_SCRIPTS
            / "__pycache__"
            / f"{module_name}.cpython-310.pyc"
        )
        if not pyc_path.exists():
            raise
        loader = importlib.machinery.SourcelessFileLoader(module_name, str(pyc_path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            raise ModuleNotFoundError(
                f"unable to create import spec for compiled module {module_name}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        return module


class OpenPLCEnvironment:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.config = cfg
        self.backend = cfg.get("backend", "auto")
        self.keep_running = bool(cfg.get("keep_running", False))
        self.started = False

        scripts_path = str(SPHERE_WT_OPENPLC_SCRIPTS)
        sys.path = [entry for entry in sys.path if entry not in ("", scripts_path)]
        sys.path.insert(0, scripts_path)
        _import_openplc_module("modbus_raw")
        _import_openplc_module("native_plc_runtime")
        openplc_testbed = _import_openplc_module("openplc_testbed")

        self._testbed_module = openplc_testbed
        self._testbed_cls = openplc_testbed.OpenPLCWaterTreatmentTestbed
        self._testbed_error = openplc_testbed.TestbedError
        self.testbed = self._testbed_cls(
            base_dir=Path(cfg.get("base_dir", SPHERE_WT_OPENPLC_DIR)),
            controller_host=cfg.get("controller_host", "127.0.0.1"),
            controller_port=cfg.get("controller_port") or None,
            simulator_host=cfg.get("simulator_host", "127.0.0.1"),
            simulator_port=cfg.get("simulator_port") or None,
            unit_id=int(cfg.get("unit_id", 1)),
        )

    def start(self) -> dict[str, Any]:
        if self.started:
            return {"started": True}
        try:
            result = self.testbed.ensure_stack(
                build_if_missing=bool(self.config.get("build_if_missing", True)),
                timeout_sec=float(self.config.get("timeout_sec", 120.0)),
                start_bridge=bool(self.config.get("start_bridge", True)),
            )
        except self._testbed_error as exc:  # pragma: no cover - exercised in live runs
            raise EnvironmentError(str(exc)) from exc
        self.started = True
        return result

    def reset(self, initial_conditions: dict[str, int] | None = None) -> dict[str, Any]:
        try:
            return self.testbed.reset(
                initial_conditions,
                wait_sec=float(self.config.get("reset_wait_sec", 1.5)),
            )
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def observe_flat(self) -> dict[str, Any]:
        try:
            return self.testbed.observe_flat()
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def observe(self) -> dict[str, Any]:
        try:
            return self.testbed.observe()
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def act(self, action: dict[str, Any] | str) -> dict[str, Any]:
        descriptor = action if isinstance(action, dict) else {"type": action}
        if str(descriptor.get("type", "")).strip().lower() == "inspect_surface":
            return self.inspect_surface(descriptor)
        try:
            return self.testbed.apply_action(action)
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def inspect_surface(self, action: dict[str, Any]) -> dict[str, Any]:
        surface = str(action.get("surface", "")).strip().lower()
        if not surface:
            raise EnvironmentError("inspect_surface requires a non-empty `surface`")

        try:
            return self._inspect_surface(surface)
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc
        except Exception as exc:
            raise EnvironmentError(str(exc)) from exc

    def _inspect_surface(self, surface: str) -> dict[str, Any]:
        if surface == "tags":
            flat = self.observe_flat()
            findings = self._findings_from_flat_observation(flat)
            return {
                "state_name": flat.get("STATE_NAME"),
                "reconnaissance": {
                    "category": "reconnaissance",
                    "surface": surface,
                    "summary": self._recon_summary(surface, findings),
                    "findings": findings,
                    "groups": {
                        "levels": {
                            key: flat.get(key)
                            for key, _ in self._testbed_module.FIELD_GROUPS["levels"]
                        },
                        "status": {
                            key: flat.get(key)
                            for key, _ in self._testbed_module.FIELD_GROUPS["status"]
                        },
                        "states": {
                            key: flat.get(key)
                            for key, _ in self._testbed_module.FIELD_GROUPS["states"]
                        },
                        "alarms": {
                            key: flat.get(key)
                            for key, _ in self._testbed_module.FIELD_GROUPS["alarms"]
                        },
                        "process_controls": {
                            key: flat.get(key)
                            for key, _ in self._testbed_module.FIELD_GROUPS[
                                "process_controls"
                            ]
                        },
                    },
                },
            }

        with self.testbed.clients() as (ctrl, sim):
            if surface == "registers":
                groups = {
                    "levels": self._read_register_group(sim, "levels"),
                    "status": self._read_register_group(sim, "status"),
                    "pump_speed_cmd": {
                        "plane": "controller",
                        "kind": "holding_registers",
                        "start_address": 100,
                        "count": 1,
                        "values": {
                            "RW_Pump_Speed_Cmd": ctrl.read_holding_registers(
                                100, 1
                            ).registers[0]
                        },
                    },
                }
            elif surface == "io_image":
                groups = {
                    "commands": self._read_coil_group(ctrl, "commands"),
                    "states": self._read_coil_group(ctrl, "states"),
                    "alarms": self._read_coil_group(ctrl, "alarms"),
                    "hmi": self._read_coil_group(ctrl, "hmi"),
                    "process_controls": self._read_coil_group(
                        ctrl, "process_controls"
                    ),
                }
            elif surface == "alarm_bits":
                groups = {"alarms": self._read_coil_group(ctrl, "alarms")}
            elif surface == "process2_control_plane":
                groups = {
                    "process_controls": self._read_coil_group(ctrl, "process_controls"),
                    "process2_status": self._read_register_subset(
                        sim,
                        plane="simulator",
                        kind="holding_registers",
                        fields=(
                            ("ChemTreat_NaCl_Valve_Sts", 325),
                            ("ChemTreat_NaOCl_Valve_Sts", 326),
                            ("ChemTreat_HCl_Valve_Sts", 327),
                        ),
                    ),
                    "process2_levels": self._read_register_subset(
                        sim,
                        plane="simulator",
                        kind="holding_registers",
                        fields=(
                            ("ChemTreat_NaCl_Level", 302),
                            ("ChemTreat_NaOCl_Level", 303),
                            ("ChemTreat_HCl_Level", 304),
                        ),
                    ),
                }
            elif surface == "actuator_state_bits":
                groups = {
                    "commands": self._read_coil_group(ctrl, "commands"),
                    "status": self._read_register_group(sim, "status"),
                    "process_controls": self._read_coil_group(
                        ctrl, "process_controls"
                    ),
                }
            else:
                raise EnvironmentError(f"unsupported reconnaissance surface: {surface}")

            state_name = self._current_state_name(ctrl)
            findings = self._findings_from_groups(surface, groups, state_name=state_name)
            return {
                "state_name": state_name,
                "reconnaissance": {
                    "category": "reconnaissance",
                    "surface": surface,
                    "summary": self._recon_summary(surface, findings),
                    "findings": findings,
                    "groups": groups,
                },
            }

    def _read_coil_group(self, client: Any, group_name: str) -> dict[str, Any]:
        fields = self._testbed_module.FIELD_GROUPS[group_name]
        start_address = int(fields[0][1])
        count = len(fields)
        bits = client.read_coils(start_address, count).bits[:count]
        return {
            "plane": "controller",
            "kind": "coils",
            "start_address": start_address,
            "count": count,
            "values": {
                name: int(bool(bits[index])) for index, (name, _) in enumerate(fields)
            },
        }

    def _read_register_group(self, client: Any, group_name: str) -> dict[str, Any]:
        fields = self._testbed_module.FIELD_GROUPS[group_name]
        start_address = int(fields[0][1])
        count = len(fields)
        registers = client.read_holding_registers(start_address, count).registers
        return {
            "plane": "simulator",
            "kind": "holding_registers",
            "start_address": start_address,
            "count": count,
            "values": {
                name: registers[index] for index, (name, _) in enumerate(fields)
            },
        }

    @staticmethod
    def _read_register_subset(
        client: Any,
        *,
        plane: str,
        kind: str,
        fields: tuple[tuple[str, int], ...],
    ) -> dict[str, Any]:
        values = {
            name: client.read_holding_registers(address, 1).registers[0]
            for name, address in fields
        }
        addresses = [address for _, address in fields]
        return {
            "plane": plane,
            "kind": kind,
            "start_address": min(addresses),
            "count": len(fields),
            "values": values,
        }

    def _current_state_name(self, ctrl: Any) -> str:
        fields = self._testbed_module.FIELD_GROUPS["states"]
        start_address = int(fields[0][1])
        count = len(fields)
        bits = ctrl.read_coils(start_address, count).bits[:count]
        sample = {
            name: 1 if bits[index] else 0 for index, (name, _) in enumerate(fields)
        }
        return self._testbed_module.state_name(sample)

    def _findings_from_flat_observation(self, flat: dict[str, Any]) -> list[str]:
        findings = [f"state={flat.get('STATE_NAME', 'UNKNOWN')}"]
        active_alarms = [
            name
            for name, _ in self._testbed_module.FIELD_GROUPS["alarms"]
            if int(flat.get(name, 0) or 0) == 1
        ]
        if active_alarms:
            findings.append("active_alarms=" + ",".join(sorted(active_alarms)))
        if int(flat.get("P2_Selected", 0) or 0) == 1:
            findings.append("process2_selected")
        for field in (
            "RW_Tank_Level",
            "UF_UFFT_Tank_Level",
            "ChemTreat_NaCl_Level",
            "ChemTreat_NaOCl_Level",
            "ChemTreat_HCl_Level",
        ):
            if field in flat:
                findings.append(f"{field}={flat[field]}")
        return findings[:8]

    def _findings_from_groups(
        self,
        surface: str,
        groups: dict[str, Any],
        *,
        state_name: str,
    ) -> list[str]:
        findings = [f"state={state_name}"]
        values: dict[str, Any] = {}
        for payload in groups.values():
            for key, value in dict(payload.get("values", {})).items():
                values[key] = value

        if surface in {"tags", "registers"}:
            for field in (
                "RW_Tank_Level",
                "UF_UFFT_Tank_Level",
                "ChemTreat_NaCl_Level",
                "ChemTreat_NaOCl_Level",
                "ChemTreat_HCl_Level",
            ):
                if field in values:
                    findings.append(f"{field}={values[field]}")
        if any(int(values.get(name, 0) or 0) == 1 for name in ("Alarm_RW_Tank_LL", "Alarm_RW_Tank_HH")):
            active_alarms = [
                key
                for key, value in values.items()
                if key.startswith("Alarm_") and int(value or 0) == 1
            ]
            findings.append("active_alarms=" + ",".join(sorted(active_alarms)))
        if int(values.get("P2_Selected", 0) or 0) == 1:
            findings.append("process2_selected")
        for field in ("P2_NaCl_Cmd", "P2_NaOCl_Cmd", "P2_HCl_Cmd"):
            if int(values.get(field, 0) or 0) == 1:
                findings.append(f"{field}=1")
        for field in (
            "ChemTreat_NaCl_Valve_Sts",
            "ChemTreat_NaOCl_Valve_Sts",
            "ChemTreat_HCl_Valve_Sts",
        ):
            if int(values.get(field, 0) or 0) == 1:
                findings.append(f"{field}=1")
        return findings[:8]

    @staticmethod
    def _recon_summary(surface: str, findings: list[str]) -> str:
        if findings:
            return f"{surface}: " + "; ".join(findings[:4])
        return f"{surface}: no notable findings"

    def stop(self) -> None:
        if self.keep_running:
            return
        try:
            self.testbed.down()
        except self._testbed_error as exc:  # pragma: no cover - exercised in live runs
            raise EnvironmentError(str(exc)) from exc
        finally:
            self.started = False
