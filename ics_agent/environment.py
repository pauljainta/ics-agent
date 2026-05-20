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
        try:
            return self.testbed.apply_action(action)
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def stop(self) -> None:
        if self.keep_running:
            return
        try:
            self.testbed.down()
        except self._testbed_error as exc:  # pragma: no cover - exercised in live runs
            raise EnvironmentError(str(exc)) from exc
        finally:
            self.started = False
