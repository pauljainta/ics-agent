from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .paths import SPHERE_WT_OPENPLC_DIR, SPHERE_WT_OPENPLC_SCRIPTS


class EnvironmentError(RuntimeError):
    """Raised when an environment backend cannot satisfy a request."""


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
        from openplc_testbed import OpenPLCWaterTreatmentTestbed, TestbedError  # type: ignore

        self._testbed_cls = OpenPLCWaterTreatmentTestbed
        self._testbed_error = TestbedError
        self.testbed = OpenPLCWaterTreatmentTestbed(
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
