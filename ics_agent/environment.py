from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .action_semantics import (
    action_type,
    attacker_surface,
    containment_metadata,
    containment_surfaces,
    is_runtime_enforced_containment_action,
    normalize_action,
)
from .paths import OPENPLC_SUPPORT_DIR, OPENPLC_SUPPORT_SCRIPTS, REPO_ROOT, WORKSPACE_ROOT
from .remote_roles import RemoteRoleAgentProxy
from .utils import stable_hash, utc_now


class EnvironmentError(RuntimeError):
    """Raised when an environment backend cannot satisfy a request."""


def _import_openplc_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        candidate_paths = [
            WORKSPACE_ROOT
            / "sphere-usecases"
            / "water-treatment"
            / "implementations"
            / "openplc"
            / "scripts"
            / "__pycache__"
            / f"{module_name}.cpython-310.pyc",
            WORKSPACE_ROOT
            / "sphere-usecases"
            / "water-treatment"
            / "implementations"
            / "openplc"
            / "scripts"
            / "__pycache__"
            / f"{module_name}.py",
            OPENPLC_SUPPORT_SCRIPTS / "__pycache__" / f"{module_name}.cpython-310.pyc",
            OPENPLC_SUPPORT_SCRIPTS / "__pycache__" / f"{module_name}.py",
        ]
        for candidate_path in candidate_paths:
            if not candidate_path.exists():
                continue
            if candidate_path.suffix == ".pyc":
                loader = importlib.machinery.SourcelessFileLoader(
                    module_name, str(candidate_path)
                )
            else:
                loader = importlib.machinery.SourceFileLoader(
                    module_name, str(candidate_path)
                )
            spec = importlib.util.spec_from_loader(module_name, loader)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            loader.exec_module(module)
            return module
        raise


def _normalized_backend_name(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    return normalized or "auto"


@contextmanager
def _temporary_testbed_backend_mode(requested_backend: str):
    previous = os.environ.get("OPENPLC_TESTBED_BACKEND")
    try:
        if requested_backend in {"native", "docker"}:
            os.environ["OPENPLC_TESTBED_BACKEND"] = requested_backend
        else:
            os.environ.pop("OPENPLC_TESTBED_BACKEND", None)
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENPLC_TESTBED_BACKEND", None)
        else:
            os.environ["OPENPLC_TESTBED_BACKEND"] = previous


class RuntimeContainmentState:
    """Track concrete runtime enforcement that can block later attacker actions."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rules: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._blocked_attempt_count = 0

    def snapshot(self) -> dict[str, Any]:
        rules = {
            surface: dict(payload)
            for surface, payload in sorted(self._rules.items(), key=lambda item: item[0])
        }
        return {
            "active": bool(rules),
            "surface_scope": "all" if "all" in rules else ("targeted" if rules else None),
            "target_surfaces": list(rules.keys()),
            "rules": rules,
            "history": list(self._history),
            "blocked_attempt_count": self._blocked_attempt_count,
        }

    def apply(self, action: dict[str, Any] | str | None) -> dict[str, Any] | None:
        normalized = normalize_action(action)
        if normalized is None or not is_runtime_enforced_containment_action(normalized):
            return None

        target_surfaces = containment_surfaces(normalized)
        action_name = action_type(normalized)
        newly_enforced: list[str] = []
        already_enforced: list[str] = []
        stronger_alias = action_name == "quarantine_signal_source"

        for surface in target_surfaces:
            existing = dict(self._rules.get(surface) or {})
            current_strength = int(existing.get("strength", 0) or 0)
            next_strength = 2 if stronger_alias else 1
            if current_strength >= next_strength:
                already_enforced.append(surface)
                continue
            self._rules[surface] = {
                "surface": surface,
                "source_action_type": action_name,
                "source_action": dict(normalized),
                "strength": next_strength,
                "mode": "runtime_enforced",
            }
            newly_enforced.append(surface)

        applied = {
            "action_type": action_name,
            "target_surfaces": target_surfaces,
            "newly_enforced_surfaces": newly_enforced,
            "already_enforced_surfaces": already_enforced,
            "stronger_alias": stronger_alias,
        }
        self._history.append(applied)
        return applied

    def block_result(
        self,
        action: dict[str, Any] | str | None,
        *,
        role: str | None,
    ) -> dict[str, Any] | None:
        if str(role or "").strip().lower() != "attacker":
            return None
        normalized = normalize_action(action)
        if normalized is None:
            return None

        surface = attacker_surface(normalized)
        rule = self._rules.get("all") or self._rules.get(surface)
        if rule is None:
            return None

        self._blocked_attempt_count += 1
        source_action = dict(rule.get("source_action") or {})
        source_action_type = str(rule.get("source_action_type") or "")
        blocked_surface = "all" if "all" in self._rules else surface
        reason = {
            "all": "engineering host isolation blocks all later attacker actions",
            "hmi_session": "disabled HMI session blocks later attacker HMI commands",
            "sensor_write_path": "contained telemetry-write surface blocks later attacker sensor writes",
            "process2_manual_path": "contained Process 2 manual path blocks later attacker Process 2 commands",
        }.get(blocked_surface, "runtime containment blocked the attacker action")
        return {
            "blocked": True,
            "blocked_by_runtime_containment": True,
            "blocked_surface": blocked_surface,
            "attempted_surface": surface,
            "reason": reason,
            "blocked_by_action": source_action,
            "blocked_by_action_type": source_action_type,
            "runtime_containment": self.snapshot(),
        }


class _SphereOpenPLCBackend:
    """Manage the SPHERE-hosted OpenPLC runtime and expose local testbed semantics."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.keep_running = bool(config.get("keep_running", False))
        self.started = False

        self.experiment_ref = str(
            config.get("experiment", "openplcwater.lgarcia")
        ).strip()
        self.merge_server = str(
            config.get(
                "merge_server",
                os.environ.get(
                    "MRG_SERVER",
                    os.environ.get("MERGE_SERVER", "merge.example.invalid"),
                ),
            )
        ).strip() or None
        self.realization_name = str(config.get("realization", "wargamestable")).strip()
        self.revision = str(config.get("revision", "")).strip() or None
        self.xdc_name = str(config.get("xdc", "")).strip() or "x0"
        self.access_host = str(config.get("access_host", "access")).strip() or None
        self.engineering_host = (
            str(config.get("engineering_host", "engineering")).strip() or None
        )

        self.controller_forward_port = int(config.get("controller_port", 2502))
        self.simulator_forward_port = int(config.get("simulator_port", 2503))
        self.remote_controller_port = int(config.get("remote_controller_port", 1502))
        self.remote_simulator_port = int(config.get("remote_simulator_port", 1503))
        self.remote_bind_host = str(config.get("remote_bind_host", "0.0.0.0")).strip()
        self.controller_process_ip = str(
            config.get("controller_process_ip", "198.18.4.10")
        ).strip()
        self.simulator_process_ip = str(
            config.get("simulator_process_ip", "198.18.4.20")
        ).strip()
        self.bridge_host = str(config.get("bridge_host", "controller")).strip()
        self.bridge_cycle_ms = int(config.get("bridge_cycle_ms", 100))
        self.bridge_package_targets = tuple(
            config.get("bridge_package_targets", ["controller"])
        )
        role_hosts = dict(
            config.get(
                "remote_roles",
                {"attacker": "attacker_foothold", "defender": "defender"},
            )
        )
        self.remote_role_hosts = {
            str(role).strip(): str(host).strip()
            for role, host in role_hosts.items()
            if str(role).strip() and str(host).strip()
        }
        self.remote_role_port = int(config.get("remote_role_port", 18600))
        configured_role_ports = dict(
            config.get(
                "remote_role_local_ports",
                {"attacker": 28601, "defender": 28602},
            )
        )
        self.remote_role_local_ports = {
            str(role).strip(): int(port)
            for role, port in configured_role_ports.items()
            if str(role).strip()
        }
        self.bridge_start_timeout_sec = float(
            config.get("bridge_start_timeout_sec", 20.0)
        )
        self.tunnel_start_timeout_sec = float(
            config.get("tunnel_start_timeout_sec", 10.0)
        )
        self.ssh_timeout_sec = float(config.get("ssh_timeout_sec", 60.0))
        self.scp_timeout_sec = float(config.get("scp_timeout_sec", 120.0))
        self.ssh_connect_timeout_sec = float(
            config.get("ssh_connect_timeout_sec", 15.0)
        )
        self.ssh_attempts = max(1, int(config.get("ssh_attempts", 3)))
        self.scp_attempts = max(1, int(config.get("scp_attempts", self.ssh_attempts)))
        self.ssh_retry_backoff_sec = float(
            config.get("ssh_retry_backoff_sec", 1.0)
        )
        self.payload_stage_timeout_sec = float(
            config.get(
                "payload_stage_timeout_sec",
                max(self.ssh_timeout_sec, self.scp_timeout_sec),
            )
        )
        self.remote_start_timeout_sec = float(
            config.get("remote_start_timeout_sec", 10.0)
        )
        self.remote_controller_scan_ms = int(
            config.get("remote_controller_scan_ms", 50)
        )
        self.remote_simulator_scan_ms = int(
            config.get("remote_simulator_scan_ms", 100)
        )
        self.remote_reset_mode = str(
            config.get("remote_reset_mode", "soft_reseed")
        ).strip().lower()
        self.start_bridge = bool(config.get("start_bridge", True))
        self.stop_remote_runtime = bool(config.get("stop_remote_runtime", False))

        helper_root_value = config.get("helper_root", "~/.cache/ics-sphere")
        self.helper_root = Path(helper_root_value).expanduser().resolve()
        self.logs_dir = self.helper_root / "logs"
        ssh_config_value = config.get("ssh_config_path", self.helper_root / "openplc_ssh_config")
        self.ssh_config_path = Path(ssh_config_value).expanduser().resolve()
        self.identity_file = Path(
            config.get("identity_file", "~/.ssh/merge_key")
        ).expanduser()

        self.remote_runtime_dir = str(
            config.get("remote_runtime_dir", "~/runtime_payload")
        )
        self.remote_state_dir = str(config.get("remote_state_dir", "~/runtime_state"))
        self.remote_workspace_dir = str(
            config.get("remote_workspace_dir", "~/distributed_workspace")
        )
        self.remote_service_state_dir = str(
            config.get("remote_service_state_dir", "~/distributed_state")
        )
        self._default_initial_conditions = {
            "RW_Tank_Level": 600,
            "UF_UFFT_Tank_Level": 400,
            "ChemTreat_NaCl_Level": 800,
            "ChemTreat_NaOCl_Level": 800,
            "ChemTreat_HCl_Level": 800,
        }
        initial_conditions = config.get("initial_conditions") or {}
        self.initial_conditions = {
            **self._default_initial_conditions,
            **dict(initial_conditions),
        }

        self._merge_user: str | None = None
        self._realization_ref: str | None = None
        self._xdc_ref: str | None = None
        self._xdc_fqdn: str | None = None
        self._xdc_host: str | None = None
        self._direct_env: OpenPLCEnvironment | None = None
        self._phase_events: list[dict[str, Any]] = []
        self._command_log: list[dict[str, Any]] = []
        self._role_runtime_info: dict[str, dict[str, Any]] = {}
        self._start_result: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        try:
            self._phase_events.clear()
            self._command_log.clear()
            self._role_runtime_info.clear()
            self._prepare_transport()
            self._direct_env = OpenPLCEnvironment(self._direct_environment_config())
            self.started = True
            self._start_result = {
                "backend": "sphere_openplc",
                "experiment": self.experiment_ref,
                "realization": self._realization_ref,
                "revision": self.revision,
                "xdc": self._xdc_ref,
                "xdc_fqdn": self._xdc_fqdn,
                "controller_forward": f"127.0.0.1:{self.controller_forward_port}",
                "simulator_forward": f"127.0.0.1:{self.simulator_forward_port}",
                "bridge": "local" if self.start_bridge else "disabled",
                "remote_roles": self.remote_role_hosts,
                "staged_hops": [
                    host
                    for host in (self.access_host, self.engineering_host)
                    if host
                ],
            }
            return dict(self._start_result)
        except Exception:
            self._cleanup_local_transport()
            self._direct_env = None
            self.started = False
            raise

    def reset(self, initial_conditions: dict[str, int] | None = None) -> dict[str, Any]:
        conditions = dict(self._default_initial_conditions)
        if initial_conditions:
            conditions.update({key: int(value) for key, value in initial_conditions.items()})
        self._require_direct_env().clear_runtime_containment()

        if self.remote_reset_mode == "restart":
            self._restart_remote_runtime(conditions)
        else:
            self._soft_reset_remote_runtime(conditions)

        snapshot = self._require_direct_env().observe_flat()
        snapshot["reset_conditions"] = conditions
        if conditions != self._default_initial_conditions:
            snapshot["warning"] = (
                "non-default initial conditions are currently experimental on the "
                "SPHERE-backed remote runtime"
            )
        return snapshot

    def _restart_remote_runtime(self, conditions: dict[str, int]) -> None:
        if self.start_bridge:
            self._stop_local_process("bridge")
        self._stop_remote_process("controller", "controller", self.remote_controller_port)
        self._stop_remote_process("simulator", "simulator", self.remote_simulator_port)
        self._ensure_remote_process(
            "controller",
            self.remote_controller_port,
            initial_conditions=conditions,
        )
        self._ensure_remote_process(
            "simulator",
            self.remote_simulator_port,
            initial_conditions=conditions,
        )

        if not self._local_port_listening(self.controller_forward_port):
            self._stop_local_process("controller_tunnel")
            self._start_tunnel(
                "controller",
                local_port=self.controller_forward_port,
                remote_port=self.remote_controller_port,
                pid_name="controller_tunnel",
                log_name="tunnel-controller.log",
            )
        if not self._local_port_listening(self.simulator_forward_port):
            self._stop_local_process("simulator_tunnel")
            self._start_tunnel(
                "simulator",
                local_port=self.simulator_forward_port,
                remote_port=self.remote_simulator_port,
                pid_name="simulator_tunnel",
                log_name="tunnel-simulator.log",
            )
        if self.start_bridge:
            self._start_bridge()

    def _soft_reset_remote_runtime(self, conditions: dict[str, int]) -> None:
        direct_env = self._require_direct_env()
        testbed = direct_env.testbed

        if self.start_bridge:
            self._stop_local_process("bridge")
        if not self._local_port_listening(self.controller_forward_port):
            self._stop_local_process("controller_tunnel")
            self._start_tunnel(
                "controller",
                local_port=self.controller_forward_port,
                remote_port=self.remote_controller_port,
                pid_name="controller_tunnel",
                log_name="tunnel-controller.log",
            )
        if not self._local_port_listening(self.simulator_forward_port):
            self._stop_local_process("simulator_tunnel")
            self._start_tunnel(
                "simulator",
                local_port=self.simulator_forward_port,
                remote_port=self.remote_simulator_port,
                pid_name="simulator_tunnel",
                log_name="tunnel-simulator.log",
            )

        testbed._clear_external_controls()
        try:
            direct_env.act({"type": "hmi_reset", "process_id": "WT-P1"})
        except EnvironmentError:
            testbed._clear_external_controls()
        testbed.set_levels(
            conditions,
            wait_sec=float(self.config.get("soft_reset_level_wait_sec", 1.0)),
        )
        testbed._clear_external_controls()
        time.sleep(float(self.config.get("reset_wait_sec", 1.5)))
        if self.start_bridge:
            self._start_bridge()

    def observe_flat(self) -> dict[str, Any]:
        return self._require_direct_env().observe_flat()

    def observe(self) -> dict[str, Any]:
        return self._require_direct_env().observe()

    def act(self, action: dict[str, Any] | str, *, role: str | None = None) -> dict[str, Any]:
        return self._require_direct_env().act(action, role=role)

    def inspect_surface(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._require_direct_env().inspect_surface(action)

    def stop(self) -> None:
        self._cleanup_local_transport()
        if self.keep_running:
            self.started = False
            return
        if self.stop_remote_runtime:
            self._stop_remote_process(
                "controller", "controller", self.remote_controller_port
            )
            self._stop_remote_process(
                "simulator", "simulator", self.remote_simulator_port
            )
            for target in sorted(set(self.remote_role_hosts.values())):
                self._stop_remote_role_service(target)
        self.started = False

    def _cleanup_local_transport(self) -> None:
        self._stop_local_process("bridge")
        self._stop_orphan_local_bridges()
        self._stop_local_process("controller_tunnel")
        self._stop_orphan_local_tunnel(self.controller_forward_port, self.remote_controller_port, "controller")
        self._stop_local_process("simulator_tunnel")
        self._stop_orphan_local_tunnel(self.simulator_forward_port, self.remote_simulator_port, "simulator")
        for role in self.remote_role_hosts:
            self._stop_local_process(f"{role}_tunnel")
            local_port = self.remote_role_local_ports.get(role)
            if local_port is not None:
                self._stop_orphan_local_tunnel(
                    local_port,
                    self.remote_role_port,
                    self.remote_role_hosts[role],
                )

    def _prepare_transport(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.helper_root.mkdir(parents=True, exist_ok=True)

        self._phase_event("phase_a", "bootstrap_start", experiment=self.experiment_ref)
        self._merge_user = self._run_checked(self._mrg_command("whoami")).strip()
        if not self._merge_user:
            raise EnvironmentError(
                "Merge authentication is unavailable; `mrg whoami` returned no user"
            )
        self._phase_event("phase_a", "merge_auth_verified", merge_user=self._merge_user)

        experiment_name, project_name = self._split_ref(self.experiment_ref)
        self._realization_ref = self._qualify_realization_ref(
            self.realization_name, experiment_name, project_name
        )
        self._xdc_ref = self._qualify_xdc_ref(self.xdc_name, self._merge_user)

        self._ensure_realization()
        self._phase_event(
            "phase_a",
            "realization_verified",
            realization=self._realization_ref,
            revision=self.revision,
        )
        self._ensure_materialization()
        self._phase_event(
            "phase_a",
            "materialization_verified",
            realization=self._realization_ref,
        )
        self._ensure_xdc()
        self._attach_xdc()
        self._write_ssh_config()
        self._phase_event(
            "phase_a",
            "xdc_attached",
            xdc=self._xdc_ref,
            xdc_host=self._xdc_host,
        )

        self._ensure_remote_packages("controller", ["python3-pymodbus", "python3-yaml"])
        for target in sorted(set(self.remote_role_hosts.values())):
            self._ensure_remote_packages(target, ["python3-psutil", "python3-yaml"])

        stage_targets = [
            target
            for target, port in (
                ("controller", self.remote_controller_port),
                ("simulator", self.remote_simulator_port),
            )
            if not self._remote_port_listening(target, port, retries=2)
        ]
        if stage_targets:
            self._stage_remote_runtime(stage_targets)
            self._phase_event(
                "phase_a",
                "process_runtime_staged",
                targets=stage_targets,
            )
        self._ensure_remote_process(
            "controller",
            self.remote_controller_port,
            initial_conditions=self._default_initial_conditions,
        )
        self._ensure_remote_process(
            "simulator",
            self.remote_simulator_port,
            initial_conditions=self._default_initial_conditions,
        )
        self._phase_event(
            "phase_a",
            "process_runtime_ready",
            controller=f"{self.controller_process_ip}:{self.remote_controller_port}",
            simulator=f"{self.simulator_process_ip}:{self.remote_simulator_port}",
        )

        role_targets = sorted(set(self.remote_role_hosts.values()))
        if role_targets:
            self._stage_remote_role_workspace(role_targets)
            for target in role_targets:
                self._ensure_remote_role_service(target)
            self._phase_event(
                "phase_a",
                "remote_role_services_ready",
                role_hosts=self.remote_role_hosts,
            )

        self._stop_local_process("bridge")
        self._stop_orphan_local_bridges()
        self._stop_local_process("controller_tunnel")
        self._stop_orphan_local_tunnel(
            self.controller_forward_port,
            self.remote_controller_port,
            "controller",
        )
        self._stop_local_process("simulator_tunnel")
        self._stop_orphan_local_tunnel(
            self.simulator_forward_port,
            self.remote_simulator_port,
            "simulator",
        )
        for role in self.remote_role_hosts:
            self._stop_local_process(f"{role}_tunnel")
            local_port = self.remote_role_local_ports.get(role)
            if local_port is not None:
                self._stop_orphan_local_tunnel(
                    local_port,
                    self.remote_role_port,
                    self.remote_role_hosts[role],
                )

        self._ensure_local_port_free(
            self.controller_forward_port, process_name="controller tunnel"
        )
        self._ensure_local_port_free(
            self.simulator_forward_port, process_name="simulator tunnel"
        )
        self._start_tunnel(
            "controller",
            local_port=self.controller_forward_port,
            remote_port=self.remote_controller_port,
            pid_name="controller_tunnel",
            log_name="tunnel-controller.log",
        )
        self._start_tunnel(
            "simulator",
            local_port=self.simulator_forward_port,
            remote_port=self.remote_simulator_port,
            pid_name="simulator_tunnel",
            log_name="tunnel-simulator.log",
        )
        for role, target in self.remote_role_hosts.items():
            local_port = self.remote_role_local_ports.get(role)
            if local_port is None:
                raise EnvironmentError(
                    f"remote role `{role}` is missing a local forwarded port"
                )
            self._ensure_local_port_free(
                local_port,
                process_name=f"{role} runtime tunnel",
            )
            self._start_tunnel(
                target,
                local_port=local_port,
                remote_port=self.remote_role_port,
                pid_name=f"{role}_tunnel",
                log_name=f"tunnel-{role}.log",
            )
        self._phase_event(
            "phase_a",
            "transport_ready",
            controller_forward=self.controller_forward_port,
            simulator_forward=self.simulator_forward_port,
            remote_role_ports=self.remote_role_local_ports,
        )
        if self.start_bridge:
            self._start_bridge()
            self._phase_event(
                "phase_a",
                "bridge_ready",
                bridge_host="local",
            )

    def _direct_environment_config(self) -> dict[str, Any]:
        local_backend = _normalized_backend_name(self.config.get("local_backend", "auto"))
        return {
            **self.config,
            "backend": local_backend,
            "keep_running": True,
            "start_bridge": False,
            "build_if_missing": False,
            "controller_host": "127.0.0.1",
            "controller_port": self.controller_forward_port,
            "simulator_host": "127.0.0.1",
            "simulator_port": self.simulator_forward_port,
        }

    def _require_direct_env(self) -> OpenPLCEnvironment:
        if self._direct_env is None:
            self.start()
        assert self._direct_env is not None
        return self._direct_env

    def remote_role_proxy(
        self,
        *,
        role: str,
        profile_name: str,
        profile: dict[str, Any],
    ) -> RemoteRoleAgentProxy | None:
        target = self.remote_role_hosts.get(role)
        local_port = self.remote_role_local_ports.get(role)
        if not target or local_port is None:
            return None
        return RemoteRoleAgentProxy(
            role=role,
            profile_name=profile_name,
            profile=profile,
            endpoint=f"http://127.0.0.1:{local_port}",
            assigned_node=target,
            request_timeout_sec=float(self.config.get("remote_role_timeout_sec", 20.0)),
        )

    def record_role_runtime(self, role: str, runtime_info: dict[str, Any] | None) -> None:
        if not runtime_info:
            return
        payload = dict(runtime_info)
        payload.setdefault("assigned_node", self.remote_role_hosts.get(role))
        self._role_runtime_info[role] = payload

    def runtime_containment_snapshot(self) -> dict[str, Any]:
        return self._require_direct_env().runtime_containment_snapshot()

    def export_run_artifacts(self, run_dir: Path, bundle: dict[str, Any]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        environment_snapshot = {
            "backend": "sphere_openplc",
            "experiment": self.experiment_ref,
            "realization": self._realization_ref,
            "revision": self.revision,
            "xdc": self._xdc_ref,
            "xdc_fqdn": self._xdc_fqdn,
            "runtime_containment": self.runtime_containment_snapshot(),
            "node_placement": {
                "controller": {
                    "runtime": "process_controller",
                    "process_ip": self.controller_process_ip,
                    "runtime_port": self.remote_controller_port,
                },
                "simulator": {
                    "runtime": "process_simulator",
                    "process_ip": self.simulator_process_ip,
                    "runtime_port": self.remote_simulator_port,
                },
                **(
                    {
                        self.access_host: {
                            "runtime": "access_hop",
                            "service_port": 22,
                        }
                    }
                    if self.access_host
                    else {}
                ),
                **(
                    {
                        self.engineering_host: {
                            "runtime": "engineering_hmi",
                            "service_port": 22,
                        }
                    }
                    if self.engineering_host
                    else {}
                ),
                **{
                    host: {
                        "runtime": role,
                        "service_port": self.remote_role_port,
                        "local_forward_port": self.remote_role_local_ports.get(role),
                    }
                    for role, host in self.remote_role_hosts.items()
                },
            },
            "role_runtime_info": self._role_runtime_info,
            "backend_config": {
                "bridge_host": "local" if self.start_bridge else "disabled",
                "remote_roles": self.remote_role_hosts,
                "merge_server": self.merge_server,
                "staged_hops": [
                    host
                    for host in (self.access_host, self.engineering_host)
                    if host
                ],
                "metric_boundary": bundle.get("config_snapshot", {}).get("phase_boundary"),
            },
            "repo_commits": bundle.get("config_snapshot", {}).get("reproducibility", {}),
            "startup": self._start_result,
        }
        self._write_json(run_dir / "environment_snapshot.json", environment_snapshot)

        phase_log_path = run_dir / "phase_log.jsonl"
        phase_log_path.parent.mkdir(parents=True, exist_ok=True)
        phase_items = list(self._phase_events)
        for item in bundle.get("system_events", []):
            if item.get("event_type") in {"wargame_t0", "role_residency"}:
                phase_items.append(
                    {
                        "timestamp_utc": item.get("timestamp_utc"),
                        "phase": "phase_b" if item.get("event_type") == "wargame_t0" else "phase_a",
                        "event": item.get("event_type"),
                        **{
                            key: value
                            for key, value in item.items()
                            if key not in {"timestamp_utc", "event_type"}
                        },
                    }
                )
        phase_items.sort(key=lambda item: str(item.get("timestamp_utc", "")))
        with phase_log_path.open("w") as handle:
            for item in phase_items:
                handle.write(json.dumps(item) + "\n")

        command_log_path = run_dir / "command_log.txt"
        command_log_path.write_text(
            "\n".join(
                f"[{item['timestamp_utc']}] {item['command']}" for item in self._command_log
            )
            + ("\n" if self._command_log else "")
        )

        journal_lines = [
            "# Distributed SPHERE Wargame Run",
            "",
            f"- Run ID: `{bundle.get('run_id')}`",
            f"- Scenario: `{bundle.get('scenario', {}).get('catalog_id')}`",
            f"- Condition: `{bundle.get('condition_name')}`",
            f"- Status: `{bundle.get('status')}`",
            "",
            "## Phase Events",
        ]
        for item in phase_items:
            journal_lines.append(
                f"- `{item.get('timestamp_utc')}` `{item.get('phase')}` `{item.get('event')}`"
            )
        (run_dir / "run_journal.md").write_text("\n".join(journal_lines) + "\n")

        remote_logs_dir = run_dir / "remote_logs"
        remote_logs_dir.mkdir(parents=True, exist_ok=True)
        self._fetch_remote_file(
            "controller",
            self._remote_file_path(self.remote_state_dir, "controller.log"),
            remote_logs_dir / "controller_runtime.log",
        )
        self._fetch_remote_file(
            "simulator",
            self._remote_file_path(self.remote_state_dir, "simulator.log"),
            remote_logs_dir / "simulator_runtime.log",
        )
        if self.start_bridge:
            bridge_log_path = self.logs_dir / "modbus_bridge.log"
            if bridge_log_path.exists():
                (remote_logs_dir / "bridge.log").write_text(bridge_log_path.read_text())
            else:
                (remote_logs_dir / "bridge.log").write_text("local bridge log missing\n")
        for role, host in self.remote_role_hosts.items():
            self._fetch_remote_file(
                host,
                self._remote_file_path(self.remote_service_state_dir, "role-service.log"),
                remote_logs_dir / f"{role}_runtime.log",
            )

    def _phase_event(self, phase: str, event: str, **fields: Any) -> None:
        self._phase_events.append(
            {
                "timestamp_utc": utc_now(),
                "phase": phase,
                "event": event,
                **fields,
            }
        )

    def _record_command(self, command: str) -> None:
        self._command_log.append({"timestamp_utc": utc_now(), "command": command})

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @staticmethod
    def _split_ref(value: str) -> tuple[str, str]:
        parts = [part for part in value.split(".") if part]
        if len(parts) != 2:
            raise EnvironmentError(
                f"expected `experiment.project` reference, got `{value}`"
            )
        return parts[0], parts[1]

    def _mrg_command(self, *args: str) -> list[str]:
        command = ["mrg"]
        if self.merge_server:
            command.extend(["-s", self.merge_server])
        command.extend(args)
        return command

    @staticmethod
    def _qualify_realization_ref(
        value: str, experiment_name: str, project_name: str
    ) -> str:
        if value.count(".") == 2:
            return value
        if "." in value:
            raise EnvironmentError(
                f"expected realization name or full `realization.experiment.project`, got `{value}`"
            )
        return f"{value}.{experiment_name}.{project_name}"

    @staticmethod
    def _qualify_xdc_ref(value: str, project_name: str) -> str:
        if value.count(".") == 1:
            return value
        if "." in value:
            raise EnvironmentError(
                f"expected XDC name or full `xdc.project`, got `{value}`"
            )
        return f"{value}.{project_name}"

    def _ensure_realization(self) -> None:
        assert self._realization_ref is not None
        payload = self._show_json("realization", self._realization_ref)
        if payload is None:
            if not self.revision:
                raise EnvironmentError(
                    f"realization `{self._realization_ref}` is missing and no revision was configured"
                )
            self._run_checked(
                self._mrg_command(
                    "--syncwait",
                    "30m",
                    "realize",
                    self._realization_ref,
                    "revision",
                    self.revision,
                )
            )
            payload = self._show_json("realization", self._realization_ref)
        if payload is None:
            raise EnvironmentError(f"unable to inspect realization `{self._realization_ref}`")
        realization = payload.get("result", {}).get("realization", {})
        realized_revision = str(realization.get("xhash", "")).strip() or None
        if self.revision is None:
            self.revision = realized_revision
        elif realized_revision and realized_revision != self.revision:
            raise EnvironmentError(
                f"realization `{self._realization_ref}` is pinned to revision `{realized_revision}`, not configured revision `{self.revision}`"
            )

    def _ensure_materialization(self) -> None:
        assert self._realization_ref is not None
        if self._show_json("materialization", self._realization_ref) is not None:
            return
        self._run_checked(
            self._mrg_command(
                "--syncwait", "30m", "materialize", self._realization_ref
            )
        )
        if self._show_json("materialization", self._realization_ref) is None:
            raise EnvironmentError(
                f"materialization for `{self._realization_ref}` did not become available"
            )

    def _ensure_xdc(self) -> None:
        assert self._xdc_ref is not None
        if self._show_json("xdc", self._xdc_ref) is not None:
            return
        self._run_checked(
            self._mrg_command(
                "--syncwait",
                "10m",
                "new",
                "xdc",
                self._xdc_ref,
                "--duration",
                str(self.config.get("xdc_duration", "2w")),
            )
        )
        if self._show_json("xdc", self._xdc_ref) is None:
            raise EnvironmentError(f"XDC `{self._xdc_ref}` was not created successfully")

    def _attach_xdc(self) -> None:
        assert self._xdc_ref is not None
        assert self._realization_ref is not None
        payload = self._show_json("xdc", self._xdc_ref)
        if payload is None:
            raise EnvironmentError(f"unable to inspect XDC `{self._xdc_ref}`")
        materialization = (
            payload.get("xdc", {}).get("materialization")
            if isinstance(payload, dict)
            else None
        )
        if materialization != self._realization_ref:
            self._run_checked(
                self._mrg_command(
                    "--syncwait",
                    "10m",
                    "xdc",
                    "attach",
                    "--force",
                    self._xdc_ref,
                    self._realization_ref,
                )
            )
            payload = self._show_json("xdc", self._xdc_ref)
            if payload is None:
                raise EnvironmentError(f"unable to verify XDC `{self._xdc_ref}` attachment")
        self._xdc_fqdn = str(payload.get("xdc", {}).get("fqdn", "")).strip() or None
        if not self._xdc_fqdn:
            raise EnvironmentError(f"XDC `{self._xdc_ref}` does not expose an FQDN")
        self._xdc_host = self._xdc_fqdn.split(".", 1)[0]

    def _write_ssh_config(self) -> None:
        assert self._merge_user is not None
        assert self._xdc_host is not None
        lines = [
            "Host *",
            "    StrictHostKeyChecking no",
            "    UserKnownHostsFile=/dev/null",
            f"    User {self._merge_user}",
            "",
            "Host jump",
            "    HostName gateway.example.invalid",
            "    Port 2022",
            f"    User {self._merge_user}",
        ]
        if self.identity_file.exists():
            lines.append(f"    IdentityFile {self.identity_file}")
        lines.extend(
            [
                "",
                f"Host {self._xdc_host}",
                f"    HostName {self._xdc_host}",
                f"    User {self._merge_user}",
            ]
        )
        if self.identity_file.exists():
            lines.append(f"    IdentityFile {self.identity_file}")
        lines.extend(
            [
                "    ProxyJump jump",
                "",
            ]
        )
        host_specs: list[tuple[str, str]] = []
        if self.access_host:
            host_specs.append((self.access_host, self._xdc_host))
        if self.engineering_host:
            host_specs.append(
                (
                    self.engineering_host,
                    self.access_host or self._xdc_host,
                )
            )

        control_proxy = self.engineering_host or self.access_host or self._xdc_host
        seen_hosts = {host for host, _ in host_specs}
        for host in [
            "controller",
            "simulator",
            *sorted(set(self.remote_role_hosts.values())),
        ]:
            if host in seen_hosts:
                continue
            host_specs.append((host, control_proxy))
            seen_hosts.add(host)

        for host, proxy_jump in host_specs:
            lines.extend(
                [
                    f"Host {host}",
                    f"    HostName {host}",
                    f"    User {self._merge_user}",
                ]
            )
            if self.identity_file.exists():
                lines.append(f"    IdentityFile {self.identity_file}")
            lines.extend(
                [
                    f"    ProxyJump {proxy_jump}",
                    "",
                ]
            )
        self.ssh_config_path.write_text("\n".join(lines))

    def _stage_remote_runtime(self, targets: list[str] | tuple[str, ...] | None = None) -> None:
        sources = [
            OPENPLC_SUPPORT_SCRIPTS / "native_plc_runtime.py",
            OPENPLC_SUPPORT_SCRIPTS / "__pycache__" / "native_plc_runtime.cpython-310.pyc",
            OPENPLC_SUPPORT_SCRIPTS / "__pycache__" / "modbus_raw.cpython-310.pyc",
            OPENPLC_SUPPORT_SCRIPTS / "modbus_bridge.py",
            OPENPLC_SUPPORT_SCRIPTS / "__pycache__" / "modbus_bridge.cpython-310.pyc",
        ]
        missing = [str(path) for path in sources if not path.exists()]
        if missing:
            raise EnvironmentError(
                "required native runtime payload is missing: " + ", ".join(missing)
            )
        for target in targets or ("controller", "simulator"):
            self._ssh_checked(
                target,
                "mkdir -p "
                + self._remote_shell_path(self.remote_runtime_dir)
                + " "
                + self._remote_shell_path(self.remote_state_dir),
                timeout_sec=self.payload_stage_timeout_sec,
            )
            for source in sources:
                destination_path = self._remote_scp_path(
                    self.remote_runtime_dir, source.name
                )
                self._scp_file_checked(
                    source,
                    f"{target}:{destination_path}",
                    timeout_sec=self.payload_stage_timeout_sec,
                )

    def _ensure_remote_packages(self, target: str, packages: list[str]) -> None:
        if not packages:
            return
        quoted = " ".join(shlex.quote(item) for item in packages)
        command = (
            "missing=''; "
            f"for pkg in {quoted}; do "
            "dpkg -s \"$pkg\" >/dev/null 2>&1 || missing=\"$missing $pkg\"; "
            "done; "
            "if [ -n \"$missing\" ]; then "
            "sudo apt-get update -qq && "
            "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $missing; "
            "fi"
        )
        self._ssh_checked(target, command, timeout_sec=max(self.ssh_timeout_sec, 300.0))

    def _remote_role_stage_hash(self) -> str:
        tracked_paths = [
            REPO_ROOT / "ics_agent",
            REPO_ROOT / "configs",
            REPO_ROOT / "vendor",
        ]
        manifest: list[tuple[str, int, int]] = []
        for path in tracked_paths:
            if not path.exists():
                raise EnvironmentError(f"remote role payload source is missing: {path}")
            if path.is_dir():
                for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
                    stat = file_path.stat()
                    manifest.append(
                        (
                            str(file_path.relative_to(path.parent)),
                            stat.st_mtime_ns,
                            stat.st_size,
                        )
                    )
            else:
                stat = path.stat()
                manifest.append((str(path.name), stat.st_mtime_ns, stat.st_size))
        return stable_hash(manifest, length=16)

    def _role_workspace_archive(self) -> tuple[Path, str]:
        stage_hash = self._remote_role_stage_hash()
        archive_path = self.helper_root / f"remote-role-workspace-{stage_hash}.tar.gz"
        if archive_path.exists():
            return archive_path, stage_hash

        with tarfile.open(archive_path, "w:gz") as handle:
            handle.add(REPO_ROOT / "ics_agent", arcname="ics-agent/ics_agent")
            handle.add(REPO_ROOT / "configs", arcname="ics-agent/configs")
            handle.add(REPO_ROOT / "vendor", arcname="ics-agent/vendor")
        return archive_path, stage_hash

    def _stage_remote_role_workspace(
        self,
        targets: list[str] | tuple[str, ...],
    ) -> None:
        archive_path, stage_hash = self._role_workspace_archive()
        marker_path = self._remote_file_path(self.remote_workspace_dir, ".stage_hash")
        remote_archive = self._remote_scp_path(self.remote_workspace_dir, ".payload.tar.gz")
        for target in targets:
            current_hash = self._ssh_checked(
                target,
                (
                    f"mkdir -p {self._remote_shell_path(self.remote_workspace_dir)} "
                    f"{self._remote_shell_path(self.remote_service_state_dir)}; "
                    f"if [ -f {marker_path} ]; then cat {marker_path}; fi"
                ),
                timeout_sec=self.payload_stage_timeout_sec,
            ).strip()
            if current_hash == stage_hash:
                continue
            self._stop_remote_role_service(target)
            self._scp_file_checked(
                archive_path,
                f"{target}:{remote_archive}",
                timeout_sec=self.payload_stage_timeout_sec,
            )
            extract_command = (
                f"rm -rf {self._remote_shell_path(self.remote_workspace_dir)}/ics-agent "
                f"{self._remote_shell_path(self.remote_workspace_dir)}/vendor && "
                f"mkdir -p {self._remote_shell_path(self.remote_workspace_dir)} && "
                f"tar -xzf {self._remote_shell_path(self.remote_workspace_dir)}/.payload.tar.gz "
                f"-C {self._remote_shell_path(self.remote_workspace_dir)} && "
                f"printf '%s\\n' {shlex.quote(stage_hash)} > {marker_path}"
            )
            self._ssh_checked(
                target,
                extract_command,
                timeout_sec=self.payload_stage_timeout_sec,
            )

    def _ensure_remote_role_service(self, target: str) -> None:
        if self._remote_port_listening(target, self.remote_role_port):
            return
        workspace_root = self._remote_shell_path(self.remote_workspace_dir)
        service_state = self._remote_shell_path(self.remote_service_state_dir)
        launcher_spec = json.dumps(
            {
                "command": [
                    "python3",
                    "-m",
                    "ics_agent.remote_service",
                    "--listen-host",
                    "127.0.0.1",
                    "--listen-port",
                    str(self.remote_role_port),
                    "--controller-host",
                    self.controller_process_ip,
                    "--controller-port",
                    str(self.remote_controller_port),
                    "--simulator-host",
                    self.simulator_process_ip,
                    "--simulator-port",
                    str(self.remote_simulator_port),
                    "--node-label",
                    target,
                ],
                "workspace_dir": self.remote_workspace_dir,
                "state_dir": self.remote_service_state_dir,
            },
            sort_keys=True,
        )
        launcher_spec_literal = json.dumps(launcher_spec)
        remote_command = (
            "python3 - <<'PY'\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import subprocess\n"
            f"spec = json.loads({launcher_spec_literal})\n"
            "workspace_dir = os.path.expanduser(spec['workspace_dir'])\n"
            "state_dir = os.path.expanduser(spec['state_dir'])\n"
            "pathlib.Path(state_dir).mkdir(parents=True, exist_ok=True)\n"
            "log_path = os.path.join(state_dir, 'role-service.log')\n"
            "env = dict(os.environ)\n"
            "env['PYTHONPATH'] = os.path.join(workspace_dir, 'ics-agent')\n"
            "with open(log_path, 'ab') as log_handle:\n"
            "    proc = subprocess.Popen(\n"
            "        spec['command'],\n"
            "        cwd=os.path.join(workspace_dir, 'ics-agent'),\n"
            "        env=env,\n"
            "        stdin=subprocess.DEVNULL,\n"
            "        stdout=log_handle,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        start_new_session=True,\n"
            "    )\n"
            "pathlib.Path(os.path.join(state_dir, 'role-service.pid')).write_text(str(proc.pid))\n"
            "PY"
        )
        self._ssh_checked(target, remote_command)
        deadline = time.time() + self.remote_start_timeout_sec
        while time.time() < deadline:
            if self._remote_port_listening(target, self.remote_role_port):
                return
            time.sleep(0.25)
        raise EnvironmentError(
            f"remote role service on `{target}` did not begin listening on port {self.remote_role_port}"
        )

    def _stop_remote_role_service(self, target: str) -> None:
        pid_path = self._remote_file_path(self.remote_service_state_dir, "role-service.pid")
        pattern = shlex.quote(r"^python3 -m ics_agent\.remote_service( |$)")
        command = (
            f"if [ -f {pid_path} ]; then kill $(cat {pid_path}) >/dev/null 2>&1 || true; fi; "
            f"pkill -f {pattern} >/dev/null 2>&1 || true; "
            f"rm -f {pid_path}"
        )
        self._ssh_checked(target, command, retries=1)

    def _ensure_remote_bridge(self) -> None:
        target = self.bridge_host
        if self._remote_bridge_running():
            return
        launcher_spec = json.dumps(
            {
                "command": [
                    "python3",
                    "modbus_bridge.py",
                    "--controller",
                    f"127.0.0.1:{self.remote_controller_port}",
                    "--simulator",
                    f"{self.simulator_process_ip}:{self.remote_simulator_port}",
                    "--cycle-ms",
                    str(self.bridge_cycle_ms),
                ],
                "runtime_dir": self.remote_runtime_dir,
                "state_dir": self.remote_state_dir,
            },
            sort_keys=True,
        )
        launcher_spec_literal = json.dumps(launcher_spec)
        remote_command = (
            "python3 - <<'PY'\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import subprocess\n"
            f"spec = json.loads({launcher_spec_literal})\n"
            "runtime_dir = os.path.expanduser(spec['runtime_dir'])\n"
            "state_dir = os.path.expanduser(spec['state_dir'])\n"
            "pathlib.Path(state_dir).mkdir(parents=True, exist_ok=True)\n"
            "log_path = os.path.join(state_dir, 'bridge.log')\n"
            "with open(log_path, 'ab') as log_handle:\n"
            "    proc = subprocess.Popen(\n"
            "        spec['command'],\n"
            "        cwd=runtime_dir,\n"
            "        stdin=subprocess.DEVNULL,\n"
            "        stdout=log_handle,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        start_new_session=True,\n"
            "    )\n"
            "pathlib.Path(os.path.join(state_dir, 'bridge.pid')).write_text(str(proc.pid))\n"
            "PY"
        )
        self._ssh_checked(target, remote_command)
        deadline = time.time() + self.bridge_start_timeout_sec
        log_path = self._remote_file_path(self.remote_state_dir, "bridge.log")
        while time.time() < deadline:
            if not self._remote_bridge_running():
                time.sleep(0.25)
                continue
            log_text = self._ssh_checked(
                target,
                f"test -f {log_path} && tail -n 40 {log_path} || true",
                retries=1,
            )
            if "Connected to both PLCs" in log_text and "Bridge loop started" in log_text:
                return
            time.sleep(0.25)
        raise EnvironmentError("remote bridge did not report a healthy start")

    def _remote_bridge_running(self) -> bool:
        pid_path = self._remote_file_path(self.remote_state_dir, "bridge.pid")
        pattern = shlex.quote(r"^python3 modbus_bridge\.py( |$)")
        command = (
            f"if [ -f {pid_path} ] && ps -p $(cat {pid_path}) -o args= | grep -Eq {pattern}; "
            "then echo ready; "
            f"elif pgrep -fo -f {pattern} >/dev/null 2>&1; then echo ready; "
            "fi"
        )
        return self._ssh_checked(self.bridge_host, command, retries=1).strip() == "ready"

    def _stop_remote_bridge(self) -> None:
        pid_path = self._remote_file_path(self.remote_state_dir, "bridge.pid")
        pattern = shlex.quote(r"^python3 modbus_bridge\.py( |$)")
        command = (
            f"if [ -f {pid_path} ]; then kill $(cat {pid_path}) >/dev/null 2>&1 || true; fi; "
            f"pkill -f {pattern} >/dev/null 2>&1 || true; "
            f"rm -f {pid_path}"
        )
        self._ssh_checked(self.bridge_host, command, retries=1)

    def _ensure_remote_process(
        self,
        target: str,
        port: int,
        *,
        initial_conditions: dict[str, int] | None,
    ) -> None:
        pid_name = "controller" if target == "controller" else "simulator"
        if self._remote_port_listening(target, port):
            self._record_remote_pid_if_missing(target, pid_name, port)
            return
        command = [
            "python3",
            "native_plc_runtime.py",
            "--role",
            pid_name,
            "--host",
            self.remote_bind_host,
            "--port",
            str(port),
            "--scan-ms",
            str(
                self.remote_controller_scan_ms
                if pid_name == "controller"
                else self.remote_simulator_scan_ms
            ),
        ]
        if pid_name == "simulator":
            command.extend(
                [
                    "--initial-conditions",
                    json.dumps(initial_conditions or self.initial_conditions, sort_keys=True),
                ]
            )
        launcher_spec = json.dumps(
            {
                "command": command,
                "role": pid_name,
                "runtime_dir": self.remote_runtime_dir,
                "state_dir": self.remote_state_dir,
            },
            sort_keys=True,
        )
        launcher_spec_literal = json.dumps(launcher_spec)
        remote_command = (
            "python3 - <<'PY'\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import subprocess\n"
            f"spec = json.loads({launcher_spec_literal})\n"
            "runtime_dir = os.path.expanduser(spec['runtime_dir'])\n"
            "state_dir = os.path.expanduser(spec['state_dir'])\n"
            "pathlib.Path(runtime_dir).mkdir(parents=True, exist_ok=True)\n"
            "pathlib.Path(state_dir).mkdir(parents=True, exist_ok=True)\n"
            "log_path = os.path.join(state_dir, spec['role'] + '.log')\n"
            "with open(log_path, 'ab') as log_handle:\n"
            "    proc = subprocess.Popen(\n"
            "        spec['command'],\n"
            "        cwd=runtime_dir,\n"
            "        stdin=subprocess.DEVNULL,\n"
            "        stdout=log_handle,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        start_new_session=True,\n"
            "    )\n"
            "pid_path = os.path.join(state_dir, spec['role'] + '.pid')\n"
            "pathlib.Path(pid_path).write_text(str(proc.pid))\n"
            "PY"
        )
        self._ssh_checked(target, remote_command)
        deadline = time.time() + self.remote_start_timeout_sec
        while time.time() < deadline:
            if self._remote_port_listening(target, port):
                return
            time.sleep(0.25)
        raise EnvironmentError(
            f"remote {pid_name} runtime on `{target}` did not begin listening on port {port}"
        )

    def _start_tunnel(
        self,
        target: str,
        *,
        local_port: int,
        remote_port: int,
        pid_name: str,
        log_name: str,
    ) -> None:
        log_path = self.logs_dir / log_name
        self._record_command(
            f"ssh -F {self.ssh_config_path} -L {local_port}:127.0.0.1:{remote_port} {target}"
        )
        with log_path.open("w") as handle:
            process = subprocess.Popen(
                [
                    "ssh",
                    "-F",
                    str(self.ssh_config_path),
                    "-N",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=30",
                    "-o",
                    "ServerAliveCountMax=3",
                    "-L",
                    f"{local_port}:127.0.0.1:{remote_port}",
                    target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._write_pid(pid_name, process.pid)
        deadline = time.time() + self.tunnel_start_timeout_sec
        while time.time() < deadline:
            if self._local_port_listening(local_port):
                return
            if process.poll() is not None:
                time.sleep(0.25)
                if self._local_port_listening(local_port):
                    return
                raise EnvironmentError(
                    f"{pid_name} exited early: {self._tail_log(log_path)}"
                )
            time.sleep(0.25)
        raise EnvironmentError(
            f"{pid_name} did not open local port {local_port}: {self._tail_log(log_path)}"
        )

    def _start_bridge(self) -> None:
        log_path = self.logs_dir / "modbus_bridge.log"
        with log_path.open("w") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(OPENPLC_SUPPORT_SCRIPTS / "modbus_bridge.py"),
                    "--controller",
                    f"127.0.0.1:{self.controller_forward_port}",
                    "--simulator",
                    f"127.0.0.1:{self.simulator_forward_port}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._write_pid("bridge", process.pid)
        deadline = time.time() + self.bridge_start_timeout_sec
        while time.time() < deadline:
            if process.poll() is not None:
                raise EnvironmentError(
                    f"modbus bridge exited early: {self._tail_log(log_path)}"
                )
            log_text = log_path.read_text() if log_path.exists() else ""
            if "Connected to both PLCs" in log_text and "Bridge loop started" in log_text:
                return
            time.sleep(0.25)
        raise EnvironmentError(
            "modbus bridge did not report a healthy start: " + self._tail_log(log_path)
        )

    def _stop_remote_process(self, target: str, role: str, port: int) -> None:
        self._record_remote_pid_if_missing(target, role, port)
        pid_path = self._remote_file_path(self.remote_state_dir, f"{role}.pid")
        role_pattern = shlex.quote(self._remote_runtime_pattern(role, port))
        kill_command = (
            f"if [ -f {pid_path} ]; then "
            f"kill $(cat {pid_path}) >/dev/null 2>&1 || true; "
            "fi; "
            f"pkill -f {role_pattern} >/dev/null 2>&1 || true; "
            f"rm -f {pid_path}"
        )
        self._ssh_checked(target, kill_command)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._remote_port_listening(target, port):
                return
            time.sleep(0.25)
        self._ssh_checked(
            target,
            f"pkill -9 -f {role_pattern} >/dev/null 2>&1 || true",
        )
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self._remote_port_listening(target, port):
                return
            time.sleep(0.25)
        raise EnvironmentError(
            f"remote {role} runtime on `{target}` did not stop listening on port {port}"
        )

    def _show_json(self, kind: str, ref: str) -> dict[str, Any] | None:
        try:
            text = self._run_checked(self._mrg_command("show", kind, ref, "--json"))
        except EnvironmentError:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise EnvironmentError(f"failed to decode `mrg show {kind}` output: {exc}") from exc

    def _run_checked(self, command: list[str]) -> str:
        self._record_command(" ".join(command))
        return self._run_capture(command)

    def _run_capture(self, command: list[str], *, timeout_sec: float | None = None) -> str:
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise EnvironmentError(
                f"command timed out after {timeout_sec}s: {' '.join(command)}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise EnvironmentError(
                f"command failed ({completed.returncode}): {' '.join(command)}"
                + (f": {detail}" if detail else "")
            )
        return completed.stdout

    def _ssh_command(self, target: str) -> list[str]:
        return [
            "ssh",
            "-F",
            str(self.ssh_config_path),
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(self.ssh_connect_timeout_sec)}",
            "-o",
            f"ConnectionAttempts={self.ssh_attempts}",
            "-T",
            target,
        ]

    def _ssh_checked(
        self,
        target: str,
        remote_command: str,
        *,
        timeout_sec: float | None = None,
        retries: int | None = None,
    ) -> str:
        self._record_command(f"ssh {target} {remote_command[:180]}")
        last_error: EnvironmentError | None = None
        for attempt in range(1, max(1, retries or self.ssh_attempts) + 1):
            try:
                return self._run_capture(
                    [*self._ssh_command(target), remote_command],
                    timeout_sec=timeout_sec or self.ssh_timeout_sec,
                )
            except EnvironmentError as exc:
                last_error = exc
                if attempt >= max(1, retries or self.ssh_attempts):
                    raise
                time.sleep(self.ssh_retry_backoff_sec * attempt)
        assert last_error is not None
        raise last_error

    def _scp_file_checked(
        self,
        source: Path,
        destination: str,
        *,
        timeout_sec: float | None = None,
        retries: int | None = None,
    ) -> None:
        self._record_command(f"scp {source} {destination}")
        timeout_value = timeout_sec or self.scp_timeout_sec
        attempt_count = max(1, retries or self.scp_attempts)
        last_error: EnvironmentError | None = None
        for attempt in range(1, attempt_count + 1):
            try:
                completed = subprocess.run(
                    [
                        "scp",
                        "-F",
                        str(self.ssh_config_path),
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        f"ConnectTimeout={int(self.ssh_connect_timeout_sec)}",
                        "-o",
                        f"ConnectionAttempts={self.ssh_attempts}",
                        str(source),
                        destination,
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=timeout_value,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = EnvironmentError(
                    f"copy timed out after {timeout_value}s: {source} -> {destination}"
                )
            else:
                if completed.returncode == 0:
                    return
                detail = (completed.stderr or completed.stdout).strip()
                last_error = EnvironmentError(
                    f"failed to copy `{source.name}` to `{destination}`"
                    + (f": {detail}" if detail else "")
                )
            if attempt >= attempt_count:
                break
            time.sleep(self.ssh_retry_backoff_sec * attempt)
        assert last_error is not None
        raise last_error

    def _fetch_remote_file(self, target: str, remote_path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._record_command(f"scp {target}:{remote_path} {destination}")
        try:
            completed = subprocess.run(
                [
                    "scp",
                    "-F",
                    str(self.ssh_config_path),
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={int(self.ssh_connect_timeout_sec)}",
                    "-o",
                    f"ConnectionAttempts={self.ssh_attempts}",
                    f"{target}:{remote_path}",
                    str(destination),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.scp_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            destination.write_text(f"log fetch timed out from {target}:{remote_path}\n")
            return
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            destination.write_text(f"log fetch failed from {target}:{remote_path}\n{detail}\n")

    def _remote_port_listening(self, target: str, port: int, *, retries: int = 1) -> bool:
        command = (
            "ss -ltn | grep -q "
            + shlex.quote(f":{port} ")
            + " && echo ready || true"
        )
        try:
            return self._ssh_checked(target, command, retries=retries).strip() == "ready"
        except EnvironmentError:
            return False

    def _remote_runtime_pattern(self, role: str, port: int) -> str:
        bind_host_pattern = re.escape(self.remote_bind_host)
        return (
            rf"^python3 native_plc_runtime\.py --role {role} --host {bind_host_pattern} "
            rf"--port {port}( |$)"
        )

    def _record_remote_pid_if_missing(self, target: str, role: str, port: int) -> None:
        pid_path = self._remote_file_path(self.remote_state_dir, f"{role}.pid")
        pattern = shlex.quote(self._remote_runtime_pattern(role, port))
        reconcile_command = (
            f"pid_path={pid_path}; "
            "existing=''; "
            f"if [ -f {pid_path} ]; then existing=$(cat {pid_path}); fi; "
            "if [ -n \"$existing\" ] "
            f"&& ps -p \"$existing\" -o args= | grep -Eq {pattern}; then "
            "printf '%s\\n' \"$existing\"; "
            "exit 0; "
            "fi; "
            f"pid=$(pgrep -fo -f {pattern} || true); "
            "if [ -n \"$pid\" ]; then "
            "printf '%s\\n' \"$pid\" > \"$pid_path\"; "
            "printf '%s\\n' \"$pid\"; "
            "else "
            "rm -f \"$pid_path\"; "
            "fi"
        )
        self._ssh_checked(target, reconcile_command)

    @staticmethod
    def _local_port_listening(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _remote_shell_path(path: str) -> str:
        if path.startswith("~/"):
            return "$HOME/" + path[2:]
        return shlex.quote(path)

    def _remote_file_path(self, directory: str, filename: str) -> str:
        base = self._remote_shell_path(directory)
        if base.endswith("/"):
            return base + shlex.quote(filename)
        return base + "/" + shlex.quote(filename)

    @staticmethod
    def _remote_scp_path(directory: str, filename: str) -> str:
        base = directory[2:] if directory.startswith("~/") else directory
        if base.endswith("/"):
            return base + filename
        return base + "/" + filename

    def _ensure_local_port_free(self, port: int, *, process_name: str) -> None:
        if self._local_port_listening(port):
            raise EnvironmentError(
                f"local port {port} is already in use; stop the conflicting process before starting the {process_name}"
            )

    def _pid_path(self, name: str) -> Path:
        return self.helper_root / f"{name}.pid"

    def _write_pid(self, name: str, pid: int) -> None:
        self._pid_path(name).write_text(f"{pid}\n")

    def _read_pid(self, name: str) -> int | None:
        path = self._pid_path(name)
        if not path.exists():
            return None
        try:
            return int(path.read_text().strip())
        except ValueError:
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _stop_local_process(self, name: str) -> None:
        pid = self._read_pid(name)
        path = self._pid_path(name)
        if pid is None:
            path.unlink(missing_ok=True)
            return
        if self._pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.time() + 5.0
            while time.time() < deadline and self._pid_alive(pid):
                time.sleep(0.1)
            if self._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        path.unlink(missing_ok=True)

    def _stop_orphan_local_bridges(self) -> None:
        pattern = (
            r"modbus_bridge\.py --controller 127\.0\.0\.1:"
            + str(self.controller_forward_port)
            + r" --simulator 127\.0\.0\.1:"
            + str(self.simulator_forward_port)
        )
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            return

    def _stop_orphan_local_tunnel(
        self,
        local_port: int,
        remote_port: int,
        target: str,
    ) -> None:
        pattern = (
            str(local_port)
            + r":127\.0\.0\.1:"
            + str(remote_port)
            + r" "
            + re.escape(target)
        )
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            return

    @staticmethod
    def _tail_log(path: Path, lines: int = 20) -> str:
        if not path.exists():
            return "log file missing"
        content = path.read_text().splitlines()
        return "\n".join(content[-lines:])


class OpenPLCEnvironment:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.config = cfg
        self.backend = cfg.get("backend", "auto")
        self._backend_name = _normalized_backend_name(self.backend)
        self.keep_running = bool(cfg.get("keep_running", False))
        self.started = False
        self._runtime_containment = RuntimeContainmentState()
        self._backend_impl: _SphereOpenPLCBackend | None = None

        if self._backend_name == "sphere_openplc":
            self._backend_impl = _SphereOpenPLCBackend(cfg)
            return

        scripts_path = str(OPENPLC_SUPPORT_SCRIPTS)
        sys.path = [entry for entry in sys.path if entry not in ("", scripts_path)]
        sys.path.insert(0, scripts_path)
        _import_openplc_module("modbus_raw")
        _import_openplc_module("native_plc_runtime")
        openplc_testbed = _import_openplc_module("openplc_testbed")

        self._testbed_module = openplc_testbed
        self._testbed_cls = openplc_testbed.OpenPLCWaterTreatmentTestbed
        self._testbed_error = openplc_testbed.TestbedError
        self.testbed = self._testbed_cls(
            base_dir=Path(cfg.get("base_dir", OPENPLC_SUPPORT_DIR)),
            controller_host=cfg.get("controller_host", "127.0.0.1"),
            controller_port=cfg.get("controller_port") or None,
            simulator_host=cfg.get("simulator_host", "127.0.0.1"),
            simulator_port=cfg.get("simulator_port") or None,
            unit_id=int(cfg.get("unit_id", 1)),
        )

    def start(self) -> dict[str, Any]:
        if self._backend_impl is not None:
            result = self._backend_impl.start()
            self.started = self._backend_impl.started
            return result
        if self.started:
            return {"started": True}
        try:
            with _temporary_testbed_backend_mode(self._backend_name):
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
        if self._backend_impl is not None:
            return self._backend_impl.reset(initial_conditions)
        self.clear_runtime_containment()
        try:
            return self.testbed.reset(
                initial_conditions,
                wait_sec=float(self.config.get("reset_wait_sec", 1.5)),
            )
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def observe_flat(self) -> dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.observe_flat()
        try:
            return self.testbed.observe_flat()
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def observe(self) -> dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.observe()
        try:
            return self.testbed.observe()
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def act(self, action: dict[str, Any] | str, *, role: str | None = None) -> dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.act(action, role=role)
        descriptor = normalize_action(action) or {"type": action}
        blocked = self._runtime_containment.block_result(descriptor, role=role)
        if blocked is not None:
            state_after = self._observe_state_snapshot()
            state_after.update(blocked)
            return state_after

        if str(descriptor.get("type", "")).strip().lower() == "inspect_surface":
            return self.inspect_surface(descriptor)

        enforcement = self._runtime_containment.apply(descriptor)
        if enforcement is not None:
            state_after = self._observe_state_snapshot()
            containment = containment_metadata(descriptor)
            containment.update(
                {
                    "is_symbolic": False,
                    "execution_mode": "runtime_enforced",
                    "newly_enforced_surfaces": enforcement.get(
                        "newly_enforced_surfaces", []
                    ),
                    "already_enforced_surfaces": enforcement.get(
                        "already_enforced_surfaces", []
                    ),
                }
            )
            state_after["containment"] = containment
            state_after["runtime_containment"] = self.runtime_containment_snapshot()
            return state_after

        try:
            return self.testbed.apply_action(action)
        except self._testbed_error as exc:
            raise EnvironmentError(str(exc)) from exc

    def runtime_containment_snapshot(self) -> dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.runtime_containment_snapshot()
        return self._runtime_containment.snapshot()

    def clear_runtime_containment(self) -> None:
        if self._backend_impl is not None:
            direct_env = self._backend_impl._require_direct_env()
            direct_env.clear_runtime_containment()
            return
        self._runtime_containment.reset()

    def _observe_state_snapshot(self) -> dict[str, Any]:
        observation = self.observe_flat()
        return {
            "state_name": observation.get("STATE_NAME")
            or observation.get("state_name")
            or "UNKNOWN",
        }

    def inspect_surface(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.inspect_surface(action)
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

    def remote_role_proxy(
        self,
        *,
        role: str,
        profile_name: str,
        profile: dict[str, Any],
    ) -> RemoteRoleAgentProxy | None:
        if self._backend_impl is None:
            return None
        return self._backend_impl.remote_role_proxy(
            role=role,
            profile_name=profile_name,
            profile=profile,
        )

    def record_role_runtime(self, role: str, runtime_info: dict[str, Any] | None) -> None:
        if self._backend_impl is None:
            return
        self._backend_impl.record_role_runtime(role, runtime_info)

    def export_run_artifacts(self, run_dir: Path, bundle: dict[str, Any]) -> None:
        if self._backend_impl is None:
            return
        self._backend_impl.export_run_artifacts(run_dir, bundle)

    def stop(self) -> None:
        if self._backend_impl is not None:
            self._backend_impl.stop()
            self.started = False
            return
        if self.keep_running:
            return
        try:
            self.testbed.down()
        except self._testbed_error as exc:  # pragma: no cover - exercised in live runs
            raise EnvironmentError(str(exc)) from exc
        finally:
            self.started = False
