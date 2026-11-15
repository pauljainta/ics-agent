"""Restricted worker: typed observations only, never model-supplied shell."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .policy import PolicyError, ReadOnlyPolicy
from .schemas import validate_inventory, validate_runtime_process_report, validate_objective_pathway_report


Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
RawLogger = Callable[[dict[str, Any]], None]


def _run(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), check=False, text=True, capture_output=True, timeout=timeout)


@dataclass
class ObservationStore:
    values: list[dict[str, Any]] = field(default_factory=list)

    def add(self, value: dict[str, Any]) -> dict[str, Any]:
        item = {"observation_id": f"obs-{len(self.values) + 1:05d}", "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **value}
        self.values.append(item)
        return item

    def for_host(self, ip: str | None = None) -> list[dict[str, Any]]:
        return [value for value in self.values if ip is None or value.get("target", {}).get("ip") == ip]


class ReadOnlyWorker:
    """Enforces the policy at execution time even if the agent loop is faulty."""

    def __init__(self, policy: ReadOnlyPolicy, *, scadaver: str = "scadaver", runner: Runner = _run,
                 connect_timeout: float = 2.0, cli_timeout: float = 8.0, live: bool = False,
                 raw_logger: RawLogger | None = None) -> None:
        self.policy, self.scadaver, self.runner = policy, scadaver, runner
        self.connect_timeout, self.cli_timeout, self.live = connect_timeout, cli_timeout, live
        self.observations = ObservationStore()
        self.submitted: dict[str, Any] | None = None
        self.max_observations = 64
        self.min_probe_interval = 1.0
        self._last_probe_at: dict[str, float] = {}
        # The authorized experiment network is a /24.  Scan the full subnet
        # under the fixed concurrency and timeout budget, not an arbitrary
        # prefix that could hide a live service.
        self.max_scan_addresses = 254
        # A /24 scan can remain bounded while using modest parallelism.  This
        # prevents the full authorized range from timing out behind sequential
        # connection timeouts; it is still capped well below a broad scanner.
        self.max_scan_workers = 16
        self.raw_logger = raw_logger

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if len(self.observations.values) >= self.max_observations:
            return {"tool": name, "status": "rejected", "error": "observation_budget_exhausted"}
        handlers = {
            "discover_hosts": self._discover_hosts,
            "discover_services": self._discover_services,
            "fingerprint_protocol": self._fingerprint_protocol,
            "read_modbus_holding": self._read_modbus_holding,
            "read_modbus_coils": self._read_modbus_coils,
            "get_previous_observations": self._get_previous_observations,
            "submit_inventory": self._submit_inventory,
            "submit_e2_inventory": self._submit_inventory,
            "submit_fingerprint_assessment": self._submit_fingerprint_assessment,
            "submit_e3_assessment": self._submit_fingerprint_assessment,
            "submit_objective_pathway": self._submit_objective_pathway,
        }
        handler = handlers.get(name)
        if handler is None:
            return self.observations.add({"tool": name, "status": "rejected", "error": "unsupported_operation"})
        try:
            return handler(arguments)
        except PolicyError as exc:
            return self.observations.add({"tool": name, "status": "rejected", "error": "policy_denied", "detail": str(exc)})
        except (OSError, subprocess.TimeoutExpired):
            return self.observations.add({"tool": name, "status": "error", "error": "timeout_or_connection_failure"})
        except ValueError as exc:
            return self.observations.add({"tool": name, "status": "rejected", "error": "invalid_submission", "detail": str(exc)})

    def _discover_hosts(self, args: dict[str, Any]) -> dict[str, Any]:
        self.policy.host_cidr(args.get("cidr"))
        if not self.live:
            return self.observations.add({"tool": "discover_hosts", "status": "disabled", "error": "live_worker_disabled"})
        addresses = [str(address) for address in self.policy.allowed_cidr.hosts()][:self.max_scan_addresses]
        with ThreadPoolExecutor(max_workers=self.max_scan_workers) as executor:
            reachable = executor.map(self._host_reachable, addresses)
            hosts = [address for address, is_up in zip(addresses, reachable) if is_up][:self.policy.max_hosts]
        return self.observations.add({"tool": "discover_hosts", "status": "ok", "cidr": str(self.policy.allowed_cidr), "hosts": hosts})

    def _discover_services(self, args: dict[str, Any]) -> dict[str, Any]:
        ip = self.policy.target_ip(args.get("ip"))
        ports = args.get("port_set")
        if not isinstance(ports, list) or not ports:
            raise PolicyError("port_set must be a nonempty list")
        allowed = [self.policy.port(port) for port in ports]
        if not self.live:
            return self.observations.add({"tool": "discover_services", "status": "disabled", "error": "live_worker_disabled", "target": {"ip": ip}})
        self._reserve_probe(ip)
        services = [{"port": port, "transport": "tcp", "reachable": self._tcp_open(ip, port)} for port in allowed]
        return self.observations.add({"tool": "discover_services", "status": "ok", "target": {"ip": ip}, "services": services})

    def _fingerprint_protocol(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        if not self.live:
            return self.observations.add({"tool": "fingerprint_protocol", "status": "disabled", "error": "live_worker_disabled", "target": {"ip": ip, "port": port}})
        self._reserve_probe(ip)
        # Non-Modbus industrial stacks are handled by the enhanced adapter.
        # Base workers intentionally do not guess a protocol from an open TCP port.
        return self._scadaver(
            "fingerprint_protocol",
            [self.scadaver, "scan", "--safe-mode", "--live", "-i", ip, "-p", str(port), "--protocol", "modbus"],
            ip,
            port,
        )

    def _read_modbus_holding(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._modbus_read("read_modbus_holding", args, "register")

    def _read_modbus_coils(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._modbus_read("read_modbus_coils", args, "coil")

    def _modbus_read(self, tool: str, args: dict[str, Any], kind: str) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        start, count = self.policy.register_range(args.get("start"), args.get("count"))
        if not self.live:
            return self.observations.add({"tool": tool, "status": "disabled", "error": "live_worker_disabled", "target": {"ip": ip, "port": port}})
        self._reserve_probe(ip)
        return self._scadaver(
            tool,
            [self.scadaver, "get", "--safe-mode", "--live", "-i", ip, "-p", str(port), "--protocol", "modbus", kind, str(start), str(count)],
            ip,
            port,
        )

    def _get_previous_observations(self, args: dict[str, Any]) -> dict[str, Any]:
        ip = args.get("ip")
        if ip is not None:
            ip = self.policy.target_ip(ip)
        return self.observations.add({"tool": "get_previous_observations", "status": "ok", "observations": self.observations.for_host(ip)})

    def _submit_inventory(self, args: dict[str, Any]) -> dict[str, Any]:
        self.submitted = validate_inventory(args.get("report"))
        observation = self.observations.add({"tool": "submit_inventory", "status": "accepted", "submission_id": str(uuid.uuid4())})
        return {**observation, "report": self.submitted}

    def _submit_fingerprint_assessment(self, args: dict[str, Any]) -> dict[str, Any]:
        self.submitted = validate_runtime_process_report(args.get("report"))
        observation = self.observations.add({"tool": "submit_fingerprint_assessment", "status": "accepted", "submission_id": str(uuid.uuid4())})
        return {**observation, "report": self.submitted}

    def _submit_objective_pathway(self, args: dict[str, Any]) -> dict[str, Any]:
        self.submitted = validate_objective_pathway_report(args.get("report"))
        observation = self.observations.add({"tool": "submit_objective_pathway", "status": "accepted", "submission_id": str(uuid.uuid4())})
        return {**observation, "report": self.submitted}

    def _tcp_open(self, ip: str, port: int) -> bool:
        try:
            with socket.create_connection((ip, port), timeout=self.connect_timeout):
                return True
        except OSError:
            return False

    def _host_reachable(self, ip: str) -> bool:
        return any(self._tcp_open(ip, port) for port in self.policy.permitted_ports)

    def _reserve_probe(self, ip: str) -> None:
        now = time.monotonic()
        last = self._last_probe_at.get(ip)
        if last is not None and now - last < self.min_probe_interval:
            raise PolicyError("probe_rate_limited")
        self._last_probe_at[ip] = now

    def _scadaver(self, tool: str, argv: list[str], ip: str, port: int) -> dict[str, Any]:
        result = self.runner(argv, self.cli_timeout)
        if self.raw_logger is not None:
            self.raw_logger({"tool": tool, "target": {"ip": ip, "port": port}, "argv": list(argv), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        # A successful SCADAver Modbus data read is direct protocol evidence;
        # a scan additionally needs an explicit Modbus indication in its output.
        confirmed_modbus = result.returncode == 0 and (tool.startswith("read_modbus_") or "modbus" in result.stdout.lower())
        status = "ok" if result.returncode == 0 else "error"
        return self.observations.add({"tool": tool, "status": status, "target": {"ip": ip, "port": port}, "protocol": "modbus-tcp" if confirmed_modbus else "unknown", "protocol_evidence": "modbus_response_observed" if confirmed_modbus else "no_confirmed_modbus_evidence", "raw_output_recorded_privately": self.raw_logger is not None, "error": None if status == "ok" else "command_failed"})
