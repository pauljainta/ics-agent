"""Deterministic tool-only baseline using the same worker interface and scope."""

from __future__ import annotations

from typing import Any


def run_f0(worker: Any, *, cidr: str, ports: list[int]) -> list[dict[str, Any]]:
    observations = [worker.execute("discover_hosts", {"cidr": cidr})]
    for ip in observations[0].get("hosts", []):
        observations.append(worker.execute("discover_services", {"ip": ip, "port_set": ports}))
        for service in observations[-1].get("services", []):
            if service.get("reachable"):
                observations.append(worker.execute("fingerprint_protocol", {"ip": ip, "port": service["port"]}))
    return observations
