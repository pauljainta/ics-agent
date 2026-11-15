"""Fail-closed validation for agent-visible read-only operations."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any


class PolicyError(ValueError):
    """Raised for requests that the restricted worker must not execute."""


@dataclass(frozen=True)
class ReadOnlyPolicy:
    allowed_cidr: ipaddress.IPv4Network
    permitted_ports: frozenset[int]
    max_register_count: int = 10
    max_hosts: int = 8

    @classmethod
    def build(cls, allowed_cidr: str, ports: tuple[int, ...], *, max_hosts: int) -> "ReadOnlyPolicy":
        network = ipaddress.ip_network(allowed_cidr, strict=True)
        if not isinstance(network, ipaddress.IPv4Network) or network.prefixlen < 24:
            raise PolicyError("allowed_cidr must be an IPv4 lab subnet no larger than /24")
        if not ports or any(port < 1 or port > 65535 for port in ports):
            raise PolicyError("permitted_ports must contain valid TCP ports")
        return cls(network, frozenset(ports), max_hosts=max_hosts)

    def target_ip(self, value: Any) -> str:
        try:
            address = ipaddress.ip_address(str(value))
        except ValueError as exc:
            raise PolicyError("target must be a literal IPv4 address") from exc
        if not isinstance(address, ipaddress.IPv4Address) or address not in self.allowed_cidr:
            raise PolicyError("target is outside the authorized lab subnet")
        if address in {self.allowed_cidr.network_address, self.allowed_cidr.broadcast_address}:
            raise PolicyError("target is not a usable lab host")
        return str(address)

    def port(self, value: Any) -> int:
        try:
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise PolicyError("port must be an integer") from exc
        if port not in self.permitted_ports:
            raise PolicyError("port is not in the approved read-only service set")
        return port

    def host_cidr(self, value: Any) -> str:
        if str(value) != str(self.allowed_cidr):
            raise PolicyError("host discovery is limited to the configured lab CIDR")
        return str(self.allowed_cidr)

    def register_range(self, start: Any, count: Any) -> tuple[int, int]:
        try:
            start_i, count_i = int(start), int(count)
        except (TypeError, ValueError) as exc:
            raise PolicyError("register range must use integer start and count") from exc
        if start_i < 0 or count_i < 1 or count_i > self.max_register_count:
            raise PolicyError("register count must be between 1 and 10")
        return start_i, count_i
