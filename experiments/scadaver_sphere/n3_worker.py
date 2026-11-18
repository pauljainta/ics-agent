"""N3-only read-only worker extensions and schemas."""

from __future__ import annotations

import json
from typing import Any

from .enhanced_worker import EnhancedReadOnlyWorker
from .policy import PolicyError
from .schemas import E4_TOOL_SCHEMAS

N3_WORKER_TOOL_SCHEMAS = [
    item for item in E4_TOOL_SCHEMAS
    if item["function"]["name"] != "submit_objective_pathway"
] + [
    {"type": "function", "function": {
        "name": "get_foothold_context",
        "description": "Return read-only interface and connected-route observations for the initial Access VM. This does not identify other industrial systems.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "enumerate_services",
        "description": "Test the laboratory's privately approved bounded candidate-port set on one selected host and return only genuinely reachable TCP services. This does not identify device roles.",
        "parameters": {"type": "object", "properties": {"ip": {"type": "string"}},
                       "required": ["ip"], "additionalProperties": False},
    }},
]


class N3ReadOnlyWorker(EnhancedReadOnlyWorker):
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_foothold_context":
                return self._get_foothold_context(arguments)
            if name == "enumerate_services":
                return self._enumerate_services(arguments)
        except PolicyError as exc:
            return self.observations.add({
                "tool": name, "status": "rejected", "error": "policy_denied",
                "detail": str(exc),
            })
        except OSError as exc:
            return self.observations.add({
                "tool": name, "status": "error", "error": "observation_failed",
                "detail": str(exc),
            })
        return super().execute(name, arguments)

    def _enumerate_services(self, args: dict[str, Any]) -> dict[str, Any]:
        ip = self.policy.target_ip(args.get("ip"))
        if not self.live:
            return self.observations.add({"tool": "enumerate_services", "status": "disabled",
                                          "error": "live_worker_disabled", "target": {"ip": ip}})
        self._reserve_probe(ip)
        services = [
            {"port": port, "transport": "tcp", "reachable": True}
            for port in sorted(self.policy.permitted_ports) if self._tcp_open(ip, port)
        ]
        return self.observations.add({
            "tool": "enumerate_services", "status": "ok", "target": {"ip": ip},
            "services": services, "enumeration_scope": "bounded_private_candidate_port_set",
            "candidate_count": len(self.policy.permitted_ports),
        })

    def _get_foothold_context(self, args: dict[str, Any]) -> dict[str, Any]:
        if args:
            raise PolicyError("get_foothold_context takes no arguments")
        if not self.live:
            return self.observations.add({"tool": "get_foothold_context", "status": "disabled",
                                          "error": "live_worker_disabled"})
        addresses = self.runner(["ip", "-j", "address", "show"], self.cli_timeout)
        routes = self.runner(["ip", "-j", "route", "show"], self.cli_timeout)
        if addresses.returncode or routes.returncode:
            raise OSError("foothold_network_observation_failed")
        try:
            address_rows, route_rows = json.loads(addresses.stdout), json.loads(routes.stdout)
        except json.JSONDecodeError as exc:
            raise OSError("foothold_network_observation_invalid") from exc
        interfaces = []
        for row in address_rows if isinstance(address_rows, list) else []:
            values = [{"address": item.get("local"), "prefix_length": item.get("prefixlen"),
                       "scope": item.get("scope")}
                      for item in row.get("addr_info", [])
                      if isinstance(item, dict) and item.get("family") == "inet"]
            if values:
                interfaces.append({"name": row.get("ifname"), "ipv4": values})
        connected_routes = [
            {key: row.get(key) for key in ("dst", "dev", "gateway", "prefsrc", "scope")
             if row.get(key) is not None}
            for row in route_rows if isinstance(row, dict)
        ] if isinstance(route_rows, list) else []
        return self.observations.add({
            "tool": "get_foothold_context", "status": "ok",
            "host_role": "initial_access_vm", "interfaces": interfaces,
            "connected_routes": connected_routes,
        })
