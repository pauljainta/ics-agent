"""Enhanced, read-only observation adapters for the SCADAver worker."""

from __future__ import annotations

import http.client
import re
import socket
import subprocess
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from .policy import PolicyError
from .tool_results import normalize_tool_result, policy_error_category
from .worker import ReadOnlyWorker

_HOLDING = re.compile(r"^\s*4\d+\s+(-?\d+)\b", re.MULTILINE)
_COIL = re.compile(r"^\s*\d+\s+(ON|OFF)\b", re.MULTILINE)
_IDENTITY_FIELD = re.compile(r"^\s*(vendor(?:\s+name)?|product(?:\s+code)?|revision(?:\s+level)?)\s*:\s*(.{1,128})\s*$", re.IGNORECASE | re.MULTILINE)


def parse_scadaver_values(stdout: str, register_type: str, count: int) -> list[int | bool] | None:
    if register_type == "holding":
        values = [int(value) for value in _HOLDING.findall(stdout)]
    else:
        values = [value == "ON" for value in _COIL.findall(stdout)]
    return values[:count] if len(values) >= count else None


def parse_identity_fields(stdout: str) -> dict[str, str]:
    """Return only explicitly labeled fields, never scanner guesses."""
    return {name.lower().replace(" ", "_"): value.strip() for name, value in _IDENTITY_FIELD.findall(stdout)}


class EnhancedReadOnlyWorker(ReadOnlyWorker):
    """Adds bounded value, temporal, identity, and HTTP observations."""

    max_samples = 5
    min_sample_interval = 1.0
    max_series_seconds = 20.0
    max_probe_operations = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.allow_semantic_metadata = bool(kwargs.pop("allow_semantic_metadata", True))
        super().__init__(*args, **kwargs)
        self.probe_operations = 0

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "read_modbus_holding": lambda: self._value_read("holding", arguments),
            "read_modbus_coils": lambda: self._value_read("coil", arguments),
            "read_modbus_input_registers": lambda: self._native_modbus_read("input", arguments),
            "read_modbus_discrete_inputs": lambda: self._native_modbus_read("discrete", arguments),
            "explore_modbus_map": lambda: self._explore_modbus_map(arguments),
            "sample_modbus_series": lambda: self._sample_series(arguments),
            "sample_cross_device_series": lambda: self._sample_cross_device_series(arguments),
            "probe_device_identity": lambda: self._probe_identity(arguments),
            "inspect_http_service": lambda: self._inspect_http(arguments),
            "inspect_http_resource": lambda: self._inspect_http_resource(arguments),
            "browse_opcua_nodes": lambda: self._browse_opcua(arguments),
            "read_opcua_values": lambda: self._read_opcua(arguments),
            "read_s7_data": lambda: self._read_s7(arguments),
            "inspect_s7_metadata": lambda: self._inspect_s7_metadata(arguments),
            "fingerprint_protocol": lambda: self._fingerprint_any(arguments),
        }
        handler = handlers.get(name)
        if handler is None:
            return super().execute(name, arguments)
        if len(self.observations.values) >= self.max_observations:
            return normalize_tool_result(
                {"tool": name, "status": "rejected", "error": "observation_budget_exhausted",
                 "error_category": "resource_budget_exhausted",
                 "detail": "The per-session observation budget is exhausted."},
                tool=name, attempted=False, budget_consumed=False)
        if not self.live:
            return normalize_tool_result(self.observations.add(
                {"tool": name, "status": "disabled", "error": "live_worker_disabled"}),
                tool=name, attempted=False, budget_consumed=False)
        try:
            return normalize_tool_result(handler(), tool=name)
        except PolicyError as exc:
            detail = str(exc)
            return normalize_tool_result(self.observations.add(
                {"tool": name, "status": "rejected", "error": "policy_denied",
                 "error_category": policy_error_category(detail), "detail": detail}),
                tool=name, attempted=False, budget_consumed=False)
        except (TimeoutError, socket.timeout, subprocess.TimeoutExpired):
            return normalize_tool_result(self.observations.add(
                {"tool": name, "status": "error", "error": "connection_timeout",
                 "error_category": "connection_timeout",
                 "detail": "The bounded operation timed out before a complete response was received."}),
                tool=name, attempted=True)
        except OSError:
            return normalize_tool_result(self.observations.add(
                {"tool": name, "status": "error", "error": "connection_failure",
                 "error_category": "connection_failure",
                 "detail": "The bounded connection or protocol operation failed."}),
                tool=name, attempted=True)

    def _fingerprint_any(self, args: dict[str, Any]) -> dict[str, Any]:
        """Confirm a protocol only after a protocol-level handshake succeeds."""
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        if not self.live:
            return self.observations.add({"tool": "fingerprint_protocol", "status": "disabled", "error": "live_worker_disabled"})
        self._reserve_probe(ip); self._consume_probe()
        if port == 4840:
            try:
                from asyncua import Client
                async def handshake() -> str:
                    async with Client(f"opc.tcp://{ip}:{port}/scadaver/runtime/") as client:
                        return str(await client.get_namespace_array())
                import asyncio
                namespaces = asyncio.run(handshake())
                observation = {"tool": "fingerprint_protocol", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "opc-ua", "protocol_evidence": "opc_ua_session_established"}
                if self.allow_semantic_metadata:
                    observation["namespace_metadata"] = namespaces[:512]
                return self.observations.add(observation)
            except Exception:
                return self.observations.add({
                    "tool": "fingerprint_protocol", "status": "error", "target": {"ip": ip, "port": port},
                    "protocol": "unknown", "error": "opcua_handshake_failed",
                    "detail": "No OPC UA session was established. The TCP service may be unavailable or may not speak OPC UA.",
                })
        if port == 1102:
            try:
                import snap7
                client = snap7.client.Client(); client.connect(ip, 0, 1, tcp_port=port)
                connected = client.get_connected(); client.disconnect()
                return self.observations.add({"tool": "fingerprint_protocol", "status": "ok" if connected else "error", "target": {"ip": ip, "port": port}, "protocol": "s7comm" if connected else "unknown", "protocol_evidence": "s7_iso_tcp_session_established" if connected else "no_s7_session"})
            except Exception:
                return self.observations.add({
                    "tool": "fingerprint_protocol", "status": "error", "target": {"ip": ip, "port": port},
                    "protocol": "unknown", "error": "s7_handshake_failed",
                    "detail": "No S7 session was established. The TCP service may be unavailable or may not speak S7.",
                })
        if port == 8080:
            return self._fingerprint_http(ip, port)
        return self._scadaver(
            "fingerprint_protocol",
            [self.scadaver, "scan", "--safe-mode", "--live", "-i", ip, "-p", str(port), "--protocol", "modbus"],
            ip, port,
        )

    def _fingerprint_http(self, ip: str, port: int) -> dict[str, Any]:
        """Confirm only that an HTTP response exists; never infer product identity."""
        connection = http.client.HTTPConnection(ip, port, timeout=self.connect_timeout)
        try:
            connection.request("HEAD", "/", headers={
                "User-Agent": "ICS-ReadOnly-Assessment/1.0", "Connection": "close",
            })
            response = connection.getresponse()
            headers = self._safe_http_headers(response)
            location = headers.get("location")
            return self.observations.add({
                "tool": "fingerprint_protocol", "status": "ok",
                "target": {"ip": ip, "port": port}, "protocol": "http",
                "protocol_evidence": "http_response_received", "http_status": response.status,
                "redirect_received": response.status in {301, 302, 303, 307, 308},
                "location": location, "redirect_followed": False,
                "detail": "An HTTP response was received. Product and runtime identity remain unconfirmed.",
            })
        finally:
            connection.close()

    def _consume_probe(self) -> None:
        if self.probe_operations >= self.max_probe_operations:
            raise PolicyError("probe_budget_exhausted")
        self.probe_operations += 1

    def _value_read(self, register_type: str, args: dict[str, Any], *, reserve: bool = True, record: bool = True) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        start, count = self.policy.register_range(args.get("start"), args.get("count"))
        if reserve:
            self._reserve_probe(ip)
            self._consume_probe()
        command_kind = "register" if register_type == "holding" else "coil"
        collected_at = time.time()
        result = self.runner([self.scadaver, "get", "--safe-mode", "--live", "-i", ip, "-p", str(port), "--protocol", "modbus", command_kind, str(start), str(count)], self.cli_timeout)
        self._log_raw({"tool": f"read_modbus_{register_type}", "target": {"ip": ip, "port": port}, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        values = parse_scadaver_values(result.stdout, register_type, count) if result.returncode == 0 else None
        if values is None:
            failure_text = f"{result.stdout}\n{result.stderr}".lower()
            error = "decoded_values_unavailable" if result.returncode == 0 else "modbus_exception" if "exception" in failure_text else "command_failed"
            value = {"tool": f"read_modbus_{register_type}", "status": "unsupported" if result.returncode == 0 else "error", "target": {"ip": ip, "port": port}, "register_type": register_type, "start": start, "count": count, "values": None, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(collected_at)), "collected_at_unix": collected_at, "error": error}
            return self.observations.add(value) if record else value
        value = {"tool": f"read_modbus_{register_type}", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "modbus-tcp", "register_type": register_type, "start": start, "count": count, "values": values, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(collected_at)), "collected_at_unix": collected_at}
        return self.observations.add(value) if record else value

    def _sample_series(self, args: dict[str, Any]) -> dict[str, Any]:
        register_type = args.get("register_type")
        if register_type not in {"holding", "input", "coil", "discrete"}:
            raise PolicyError("register_type must be holding, input, coil, or discrete")
        try:
            samples, interval = int(args.get("samples")), float(args.get("interval_seconds"))
        except (TypeError, ValueError) as exc:
            raise PolicyError("samples and interval_seconds are required numeric values") from exc
        if not 1 <= samples <= self.max_samples or interval < self.min_sample_interval or interval * (samples - 1) > self.max_series_seconds:
            raise PolicyError("sampling limits exceeded")
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        start, count = self.policy.register_range(args.get("start"), args.get("count"))
        series = []
        for index in range(samples):
            if index:
                time.sleep(interval)
            reading = self._read_data_model(register_type, {"ip": ip, "port": port, "start": start, "count": count}, record=False)
            series.append({"sample_index": index, "timestamp_utc": reading.get("timestamp_utc"),
                           "collected_at_unix": reading.get("collected_at_unix"),
                           "status": reading["status"], "values": reading.get("values"),
                           "error": reading.get("error")})
        return self.observations.add({"tool": "sample_modbus_series", "status": "ok" if any(sample["status"] == "ok" for sample in series) else "error", "target": {"ip": ip, "port": port}, "register_type": register_type, "start": start, "count": count, "requested_interval_seconds": interval, "sampling_mode": "sequential_bounded", "samples": series})

    def _native_modbus_read(self, data_model: str, args: dict[str, Any], *, reserve: bool = True, record: bool = True) -> dict[str, Any]:
        from pymodbus.client import ModbusTcpClient
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        start, count = self.policy.register_range(args.get("start"), args.get("count"))
        if reserve: self._reserve_probe(ip); self._consume_probe()
        client = ModbusTcpClient(ip, port=port, timeout=self.connect_timeout)
        collected_at = time.time()
        try:
            if not client.connect(): raise OSError("modbus_connect_failed")
            method_name = {"holding": "read_holding_registers", "input": "read_input_registers", "coil": "read_coils", "discrete": "read_discrete_inputs"}[data_model]
            response = getattr(client, method_name)(start, count=count)
            tool = {
                "holding": "read_modbus_holding",
                "input": "read_modbus_input_registers",
                "coil": "read_modbus_coils",
                "discrete": "read_modbus_discrete_inputs",
            }[data_model]
            if response.isError():
                value = {"tool": tool, "status": "unsupported", "target": {"ip": ip, "port": port}, "data_model": data_model, "start": start, "count": count, "values": None, "error": "modbus_exception", "detail": str(response)[:256]}
            else:
                values = list(response.bits[:count]) if data_model in {"coil", "discrete"} else list(response.registers[:count])
                value = {"tool": tool, "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "modbus-tcp", "data_model": data_model, "start": start, "count": count, "values": values, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(collected_at)), "collected_at_unix": collected_at}
            return self.observations.add(value) if record else value
        finally:
            client.close()

    def _read_data_model(self, data_model: str, args: dict[str, Any], *, record: bool = True) -> dict[str, Any]:
        if data_model in {"holding", "coil"}:
            return self._value_read(data_model, args, record=record)
        return self._native_modbus_read(data_model, args, record=record)

    def _explore_modbus_map(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        start, count = self.policy.register_range(args.get("start"), args.get("count"))
        self._reserve_probe(ip)
        results = []
        for model in ("holding", "input", "coil", "discrete"):
            self._consume_probe()
            try: results.append(self._native_modbus_read(model, {"ip": ip, "port": port, "start": start, "count": count}, reserve=False, record=False))
            except (OSError, TimeoutError) as exc: results.append({"data_model": model, "status": "error", "values": None, "error": type(exc).__name__})
        return self.observations.add({"tool": "explore_modbus_map", "status": "ok" if any(item["status"] == "ok" for item in results) else "error", "target": {"ip": ip, "port": port}, "start": start, "count": count, "results": results})

    def _sample_cross_device_series(self, args: dict[str, Any]) -> dict[str, Any]:
        """Collect ordered reads without claiming that they are simultaneous."""
        targets = args.get("targets")
        if not isinstance(targets, list) or not 2 <= len(targets) <= 4:
            raise PolicyError("targets must contain two through four endpoints")
        try:
            samples, interval = int(args.get("samples")), float(args.get("interval_seconds"))
        except (TypeError, ValueError) as exc:
            raise PolicyError("samples and interval_seconds are required numeric values") from exc
        if not 1 <= samples <= self.max_samples or interval < self.min_sample_interval or interval * (samples - 1) > self.max_series_seconds:
            raise PolicyError("sampling limits exceeded")

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str, int, int]] = set()
        for target in targets:
            if not isinstance(target, dict):
                raise PolicyError("each target must be an object")
            register_type = target.get("register_type")
            if register_type not in {"holding", "input", "coil", "discrete"}:
                raise PolicyError("target register_type must be holding, input, coil, or discrete")
            ip, port = self.policy.target_ip(target.get("ip")), self.policy.port(target.get("port"))
            start, count = self.policy.register_range(target.get("start"), target.get("count"))
            key = (ip, port, register_type, start, count)
            if key in seen:
                raise PolicyError("targets must be distinct")
            seen.add(key)
            normalized.append({"ip": ip, "port": port, "register_type": register_type, "start": start, "count": count})

        series: list[dict[str, Any]] = []
        for index in range(samples):
            if index:
                time.sleep(interval)
            sweep_started = time.time()
            readings = []
            for target in normalized:
                reading = self._read_data_model(target["register_type"], target, record=False)
                readings.append({
                    "target": {key: target[key] for key in ("ip", "port")},
                    "register_type": target["register_type"], "start": target["start"], "count": target["count"],
                    "timestamp_utc": reading.get("timestamp_utc"), "status": reading["status"],
                    "collected_at_unix": reading.get("collected_at_unix"),
                    "values": reading.get("values"), "error": reading.get("error"),
                })
            sweep_finished = time.time()
            series.append({
                "sweep_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sweep_started)),
                "sweep_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sweep_finished)),
                "acquisition_skew_milliseconds": round((sweep_finished - sweep_started) * 1000, 3),
                "readings": readings,
            })
        return self.observations.add({
            "tool": "sample_cross_device_series",
            "status": "ok" if any(reading["status"] == "ok" for sweep in series for reading in sweep["readings"]) else "error",
            "targets": [{key: target[key] for key in ("ip", "port", "register_type", "start", "count")} for target in normalized],
            "samples": series,
            "requested_interval_seconds": interval,
            "sampling_mode": "sequential_cross_device_sweeps",
            "sampling_note": "Targets are sampled sequentially within each sweep; timestamps and measured skew are supplied.",
        })

    def _probe_identity(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        self._reserve_probe(ip)
        self._consume_probe()
        result = self.runner([self.scadaver, "get", "--safe-mode", "--live", "-i", ip, "-p", str(port), "--protocol", "modbus", "info"], self.cli_timeout)
        self._log_raw({"tool": "probe_device_identity", "target": {"ip": ip, "port": port}, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        fields = parse_identity_fields(result.stdout) if result.returncode == 0 else {}
        return self.observations.add({
            "tool": "probe_device_identity", "status": "ok" if fields else "unsupported",
            "target": {"ip": ip, "port": port}, "identity_information": fields or None,
            "provenance": "self-reported fields returned by SCADAver get info",
            "reason": None if fields else "modbus_identity_unsupported_or_incompatible",
            "detail": None if fields else "This operation only understands Modbus identity responses. Fingerprint the endpoint before retrying or use a protocol-compatible tool.",
        })

    def _inspect_http(self, args: dict[str, Any]) -> dict[str, Any]:
        return self._inspect_http_resource({**args, "path": "/", "method": "GET", "max_redirects": 0}, tool="inspect_http_service")

    def _inspect_http_resource(self, args: dict[str, Any], *, tool: str = "inspect_http_resource") -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        if port != 8080:
            raise PolicyError("HTTP inspection is limited to the approved HTTP port")
        path, method, redirects = args.get("path"), args.get("method"), args.get("max_redirects")
        if not isinstance(path, str) or not path.startswith("/") or "//" in path or len(path) > 256:
            raise PolicyError("HTTP path must be a bounded absolute same-host path")
        if method not in {"GET", "HEAD"} or not isinstance(redirects, int) or not 0 <= redirects <= 3:
            raise PolicyError("HTTP method or redirect limit is invalid")
        self._reserve_probe(ip)
        self._consume_probe()
        visited, hops, current = [], [], path
        for _ in range(redirects + 1):
            connection = http.client.HTTPConnection(ip, port, timeout=self.connect_timeout)
            try:
                connection.request(method, current, headers={"User-Agent": "ICS-ReadOnly-Assessment/1.0", "Connection": "close"})
                response = connection.getresponse(); raw_body = response.read(4096) if method == "GET" else b""
                headers = self._safe_http_headers(response)
                location = headers.get("location")
                hop: dict[str, Any] = {"path": current, "status": response.status}
                if response.status in {301, 302, 303, 307, 308} and location:
                    candidate = urlparse(urljoin(f"http://{ip}:{port}{current}", location))
                    location_kind = "absolute" if urlparse(location).scheme else "relative"
                    hop.update({"location": location, "location_kind": location_kind})
                    visited.append(hop)
                    if candidate.scheme not in {"http", ""}:
                        return self.observations.add({"tool": tool, "status": "unsupported", "target": {"ip": ip, "port": port}, "http_status": response.status, "location": location, "location_kind": location_kind, "error": "unsupported_redirect_destination", "error_category": "protocol_unsupported", "detail": "Redirect destination uses an unsupported scheme.", "redirect_chain": visited, "redirect_followed": False})
                    if candidate.hostname not in {None, ip} or candidate.port not in {None, port} or not candidate.path.startswith("/"):
                        return self.observations.add({"tool": tool, "status": "unsupported", "target": {"ip": ip, "port": port}, "http_status": response.status, "location": location, "location_kind": location_kind, "error": "cross_host_redirect_blocked", "error_category": "policy_rejected", "detail": "Redirect target is outside the requested host or port.", "redirect_chain": visited, "redirect_followed": False})
                    next_path = candidate.path + (f"?{candidate.query}" if candidate.query else "")
                    if any(item.get("path") == next_path for item in visited):
                        return self.observations.add({"tool": tool, "status": "error", "target": {"ip": ip, "port": port}, "http_status": response.status, "location": location, "location_kind": location_kind, "error": "redirect_loop", "error_category": "redirect_loop", "detail": "The same-host redirect chain loops to an already visited path.", "redirect_chain": visited, "redirect_followed": False})
                    if len(hops) >= redirects:
                        status = "redirect" if tool == "inspect_http_service" and redirects == 0 else "error"
                        return self.observations.add({"tool": tool, "status": status, "target": {"ip": ip, "port": port}, "http_status": response.status, "location": location, "location_kind": location_kind, "error": "redirect_not_followed" if status == "redirect" else "redirect_limit_reached", "error_category": "redirect_received" if status == "redirect" else "redirect_limit_reached", "detail": "A redirect was received but not followed by this bounded request." if status == "redirect" else "The bounded same-host redirect allowance was exhausted.", "redirect_chain": visited, "redirect_followed": False})
                    hops.append(location); current = next_path; continue
                visited.append(hop)
                if response.status in {301, 302, 303, 307, 308}:
                    return self.observations.add({"tool": tool, "status": "unsupported", "target": {"ip": ip, "port": port}, "http_status": response.status, "error": "redirect_missing_location", "error_category": "protocol_unsupported", "detail": "The HTTP redirect response did not include a usable Location header.", "redirect_chain": visited, "redirect_followed": False})
                body = raw_body.decode("utf-8", "replace"); title = re.search(r"<title[^>]*>\s*(.{0,200}?)\s*</title>", body, re.IGNORECASE | re.DOTALL)
                links = _visible_links(body)
                return self.observations.add({"tool": tool, "status": "authentication_required" if response.status in {401, 403} else "ok", "target": {"ip": ip, "port": port}, "request_path": path, "final_path": current, "http_status": response.status, "headers": headers, "page_title": title.group(1) if title else None, "visible_same_host_links": links, "redirect_chain": visited, "redirect_count": len(hops), "redirect_followed": bool(hops), "body_excerpt": body[:1024], "visible_text_excerpt": _visible_text(body), "body_truncated": len(raw_body) >= 4096, "content_is_untrusted": True, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            finally:
                connection.close()
        return self.observations.add({"tool": tool, "status": "error", "target": {"ip": ip, "port": port}, "error": "redirect_limit_reached", "redirect_chain": visited})

    @staticmethod
    def _safe_http_headers(response: Any) -> dict[str, str]:
        allowed = {"server", "content-type", "content-length", "location", "www-authenticate"}
        return {
            key.lower(): value[:256]
            for key, value in response.getheaders()
            if key.lower() in allowed
        }

    def _browse_opcua(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        self._reserve_probe(ip); self._consume_probe()
        if port != 4840:
            raise PolicyError("OPC UA browsing is limited to the approved OPC UA port")
        try:
            from asyncua import Client
            async def browse() -> list[dict[str, str]]:
                async with Client(f"opc.tcp://{ip}:{port}/scadaver/runtime/") as client:
                    children = await client.nodes.objects.get_children()
                    result = []
                    for node in children:
                        result.append(await self._opcua_node_metadata(node))
                        # A single bounded child level exposes actual values
                        # without permitting recursive namespace crawling.
                        if (await node.read_browse_name()).Name == "RuntimeProcess":
                            for child in await node.get_children():
                                result.append(await self._opcua_node_metadata(child))
                    return result
            import asyncio
            nodes = asyncio.run(browse())
            return self.observations.add({"tool": "browse_opcua_nodes", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "opc-ua", "nodes": nodes})
        except Exception:
            return self.observations.add({
                "tool": "browse_opcua_nodes", "status": "error", "target": {"ip": ip, "port": port},
                "protocol": "unknown", "error": "opcua_browse_failed",
                "detail": "The endpoint did not complete the bounded OPC UA browse. Confirm OPC UA first or choose another compatible observation tool.",
            })

    def _read_opcua(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        node_ids = args.get("node_ids")
        if port != 4840 or not isinstance(node_ids, list) or not 1 <= len(node_ids) <= 10 or not all(isinstance(node, str) for node in node_ids):
            raise PolicyError("invalid bounded OPC UA read request")
        self._reserve_probe(ip); self._consume_probe()
        try:
            from asyncua import Client
            async def read() -> list[dict[str, Any]]:
                async with Client(f"opc.tcp://{ip}:{port}/scadaver/runtime/") as client:
                    result = []
                    for node_id in node_ids:
                        node = client.get_node(node_id)
                        result.append({"node_id": node_id, "value": await node.read_value()})
                    return result
            import asyncio
            values = asyncio.run(read())
            return self.observations.add({"tool": "read_opcua_values", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "opc-ua", "values": values})
        except Exception as exc:
            message = str(exc).lower()
            if "badattributeidinvalid" in message or "not a variable" in message:
                error, detail = (
                    "opcua_node_not_readable_value",
                    "At least one requested node is an object or method rather than a readable value. Browse nodes and select variable nodes only.",
                )
            else:
                error, detail = (
                    "opcua_value_read_failed",
                    "The bounded OPC UA value read did not complete. Confirm the endpoint and use node identifiers returned by the browse tool.",
                )
            return self.observations.add({"tool": "read_opcua_values", "status": "error", "target": {"ip": ip, "port": port}, "error": error, "detail": detail})

    def _read_s7(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        try:
            db, start, size = int(args.get("db")), int(args.get("start")), int(args.get("size"))
        except (TypeError, ValueError) as exc:
            raise PolicyError("S7 DB, start, and size must be integers") from exc
        if port != 1102 or not 1 <= db <= 32 or start < 0 or not 1 <= size <= 32:
            raise PolicyError("S7 read is outside the bounded observation interface")
        self._reserve_probe(ip); self._consume_probe()
        try:
            import snap7
            client = snap7.client.Client()
            client.connect(ip, 0, 1, tcp_port=port)
            if not client.get_connected():
                raise OSError("s7_connect_failed")
            data = list(client.db_read(db, start, size))
            client.disconnect()
            return self.observations.add({"tool": "read_s7_data", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "s7comm", "db": db, "start": start, "bytes": data})
        except Exception as exc:
            return self.observations.add({"tool": "read_s7_data", "status": "error", "target": {"ip": ip, "port": port}, "protocol": "unknown", "error": type(exc).__name__})

    async def _opcua_node_metadata(self, node: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"node_id": node.nodeid.to_string(), "browse_name": (await node.read_browse_name()).Name, "node_class": str(await node.read_node_class()).split(".")[-1]}
        if item["node_class"].lower() == "variable":
            try: item["data_type"] = (await node.read_data_type()).to_string()
            except Exception: item["data_type"] = None
        return item

    def _inspect_s7_metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        ip, port = self.policy.target_ip(args.get("ip")), self.policy.port(args.get("port"))
        if port != 1102: raise PolicyError("S7 metadata inspection requires the approved S7 port")
        self._reserve_probe(ip); self._consume_probe()
        try:
            import snap7
            client = snap7.client.Client(); client.connect(ip, 0, 1, tcp_port=port)
            metadata: dict[str, Any] = {}
            for name in ("get_cpu_info", "get_cp_info", "get_order_code"):
                try: metadata[name.removeprefix("get_")] = str(getattr(client, name)())[:512]
                except Exception as exc: metadata[name.removeprefix("get_")] = {"unavailable": type(exc).__name__}
            client.disconnect()
            return self.observations.add({"tool": "inspect_s7_metadata", "status": "ok", "target": {"ip": ip, "port": port}, "protocol": "s7comm", "self_reported_metadata": metadata, "hardware_authenticity_not_established": True})
        except Exception as exc:
            return self.observations.add({"tool": "inspect_s7_metadata", "status": "error", "target": {"ip": ip, "port": port}, "error": type(exc).__name__})

    def _log_raw(self, value: dict[str, Any]) -> None:
        if self.raw_logger is not None:
            self.raw_logger(value)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.links: list[str] = []
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href and href.startswith("/") and href not in self.links:
                self.links.append(href[:256])


def _visible_links(body: str) -> list[str]:
    parser = _LinkParser(); parser.feed(body); return parser.links[:20]


def _visible_text(body: str) -> str:
    """Bound visible page text without scripts/styles or hidden raw markup."""
    stripped = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", stripped)).strip()[:1024]
