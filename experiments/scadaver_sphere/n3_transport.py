"""Transport for the isolated N3 worker service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class N3TransportError(RuntimeError):
    pass


@dataclass
class N3Transport:
    base_url: str
    timeout: float = 75.0
    session_id: str | None = None

    def start_episode(self, required_tool_names: set[str], *, autonomous_actions: bool = False) -> dict[str, Any]:
        response = self._request("/session", {"autonomous_actions": autonomous_actions})
        actual = response.get("tool_names")
        if response.get("harness_profile") != "heterogeneous_n3":
            raise N3TransportError("N3 worker profile mismatch")
        if not isinstance(actual, list) or not required_tool_names.issubset(set(actual)):
            raise N3TransportError("N3 worker tool surface mismatch")
        self.session_id = response["session_id"]
        return response

    def close_episode(self) -> None:
        if self.session_id:
            self._request("/session/close", {"session_id": self.session_id})
            self.session_id = None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.session_id:
            raise N3TransportError("N3 session is not active")
        return self._request("/tool", {"session_id": self.session_id,
                                        "name": name, "arguments": arguments})

    def execute_action(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.session_id:
            raise N3TransportError("N3 session is not active")
        return self._request("/action", {"session_id": self.session_id,
                                          "arguments": arguments})

    def operator_check(self, operation: str) -> dict[str, Any]:
        if operation not in {"status", "verify", "baseline"}:
            raise N3TransportError("unsupported operator check")
        return self._request("/operator", {"operation": operation})

    def _request(self, suffix: str, body: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(self.base_url.rstrip("/") + suffix,
                          data=json.dumps(body).encode(), method="POST", headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise N3TransportError("N3 worker unavailable") from exc
        if not isinstance(value, dict):
            raise N3TransportError("N3 worker returned invalid JSON")
        return value
