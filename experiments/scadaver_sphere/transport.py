"""External orchestrator transport for the loopback-only access worker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class WorkerTransportError(RuntimeError):
    pass


@dataclass
class AccessWorkerTransport:
    """Calls an access-node worker through an administrator-created SSH tunnel."""

    base_url: str
    timeout: float = 45.0
    session_id: str | None = None

    def start_episode(self, profile: str = "fingerprinting_basic", *, required_tool_names: set[str] | None = None) -> dict[str, Any]:
        response = self._request("/session", {"harness_profile": profile})
        if response.get("harness_profile") != profile:
            raise WorkerTransportError("access worker did not confirm the requested harness profile")
        if required_tool_names is not None:
            actual = response.get("tool_names")
            if not isinstance(actual, list) or not required_tool_names.issubset(set(actual)):
                raise WorkerTransportError("access worker did not confirm the required tool surface")
        self.session_id = response["session_id"]
        return response

    def close_episode(self) -> None:
        if self.session_id is not None:
            self._request("/session/close", {"session_id": self.session_id})
            self.session_id = None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.session_id is None:
            raise WorkerTransportError("episode session has not been started")
        return self._request("/tool", {"session_id": self.session_id, "name": name, "arguments": arguments})

    def _request(self, suffix: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode()
        request = Request(
            self.base_url.rstrip("/") + suffix,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise WorkerTransportError("access worker unavailable") from exc
        if not isinstance(result, dict):
            raise WorkerTransportError("access worker returned invalid JSON")
        return result
