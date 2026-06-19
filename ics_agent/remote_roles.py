from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .types import RoleDecision, ScenarioDefinition


class RemoteRoleError(RuntimeError):
    """Raised when a remote role runtime fails."""


def _scenario_payload(scenario: ScenarioDefinition) -> dict[str, Any]:
    return {
        "catalog_id": scenario.catalog_id,
        "suite": scenario.suite,
        "scenario_id": scenario.scenario_id,
        "execution_path": str(scenario.execution_path),
        "execution": scenario.execution,
        "metadata": scenario.metadata,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, ScenarioDefinition):
        return _scenario_payload(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


class RemoteRoleAgentProxy:
    """HTTP proxy for a role runtime executing on a remote SPHERE node."""

    def __init__(
        self,
        *,
        role: str,
        profile_name: str,
        profile: dict[str, Any],
        endpoint: str,
        assigned_node: str,
        request_timeout_sec: float = 15.0,
    ) -> None:
        self.role = role
        self.profile_name = profile_name
        self.profile = profile
        self.mode = str(profile.get("mode", "online"))
        self.endpoint = endpoint.rstrip("/")
        self.assigned_node = assigned_node
        self.request_timeout_sec = float(request_timeout_sec)
        self.runtime_info: dict[str, Any] = {}

    def reset(self, scenario: ScenarioDefinition) -> None:
        response = self._request(
            "/configure",
            {
                "role": self.role,
                "profile_name": self.profile_name,
                "profile": self.profile,
                "scenario": _scenario_payload(scenario),
            },
        )
        self.runtime_info = dict(response.get("runtime") or {})

    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        response = self._request("/step", {"context": _jsonable(context)})
        decisions: list[RoleDecision] = []
        for item in response.get("decisions", []):
            if not isinstance(item, dict):
                continue
            decisions.append(RoleDecision(**item))
        runtime = response.get("runtime")
        if isinstance(runtime, dict):
            self.runtime_info = dict(runtime)
        return decisions

    def finalize(self, context: dict[str, Any]) -> list[RoleDecision]:
        return []

    def apply_action(self, action: dict[str, Any] | str) -> dict[str, Any]:
        response = self._request("/apply_action", {"action": action})
        runtime = response.get("runtime")
        if isinstance(runtime, dict):
            self.runtime_info = dict(runtime)
        return dict(response.get("result") or {})

    def observe_flat(self) -> dict[str, Any]:
        response = self._request("/observe", {})
        runtime = response.get("runtime")
        if isinstance(runtime, dict):
            self.runtime_info = dict(runtime)
        return dict(response.get("observation") or {})

    def health(self) -> dict[str, Any]:
        return self._request("/health", {})

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_sec,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteRoleError(
                f"remote role `{self.role}` request `{path}` failed: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteRoleError(
                f"remote role `{self.role}` request `{path}` failed: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise RemoteRoleError(
                f"remote role `{self.role}` returned non-object payload for `{path}`"
            )
        if body.get("status") == "error":
            raise RemoteRoleError(
                str(body.get("error") or f"remote role `{self.role}` returned an error")
            )
        return body
