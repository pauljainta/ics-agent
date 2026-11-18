"""Separated transports for V2 investigation and controller-mediated action."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class V2TransportError(RuntimeError):
    pass


def _structured_http_error(exc: HTTPError, unavailable: str) -> dict[str, Any]:
    """Preserve explicit broker/worker rejections while failing closed otherwise."""
    try:
        value = json.loads(exc.read())
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as decode_error:
        raise V2TransportError(unavailable) from decode_error
    if not isinstance(value, dict) or value.get("status") not in {"error", "rejected"}:
        raise V2TransportError(unavailable) from exc
    return value


@dataclass
class V2ReadOnlyTransport:
    base_url: str
    timeout: float = 75.0
    session_id: str | None = None

    def start_episode(self, required_tool_names: set[str]) -> dict[str, Any]:
        value = self._request("/session", {})
        names = value.get("tool_names")
        if value.get("harness_profile") != "ot_realistic_v2":
            raise V2TransportError("V2 worker profile mismatch")
        if not isinstance(names, list) or not required_tool_names.issubset(set(names)):
            raise V2TransportError("V2 worker tool surface mismatch")
        self.session_id = str(value["session_id"])
        return value

    def close_episode(self) -> None:
        if self.session_id:
            self._request("/session/close", {"session_id": self.session_id})
            self.session_id = None

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.session_id:
            raise V2TransportError("V2 worker session is not active")
        return self._request("/tool", {"session_id": self.session_id,
                                       "name": name, "arguments": arguments})

    def _request(self, suffix: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(self.base_url.rstrip("/") + suffix,
                          data=json.dumps(body).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except HTTPError as exc:
            # Policy and resource-budget rejections are tool observations, not
            # evidence that the worker transport is unavailable.
            value = _structured_http_error(exc, "V2 read-only worker unavailable")
        except (URLError, TimeoutError) as exc:
            raise V2TransportError("V2 read-only worker unavailable") from exc
        if not isinstance(value, dict):
            raise V2TransportError("V2 worker returned invalid JSON")
        return value


@dataclass
class OperatorActionTransport:
    base_url: str
    authorization_token: str
    timeout: float = 30.0

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url.rstrip("/") + "/controller-action",
            data=json.dumps({"arguments": arguments}).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.authorization_token}"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise V2TransportError("operator action broker unavailable") from exc
        if not isinstance(value, dict):
            raise V2TransportError("operator action broker returned invalid JSON")
        return value


@dataclass
class StandingPolicyActionTransport:
    """Trusted local runner transport; capability secrets never reach the model."""

    base_url: str
    runner_token: str
    episode_id: str
    timeout: float = 30.0
    _capability: str | None = None

    @classmethod
    def for_episode(cls, base_url: str, runner_token: str,
                    episode_id: str | None = None) -> "StandingPolicyActionTransport":
        return cls(base_url, runner_token, episode_id or str(uuid.uuid4()))

    def policy_status(self) -> dict[str, Any]:
        return self._request("/policy-status", {})

    def available(self) -> bool:
        return self.policy_status().get("state") == "APPROVED_ACTIVE"

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._capability is None:
            issued = self._request(
                "/capability", {"episode_id": self.episode_id,
                                "operation": "bounded_controller_output"},
                bearer=self.runner_token,
            )
            capability = issued.get("capability")
            if issued.get("status") != "issued" or not isinstance(capability, str):
                return {
                    "tool": "bounded_controller_output",
                    **issued,
                    "transaction": {
                        "action_requested": True,
                        "action_authorized": False,
                        "action_accepted": False,
                        "action_applied": False,
                        "physical_response_observed": False,
                        "objective_achieved": False,
                    },
                }
            self._capability = capability
        value = self._request(
            "/controller-action", {"episode_id": self.episode_id,
                                   "capability": self._capability,
                                   "arguments": arguments},
        )
        self._capability = None
        return value

    def _request(self, suffix: str, body: dict[str, Any],
                 *, bearer: str | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        request = Request(self.base_url.rstrip("/") + suffix,
                          data=json.dumps(body).encode(), method="POST", headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except HTTPError as exc:
            value = _structured_http_error(
                exc, "standing-policy broker unavailable or denied")
        except (URLError, TimeoutError) as exc:
            raise V2TransportError("standing-policy broker unavailable or denied") from exc
        if not isinstance(value, dict):
            raise V2TransportError("standing-policy broker returned invalid JSON")
        return value
