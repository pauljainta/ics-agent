from __future__ import annotations

import argparse
import json
import os
import socket
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .agents import AgenticRoleAgent, RuleBasedRoleAgent, ScriptedRoleAgent
from .environment import EnvironmentError, OpenPLCEnvironment
from .types import ScenarioDefinition
from .utils import utc_now


def _create_agent(profile_name: str, profile: dict[str, Any]):
    family = str(profile.get("family", "scripted")).strip().lower()
    if family == "agentic":
        return AgenticRoleAgent(profile_name, dict(profile))
    if family == "rule_based":
        return RuleBasedRoleAgent(profile_name, dict(profile))
    return ScriptedRoleAgent(profile_name, dict(profile))


def _scenario_from_payload(payload: dict[str, Any]) -> ScenarioDefinition:
    return ScenarioDefinition(
        catalog_id=str(payload["catalog_id"]),
        suite=str(payload["suite"]),
        scenario_id=str(payload["scenario_id"]),
        execution_path=Path(str(payload["execution_path"])),
        execution=dict(payload.get("execution") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )


def _agentic_capabilities(profile: dict[str, Any]) -> dict[str, Any]:
    llm = dict(profile.get("llm") or {})
    require_cuda = bool(llm.get("require_cuda", True))
    libraries: dict[str, bool] = {}
    for module_name in (
        "torch",
        "transformers",
        "langchain",
        "langchain_core",
        "langchain_huggingface",
        "langgraph",
    ):
        try:
            __import__(module_name)
            libraries[module_name] = True
        except Exception:
            libraries[module_name] = False

    cuda_available = False
    if libraries.get("torch"):
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    supported = all(libraries.values()) and (cuda_available or not require_cuda)
    reasons: list[str] = []
    missing = sorted(name for name, available in libraries.items() if not available)
    if missing:
        reasons.append("missing libraries: " + ", ".join(missing))
    if require_cuda and not cuda_available:
        reasons.append("CUDA GPU unavailable on remote node")
    return {
        "family": str(profile.get("family", "scripted")),
        "require_cuda": require_cuda,
        "cuda_available": cuda_available,
        "libraries": libraries,
        "supported": supported,
        "reasons": reasons,
    }


class RemoteRoleService:
    def __init__(self, env_config: dict[str, Any], *, node_label: str | None = None) -> None:
        self.env = OpenPLCEnvironment(env_config)
        self.agent: Any = None
        self.role: str | None = None
        self.profile_name: str | None = None
        self.profile: dict[str, Any] | None = None
        self.node_label = node_label or socket.gethostname()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.started_utc = utc_now()
        self.capabilities: dict[str, Any] = {}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self.runtime_info(),
            "configured": self.agent is not None,
            "capabilities": self.capabilities,
        }

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = dict(payload.get("profile") or {})
        profile_name = str(payload.get("profile_name") or "remote_profile")
        scenario_payload = dict(payload.get("scenario") or {})
        if not profile:
            raise RuntimeError("remote role configure request is missing `profile`")
        if not scenario_payload:
            raise RuntimeError("remote role configure request is missing `scenario`")

        role = str(payload.get("role") or profile.get("role") or "").strip()
        if not role:
            raise RuntimeError("remote role configure request is missing `role`")

        family = str(profile.get("family", "scripted")).strip().lower()
        self.capabilities = (
            _agentic_capabilities(profile) if family == "agentic" else {"supported": True}
        )
        if family == "agentic" and not self.capabilities.get("supported"):
            reasons = ", ".join(self.capabilities.get("reasons") or ["unsupported"])
            raise RuntimeError(
                f"agentic remote execution is unavailable on `{self.hostname}`: {reasons}"
            )

        scenario = _scenario_from_payload(scenario_payload)
        agent = _create_agent(profile_name, profile)
        agent.reset(scenario)

        self.agent = agent
        self.role = role
        self.profile_name = profile_name
        self.profile = profile

        return {
            "status": "ok",
            "runtime": self.runtime_info(),
            "capabilities": self.capabilities,
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.agent is None:
            raise RuntimeError("remote role runtime is not configured")
        context = dict(payload.get("context") or {})
        scenario_payload = context.get("scenario")
        if isinstance(scenario_payload, dict) and "catalog_id" in scenario_payload:
            context["scenario"] = _scenario_from_payload(scenario_payload)
        decisions = []
        for decision in self.agent.step(context):
            rendered = decision.as_dict()
            metadata = dict(rendered.get("metadata") or {})
            metadata.update(
                {
                    "execution_residency": "remote",
                    "runtime_host": self.hostname,
                    "runtime_pid": self.pid,
                    "assigned_node": self.node_label,
                }
            )
            rendered["metadata"] = metadata
            decisions.append(rendered)
        return {
            "status": "ok",
            "runtime": self.runtime_info(),
            "decisions": decisions,
        }

    def apply_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        role = payload.get("role") or self.role
        result = self.env.act(action, role=role)
        return {
            "status": "ok",
            "runtime": self.runtime_info(),
            "result": result,
        }

    def observe(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": self.runtime_info(),
            "observation": self.env.observe_flat(),
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "pid": self.pid,
            "started_utc": self.started_utc,
            "role": self.role,
            "profile_name": self.profile_name,
            "family": None if self.profile is None else self.profile.get("family"),
            "execution_residency": "remote",
            "assigned_node": self.node_label,
            "capabilities": self.capabilities,
        }


def _write_json(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote ICS role runtime service")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18600)
    parser.add_argument("--controller-host", default="198.18.4.10")
    parser.add_argument("--controller-port", type=int, default=1502)
    parser.add_argument("--simulator-host", default="198.18.4.20")
    parser.add_argument("--simulator-port", type=int, default=1503)
    parser.add_argument("--node-label", default=None)
    args = parser.parse_args()

    service = RemoteRoleService(
        {
            "backend": "auto",
            "keep_running": True,
            "start_bridge": False,
            "build_if_missing": False,
            "controller_host": args.controller_host,
            "controller_port": args.controller_port,
            "simulator_host": args.simulator_host,
            "simulator_port": args.simulator_port,
        },
        node_label=args.node_label,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body.decode("utf-8"))
                if self.path == "/health":
                    _write_json(self, 200, service.health())
                    return
                if self.path == "/configure":
                    _write_json(self, 200, service.configure(payload))
                    return
                if self.path == "/step":
                    _write_json(self, 200, service.step(payload))
                    return
                if self.path == "/apply_action":
                    _write_json(self, 200, service.apply_action(payload))
                    return
                if self.path == "/observe":
                    _write_json(self, 200, service.observe())
                    return
                _write_json(
                    self,
                    404,
                    {"status": "error", "error": f"unknown endpoint `{self.path}`"},
                )
            except EnvironmentError as exc:
                _write_json(self, 500, {"status": "error", "error": str(exc)})
            except Exception as exc:
                _write_json(
                    self,
                    500,
                    {
                        "status": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    },
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(
                json.dumps(
                    {
                        "timestamp_utc": utc_now(),
                        "client": self.address_string(),
                        "message": format % args,
                    }
                )
            )

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
