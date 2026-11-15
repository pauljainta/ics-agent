"""Configuration loading and safe defaults for the SCADAver experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: str
    allowed_cidr: str
    permitted_ports: tuple[int, ...]
    model: str
    api_key_env: str
    max_steps: int = 40
    max_hosts: int = 8
    max_observations: int = 64
    live_openrouter: bool = False
    live_worker: bool = False
    max_output_tokens: int = 1536
    final_submission_max_output_tokens: int = 2048
    request_timeout_seconds: int = 240
    inference_retries: int = 1
    episode_deadline_seconds: int = 900
    harness_profile: str = "fingerprinting_basic"
    max_model_calls: int = 80
    max_tool_calls: int = 160
    max_input_tokens: int = 1_000_000
    max_total_output_tokens: int = 120_000
    max_network_observation_seconds: float = 900.0
    max_time_series_samples: int = 200
    max_api_cost_usd: float = 25.0
    reserved_completion_steps: int = 3
    max_context_chars: int = 600_000
    context_tail_groups: int = 24
    max_action_duration_seconds: float = 5.0
    max_actions: int = 1

    @classmethod
    def from_json(cls, path: Path) -> "ScenarioConfig":
        raw: dict[str, Any] = json.loads(path.read_text())
        profile = str(raw.get("harness_profile", "fingerprinting_basic"))
        if profile not in {"fingerprinting_basic", "fingerprinting_enhanced", "fingerprinting_runtime_process", "heterogeneous_runtime_process", "heterogeneous_e2", "heterogeneous_e3_telemetry", "heterogeneous_e3_metadata", "heterogeneous_e4"}:
            raise ValueError("harness_profile is not supported")
        max_output_tokens = int(raw.get("max_output_tokens", 1536))
        final_submission_max_output_tokens = int(raw.get("final_submission_max_output_tokens", 2048))
        if final_submission_max_output_tokens < max_output_tokens:
            raise ValueError("final_submission_max_output_tokens must be at least max_output_tokens")
        if float(raw.get("max_action_duration_seconds", 5.0)) <= 0:
            raise ValueError("max_action_duration_seconds must be positive")
        if int(raw.get("max_actions", 1)) < 1:
            raise ValueError("max_actions must be positive")
        return cls(
            scenario_id=str(raw["scenario_id"]),
            allowed_cidr=str(raw["allowed_cidr"]),
            permitted_ports=tuple(int(port) for port in raw["permitted_ports"]),
            model=str(raw["model"]),
            api_key_env=str(raw.get("api_key_env", "OPENROUTER_API_KEY")),
            max_steps=int(raw.get("max_steps", 40)),
            max_hosts=int(raw.get("max_hosts", 8)),
            max_observations=int(raw.get("max_observations", 64)),
            live_openrouter=bool(raw.get("live_openrouter", False)),
            live_worker=bool(raw.get("live_worker", False)),
            max_output_tokens=max_output_tokens,
            final_submission_max_output_tokens=final_submission_max_output_tokens,
            request_timeout_seconds=int(raw.get("request_timeout_seconds", 240)),
            inference_retries=int(raw.get("inference_retries", 1)),
            episode_deadline_seconds=int(raw.get("episode_deadline_seconds", 900)),
            harness_profile=profile,
            max_model_calls=int(raw.get("max_model_calls", 80)),
            max_tool_calls=int(raw.get("max_tool_calls", 160)),
            max_input_tokens=int(raw.get("max_input_tokens", 1_000_000)),
            max_total_output_tokens=int(raw.get("max_total_output_tokens", 120_000)),
            max_network_observation_seconds=float(raw.get("max_network_observation_seconds", 900)),
            max_time_series_samples=int(raw.get("max_time_series_samples", 200)),
            max_api_cost_usd=float(raw.get("max_api_cost_usd", 25.0)),
            reserved_completion_steps=int(raw.get("reserved_completion_steps", 3)),
            max_context_chars=int(raw.get("max_context_chars", 600_000)),
            context_tail_groups=int(raw.get("context_tail_groups", 24)),
            max_action_duration_seconds=float(raw.get("max_action_duration_seconds", 5.0)),
            max_actions=int(raw.get("max_actions", 1)),
        )

    def public_snapshot(self) -> dict[str, Any]:
        """Return only information that may be shown to the agent or stored publicly."""
        return {
            "scenario_id": self.scenario_id,
            "allowed_cidr": self.allowed_cidr,
            "permitted_ports": list(self.permitted_ports),
            "model": self.model,
            "max_steps": self.max_steps,
            "max_hosts": self.max_hosts,
            "max_observations": self.max_observations,
            "max_output_tokens": self.max_output_tokens,
            "final_submission_max_output_tokens": self.final_submission_max_output_tokens,
            "request_timeout_seconds": self.request_timeout_seconds,
            "inference_retries": self.inference_retries,
            "episode_deadline_seconds": self.episode_deadline_seconds,
            "harness_profile": self.harness_profile,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_total_output_tokens": self.max_total_output_tokens,
            "max_network_observation_seconds": self.max_network_observation_seconds,
            "max_time_series_samples": self.max_time_series_samples,
            "max_api_cost_usd": self.max_api_cost_usd,
            "reserved_completion_steps": self.reserved_completion_steps,
            "max_context_chars": self.max_context_chars,
            "context_tail_groups": self.context_tail_groups,
            "max_action_duration_seconds": self.max_action_duration_seconds,
            "max_actions": self.max_actions,
        }
