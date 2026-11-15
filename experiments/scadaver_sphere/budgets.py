"""Global episode budgets for continuous SCADAver investigations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EpisodeLimits:
    max_model_calls: int = 80
    max_tool_calls: int = 160
    max_input_tokens: int = 1_000_000
    max_output_tokens: int = 120_000
    max_network_observation_seconds: float = 900.0
    max_time_series_samples: int = 200
    max_api_cost_usd: float = 25.0
    reserved_completion_steps: int = 3

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(value < 0 for value in numeric.values()):
            raise ValueError("episode limits must be nonnegative")
        if self.max_model_calls < 1 or self.max_tool_calls < 1:
            raise ValueError("model and tool-call limits must be positive")


@dataclass
class EpisodeBudget:
    limits: EpisodeLimits
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    network_observation_seconds: float = 0.0
    time_series_samples: int = 0
    api_cost_usd: float = 0.0
    exhausted: list[str] = field(default_factory=list)
    _usage_records_seen: int = 0

    def can_model_call(self) -> bool:
        return self.model_calls < self.limits.max_model_calls

    def record_model_call(self) -> None:
        self.model_calls += 1
        self._mark("model_calls", self.model_calls > self.limits.max_model_calls)

    def ingest_request_records(self, records: list[dict[str, Any]]) -> None:
        for record in records[self._usage_records_seen:]:
            usage = record.get("usage") or {}
            self.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            cost = usage.get("cost", record.get("cost"))
            if isinstance(cost, (int, float)):
                self.api_cost_usd += float(cost)
        self._usage_records_seen = len(records)
        self._mark("input_tokens", self.input_tokens > self.limits.max_input_tokens)
        self._mark("output_tokens", self.output_tokens > self.limits.max_output_tokens)
        self._mark("api_cost_usd", self.api_cost_usd > self.limits.max_api_cost_usd)

    def check_tool(self, name: str, arguments: dict[str, Any]) -> str | None:
        if self.tool_calls >= self.limits.max_tool_calls:
            return "tool_calls"
        samples = int(arguments.get("samples", 0)) if name in {"sample_modbus_series", "sample_cross_device_series"} else 0
        interval = float(arguments.get("interval_seconds", 0)) if samples else 0.0
        observation_seconds = max(0.0, interval * max(0, samples - 1))
        if self.time_series_samples + samples > self.limits.max_time_series_samples:
            return "time_series_samples"
        if self.network_observation_seconds + observation_seconds > self.limits.max_network_observation_seconds:
            return "network_observation_seconds"
        return None

    def record_tool(self, name: str, arguments: dict[str, Any]) -> None:
        self.tool_calls += 1
        if name in {"sample_modbus_series", "sample_cross_device_series"}:
            samples = int(arguments.get("samples", 0))
            self.time_series_samples += samples
            self.network_observation_seconds += float(arguments.get("interval_seconds", 0)) * max(0, samples - 1)

    def should_finalize(self) -> bool:
        return bool(self.exhausted) or not self.can_model_call()

    def remaining(self) -> dict[str, int | float]:
        """Return a secret-free snapshot of capacity before the next request."""
        return {
            "model_calls": max(0, self.limits.max_model_calls - self.model_calls),
            "tool_calls": max(0, self.limits.max_tool_calls - self.tool_calls),
            "input_tokens": max(0, self.limits.max_input_tokens - self.input_tokens),
            "output_tokens": max(0, self.limits.max_output_tokens - self.output_tokens),
            "network_observation_seconds": max(
                0.0, self.limits.max_network_observation_seconds - self.network_observation_seconds
            ),
            "time_series_samples": max(
                0, self.limits.max_time_series_samples - self.time_series_samples
            ),
            "api_cost_usd": max(0.0, self.limits.max_api_cost_usd - self.api_cost_usd),
        }

    def snapshot(self) -> dict[str, Any]:
        return {"limits": asdict(self.limits), "usage": {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "network_observation_seconds": round(self.network_observation_seconds, 3),
            "time_series_samples": self.time_series_samples,
            "api_cost_usd": round(self.api_cost_usd, 8),
        }, "exhausted": list(self.exhausted)}

    def _mark(self, name: str, condition: bool) -> None:
        if condition and name not in self.exhausted:
            self.exhausted.append(name)
