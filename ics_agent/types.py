from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScenarioDefinition:
    catalog_id: str
    suite: str
    scenario_id: str
    execution_path: Path
    execution: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class RoleDecision:
    role: str
    agent_name: str
    decision_type: str
    summary: str
    time_sec: float
    timestamp_utc: str
    action: dict[str, Any] | None = None
    confidence: float | None = None
    diagnosis: str | None = None
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "agent_name": self.agent_name,
            "decision_type": self.decision_type,
            "summary": self.summary,
            "time_sec": round(self.time_sec, 3),
            "timestamp_utc": self.timestamp_utc,
            "action": self.action,
            "confidence": self.confidence,
            "diagnosis": self.diagnosis,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass
class RunBundle:
    experiment_id: str
    condition_name: str
    run_id: str
    scenario: ScenarioDefinition
    run_dir: Path
    config_snapshot: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    system_events: list[dict[str, Any]] = field(default_factory=list)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    offline_analyses: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    passed: bool | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

