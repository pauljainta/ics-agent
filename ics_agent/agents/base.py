from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import RoleDecision, ScenarioDefinition
from ..utils import utc_now


class RoleAgent(ABC):
    def __init__(self, profile_name: str, profile: dict[str, Any]) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.role = str(profile["role"])
        self.mode = str(profile.get("mode", "online"))
        self._state: dict[str, Any] = {}

    def reset(self, scenario: ScenarioDefinition) -> None:
        self._state = {
            "scenario_id": scenario.catalog_id,
            "emitted_notes": set(),
            "triggered_rules": set(),
            "cursor": 0,
        }

    def decision(
        self,
        *,
        decision_type: str,
        summary: str,
        time_sec: float,
        action: dict[str, Any] | None = None,
        confidence: float | None = None,
        diagnosis: str | None = None,
        labels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RoleDecision:
        return RoleDecision(
            role=self.role,
            agent_name=self.profile_name,
            decision_type=decision_type,
            summary=summary,
            time_sec=time_sec,
            timestamp_utc=utc_now(),
            action=action,
            confidence=confidence,
            diagnosis=diagnosis,
            labels=labels or [],
            metadata=metadata or {},
        )

    @abstractmethod
    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        raise NotImplementedError

    def finalize(self, context: dict[str, Any]) -> list[RoleDecision]:
        return []
