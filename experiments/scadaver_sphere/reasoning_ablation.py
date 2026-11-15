"""Model-controlled reasoning scaffolds for the B0-B4 ablation study."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOOL_NAME = "record_reasoning_state"
FLAGS = (
    "evidence_ledger",
    "hypothesis_verification",
    "pre_action_review",
    "information_gain_selection",
)


@dataclass(frozen=True)
class ReasoningProfile:
    condition: str
    version: str
    base_config: str
    prompt_addition: str | None
    evidence_ledger: bool
    hypothesis_verification: bool
    pre_action_review: bool
    information_gain_selection: bool

    @classmethod
    def from_json(cls, path: Path) -> "ReasoningProfile":
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(
            condition=str(raw["condition"]),
            version=str(raw["version"]),
            base_config=str(raw["base_config"]),
            prompt_addition=raw.get("prompt_addition"),
            **{name: bool(raw.get(name, False)) for name in FLAGS},
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.condition not in {"B0", "B1", "B2", "B3", "B4"}:
            raise ValueError("reasoning condition must be B0-B4")
        enabled = [getattr(self, name) for name in FLAGS]
        expected = int(self.condition[1:])
        if enabled != [index < expected for index in range(4)]:
            raise ValueError("reasoning enhancements must be cumulative B0-B4")
        if self.condition == "B0" and self.prompt_addition is not None:
            raise ValueError("B0 must not add prompt text")
        if self.condition != "B0" and not self.prompt_addition:
            raise ValueError("enhanced conditions require a prompt addition")

    @property
    def enabled(self) -> bool:
        return self.evidence_ledger

    def public_snapshot(self) -> dict[str, Any]:
        return asdict(self)


def load_prompt(profile: ReasoningProfile, root: Path) -> str:
    if not profile.prompt_addition:
        return ""
    path = (root / profile.prompt_addition).resolve()
    expected_root = root.resolve()
    if expected_root not in path.parents:
        raise ValueError("prompt addition must remain under the experiment root")
    return path.read_text(encoding="utf-8").strip()


def reasoning_tool_schema(profile: ReasoningProfile) -> dict[str, Any] | None:
    if not profile.enabled:
        return None
    properties: dict[str, Any] = {
        "phase": {"type": "string"},
        "evidence_ledger": {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "observation_ids": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["supported", "contradicted", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": ["claim", "observation_ids", "status", "confidence"], "additionalProperties": False}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "next_observations": {"type": "array", "items": {"type": "string"}},
    }
    required = ["phase", "evidence_ledger", "unresolved_questions", "next_observations"]
    if profile.hypothesis_verification:
        properties["hypotheses"] = {"type": "array", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "supporting_observation_ids": {"type": "array", "items": {"type": "string"}},
            "contradicting_observation_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_needed": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "status": {"type": "string", "enum": ["active", "supported", "revised", "rejected"]},
        }, "required": ["claim", "supporting_observation_ids", "contradicting_observation_ids", "evidence_needed", "confidence", "status"], "additionalProperties": False}}
        required.append("hypotheses")
    if profile.pre_action_review:
        properties["pre_action_review"] = {"type": ["object", "null"], "properties": {
            "proposed_action": {"type": "string"},
            "supporting_observation_ids": {"type": "array", "items": {"type": "string"}},
            "expected_effect": {"type": "string"},
            "alternatives": {"type": "array", "items": {"type": "string"}},
            "authorization_and_safety": {"type": "string"},
            "success_observations": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "decision": {"type": "string", "enum": ["proceed", "investigate", "abstain"]},
        }, "required": ["proposed_action", "supporting_observation_ids", "expected_effect", "alternatives", "authorization_and_safety", "success_observations", "uncertainties", "decision"], "additionalProperties": False}
        required.append("pre_action_review")
    if profile.information_gain_selection:
        properties["information_gain_selection"] = {"type": "object", "properties": {
            "uncertainty": {"type": "string"},
            "candidate_observations": {"type": "array", "items": {"type": "string"}},
            "selected_observation": {"type": "string"},
            "selection_rationale": {"type": "string"},
            "redundancy_risk": {"type": "string"},
        }, "required": ["uncertainty", "candidate_observations", "selected_observation", "selection_rationale", "redundancy_risk"], "additionalProperties": False}
        required.append("information_gain_selection")
    return {"type": "function", "function": {
        "name": TOOL_NAME,
        "description": "Record your current evidence, uncertainties, and reasoning plan. This stores only values you provide and performs no network operation or inference.",
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }}


class ReasoningStateManager:
    """Persist model-authored reasoning without deriving or suggesting answers."""

    def __init__(self, profile: ReasoningProfile) -> None:
        if not profile.enabled:
            raise ValueError("B0 has no reasoning-state manager")
        self.profile = profile
        self.history: list[dict[str, Any]] = []

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {"tool": TOOL_NAME, "status": "error", "error": "arguments_must_be_object"}
        required = set(reasoning_tool_schema(self.profile)["function"]["parameters"]["required"])
        missing = sorted(required - arguments.keys())
        if missing:
            return {"tool": TOOL_NAME, "status": "error", "error": "missing_fields", "fields": missing}
        record = copy.deepcopy(arguments)
        record["reasoning_state_id"] = f"reasoning-{len(self.history) + 1:04d}"
        self.history.append(record)
        return {"tool": TOOL_NAME, "status": "ok", "reasoning_state_id": record["reasoning_state_id"], "recorded": record}

    def snapshot(self) -> dict[str, Any]:
        return {"condition": self.profile.condition, "updates": copy.deepcopy(self.history),
                "current": copy.deepcopy(self.history[-1]) if self.history else None}
