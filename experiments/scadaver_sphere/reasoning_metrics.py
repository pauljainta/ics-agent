"""Non-scoring diagnostics for reasoning-ablation episodes."""

from __future__ import annotations

import json
from typing import Any


def compute_reasoning_diagnostics(messages: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
                                  reasoning_state: dict[str, Any] | None) -> dict[str, Any]:
    tool_calls: list[str] = []
    evidence_ids: set[str] = set()
    failed = 0
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []) or []:
                name = (call.get("function") or {}).get("name")
                if isinstance(name, str):
                    tool_calls.append(name)
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            try:
                value = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                if isinstance(value.get("observation_id"), str):
                    evidence_ids.add(value["observation_id"])
                if value.get("status") in {"error", "rejected", "unsupported", "disabled"}:
                    failed += 1
    updates = (reasoning_state or {}).get("updates", [])
    hypotheses = [item for update in updates for item in update.get("hypotheses", [])]
    return {
        "tool_calls_total": len(tool_calls),
        "distinct_tools": len(set(tool_calls)),
        "reasoning_state_updates": len(updates),
        "evidence_observations": len(evidence_ids),
        "hypotheses_recorded": len(hypotheses),
        "hypotheses_revised_or_rejected": sum(item.get("status") in {"revised", "rejected"} for item in hypotheses),
        "pre_action_reviews": sum(update.get("pre_action_review") is not None for update in updates),
        "information_gain_selections": sum("information_gain_selection" in update for update in updates),
        "redundant_probe_requests": sum(item.get("kind") == "redundant_probe_request" for item in diagnostics),
        "unsuccessful_tool_calls": failed,
        "submission_recovery_events": sum(str(item.get("kind", "")).startswith("submission_recovery") for item in diagnostics),
    }
