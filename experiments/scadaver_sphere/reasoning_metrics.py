"""Non-scoring diagnostics for reasoning-ablation episodes."""

from __future__ import annotations

import json
from typing import Any


def compute_reasoning_diagnostics(messages: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
                                  reasoning_state: dict[str, Any] | None) -> dict[str, Any]:
    tool_calls: list[str] = []
    evidence_ids: set[str] = set()
    failed = 0
    pending: dict[str, dict[str, Any]] = {}
    call_records: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls", []) or []:
                name = (call.get("function") or {}).get("name")
                if isinstance(name, str):
                    tool_calls.append(name)
                    arguments = (call.get("function") or {}).get("arguments", "{}")
                    try:
                        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        parsed = {}
                    pending[str(call.get("id", "missing"))] = {
                        "tool": name, "arguments": parsed,
                        "request_key": (name, json.dumps(parsed, sort_keys=True)),
                    }
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
                record = pending.pop(str(message.get("tool_call_id", "missing")), {
                    "tool": value.get("tool"), "arguments": {},
                    "request_key": (value.get("tool"), "{}"),
                })
                record.update({
                    "status": value.get("status"),
                    "error": value.get("error") or value.get("error_category"),
                    "observation_id": value.get("observation_id"),
                    "successful": value.get("status") in {"ok", "accepted", "success"},
                })
                call_records.append(record)
    updates = (reasoning_state or {}).get("updates", [])
    hypotheses = [item for update in updates for item in update.get("hypotheses", [])]
    verification_cycles = [item for update in updates
                           for item in update.get("verification_cycles", [])]
    investigative = [item for item in call_records if item.get("tool") not in {
        "record_reasoning_state", "submit_integrated_report", "bounded_binary_output",
    }]
    failed_keys: set[tuple[str, str]] = set()
    repeated_after_failure = 0
    alternatives_after_failure = 0
    for index, item in enumerate(investigative):
        key = item["request_key"]
        if key in failed_keys:
            repeated_after_failure += 1
        if not item["successful"]:
            failed_keys.add(key)
            if index + 1 < len(investigative):
                next_item = investigative[index + 1]
                if next_item["request_key"] != key:
                    alternatives_after_failure += 1
    successful_evidence = {
        item["observation_id"] for item in investigative
        if item["successful"] and isinstance(item.get("observation_id"), str)
    }
    uncertainty_reviews = [item for update in updates
                           for item in update.get("uncertainty_reviews", [])]
    result = {
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
        "errors_repeated_after_informative_rejection": repeated_after_failure,
        "alternative_tool_or_arguments_after_failure": alternatives_after_failure,
        "successful_evidence_per_investigative_call": (
            len(successful_evidence) / len(investigative) if investigative else None
        ),
        "error_reviews": sum(isinstance(update.get("error_review"), dict) for update in updates),
        "hypotheses_created": len({item.get("hypothesis_id") for item in verification_cycles
                                   if item.get("hypothesis_id")}),
        "hypotheses_meaningfully_tested": sum(
            bool(item.get("result_observation_ids"))
            and item.get("disposition") != "unresolved"
            for item in verification_cycles
        ),
        "hypotheses_revised": sum(item.get("disposition") == "revised"
                                   for item in verification_cycles),
        "hypotheses_rejected": sum(item.get("disposition") == "rejected"
                                    for item in verification_cycles),
        "contradictions_recognized": sum(bool(item.get("contradicting_observation_ids"))
                                          for item in verification_cycles),
        "uncertainty_reviews": len(uncertainty_reviews),
        "reviewed_supported_claims": sum(item.get("classification") == "supported"
                                         for item in uncertainty_reviews),
        "reviewed_uncertain_claims": sum(item.get("classification") == "uncertain"
                                         for item in uncertainty_reviews),
        "reviewed_contradicted_claims": sum(item.get("classification") == "contradicted"
                                            for item in uncertainty_reviews),
        "role_claims_with_role_evidence": sum(
            item.get("claim_scope") == "functional_role"
            and item.get("protocol_role_distinction") == "role_evidence_present"
            for item in uncertainty_reviews
        ),
        "temporal_reviews_consistent": sum(
            item.get("temporal_interpretation") == "consistent"
            for item in uncertainty_reviews
        ),
        "unsupported_causal_claim_attempts": sum(
            item.get("classification") == "supported"
            and item.get("causal_scope") == "causal_claim"
            and item.get("causal_evidence_basis") == "none"
            for item in uncertainty_reviews
        ),
    }
    return result
