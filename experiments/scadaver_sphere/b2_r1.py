"""Answer-blind reasoning support for the B2-R1 C0-C4 study."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .reasoning_ablation import (
    TOOL_NAME,
    ReasoningProfile,
    reasoning_tool_schema as base_reasoning_tool_schema,
)


ROOT = Path(__file__).resolve().parent
PROFILE_FAMILY = "b2-r1-v1"
COMPONENTS = (
    "error_aware_investigation",
    "evidence_verification",
    "uncertainty_review",
)
EXPECTED_COMPONENTS = {
    "C0": (False, False, False),
    "C1": (True, False, False),
    "C2": (False, True, False),
    "C3": (False, False, True),
    "C4": (True, True, True),
}
SUCCESS_STATUSES = {"ok", "accepted", "success"}
# These fingerprints identify the sanitized public inputs, not an old deployment.
FROZEN_BASELINE = {
    "harness_commit": None,
    "baseline_provenance_note": (
        "Public inputs were sanitized and refingerprinted. Historical deployment "
        "results require revalidation; the original private baseline is unavailable."
    ),
    "original_b2_config_sha256": "9ebf50871aaf09967394d1b12f5dbdcf1657ed0040be6d49c424ed5cf93e4b5b",
    "original_b2_prompt_sha256": "57ac35a48f1456abf706cedf8965d73a4c92d080b5efc11d0c815c3b594ea100",
    "base_scenario_sha256": "bbda711efb87704a0ee3a840741908d53114595c648f88600cdc2ffbaea5cc8c",
    "system_prompt_sha256": "756d650237553d71f640276932cf9e2a947db8edd33c800ac3fb84ba2f8af1fd",
    "task_prompt_sha256": "0d1432093b7ff8857a64b9a918191f703991b00ed07f6cc937df2791218eade5",
    "integrated_evaluator_sha256": "6f137b0474f1774c75dc0edf09ca3b573655c471ce0c20d03374a1dfb6ad4050",
    "e5_evaluator_sha256": "22a6c8e749523612a2b44d63aa3889113782f50727968e3e39c51ae3f68eb553",
    "action_broker_sha256": "aa01e5ebaf4fcfd6fd3fec20934ee734446919982cf36fd9da33c561238af734",
    "investigative_tools_schema_sha256": "236fc6e3c617cca38d659f59d90534a5b071c4cd08b2ef3935eb4f20989b5d4f",
    "action_schema_sha256": "7f3fc05340e8ebce99ee27e852f38183c7ddda2a6675d650d6845263512d5239",
    "final_report_schema_sha256": "386fc5b3ed2eabfb8630fd32ad0e3fe4486c99552f050331ad006d69f1547700",
    "evaluator_version": "integrated-e1-e5-causal-v3",
}


@dataclass(frozen=True)
class B2R1Profile:
    condition: str
    version: str
    profile_family: str
    base_config: str
    base_reasoning_profile: str
    component_prompts: dict[str, str]
    error_aware_investigation: bool
    evidence_verification: bool
    uncertainty_review: bool
    freshness_max_age_seconds: float

    @classmethod
    def from_json(cls, path: Path) -> "B2R1Profile":
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(
            condition=str(raw["condition"]),
            version=str(raw["version"]),
            profile_family=str(raw["profile_family"]),
            base_config=str(raw["base_config"]),
            base_reasoning_profile=str(raw["base_reasoning_profile"]),
            component_prompts={str(key): str(value)
                               for key, value in raw["component_prompts"].items()},
            **{name: bool(raw.get(name, False)) for name in COMPONENTS},
            freshness_max_age_seconds=float(raw.get("freshness_max_age_seconds", 30.0)),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.profile_family != PROFILE_FAMILY:
            raise ValueError(f"profile_family must be {PROFILE_FAMILY}")
        if self.condition not in EXPECTED_COMPONENTS:
            raise ValueError("B2-R1 condition must be C0-C4")
        enabled = tuple(getattr(self, name) for name in COMPONENTS)
        if enabled != EXPECTED_COMPONENTS[self.condition]:
            raise ValueError("B2-R1 component selection does not match C0-C4 design")
        if set(self.component_prompts) != set(COMPONENTS):
            raise ValueError("component_prompts must name exactly the three R1 mechanisms")
        if self.freshness_max_age_seconds <= 0:
            raise ValueError("freshness_max_age_seconds must be positive")

    @property
    def enabled(self) -> bool:
        return True

    def public_snapshot(self) -> dict[str, Any]:
        return asdict(self)


def is_b2_r1_config(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return raw.get("profile_family") == PROFILE_FAMILY


def _safe_read(root: Path, relative: str) -> str:
    path = (root / relative).resolve()
    expected_root = root.resolve()
    if expected_root not in path.parents:
        raise ValueError("B2-R1 prompt/profile path must remain under the experiment root")
    return path.read_text(encoding="utf-8").strip()


def load_prompt(profile: B2R1Profile, root: Path = ROOT) -> str:
    """Render original B2 text plus only the selected R1 component text."""
    base = _safe_read(root, "prompts/reasoning_ablation_v1/B2.md")
    additions = [
        _safe_read(root, profile.component_prompts[name])
        for name in COMPONENTS if getattr(profile, name)
    ]
    return "\n\n".join([base, *additions])


def _base_b2_schema() -> dict[str, Any]:
    base = ReasoningProfile.from_json(ROOT / "config/reasoning_ablation_v1/B2.json")
    schema = base_reasoning_tool_schema(base)
    assert schema is not None
    return copy.deepcopy(schema)


def reasoning_tool_schema(profile: B2R1Profile) -> dict[str, Any]:
    schema = _base_b2_schema()
    parameters = schema["function"]["parameters"]
    properties = parameters["properties"]
    required = parameters["required"]

    if profile.error_aware_investigation:
        properties["error_review"] = {
            "type": ["object", "null"],
            "properties": {
                "purpose": {"type": "string"},
                "result_observation_ids": {"type": "array", "items": {"type": "string"}},
                "failure_category": {"type": "string"},
                "useful_information": {"type": "string"},
                "remaining_uncertainty": {"type": "string"},
                "alternative_observations": {"type": "array", "items": {"type": "string"}},
                "repeat_assessment": {
                    "type": "string",
                    "enum": ["likely_helpful", "unlikely_helpful", "unknown"],
                },
                "disposition": {
                    "type": "string",
                    "enum": ["continue", "change_direction", "abandon"],
                },
            },
            "required": [
                "purpose", "result_observation_ids", "failure_category",
                "useful_information", "remaining_uncertainty",
                "alternative_observations", "repeat_assessment", "disposition",
            ],
            "additionalProperties": False,
        }
        required.append("error_review")

    if profile.evidence_verification:
        properties["verification_cycles"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string", "minLength": 1},
                    "claim": {"type": "string"},
                    "supporting_observation_ids": {"type": "array", "items": {"type": "string"}},
                    "contradicting_observation_ids": {"type": "array", "items": {"type": "string"}},
                    "alternative_explanations": {"type": "array", "items": {"type": "string"}},
                    "discriminating_evidence": {"type": "string"},
                    "result_observation_ids": {"type": "array", "items": {"type": "string"}},
                    "result_interpretation": {"type": "string"},
                    "disposition": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "revised", "rejected", "unresolved"],
                    },
                    "revised_claim": {"type": ["string", "null"]},
                },
                "required": [
                    "hypothesis_id", "claim", "supporting_observation_ids",
                    "contradicting_observation_ids", "alternative_explanations",
                    "discriminating_evidence", "result_observation_ids",
                    "result_interpretation", "disposition", "revised_claim",
                ],
                "additionalProperties": False,
            },
        }
        required.append("verification_cycles")

    if profile.uncertainty_review:
        properties["uncertainty_reviews"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "claim": {"type": "string"},
                    "claim_scope": {
                        "type": "string",
                        "enum": ["protocol_identity", "functional_role", "temporal_relationship", "process_state", "other"],
                    },
                    "observation_ids": {"type": "array", "items": {"type": "string"}},
                    "recency_requirement": {
                        "type": "string", "enum": ["time_sensitive", "time_insensitive", "unknown"],
                    },
                    "protocol_role_distinction": {
                        "type": "string",
                        "enum": ["protocol_only", "role_evidence_present", "unresolved", "not_applicable"],
                    },
                    "alternative_explanations": {"type": "array", "items": {"type": "string"}},
                    "temporal_interpretation": {
                        "type": "string",
                        "enum": ["consistent", "conflicting", "incomplete", "not_applicable"],
                    },
                    "causal_scope": {
                        "type": "string", "enum": ["observation", "correlation", "causal_claim"],
                    },
                    "causal_evidence_basis": {
                        "type": "string", "enum": ["intervention", "passive_directionality", "none"],
                    },
                    "classification": {
                        "type": "string", "enum": ["supported", "uncertain", "contradicted"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "claim_id", "claim", "claim_scope", "observation_ids",
                    "recency_requirement", "protocol_role_distinction",
                    "alternative_explanations", "temporal_interpretation",
                    "causal_scope", "causal_evidence_basis", "classification", "rationale",
                ],
                "additionalProperties": False,
            },
        }
        required.append("uncertainty_reviews")
    return schema


class B2R1StateManager:
    """Validate model-authored reasoning provenance without evaluating its answer."""

    def __init__(self, profile: B2R1Profile, *, now: Callable[[], float] = time.time) -> None:
        self.profile = profile
        self.now = now
        self.history: list[dict[str, Any]] = []
        self.observations: dict[str, dict[str, Any]] = {}
        self.observation_order: list[str] = []
        self.known_contradictions: dict[str, set[str]] = {}

    def observe(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        observation_id = result.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            return
        if observation_id in self.observations:
            return
        timestamp = _observation_timestamp(result)
        status = str(result.get("status", "unknown"))
        self.observations[observation_id] = {
            "observation_id": observation_id,
            "tool": tool,
            "arguments": copy.deepcopy(arguments),
            "status": status,
            "successful": status in SUCCESS_STATUSES,
            "error": result.get("error") or result.get("error_category"),
            "timestamp": timestamp,
            "sequence": len(self.observation_order) + 1,
        }
        self.observation_order.append(observation_id)

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {"tool": TOOL_NAME, "status": "error", "error": "arguments_must_be_object"}
        schema = reasoning_tool_schema(self.profile)
        required = set(schema["function"]["parameters"]["required"])
        missing = sorted(required - arguments.keys())
        if missing:
            return {"tool": TOOL_NAME, "status": "error", "error": "missing_fields", "fields": missing}

        # C0 is the original B2 control. Preserve its model-visible state-tool
        # behavior exactly; the registry remains private experiment metadata.
        if not any(getattr(self.profile, name) for name in COMPONENTS):
            record = copy.deepcopy(arguments)
            record["reasoning_state_id"] = f"reasoning-{len(self.history) + 1:04d}"
            self.history.append(record)
            return {
                "tool": TOOL_NAME,
                "status": "ok",
                "reasoning_state_id": record["reasoning_state_id"],
                "recorded": copy.deepcopy(record),
            }

        issues = self._validate_evidence(arguments)
        if issues:
            return {
                "tool": TOOL_NAME,
                "status": "error",
                "error": "evidence_validation_failed",
                "issues": issues,
            }

        record = copy.deepcopy(arguments)
        record["reasoning_state_id"] = f"reasoning-{len(self.history) + 1:04d}"
        record["validation"] = self._validation_summary(record)
        self.history.append(record)
        for cycle in record.get("verification_cycles", []):
            hypothesis_id = cycle["hypothesis_id"]
            self.known_contradictions.setdefault(hypothesis_id, set()).update(
                cycle["contradicting_observation_ids"]
            )
        return {
            "tool": TOOL_NAME,
            "status": "ok",
            "reasoning_state_id": record["reasoning_state_id"],
            "recorded": copy.deepcopy(record),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "condition": self.profile.condition,
            "profile_family": self.profile.profile_family,
            "updates": copy.deepcopy(self.history),
            "current": copy.deepcopy(self.history[-1]) if self.history else None,
            "observation_registry": copy.deepcopy(self.observations),
        }

    def _validate_evidence(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        def validate_ids(field: str, values: Any, *, require_success: bool = False) -> None:
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                issues.append({"field": field, "error": "observation_ids_must_be_strings"})
                return
            for observation_id in values:
                observation = self.observations.get(observation_id)
                if observation is None:
                    issues.append({"field": field, "observation_id": observation_id,
                                   "error": "unknown_or_cross_episode_observation"})
                elif require_success and not observation["successful"]:
                    issues.append({"field": field, "observation_id": observation_id,
                                   "error": "unsuccessful_observation_cannot_support_claim"})

        if self.profile.evidence_verification:
            for index, item in enumerate(arguments.get("evidence_ledger", [])):
                status = item.get("status")
                validate_ids(f"evidence_ledger[{index}].observation_ids",
                             item.get("observation_ids"),
                             require_success=status == "supported")
            for index, item in enumerate(arguments.get("hypotheses", [])):
                validate_ids(f"hypotheses[{index}].supporting_observation_ids",
                             item.get("supporting_observation_ids"), require_success=True)
                validate_ids(f"hypotheses[{index}].contradicting_observation_ids",
                             item.get("contradicting_observation_ids"))

        review = arguments.get("error_review")
        if self.profile.error_aware_investigation and isinstance(review, dict):
            validate_ids("error_review.result_observation_ids",
                         review.get("result_observation_ids"))

        hypothesis_ids: set[str] = set()
        for index, cycle in enumerate(
                arguments.get("verification_cycles", []) if self.profile.evidence_verification else []):
            hypothesis_id = cycle.get("hypothesis_id")
            if not isinstance(hypothesis_id, str) or not hypothesis_id:
                issues.append({"field": f"verification_cycles[{index}].hypothesis_id",
                               "error": "hypothesis_id_required"})
                continue
            if hypothesis_id in hypothesis_ids:
                issues.append({"field": f"verification_cycles[{index}].hypothesis_id",
                               "error": "duplicate_hypothesis_id_in_update"})
            hypothesis_ids.add(hypothesis_id)
            support = cycle.get("supporting_observation_ids")
            contradict = cycle.get("contradicting_observation_ids")
            result_ids = cycle.get("result_observation_ids")
            validate_ids(f"verification_cycles[{index}].supporting_observation_ids",
                         support, require_success=True)
            validate_ids(f"verification_cycles[{index}].contradicting_observation_ids",
                         contradict)
            requires_test = cycle.get("disposition") in {
                "supported", "contradicted", "revised", "rejected",
            }
            validate_ids(f"verification_cycles[{index}].result_observation_ids",
                         result_ids, require_success=requires_test)
            if requires_test and not result_ids:
                issues.append({"field": f"verification_cycles[{index}].result_observation_ids",
                               "error": "meaningful_test_required_for_disposition"})
            if cycle.get("disposition") == "revised" and not cycle.get("revised_claim"):
                issues.append({"field": f"verification_cycles[{index}].revised_claim",
                               "error": "revised_claim_required"})
            previous = self.known_contradictions.get(hypothesis_id, set())
            current = set(contradict or [])
            discarded = sorted(previous - current)
            if discarded:
                issues.append({"field": f"verification_cycles[{index}].contradicting_observation_ids",
                               "error": "contradictory_evidence_discarded",
                               "observation_ids": discarded})

        for index, review in enumerate(
                arguments.get("uncertainty_reviews", []) if self.profile.uncertainty_review else []):
            field = f"uncertainty_reviews[{index}]"
            observation_ids = review.get("observation_ids")
            validate_ids(f"{field}.observation_ids", observation_ids,
                         require_success=review.get("classification") == "supported")
            freshness = self._freshness(observation_ids or [])
            if (review.get("classification") == "supported"
                    and review.get("recency_requirement") == "time_sensitive"
                    and freshness != "fresh"):
                issues.append({"field": f"{field}.recency_requirement",
                               "error": "stale_or_unknown_evidence_cannot_support_time_sensitive_claim",
                               "computed": freshness})
            if (review.get("classification") == "supported"
                    and review.get("claim_scope") == "functional_role"
                    and review.get("protocol_role_distinction") != "role_evidence_present"):
                issues.append({"field": f"{field}.protocol_role_distinction",
                               "error": "protocol_identity_alone_cannot_support_functional_role"})
            temporal = self._temporal_quality(observation_ids or [])
            if (review.get("classification") == "supported"
                    and review.get("claim_scope") in {"temporal_relationship", "process_state"}
                    and (review.get("temporal_interpretation") != "consistent"
                         or temporal != "consistent")):
                issues.append({"field": f"{field}.temporal_interpretation",
                               "error": "temporal_evidence_not_consistent", "computed": temporal})
            if (review.get("classification") == "supported"
                    and review.get("causal_scope") == "causal_claim"
                    and review.get("causal_evidence_basis") == "none"):
                issues.append({"field": f"{field}.causal_evidence_basis",
                               "error": "causal_claim_lacks_causal_evidence"})
        return issues

    def _freshness(self, observation_ids: list[str]) -> str:
        timestamps = [self.observations[item].get("timestamp") for item in observation_ids
                      if item in self.observations]
        if not timestamps or any(value is None for value in timestamps):
            return "unknown"
        newest = max(float(value) for value in timestamps if value is not None)
        return "fresh" if 0 <= self.now() - newest <= self.profile.freshness_max_age_seconds else "stale"

    def _temporal_quality(self, observation_ids: list[str]) -> str:
        values = [self.observations[item] for item in observation_ids
                  if item in self.observations]
        if len(values) < 2 or any(item.get("timestamp") is None for item in values):
            return "incomplete"
        ordered = sorted(values, key=lambda item: item["sequence"])
        timestamps = [float(item["timestamp"]) for item in ordered]
        return "consistent" if timestamps == sorted(timestamps) else "conflicting"

    def _validation_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        cycles = record.get("verification_cycles", [])
        meaningful = [
            cycle["hypothesis_id"] for cycle in cycles
            if cycle.get("disposition") != "unresolved" and cycle.get("result_observation_ids")
        ]
        reviews = record.get("uncertainty_reviews", [])
        return {
            "citations_current_episode": True,
            "meaningfully_tested_hypothesis_ids": meaningful,
            "reviewed_claim_ids": [review["claim_id"] for review in reviews],
            "supported_claim_ids": [review["claim_id"] for review in reviews
                                    if review["classification"] == "supported"],
            "uncertain_claim_ids": [review["claim_id"] for review in reviews
                                    if review["classification"] == "uncertain"],
            "contradicted_claim_ids": [review["claim_id"] for review in reviews
                                       if review["classification"] == "contradicted"],
            "review_freshness": {review["claim_id"]: self._freshness(review["observation_ids"])
                                 for review in reviews},
            "review_temporal_quality": {
                review["claim_id"]: self._temporal_quality(review["observation_ids"])
                for review in reviews
            },
        }


def _observation_timestamp(result: dict[str, Any]) -> float | None:
    for field in ("collected_at_unix", "timestamp", "time"):
        value = result.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    value = result.get("time_utc") or result.get("observed_at")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
