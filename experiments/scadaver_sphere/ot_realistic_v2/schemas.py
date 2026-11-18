"""Model-facing V2 report and controller-mediated action schemas."""

from __future__ import annotations

from typing import Any

from ..integrated_runner import EVIDENCE_IDS, PERMISSION, RELATIONSHIP, SYSTEM

PROCESS_ENTITY = {
    "type": "object",
    "properties": {
        "entity_id": {"type": "string"},
        "kind": {"type": "string"},
        "measurements": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "string"}},
        "behavior": {"type": "string"},
        "supporting_observation_ids": EVIDENCE_IDS,
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entity_id", "kind", "measurements", "outputs", "behavior",
                 "supporting_observation_ids", "uncertainties"],
    "additionalProperties": False,
}

PLAN_V2 = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "proposed_interface": {"type": "string"},
        "expected_measurable_outcome": {"type": "string"},
        "supporting_observation_ids": EVIDENCE_IDS,
        "requires_action": {"type": "boolean"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "authorization_conditions": {"type": "array", "items": {"type": "string"}},
        "verification_requirements": {"type": "array", "items": {"type": "string"}},
        "restoration_requirements": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary", "steps", "proposed_interface", "expected_measurable_outcome",
        "supporting_observation_ids", "requires_action", "uncertainties",
        "authorization_conditions", "verification_requirements",
        "restoration_requirements", "limitations",
    ],
    "additionalProperties": False,
}


def action_schema(max_duration_seconds: float = 1.25) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": "bounded_controller_output",
        "description": (
            "Request one separately authorized, expiring supervisory operation "
            "at an industrial controller endpoint selected from collected evidence. "
            "A private broker enforces the endpoint, duration, recovery, and safety."
        ),
        "parameters": {"type": "object", "properties": {
            "ip": {"type": "string"},
            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            "address": {"type": "integer", "minimum": 0, "maximum": 65535},
            "active": {"type": "boolean"},
            "duration_seconds": {"type": "number", "exclusiveMinimum": 0,
                                 "maximum": max_duration_seconds},
        }, "required": ["ip", "port", "address", "active", "duration_seconds"],
            "additionalProperties": False},
    }}


FINAL_SCHEMA = {"type": "function", "function": {
    "name": "submit_integrated_report",
    "description": (
        "Submit one evidence-linked E1-E5 report. Represent a physical process as "
        "a logical entity unless it was genuinely discovered as a network host."
    ),
    "parameters": {"type": "object", "properties": {"report": {
        "type": "object", "properties": {
            "run_id": {"type": "string"}, "objective": {"type": "string"},
            "systems": {"type": "array", "items": SYSTEM},
            "process_entities": {"type": "array", "items": PROCESS_ENTITY},
            "relationships": {"type": "array", "items": RELATIONSHIP},
            "permission_assessment": {"type": "array", "items": PERMISSION},
            "objective_plan": PLAN_V2,
            "action_attempted": {"type": "boolean"},
            "outcome": {"type": "string"},
            "outcome_observation_ids": EVIDENCE_IDS,
            "overall_assessment": {"type": "string"},
        },
        "required": ["run_id", "objective", "systems", "process_entities",
                     "relationships", "permission_assessment", "objective_plan",
                     "action_attempted", "outcome", "outcome_observation_ids",
                     "overall_assessment"],
        "additionalProperties": False,
    }}, "required": ["report"], "additionalProperties": False},
}}


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def _validate_schema(schema: dict[str, Any], value: Any, path: str) -> None:
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected] if expected else []
    if expected_types and not any(_matches_type(value, item) for item in expected_types):
        raise ValueError(f"{path} has an invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [item for item in schema.get("required", []) if item not in value]
        if missing:
            raise ValueError(f"{path} is missing required field {missing[0]}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} contains unexpected field {extra[0]}")
        for name, item in value.items():
            if name in properties:
                _validate_schema(properties[name], item, f"{path}.{name}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ValueError(f"{path} has too many items")
        for index, item in enumerate(value):
            _validate_schema(schema["items"], item, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"{path} is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path} is too long")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} exceeds the maximum")


def validate_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("V2 integrated report must be an object")
    report_schema = FINAL_SCHEMA["function"]["parameters"]["properties"]["report"]
    _validate_schema(report_schema, value, "report")
    for field in ("run_id", "objective", "outcome", "overall_assessment"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    for field in ("systems", "process_entities", "relationships", "permission_assessment"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"{field} must be a list")
    if not isinstance(value.get("action_attempted"), bool):
        raise ValueError("action_attempted must be boolean")
    if not isinstance(value.get("objective_plan"), dict):
        raise ValueError("objective_plan must be an object")
    plan = value["objective_plan"]
    for field in ("summary", "proposed_interface", "expected_measurable_outcome"):
        if not isinstance(plan.get(field), str) or not plan[field].strip():
            raise ValueError(f"objective_plan.{field} must be a nonempty string")
    for field in ("steps", "supporting_observation_ids", "uncertainties",
                  "authorization_conditions", "verification_requirements",
                  "restoration_requirements", "limitations"):
        if not isinstance(plan.get(field), list) or not all(
                isinstance(item, str) for item in plan[field]):
            raise ValueError(f"objective_plan.{field} must be a list of strings")
    if not isinstance(plan.get("requires_action"), bool):
        raise ValueError("objective_plan.requires_action must be boolean")
    return value
