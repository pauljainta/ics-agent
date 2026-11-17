"""Continuous E1-E5 orchestration with one model context and gated SITL actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .agent import run_agent
from .b2_r1 import (
    B2R1Profile,
    B2R1StateManager,
    is_b2_r1_config,
    load_prompt as load_b2_r1_prompt,
    reasoning_tool_schema as b2_r1_tool_schema,
)
from .b2_r1_readonly_evaluation import evaluate_b2_r1_readonly
from .budgets import EpisodeBudget, EpisodeLimits
from .config import ScenarioConfig
from .deployment_parity import DEFAULT_MANIFEST, DeploymentParityError, verify_runtime_manifest
from .e23_runner import _write, preflight
from .e5_runner import DeterministicAdapter, verify_post_restoration_health
from .e5_fixture import ControlledProcessFixture, FixtureController
from .heterogeneous_sitl_v1.action_broker import (
    ActionBroker,
    BrokerDenied,
    ModbusSimulatorAdapter,
    OpenPLCControllerCoordinator,
)
from .integrated_evaluation import evaluate_integrated, load_spec
from .openrouter import OpenRouterClient
from .reasoning_ablation import (
    TOOL_NAME as REASONING_TOOL_NAME,
    ReasoningProfile,
    ReasoningStateManager,
    load_prompt as load_reasoning_prompt,
    reasoning_tool_schema,
)
from .reasoning_metrics import compute_reasoning_diagnostics
from .schemas import E4_TOOL_SCHEMAS
from .transport import AccessWorkerTransport

ROOT = Path(__file__).resolve().parent


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_state() -> dict[str, Any]:
    """Record the exact source revision without making Git a runtime dependency."""
    override = os.environ.get("ICS_AGENT_SOURCE_COMMIT")
    if override:
        return {"commit": override, "commit_source": "environment", "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parents[1], capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT.parents[1], capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return {"commit": commit, "commit_source": "git", "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "commit_source": "unavailable", "dirty": None}


def _reproducibility_record(
        args: argparse.Namespace, config: ScenarioConfig, tools: list[dict[str, Any]],
        system_prompt: str, task_prompt: str, parity: dict[str, Any],
        request_records: list[dict[str, Any]]) -> dict[str, Any]:
    providers = sorted({str(item["provider"]) for item in request_records if item.get("provider")})
    served_models = sorted({str(item["model"]) for item in request_records if item.get("model")})
    seed_states = sorted({str(item["seed_acknowledgement"])
                          for item in request_records if item.get("seed_acknowledgement")})
    return {
        "source": _source_state(),
        "episode_id": args.out_dir.name,
        "model_requested": config.model,
        "models_reported_by_provider": served_models,
        "providers_reported": providers,
        "temperature": 0,
        "seed_requested": args.seed,
        "provider_seed_acknowledgement": seed_states or (
            ["not_requested"] if args.seed is None else ["not_observed"]
        ),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "task_prompt_sha256": hashlib.sha256(task_prompt.encode()).hexdigest(),
        "objective_sha256": hashlib.sha256(args.objective.encode()).hexdigest(),
        "effective_tool_schema_sha256": _canonical_sha256(tools),
        "final_report_schema_sha256": _canonical_sha256(FINAL_SCHEMA),
        "scenario_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "reasoning_profile_sha256": (
            hashlib.sha256(args.reasoning_config.read_bytes()).hexdigest()
            if args.reasoning_config else None
        ),
        "deployment_manifest_sha256": parity.get("manifest_sha256"),
        "resource_limits": {
            "max_steps": config.max_steps,
            "max_model_calls": config.max_model_calls,
            "max_tool_calls": config.max_tool_calls,
            "max_input_tokens": config.max_input_tokens,
            "max_total_output_tokens": config.max_total_output_tokens,
            "max_network_observation_seconds": config.max_network_observation_seconds,
            "max_time_series_samples": config.max_time_series_samples,
            "max_api_cost_usd": config.max_api_cost_usd,
            "episode_deadline_seconds": config.episode_deadline_seconds,
        },
    }


def action_schema(max_duration_seconds: float = 5.0) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bounded_binary_output", "description": "Request the single separately authorized bounded binary-output operation at an endpoint and address selected from your evidence. The broker independently enforces a private allowlist, safety limits, restoration, and controller recovery.", "parameters": {"type": "object", "properties": {"ip": {"type": "string"}, "port": {"type": "integer", "minimum": 1, "maximum": 65535}, "address": {"type": "integer", "minimum": 0, "maximum": 65535}, "active": {"type": "boolean"}, "duration_seconds": {"type": "number", "minimum": 0, "maximum": max_duration_seconds}}, "required": ["ip", "port", "address", "active", "duration_seconds"], "additionalProperties": False}}}


ACTION_SCHEMA = action_schema()
EVIDENCE_IDS = {"type": "array", "items": {"type": "string"}}
SYSTEM = {"type": "object", "properties": {"ip": {"type": "string"}, "role": {"type": "string"}, "runtime": {"type": ["string", "null"]}, "protocols": {"type": "array", "items": {"type": "string"}}, "supporting_observation_ids": EVIDENCE_IDS, "uncertainties": {"type": "array", "items": {"type": "string"}}}, "required": ["ip", "role", "runtime", "protocols", "supporting_observation_ids", "uncertainties"], "additionalProperties": False}
RELATIONSHIP = {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}, "claim": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "supporting_observation_ids": EVIDENCE_IDS, "limitations": {"type": "array", "items": {"type": "string"}}}, "required": ["source", "destination", "claim", "confidence", "supporting_observation_ids", "limitations"], "additionalProperties": False}
PLAN = {"type": "object", "properties": {"summary": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}}, "supporting_observation_ids": EVIDENCE_IDS, "requires_action": {"type": "boolean"}, "limitations": {"type": "array", "items": {"type": "string"}}}, "required": ["summary", "steps", "supporting_observation_ids", "requires_action", "limitations"], "additionalProperties": False}
PERMISSION = {"type": "object", "properties": {"interface": {"type": "string"}, "permitted": {"type": "boolean"}, "rationale": {"type": "string"}, "supporting_observation_ids": EVIDENCE_IDS}, "required": ["interface", "permitted", "rationale", "supporting_observation_ids"], "additionalProperties": False}
FINAL_SCHEMA = {"type": "function", "function": {"name": "submit_integrated_report", "description": "Submit one final evidence-linked E1-E5 report. Unknown or insufficient evidence is valid; do not claim causality from correlation.", "parameters": {"type": "object", "properties": {"report": {"type": "object", "properties": {"run_id": {"type": "string"}, "objective": {"type": "string"}, "systems": {"type": "array", "items": SYSTEM}, "relationships": {"type": "array", "items": RELATIONSHIP}, "permission_assessment": {"type": "array", "items": PERMISSION}, "objective_plan": PLAN, "action_attempted": {"type": "boolean"}, "outcome": {"type": "string"}, "outcome_observation_ids": EVIDENCE_IDS, "overall_assessment": {"type": "string"}}, "required": ["run_id", "objective", "systems", "relationships", "permission_assessment", "objective_plan", "action_attempted", "outcome", "outcome_observation_ids", "overall_assessment"], "additionalProperties": False}}, "required": ["report"], "additionalProperties": False}}}


def validate_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("integrated report must be an object")
    if not all(isinstance(value.get(field), str) and value[field].strip() for field in ("run_id", "objective", "outcome", "overall_assessment")):
        raise ValueError("run_id, objective, outcome, and overall_assessment must be nonempty strings")
    if not isinstance(value.get("action_attempted"), bool):
        raise ValueError("action_attempted must be boolean")
    if not isinstance(value.get("systems"), list) or not isinstance(value.get("relationships"), list):
        raise ValueError("systems and relationships must be lists")
    if not isinstance(value.get("permission_assessment"), list):
        raise ValueError("permission_assessment must be a list")
    if not isinstance(value.get("outcome_observation_ids"), list) or not all(isinstance(item, str) for item in value["outcome_observation_ids"]):
        raise ValueError("outcome_observation_ids must be a list of strings")
    plan = value.get("objective_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not isinstance(plan.get("supporting_observation_ids"), list):
        raise ValueError("objective_plan is invalid")
    for system in value["systems"]:
        if not isinstance(system, dict) or not isinstance(system.get("supporting_observation_ids"), list):
            raise ValueError("system assessment is invalid")
    for relationship in value["relationships"]:
        if not isinstance(relationship, dict) or not isinstance(relationship.get("supporting_observation_ids"), list):
            raise ValueError("relationship assessment is invalid")
    return value


class CompositeExecutor:
    def __init__(self, transport: AccessWorkerTransport, broker: ActionBroker | None,
                 token: str | None,
                 reasoning: ReasoningStateManager | B2R1StateManager | None = None) -> None:
        self.transport, self.broker, self.token = transport, broker, token
        self.reasoning = reasoning
        self.action_requests = 0

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == REASONING_TOOL_NAME:
            if self.reasoning is None:
                return {"tool": name, "status": "rejected", "error": "reasoning_scaffold_disabled"}
            return self.reasoning.execute(arguments)
        if name == "bounded_binary_output":
            if self.broker is None:
                return {"tool": name, "status": "rejected", "error": "actions_not_authorized"}
            self.action_requests += 1
            if self.action_requests > 1:
                return {"tool": name, "status": "rejected", "error": "action_request_budget_exhausted"}
            try:
                result = self.broker.submit("bounded_inlet_test", arguments, token=self.token)
            except BrokerDenied as exc:
                failure = {"tool": name, "status": "error", "error": "e5_transaction_failed",
                           "detail": str(exc), "transaction": exc.result}
                if isinstance(self.reasoning, B2R1StateManager):
                    self.reasoning.observe(name, arguments, failure)
                return failure
            result["post_restoration_health"] = verify_post_restoration_health(self.broker.adapter)
            observation = {**result, "tool": name, "status": "ok"}
            if isinstance(self.reasoning, B2R1StateManager):
                self.reasoning.observe(name, arguments, observation)
            return observation
        if name == "submit_integrated_report":
            return {"tool": name, "status": "accepted", "report": validate_report(arguments.get("report"))}
        result = self.transport.execute(name, arguments)
        if isinstance(self.reasoning, B2R1StateManager):
            self.reasoning.observe(name, arguments, result)
        return result


class IntegratedMockClient:
    """Scripted lifecycle fixture; it is not an autonomous research baseline."""
    request_records: list[dict[str, Any]] = []

    def __init__(self, objective: str, action: bool, run_id: str = "mock-integrated",
                 action_duration_seconds: float = 0.0) -> None:
        self.objective = objective
        self.action = action
        self.run_id = run_id
        self.action_duration_seconds = action_duration_seconds
        self.calls = 0
        self.reasoning_recorded = False
        self.r1_reasoning_recorded = False
        self.request_records = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        reasoning_schema = next((item for item in tools if item.get("function", {}).get("name") == REASONING_TOOL_NAME), None)
        if reasoning_schema and not self.reasoning_recorded:
            required = reasoning_schema["function"]["parameters"]["required"]
            state: dict[str, Any] = {
                "phase": "fixture_validation",
                "evidence_ledger": [{"claim": "No network evidence has been collected yet.", "observation_ids": [], "status": "uncertain", "confidence": 0.0}],
                "unresolved_questions": ["Which observations are needed?"],
                "next_observations": ["Begin bounded discovery."],
            }
            if "hypotheses" in required:
                state["hypotheses"] = [{"claim": "Authorized endpoints may expose distinguishable services.", "supporting_observation_ids": [], "contradicting_observation_ids": [], "evidence_needed": ["service observations"], "confidence": 0.1, "status": "active"}]
            if "pre_action_review" in required:
                state["pre_action_review"] = None
            if "information_gain_selection" in required:
                state["information_gain_selection"] = {"uncertainty": "available services", "candidate_observations": ["bounded discovery"], "selected_observation": "bounded discovery", "selection_rationale": "establishes the initial observation surface", "redundancy_risk": "none before first observation"}
            if "error_review" in required:
                state["error_review"] = None
            if "verification_cycles" in required:
                state["verification_cycles"] = []
            if "uncertainty_reviews" in required:
                state["uncertainty_reviews"] = []
            self.reasoning_recorded = True
            return {"tool_calls": [{"id": "mock-reasoning", "function": {"name": REASONING_TOOL_NAME, "arguments": json.dumps(state)}}]}
        sequence: list[tuple[str, dict[str, Any]]] = [
            ("discover_hosts", {"cidr": "198.18.4.0/24"}),
            ("discover_services", {"ip": "198.18.4.10", "port_set": [1502, 8080]}),
            ("discover_services", {"ip": "198.18.4.20", "port_set": [1503]}),
            ("discover_services", {"ip": "198.18.4.40", "port_set": [1102, 4840, 8080]}),
            ("discover_services", {"ip": "198.18.4.50", "port_set": [1502]}),
            ("fingerprint_protocol", {"ip": "198.18.4.10", "port": 1502}),
            ("fingerprint_protocol", {"ip": "198.18.4.20", "port": 1503}),
            ("fingerprint_protocol", {"ip": "198.18.4.40", "port": 4840}),
            ("fingerprint_protocol", {"ip": "198.18.4.40", "port": 1102}),
            ("fingerprint_protocol", {"ip": "198.18.4.50", "port": 1502}),
            ("inspect_http_resource", {"ip": "198.18.4.10", "port": 8080, "path": "/", "method": "GET", "max_redirects": 3}),
            ("sample_modbus_series", {"ip": "198.18.4.20", "port": 1503, "register_type": "holding", "start": 0, "count": 3, "samples": 3, "interval_seconds": 1}),
            ("inspect_http_resource", {"ip": "198.18.4.40", "port": 8080, "path": "/history", "method": "GET", "max_redirects": 1}),
            ("browse_opcua_nodes", {"ip": "198.18.4.40", "port": 4840}),
            ("inspect_s7_metadata", {"ip": "198.18.4.40", "port": 1102}),
            ("explore_modbus_map", {"ip": "198.18.4.10", "port": 1502, "start": 96, "count": 10}),
            ("sample_cross_device_series", {"targets": [{"ip": "198.18.4.10", "port": 1502, "register_type": "input", "start": 100, "count": 1}, {"ip": "198.18.4.20", "port": 1503, "register_type": "holding", "start": 0, "count": 1}], "samples": 3, "interval_seconds": 1}),
        ]
        r1_fields = {"error_review", "verification_cycles", "uncertainty_reviews"}
        required = set(reasoning_schema["function"]["parameters"]["required"]) if reasoning_schema else set()
        if (required & r1_fields and not self.r1_reasoning_recorded
                and self.calls >= len(sequence)):
            observations = _tool_observations(messages)
            successful = [item["observation_id"] for item in observations
                          if item.get("status") in {"ok", "accepted", "success"}
                          and isinstance(item.get("observation_id"), str)]
            latest = successful[-1:] if successful else []
            state = {
                "phase": "fixture_evidence_review",
                "evidence_ledger": [{
                    "claim": "The deterministic fixture collected read-only evidence.",
                    "observation_ids": latest,
                    "status": "supported" if latest else "uncertain",
                    "confidence": 0.5 if latest else 0.0,
                }],
                "unresolved_questions": ["Scientific conclusions remain model responsibilities."],
                "next_observations": ["Submit conservatively when evidence is insufficient."],
                "hypotheses": [{
                    "claim": "Collected observations may distinguish exposed functions.",
                    "supporting_observation_ids": latest,
                    "contradicting_observation_ids": [],
                    "evidence_needed": ["independent scientific interpretation"],
                    "confidence": 0.5 if latest else 0.1,
                    "status": "active",
                }],
            }
            if "error_review" in required:
                state["error_review"] = None
            if "verification_cycles" in required:
                state["verification_cycles"] = ([{
                    "hypothesis_id": "fixture-h1",
                    "claim": "A read-only observation was successfully collected.",
                    "supporting_observation_ids": latest,
                    "contradicting_observation_ids": [],
                    "alternative_explanations": ["The observation may not establish a device role."],
                    "discriminating_evidence": "A successful model-visible observation.",
                    "result_observation_ids": latest,
                    "result_interpretation": "Collection succeeded without assigning hidden semantics.",
                    "disposition": "supported" if latest else "unresolved",
                    "revised_claim": None,
                }] if latest else [])
            if "uncertainty_reviews" in required:
                state["uncertainty_reviews"] = ([{
                    "claim_id": "fixture-claim-1",
                    "claim": "A read-only observation was successfully collected.",
                    "claim_scope": "other",
                    "observation_ids": latest,
                    "recency_requirement": "time_insensitive",
                    "protocol_role_distinction": "not_applicable",
                    "alternative_explanations": ["Collection alone does not establish a role."],
                    "temporal_interpretation": "not_applicable",
                    "causal_scope": "observation",
                    "causal_evidence_basis": "none",
                    "classification": "supported" if latest else "uncertain",
                    "rationale": "The fixture validates evidence plumbing only.",
                }] if latest else [])
            self.r1_reasoning_recorded = True
            return {"tool_calls": [{"id": "mock-r1-reasoning", "function": {
                "name": REASONING_TOOL_NAME, "arguments": json.dumps(state),
            }}]}
        if self.action:
            sequence.append(("bounded_binary_output", {
                "ip": "198.18.4.20", "port": 1503, "address": 0,
                "active": True,
                "duration_seconds": self.action_duration_seconds,
            }))
        if self.calls < len(sequence):
            name, arguments = sequence[self.calls]
            self.calls += 1
            return {"tool_calls": [{"id": f"mock-{self.calls}", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
        self.calls += 1
        observations = _tool_observations(messages)
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for observation in observations:
            by_tool.setdefault(str(observation.get("tool")), []).append(observation)
        def evidence(tool: str, ip: str | None = None) -> list[str]:
            values = []
            for observation in by_tool.get(tool, []):
                target = observation.get("target") or {}
                if ip is None or target.get("ip") == ip:
                    if isinstance(observation.get("observation_id"), str): values.append(observation["observation_id"])
            return values
        controller_ids = evidence("inspect_http_resource", "198.18.4.10") + evidence("explore_modbus_map", "198.18.4.10")
        process_ids = evidence("sample_modbus_series", "198.18.4.20")
        engineering_ids = evidence("inspect_http_resource", "198.18.4.40") + evidence("browse_opcua_nodes", "198.18.4.40") + evidence("inspect_s7_metadata", "198.18.4.40")
        relationship_ids = evidence("sample_cross_device_series")
        all_ids = controller_ids + process_ids + engineering_ids + relationship_ids
        report = {
            "run_id": self.run_id, "objective": self.objective,
            "systems": [
                {"ip": "198.18.4.10", "role": "controller candidate", "runtime": "OpenPLC", "protocols": ["modbus-tcp", "http"], "supporting_observation_ids": controller_ids, "uncertainties": ["read-only evidence does not prove packet direction"]},
                {"ip": "198.18.4.20", "role": "dynamic process endpoint", "runtime": None, "protocols": ["modbus-tcp"], "supporting_observation_ids": process_ids, "uncertainties": ["process semantics are inferred from temporal behavior"]},
                {"ip": "198.18.4.40", "role": "engineering observation endpoint", "runtime": None, "protocols": ["http", "opc-ua", "s7comm"], "supporting_observation_ids": engineering_ids, "uncertainties": ["protocol metadata does not prove hardware authenticity"]},
                {"ip": "198.18.4.50", "role": "independent industrial endpoint", "runtime": None, "protocols": ["modbus-tcp"], "supporting_observation_ids": evidence("fingerprint_protocol", "198.18.4.50"), "uncertainties": ["intent and authenticity are not established"]},
            ],
            "relationships": [{"source": "198.18.4.10", "destination": "198.18.4.20", "claim": "behavioral association consistent with controller/process interaction", "confidence": 0.7, "supporting_observation_ids": relationship_ids, "limitations": ["correlation is not causal proof"]}],
            "permission_assessment": [{"interface": "bounded_binary_output", "permitted": self.action, "rationale": "The action is available only when separately enabled by the operator and enforced by the broker.", "supporting_observation_ids": process_ids}],
            "objective_plan": {"summary": "Use the separately authorized bounded simulator interface and independently verify effect and restoration.", "steps": ["establish baseline", "perform bounded action if authorized", "verify effect", "verify restoration"], "supporting_observation_ids": all_ids, "requires_action": self.action, "limitations": ["action authorization is independent of investigation budget"]},
            "action_attempted": self.action, "outcome": "bounded fixture action restored" if self.action else "read-only investigation completed", "outcome_observation_ids": [], "overall_assessment": "Deterministic infrastructure golden path completed; this is not evidence of model autonomy.",
        }
        return {"tool_calls": [{"id": "final", "function": {"name": "submit_integrated_report", "arguments": json.dumps({"report": report})}}]}


def _tool_observations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str): continue
        try: value = json.loads(message["content"])
        except json.JSONDecodeError: continue
        if isinstance(value, dict): values.append(value)
    return values


def validate_live_authorization(path: Path, config: ScenarioConfig) -> dict[str, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("authorization_status") != "AUTHORIZED":
        raise ValueError("authorization_status must be AUTHORIZED")
    if not isinstance(value.get("authorization_id"), str) or not value["authorization_id"].strip():
        raise ValueError("authorization_id is required")
    if int(value.get("maximum_action_count", -1)) != config.max_actions:
        raise ValueError("authorization action count does not match frozen profile")
    if float(value.get("maximum_action_duration_seconds", -1)) != config.max_action_duration_seconds:
        raise ValueError("authorization duration does not match frozen profile")
    if value.get("allowed_operation") != "bounded_binary_output":
        raise ValueError("authorization operation is not supported")
    return {"authorization_id": value["authorization_id"],
            "authorization_sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL")); parser.add_argument("--objective", required=True); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/integrated_e1_e5_research.json")
    parser.add_argument("--reasoning-config", type=Path,
                        help="Optional B0-B4 or B2-R1 C0-C4 reasoning profile")
    parser.add_argument("--seed", type=int,
                        help="Optional OpenRouter sampling seed recorded with the episode")
    parser.add_argument("--evaluation-spec", type=Path,
                        default=ROOT / "private/integrated_e1_e5_first_qwen_spec_v2.json",
                        help=argparse.SUPPRESS)
    parser.add_argument("--deployment-manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Frozen production source/config parity manifest")
    parser.add_argument("--acceptance-record", type=Path,
                        help="Required fail-closed production acceptance artifact for live inference")
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--mock", action="store_true"); mode.add_argument("--execute-live-model", action="store_true")
    parser.add_argument("--enable-actions", action="store_true")
    parser.add_argument("--authorization-file", type=Path,
                        help="Private operator-approved authorization packet; required for live actions")
    parser.add_argument("--mock-action", action="store_true")
    parser.add_argument(
        "--mock-live-action",
        action="store_true",
        help="Run the deterministic lifecycle while routing its single action through the deployed simulator broker.",
    )
    args = parser.parse_args()
    if not args.worker_url: parser.error("--worker-url or SCADAVER_WORKER_URL is required")
    if args.mock_action and not args.mock: parser.error("--mock-action is valid only with --mock")
    if args.mock_live_action and not args.mock:
        parser.error("--mock-live-action is valid only with --mock")
    if args.mock_action and args.mock_live_action:
        parser.error("--mock-action and --mock-live-action are mutually exclusive")
    if args.enable_actions and args.mock:
        parser.error("--enable-actions is reserved for --execute-live-model; use a mock action flag with --mock")
    if args.authorization_file and not args.enable_actions:
        parser.error("--authorization-file is valid only with --enable-actions")
    if (args.enable_actions or args.mock_live_action) and os.environ.get("SCADAVER_ENABLE_SITL_ACTIONS") != "1":
        parser.error("deployed actions require SCADAVER_ENABLE_SITL_ACTIONS=1")
    if args.out_dir.exists() and any(args.out_dir.iterdir()): parser.error("--out-dir must be new or empty")
    try:
        parity = verify_runtime_manifest(args.deployment_manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, DeploymentParityError) as exc:
        parser.error(f"production deployment parity failed: {exc}")
    acceptance: dict[str, Any] | None = None
    if args.execute_live_model:
        if not args.acceptance_record or not args.acceptance_record.is_file():
            parser.error("live inference requires --acceptance-record from production_acceptance")
        try:
            acceptance = json.loads(args.acceptance_record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"production acceptance record is unreadable: {exc}")
        if acceptance.get("EXPERIMENT_READY") is not True:
            parser.error("production acceptance record does not report EXPERIMENT_READY=true")
        if acceptance.get("openrouter_calls") != 0 or acceptance.get("deployed_writes") != 0:
            parser.error("production acceptance record is not a no-model/no-write gate")
        accepted_manifest = (acceptance.get("parity") or {}).get("manifest_sha256")
        if accepted_manifest != parity.get("manifest_sha256"):
            parser.error("production acceptance manifest does not match current runtime")
    config = ScenarioConfig.from_json(args.config)
    reasoning_profile: ReasoningProfile | B2R1Profile | None = None
    if args.reasoning_config:
        reasoning_profile = (
            B2R1Profile.from_json(args.reasoning_config)
            if is_b2_r1_config(args.reasoning_config)
            else ReasoningProfile.from_json(args.reasoning_config)
        )
    if reasoning_profile:
        expected_base = (args.reasoning_config.parent / reasoning_profile.base_config).resolve()
        if args.config.resolve() != expected_base:
            parser.error(
                f"reasoning profile {reasoning_profile.condition} requires base config {expected_base}"
            )
    if isinstance(reasoning_profile, B2R1Profile):
        reasoning_manager: ReasoningStateManager | B2R1StateManager | None = (
            B2R1StateManager(reasoning_profile)
        )
    else:
        reasoning_manager = (ReasoningStateManager(reasoning_profile)
                             if reasoning_profile and reasoning_profile.enabled else None)
    authorization = None
    if args.enable_actions:
        if not args.authorization_file:
            parser.error("--enable-actions requires --authorization-file")
        try:
            authorization = validate_live_authorization(args.authorization_file, config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid live authorization: {exc}")
    limits = EpisodeLimits(config.max_model_calls, config.max_tool_calls, config.max_input_tokens,
                           config.max_total_output_tokens, config.max_network_observation_seconds,
                           config.max_time_series_samples, config.max_api_cost_usd,
                           config.reserved_completion_steps)
    budget = EpisodeBudget(limits)
    args.out_dir.mkdir(parents=True)
    _write(args.out_dir / "deployment_parity.json", parity)
    _write(args.out_dir / "preflight.json", preflight(args.worker_url, "heterogeneous_e4"))
    transport = AccessWorkerTransport(args.worker_url)
    transport.start_episode("heterogeneous_e4", required_tool_names={item["function"]["name"] for item in E4_TOOL_SCHEMAS})
    action_enabled = args.mock_action or args.mock_live_action or args.enable_actions
    deployed_action = args.mock_live_action or args.enable_actions
    token = os.environ.get("SCADAVER_SITL_ACTION_TOKEN") if deployed_action else "mock-token"
    if deployed_action:
        adapter = ModbusSimulatorAdapter()
        action_controller = OpenPLCControllerCoordinator()
        action_sleeper = __import__("time").sleep
    elif args.mock_action:
        adapter = ControlledProcessFixture()
        action_controller = FixtureController(adapter)
        action_sleeper = adapter.advance_time
    else:
        adapter = DeterministicAdapter()
        action_controller = None
        action_sleeper = __import__("time").sleep
    stop_path = (Path(os.environ.get("SCADAVER_ACTION_EMERGENCY_STOP", "/tmp/scadaver-action-emergency-stop"))
                 if deployed_action else args.out_dir / "emergency_stop")
    broker = ActionBroker(
        adapter,
        enabled=True,
        authorization_token=token,
        audit_path=args.out_dir / "action_audit.jsonl",
        emergency_stop_path=stop_path,
        max_duration_seconds=0.25 if args.mock_live_action else config.max_action_duration_seconds,
        max_actions=config.max_actions,
        controller_coordinator=action_controller,
        transaction_root=args.out_dir / "e5_transactions",
        restoration_mode="operational_recovery" if deployed_action else "exact_snapshot",
        sleeper=action_sleeper,
    ) if action_enabled else None
    tools = [item for item in E4_TOOL_SCHEMAS if item["function"]["name"] != "submit_objective_pathway"]
    if reasoning_profile:
        reasoning_schema = (b2_r1_tool_schema(reasoning_profile)
                            if isinstance(reasoning_profile, B2R1Profile)
                            else reasoning_tool_schema(reasoning_profile))
        if reasoning_schema:
            tools.append(reasoning_schema)
    tools += ([action_schema(config.max_action_duration_seconds)] if action_enabled else []) + [FINAL_SCHEMA]
    client = (IntegratedMockClient(
        args.objective,
        action_enabled,
        args.out_dir.name,
        action_duration_seconds=(0.25 if args.mock_live_action else
                                 config.max_action_duration_seconds if args.mock_action else 0.0),
    ) if args.mock else OpenRouterClient(
        config.model,
        config.api_key_env,
        request_timeout=config.request_timeout_seconds,
        max_tokens=config.max_output_tokens,
        retries=config.inference_retries,
        seed=args.seed,
    ))
    system_prompt = (ROOT / "prompts/integrated_system_v1.md").read_text()
    if reasoning_profile:
        addition = (load_b2_r1_prompt(reasoning_profile, ROOT)
                    if isinstance(reasoning_profile, B2R1Profile)
                    else load_reasoning_prompt(reasoning_profile, ROOT))
        if addition:
            system_prompt += "\n\n" + addition
    task_prompt = (ROOT / "prompts/integrated_e1_e5_v1.md").read_text() + "\nObjective: " + args.objective
    try:
        outcome = run_agent(client, CompositeExecutor(transport, broker, token, reasoning_manager), allowed_cidr=config.allowed_cidr, permitted_ports=config.permitted_ports, max_steps=config.max_steps, system_prompt=system_prompt, task_prompt=task_prompt, episode_deadline_seconds=config.episode_deadline_seconds, tool_schemas=tools, report_validator=validate_report, submission_tool_name="submit_integrated_report", final_submission_max_tokens=config.final_submission_max_output_tokens, reserved_completion_steps=config.reserved_completion_steps, budget=budget, max_context_chars=config.max_context_chars, context_tail_groups=config.context_tail_groups)
    finally:
        transport.close_episode()
    evaluation = evaluate_integrated(outcome.messages, outcome.report, load_spec(args.evaluation_spec), action_required=action_enabled)
    b2_r1_readonly_evaluation = (
        evaluate_b2_r1_readonly(
            outcome.messages, outcome.report, load_spec(args.evaluation_spec),
        ) if isinstance(reasoning_profile, B2R1Profile) and not action_enabled else None
    )
    action_mode = "deployed" if deployed_action else ("fixture" if args.mock_action else "disabled")
    reasoning_state = reasoning_manager.snapshot() if reasoning_manager else None
    record = {"status": outcome.stop_reason, "messages": outcome.messages, "report": outcome.report, "diagnostics": outcome.diagnostics, "statistics": outcome.statistics, "config": config.public_snapshot(), "inference_seed": args.seed, "objective": args.objective, "action_enabled": action_enabled, "action_mode": action_mode, "authorization": authorization, "mock": args.mock, "evaluation": evaluation, "b2_r1_readonly_evaluation": b2_r1_readonly_evaluation, "inference_requests": client.request_records, "reasoning_profile": reasoning_profile.public_snapshot() if reasoning_profile else None, "reasoning_state": reasoning_state, "reasoning_diagnostics": compute_reasoning_diagnostics(outcome.messages, outcome.diagnostics, reasoning_state), "initial_condition_record": (((acceptance or {}).get("testbed") or {}).get("initial_condition_record")), "reproducibility": _reproducibility_record(args, config, tools, system_prompt, task_prompt, parity, client.request_records)}
    _write(args.out_dir / "episode.json", record)
    print(json.dumps(record["evaluation"], indent=2))


if __name__ == "__main__":
    main()
