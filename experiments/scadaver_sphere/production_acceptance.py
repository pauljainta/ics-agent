"""Single fail-closed, no-model, no-write production readiness gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .acceptance import fault_security, tool_coverage
from .budgets import EpisodeBudget, EpisodeLimits, ModelBudgetAdmissionError
from .deployment_parity import verify_runtime_manifest
from .e5_evaluation import EVALUATOR_VERSION, evaluate_e5
from .e5_fixture import ControlledProcessFixture
from .evidence_sufficiency import run as run_evidence_sufficiency
from .heterogeneous_sitl_v1.action_broker import ActionBroker, BrokerDenied
from .operator_workflow import baseline as operator_baseline
from .production_path_smoke import production_mock, reasoning_profile_mocks, synthetic_checks
from .schemas import E4_TOOL_SCHEMAS
from .tool_results import CONTRACT_VERSION, normalize_tool_result


ROOT = Path(__file__).resolve().parent


def _budget_acceptance() -> dict[str, Any]:
    cases: dict[str, bool] = {}
    exact_calls = EpisodeBudget(EpisodeLimits(max_model_calls=1))
    exact_calls.record_model_call()
    cases["exact_model_call_limit"] = not exact_calls.can_model_call()

    exact_tools = EpisodeBudget(EpisodeLimits(max_tool_calls=1))
    exact_tools.record_tool("discover_hosts", {})
    cases["exact_tool_call_limit"] = exact_tools.check_tool("discover_hosts", {}) == "tool_calls"

    near = EpisodeBudget(EpisodeLimits(max_input_tokens=100, max_output_tokens=50))
    try:
        near.admit_model_request(estimated_input_tokens=60, requested_output_tokens=10,
                                 reserve_input_tokens=60, reserve_output_tokens=10)
    except ModelBudgetAdmissionError as exc:
        cases["conservative_input_admission"] = exc.resource == "input_tokens"
    else:
        cases["conservative_input_admission"] = False

    crossed = EpisodeBudget(EpisodeLimits(max_input_tokens=100))
    crossed.ingest_request_records([
        {"attempt": 1, "status": "transport_error"},
        {"attempt": 2, "usage": {"prompt_tokens": 101, "completion_tokens": 1}},
    ])
    snapshot = crossed.snapshot()
    cases["retry_and_unknown_usage_accounted"] = (
        snapshot["usage"]["provider_attempts"] == 2
        and snapshot["usage"]["unknown_usage_attempts"] == 1
    )
    cases["provider_crossing_explicit"] = (
        snapshot["crossings"] and snapshot["crossings"][0]["resource"] == "input_tokens"
        and "input_tokens" in snapshot["exhausted"]
    )
    return {"passed": all(cases.values()), "cases": cases,
            "accounting_semantics": snapshot["accounting_semantics"]}


def _tool_contract_acceptance() -> dict[str, Any]:
    names = [item["function"]["name"] for item in E4_TOOL_SCHEMAS]
    schemas_closed = all(
        item["function"]["parameters"].get("additionalProperties") is False
        for item in E4_TOOL_SCHEMAS
    )
    failure = normalize_tool_result(
        {"status": "error", "error": "connection_timeout",
         "error_category": "connection_timeout"},
        tool="fixture", attempted=True,
    )
    fields = {
        "result_contract", "operation_attempted", "operation_executed", "retryable",
        "budget_consumed", "state_change_possible", "error_category",
    }
    cases = {
        "unique_tool_names": len(names) == len(set(names)),
        "closed_argument_schemas": schemas_closed,
        "versioned_error_contract": failure["result_contract"] == CONTRACT_VERSION,
        "required_error_fields": fields.issubset(failure),
        "read_only_failure_metadata": failure["state_change_possible"] is False,
    }
    return {"passed": all(cases.values()), "cases": cases, "tool_names": names,
            "result_contract": CONTRACT_VERSION}


def _broker_evaluator_acceptance() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = ControlledProcessFixture()
    denied = False
    try:
        ActionBroker(fixture).submit(
            "bounded_inlet_test", {"active": True, "duration_seconds": 1.0}, token="absent")
    except BrokerDenied:
        denied = True
    broker = {"passed": denied, "cases": {
        "disabled_by_default": denied,
        "no_deployed_adapter_used": True,
        "no_authorization_token_created": True,
    }}

    base = {"inlet": False, "level_raw": 500, "flow_raw": 0,
            "scan_counter": 10, "time": 1.0}
    acknowledged = {**base, "inlet": True, "flow_raw": 100, "scan_counter": 11, "time": 2.0}
    during = {**acknowledged, "level_raw": 520, "scan_counter": 12, "time": 3.0}
    restored = {**base, "time": 4.0}
    success = evaluate_e5({
        "action_requested": True, "action_authorized": True, "action_accepted": True,
        "action_applied": True, "resolved_active": True, "before": base,
        "acknowledged": acknowledged, "during": during, "restored": restored,
        "exact_state_restored": True, "restoration_verified": True,
        "controller_health": {"passed": True},
    }, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    recovery_only = evaluate_e5({
        "action_requested": True, "action_authorized": True, "action_accepted": True,
        "action_applied": False, "resolved_active": True, "before": base,
        "acknowledged": base, "during": base,
        "restored": {**restored, "level_raw": 550},
        "restoration_verified": False,
    }, {"minimum_level_increase_raw": 10, "minimum_scan_advance": 1})
    evaluator_cases = {
        "version_is_causal_v3": EVALUATOR_VERSION == "e5-causal-attribution-v3",
        "genuine_intervention_supported": success["physical_objective_achieved"] is True,
        "recovery_only_change_rejected": recovery_only["physical_objective_achieved"] is False,
        "unapplied_is_not_success": recovery_only["action_applied"] is False,
    }
    return broker, {"passed": all(evaluator_cases.values()), "cases": evaluator_cases,
                    "evaluator_version": EVALUATOR_VERSION}


def run_acceptance(worker_url: str | None, out_dir: Path, *, offline: bool = False) -> dict[str, Any]:
    """Run all mandatory gates without model inference, tokens, or deployed writes."""
    out_dir.mkdir(parents=True, exist_ok=False)
    parity = verify_runtime_manifest()
    tools = _tool_contract_acceptance()
    budgets = _budget_acceptance()
    broker, evaluator = _broker_evaluator_acceptance()
    synthetic = synthetic_checks()

    testbed: dict[str, Any] = {"skipped": offline, "TESTBED_READY": False}
    deployed_tools: dict[str, Any] = {"skipped": offline, "passed": False}
    evidence: dict[str, Any] = {"skipped": offline, "all_ready_tasks_pass": False}
    production: dict[str, Any] = {"skipped": offline, "passed": False}
    profiles: dict[str, Any] = {"skipped": offline, "passed": False}
    if not offline:
        if not worker_url:
            raise ValueError("worker_url is required for deployed acceptance")
        testbed = operator_baseline()
        deployed_tools = tool_coverage(worker_url)
        evidence = run_evidence_sufficiency(worker_url)
        production = production_mock(worker_url, out_dir)
        profiles = reasoning_profile_mocks(worker_url, out_dir)
        security = fault_security(worker_url)
        deployed_tools["fault_security"] = security
        deployed_tools["passed"] = deployed_tools.get("passed") is True and security["passed"] is True

    statuses = {
        "DEPLOYMENT_READY": parity["passed"] is True,
        "TOOLS_READY": tools["passed"] is True and (offline or deployed_tools["passed"] is True),
        "BUDGETS_READY": budgets["passed"] is True,
        "TESTBED_READY": (not offline and testbed.get("TESTBED_READY") is True),
        "BROKER_READY": broker["passed"] is True,
        "EVALUATOR_READY": evaluator["passed"] is True,
        "FULL_SMOKE_READY": synthetic["passed"] is True and (
            not offline and production["passed"] is True and profiles["passed"] is True),
        "SCIENTIFIC_CONFIG_FROZEN": parity["scientific_configuration"]["passed"] is True,
    }
    statuses["EXPERIMENT_READY"] = all(statuses.values())
    result = {
        "acceptance_version": "scadaver-production-acceptance-v1",
        **statuses,
        "offline": offline,
        "openrouter_calls": 0,
        "deployed_writes": 0,
        "action_tokens_created": 0,
        "parity": parity,
        "tools": tools,
        "deployed_tool_coverage": deployed_tools,
        "budgets": budgets,
        "testbed": testbed,
        "broker": broker,
        "evaluator": evaluator,
        "synthetic_smoke": synthetic,
        "production_mock": production,
        "reasoning_profile_mocks": profiles,
        "evidence_sufficiency": evidence,
    }
    (out_dir / "production_acceptance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must not already exist")
    try:
        result = run_acceptance(args.worker_url, args.out_dir, offline=args.offline)
    except Exception as exc:
        print(json.dumps({"EXPERIMENT_READY": False, "error": type(exc).__name__,
                          "detail": str(exc)}, indent=2))
        return 1
    print(json.dumps({key: value for key, value in result.items()
                      if key.endswith("_READY") or key in {"openrouter_calls", "deployed_writes"}},
                     indent=2, sort_keys=True))
    return 0 if result["EXPERIMENT_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
