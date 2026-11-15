"""E1-B enhanced industrial-device fingerprinting orchestration."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.scadaver_sphere.agent import run_agent
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.e1b_evaluation import evaluate_e1b, reliability, summarize_e1b
from experiments.scadaver_sphere.openrouter import OpenRouterClient, OpenRouterError
from experiments.scadaver_sphere.schemas import tool_schemas
from experiments.scadaver_sphere.transport import AccessWorkerTransport, WorkerTransportError

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
TRUTH_PATH = ROOT / "evaluation" / "e1_ground_truth.json"
SYSTEM_PROMPT_PATH = PROMPTS / "fingerprinting_roles_system_v1.md"
TASK_PROMPT_PATH = PROMPTS / "fingerprinting_roles_task_v1.md"
ENHANCED_PROFILE = "fingerprinting_enhanced"
SAFETY_LIMITS = {
    "read_only": True,
    "max_register_or_coil_count": 10,
    "max_temporal_samples": 5,
    "min_probe_interval_seconds": 1,
    "max_temporal_duration_seconds": 20,
    "max_probe_operations_per_session": 80,
    "http_path": "/",
    "http_response_bytes": 4096,
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _usage(client: OpenRouterClient) -> dict[str, Any]:
    usage = [request.get("usage") or {} for request in client.request_records]
    return {
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in usage),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        "cost": sum(item.get("cost", 0) for item in usage),
    }


def _validate_config(config: ScenarioConfig) -> None:
    if config.harness_profile != ENHANCED_PROFILE:
        raise ValueError("E1-B requires harness_profile=fingerprinting_enhanced")
    if config.max_observations < 1 or config.max_observations > 64:
        raise ValueError("E1-B max_observations must be between 1 and 64")


def _episode_record(config: ScenarioConfig, episode_id: str, worker_url: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    system_prompt, task_prompt = SYSTEM_PROMPT_PATH.read_text(), TASK_PROMPT_PATH.read_text()
    schema = tool_schemas(ENHANCED_PROFILE)
    required_tools = {tool["function"]["name"] for tool in schema}
    client = OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
    transport = AccessWorkerTransport(worker_url)
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.public_snapshot(),
        "provider": "openrouter",
        "system_prompt": system_prompt,
        "task_prompt": task_prompt,
        "tool_schemas": schema,
        "safety_limits": {**SAFETY_LIMITS, "max_public_observations": config.max_observations},
        "status": "infrastructure_failure",
    }
    started = time.monotonic()
    evaluation: dict[str, Any] | None = None
    try:
        transport.start_episode(ENHANCED_PROFILE, required_tool_names=required_tools)
        outcome = run_agent(
            client,
            transport,
            allowed_cidr=config.allowed_cidr,
            permitted_ports=config.permitted_ports,
            max_steps=config.max_steps,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            episode_deadline_seconds=config.episode_deadline_seconds,
            tool_schemas=schema,
        )
        record.update({"stop_reason": outcome.stop_reason, "messages": outcome.messages, "report": outcome.report})
        output_limited = any(request.get("finish_reason") == "length" for request in client.request_records)
        if outcome.stop_reason == "submitted_inventory":
            record["status"] = "completed"
        elif output_limited:
            record["status"] = "output_limit_termination"
        else:
            record["status"] = outcome.stop_reason
        evaluation = evaluate_e1b(outcome.report, json.loads(TRUTH_PATH.read_text()), outcome.messages)
    except (OpenRouterError, WorkerTransportError, RuntimeError, ValueError) as exc:
        record["error"] = str(exc)
    finally:
        try:
            transport.close_episode()
        except RuntimeError:
            pass
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    record["inference_requests"] = client.request_records
    record["usage"] = _usage(client)
    record["reliability"] = reliability(record)
    return record, evaluation


def run_e1b(config: ScenarioConfig, worker_url: str, out_dir: Path, episodes: int) -> list[dict[str, Any]]:
    _validate_config(config)
    truth = json.loads(TRUTH_PATH.read_text())
    _write_json(out_dir / "config_snapshot.json", {
        "config": config.public_snapshot(), "provider": "openrouter", "tool_schemas": tool_schemas(ENHANCED_PROFILE),
        "safety_limits": {**SAFETY_LIMITS, "max_public_observations": config.max_observations},
        "ground_truth_path": str(TRUTH_PATH), "ground_truth_is_evaluator_only": True,
    })
    records: list[dict[str, Any]] = []
    for index in range(1, episodes + 1):
        episode_id = f"e1b-{index:03d}"
        episode_dir = out_dir / episode_id
        episode_dir.mkdir()
        record, evaluation = _episode_record(config, episode_id, worker_url)
        _write_json(episode_dir / "episode.json", record)
        if evaluation is not None:
            _write_json(episode_dir / "evaluation.json", evaluation)
            record["evaluation"] = evaluation
        records.append(record)
    return records


def _metric(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    return value.get("mean") if isinstance(value, dict) else "Not available"


def render_comparison(e1a_dir: Path | None, e1b: dict[str, Any]) -> str:
    baseline: dict[str, Any] = {}
    if e1a_dir is not None and (e1a_dir / "e1_aggregate.json").is_file():
        baseline = json.loads((e1a_dir / "e1_aggregate.json").read_text())
    e1a_submission_rate = baseline.get("completed", 0) / baseline["episodes"] if baseline.get("episodes") else "Not available"
    rows = [
        ("Host discovery F1", _metric(baseline, "host_discovery_f1"), _metric(e1b, "host_discovery_f1")),
        ("Service discovery F1", _metric(baseline, "service_discovery_f1"), _metric(e1b, "service_discovery_f1")),
        ("Protocol identification F1", _metric(baseline, "protocol_identification_f1"), _metric(e1b, "protocol_identification_f1")),
        ("Functional-role accuracy", "Not available", _metric(e1b, "accuracy_over_all_ground_truth_devices")),
        ("Classification coverage", "Not available", _metric(e1b, "classification_coverage")),
        ("Unsupported authenticity claims", baseline.get("unsupported_runtime_claims", "Not available"), e1b.get("unsupported_authenticity_claims", "Not available")),
        ("Valid submission rate", e1a_submission_rate, e1b.get("reliability", {}).get("valid_submission_rate", "Not available")),
        ("Tool usage and cost", "See E1-A episodes", "See E1-B episode usage"),
    ]
    lines = ["# E1-A Versus E1-B", "", "E1-A artifacts are read-only inputs. E1-B uses the enhanced observation profile; no causal improvement claim follows until E1-B is executed.", "", "| Metric | E1-A baseline | E1-B enhanced |", "|---|---:|---:|"]
    lines += [f"| {name} | {left} | {right} |" for name, left, right in rows]
    return "\n".join(lines) + "\n"


def render_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# E1-B Five-Run Report", "", "Preliminary, sequential enhanced-fingerprinting experiment. Functional role is distinct from exact runtime identity or physical PLC identification.", "", "| Episode | Status | Host F1 | Service F1 | Modbus F1 | Role accuracy/all | Coverage | Tokens | Cost |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        evaluation = record.get("evaluation", {})
        role = evaluation.get("functional_role_identification", {})
        usage = record.get("usage", {})
        lines.append(f"| {record['episode_id']} | {record['status']} | {evaluation.get('host_discovery', {}).get('f1', 'N/A')} | {evaluation.get('service_discovery', {}).get('f1', 'N/A')} | {evaluation.get('protocol_identification', {}).get('f1', 'N/A')} | {role.get('accuracy_over_all_ground_truth_devices', 'N/A')} | {role.get('classification_coverage', 'N/A')} | {usage.get('total_tokens', 0)} | {usage.get('cost', 0):.8f} |")
    lines += ["", "## Aggregate", "", f"- Completed inventories: {summary['completed']} / {summary['episodes']}", f"- Status counts: {summary['status_counts']}", f"- Reliability: {summary['reliability']}", "", "## Interpretation Guardrails", "", "- Correct functional-controller classification does not identify a physical PLC or exact controller runtime.", "- Constant values do not by themselves establish emulation; changing values do not by themselves establish runtime authenticity.", "- Semantic support for non-unknown role claims is flagged for manual review in each episode evaluation."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model")
    parser.add_argument("--harness-profile", default=ENHANCED_PROFILE)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout-seconds", type=int)
    parser.add_argument("--episode-deadline-seconds", type=int)
    parser.add_argument("--e1a-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    config = ScenarioConfig.from_json(args.config)
    config = replace(config, model=args.model or config.model, harness_profile=args.harness_profile, max_steps=args.max_steps or config.max_steps, max_output_tokens=args.max_output_tokens or config.max_output_tokens, request_timeout_seconds=args.request_timeout_seconds or config.request_timeout_seconds, episode_deadline_seconds=args.episode_deadline_seconds or config.episode_deadline_seconds)
    try:
        _validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    validation = {"config": config.public_snapshot(), "episodes": args.episodes, "prompts": [str(SYSTEM_PROMPT_PATH), str(TASK_PROMPT_PATH)], "prompts_present": SYSTEM_PROMPT_PATH.is_file() and TASK_PROMPT_PATH.is_file(), "tool_names": [tool["function"]["name"] for tool in tool_schemas(ENHANCED_PROFILE)], "truth_present": TRUTH_PATH.is_file(), "api_key_env": config.api_key_env, "worker_url_supplied": bool(args.worker_url), "planned_output": ["config_snapshot.json", *[f"e1b-{index:03d}/episode.json" for index in range(1, args.episodes + 1)], "e1b_aggregate.json", "e1b_five_run_report.md", "e1a_vs_e1b_comparison.md"]}
    if args.dry_run:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if not config.live_openrouter or not config.live_worker:
        parser.error("E1-B execution requires a reviewed config with both live gates enabled")
    if not args.worker_url:
        parser.error("--worker-url is required unless --dry-run is used")
    args.out_dir.mkdir(parents=True)
    records = run_e1b(config, args.worker_url, args.out_dir, args.episodes)
    summary = summarize_e1b(records)
    _write_json(args.out_dir / "e1b_aggregate.json", summary)
    (args.out_dir / "e1b_five_run_report.md").write_text(render_report(records, summary))
    (args.out_dir / "e1a_vs_e1b_comparison.md").write_text(render_comparison(args.e1a_dir, summary))


if __name__ == "__main__":
    main()
