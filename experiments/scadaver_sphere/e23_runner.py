"""Reusable E2/E3 autonomous read-only runner. Live mode uses OpenRouter; mock mode never does."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import run_agent
from .config import ScenarioConfig
from .e23_evaluation import evaluate_e2, evaluate_e3, summarize
from .openrouter import OpenRouterClient
from .schemas import E2_TOOL_SCHEMAS, E3_METADATA_TOOL_SCHEMAS, E3_TELEMETRY_TOOL_SCHEMAS, validate_inventory, validate_runtime_process_report
from .transport import AccessWorkerTransport, WorkerTransportError

ROOT = Path(__file__).resolve().parent
TRUTH = ROOT / "e23_ground_truth.json"
PROMPTS = ROOT / "prompts"


class MockClient:
    """Deterministic fixture client used only for offline harness validation."""
    request_records: list[dict[str, Any]] = []
    def __init__(self, experiment: str) -> None: self.experiment, self.calls = experiment, 0; self.request_records = []
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"tool_calls": [{"id": "mock-discover", "function": {"name": "discover_hosts", "arguments": '{"cidr":"198.18.4.0/24"}'}}]}
        if self.experiment == "e2":
            report = {"run_id": "mock", "hosts": [], "overall_assessment": "No claims without evidence."}
            name = "submit_inventory"
        else:
            report = {"run_id": "mock", "device_assessments": [], "runtime_assessments": [], "process_assessment": {"process_class": "unknown", "confidence": 0.0, "supporting_observation_ids": [], "alternative_explanations": ["no evidence"], "relationships": []}, "overall_assessment": "No claims without evidence."}
            name = "submit_fingerprint_assessment"
        return {"tool_calls": [{"id": "mock-submit", "function": {"name": name, "arguments": json.dumps({"report": report})}}]}


def _write(path: Path, value: Any) -> None: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def profile(experiment: str, condition: str) -> tuple[str, list[dict[str, Any]], str, Any, str]:
    if experiment == "e2": return "heterogeneous_e2", E2_TOOL_SCHEMAS, "e2_heterogeneous_discovery_v1.md", validate_inventory, "submit_inventory"
    if condition == "telemetry": return "heterogeneous_e3_telemetry", E3_TELEMETRY_TOOL_SCHEMAS, "e3_cyber_physical_inference_v1.md", validate_runtime_process_report, "submit_fingerprint_assessment"
    return "heterogeneous_e3_metadata", E3_METADATA_TOOL_SCHEMAS, "e3_cyber_physical_inference_v1.md", validate_runtime_process_report, "submit_fingerprint_assessment"


def preflight(worker_url: str, harness_profile: str) -> dict[str, Any]:
    transport = AccessWorkerTransport(worker_url)
    try:
        transport.start_episode(harness_profile, required_tool_names={"discover_hosts", "discover_services", "fingerprint_protocol"})
        result = transport.execute("discover_hosts", {"cidr": "198.18.4.0/24"})
        return {"worker": "reachable", "discovery_status": result.get("status"), "reachable_hosts": result.get("hosts", [])}
    finally:
        transport.close_episode()


def run(config: ScenarioConfig, experiment: str, condition: str, worker_url: str, out_dir: Path, episodes: int, mock: bool) -> list[dict[str, Any]]:
    harness_profile, tools, prompt_file, validator, submission = profile(experiment, condition)
    if config.harness_profile != harness_profile: raise ValueError("config harness_profile does not match experiment/condition")
    truth = json.loads(TRUTH.read_text())
    system = (PROMPTS / "system_v1.md").read_text(); task = (PROMPTS / prompt_file).read_text()
    records: list[dict[str, Any]] = []
    for index in range(1, episodes + 1):
        episode_id = f"{experiment}-{condition}-{index:03d}"; episode_dir = out_dir / episode_id; episode_dir.mkdir(parents=True)
        started = time.monotonic(); client = MockClient(experiment) if mock else OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
        transport = AccessWorkerTransport(worker_url); record: dict[str, Any] = {"episode_id": episode_id, "started_utc": datetime.now(timezone.utc).isoformat(), "config": config.public_snapshot(), "condition": condition, "system_prompt": system, "task_prompt": task, "tool_schemas": tools, "status": "infrastructure_failure", "mock": mock}
        try:
            transport.start_episode(harness_profile, required_tool_names={tool["function"]["name"] for tool in tools})
            outcome = run_agent(client, transport, allowed_cidr=config.allowed_cidr, permitted_ports=config.permitted_ports, max_steps=config.max_steps, system_prompt=system, task_prompt=task, episode_deadline_seconds=config.episode_deadline_seconds, tool_schemas=tools, report_validator=validator, submission_tool_name=submission, final_submission_max_tokens=config.final_submission_max_output_tokens)
            record.update({"stop_reason": outcome.stop_reason, "messages": outcome.messages, "report": outcome.report, "diagnostics": outcome.diagnostics, "status": "completed" if outcome.stop_reason.startswith("submitted_") else outcome.stop_reason})
            record["tool_calls"] = sum(1 for message in outcome.messages if message.get("role") == "tool")
            record["evaluation"] = (evaluate_e2 if experiment == "e2" else evaluate_e3)(outcome.report, truth) if record["status"] == "completed" else {"valid_submission": False}
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try: transport.close_episode()
            except RuntimeError: pass
        record["duration_seconds"] = round(time.monotonic() - started, 3); record["inference_requests"] = client.request_records
        usage = [item.get("usage") or {} for item in client.request_records]
        record["usage"] = {"prompt_tokens": sum(x.get("prompt_tokens", 0) for x in usage), "completion_tokens": sum(x.get("completion_tokens", 0) for x in usage), "total_tokens": sum(x.get("total_tokens", 0) for x in usage), "cost": sum(x.get("cost", 0) for x in usage)}
        _write(episode_dir / "episode.json", record); records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--experiment", choices=("e2", "e3"), required=True); parser.add_argument("--condition", choices=("telemetry", "metadata"), default="metadata"); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL")); parser.add_argument("--out-dir", type=Path, required=True); parser.add_argument("--episodes", type=int, default=1); parser.add_argument("--mock", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--model"); args = parser.parse_args()
    config = ScenarioConfig.from_json(args.config); config = replace(config, model=args.model or config.model)
    expected, tools, prompt, _, _ = profile(args.experiment, args.condition)
    if config.harness_profile != expected: parser.error("configuration profile does not match selected experiment/condition")
    if args.dry_run: print(json.dumps({"experiment": args.experiment, "condition": args.condition, "profile": expected, "prompt": prompt, "tools": [x["function"]["name"] for x in tools], "openrouter_called": False}, indent=2)); return
    if not args.worker_url: parser.error("--worker-url or SCADAVER_WORKER_URL is required")
    if not args.mock and (not config.live_openrouter or not config.live_worker): parser.error("live execution requires reviewed live_openrouter/live_worker configuration")
    if args.out_dir.exists() and any(args.out_dir.iterdir()): parser.error("--out-dir must be new or empty")
    args.out_dir.mkdir(parents=True); _write(args.out_dir / "preflight.json", preflight(args.worker_url, expected)); records = run(config, args.experiment, args.condition, args.worker_url, args.out_dir, args.episodes, args.mock); summary = summarize(records, args.experiment); _write(args.out_dir / f"{args.experiment}_summary.json", summary); (args.out_dir / "summary.md").write_text("# Experiment Summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")

if __name__ == "__main__": main()
