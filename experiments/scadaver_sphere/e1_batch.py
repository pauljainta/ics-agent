"""Sequential, session-isolated E1 batch runner. Ground truth is evaluator-only."""

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
from experiments.scadaver_sphere.e1_evaluation import evaluate_e1, summarize
from experiments.scadaver_sphere.openrouter import OpenRouterClient, OpenRouterError
from experiments.scadaver_sphere.schemas import TOOL_SCHEMAS
from experiments.scadaver_sphere.scripts.run_live import PROMPTS
from experiments.scadaver_sphere.transport import AccessWorkerTransport, WorkerTransportError

TRUTH_PATH = Path(__file__).resolve().parent / "evaluation" / "e1_ground_truth.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _render_report(records: list[dict[str, Any]], summary: dict[str, Any], config: ScenarioConfig) -> str:
    lines = [
        "# E1 Five-Run Report", "", "Preliminary, sequential, Modbus-focused evidence only.", "",
        "## Configuration", "", f"- Model: {config.model}", f"- Episodes requested: {len(records)}",
        f"- Action budget: {config.max_steps}", f"- Output budget: {config.max_output_tokens}",
        f"- Request timeout: {config.request_timeout_seconds} seconds", f"- Episode deadline: {config.episode_deadline_seconds} seconds", "",
        "## Per-Episode Results", "", "| Episode | Status | Host F1 | Service F1 | Modbus F1 | Role coverage | Runtime claims | Tokens | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        evaluation = record.get("evaluation", {})
        usage = record.get("usage", {})
        role = evaluation.get("role_inference", {})
        lines.append(f"| {record['episode_id']} | {record['status']} | {evaluation.get('host_discovery', {}).get('f1', 'N/A')} | {evaluation.get('service_discovery', {}).get('f1', 'N/A')} | {evaluation.get('protocol_identification', {}).get('f1', 'N/A')} | {role.get('coverage', 'N/A')} | {evaluation.get('unsupported_runtime_claims', 'N/A')} | {usage.get('total_tokens', 0)} | {usage.get('cost', 0):.8f} |")
    lines += [
        "", "## Aggregate", "", f"- Completed inventories: {summary['completed']} / {summary['episodes']}",
        f"- Host-discovery F1 mean/stdev: {summary['host_discovery_f1']}",
        f"- Service-discovery F1 mean/stdev: {summary['service_discovery_f1']}",
        f"- Modbus-identification F1 mean/stdev: {summary['protocol_identification_f1']}",
        f"- Status counts: {dict(summary['failures'])}", "", "## Limitations", "",
        "Five runs are preliminary. This evaluates bounded discovery and Modbus TCP evidence in one realization; it does not establish arbitrary protocol discovery or real PLC-runtime identification.",
    ]
    return "\n".join(lines) + "\n"


def run_batch(config: ScenarioConfig, worker_url: str, out_dir: Path, episodes: int) -> list[dict[str, Any]]:
    truth = json.loads(TRUTH_PATH.read_text())
    system_prompt = (PROMPTS / "system_v1.md").read_text()
    task_prompt = (PROMPTS / "fingerprinting_v1.md").read_text()
    records: list[dict[str, Any]] = []
    for index in range(1, episodes + 1):
        episode_id = f"e1-{index:03d}"
        episode_dir = out_dir / episode_id
        episode_dir.mkdir()
        started = time.monotonic()
        client = OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
        transport = AccessWorkerTransport(worker_url)
        record: dict[str, Any] = {"episode_id": episode_id, "started_utc": datetime.now(timezone.utc).isoformat(), "config": config.public_snapshot(), "system_prompt": system_prompt, "task_prompt": task_prompt, "tool_schemas": TOOL_SCHEMAS, "status": "infrastructure_failure"}
        try:
            transport.start_episode()
            outcome = run_agent(client, transport, allowed_cidr=config.allowed_cidr, permitted_ports=config.permitted_ports, max_steps=config.max_steps, system_prompt=system_prompt, task_prompt=task_prompt, episode_deadline_seconds=config.episode_deadline_seconds)
            record.update({"stop_reason": outcome.stop_reason, "messages": outcome.messages, "report": outcome.report})
            if outcome.stop_reason == "submitted_inventory":
                record["status"] = "completed"
                record["evaluation"] = evaluate_e1(outcome.report, truth)
            else:
                record["status"] = outcome.stop_reason
        except (OpenRouterError, WorkerTransportError, RuntimeError) as exc:
            record["error"] = str(exc)
        finally:
            try:
                transport.close_episode()
            except RuntimeError:
                pass
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        record["inference_requests"] = client.request_records
        usage = [request.get("usage") or {} for request in client.request_records]
        record["usage"] = {"prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage), "completion_tokens": sum(item.get("completion_tokens", 0) for item in usage), "total_tokens": sum(item.get("total_tokens", 0) for item in usage), "cost": sum(item.get("cost", 0) for item in usage)}
        _write_json(episode_dir / "episode.json", record)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--request-timeout-seconds", type=int)
    parser.add_argument("--episode-deadline-seconds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    config = ScenarioConfig.from_json(args.config)
    if config.harness_profile != "fingerprinting_basic":
        parser.error("the completed E1-A batch is reserved for the fingerprinting_basic profile")
    config = replace(config, model=args.model or config.model, max_steps=args.max_steps or config.max_steps, max_output_tokens=args.max_output_tokens or config.max_output_tokens, request_timeout_seconds=args.request_timeout_seconds or config.request_timeout_seconds, episode_deadline_seconds=args.episode_deadline_seconds or config.episode_deadline_seconds)
    if not config.live_openrouter or not config.live_worker:
        parser.error("batch requires an explicitly reviewed live configuration")
    if args.dry_run:
        validation = {"config": config.public_snapshot(), "episodes": args.episodes, "truth_path": str(TRUTH_PATH), "truth_present": TRUTH_PATH.is_file(), "prompt_files_present": all((PROMPTS / name).is_file() for name in ("system_v1.md", "fingerprinting_v1.md")), "tool_names": [tool["function"]["name"] for tool in TOOL_SCHEMAS], "worker_url_supplied": bool(args.worker_url)}
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if not args.worker_url:
        parser.error("--worker-url is required unless --dry-run is used")
    args.out_dir.mkdir(parents=True)
    records = run_batch(config, args.worker_url, args.out_dir, args.episodes)
    summary = summarize(records)
    _write_json(args.out_dir / "e1_aggregate.json", summary)
    (args.out_dir / "e1_five_run_report.md").write_text(_render_report(records, summary, config))


if __name__ == "__main__":
    main()
