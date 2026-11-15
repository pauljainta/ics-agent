"""Run one or more opt-in runtime/process fingerprinting episodes."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.scadaver_sphere.agent import run_agent
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.openrouter import OpenRouterClient, OpenRouterError
from experiments.scadaver_sphere.runtime_process_evaluation import evaluate_runtime_process
from experiments.scadaver_sphere.schemas import tool_schemas, validate_runtime_process_report
from experiments.scadaver_sphere.transport import AccessWorkerTransport, WorkerTransportError

ROOT = Path(__file__).resolve().parent
PROFILE = "fingerprinting_runtime_process"
TRUTH_PATH = ROOT / "openplc_v3" / "private" / "runtime_process_ground_truth.json"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _episode(config: ScenarioConfig, episode_id: str, worker_url: str, system: str, task: str) -> dict[str, Any]:
    client = OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
    transport = AccessWorkerTransport(worker_url)
    schema = tool_schemas(PROFILE)
    record: dict[str, Any] = {"episode_id": episode_id, "started_utc": datetime.now(timezone.utc).isoformat(), "config": config.public_snapshot(), "system_prompt": system, "task_prompt": task, "tool_schemas": schema, "status": "infrastructure_failure"}
    started = time.monotonic()
    try:
        transport.start_episode(PROFILE, required_tool_names={item["function"]["name"] for item in schema})
        outcome = run_agent(client, transport, allowed_cidr=config.allowed_cidr, permitted_ports=config.permitted_ports, max_steps=config.max_steps, system_prompt=system, task_prompt=task, episode_deadline_seconds=config.episode_deadline_seconds, tool_schemas=schema, report_validator=validate_runtime_process_report, submission_tool_name="submit_fingerprint_assessment")
        record.update({"stop_reason": outcome.stop_reason, "messages": outcome.messages, "report": outcome.report, "status": "completed" if outcome.stop_reason == "submitted_inventory" else outcome.stop_reason})
        if outcome.report is not None:
            record["evaluation"] = evaluate_runtime_process(outcome.report, json.loads(TRUTH_PATH.read_text()), outcome.messages)
    except (OpenRouterError, WorkerTransportError, RuntimeError, ValueError) as exc:
        record["error"] = str(exc)
    finally:
        try:
            transport.close_episode()
        except RuntimeError:
            pass
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    record["inference_requests"] = client.request_records
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--condition", choices=("network", "service"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = ScenarioConfig.from_json(args.config)
    if config.harness_profile != PROFILE:
        parser.error(f"config must set harness_profile={PROFILE}")
    prompts = ROOT / "prompts"
    system = (prompts / "runtime_process_system_v1.md").read_text()
    task = (prompts / f"runtime_process_{args.condition}_task_v1.md").read_text()
    plan = {"config": config.public_snapshot(), "condition": args.condition, "episodes": args.episodes, "tool_names": [item["function"]["name"] for item in tool_schemas(PROFILE)], "truth_is_evaluator_only": True}
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if not config.live_openrouter or not config.live_worker or not args.worker_url:
        parser.error("a reviewed live config and --worker-url are required")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    args.out_dir.mkdir(parents=True)
    _write(args.out_dir / "config_snapshot.json", plan)
    for index in range(1, args.episodes + 1):
        episode = _episode(config, f"runtime-process-{index:03d}", args.worker_url, system, task)
        directory = args.out_dir / episode["episode_id"]
        directory.mkdir(); _write(directory / "episode.json", episode)


if __name__ == "__main__":
    main()
