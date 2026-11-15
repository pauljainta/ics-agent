"""Run a live OpenRouter experiment through a pre-established access-worker tunnel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.scadaver_sphere.agent import run_agent
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.schemas import tool_schemas
from experiments.scadaver_sphere.openrouter import OpenRouterClient, OpenRouterError
from experiments.scadaver_sphere.transport import AccessWorkerTransport, WorkerTransportError


PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
TASK_PROMPTS = {
    "fingerprinting": "fingerprinting_v1.md",
    "deception-blind": "deception_blind_v1.md",
    "deception-aware": "deception_aware_v1.md",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_PROMPTS), required=True)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker_url:
        parser.error("--worker-url or SCADAVER_WORKER_URL is required")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    config = ScenarioConfig.from_json(args.config)
    if not config.live_openrouter or not config.live_worker:
        parser.error("live runner requires a reviewed configuration with both live gates enabled")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = (PROMPTS / "system_v1.md").read_text()
    task_prompt = (PROMPTS / TASK_PROMPTS[args.task]).read_text()
    client = OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
    transport = AccessWorkerTransport(args.worker_url)
    try:
        transport.start_episode(config.harness_profile)
        episode = run_agent(
            client,
            transport,
            allowed_cidr=config.allowed_cidr,
            max_steps=config.max_steps,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            episode_deadline_seconds=config.episode_deadline_seconds,
            tool_schemas=tool_schemas(config.harness_profile),
            permitted_ports=config.permitted_ports,
        )
    except (OpenRouterError, WorkerTransportError, RuntimeError) as exc:
        result = {
            "config": config.public_snapshot(), "task": args.task,
            "status": "infrastructure_failure", "component": "external_openrouter_inference",
            "detail": str(exc), "scientific_result": False,
            "inference_requests": client.request_records,
        }
        (args.out_dir / "infrastructure_failure.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    finally:
        try:
            transport.close_episode()
        except RuntimeError:
            pass
    result = {
        "config": config.public_snapshot(),
        "task": args.task,
        "system_prompt": system_prompt,
        "task_prompt": task_prompt,
        "stop_reason": episode.stop_reason,
        "report": episode.report,
        "messages": episode.messages,
        "inference_requests": client.request_records,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
