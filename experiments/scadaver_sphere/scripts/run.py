"""Run F0 locally or an agent loop only after explicitly enabling both live gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.scadaver_sphere.agent import run_agent
from experiments.scadaver_sphere.config import ScenarioConfig
from experiments.scadaver_sphere.f0 import run_f0
from experiments.scadaver_sphere.openrouter import OpenRouterClient
from experiments.scadaver_sphere.policy import ReadOnlyPolicy
from experiments.scadaver_sphere.worker import ReadOnlyWorker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("f0", "agent"), required=True)
    parser.add_argument("--allow-live-worker", action="store_true")
    parser.add_argument("--allow-live-openrouter", action="store_true")
    args = parser.parse_args()
    config = ScenarioConfig.from_json(args.config)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    if args.mode == "agent" and not args.allow_live_openrouter:
        parser.error("agent mode requires --allow-live-openrouter")
    if args.allow_live_worker != config.live_worker or args.allow_live_openrouter != config.live_openrouter:
        parser.error("live flags must match an explicitly reviewed live configuration")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = ReadOnlyPolicy.build(config.allowed_cidr, config.permitted_ports, max_hosts=config.max_hosts)
    worker = ReadOnlyWorker(policy, live=args.allow_live_worker)
    worker.max_observations = config.max_observations
    metadata = {
        "scenario": config.public_snapshot(),
        "mode": args.mode,
        "worker_live": worker.live,
        "tool_policy": {
            "allowed_cidr": str(policy.allowed_cidr),
            "permitted_ports": list(policy.permitted_ports),
            "max_hosts": policy.max_hosts,
            "max_register_read": 10,
            "max_observations": worker.max_observations,
        },
    }
    if args.mode == "f0":
        result = {
            "metadata": metadata,
            "observations": run_f0(worker, cidr=config.allowed_cidr, ports=list(config.permitted_ports)),
        }
    else:
        client = OpenRouterClient(config.model, config.api_key_env)
        episode = run_agent(client, worker, allowed_cidr=config.allowed_cidr, max_steps=config.max_steps)
        result = {
            "metadata": metadata,
            "stop_reason": episode.stop_reason,
            "report": episode.report,
            "messages": episode.messages,
            "observations": worker.observations.values,
        }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
