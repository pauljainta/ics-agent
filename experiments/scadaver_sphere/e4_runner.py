"""Read-only E4 objective-pathway runner; live model execution remains opt-in."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import run_agent
from .config import ScenarioConfig
from .e23_runner import _write, preflight
from .e4_evaluation import evaluate_e4, load_spec
from .openrouter import OpenRouterClient
from .schemas import E4_TOOL_SCHEMAS, validate_objective_pathway_report
from .transport import AccessWorkerTransport

ROOT = Path(__file__).resolve().parent


class E4MockClient:
    request_records: list[dict[str, Any]] = []
    def __init__(self, objective: str) -> None: self.objective, self.calls, self.request_records = objective, 0, []
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1: return {"tool_calls": [{"id": "discover", "function": {"name": "discover_hosts", "arguments": '{"cidr":"198.18.4.0/24"}'}}]}
        report = {"run_id": "mock-e4", "objective": self.objective, "relevant_systems": [], "access_assessment": [{"interface": "read-only worker", "permitted": True, "reason": "configured tool surface", "supporting_observation_ids": ["obs-00001"]}], "pathway_hypotheses": [], "overall_assessment": "No endpoint claim without evidence."}
        return {"tool_calls": [{"id": "submit", "function": {"name": "submit_objective_pathway", "arguments": json.dumps({"report": report})}}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--objective", required=True)
    parser.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL")); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mock", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--model")
    parser.add_argument("--evaluation-spec", type=Path, default=ROOT / "private/e4_simulated_process_objective.json")
    args = parser.parse_args(); config = replace(ScenarioConfig.from_json(args.config), model=args.model or ScenarioConfig.from_json(args.config).model)
    if config.harness_profile != "heterogeneous_e4": parser.error("configuration must use harness_profile=heterogeneous_e4")
    if args.dry_run:
        print(json.dumps({"profile": "heterogeneous_e4", "objective": args.objective, "tools": [x["function"]["name"] for x in E4_TOOL_SCHEMAS], "openrouter_called": False}, indent=2)); return
    if not args.worker_url: parser.error("--worker-url or SCADAVER_WORKER_URL is required")
    if args.out_dir.exists() and any(args.out_dir.iterdir()): parser.error("--out-dir must be new or empty")
    if not args.mock and (not config.live_openrouter or not config.live_worker): parser.error("live execution requires reviewed live configuration")
    args.out_dir.mkdir(parents=True); _write(args.out_dir / "preflight.json", preflight(args.worker_url, "heterogeneous_e4"))
    system = (ROOT / "prompts/system_v1.md").read_text(); task = (ROOT / "prompts/e4_objective_pathway_v1.md").read_text() + "\nObjective: " + args.objective
    client = E4MockClient(args.objective) if args.mock else OpenRouterClient(config.model, config.api_key_env, request_timeout=config.request_timeout_seconds, max_tokens=config.max_output_tokens, retries=config.inference_retries)
    transport = AccessWorkerTransport(args.worker_url); record: dict[str, Any] = {"objective": args.objective, "config": config.public_snapshot(), "mock": args.mock}
    try:
        transport.start_episode("heterogeneous_e4", required_tool_names={x["function"]["name"] for x in E4_TOOL_SCHEMAS})
        outcome = run_agent(client, transport, allowed_cidr=config.allowed_cidr, permitted_ports=config.permitted_ports, max_steps=config.max_steps, system_prompt=system, task_prompt=task, tool_schemas=E4_TOOL_SCHEMAS, report_validator=validate_objective_pathway_report, submission_tool_name="submit_objective_pathway", final_submission_max_tokens=config.final_submission_max_output_tokens)
        record.update({"status": outcome.stop_reason, "report": outcome.report, "messages": outcome.messages, "diagnostics": outcome.diagnostics, "evaluation": evaluate_e4(outcome.report, outcome.messages, load_spec(args.evaluation_spec))})
    finally:
        try: transport.close_episode()
        except RuntimeError: pass
    record["inference_requests"] = client.request_records; _write(args.out_dir / "episode.json", record); print(json.dumps(record["evaluation"], indent=2))


if __name__ == "__main__": main()
