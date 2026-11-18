"""Run the isolated N3-S1 planner/selector scaffold variant."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .budgets import EpisodeBudget
from .config import ScenarioConfig
from .integrated_evaluation import evaluate_integrated, load_spec
from .integrated_runner import validate_report
from .n3 import (
    CONFIG as BASE_CONFIG,
    PROMPTS,
    RESULTS_ROOT,
    SEEDS,
    SKILL_INDEX,
    SPEC,
    EventJournal,
    N3Executor,
    N3ReliabilityOptions,
    _atomic_json,
    _episode_summary,
    _evaluation_messages,
    _limits,
    _load_local_key,
    _runtime_tool_manifest,
    prepare_root,
    tool_schemas,
)
from .n3_inference import N3OpenRouterClient
from .n3_scaffolded_agent import run_scaffolded_agent
from .n3_transport import N3Transport
from .n3_worker import N3_WORKER_TOOL_SCHEMAS


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/n3_qwen38_27b_scaffolded_v1.json"
SCAFFOLD_PROMPT = PROMPTS / "n3_scaffolded_addition.md"


@dataclass(frozen=True)
class ScaffoldOptions:
    planner_max_output_tokens: int
    selector_max_output_tokens: int

    @classmethod
    def from_json(cls, path: Path) -> "ScaffoldOptions":
        raw = json.loads(path.read_text(encoding="utf-8"))
        value = cls(int(raw.get("planner_max_output_tokens", 4096)),
                    int(raw.get("selector_max_output_tokens", 2048)))
        if value.planner_max_output_tokens <= 0 or value.selector_max_output_tokens <= 0:
            raise ValueError("scaffold output-token limits must be positive")
        return value


def run_seed(config: ScenarioConfig, reliability: N3ReliabilityOptions,
             scaffold: ScaffoldOptions, seed: int, out_dir: Path,
             worker_url: str, *, read_only: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=False)
    journal = EventJournal(out_dir)
    transport = N3Transport(worker_url, timeout=75)
    session = transport.start_episode(
        required_tool_names={item["function"]["name"] for item in N3_WORKER_TOOL_SCHEMAS},
        autonomous_actions=(reliability.automatic_session_action_authorization and not read_only),
    )
    action_enabled = session.get("action_broker_enabled") is True
    if reliability.automatic_session_action_authorization and not read_only and not action_enabled:
        transport.close_episode()
        raise RuntimeError("N3-S1 automatic action broker was not enabled")
    tools = tool_schemas(action_enabled, config.max_action_duration_seconds)
    foothold = transport.execute("get_foothold_context", {})
    journal.record("initial_foothold_observation", {"observation": foothold})
    executor = N3Executor(transport, journal, action_enabled)
    client = N3OpenRouterClient(
        config.model, config.api_key_env,
        request_timeout=config.request_timeout_seconds,
        max_tokens=config.max_output_tokens,
        retries=config.inference_retries,
        seed=seed,
        output_token_fallbacks=reliability.provider_output_token_fallbacks,
    )
    system_prompt = (PROMPTS / "n3_system.md").read_text(encoding="utf-8")
    mission_prompt = (PROMPTS / "n3_mission.md").read_text(encoding="utf-8")
    scaffold_prompt = SCAFFOLD_PROMPT.read_text(encoding="utf-8")
    initial_context = (
        SKILL_INDEX + "\n\nInitial read-only foothold observation:\n" +
        json.dumps(foothold, sort_keys=True) + "\n\n" + _runtime_tool_manifest(tools)
    )
    started = time.monotonic()
    budget = EpisodeBudget(_limits(config))
    try:
        outcome, progress = run_scaffolded_agent(
            client, executor, max_cycles=config.max_steps,
            system_prompt=system_prompt, mission_prompt=mission_prompt,
            scaffold_prompt=scaffold_prompt, initial_context=initial_context,
            tool_schemas=tools, report_validator=validate_report,
            planner_max_tokens=scaffold.planner_max_output_tokens,
            selector_max_tokens=scaffold.selector_max_output_tokens,
            final_submission_max_tokens=config.final_submission_max_output_tokens,
            episode_deadline_seconds=config.episode_deadline_seconds,
            budget=budget, max_context_chars=config.max_context_chars,
            context_tail_groups=config.context_tail_groups,
            event_sink=journal.record,
        )
    finally:
        transport.close_episode()
    evaluation = evaluate_integrated(
        _evaluation_messages(outcome.messages), outcome.report, load_spec(SPEC),
        action_required=action_enabled,
    )
    record = {
        "experiment": "N3-S1", "seed": seed, "status": outcome.stop_reason,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "messages": outcome.messages, "report": outcome.report,
        "progress_states": progress,
        "diagnostics": outcome.diagnostics, "statistics": outcome.statistics,
        "inference_requests": client.request_records, "evaluation": evaluation,
        "worker_session": {"profile": session.get("harness_profile"),
                           "tool_names": session.get("tool_names")},
        "action_enabled": action_enabled,
    }
    _atomic_json(out_dir / "episode.json", record)
    _atomic_json(out_dir / "evaluation.json", evaluation)
    _atomic_json(out_dir / "progress_states.json", progress)
    if outcome.report is not None:
        _atomic_json(out_dir / "final_report.json", outcome.report)
    (out_dir / "run_summary.md").write_text(
        _episode_summary(seed, outcome, evaluation, client.request_records, executor, started),
        encoding="utf-8",
    )
    return record


def finalize(out_dir: Path, episodes: dict[int, dict[str, Any]]) -> None:
    comparison = {
        str(seed): {
            "status": record["status"], "evaluation": record["evaluation"],
            "model_calls": len(record["inference_requests"]),
            "progress_updates": len(record["progress_states"]),
            "duration_seconds": record["duration_seconds"],
        } for seed, record in episodes.items()
    }
    _atomic_json(out_dir / "comparison.json", comparison)
    lines = ["# N3-S1 Scaffolded Qwen3.8-27B Results", "",
             "This is a separate planner/selector scaffold condition; the N3 baseline is unchanged.", "",
             "| Seed | E1 | E2 | E3 | E4 | E5 | Overall |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        evaluation = episodes[seed]["evaluation"]
        values = [evaluation[key]["passed"] for key in (
            "e1_discovery", "e2_service_protocol_identification", "e3_understanding",
            "e4_planning", "e5_execution")]
        lines.append(f"| {seed} | " + " | ".join(str(value) for value in values) +
                     f" | {evaluation['overall_success']} |")
    (out_dir / "final_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", default=os.environ.get(
        "SCADAVER_N3_WORKER_URL", "http://127.0.0.1:18778"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)
    config = ScenarioConfig.from_json(args.config)
    reliability = N3ReliabilityOptions.from_json(args.config)
    scaffold = ScaffoldOptions.from_json(args.config)
    if config.model != "qwen/qwen3.8-27b":
        parser.error("N3-S1 requires qwen/qwen3.8-27b")
    _load_local_key(config.api_key_env)
    if not os.environ.get(config.api_key_env):
        parser.error(f"{config.api_key_env} is not set")
    out_dir = args.out_dir or RESULTS_ROOT / datetime.now(timezone.utc).strftime(
        "N3s1_qwen38_27b_%Y%m%dT%H%M%SZ")
    manifest = prepare_root(out_dir, config, reliability, config_path=args.config)
    shutil.copy2(SCAFFOLD_PROMPT, out_dir / "scaffold_prompt.md")
    manifest.update({
        "experiment": "N3-S1", "baseline_unchanged": str(BASE_CONFIG),
        "planner_max_output_tokens": scaffold.planner_max_output_tokens,
        "selector_max_output_tokens": scaffold.selector_max_output_tokens,
        "decision_architecture": "model-authored progress planner then one-tool selector",
    })
    _atomic_json(out_dir / "experiment_manifest.json", manifest)
    transport = N3Transport(args.worker_url, timeout=75)
    startup = transport.operator_check("status")
    readiness = transport.operator_check("baseline")
    _atomic_json(out_dir / "testbed_startup.json", startup)
    _atomic_json(out_dir / "readiness_checks.json", readiness)
    if startup.get("TESTBED_READY") is not True or readiness.get("TESTBED_READY") is not True:
        _atomic_json(out_dir / "comparison.json", {"status": "preflight_failed"})
        return 2
    episodes: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        episodes[seed] = run_seed(config, reliability, scaffold, seed,
                                  out_dir / f"seed_{seed}", args.worker_url,
                                  read_only=args.read_only)
        if seed == 101:
            between = transport.operator_check("verify")
            _atomic_json(out_dir / "between_seed_readiness.json", between)
            e5 = episodes[seed]["evaluation"]["e5_execution"]
            if (between.get("TESTBED_READY") is not True
                    or episodes[seed]["status"] in {"tool_transport_error", "inference_error"}
                    or (e5.get("action_applied") and not e5.get("restoration_verified"))):
                break
    if set(episodes) != set(SEEDS):
        _atomic_json(out_dir / "comparison.json", {
            "status": "campaign_stopped_before_seed_102", "completed_seeds": sorted(episodes)})
        return 2
    post = transport.operator_check("verify")
    _atomic_json(out_dir / "post_run_readiness.json", post)
    finalize(out_dir, episodes)
    return 0 if post.get("TESTBED_READY") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
