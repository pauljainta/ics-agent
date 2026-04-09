#!/usr/bin/env python3
"""
Scripted nominal/attack agents for the water-treatment testbed.

These agents are intentionally simple: they drive curated scenario suites
through the host-side testbed controller and collect reproducible artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OPENPLC_DIR = (
    WORKSPACE_ROOT
    / "sphere-usecases"
    / "water-treatment"
    / "implementations"
    / "openplc"
)
OPENPLC_SCRIPTS = OPENPLC_DIR / "scripts"
sys.path = [entry for entry in sys.path if entry not in ("", str(OPENPLC_SCRIPTS))]
sys.path.insert(0, str(OPENPLC_SCRIPTS))

from openplc_testbed import DEFAULT_READINESS_SCENARIOS, OpenPLCWaterTreatmentTestbed, TestbedError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    candidates: list[str] = []
    for anchor in (WORKSPACE_ROOT, OPENPLC_DIR):
        try:
            candidate = resolved.relative_to(anchor.resolve()).as_posix()
        except ValueError:
            continue
        if candidate:
            candidates.append(candidate)
    if candidates:
        return min(candidates, key=len)
    return str(path)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def readiness_paths() -> list[Path]:
    return unique_paths([(OPENPLC_DIR / path).resolve() for path in DEFAULT_READINESS_SCENARIOS])


def agent_smoke_paths(group: str) -> list[Path]:
    return unique_paths(sorted((WORKSPACE_ROOT / "artifacts" / "agent_smoke" / group).glob("*/scenario.yaml")))


def recheck_paths() -> list[Path]:
    return unique_paths(sorted((WORKSPACE_ROOT / "artifacts" / "agent_smoke" / "recheck").glob("*/scenario.yaml")))


@dataclass
class ScriptedAgent:
    name: str
    scenario_paths: list[Path]

    def run(
        self,
        testbed: OpenPLCWaterTreatmentTestbed,
        output_root: Path,
    ) -> dict[str, Any]:
        agent_root = output_root / self.name
        agent_root.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []

        for scenario_path in self.scenario_paths:
            scenario = testbed.load_scenario(scenario_path)
            scenario_id = scenario.get("scenario_id", scenario_path.stem)
            scenario_output = agent_root / scenario_id
            try:
                result = testbed.run_scenario(
                    scenario_path,
                    output_dir=scenario_output,
                    ensure_stack=False,
                )
            except Exception as exc:  # pragma: no cover - exercised in live runs
                result = {
                    "scenario_id": scenario_id,
                    "scenario_path": portable_path(Path(scenario_path)),
                    "passed": False,
                    "error": str(exc),
                    "output_dir": portable_path(scenario_output),
                }
                scenario_output.mkdir(parents=True, exist_ok=True)
                (scenario_output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)

        summary = {
            "timestamp_utc": utc_now(),
            "agent": self.name,
            "passed": all(result.get("passed", False) for result in results),
            "scenario_count": len(results),
            "results": results,
        }
        (agent_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


class NominalOperationsAgent(ScriptedAgent):
    pass


class AttackAgent(ScriptedAgent):
    pass


class WargameOrchestrator:
    def __init__(self) -> None:
        self.testbed = OpenPLCWaterTreatmentTestbed(base_dir=OPENPLC_DIR)
        self.agents = {
            "readiness": NominalOperationsAgent("readiness", readiness_paths()),
            "normal": NominalOperationsAgent("normal", agent_smoke_paths("normal")),
            "attack": AttackAgent("attack", agent_smoke_paths("attack")),
            "recheck": NominalOperationsAgent("recheck", recheck_paths()),
        }

    def run_suite(self, suite_name: str, output_root: Path) -> dict[str, Any]:
        if suite_name == "all":
            selected = [self.agents[name] for name in ("readiness", "normal", "attack", "recheck")]
        else:
            if suite_name not in self.agents:
                raise TestbedError(f"unknown suite: {suite_name}")
            selected = [self.agents[suite_name]]

        output_root.mkdir(parents=True, exist_ok=True)
        self.testbed.ensure_stack()
        summaries = [agent.run(self.testbed, output_root) for agent in selected]

        summary = {
            "timestamp_utc": utc_now(),
            "suite": suite_name,
            "passed": all(item["passed"] for item in summaries),
            "agents": summaries,
        }
        (output_root / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scripted water-treatment wargame agents")
    parser.add_argument(
        "--suite",
        choices=("readiness", "normal", "attack", "recheck", "all"),
        default="all",
        help="Scenario suite to execute",
    )
    parser.add_argument(
        "--output-root",
        default=str(WORKSPACE_ROOT / "artifacts" / "scripted_agents"),
        help="Directory for suite artifacts",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    orchestrator = WargameOrchestrator()
    summary = orchestrator.run_suite(args.suite, Path(args.output_root))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
