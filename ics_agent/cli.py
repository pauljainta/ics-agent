from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import ExperimentOrchestrator
from .scenarios import ScenarioCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ICS water-treatment experiment runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser(
        "validate", help="Validate scenario catalog and agent profiles"
    )
    validate_parser.add_argument("--catalog", default="scenario_catalog.yaml")

    list_parser = sub.add_parser("list-scenarios", help="List known scenarios")
    list_parser.add_argument("--catalog", default="scenario_catalog.yaml")

    run_parser = sub.add_parser("run", help="Run an experiment config")
    run_parser.add_argument("config", help="Experiment YAML relative to configs/")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        orchestrator = ExperimentOrchestrator()
        runtime = orchestrator.validate_runtime()
        catalog = ScenarioCatalog(args.catalog)
        payload = {
            "validated": True,
            "scenario_count": len(catalog.list()),
            "profile_count": runtime["profile_count"],
            "agentic_profile_count": runtime["agentic_profile_count"],
            "knowledge_file_count": runtime["knowledge_file_count"],
            "scenarios": [scenario.catalog_id for scenario in catalog.list()],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "list-scenarios":
        catalog = ScenarioCatalog(args.catalog)
        payload = [
            {
                "catalog_id": scenario.catalog_id,
                "name": scenario.metadata["name"],
                "intent": scenario.metadata["intent"],
            }
            for scenario in catalog.list()
        ]
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run":
        orchestrator = ExperimentOrchestrator()
        summary = orchestrator.run(Path(args.config))
        print(json.dumps(summary, indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
