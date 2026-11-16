"""Fail-closed production runtime parity validation for integrated experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "config/production_runtime_manifest_v1.json"


class DeploymentParityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_runtime_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    file_results: dict[str, Any] = {}
    for relative, expected in manifest["files"].items():
        target = ROOT / relative
        actual = _sha256(target) if target.is_file() else None
        passed = actual == expected
        file_results[relative] = {"expected": expected, "actual": actual, "passed": passed}
        if not passed:
            failures.append(f"file_hash:{relative}")

    import_results: dict[str, Any] = {}
    for module_name, expected_relative in manifest["imports"].items():
        module = importlib.import_module(module_name)
        actual_path = Path(module.__file__).resolve()
        expected_path = (ROOT / expected_relative).resolve()
        passed = actual_path == expected_path
        import_results[module_name] = {
            "actual": str(actual_path), "expected": str(expected_path), "passed": passed,
        }
        if not passed:
            failures.append(f"import_path:{module_name}")

    dependency_results = {}
    for package, expected in manifest["dependencies"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        passed = actual == expected
        dependency_results[package] = {"expected": expected, "actual": actual, "passed": passed}
        if not passed:
            failures.append(f"dependency:{package}")

    from .budgets import EpisodeBudget, EpisodeLimits
    from .config import ScenarioConfig
    from .heterogeneous_sitl_v1.action_broker import ActionBroker
    from .integrated_evaluation import load_spec
    from .openrouter import OpenRouterClient
    from .reasoning_ablation import ReasoningProfile, reasoning_tool_schema
    from .schemas import E4_TOOL_SCHEMAS
    from .transport import AccessWorkerTransport

    required_interfaces = {
        "EpisodeBudget.remaining": callable(getattr(EpisodeBudget, "remaining", None)),
        "EpisodeBudget.ingest_request_records": callable(getattr(EpisodeBudget, "ingest_request_records", None)),
        "OpenRouterClient.chat": callable(getattr(OpenRouterClient, "chat", None)),
        "OpenRouterClient.mark_termination": callable(getattr(OpenRouterClient, "mark_termination", None)),
        "ActionBroker.submit": callable(getattr(ActionBroker, "submit", None)),
        "AccessWorkerTransport.execute": callable(getattr(AccessWorkerTransport, "execute", None)),
    }
    failures.extend(f"interface:{name}" for name, passed in required_interfaces.items() if not passed)

    scenario = ScenarioConfig.from_json(ROOT / manifest["scientific_configuration"]["scenario_config"])
    expected_science = manifest["scientific_configuration"]["expected"]
    actual_science = {key: scenario.public_snapshot()[key] for key in expected_science}
    science_passed = actual_science == expected_science
    if not science_passed:
        failures.append("scientific_configuration")

    profile = ReasoningProfile.from_json(ROOT / manifest["scientific_configuration"]["reasoning_profile"])
    b0_passed = profile.condition == "B0" and not profile.enabled and reasoning_tool_schema(profile) is None
    if not b0_passed:
        failures.append("b0_profile")

    evaluator = load_spec(ROOT / manifest["scientific_configuration"]["evaluation_spec"])
    evaluator_passed = evaluator.get("evaluator_version") == manifest["scientific_configuration"]["evaluator_version"]
    if not evaluator_passed:
        failures.append("evaluator_version")

    tool_names = [item["function"]["name"] for item in E4_TOOL_SCHEMAS]
    tools_passed = tool_names == manifest["tool_names"]
    if not tools_passed:
        failures.append("tool_catalog")

    limits = EpisodeLimits(
        scenario.max_model_calls, scenario.max_tool_calls, scenario.max_input_tokens,
        scenario.max_total_output_tokens, scenario.max_network_observation_seconds,
        scenario.max_time_series_samples, scenario.max_api_cost_usd,
        scenario.reserved_completion_steps,
    )
    budget = EpisodeBudget(limits)
    budget_passed = budget.remaining()["model_calls"] == scenario.max_model_calls
    if not budget_passed:
        failures.append("budget_initialization")

    python_expected = manifest["python"]
    python_actual = {"major": sys.version_info.major, "minor": sys.version_info.minor}
    python_passed = python_actual == python_expected
    if not python_passed:
        failures.append("python_version")

    result = {
        "manifest_version": manifest["manifest_version"],
        "passed": not failures,
        "failures": failures,
        "manifest_sha256": _sha256(path),
        "python": {"expected": python_expected, "actual": python_actual, "passed": python_passed},
        "dependencies": dependency_results,
        "files": file_results,
        "imports": import_results,
        "interfaces": required_interfaces,
        "scientific_configuration": {"expected": expected_science, "actual": actual_science,
                                      "passed": science_passed},
        "b0_profile_passed": b0_passed,
        "evaluator_passed": evaluator_passed,
        "tool_catalog_passed": tools_passed,
        "budget_initialization_passed": budget_passed,
    }
    if failures:
        raise DeploymentParityError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        result = verify_runtime_manifest(args.manifest)
    except DeploymentParityError as exc:
        print(json.dumps(json.loads(str(exc)), indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
