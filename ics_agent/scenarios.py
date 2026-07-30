from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .testbeds import get_testbed_spec, normalize_testbed_id
from .types import ScenarioDefinition
from .utils import load_yaml


REQUIRED_METADATA_FIELDS = [
    "name",
    "intent",
    "prerequisites",
    "start_state",
    "attacker_actions",
    "defender_responses",
    "expected_outcomes",
    "success_conditions",
    "failure_conditions",
    "required_metrics",
]


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_catalog_config(testbed_id: str) -> dict[str, Any]:
    config = load_config("scenario_catalog.yaml")
    overlay_path = (
        get_testbed_spec(testbed_id).scenario_root.parent / "scenario_catalog.yaml"
    )
    if overlay_path.exists():
        overlay = load_yaml(overlay_path)
        if not isinstance(overlay, dict):
            raise ConfigError(f"expected mapping at {overlay_path}")
        config = _merge_dict(config, overlay)
    return config


def _default_role_bindings(
    suite: str, execution: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, list[int]]:
    timeline = execution.get("timeline", [])
    explicit = metadata.get("role_bindings")
    if isinstance(explicit, dict):
        return {
            role: [int(item) for item in values] for role, values in explicit.items()
        }

    bindings: dict[str, list[int]] = {"operator": [], "attacker": [], "system": []}
    incident_type = metadata.get("ground_truth", {}).get("incident_type", "nominal")

    for index, item in enumerate(timeline):
        action = item.get("action")
        action_type = action if isinstance(action, str) else action.get("type")
        if suite == "attack" or incident_type == "attack":
            if index == 0 and action_type == "hmi_start":
                bindings["operator"].append(index)
            else:
                bindings["attacker"].append(index)
            continue
        if suite in {"normal", "readiness", "recheck", "smoke"} or incident_type in {
            "nominal",
            "fault",
            "regression",
            "smoke",
        }:
            bindings["operator"].append(index)
            continue
        bindings["system"].append(index)

    return {role: values for role, values in bindings.items() if values}


def _validate_execution(path: Path, payload: dict[str, Any]) -> None:
    required = [
        "scenario_id",
        "description",
        "duration_sec",
        "poll_interval_ms",
        "timeline",
        "expectations",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ConfigError(
            f"scenario execution {path} missing fields: {', '.join(missing)}"
        )


class ScenarioCatalog:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        testbed_id: str | None = None,
    ) -> None:
        self.testbed_id = normalize_testbed_id(testbed_id)
        self.testbed = get_testbed_spec(self.testbed_id)
        if config_path is not None:
            self.config = load_config(config_path)
            overlay_path = (
                self.testbed.scenario_root.parent / "scenario_catalog.yaml"
            )
            if overlay_path.exists():
                overlay = load_yaml(overlay_path)
                if not isinstance(overlay, dict):
                    raise ConfigError(f"expected mapping at {overlay_path}")
                self.config = _merge_dict(self.config, overlay)
        else:
            self.config = _load_catalog_config(self.testbed_id)
        self.scenarios = self._load_definitions()

    def _load_definitions(self) -> dict[str, ScenarioDefinition]:
        suite_defaults = self.config.get("suite_defaults", {})
        scenario_overrides = self.config.get("scenario_overrides", {})
        definitions: dict[str, ScenarioDefinition] = {}

        scenario_root = self.testbed.scenario_root
        for path in sorted(scenario_root.rglob("*.yaml")):
            if path.name.startswith("."):
                continue
            execution = load_yaml(path)
            if not isinstance(execution, dict):
                raise ConfigError(f"scenario file {path} must contain a mapping")
            _validate_execution(path, execution)
            suite = path.parent.name
            catalog_id = f"{suite}/{execution['scenario_id']}"

            base_metadata = deepcopy(suite_defaults.get(suite, {}))
            override = deepcopy(scenario_overrides.get(catalog_id, {}))
            metadata = _merge_dict(base_metadata, override)

            metadata.setdefault("name", execution["scenario_id"])
            metadata.setdefault("intent", execution.get("description", ""))
            metadata.setdefault(
                "prerequisites",
                ["OpenPLC stack running or launchable via configured backend"],
            )
            metadata.setdefault(
                "start_state",
                {"initial_conditions": execution.get("initial_conditions", {})},
            )
            metadata.setdefault("attacker_actions", [])
            metadata.setdefault("defender_responses", [])
            metadata.setdefault("expected_outcomes", [execution.get("description", "")])
            metadata.setdefault(
                "success_conditions",
                [check for check in execution.get("expectations", {}).keys()],
            )
            metadata.setdefault("failure_conditions", ["scenario expectations fail"])
            metadata.setdefault(
                "required_metrics", self.config.get("global_required_metrics", [])
            )
            metadata.setdefault("ground_truth", {})
            metadata.setdefault("safety_bounds", {})
            metadata.setdefault("tags", [])
            metadata["role_bindings"] = _default_role_bindings(
                suite, execution, metadata
            )

            missing = [
                field for field in REQUIRED_METADATA_FIELDS if field not in metadata
            ]
            if missing:
                raise ConfigError(
                    f"scenario metadata for {catalog_id} missing fields: {', '.join(missing)}"
                )

            metadata["testbed_id"] = self.testbed_id
            definitions[catalog_id] = ScenarioDefinition(
                testbed_id=self.testbed_id,
                catalog_id=catalog_id,
                suite=suite,
                scenario_id=execution["scenario_id"],
                execution_path=path.resolve(),
                execution=execution,
                metadata=metadata,
            )

        return definitions

    def list(self) -> list[ScenarioDefinition]:
        return [self.scenarios[key] for key in sorted(self.scenarios)]

    def get(self, catalog_id: str) -> ScenarioDefinition:
        try:
            return self.scenarios[catalog_id]
        except KeyError as exc:
            raise ConfigError(f"unknown scenario: {catalog_id}") from exc

    def resolve_selectors(self, selectors: list[str]) -> list[ScenarioDefinition]:
        if not selectors:
            return self.list()

        selected: list[ScenarioDefinition] = []
        seen: set[str] = set()
        for selector in selectors:
            selector = selector.strip()
            if selector == "*":
                matches = self.list()
            elif selector.endswith("/*"):
                prefix = selector[:-1]
                matches = [
                    scenario
                    for scenario in self.list()
                    if scenario.catalog_id.startswith(prefix)
                ]
            else:
                matches = [self.get(selector)]
            for match in matches:
                if match.catalog_id in seen:
                    continue
                seen.add(match.catalog_id)
                selected.append(match)
        return selected
