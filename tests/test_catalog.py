from __future__ import annotations

from pathlib import Path

from ics_agent.config import load_config
from ics_agent.paths import RESULTS_ROOT, WORKSPACE_ROOT
from ics_agent.scenarios import ScenarioCatalog


def test_catalog_loads_repo_local_scenarios() -> None:
    catalog = ScenarioCatalog()
    scenario_ids = {scenario.catalog_id for scenario in catalog.list()}
    assert "smoke/wt-smoke-p1-short-nominal" in scenario_ids
    assert "smoke/wt-smoke-rw-refill-spoof" in scenario_ids
    assert "attack/wt-attack-sensor-rw-refill-spoof" in scenario_ids
    assert len(scenario_ids) >= 20


def test_catalog_applies_metadata_schema_defaults() -> None:
    catalog = ScenarioCatalog()
    scenario = catalog.get("smoke/wt-smoke-rw-refill-spoof")
    metadata = scenario.metadata
    for key in (
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
    ):
        assert key in metadata
    assert metadata["ground_truth"]["incident_type"] == "attack"


def test_runtime_config_loader_resolves_experiment_aliases() -> None:
    payload = load_config("smoke_core_matrix.yaml")
    assert payload["experiment_name"] == "smoke_core_matrix"
    assert Path(payload["_path"]).parent.name == "Experiments"


def test_smoke_nominal_bindings_assign_operator_and_artifact_results_root() -> None:
    catalog = ScenarioCatalog()
    scenario = catalog.get("smoke/wt-smoke-p1-short-nominal")
    assert scenario.metadata["role_bindings"]["operator"] == [0, 1]
    assert RESULTS_ROOT == WORKSPACE_ROOT / "Artifact" / "Results"
    assert RESULTS_ROOT.name == "Results"
