from __future__ import annotations

import json
from pathlib import Path

from ics_agent.experiments import model_variant_runner as runner


def _write_summary(path: Path, *, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status}) + "\n")


def test_model_variant_resume_cleanup_removes_failed_gray_box_trial_only(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "WORKSPACE_ROOT", tmp_path)
    model = runner.ModelVariant(
        key="phi4_mini_instruct",
        label="Phi-4-mini-instruct",
        model_id="microsoft/Phi-4-mini-instruct",
        local_dir=Path("models/local_models/phi-4-mini-instruct"),
        quantization="bf16",
    )
    testbed = runner.TestbedVariant(
        key="water_distribution",
        testbed_id="water_distribution",
        result_experiment_name="water_distribution_openplc",
        expected_scenarios=11,
        family_configs=(),
    )
    root = (
        tmp_path
        / "Artifact"
        / "Results"
        / model.label
        / testbed.result_experiment_name
    )
    completed_trial = (
        root
        / "gray_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-completed"
        / "trial-001"
    )
    failed_trial = (
        root
        / "gray_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-grid-low-spoof"
        / "trial-001"
    )
    other_family_trial = (
        root
        / "black_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-completed"
        / "trial-001"
    )

    _write_summary(completed_trial / "summary.json", status="completed")
    _write_summary(failed_trial / "summary.json", status="failed")
    _write_summary(other_family_trial / "summary.json", status="completed")

    removed = runner._purge_failed_or_incomplete_runs(
        model=model,
        testbed=testbed,
        family_names={"gray_box_attacker__agentic_defender"},
    )

    assert completed_trial.exists()
    assert other_family_trial.exists()
    assert not failed_trial.exists()
    assert [item["family"] for item in removed] == [
        "gray_box_attacker__agentic_defender"
    ]
