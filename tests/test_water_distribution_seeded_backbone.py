from __future__ import annotations

import subprocess

from ics_agent.scenarios import ScenarioCatalog
from ics_agent.water_distribution_seeded_backbone import (
    DEFAULT_TMUX_SESSION_NAME,
    SeededBackboneRunner,
    TESTBED_ID,
    _format_mean_std,
    _normalize_csv_rows,
    _parse_args,
    build_manifest_rows,
    build_split_rows,
)


def test_water_distribution_seeded_manifest_and_splits_are_fixed() -> None:
    catalog = ScenarioCatalog(testbed_id=TESTBED_ID)
    scenarios = catalog.resolve_selectors(["attack/*"])

    manifest_rows = build_manifest_rows(scenarios)
    split_rows = build_split_rows(manifest_rows)

    assert len(manifest_rows) == 11
    assert all(row["testbed"] == TESTBED_ID for row in manifest_rows)
    assert all(row["enabled_for_final"] == "true" for row in manifest_rows)
    assert manifest_rows[0]["scenario_id"] == "wd-attack-coordinated-grid-rws-delay"
    assert manifest_rows[-1]["scenario_id"] == "wd-attack-supply-low-spoof"

    split_counts = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    assert split_counts == {
        "train_dev": 7,
        "validation": 2,
        "heldout_test": 2,
    }


def test_distribution_format_mean_std_handles_missing_values() -> None:
    assert _format_mean_std(None, None) == "n/a"
    assert _format_mean_std(8.3897, 0.4123) == "8.39 ± 0.41"


def test_distribution_normalize_csv_rows_stabilizes_string_comparison() -> None:
    assert _normalize_csv_rows(
        [{"manifest_order": 1, "split": "train_dev", "enabled_for_final": True}]
    ) == [
        {
            "manifest_order": "1",
            "split": "train_dev",
            "enabled_for_final": "True",
        }
    ]


def test_distribution_parse_args_launch_defaults() -> None:
    args = _parse_args(["launch"])
    assert args.mode == "launch"
    assert args.session_name == DEFAULT_TMUX_SESSION_NAME
    assert args.force_restart is False


def test_distribution_launch_controller_uses_detached_tmux(
    monkeypatch, tmp_path
) -> None:
    runner = SeededBackboneRunner(tmp_path / "seeded_backbone_qwen35_9b")
    calls: list[list[str]] = []

    monkeypatch.setattr("ics_agent.water_distribution_seeded_backbone.shutil.which", lambda name: "/usr/bin/tmux" if name == "tmux" else None)
    monkeypatch.setattr(runner, "_tmux_session_exists", lambda session_name: False)

    def fake_run(command, text=True, capture_output=True):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("ics_agent.water_distribution_seeded_backbone.subprocess.run", fake_run)

    result = runner.launch_controller("wd_seeded_backbone_test")

    assert result == 0
    assert calls
    assert calls[0][:4] == ["/usr/bin/tmux", "new-session", "-d", "-s"]
    assert calls[0][4] == "wd_seeded_backbone_test"
    launch_command = calls[0][5]
    assert "ics_agent.water_distribution_seeded_backbone controller" in launch_command
    assert str(runner.result_root) in launch_command
    assert str(runner.controller_stdout_path) in launch_command
    assert runner.launcher_state_path.exists()
