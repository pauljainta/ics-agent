from __future__ import annotations

from ics_agent.scenarios import ScenarioCatalog
from ics_agent.water_treatment_seeded_backbone import (
    TESTBED_ID,
    _format_mean_std,
    _normalize_csv_rows,
    build_manifest_rows,
    build_split_rows,
)


def test_water_treatment_seeded_manifest_and_splits_are_fixed() -> None:
    catalog = ScenarioCatalog(testbed_id=TESTBED_ID)
    scenarios = catalog.resolve_selectors(["attack/*"])

    manifest_rows = build_manifest_rows(scenarios)
    split_rows = build_split_rows(manifest_rows)

    assert len(manifest_rows) == 18
    assert all(row["testbed"] == TESTBED_ID for row in manifest_rows)
    assert all(row["enabled_for_final"] == "true" for row in manifest_rows)
    assert manifest_rows[0]["scenario_id"] == "wt-attack-hmi-chatter"
    assert manifest_rows[-1]["scenario_id"] == "wt-attack-sensor-uf-high-disrupt"

    split_counts = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    assert split_counts == {
        "train_dev": 12,
        "validation": 3,
        "heldout_test": 3,
    }


def test_format_mean_std_handles_missing_values() -> None:
    assert _format_mean_std(None, None) == "n/a"
    assert _format_mean_std(13.857, 0.104) == "13.86 ± 0.10"


def test_normalize_csv_rows_stabilizes_string_comparison() -> None:
    assert _normalize_csv_rows(
        [{"manifest_order": 1, "split": "train_dev", "enabled_for_final": True}]
    ) == [
        {
            "manifest_order": "1",
            "split": "train_dev",
            "enabled_for_final": "True",
        }
    ]
