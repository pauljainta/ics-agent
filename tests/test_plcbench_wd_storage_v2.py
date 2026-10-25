"""Admission guards for the experimental WD storage realization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ics_agent.plcbench_matched_run import wd_command_menu
from ics_agent.plcbench_wd_storage_v2 import reset_wd_storage_v2
from testbeds.water_distribution.openplc.scripts.modbus_bridge import build_blocks

ROOT = Path(__file__).resolve().parents[1]
MAP_YAML = ROOT / "testbeds/water_distribution/openplc/configs/storage_v2/openplc_map.yaml"
MAP_JSON = MAP_YAML.with_suffix(".json")


def test_generated_map_is_exact_and_fixture_reset_is_not_an_actor_action():
    generated = json.loads(MAP_JSON.read_text())
    source = yaml.safe_load(MAP_YAML.read_text())
    assert generated["source_yaml_sha256"] == hashlib.sha256(MAP_YAML.read_bytes()).hexdigest()
    assert generated["mappings"] == source["mappings"]
    mapping = source["mappings"]
    assert mapping["Fixture_Reset_PB"]["owner"] == "fixture"
    assert all(not (b["kind"] == "coil" and b["lo"] <= 240 < b["lo"] + b["count"])
               for b in build_blocks(mapping))
    assert all(candidate.address != 240 for candidate in wd_command_menu(mapping))


def test_reset_rejects_missing_generation_edge(monkeypatch):
    mapping = yaml.safe_load(MAP_YAML.read_text())["mappings"]
    def fake_request(_endpoint, operation, address, value=None):
        if operation == "coil":
            return 1
        if operation == "real" and address == 2076:
            return 0.0
        return None
    monkeypatch.setattr("ics_agent.plcbench_wd_storage_v2._request", fake_request)
    monkeypatch.setattr("ics_agent.plcbench_wd_storage_v2.time.sleep", lambda _t: None)
    with pytest.raises(RuntimeError, match="reset edge not proven"):
        reset_wd_storage_v2(("controller", 502), ("simulator", 502), mapping)


def test_reset_rejects_dirty_readback(monkeypatch):
    mapping = yaml.safe_load(MAP_YAML.read_text())["mappings"]
    generations = iter((1.0, 2.0))
    values = {2058: 450.0, 2066: 600.0, 2068: 500.0,
              2072: 100.0, 2074: 100.0, 2078: 101.0, 2080: 0.0}
    def fake_request(_endpoint, operation, address, value=None):
        if operation == "coil":
            return 1
        if operation == "real" and address == 2076:
            return next(generations)
        if operation == "real":
            return values[address]
        return None
    monkeypatch.setattr("ics_agent.plcbench_wd_storage_v2._request", fake_request)
    monkeypatch.setattr("ics_agent.plcbench_wd_storage_v2.time.sleep", lambda _t: None)
    with pytest.raises(RuntimeError, match="outside fixture band"):
        reset_wd_storage_v2(("controller", 502), ("simulator", 502), mapping)
