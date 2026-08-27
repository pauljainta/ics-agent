from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from ics_agent.config import ConfigError, load_config
from ics_agent.infrastructure import SPHERE_INFRASTRUCTURE_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "configs" / "experiments"


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [REPO_ROOT / line for line in completed.stdout.splitlines() if line.strip()]


def test_tracked_sphere_experiment_configs_use_logical_profiles_only() -> None:
    direct_infra_keys = SPHERE_INFRASTRUCTURE_FIELDS - {
        "backend",
        "testbed",
        "infrastructure_profile",
    }
    for path in sorted(EXPERIMENT_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            continue
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            continue
        if environment.get("backend") != "sphere_openplc":
            continue
        assert set(environment) == {
            "backend",
            "testbed",
            "infrastructure_profile",
        }, path
        assert not (set(environment) & direct_infra_keys), path


def test_load_config_resolves_sphere_profile_from_local_config() -> None:
    config = load_config(
        "water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml"
    )

    environment = config["environment"]
    assert environment["backend"] == "sphere_openplc"
    assert environment["infrastructure_profile"] == "water_treatment"
    assert environment["merge_server"] == "merge-test-host"
    assert environment["xdc"] == "treatment-xdc"
    assert environment["_infrastructure"]["source"] == "local"


def test_load_config_reports_missing_local_infrastructure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ICS_AGENT_SPHERE_CONFIG", str(tmp_path / "missing.yaml"))

    with pytest.raises(ConfigError, match="local infrastructure config is missing"):
        load_config(
            "water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml"
        )


def test_xdc_override_is_explicit_and_recorded(monkeypatch) -> None:
    monkeypatch.setenv("ICS_AGENT_SPHERE_XDC_OVERRIDE", "override-test-xdc.user")

    config = load_config("water_distribution_black_box_vs_agentic_defender.yaml")

    environment = config["environment"]
    assert environment["xdc"] == "override-test-xdc"
    assert environment["_infrastructure"]["xdc_overridden"] is True


def test_tracked_files_do_not_embed_concrete_infrastructure_identifiers() -> None:
    forbidden_literals = [
        "".join(parts)
        for parts in (
            ("sphere", "-testbed"),
            ("openplc", "water"),
            ("l", "garcia"),
            ("jainta", "pauljp"),
            ("agentic", "wt"),
            ("agentic", "wd"),
            ("wargame", "stable"),
        )
    ]
    private_network = re.compile(r"\b(?:10\.100\.0|172\.30\.0)\.\d+\b")

    offenders: list[str] = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for literal in forbidden_literals:
            if literal in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains `{literal}`")
        if private_network.search(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)} contains private node address")

    assert offenders == []
