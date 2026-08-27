from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "Artifact"
OPENPLC_SUPPORT_DIR = REPO_ROOT / "vendor" / "openplc"
OPENPLC_SUPPORT_SCRIPTS = OPENPLC_SUPPORT_DIR / "scripts"
RUNTIME_CONFIG_ROOT = REPO_ROOT / "configs"
RUNTIME_EXPERIMENT_ROOT = RUNTIME_CONFIG_ROOT / "experiments"
RUNTIME_SCENARIO_ROOT = REPO_ROOT / "testbeds" / "water_treatment" / "scenarios"
SHARED_PROFILE_REGISTRY = RUNTIME_CONFIG_ROOT / "profiles" / "shared.yaml"


def artifact_root() -> Path:
    override = os.getenv("ICS_AGENT_ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ARTIFACT_ROOT.resolve()


def artifact_config_root() -> Path:
    return RUNTIME_CONFIG_ROOT


def artifact_experiment_root() -> Path:
    return RUNTIME_EXPERIMENT_ROOT


def artifact_scenario_root() -> Path:
    return RUNTIME_SCENARIO_ROOT


def results_root() -> Path:
    return artifact_root() / "Results"


def config_roots() -> tuple[Path, ...]:
    return (RUNTIME_CONFIG_ROOT,)


def experiment_roots() -> tuple[Path, ...]:
    return (RUNTIME_EXPERIMENT_ROOT,)


def scenario_roots() -> tuple[Path, ...]:
    return (RUNTIME_SCENARIO_ROOT,)


def profile_registry_paths(testbed_id: str = "water_treatment") -> tuple[Path, ...]:
    return (
        SHARED_PROFILE_REGISTRY,
        REPO_ROOT / "testbeds" / testbed_id / "profiles.yaml",
    )


def active_scenario_config_root() -> Path:
    return RUNTIME_SCENARIO_ROOT


CONFIG_ROOT = RUNTIME_CONFIG_ROOT
RESULTS_ROOT = results_root()
SCENARIO_CONFIG_ROOT = RUNTIME_SCENARIO_ROOT


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    artifact_anchor = artifact_root().resolve()
    try:
        return (Path("Artifact") / resolved.relative_to(artifact_anchor)).as_posix()
    except ValueError:
        pass

    for anchor in (REPO_ROOT, WORKSPACE_ROOT, OPENPLC_SUPPORT_DIR):
        try:
            return resolved.relative_to(anchor.resolve()).as_posix()
        except ValueError:
            continue
    return str(resolved)
