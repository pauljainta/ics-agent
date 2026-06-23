from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_ARTIFACT_ROOT = WORKSPACE_ROOT / "Artifact"
OPENPLC_SUPPORT_DIR = REPO_ROOT / "vendor" / "openplc"
OPENPLC_SUPPORT_SCRIPTS = OPENPLC_SUPPORT_DIR / "scripts"


def artifact_root() -> Path:
    override = os.getenv("ICS_AGENT_ARTIFACT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ARTIFACT_ROOT.resolve()


def artifact_config_root() -> Path:
    return artifact_root() / "Configs"


def artifact_experiment_root() -> Path:
    return artifact_root() / "Experiments"


def artifact_scenario_root() -> Path:
    return artifact_root() / "Scenarios"


def results_root() -> Path:
    return artifact_root() / "Results"


def config_roots() -> tuple[Path, ...]:
    return (artifact_config_root(),)


def experiment_roots() -> tuple[Path, ...]:
    return (artifact_experiment_root(),)


def scenario_roots() -> tuple[Path, ...]:
    return (artifact_scenario_root(),)


def active_scenario_config_root() -> Path:
    return artifact_scenario_root()


CONFIG_ROOT = artifact_config_root()
RESULTS_ROOT = results_root()
SCENARIO_CONFIG_ROOT = active_scenario_config_root()


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
