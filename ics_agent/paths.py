from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_ROOT = REPO_ROOT / "configs"
RESULTS_ROOT = WORKSPACE_ROOT / "Results"
SCENARIO_CONFIG_ROOT = CONFIG_ROOT / "scenarios"
SPHERE_WT_OPENPLC_DIR = (
    WORKSPACE_ROOT
    / "sphere-usecases"
    / "water-treatment"
    / "implementations"
    / "openplc"
)
SPHERE_WT_OPENPLC_SCRIPTS = SPHERE_WT_OPENPLC_DIR / "scripts"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    for anchor in (REPO_ROOT, WORKSPACE_ROOT, SPHERE_WT_OPENPLC_DIR):
        try:
            return resolved.relative_to(anchor.resolve()).as_posix()
        except ValueError:
            continue
    return str(resolved)
