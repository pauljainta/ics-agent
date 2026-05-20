from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, artifact_config_root, artifact_experiment_root, artifact_scenario_root
from .utils import load_yaml


class ConfigError(ValueError):
    """Raised when a runtime configuration file is invalid."""


def resolve_repo_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = base_dir or REPO_ROOT
    return (anchor / path).resolve()


def _strip_prefix(parts: tuple[str, ...], prefix: tuple[str, ...]) -> Path | None:
    if parts[: len(prefix)] != prefix:
        return None
    suffix = parts[len(prefix) :]
    return Path(*suffix) if suffix else Path()


def resolve_runtime_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()

    normalized = path.as_posix().lstrip("./")
    parts = Path(normalized).parts
    search_plan: list[tuple[Path, tuple[Path, ...]]] = []

    for prefix, roots in (
        (("configs", "experiments"), (artifact_experiment_root(),)),
        (("experiments",), (artifact_experiment_root(),)),
        (("configs", "scenarios"), (artifact_scenario_root(),)),
        (("scenarios",), (artifact_scenario_root(),)),
        (("configs",), (artifact_config_root(),)),
    ):
        stripped = _strip_prefix(parts, prefix)
        if stripped is not None:
            search_plan = [(stripped, roots)]
            break

    if not search_plan:
        search_plan = [
            (path, (artifact_config_root(),)),
            (path, (artifact_experiment_root(),)),
            (path, (artifact_scenario_root(),)),
        ]

    first_candidate: Path | None = None
    for relative_path, roots in search_plan:
        for root in roots:
            candidate = (root / relative_path).resolve()
            if first_candidate is None:
                first_candidate = candidate
            if candidate.exists():
                return candidate

    if first_candidate is None:
        raise ConfigError(f"unable to resolve runtime config path: {value}")
    raise ConfigError(
        f"runtime config not found for `{value}` under Artifact roots: "
        f"{artifact_config_root()}, {artifact_experiment_root()}, {artifact_scenario_root()}"
    )


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_runtime_config_path(path)
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise ConfigError(f"expected mapping at {resolved}")
    payload["_path"] = resolved
    return payload


def require_fields(payload: dict[str, Any], fields: list[str], *, ctx: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ConfigError(f"{ctx} missing required fields: {', '.join(missing)}")
