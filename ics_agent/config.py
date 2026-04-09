from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import CONFIG_ROOT, REPO_ROOT
from .utils import load_yaml


class ConfigError(ValueError):
    """Raised when a runtime configuration file is invalid."""


def resolve_repo_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = base_dir or REPO_ROOT
    return (anchor / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path, base_dir=CONFIG_ROOT)
    payload = load_yaml(resolved)
    if not isinstance(payload, dict):
        raise ConfigError(f"expected mapping at {resolved}")
    payload["_path"] = resolved
    return payload


def require_fields(payload: dict[str, Any], fields: list[str], *, ctx: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ConfigError(f"{ctx} missing required fields: {', '.join(missing)}")
