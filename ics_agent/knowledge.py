from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ConfigError, resolve_repo_path
from .paths import REPO_ROOT


_MANUAL_CACHE: dict[tuple[str, ...], str] = {}


def load_manual_bundle(paths: list[str | Path]) -> str:
    if not paths:
        return ""

    resolved_paths = tuple(
        str(resolve_repo_path(path, base_dir=REPO_ROOT)) for path in paths
    )
    cached = _MANUAL_CACHE.get(resolved_paths)
    if cached is not None:
        return cached

    sections: list[str] = []
    for path_text in resolved_paths:
        path = Path(path_text)
        body = path.read_text().strip()
        if not body:
            continue
        title = path.stem.replace("_", " ").replace("-", " ").title()
        sections.append(f"[{title}]\n{body}")

    manual_text = "\n\n".join(sections)
    _MANUAL_CACHE[resolved_paths] = manual_text
    return manual_text


def validate_manual_paths(paths: Any, *, profile_id: str) -> list[str]:
    if paths is None:
        return []
    if not isinstance(paths, list):
        raise ConfigError(
            f"profile {profile_id} knowledge_files must be a list when provided"
        )

    validated: list[str] = []
    for index, item in enumerate(paths):
        if not isinstance(item, (str, Path)):
            raise ConfigError(
                f"profile {profile_id} knowledge_files[{index}] must be a path string"
            )
        resolved = resolve_repo_path(item, base_dir=REPO_ROOT)
        if not resolved.exists():
            raise ConfigError(
                f"profile {profile_id} references missing knowledge file: {resolved}"
            )
        validated.append(str(resolved))
    return validated
