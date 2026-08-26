from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, portable_path
from .utils import load_yaml


class InfrastructureConfigError(ValueError):
    """Raised when a logical infrastructure profile cannot be resolved."""


DEFAULT_INFRASTRUCTURE_CONFIG = REPO_ROOT / "configs" / "private" / "sphere.local.yaml"

SPHERE_REQUIRED_FIELDS = (
    "merge_server",
    "experiment",
    "realization",
    "revision",
    "xdc",
    "access_host",
    "engineering_host",
    "jump_host",
    "remote_bind_host",
    "controller_port",
    "simulator_port",
    "controller_process_ip",
    "simulator_process_ip",
)

SPHERE_INFRASTRUCTURE_FIELDS = frozenset(
    {
        *SPHERE_REQUIRED_FIELDS,
        "keep_running",
        "jump_port",
        "start_bridge",
        "remote_roles",
        "remote_role_port",
        "remote_role_local_ports",
        "remote_controller_port",
        "remote_simulator_port",
        "reset_wait_sec",
        "loop_sleep_sec",
        "bridge_host",
        "bridge_cycle_ms",
        "bridge_package_targets",
        "bridge_start_timeout_sec",
        "tunnel_start_timeout_sec",
        "preferred_xdcs",
        "xdc_candidate_prefix",
        "allow_xdc_discovery",
    }
)


def _truthy(value: str | bool | int | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_config_path(path_value: str | Path | None) -> Path:
    if path_value is None or str(path_value).strip() == "":
        return DEFAULT_INFRASTRUCTURE_CONFIG
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_profile(config_path: Path, profile_name: str) -> dict[str, Any]:
    if not config_path.exists():
        raise InfrastructureConfigError(
            "SPHERE backend config uses logical infrastructure profile "
            f"`{profile_name}`, but local infrastructure config is missing. "
            "Create an ignored local file from "
            "`configs/infrastructure/sphere.example.yaml`, or set "
            "`ICS_AGENT_SPHERE_CONFIG` to its path."
        )
    payload = load_yaml(config_path)
    if not isinstance(payload, dict):
        raise InfrastructureConfigError(
            f"expected mapping in local infrastructure config `{portable_path(config_path)}`"
        )
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise InfrastructureConfigError(
            f"local infrastructure config `{portable_path(config_path)}` is missing `profiles`"
        )
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise InfrastructureConfigError(
            f"local infrastructure config `{portable_path(config_path)}` has no "
            f"profile `{profile_name}`; available profiles: {available}"
        )
    return dict(profile)


def _required_missing(payload: dict[str, Any]) -> list[str]:
    return [
        field
        for field in SPHERE_REQUIRED_FIELDS
        if str(payload.get(field, "")).strip() == ""
    ]


def resolve_infrastructure(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Expand logical runtime infrastructure profiles into executable config.

    Tracked experiment configs should describe the testbed and profile name only.
    Concrete SPHERE deployment coordinates are loaded from an ignored local file.
    """

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        return payload
    if str(environment.get("backend", "")).strip() != "sphere_openplc":
        return payload

    resolved_payload = deepcopy(payload)
    resolved_env = dict(resolved_payload.get("environment") or {})
    profile_name = str(
        resolved_env.get("infrastructure_profile")
        or resolved_env.get("testbed")
        or resolved_payload.get("testbed")
        or "water_treatment"
    ).strip()
    if not profile_name:
        raise InfrastructureConfigError(
            f"{portable_path(source_path) if source_path else 'runtime config'} "
            "uses `sphere_openplc` but does not declare `environment.infrastructure_profile`"
        )

    config_path = _resolve_config_path(
        resolved_env.get("infrastructure_config")
        or os.environ.get("ICS_AGENT_SPHERE_CONFIG")
    )
    profile = _load_profile(config_path, profile_name)

    merged_env: dict[str, Any] = {
        "backend": "sphere_openplc",
        "testbed": resolved_env.get("testbed") or resolved_payload.get("testbed") or profile_name,
        "infrastructure_profile": profile_name,
    }
    for key, value in profile.items():
        if key in SPHERE_INFRASTRUCTURE_FIELDS:
            merged_env[key] = value
    for key, value in resolved_env.items():
        if key not in SPHERE_INFRASTRUCTURE_FIELDS and key != "infrastructure_config":
            merged_env[key] = value

    merge_server_override = (
        os.environ.get("ICS_AGENT_MERGE_SERVER_OVERRIDE")
        or os.environ.get("MRG_SERVER")
        or os.environ.get("MERGE_SERVER")
    )
    if merge_server_override:
        merged_env["merge_server"] = merge_server_override

    xdc_override = os.environ.get("ICS_AGENT_SPHERE_XDC_OVERRIDE")
    if xdc_override:
        merged_env["xdc"] = xdc_override.split(".", 1)[0]

    allow_discovery = _truthy(os.environ.get("ICS_AGENT_ALLOW_XDC_DISCOVERY"))
    if allow_discovery:
        merged_env["allow_xdc_discovery"] = True

    missing = _required_missing(merged_env)
    if missing:
        raise InfrastructureConfigError(
            f"infrastructure profile `{profile_name}` is missing required fields: "
            + ", ".join(missing)
        )

    merged_env["_infrastructure"] = {
        "profile": profile_name,
        "config_path": portable_path(config_path),
        "source": "local",
        "merge_server_overridden": bool(merge_server_override),
        "xdc_overridden": bool(xdc_override),
        "xdc_discovery_allowed": allow_discovery,
    }
    resolved_payload["environment"] = merged_env
    return resolved_payload
