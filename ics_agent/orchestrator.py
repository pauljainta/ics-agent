from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .action_semantics import action_type, containment_metadata, is_symbolic_containment_action
from .agents import AgenticRoleAgent, RuleBasedRoleAgent, ScriptedRoleAgent
from .config import ConfigError, load_config, require_fields, resolve_repo_path
from .environment import EnvironmentError, OpenPLCEnvironment
from .knowledge import validate_manual_paths
from .mitre import ATTACKER_PHASES
from .metrics import compute_metrics
from .paths import REPO_ROOT, portable_path, results_root
from .reporting import write_json, write_run_bundle
from .scenarios import ScenarioCatalog
from .types import RunBundle, ScenarioDefinition
from .utils import load_yaml, stable_hash, utc_now


ROLE_ORDER = ["planner", "observer", "defender", "operator", "attacker"]
PLANNING_ROLES = ["planner"]
ACTION_ROLES = ["operator", "attacker"]
MONITORING_ROLES = ["observer", "defender"]
OBSERVER_ROLES = ["observer"]
DEFENDER_ROLES = ["defender"]


def canonical_action_name(action: dict[str, Any] | str) -> str:
    if isinstance(action, str):
        return action
    action_type = action.get("type", "unknown")
    process_id = action.get("process_id")
    if action_type == "inspect_surface":
        return f"inspect_surface:{action.get('surface', 'unknown')}"
    if action_type == "set_manual_valve":
        chemical = action.get("chemical", "unknown")
        state = "open" if action.get("open", True) else "close"
        return f"{process_id}:{action_type}:{chemical}:{state}"
    if action_type == "select_process":
        state = "on" if action.get("active", True) else "off"
        return f"{process_id}:{action_type}:{state}"
    return str(action_type)


class ExperimentOrchestrator:
    def __init__(self, catalog: ScenarioCatalog | None = None) -> None:
        self.catalog = catalog or ScenarioCatalog()
        self._current_reproducibility: dict[str, Any] = {}
        raw_profiles = load_config("agent_profiles.yaml").get("profiles", {})
        overlay_path = REPO_ROOT / "configs" / "agent_profiles_overrides.yaml"
        if overlay_path.exists():
            overlay_payload = load_yaml(overlay_path)
            if not isinstance(overlay_payload, dict):
                raise ConfigError(
                    f"expected mapping at repo profile overlay {overlay_path}"
                )
            overlay_profiles = overlay_payload.get("profiles", {})
            if not isinstance(overlay_profiles, dict):
                raise ConfigError(
                    f"repo profile overlay {overlay_path} must define a `profiles` mapping"
                )
            for profile_id, profile in overlay_profiles.items():
                base_profile = raw_profiles.get(profile_id)
                if isinstance(base_profile, dict) and isinstance(profile, dict):
                    raw_profiles[profile_id] = self._deep_merge_mapping(
                        base_profile,
                        profile,
                    )
                else:
                    raw_profiles[profile_id] = deepcopy(profile)
        self.agent_profiles = {
            profile_id: self._normalized_profile(profile)
            for profile_id, profile in raw_profiles.items()
        }
        if not isinstance(self.agent_profiles, dict):
            raise ConfigError("agent_profiles.yaml must define a `profiles` mapping")

    @staticmethod
    def _normalized_profile(profile: Any) -> Any:
        if not isinstance(profile, dict):
            return profile
        normalized = dict(profile)
        role = str(normalized.get("role", "")).strip().lower()
        knowledge_mode = str(normalized.get("knowledge_mode", "")).strip().lower()
        if knowledge_mode == "practical_black_box":
            knowledge_mode = "gray_box"
            normalized["knowledge_mode"] = knowledge_mode
        if role == "attacker" and knowledge_mode in {
            "gray_box",
            "black_box",
            "white_box",
        }:
            allowed = list(normalized.get("allowed_action_types", []))
            if "inspect_surface" not in allowed:
                insertion_index = (
                    allowed.index("hmi_estop") + 1 if "hmi_estop" in allowed else 0
                )
                allowed.insert(insertion_index, "inspect_surface")
                normalized["allowed_action_types"] = allowed
        return normalized

    @staticmethod
    def _deep_merge_mapping(
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ExperimentOrchestrator._deep_merge_mapping(
                    merged[key],
                    value,
                )
            else:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _initial_runtime_containment() -> dict[str, Any]:
        return {
            "active": False,
            "activated_at_sec": None,
            "activated_by_action": None,
            "activated_by_role": None,
            "surface_scope": None,
            "target_surfaces": [],
            "is_symbolic": False,
            "blocked_action_count": 0,
        }

    @staticmethod
    def _symbolic_action_state_after(
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        state_name = (
            observation.get("STATE_NAME")
            or observation.get("state_name")
            or "UNKNOWN"
        )
        metadata = containment_metadata(action)
        return {
            "state_name": state_name,
            "symbolic_containment": metadata,
        }

    @staticmethod
    def _record_containment_activation(
        *,
        runtime_containment: dict[str, Any],
        bundle: RunBundle,
        action: dict[str, Any],
        role: str,
        elapsed: float,
    ) -> None:
        metadata = containment_metadata(action)
        if runtime_containment.get("active"):
            return
        runtime_containment.update(
            {
                "active": True,
                "activated_at_sec": round(elapsed, 3),
                "activated_by_action": dict(action),
                "activated_by_role": role,
                "surface_scope": metadata.get("surface_scope"),
                "target_surfaces": list(metadata.get("target_surfaces", [])),
                "is_symbolic": bool(metadata.get("is_symbolic")),
            }
        )
        bundle.system_events.append(
            {
                "timestamp_utc": utc_now(),
                "time_sec": round(elapsed, 3),
                "event_type": "attacker_execution_cutoff_activated",
                "role": role,
                "action": dict(action),
                "containment": metadata,
            }
        )

    @staticmethod
    def _record_blocked_attacker_action(
        *,
        runtime_containment: dict[str, Any],
        bundle: RunBundle,
        decision: Any,
        action: dict[str, Any],
        elapsed: float,
    ) -> None:
        runtime_containment["blocked_action_count"] = int(
            runtime_containment.get("blocked_action_count", 0) or 0
        ) + 1
        bundle.system_events.append(
            {
                "timestamp_utc": utc_now(),
                "time_sec": round(elapsed, 3),
                "event_type": "attacker_action_blocked",
                "role": "attacker",
                "agent_name": decision.agent_name,
                "attempted_action": dict(action),
                "attempted_action_type": action_type(action),
                "blocked_by_action": dict(
                    runtime_containment.get("activated_by_action") or {}
                ),
                "blocked_by_time_sec": runtime_containment.get("activated_at_sec"),
                "containment_scope": runtime_containment.get("surface_scope"),
                "target_surfaces": list(
                    runtime_containment.get("target_surfaces", [])
                ),
            }
        )

    def _validate_runtime_profile(self, profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(profile, dict):
            raise ConfigError(f"profile {profile_id} must be a mapping")
        role = str(profile.get("role", "")).strip()
        if not role:
            raise ConfigError(f"profile {profile_id} is missing a role")
        allowed = profile.get("allowed_action_types")
        if not isinstance(allowed, list) or not allowed:
            raise ConfigError(
                f"profile {profile_id} must define non-empty allowed_action_types"
            )

        family = str(profile.get("family", "scripted"))
        knowledge_file_count = 0
        if family == "agentic":
            llm = profile.get("llm", {})
            if not isinstance(llm, dict):
                raise ConfigError(f"profile {profile_id} llm config must be a mapping")
            prompt_file = llm.get("prompt_file")
            if not prompt_file:
                raise ConfigError(
                    f"profile {profile_id} must define llm.prompt_file for agentic use"
                )
            prompt_path = resolve_repo_path(prompt_file, base_dir=REPO_ROOT)
            if not prompt_path.exists():
                raise ConfigError(
                    f"profile {profile_id} references missing prompt file: {prompt_path}"
                )

            knowledge_paths = validate_manual_paths(
                profile.get("knowledge_files"),
                profile_id=profile_id,
            )
            knowledge_file_count = len(knowledge_paths)

        return {
            "role": role,
            "family": family,
            "knowledge_file_count": knowledge_file_count,
        }

    def _validated_profile_overrides(
        self,
        *,
        profiles: dict[str, str],
        raw_overrides: Any,
        condition_label: str,
    ) -> dict[str, dict[str, Any]]:
        if raw_overrides in (None, {}):
            return {}
        if not isinstance(raw_overrides, dict):
            raise ConfigError(f"{condition_label} profile_overrides must be a mapping")

        validated: dict[str, dict[str, Any]] = {}
        for role, override in raw_overrides.items():
            if role not in profiles:
                raise ConfigError(
                    f"{condition_label} profile_overrides references unknown role `{role}`"
                )
            if not isinstance(override, dict):
                raise ConfigError(
                    f"{condition_label} profile_overrides[{role}] must be a mapping"
                )
            validated[str(role)] = deepcopy(override)
        return validated

    def _effective_condition_profiles(
        self,
        *,
        profiles: dict[str, str],
        profile_overrides: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        effective: dict[str, dict[str, Any]] = {}
        for role, profile_id in profiles.items():
            base_profile = deepcopy(self.agent_profiles[profile_id])
            override = dict(profile_overrides.get(role, {}))
            merged = (
                self._deep_merge_mapping(base_profile, override)
                if override
                else base_profile
            )
            normalized = self._normalized_profile(merged)
            metadata = self._validate_runtime_profile(profile_id, normalized)
            if metadata["role"] != role:
                raise ConfigError(
                    f"profile override for `{profile_id}` changed role to `{metadata['role']}`; expected `{role}`"
                )
            effective[str(role)] = normalized
        return effective

    def validate_runtime(self) -> dict[str, Any]:
        agentic_profile_count = 0
        knowledge_file_count = 0

        for profile_id, profile in self.agent_profiles.items():
            metadata = self._validate_runtime_profile(profile_id, profile)
            if metadata["family"] == "agentic":
                agentic_profile_count += 1
                knowledge_file_count += int(metadata["knowledge_file_count"])

        return {
            "profile_count": len(self.agent_profiles),
            "agentic_profile_count": agentic_profile_count,
            "knowledge_file_count": knowledge_file_count,
        }

    @staticmethod
    def _use_canonical_result_root(config: dict[str, Any]) -> bool:
        return bool(config.get("use_canonical_result_root", False))

    @staticmethod
    def _git_commit(path: Path) -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", "HEAD"],
                    text=True,
                )
                .strip()
                or None
            )
        except Exception:
            return None

    def _collect_decoding_settings(
        self,
        profiles: dict[str, str],
        *,
        effective_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        settings: dict[str, dict[str, Any]] = {}
        for role, profile_id in profiles.items():
            profile = (
                dict((effective_profiles or {}).get(role) or {})
                or self.agent_profiles.get(profile_id, {})
            )
            if not isinstance(profile, dict) or profile.get("family") != "agentic":
                continue
            llm = profile.get("llm", {})
            if not isinstance(llm, dict):
                continue
            settings[str(role)] = {
                "profile_id": str(profile_id),
                "backend": llm.get("backend"),
                "framework": llm.get("framework"),
                "model_id": llm.get("model_id"),
                "model_path": os.environ.get("QWEN_MODEL_PATH", "").strip()
                or llm.get("model_id"),
                "temperature": float(llm.get("temperature", 0.0) or 0.0),
                "do_sample": float(llm.get("temperature", 0.0) or 0.0) > 0.0,
                "max_new_tokens": llm.get("max_new_tokens"),
                "quantization": llm.get("quantization"),
                "local_files_only": bool(llm.get("local_files_only", True)),
                "require_cuda": bool(llm.get("require_cuda", True)),
                "strict_backend": bool(llm.get("strict_backend", True)),
                "allow_langchain_fallback": bool(
                    llm.get("allow_langchain_fallback", True)
                ),
            }
        return settings

    def _suite_decoding_settings(
        self,
        conditions: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        suite_settings: dict[str, dict[str, dict[str, Any]]] = {}
        for condition_spec in conditions:
            profiles = condition_spec.get("profiles", {})
            condition_name = self._condition_name(
                profiles,
                override=condition_spec.get("name"),
            )
            suite_settings[condition_name] = self._collect_decoding_settings(
                profiles,
                effective_profiles=condition_spec.get("effective_profiles"),
            )
        return suite_settings

    def _condition_reproducibility(
        self,
        condition_name: str,
        profiles: dict[str, str],
        *,
        effective_profiles: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = dict(self._current_reproducibility)
        suite_decoding = dict(payload.get("suite_decoding") or {})
        payload["decoding"] = self._collect_decoding_settings(
            profiles,
            effective_profiles=effective_profiles,
        )
        payload["suite_decoding"] = suite_decoding
        payload["condition_name"] = condition_name
        return payload

    def _configure_reproducibility(
        self,
        *,
        config: dict[str, Any],
        conditions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = config.get("reproducibility", {})
        if not isinstance(payload, dict):
            payload = {}
        seed_value = payload.get("seed", config.get("seed"))
        seed = int(seed_value) if seed_value is not None else None
        remaining_nondeterminism: list[str] = []

        numpy_available = False
        torch_available = False
        torch_cuda_available = False
        cudnn_deterministic = None
        cudnn_benchmark = None
        deterministic_algorithms = None

        if seed is not None:
            random.seed(seed)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                import numpy as np  # type: ignore

                np.random.seed(seed)
                numpy_available = True
            except Exception:
                remaining_nondeterminism.append(
                    "NumPy was unavailable, so NumPy RNG could not be seeded."
                )
            try:
                import torch  # type: ignore

                torch_available = True
                torch.manual_seed(seed)
                torch_cuda_available = bool(torch.cuda.is_available())
                if torch_cuda_available:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    torch.use_deterministic_algorithms(True)
                deterministic_algorithms = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
                    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            except Exception as exc:
                remaining_nondeterminism.append(
                    f"PyTorch deterministic seeding was unavailable: {exc}"
                )

        if seed is not None and os.environ.get("PYTHONHASHSEED") != str(seed):
            remaining_nondeterminism.append(
                "PYTHONHASHSEED was not set before process start, so set-iteration order may retain minor nondeterminism."
            )

        return {
            "seed": seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "numpy_seeded": numpy_available,
            "torch_seeded": torch_available,
            "torch_cuda_seeded": torch_cuda_available,
            "torch_deterministic_algorithms": deterministic_algorithms,
            "cudnn_deterministic": cudnn_deterministic,
            "cudnn_benchmark": cudnn_benchmark,
            "experiment_config_path": portable_path(Path(config["_path"])),
            "ics_agent_repo_commit": self._git_commit(REPO_ROOT),
            "suite_decoding": self._suite_decoding_settings(conditions),
            "remaining_nondeterminism": remaining_nondeterminism,
        }

    def run(self, experiment_config_path: str | Path) -> dict[str, Any]:
        config = load_config(experiment_config_path)
        require_fields(
            config, ["experiment_name", "scenario_selectors"], ctx="experiment config"
        )

        experiment_name = str(config["experiment_name"])
        scenarios = self.catalog.resolve_selectors(
            list(config.get("scenario_selectors", []))
        )
        if not scenarios:
            raise ConfigError(f"no scenarios matched selectors in {config['_path']}")

        conditions = self._enumerate_conditions(config)
        reproducibility = self._configure_reproducibility(
            config=config,
            conditions=conditions,
        )
        self._current_reproducibility = reproducibility
        timestamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
        if self._use_canonical_result_root(config):
            experiment_id = str(config.get("experiment_id", experiment_name))
        else:
            experiment_id = f"{timestamp[:15]}-{stable_hash({'config': portable_path(config['_path']), 'conditions': conditions})}"
        results_base_dir = (
            Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
        )
        resolved_results_root = resolve_repo_path(
            config.get("results_root", str(results_root())), base_dir=results_base_dir
        )
        if self._use_canonical_result_root(config):
            experiment_root = resolved_results_root / experiment_name
        else:
            experiment_root = resolved_results_root / experiment_name / experiment_id
        experiment_root.mkdir(parents=True, exist_ok=True)

        suite_summary: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "config_path": portable_path(config["_path"]),
            "results_root": portable_path(experiment_root),
            "timestamp_utc": utc_now(),
            "status": "completed",
            "use_canonical_result_root": self._use_canonical_result_root(config),
            "reproducibility": reproducibility,
            "conditions": [],
        }
        suite_trajectory_baseline_runs: list[dict[str, Any]] = []
        trajectory_baseline_roots = self._trajectory_baseline_roots(config)

        for condition_spec in conditions:
            condition = condition_spec["profiles"]
            condition_name = self._condition_name(
                condition, override=condition_spec.get("name")
            )
            condition_root = experiment_root / condition_name
            condition_reproducibility = self._condition_reproducibility(
                condition_name,
                condition,
                effective_profiles=condition_spec.get("effective_profiles"),
            )
            condition_summary: dict[str, Any] = {
                "condition_name": condition_name,
                "profiles": condition,
                "runs": [],
                "status": "completed",
                "reproducibility": condition_reproducibility,
            }
            for scenario in scenarios:
                for repeat in range(1, int(config.get("repeats", 1)) + 1):
                    run_id = (
                        f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"
                    )
                    run_dir = (
                        condition_root
                        / scenario.suite
                        / scenario.scenario_id
                        / f"trial-{repeat:03d}"
                    )
                    if config.get("resume") and (run_dir / "metrics.json").exists():
                        condition_summary["runs"].append(
                            {
                                "run_id": run_id,
                                "scenario": scenario.catalog_id,
                                "status": "skipped_existing",
                                "run_dir": portable_path(run_dir),
                            }
                        )
                        continue
                    bundle = self._run_single(
                        experiment_id=experiment_id,
                        condition_name=condition_name,
                        condition=condition,
                        effective_profiles=condition_spec.get("effective_profiles"),
                        scenario=scenario,
                        run_id=run_id,
                        run_dir=run_dir,
                        config=config,
                    )
                    condition_summary["runs"].append(bundle["summary"])
                    if bundle["summary"]["status"] != "completed":
                        condition_summary["status"] = "failed"
                        suite_summary["status"] = "failed"
            trajectory_baseline_runs = self._load_trajectory_baseline_runs(
                trajectory_baseline_roots,
                condition_name=condition_name,
            )
            suite_trajectory_baseline_runs.extend(trajectory_baseline_runs)
            condition_summary["nominal_metrics"] = self._aggregate_nominal_metrics(
                condition_summary["runs"]
            )
            condition_summary["attack_metrics"] = self._aggregate_attack_metrics(
                condition_summary["runs"]
            )
            condition_summary["mitre_phase_metrics"] = self._aggregate_mitre_phase_metrics(
                condition_summary["runs"]
            )
            condition_summary["trajectory_metrics"] = self._aggregate_trajectory_metrics(
                condition_summary["runs"] + trajectory_baseline_runs,
                condition=condition,
            )
            if trajectory_baseline_roots:
                condition_summary["trajectory_metrics"]["baseline_roots"] = [
                    portable_path(path) for path in trajectory_baseline_roots
                ]
                condition_summary["trajectory_metrics"]["baseline_run_count"] = len(
                    trajectory_baseline_runs
                )
            write_json(condition_root / "condition_summary.json", condition_summary)
            write_json(
                condition_root / "attack_metrics_summary.json",
                condition_summary["attack_metrics"],
            )
            write_json(
                condition_root / "nominal_metrics_summary.json",
                condition_summary["nominal_metrics"],
            )
            write_json(
                condition_root / "mitre_phase_metrics.json",
                condition_summary["mitre_phase_metrics"],
            )
            write_json(
                condition_root / "trajectory_metrics_summary.json",
                condition_summary["trajectory_metrics"],
            )
            suite_summary["conditions"].append(condition_summary)

        suite_summary["nominal_metrics"] = self._aggregate_nominal_metrics(
            [
                run
                for condition in suite_summary["conditions"]
                for run in condition.get("runs", [])
            ]
        )
        suite_summary["attack_metrics"] = self._aggregate_attack_metrics(
            [
                run
                for condition in suite_summary["conditions"]
                for run in condition.get("runs", [])
            ]
        )
        suite_summary["mitre_phase_metrics"] = self._aggregate_mitre_phase_metrics(
            [
                run
                for condition in suite_summary["conditions"]
                for run in condition.get("runs", [])
            ]
        )
        suite_summary["trajectory_metrics"] = self._aggregate_trajectory_metrics(
            [
                run
                for condition in suite_summary["conditions"]
                for run in condition.get("runs", [])
            ]
            + suite_trajectory_baseline_runs
        )
        if trajectory_baseline_roots:
            suite_summary["trajectory_metrics"]["baseline_roots"] = [
                portable_path(path) for path in trajectory_baseline_roots
            ]
            suite_summary["trajectory_metrics"]["baseline_run_count"] = len(
                suite_trajectory_baseline_runs
            )
        write_json(experiment_root / "experiment_summary.json", suite_summary)
        write_json(
            experiment_root / "attack_metrics_summary.json",
            suite_summary["attack_metrics"],
        )
        write_json(
            experiment_root / "nominal_metrics_summary.json",
            suite_summary["nominal_metrics"],
        )
        write_json(
            experiment_root / "mitre_phase_metrics.json",
            suite_summary["mitre_phase_metrics"],
        )
        write_json(
            experiment_root / "trajectory_metrics_summary.json",
            suite_summary["trajectory_metrics"],
        )
        return suite_summary

    def _trajectory_baseline_roots(self, config: dict[str, Any]) -> list[Path]:
        baseline_roots = config.get("trajectory_baseline_roots", [])
        if baseline_roots in (None, []):
            return []
        if not isinstance(baseline_roots, list):
            raise ConfigError("trajectory_baseline_roots must be a list when provided")
        config_base_dir = Path(config["_path"]).parent
        resolved: list[Path] = []
        for index, item in enumerate(baseline_roots):
            if not isinstance(item, (str, Path)):
                raise ConfigError(
                    f"trajectory_baseline_roots[{index}] must be a path-like value"
                )
            resolved_path = resolve_repo_path(item, base_dir=config_base_dir)
            if not resolved_path.exists():
                raise ConfigError(
                    f"trajectory baseline root does not exist: {resolved_path}"
                )
            resolved.append(resolved_path)
        return resolved

    def _load_trajectory_baseline_runs(
        self,
        baseline_roots: list[Path],
        *,
        condition_name: str,
    ) -> list[dict[str, Any]]:
        loaded_runs: list[dict[str, Any]] = []
        for root in baseline_roots:
            summary_path = root / condition_name / "condition_summary.json"
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            runs = payload.get("runs", [])
            if not isinstance(runs, list):
                continue
            for item in runs:
                if not isinstance(item, dict):
                    continue
                run = deepcopy(item)
                run["trajectory_source_root"] = portable_path(root)
                loaded_runs.append(run)
        return loaded_runs

    def _enumerate_conditions(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        explicit_conditions = config.get("conditions")
        if explicit_conditions is not None:
            if not isinstance(explicit_conditions, list) or not explicit_conditions:
                raise ConfigError(
                    "experiment config `conditions` must be a non-empty list"
                )
            resolved: list[dict[str, Any]] = []
            for index, item in enumerate(explicit_conditions):
                if not isinstance(item, dict):
                    raise ConfigError(f"conditions[{index}] must be a mapping")
                profiles = item.get("profiles")
                if not isinstance(profiles, dict) or not profiles:
                    raise ConfigError(
                        f"conditions[{index}] must define a non-empty `profiles` mapping"
                    )
                validated = self._validate_condition_profiles(profiles)
                condition_label = (
                    f"conditions[{index}]"
                    if not item.get("name")
                    else f"condition `{item.get('name')}`"
                )
                profile_overrides = self._validated_profile_overrides(
                    profiles=validated,
                    raw_overrides=item.get("profile_overrides"),
                    condition_label=condition_label,
                )
                resolved.append(
                    {
                        "name": item.get("name"),
                        "profiles": validated,
                        "profile_overrides": profile_overrides,
                        "effective_profiles": self._effective_condition_profiles(
                            profiles=validated,
                            profile_overrides=profile_overrides,
                        ),
                    }
                )
            return resolved

        role_matrix = config.get("role_matrix")
        if not isinstance(role_matrix, dict) or not role_matrix:
            raise ConfigError("experiment config must define role_matrix")
        roles = list(role_matrix.keys())
        choices = []
        for role, profile_ids in role_matrix.items():
            if not isinstance(profile_ids, list) or not profile_ids:
                raise ConfigError(f"role_matrix[{role}] must be a non-empty list")
            for profile_id in profile_ids:
                profile = self.agent_profiles.get(profile_id)
                if not isinstance(profile, dict):
                    raise ConfigError(f"unknown profile: {profile_id}")
                if profile.get("role") != role:
                    raise ConfigError(
                        f"profile {profile_id} role mismatch: expected {role}, got {profile.get('role')}"
                    )
            choices.append(profile_ids)
        return [
            {
                "name": None,
                "profiles": dict(zip(roles, combo, strict=True)),
                "profile_overrides": {},
                "effective_profiles": self._effective_condition_profiles(
                    profiles=dict(zip(roles, combo, strict=True)),
                    profile_overrides={},
                ),
            }
            for combo in itertools.product(*choices)
        ]

    def _validate_condition_profiles(self, profiles: dict[str, Any]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for role, profile_id in profiles.items():
            profile = self.agent_profiles.get(profile_id)
            if not isinstance(profile, dict):
                raise ConfigError(f"unknown profile: {profile_id}")
            if profile.get("role") != role:
                raise ConfigError(
                    f"profile {profile_id} role mismatch: expected {role}, got {profile.get('role')}"
                )
            validated[str(role)] = str(profile_id)
        expected_roles = set(ROLE_ORDER + ["offline_analysis"])
        missing = sorted(expected_roles.difference(validated))
        if missing:
            raise ConfigError(f"condition missing required roles: {', '.join(missing)}")
        return validated

    def _condition_name(
        self, condition: dict[str, str], *, override: str | None = None
    ) -> str:
        if override:
            return str(override)
        return "__".join(
            f"{role}-{profile}" for role, profile in sorted(condition.items())
        )

    def _create_agent(self, profile_id: str, *, profile: dict[str, Any] | None = None):
        profile = deepcopy(profile or self.agent_profiles[profile_id])
        family = str(profile.get("family", "scripted"))
        if family == "agentic":
            return AgenticRoleAgent(profile_id, profile)
        if family == "rule_based":
            return RuleBasedRoleAgent(profile_id, profile)
        return ScriptedRoleAgent(profile_id, profile)

    def _create_condition_agent(
        self,
        *,
        role: str,
        profile_id: str,
        effective_profiles: dict[str, dict[str, Any]] | None,
    ):
        base_profile = self.agent_profiles[profile_id]
        profile = deepcopy((effective_profiles or {}).get(role) or base_profile)
        if profile == base_profile:
            return self._create_agent(profile_id)
        return self._create_agent(profile_id, profile=profile)

    def _build_plan(
        self, scenario: ScenarioDefinition
    ) -> dict[str, list[dict[str, Any]]]:
        bindings = scenario.metadata.get("role_bindings", {})
        timeline = scenario.execution.get("timeline", [])
        plan: dict[str, list[dict[str, Any]]] = {}
        for role, indices in bindings.items():
            plan[role] = [timeline[int(index)] for index in indices]
        return plan

    def _remaining_role_events(
        self,
        *,
        role: str,
        planned_events: dict[str, list[dict[str, Any]]],
        executed_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        role_plan = planned_events.get(role, [])
        role_actions = [
            item.get("action")
            for item in executed_actions
            if item.get("role") == role and item.get("action") is not None
        ]
        cursor = 0
        for planned in role_plan:
            planned_action = planned.get("action")
            if cursor < len(role_actions) and canonical_action_name(
                role_actions[cursor]
            ) == canonical_action_name(planned_action):
                cursor += 1
                continue
            break
        return role_plan[cursor:]

    def _sanitize_action(
        self,
        *,
        role: str,
        action: dict[str, Any] | None,
        scenario: ScenarioDefinition,
        profile_id: str,
        effective_profile: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if action is None:
            return None, None
        profile = dict(effective_profile or {}) or self.agent_profiles[profile_id]
        allowed_types = set(
            profile.get(
                "allowed_action_types",
                [
                    "hmi_start",
                    "hmi_stop",
                    "hmi_reset",
                    "hmi_estop",
                    "inspect_surface",
                    "select_process",
                    "set_manual_valve",
                    "set_sensor",
                    "noop",
                ],
            )
        )
        action_type = str(action.get("type"))
        if action_type not in allowed_types:
            return (
                None,
                f"action `{action_type}` rejected by guardrails for role {role}",
            )
        if (
            role in {"observer", "planner", "offline_analysis"}
            and action_type != "noop"
        ):
            return None, f"role {role} may not actuate the environment"
        return action, None

    def _run_single(
        self,
        *,
        experiment_id: str,
        condition_name: str,
        condition: dict[str, str],
        effective_profiles: dict[str, dict[str, Any]] | None = None,
        scenario: ScenarioDefinition,
        run_id: str,
        run_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = RunBundle(
            experiment_id=experiment_id,
            condition_name=condition_name,
            run_id=run_id,
            scenario=scenario,
            run_dir=run_dir,
            config_snapshot={
                "experiment_config_path": portable_path(config["_path"]),
                "condition": condition,
                "scenario": scenario.catalog_id,
                "environment": config.get("environment", {}),
                "experiment_rules": {
                    "containment_class_actions_cut_off_later_attacker_execution": True,
                    "containment_rule_scope": "rest_of_run",
                    "timing_metrics_exclude_pre_run_setup_latency": True,
                },
                "reproducibility": self._condition_reproducibility(
                    condition_name,
                    condition,
                    effective_profiles=effective_profiles,
                ),
                "profile_overrides": {
                    role: profile
                    for role, profile in (effective_profiles or {}).items()
                    if profile != self.agent_profiles.get(condition.get(role, ""))
                },
                "profile_assets": {
                    role: {
                        "prompt_file": (
                            dict((effective_profiles or {}).get(role) or {})
                            or self.agent_profiles[profile_id]
                        )
                        .get("llm", {})
                        .get("prompt_file"),
                        "knowledge_files": (
                            dict((effective_profiles or {}).get(role) or {})
                            or self.agent_profiles[profile_id]
                        ).get("knowledge_files", []),
                    }
                    for role, profile_id in condition.items()
                },
            },
        )

        env = OpenPLCEnvironment(config.get("environment", {}))
        started = False
        phase_a_started_monotonic = time.monotonic()
        phase_a_started_utc = utc_now()
        agents: dict[str, Any] = {}
        try:
            if config.get("execution_mode", "live") == "live":
                env.start()
                started = True
                for role, profile_id in condition.items():
                    profile = deepcopy(
                        (effective_profiles or {}).get(role)
                        or self.agent_profiles[profile_id]
                    )
                    agent = env.remote_role_proxy(
                        role=role,
                        profile_name=profile_id,
                        profile=profile,
                    )
                    if agent is None:
                        agent = self._create_condition_agent(
                            role=role,
                            profile_id=profile_id,
                            effective_profiles=effective_profiles,
                        )
                    agents[role] = agent
                for role, agent in agents.items():
                    agent.reset(scenario)
                    runtime_info = getattr(agent, "runtime_info", None)
                    if isinstance(runtime_info, dict) and runtime_info:
                        env.record_role_runtime(role, runtime_info)
                        bundle.system_events.append(
                            {
                                "timestamp_utc": utc_now(),
                                "time_sec": None,
                                "event_type": "role_residency",
                                "role": role,
                                "execution_residency": runtime_info.get(
                                    "execution_residency",
                                    "remote",
                                ),
                                "assigned_node": runtime_info.get("assigned_node"),
                                "runtime_host": runtime_info.get("hostname"),
                                "runtime_pid": runtime_info.get("pid"),
                            }
                        )
                env.reset(scenario.execution.get("initial_conditions"))
                phase_a_duration_sec = round(
                    max(0.0, time.monotonic() - phase_a_started_monotonic),
                    3,
                )
                phase_boundary = {
                    "phase_a_started_utc": phase_a_started_utc,
                    "phase_a_completed_utc": utc_now(),
                    "phase_a_duration_sec": phase_a_duration_sec,
                    "wargame_t0_utc": utc_now(),
                    "wargame_t0_elapsed_sec": 0.0,
                    "metric_window": "phase_b_only",
                    "metric_clock_origin": "wargame_t0",
                }
                bundle.config_snapshot["phase_boundary"] = phase_boundary
                bundle.system_events.append(
                    {
                        "timestamp_utc": phase_boundary["wargame_t0_utc"],
                        "time_sec": 0.0,
                        "event_type": "wargame_t0",
                        "phase_a_duration_sec": phase_a_duration_sec,
                    }
                )
                self._run_live_loop(
                    bundle,
                    scenario,
                    agents,
                    condition,
                    effective_profiles,
                    env,
                )
            else:
                agents = {
                    role: self._create_condition_agent(
                        role=role,
                        profile_id=profile_id,
                        effective_profiles=effective_profiles,
                    )
                    for role, profile_id in condition.items()
                }
                for agent in agents.values():
                    agent.reset(scenario)
                replay_source = config.get("replay_source")
                if not replay_source:
                    raise ConfigError("replay execution requires replay_source")
                self._run_replay_loop(
                    bundle,
                    scenario,
                    agents,
                    self._resolve_replay_source_path(
                        replay_source, config_path=Path(config["_path"])
                    ),
                )
            bundle.status = "completed"
        except Exception as exc:  # pragma: no cover - exercised in live runs
            bundle.status = "failed"
            bundle.passed = False
            bundle.errors.append({"timestamp_utc": utc_now(), "error": str(exc)})

        bundle.metrics = compute_metrics(
            bundle.__dict__,
            {
                "metadata": scenario.metadata,
                "suite": scenario.suite,
                "scenario_id": scenario.scenario_id,
            },
        )
        offline_context = {
            "scenario": scenario,
            "metrics": bundle.metrics,
            "passed": bundle.passed,
            "duration_sec": float(scenario.execution.get("duration_sec", 0.0)),
            "observations": bundle.observations,
            "executed_actions": bundle.executed_actions,
            "decisions": bundle.decisions,
            "alerts": bundle.alerts,
        }
        for role, agent in agents.items():
            if getattr(agent, "mode", "online") != "offline":
                continue
            try:
                for decision in agent.finalize(offline_context):
                    rendered = decision.as_dict()
                    bundle.decisions.append(rendered)
                    bundle.offline_analyses.append(rendered)
            except (
                Exception
            ) as exc:  # pragma: no cover - depends on local model environment
                bundle.status = "failed"
                bundle.passed = False
                bundle.errors.append(
                    {
                        "timestamp_utc": utc_now(),
                        "error": f"offline agent `{role}` failed: {exc}",
                    }
                )

        bundle.summary = {
            "run_id": run_id,
            "scenario": scenario.catalog_id,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "condition_name": condition_name,
            "status": bundle.status,
            "passed": bundle.passed,
            "run_dir": portable_path(run_dir),
            "errors": bundle.errors,
            "metrics": bundle.metrics,
        }
        resolved_bundle = self._bundle_dict(bundle)
        write_run_bundle(run_dir, resolved_bundle)
        if started:
            try:
                env.export_run_artifacts(run_dir, resolved_bundle)
            except Exception as exc:
                bundle.errors.append(
                    {
                        "timestamp_utc": utc_now(),
                        "error": f"environment artifact export failed: {exc}",
                    }
                )
                resolved_bundle = self._bundle_dict(bundle)
                write_run_bundle(run_dir, resolved_bundle)
            try:
                env.stop()
            except EnvironmentError as exc:
                bundle.errors.append(
                    {
                        "timestamp_utc": utc_now(),
                        "error": f"environment teardown failed: {exc}",
                    }
                )
                resolved_bundle = self._bundle_dict(bundle)
                write_run_bundle(run_dir, resolved_bundle)
        return resolved_bundle

    def _aggregate_attack_metrics(
        self,
        runs: list[dict[str, Any]],
        *,
        primary_timing_origin: str = "attack_start",
    ) -> dict[str, Any]:
        attack_runs = [run for run in runs if run.get("suite") == "attack"]
        completed = [run for run in attack_runs if run.get("status") == "completed"]
        per_run_metrics = [
            {
                "condition_name": run.get("condition_name"),
                "scenario": run.get("scenario"),
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "passed": run.get("passed"),
                "time_to_detection_sec": run.get("metrics", {}).get(
                    "time_to_detection_sec"
                ),
                "time_to_detection_from_attack_start_sec": run.get("metrics", {}).get(
                    "time_to_detection_sec"
                ),
                "time_to_detection_from_wargame_t0_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("time_to_first_detection_sec"),
                "time_to_recognition_sec": run.get("metrics", {}).get(
                    "time_to_recognition_sec"
                ),
                "time_to_recognition_from_wargame_t0_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("time_to_first_recognition_sec"),
                "observer_time_to_detection_sec": run.get("metrics", {})
                .get("observer_detection", {})
                .get("time_to_detection_sec"),
                "observer_time_to_recognition_sec": run.get("metrics", {})
                .get("observer_recognition", {})
                .get("time_to_recognition_sec"),
                "defender_time_to_detection_sec": run.get("metrics", {})
                .get("defender_detection", {})
                .get("time_to_detection_sec"),
                "defender_time_to_recognition_sec": run.get("metrics", {})
                .get("defender_recognition", {})
                .get("time_to_recognition_sec"),
                "time_to_mitigation_sec": run.get("metrics", {}).get(
                    "time_to_mitigation_sec"
                ),
                "time_to_mitigation_from_wargame_t0_sec": run.get("metrics", {}).get(
                    "time_to_mitigation_from_wargame_t0_sec"
                ),
                "attacker_action_count": run.get("metrics", {}).get(
                    "attacker_action_count"
                ),
                "attacker_recon_action_count": run.get("metrics", {}).get(
                    "attacker_recon_action_count"
                ),
                "attacker_time_to_impact_sec": run.get("metrics", {})
                .get("attacker", {})
                .get("time_to_impact_sec"),
                "time_to_physical_impact_sec": run.get("metrics", {})
                .get("attacker", {})
                .get("time_to_physical_impact_sec"),
                "time_to_physical_impact_from_attack_start_sec": run.get("metrics", {})
                .get("attacker", {})
                .get("time_to_physical_impact_sec"),
                "time_to_physical_impact_from_wargame_t0_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("time_to_first_safety_violation_sec"),
                "attacker_execution_success_rate": run.get("metrics", {})
                .get("attacker", {})
                .get("execution_success_rate"),
                "attacker_scenario_expectation_passed": run.get("metrics", {})
                .get("attacker", {})
                .get("scenario_expectation_passed"),
                "attacker_caused_safety_violation": run.get("metrics", {})
                .get("attacker", {})
                .get("caused_safety_violation"),
                "attacker_undetected_impact": run.get("metrics", {})
                .get("attacker", {})
                .get("undetected_impact"),
                "true_positive_rate": run.get("metrics", {}).get("true_positive_rate"),
                "observer_detection_rate": run.get("metrics", {})
                .get("observer_detection", {})
                .get("detection_rate"),
                "defender_detection_rate": run.get("metrics", {})
                .get("defender_detection", {})
                .get("detection_rate"),
                "observer_recognition_rate": run.get("metrics", {})
                .get("observer_recognition", {})
                .get("recognition_rate"),
                "defender_recognition_rate": run.get("metrics", {})
                .get("defender_recognition", {})
                .get("recognition_rate"),
                "false_positive_rate": run.get("metrics", {}).get(
                    "false_positive_rate"
                ),
                "true_positive_count": run.get("metrics", {}).get(
                    "true_positive_count"
                ),
                "observer_true_positive_count": run.get("metrics", {})
                .get("observer_detection", {})
                .get("true_positive_count"),
                "defender_true_positive_count": run.get("metrics", {})
                .get("defender_detection", {})
                .get("true_positive_count"),
                "false_positive_count": run.get("metrics", {}).get(
                    "false_positive_count"
                ),
                "blue_team_alert_count": run.get("metrics", {}).get(
                    "blue_team_alert_count"
                ),
                "planner_alert_count": run.get("metrics", {}).get(
                    "planner_alert_count"
                ),
                "decision_correctness": run.get("metrics", {}).get(
                    "decision_correctness"
                ),
                "defender_action_count": run.get("metrics", {}).get(
                    "defender_action_count"
                ),
                "defender_shutdown_action_count": run.get("metrics", {}).get(
                    "defender_shutdown_action_count"
                ),
                "defender_containment_action_count": run.get("metrics", {}).get(
                    "defender_containment_action_count"
                ),
                "defender_symbolic_containment_action_count": run.get("metrics", {}).get(
                    "defender_symbolic_containment_action_count"
                ),
                "first_defender_action_type": run.get("metrics", {}).get(
                    "first_defender_action_type"
                ),
                "first_defender_action_is_symbolic_containment": run.get("metrics", {}).get(
                    "first_defender_action_is_symbolic_containment"
                ),
                "defender_intervention_action_count": run.get("metrics", {})
                .get("operational_cost", {})
                .get("intervention_action_count"),
                "downtime_timestep_count": run.get("metrics", {}).get(
                    "downtime_timestep_count"
                ),
                "time_to_first_defender_action_sec": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("time_to_first_action_sec"),
                "time_to_first_defender_action_from_attack_start_sec": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("time_to_first_action_sec"),
                "time_to_first_defender_action_from_wargame_t0_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("time_to_first_defender_action_sec"),
                "time_to_first_containment_from_wargame_t0_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("time_to_first_containment_sec"),
                "mitigation_action_count": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("mitigation_action_count"),
                "containment_action_count": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("containment_action_count"),
                "symbolic_containment_action_count": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("symbolic_containment_action_count"),
                "contained_before_safety_violation": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("contained_before_safety_violation"),
                "attacker_execution_cutoff_activated": run.get("metrics", {})
                .get("runtime_containment", {})
                .get("attacker_execution_cutoff_activated"),
                "blocked_attacker_action_count": run.get("metrics", {})
                .get("runtime_containment", {})
                .get("blocked_attacker_action_count"),
                "attacker_actions_stopped_after_containment": run.get("metrics", {})
                .get("runtime_containment", {})
                .get("attacker_actions_stopped_after_containment"),
                "safety_violation_count": run.get("metrics", {}).get(
                    "safety_violation_count"
                ),
                "recovery_success_rate": run.get("metrics", {}).get(
                    "recovery_success_rate"
                ),
                "phase_a_duration_sec": run.get("metrics", {})
                .get("phase_window", {})
                .get("phase_a_duration_sec"),
                "wargame_t0_utc": run.get("metrics", {})
                .get("phase_window", {})
                .get("wargame_t0_utc"),
            }
            for run in attack_runs
        ]

        completed_metrics = [
            item for item in per_run_metrics if item.get("status") == "completed"
        ]

        def average(metric_name: str) -> float | None:
            values = [
                float(run[metric_name])
                for run in completed_metrics
                if run.get(metric_name) is not None
            ]
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        detected = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_detection_sec") is not None
        ]
        recognized = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_recognition_sec") is not None
        ]
        observer_detected = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("observer_detection", {})
            .get("time_to_detection_sec")
            is not None
        ]
        defender_detected = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("defender_detection", {})
            .get("time_to_detection_sec")
            is not None
        ]
        mitigated = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_mitigation_sec") is not None
        ]
        defender_acted = [
            run
            for run in completed
            if (run.get("metrics", {}).get("defender_action_count") or 0) > 0
        ]
        timely_contained = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("defender_intervention", {})
            .get("contained_before_safety_violation")
        ]
        first_symbolic_containment_runs = [
            run
            for run in completed
            if run.get("metrics", {}).get("first_defender_action_is_symbolic_containment")
        ]
        attacker_cutoff_runs = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("runtime_containment", {})
            .get("attacker_execution_cutoff_activated")
        ]
        attacker_stopped_runs = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("runtime_containment", {})
            .get("attacker_actions_stopped_after_containment")
        ]
        attacker_safety_success = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("attacker", {})
            .get("caused_safety_violation")
        ]
        undetected_impact_runs = [
            run
            for run in completed
            if run.get("metrics", {}).get("attacker", {}).get("undetected_impact")
        ]

        primary_origin = (
            "wargame_t0"
            if str(primary_timing_origin).strip().lower() == "wargame_t0"
            else "attack_start"
        )
        primary_detection_metric = (
            "time_to_detection_from_wargame_t0_sec"
            if primary_origin == "wargame_t0"
            else "time_to_detection_from_attack_start_sec"
        )
        primary_first_action_metric = (
            "time_to_first_defender_action_from_wargame_t0_sec"
            if primary_origin == "wargame_t0"
            else "time_to_first_defender_action_from_attack_start_sec"
        )
        primary_impact_metric = (
            "time_to_physical_impact_from_wargame_t0_sec"
            if primary_origin == "wargame_t0"
            else "time_to_physical_impact_from_attack_start_sec"
        )

        return {
            "primary_timing_origin": primary_origin,
            "attack_run_count": len(attack_runs),
            "completed_attack_run_count": len(completed),
            "failed_attack_run_count": len(attack_runs) - len(completed),
            "detected_attack_run_count": len(detected),
            "recognized_attack_run_count": len(recognized),
            "observer_detected_attack_run_count": len(observer_detected),
            "defender_detected_attack_run_count": len(defender_detected),
            "defender_acted_attack_run_count": len(defender_acted),
            "mitigated_attack_run_count": len(mitigated),
            "contained_before_safety_violation_run_count": len(timely_contained),
            "attacker_caused_safety_violation_run_count": len(attacker_safety_success),
            "undetected_impact_run_count": len(undetected_impact_runs),
            "time_to_detection_sec_avg": average(primary_detection_metric),
            "time_to_detection_from_attack_start_sec_avg": average(
                "time_to_detection_from_attack_start_sec"
            ),
            "time_to_detection_from_wargame_t0_sec_avg": average(
                "time_to_detection_from_wargame_t0_sec"
            ),
            "time_to_recognition_sec_avg": average("time_to_recognition_sec"),
            "time_to_recognition_from_wargame_t0_sec_avg": average(
                "time_to_recognition_from_wargame_t0_sec"
            ),
            "observer_time_to_detection_sec_avg": average(
                "observer_time_to_detection_sec"
            ),
            "observer_time_to_recognition_sec_avg": average(
                "observer_time_to_recognition_sec"
            ),
            "defender_time_to_detection_sec_avg": average(
                "defender_time_to_detection_sec"
            ),
            "defender_time_to_recognition_sec_avg": average(
                "defender_time_to_recognition_sec"
            ),
            "time_to_mitigation_sec_avg": average("time_to_mitigation_sec"),
            "time_to_mitigation_from_wargame_t0_sec_avg": average(
                "time_to_mitigation_from_wargame_t0_sec"
            ),
            "attacker_action_count_avg": average("attacker_action_count"),
            "attacker_recon_action_count_avg": average(
                "attacker_recon_action_count"
            ),
            "attacker_time_to_impact_sec_avg": average("attacker_time_to_impact_sec"),
            "time_to_physical_impact_sec_avg": average(primary_impact_metric),
            "time_to_physical_impact_from_attack_start_sec_avg": average(
                "time_to_physical_impact_from_attack_start_sec"
            ),
            "time_to_physical_impact_from_wargame_t0_sec_avg": average(
                "time_to_physical_impact_from_wargame_t0_sec"
            ),
            "attacker_execution_success_rate_avg": average(
                "attacker_execution_success_rate"
            ),
            "attacker_scenario_expectation_passed_avg": average(
                "attacker_scenario_expectation_passed"
            ),
            "attacker_caused_safety_violation_avg": average(
                "attacker_caused_safety_violation"
            ),
            "attacker_safety_success_rate": average(
                "attacker_caused_safety_violation"
            ),
            "undetected_impact_rate": average("attacker_undetected_impact"),
            "defender_action_count_avg": average("defender_action_count"),
            "defender_shutdown_action_count_avg": average(
                "defender_shutdown_action_count"
            ),
            "defender_containment_action_count_avg": average(
                "defender_containment_action_count"
            ),
            "defender_symbolic_containment_action_count_avg": average(
                "defender_symbolic_containment_action_count"
            ),
            "defender_intervention_action_count_avg": average(
                "defender_intervention_action_count"
            ),
            "downtime_timestep_count_avg": average("downtime_timestep_count"),
            "time_to_first_defender_action_sec_avg": average(
                primary_first_action_metric
            ),
            "time_to_first_defender_action_from_attack_start_sec_avg": average(
                "time_to_first_defender_action_from_attack_start_sec"
            ),
            "time_to_first_defender_action_from_wargame_t0_sec_avg": average(
                "time_to_first_defender_action_from_wargame_t0_sec"
            ),
            "time_to_first_containment_from_wargame_t0_sec_avg": average(
                "time_to_first_containment_from_wargame_t0_sec"
            ),
            "mitigation_action_count_avg": average("mitigation_action_count"),
            "containment_action_count_avg": average("containment_action_count"),
            "symbolic_containment_action_count_avg": average(
                "symbolic_containment_action_count"
            ),
            "first_defender_symbolic_containment_run_count": len(
                first_symbolic_containment_runs
            ),
            "first_defender_symbolic_containment_rate": (
                None
                if not completed
                else round(len(first_symbolic_containment_runs) / len(completed), 4)
            ),
            "attacker_execution_cutoff_run_count": len(attacker_cutoff_runs),
            "attacker_execution_cutoff_rate": (
                None
                if not completed
                else round(len(attacker_cutoff_runs) / len(completed), 4)
            ),
            "blocked_attacker_action_count_avg": average(
                "blocked_attacker_action_count"
            ),
            "attacker_actions_stopped_after_containment_run_count": len(
                attacker_stopped_runs
            ),
            "attacker_actions_stopped_after_containment_rate": (
                None
                if not completed
                else round(len(attacker_stopped_runs) / len(completed), 4)
            ),
            "false_positive_rate_avg": average("false_positive_rate"),
            "true_positive_rate_avg": average("true_positive_rate"),
            "observer_detection_rate_avg": average("observer_detection_rate"),
            "defender_detection_rate_avg": average("defender_detection_rate"),
            "observer_recognition_rate_avg": average("observer_recognition_rate"),
            "defender_recognition_rate_avg": average("defender_recognition_rate"),
            "blue_team_alert_count_avg": average("blue_team_alert_count"),
            "planner_alert_count_avg": average("planner_alert_count"),
            "safety_violation_count_avg": average("safety_violation_count"),
            "recovery_success_rate_avg": average("recovery_success_rate"),
            "decision_correctness_avg": average("decision_correctness"),
            "per_run_metrics": per_run_metrics,
        }

    def _aggregate_mitre_phase_metrics(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        attack_runs = [run for run in runs if run.get("suite") == "attack"]
        completed = [run for run in attack_runs if run.get("status") == "completed"]
        per_run_metrics = []
        distribution = {phase: 0 for phase in ATTACKER_PHASES}
        per_condition_distribution: dict[str, dict[str, int]] = {}

        def average(metric_name: str) -> float | None:
            values = []
            for run in per_run_metrics:
                value = run.get(metric_name)
                if value is None:
                    continue
                values.append(float(value))
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        for run in completed:
            mitre = dict(run.get("metrics", {}).get("mitre_phase") or {})
            item = {
                "condition_name": run.get("condition_name"),
                "scenario": run.get("scenario"),
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "highest_attacker_phase_reached": mitre.get(
                    "highest_attacker_phase_reached"
                ),
                "first_blue_team_detection_phase": mitre.get(
                    "first_blue_team_detection_phase"
                ),
                "first_defender_detection_phase": mitre.get(
                    "first_defender_detection_phase"
                ),
                "first_defender_action_phase": mitre.get(
                    "first_defender_action_phase"
                ),
                "blue_team_detection_before_manipulation": mitre.get(
                    "blue_team_detection_before_manipulation"
                ),
                "defender_detection_before_manipulation": mitre.get(
                    "defender_detection_before_manipulation"
                ),
                "first_defender_action_before_manipulation": mitre.get(
                    "first_defender_action_before_manipulation"
                ),
                "manipulation_to_blue_team_detection_delay_sec": mitre.get(
                    "manipulation_to_blue_team_detection_delay_sec"
                ),
                "manipulation_to_defender_detection_delay_sec": mitre.get(
                    "manipulation_to_defender_detection_delay_sec"
                ),
                "manipulation_to_first_defender_action_delay_sec": mitre.get(
                    "manipulation_to_first_defender_action_delay_sec"
                ),
            }
            per_run_metrics.append(item)

            highest_phase = str(item.get("highest_attacker_phase_reached") or "")
            if highest_phase in distribution:
                distribution[highest_phase] += 1
                condition_name = str(item.get("condition_name") or "")
                condition_bucket = per_condition_distribution.setdefault(
                    condition_name,
                    {phase: 0 for phase in ATTACKER_PHASES},
                )
                condition_bucket[highest_phase] += 1

        return {
            "attack_run_count": len(attack_runs),
            "completed_attack_run_count": len(completed),
            "phase_model_present_run_count": sum(
                1
                for run in completed
                if run.get("metrics", {}).get("mitre_phase", {}).get("phase_model")
            ),
            "highest_attacker_phase_distribution": distribution,
            "per_condition_highest_phase_distribution": per_condition_distribution,
            "blue_team_detection_before_manipulation_run_count": sum(
                1
                for item in per_run_metrics
                if item.get("blue_team_detection_before_manipulation") is True
            ),
            "defender_detection_before_manipulation_run_count": sum(
                1
                for item in per_run_metrics
                if item.get("defender_detection_before_manipulation") is True
            ),
            "first_defender_action_before_manipulation_run_count": sum(
                1
                for item in per_run_metrics
                if item.get("first_defender_action_before_manipulation") is True
            ),
            "blue_team_detection_before_manipulation_rate": average(
                "blue_team_detection_before_manipulation"
            ),
            "defender_detection_before_manipulation_rate": average(
                "defender_detection_before_manipulation"
            ),
            "first_defender_action_before_manipulation_rate": average(
                "first_defender_action_before_manipulation"
            ),
            "manipulation_to_blue_team_detection_delay_sec_avg": average(
                "manipulation_to_blue_team_detection_delay_sec"
            ),
            "manipulation_to_defender_detection_delay_sec_avg": average(
                "manipulation_to_defender_detection_delay_sec"
            ),
            "manipulation_to_first_defender_action_delay_sec_avg": average(
                "manipulation_to_first_defender_action_delay_sec"
            ),
            "per_run_metrics": per_run_metrics,
        }

    def _aggregate_nominal_metrics(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        nominal_runs = [run for run in runs if run.get("suite") == "normal"]
        completed = [run for run in nominal_runs if run.get("status") == "completed"]
        per_run_metrics = [
            {
                "condition_name": run.get("condition_name"),
                "scenario": run.get("scenario"),
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "passed": run.get("passed"),
                "blue_team_alert_count": run.get("metrics", {}).get(
                    "blue_team_alert_count"
                ),
                "defender_action_count": run.get("metrics", {}).get(
                    "defender_action_count"
                ),
                "safety_violation_count": run.get("metrics", {}).get(
                    "safety_violation_count"
                ),
                "false_positive_episode": run.get("metrics", {})
                .get("nominal", {})
                .get("false_positive_episode"),
                "false_intervention_episode": run.get("metrics", {})
                .get("nominal", {})
                .get("false_intervention_episode"),
                "time_to_first_false_alert_sec": run.get("metrics", {})
                .get("nominal", {})
                .get("time_to_first_false_alert_sec"),
                "time_to_first_false_intervention_sec": run.get("metrics", {})
                .get("nominal", {})
                .get("time_to_first_false_intervention_sec"),
            }
            for run in nominal_runs
        ]

        def average(metric_name: str) -> float | None:
            values = [
                float(run[metric_name])
                for run in per_run_metrics
                if run.get("status") == "completed" and run.get(metric_name) is not None
            ]
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        false_positive_runs = [
            run
            for run in completed
            if run.get("metrics", {}).get("nominal", {}).get("false_positive_episode")
        ]
        false_intervention_runs = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("nominal", {})
            .get("false_intervention_episode")
        ]
        safety_violation_runs = [
            run
            for run in completed
            if (run.get("metrics", {}).get("safety_violation_count") or 0) > 0
        ]
        passed_runs = [run for run in completed if run.get("passed")]

        return {
            "nominal_run_count": len(nominal_runs),
            "completed_nominal_run_count": len(completed),
            "failed_nominal_run_count": len(nominal_runs) - len(completed),
            "passed_nominal_run_count": len(passed_runs),
            "false_positive_nominal_run_count": len(false_positive_runs),
            "false_intervention_nominal_run_count": len(false_intervention_runs),
            "safety_violation_nominal_run_count": len(safety_violation_runs),
            "nominal_pass_rate": (
                None if not completed else round(len(passed_runs) / len(completed), 4)
            ),
            "false_positive_episode_rate": (
                None
                if not completed
                else round(len(false_positive_runs) / len(completed), 4)
            ),
            "false_intervention_episode_rate": (
                None
                if not completed
                else round(len(false_intervention_runs) / len(completed), 4)
            ),
            "blue_team_alert_count_avg": average("blue_team_alert_count"),
            "defender_action_count_avg": average("defender_action_count"),
            "safety_violation_count_avg": average("safety_violation_count"),
            "time_to_first_false_alert_sec_avg": average(
                "time_to_first_false_alert_sec"
            ),
            "time_to_first_false_intervention_sec_avg": average(
                "time_to_first_false_intervention_sec"
            ),
            "per_run_metrics": per_run_metrics,
        }

    def _aggregate_trajectory_metrics(
        self,
        runs: list[dict[str, Any]],
        *,
        condition: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for run in runs:
            metrics = run.get("metrics", {})
            if run.get("status") != "completed":
                continue
            if metrics.get("suite_type") != "attack" and run.get("suite") != "attack":
                continue
            key = (str(run.get("condition_name", "")), str(run.get("scenario", "")))
            grouped.setdefault(key, []).append(run)

        repeated_groups = {
            key: value for key, value in grouped.items() if len(value) > 1
        }
        configured_sources = self._trajectory_variation_sources(condition)
        if not repeated_groups:
            return {
                "status": "not_meaningful_yet",
                "reason": "Each scenario-condition pair was executed once; repeated same-scenario runs are required for honest diversity measurement.",
                "configured_variation_sources": configured_sources,
                "per_scenario": [],
            }

        per_scenario: list[dict[str, Any]] = []
        diversity_rates: list[float] = []
        pairwise_distances: list[float] = []
        unique_counts: list[float] = []
        for (condition_name, scenario_name), scenario_runs in sorted(repeated_groups.items()):
            trajectories = [
                tuple(
                    item
                    for item in run.get("metrics", {})
                    .get("trajectory", {})
                    .get("paired_sequence", [])
                )
                for run in scenario_runs
            ]
            unique_count = len(set(trajectories))
            diversity_rate = round(unique_count / len(trajectories), 4)
            avg_edit_distance = self._average_pairwise_edit_distance(trajectories)
            per_scenario.append(
                {
                    "condition_name": condition_name,
                    "scenario": scenario_name,
                    "run_count": len(trajectories),
                    "unique_trajectory_count": unique_count,
                    "trajectory_diversity_rate": diversity_rate,
                    "avg_pairwise_edit_distance": avg_edit_distance,
                    "trajectories": [list(item) for item in trajectories],
                }
            )
            diversity_rates.append(diversity_rate)
            unique_counts.append(float(unique_count))
            if avg_edit_distance is not None:
                pairwise_distances.append(avg_edit_distance)

        return {
            "status": "computed",
            "configured_variation_sources": configured_sources,
            "repeated_scenario_count": len(per_scenario),
            "average_unique_trajectory_count": round(
                sum(unique_counts) / len(unique_counts),
                4,
            ),
            "average_trajectory_diversity_rate": round(
                sum(diversity_rates) / len(diversity_rates),
                4,
            ),
            "average_pairwise_edit_distance": (
                None
                if not pairwise_distances
                else round(sum(pairwise_distances) / len(pairwise_distances), 4)
            ),
            "per_scenario": per_scenario,
        }

    def _trajectory_variation_sources(
        self,
        condition: dict[str, str] | None,
    ) -> list[str]:
        if not condition:
            return []
        sources = ["environment_nondeterminism"]
        sampled_roles = [
            role
            for role, profile_id in condition.items()
            if float(
                self.agent_profiles.get(profile_id, {})
                .get("llm", {})
                .get("temperature", 0.0)
                or 0.0
            )
            > 0.0
        ]
        if sampled_roles:
            sources.insert(0, "model_sampling")
        return sources

    @staticmethod
    def _average_pairwise_edit_distance(
        trajectories: list[tuple[str, ...]],
    ) -> float | None:
        if len(trajectories) < 2:
            return None
        distances: list[int] = []
        for index, left in enumerate(trajectories):
            for right in trajectories[index + 1 :]:
                distances.append(ExperimentOrchestrator._trajectory_edit_distance(left, right))
        if not distances:
            return None
        return round(sum(distances) / len(distances), 4)

    @staticmethod
    def _trajectory_edit_distance(
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> int:
        rows = len(left) + 1
        cols = len(right) + 1
        dp = [[0] * cols for _ in range(rows)]
        for row in range(rows):
            dp[row][0] = row
        for col in range(cols):
            dp[0][col] = col
        for row in range(1, rows):
            for col in range(1, cols):
                cost = 0 if left[row - 1] == right[col - 1] else 1
                dp[row][col] = min(
                    dp[row - 1][col] + 1,
                    dp[row][col - 1] + 1,
                    dp[row - 1][col - 1] + cost,
                )
        return dp[-1][-1]

    def _run_live_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        condition: dict[str, str],
        effective_profiles: dict[str, dict[str, Any]] | None,
        env: OpenPLCEnvironment,
    ) -> None:
        poll_interval = float(scenario.execution.get("poll_interval_ms", 500)) / 1000.0
        loop_sleep_sec = float(env.config.get("loop_sleep_sec", poll_interval))
        duration = float(scenario.execution.get("duration_sec", 60.0))
        planned_events = self._build_plan(scenario)
        runtime_containment = self._initial_runtime_containment()
        last_state: str | None = None
        elapsed = 0.0

        while elapsed <= duration:
            observation = env.observe_flat()
            last_state = self._record_observation(
                bundle,
                observation,
                elapsed=elapsed,
                phase="pre_action",
                last_state=last_state,
            )

            planning_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            self._execute_online_roles(
                roles=PLANNING_ROLES,
                agents=agents,
                context=planning_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                effective_profiles=effective_profiles,
                env=env,
                elapsed=elapsed,
                runtime_containment=runtime_containment,
            )

            action_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            action_taken = self._execute_online_roles(
                roles=ACTION_ROLES,
                agents=agents,
                context=action_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                effective_profiles=effective_profiles,
                env=env,
                elapsed=elapsed,
                runtime_containment=runtime_containment,
            )

            if action_taken:
                post_action_observation = env.observe_flat()
                last_state = self._record_observation(
                    bundle,
                    post_action_observation,
                    elapsed=elapsed,
                    phase="post_action",
                    last_state=last_state,
                )

            observer_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            self._execute_online_roles(
                roles=OBSERVER_ROLES,
                agents=agents,
                context=observer_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                effective_profiles=effective_profiles,
                env=env,
                elapsed=elapsed,
                runtime_containment=runtime_containment,
            )

            defender_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            defense_taken = self._execute_online_roles(
                roles=DEFENDER_ROLES,
                agents=agents,
                context=defender_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                effective_profiles=effective_profiles,
                env=env,
                elapsed=elapsed,
                runtime_containment=runtime_containment,
            )

            if defense_taken:
                post_defense_observation = env.observe_flat()
                last_state = self._record_observation(
                    bundle,
                    post_defense_observation,
                    elapsed=elapsed,
                    phase="post_defense",
                    last_state=last_state,
                )
            time.sleep(loop_sleep_sec)
            elapsed = round(elapsed + poll_interval, 3)

        expectation_events = [
            {
                "timestamp_utc": item["timestamp_utc"],
                "time_sec": item["time_sec"],
                "action": canonical_action_name(item["action"]),
                "state_after": item["state_after"],
            }
            for item in bundle.executed_actions
        ]
        if hasattr(env, "testbed") and hasattr(env.testbed, "evaluate_expectations"):
            passed, checks = env.testbed.evaluate_expectations(
                bundle.observations,
                expectation_events,
                scenario.execution.get("expectations"),
            )
            bundle.passed = passed
            bundle.expectation_checks = [check.__dict__ for check in checks]
        else:
            passed = scenario.execution.get("expectations", {}).get("final_state")
            bundle.passed = bool(bundle.observations) and (
                passed is None or bundle.observations[-1].get("STATE_NAME") == passed
            )
            bundle.expectation_checks = []

    def _record_observation(
        self,
        bundle: RunBundle,
        observation: dict[str, Any],
        *,
        elapsed: float,
        phase: str,
        last_state: str | None,
    ) -> str:
        sample = dict(observation)
        sample["elapsed_sec"] = round(elapsed, 3)
        sample["sample_phase"] = phase
        bundle.observations.append(sample)

        state_name = str(sample.get("STATE_NAME", "UNKNOWN"))
        if state_name != last_state:
            bundle.system_events.append(
                {
                    "timestamp_utc": utc_now(),
                    "time_sec": round(elapsed, 3),
                    "event_type": "state_transition",
                    "from_state": last_state,
                    "to_state": state_name,
                    "phase": phase,
                }
            )
        return state_name

    def _live_context(
        self,
        *,
        scenario: ScenarioDefinition,
        bundle: RunBundle,
        planned_events: dict[str, list[dict[str, Any]]],
        observation: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "scenario": scenario,
            "time_sec": elapsed,
            "observation": observation,
            "observations": bundle.observations,
            "alerts": bundle.alerts,
            "decisions": bundle.decisions,
            "executed_actions": bundle.executed_actions,
            "planned_events": planned_events,
            "playbooks": planned_events,
        }

    def _execute_online_roles(
        self,
        *,
        roles: list[str],
        agents: dict[str, Any],
        context: dict[str, Any],
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        condition: dict[str, str],
        env: OpenPLCEnvironment,
        elapsed: float,
        effective_profiles: dict[str, dict[str, Any]] | None = None,
        runtime_containment: dict[str, Any] | None = None,
    ) -> bool:
        runtime_containment = runtime_containment or self._initial_runtime_containment()
        executed_action = False
        for role in roles:
            agent = agents.get(role)
            if agent is None or getattr(agent, "mode", "online") != "online":
                continue
            for decision in agent.step(context):
                rendered = decision.as_dict()
                bundle.decisions.append(rendered)
                runtime_info = getattr(agent, "runtime_info", None)
                if isinstance(runtime_info, dict) and runtime_info:
                    env.record_role_runtime(role, runtime_info)
                if decision.decision_type == "alert":
                    bundle.alerts.append(rendered)
                action, error = self._sanitize_action(
                    role=role,
                    action=decision.action,
                    scenario=scenario,
                    profile_id=condition[role],
                    effective_profile=(effective_profiles or {}).get(role),
                )
                if error:
                    bundle.system_events.append(
                        {
                            "timestamp_utc": utc_now(),
                            "time_sec": round(elapsed, 3),
                            "event_type": "guardrail_rejection",
                            "role": role,
                            "detail": error,
                        }
                    )
                    continue
                if action is None:
                    continue

                if role == "attacker" and runtime_containment.get("active"):
                    self._record_blocked_attacker_action(
                        runtime_containment=runtime_containment,
                        bundle=bundle,
                        decision=decision,
                        action=action,
                        elapsed=elapsed,
                    )
                    continue

                action_executor = getattr(agent, "apply_action", None)
                if role == "defender" and is_symbolic_containment_action(action):
                    state_after = self._symbolic_action_state_after(
                        action,
                        context.get("observation", {}),
                    )
                elif callable(action_executor):
                    state_after = action_executor(action)
                else:
                    state_after = env.act(action)
                runtime_info = getattr(agent, "runtime_info", runtime_info)
                if isinstance(runtime_info, dict) and runtime_info:
                    env.record_role_runtime(role, runtime_info)
                executed_action = True
                record = {
                    "timestamp_utc": utc_now(),
                    "time_sec": round(elapsed, 3),
                    "role": role,
                    "agent_name": decision.agent_name,
                    "action": action,
                    "state_after": state_after.get("state_name")
                    or state_after.get("STATE_NAME"),
                    "labels": decision.labels,
                    "diagnosis": decision.diagnosis,
                    "metadata": dict(decision.metadata or {}),
                }
                if isinstance(runtime_info, dict) and runtime_info:
                    record["execution_residency"] = runtime_info.get(
                        "execution_residency"
                    )
                    record["runtime_host"] = runtime_info.get("hostname")
                    record["assigned_node"] = runtime_info.get("assigned_node")
                recon_result = state_after.get("reconnaissance")
                if isinstance(recon_result, dict):
                    record["result"] = recon_result
                containment = containment_metadata(action)
                if role == "defender" and containment.get("is_containment_action"):
                    record["containment"] = containment
                    self._record_containment_activation(
                        runtime_containment=runtime_containment,
                        bundle=bundle,
                        action=action,
                        role=role,
                        elapsed=elapsed,
                    )
                bundle.executed_actions.append(record)
        return executed_action

    def _resolve_replay_source_path(
        self, replay_source: str | Path, *, config_path: Path
    ) -> Path:
        path = Path(replay_source)
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()

        for base_dir in (config_path.parent, results_root(), REPO_ROOT):
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                return candidate
        return (config_path.parent / path).resolve()

    def _run_replay_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        replay_source: Path,
    ) -> None:
        source = replay_source.resolve()
        observations_path = source / "observations.csv"
        metrics_path = source / "metrics.json"
        if not observations_path.exists():
            raise ConfigError(f"replay source missing observations.csv: {source}")
        import csv
        import json

        with observations_path.open() as handle:
            reader = csv.DictReader(handle)
            bundle.observations = [dict(row) for row in reader]
        for name, target in (
            ("actions.json", "executed_actions"),
            ("decisions.json", "decisions"),
            ("alerts.json", "alerts"),
            ("checks.json", "expectation_checks"),
            ("system_events.json", "system_events"),
            ("offline_analysis.json", "offline_analyses"),
        ):
            path = source / name
            if not path.exists():
                continue
            setattr(bundle, target, json.loads(path.read_text()))
        if metrics_path.exists():
            bundle.metrics = json.loads(metrics_path.read_text())
        summary_path = source / "summary.json"
        if summary_path.exists():
            existing_summary = json.loads(summary_path.read_text())
            bundle.passed = existing_summary.get("passed", True)
        if bundle.passed is None:
            bundle.passed = True

    def _bundle_dict(self, bundle: RunBundle) -> dict[str, Any]:
        return {
            "experiment_id": bundle.experiment_id,
            "condition_name": bundle.condition_name,
            "run_id": bundle.run_id,
            "scenario": {
                "catalog_id": bundle.scenario.catalog_id,
                "scenario_id": bundle.scenario.scenario_id,
                "suite": bundle.scenario.suite,
                "execution_path": portable_path(bundle.scenario.execution_path),
                "execution": bundle.scenario.execution,
                "metadata": bundle.scenario.metadata,
            },
            "run_dir": portable_path(bundle.run_dir),
            "config_snapshot": bundle.config_snapshot,
            "observations": bundle.observations,
            "system_events": bundle.system_events,
            "executed_actions": bundle.executed_actions,
            "decisions": bundle.decisions,
            "alerts": bundle.alerts,
            "offline_analyses": bundle.offline_analyses,
            "expectation_checks": bundle.expectation_checks,
            "errors": bundle.errors,
            "status": bundle.status,
            "passed": bundle.passed,
            "summary": bundle.summary,
            "metrics": bundle.metrics,
        }
