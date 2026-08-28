from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ConfigError, load_config
from ..environment import OpenPLCEnvironment
from ..orchestrator import ExperimentOrchestrator
from ..paths import REPO_ROOT, WORKSPACE_ROOT, portable_path
from ..reporting import write_json
from ..utils import dump_yaml, utc_now


SEED = 101


@dataclass(frozen=True)
class ModelVariant:
    key: str
    label: str
    model_id: str
    local_dir: Path
    quantization: str
    trust_remote_code: bool = False
    max_new_tokens_override: int | None = None
    requires_bitsandbytes: bool = False


@dataclass(frozen=True)
class FamilyConfig:
    family_name: str
    config_path: Path


@dataclass(frozen=True)
class TestbedVariant:
    key: str
    testbed_id: str
    result_experiment_name: str
    expected_scenarios: int
    family_configs: tuple[FamilyConfig, ...]


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "qwen3_14b": ModelVariant(
        key="qwen3_14b",
        label="Qwen3-14B",
        model_id="Qwen/Qwen3-14B",
        local_dir=REPO_ROOT / "models" / "local_models" / "qwen3-14b",
        quantization="4bit",
        requires_bitsandbytes=True,
    ),
    "phi4_mini_instruct": ModelVariant(
        key="phi4_mini_instruct",
        label="Phi-4-mini-instruct",
        model_id="microsoft/Phi-4-mini-instruct",
        local_dir=REPO_ROOT / "models" / "local_models" / "phi-4-mini-instruct",
        quantization="bf16",
    ),
}

MODEL_ALIASES = {
    "qwen3-14b": "qwen3_14b",
    "qwen_3_14b": "qwen3_14b",
    "qwen": "qwen3_14b",
    "phi": "phi4_mini_instruct",
    "phi4": "phi4_mini_instruct",
    "phi-4-mini": "phi4_mini_instruct",
    "phi-4-mini-instruct": "phi4_mini_instruct",
}

TESTBED_VARIANTS: dict[str, TestbedVariant] = {
    "water_treatment": TestbedVariant(
        key="water_treatment",
        testbed_id="water_treatment",
        result_experiment_name="water_treatment",
        expected_scenarios=18,
        family_configs=(
            FamilyConfig(
                "black_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_treatment_agentic_black_box_attacker_vs_agentic_full_containment_defender.yaml",
            ),
            FamilyConfig(
                "gray_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_treatment_agentic_gray_box_attacker_vs_agentic_full_containment_defender.yaml",
            ),
            FamilyConfig(
                "white_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_treatment_agentic_white_box_attacker_vs_agentic_full_containment_defender.yaml",
            ),
            FamilyConfig(
                "black_box_high_cyber_high_domain__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_treatment_agentic_black_box_high_cyber_high_domain_attacker_vs_agentic_full_containment_defender.yaml",
            ),
        ),
    ),
    "water_distribution": TestbedVariant(
        key="water_distribution",
        testbed_id="water_distribution",
        result_experiment_name="water_distribution_openplc",
        expected_scenarios=11,
        family_configs=(
            FamilyConfig(
                "black_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_distribution_black_box_vs_agentic_defender.yaml",
            ),
            FamilyConfig(
                "gray_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_distribution_gray_box_vs_agentic_defender.yaml",
            ),
            FamilyConfig(
                "white_box_attacker__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_distribution_white_box_vs_agentic_defender.yaml",
            ),
            FamilyConfig(
                "black_box_high_cyber_high_domain__agentic_defender",
                REPO_ROOT
                / "configs"
                / "experiments"
                / "water_distribution_domain_ablation_black_box_high_cyber_high_domain_vs_agentic_defender.yaml",
            ),
        ),
    ),
}


def _normalize_model_key(value: str) -> str:
    cleaned = value.strip()
    lowered = cleaned.lower()
    return MODEL_ALIASES.get(lowered, lowered)


def _model_variant(value: str) -> ModelVariant:
    key = _normalize_model_key(value)
    try:
        return MODEL_VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(sorted([*MODEL_VARIANTS, *MODEL_ALIASES]))
        raise ConfigError(f"unknown model variant `{value}`; choose one of: {known}") from exc


def _testbed_variant(value: str) -> TestbedVariant:
    key = value.strip().lower().replace("-", "_")
    try:
        return TESTBED_VARIANTS[key]
    except KeyError as exc:
        known = ", ".join(sorted(TESTBED_VARIANTS))
        raise ConfigError(f"unknown testbed `{value}`; choose one of: {known}") from exc


def _deep_merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _model_env(model: ModelVariant) -> dict[str, str]:
    local_dir = str(model.local_dir.resolve())
    return {
        "ICS_AGENT_MODEL_ID": model.model_id,
        "ICS_AGENT_MODEL_PATH": local_dir,
        "QWEN_MODEL_ID": model.model_id,
        "QWEN_MODEL_PATH": local_dir,
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
    }


def _llm_override(model: ModelVariant) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": model.model_id,
        "local_files_only": True,
        "quantization": model.quantization,
        "trust_remote_code": model.trust_remote_code,
    }
    if model.max_new_tokens_override is not None:
        payload["max_new_tokens"] = model.max_new_tokens_override
    return payload


TREATMENT_HISTORICAL_ATTACKER_ACTIONS = [
    "hmi_start",
    "hmi_stop",
    "hmi_estop",
    "inspect_surface",
    "select_process",
    "set_manual_valve",
    "set_sensor",
    "noop",
]

DISTRIBUTION_HISTORICAL_ATTACKER_ACTIONS = [
    "inspect_surface",
    "hmi_start",
    "hmi_stop",
    "hmi_estop",
    "set_sensor",
    "set_actuator",
    "noop",
]

TREATMENT_HISTORICAL_DEFENDER_ACTIONS = [
    "hmi_stop",
    "hmi_estop",
    "select_process",
    "disable_hmi_session",
    "isolate_engineering_host",
    "block_sensor_write",
    "block_process2_manual_path",
    "quarantine_signal_source",
    "noop",
]

DISTRIBUTION_HISTORICAL_DEFENDER_ACTIONS = [
    "hmi_stop",
    "hmi_estop",
    "disable_hmi_session",
    "isolate_engineering_host",
    "block_sensor_write",
    "block_actuator_write",
    "quarantine_signal_source",
    "noop",
]

TARGET_BLIND_PHASE35_SHARED_KNOWLEDGE = [
    "configs/knowledge/phase35/black_box_constraints_manual.md",
    "configs/knowledge/phase35/high_cyber_ics_methodology_manual.md",
]


def _disable_white_box_commit_guardrail(attacker: dict[str, Any]) -> dict[str, Any]:
    compatible = dict(attacker)
    behavior_model = dict(compatible.get("behavior_model") or {})
    compatible["white_box_commit_guardrail"] = False
    behavior_model["white_box_commit_guardrail"] = False
    compatible["behavior_model"] = behavior_model
    return compatible


def _target_blind_knowledge_files(testbed: TestbedVariant) -> list[str]:
    if testbed.key == "water_treatment":
        return [
            *TARGET_BLIND_PHASE35_SHARED_KNOWLEDGE,
            "testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md",
        ]
    if testbed.key == "water_distribution":
        return [
            *TARGET_BLIND_PHASE35_SHARED_KNOWLEDGE,
            "testbeds/water_distribution/knowledge/phase35/water_distribution_domain_primer.md",
        ]
    return list(TARGET_BLIND_PHASE35_SHARED_KNOWLEDGE)


def _apply_historical_qwen35_profile_compatibility(
    *,
    config: dict[str, Any],
    family: FamilyConfig,
    testbed: TestbedVariant,
) -> dict[str, Any]:
    """Keep model-swap profiles capability-equivalent to the Qwen3.5 anchor."""
    if testbed.key not in {"water_treatment", "water_distribution"}:
        return config

    compatible = deepcopy(config)
    condition = compatible["conditions"][0]
    profile_overrides = dict(condition.get("profile_overrides") or {})
    attacker = dict(profile_overrides.get("attacker") or {})
    defender = dict(profile_overrides.get("defender") or {})

    if testbed.key == "water_treatment":
        attacker["allowed_action_types"] = list(TREATMENT_HISTORICAL_ATTACKER_ACTIONS)
        defender["allowed_action_types"] = list(TREATMENT_HISTORICAL_DEFENDER_ACTIONS)
    elif testbed.key == "water_distribution":
        attacker["allowed_action_types"] = list(DISTRIBUTION_HISTORICAL_ATTACKER_ACTIONS)
        defender["allowed_action_types"] = list(DISTRIBUTION_HISTORICAL_DEFENDER_ACTIONS)

    if family.family_name == "white_box_attacker__agentic_defender":
        # The historical Qwen3.5 seeded backbone did not use this strengthened
        # direct-commit wrapper. Disable it so Qwen3/Phi swaps keep the same
        # attacker capability boundary as the seeded anchor.
        attacker = _disable_white_box_commit_guardrail(attacker)

    if family.family_name == "black_box_high_cyber_high_domain__agentic_defender":
        attacker["knowledge_files"] = _target_blind_knowledge_files(testbed)

    profile_overrides["attacker"] = attacker
    profile_overrides["defender"] = defender
    condition["profile_overrides"] = profile_overrides
    compatible["conditions"] = [condition]
    return compatible


def _apply_model_override(
    *,
    raw_config: dict[str, Any],
    family: FamilyConfig,
    testbed: TestbedVariant,
    model: ModelVariant,
) -> dict[str, Any]:
    config = deepcopy(raw_config)
    config["experiment_name"] = testbed.result_experiment_name
    config["experiment_id"] = f"{testbed.key}_{model.key}_seed_{SEED}"
    config["testbed"] = testbed.testbed_id
    config["repeats"] = 1
    config["resume"] = True
    config["use_canonical_result_root"] = True
    config["results_root"] = "../.."
    config.setdefault("reproducibility", {})
    config["reproducibility"]["seed"] = SEED
    config.setdefault("study_metadata", {})
    config["study_metadata"] = _deep_merge_mapping(
        dict(config["study_metadata"]),
        {
            "model_variant": model.label,
            "model_id": model.model_id,
            "model_local_dir": portable_path(model.local_dir),
            "model_quantization": model.quantization,
            "trust_remote_code": model.trust_remote_code,
            "seed": SEED,
            "scope": "model-only robustness runner",
        },
    )
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        raise ConfigError(f"{family.config_path} must define exactly one condition")
    condition = deepcopy(conditions[0])
    if condition.get("name") != family.family_name:
        raise ConfigError(
            f"expected condition `{family.family_name}` in {family.config_path}, "
            f"found `{condition.get('name')}`"
        )
    profile_overrides = dict(condition.get("profile_overrides") or {})
    for role in ("attacker", "defender", "observer"):
        role_override = dict(profile_overrides.get(role) or {})
        role_override["llm"] = _deep_merge_mapping(
            dict(role_override.get("llm") or {}),
            {
                **_llm_override(model),
                "fatal_on_parse_error": False,
            },
        )
        profile_overrides[role] = role_override
    condition["profile_overrides"] = profile_overrides
    config["conditions"] = [condition]
    return _apply_historical_qwen35_profile_compatibility(
        config=config,
        family=family,
        testbed=testbed,
    )


def _result_model_root(model: ModelVariant) -> Path:
    return WORKSPACE_ROOT / "Artifact" / "Results" / model.label


def _generated_config_path(model: ModelVariant, testbed: TestbedVariant, family: FamilyConfig) -> Path:
    return (
        _result_model_root(model)
        / "_generated_configs"
        / testbed.key
        / f"{family.family_name}.yaml"
    )


def _write_generated_config(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
    family: FamilyConfig,
    config: dict[str, Any],
) -> Path:
    path = _generated_config_path(model, testbed, family)
    serializable = deepcopy(config)
    serializable.pop("_path", None)
    dump_yaml(path, serializable)
    return path


def _model_dir_status(model: ModelVariant) -> dict[str, Any]:
    files = []
    if model.local_dir.exists():
        files = sorted(path.name for path in model.local_dir.iterdir())[:12]
    return {
        "model_id": model.model_id,
        "local_dir": portable_path(model.local_dir),
        "exists": model.local_dir.exists(),
        "has_config": (model.local_dir / "config.json").exists(),
        "has_tokenizer": any(
            (model.local_dir / name).exists()
            for name in ("tokenizer.json", "tokenizer.model")
        ),
        "has_weight_index": (model.local_dir / "model.safetensors.index.json").exists(),
        "has_safetensors": bool(list(model.local_dir.glob("*.safetensors")))
        if model.local_dir.exists()
        else False,
        "sample_files": files,
    }


def _dependency_status(model: ModelVariant) -> dict[str, Any]:
    status = {
        "python": sys.executable,
        "requires_bitsandbytes": model.requires_bitsandbytes,
        "bitsandbytes_available": False,
        "hf_cli_available": bool(shutil.which("hf") or shutil.which("huggingface-cli")),
    }
    try:
        import transformers  # type: ignore

        status["transformers_version"] = getattr(transformers, "__version__", "unknown")
        status["transformers_file"] = getattr(transformers, "__file__", "")
    except Exception as exc:  # pragma: no cover - environment-dependent
        status["transformers_error"] = str(exc)
    if model.requires_bitsandbytes:
        try:
            import bitsandbytes  # type: ignore  # noqa: F401

            status["bitsandbytes_available"] = True
        except Exception as exc:  # pragma: no cover - environment-dependent
            status["bitsandbytes_error"] = str(exc)
    return status


def _validate_model_backend_imports(model: ModelVariant) -> dict[str, Any]:
    """Catch local model/backend incompatibilities before scenario execution."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ConfigError(f"transformers backend import failed: {exc}") from exc

    try:
        config = AutoConfig.from_pretrained(
            model.local_dir,
            trust_remote_code=model.trust_remote_code,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model.local_dir,
            trust_remote_code=model.trust_remote_code,
            local_files_only=True,
        )
        # Resolve the model class without loading checkpoint weights. This catches
        # stale cached remote modules and incompatible transformers releases early.
        model_cls = AutoModelForCausalLM._model_mapping[type(config)]  # noqa: SLF001
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ConfigError(
            f"{model.label} local backend is not loadable with the current Python "
            f"environment: {exc}"
        ) from exc

    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": getattr(model_cls, "__name__", str(model_cls)),
        "trust_remote_code": model.trust_remote_code,
    }


def _validate_generated_config(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
    family: FamilyConfig,
    config_path: Path,
    require_model: bool,
) -> dict[str, Any]:
    env_before = {key: os.environ.get(key) for key in _model_env(model)}
    os.environ.update(_model_env(model))
    try:
        orchestrator = ExperimentOrchestrator(testbed_id=testbed.testbed_id)
        config = load_config(config_path)
        scenarios = orchestrator.catalog.resolve_selectors(config.get("scenario_selectors", []))
        if len(scenarios) != testbed.expected_scenarios:
            raise ConfigError(
                f"{testbed.key} expected {testbed.expected_scenarios} scenarios, "
                f"found {len(scenarios)}"
            )
        condition_specs = orchestrator._enumerate_conditions(config)
        if len(condition_specs) != 1:
            raise ConfigError(f"{config_path} produced {len(condition_specs)} conditions")
        condition_spec = condition_specs[0]
        condition_name = orchestrator._condition_name(
            condition_spec["profiles"],
            override=condition_spec.get("name"),
        )
        if condition_name != family.family_name:
            raise ConfigError(
                f"expected condition `{family.family_name}`, found `{condition_name}`"
            )
        settings = orchestrator._collect_decoding_settings(
            condition_spec["profiles"],
            effective_profiles=condition_spec.get("effective_profiles"),
        )
        for role, llm_settings in settings.items():
            if llm_settings.get("model_id") != model.model_id:
                raise ConfigError(
                    f"{family.family_name}/{role} model_id was not overridden: "
                    f"{llm_settings.get('model_id')}"
                )
            if llm_settings.get("quantization") != model.quantization:
                raise ConfigError(
                    f"{family.family_name}/{role} quantization was not overridden: "
                    f"{llm_settings.get('quantization')}"
                )
        model_status = _model_dir_status(model)
        if require_model and not (
            model_status["has_config"]
            and model_status["has_tokenizer"]
            and model_status["has_safetensors"]
        ):
            raise ConfigError(
                f"local model files are incomplete for {model.label}; run "
                f"`python scripts/download_local_model.py --model {model.key}` first"
            )
        deps = _dependency_status(model)
        if model.requires_bitsandbytes and not deps.get("bitsandbytes_available"):
            raise ConfigError(
                f"{model.label} is configured for 4-bit loading, but bitsandbytes is unavailable"
            )
        backend_import_status = _validate_model_backend_imports(model)
        return {
            "family": family.family_name,
            "config_path": portable_path(config_path),
            "scenario_count": len(scenarios),
            "condition_name": condition_name,
            "profiles": condition_spec["profiles"],
            "llm_roles": sorted(settings),
            "model_status": model_status,
            "dependency_status": deps,
            "backend_import_status": backend_import_status,
            "result_root": portable_path(_result_model_root(model) / testbed.result_experiment_name),
        }
    finally:
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prepare_configs(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
) -> list[tuple[FamilyConfig, Path]]:
    generated: list[tuple[FamilyConfig, Path]] = []
    for family in testbed.family_configs:
        raw_config = load_config(family.config_path)
        config = _apply_model_override(
            raw_config=raw_config,
            family=family,
            testbed=testbed,
            model=model,
        )
        generated.append(
            (
                family,
                _write_generated_config(
                    model=model,
                    testbed=testbed,
                    family=family,
                    config=config,
                ),
            )
        )
    return generated


def _sphere_merge_server(config_path: Path) -> str | None:
    config = load_config(config_path)
    environment = dict(config.get("environment") or {})
    if str(environment.get("backend", "")).strip() != "sphere_openplc":
        return None
    return str(environment.get("merge_server", "")).strip() or None


def _require_mrg_auth(config_paths: list[Path]) -> dict[str, Any]:
    merge_servers = {
        server
        for config_path in config_paths
        if (server := _sphere_merge_server(config_path))
    }
    if not merge_servers:
        return {"required": False, "status": "skipped"}
    if len(merge_servers) != 1:
        raise ConfigError(
            "model-variant run selected multiple Merge servers; split the run first"
        )
    mrg_path = shutil.which("mrg")
    if mrg_path is None:
        raise ConfigError("SPHERE backend requires `mrg`, but it was not found on PATH")
    merge_server = next(iter(merge_servers))
    command = [mrg_path, "-s", merge_server, "whoami"]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ConfigError(
            "SPHERE/MRG authentication is not active. Run `mrg login` first, "
            "then rerun the same model-variant command. "
            f"Preflight failed with: {stderr}"
        )
    return {
        "required": True,
        "status": "passed",
        "merge_server": merge_server,
        "identity": completed.stdout.strip(),
    }


def _cleanup_sphere_local_transports(config_paths: list[Path]) -> list[dict[str, Any]]:
    cleanup: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        environment = dict(config.get("environment") or {})
        if str(environment.get("backend", "")).strip() != "sphere_openplc":
            continue
        environment["keep_running"] = True
        environment["stop_remote_runtime"] = False
        try:
            env = OpenPLCEnvironment(environment)
            env.stop()
            cleanup.append(
                {
                    "config_path": portable_path(config_path),
                    "status": "cleaned",
                    "controller_port": environment.get("controller_port"),
                    "simulator_port": environment.get("simulator_port"),
                }
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            cleanup.append(
                {
                    "config_path": portable_path(config_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return cleanup


def _purge_failed_or_incomplete_runs(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
    family_names: set[str] | None,
) -> list[dict[str, Any]]:
    experiment_root = _result_model_root(model) / testbed.result_experiment_name
    selected_families = family_names or {
        family.family_name for family in testbed.family_configs
    }
    removed: list[dict[str, Any]] = []
    for family_name in sorted(selected_families):
        family_root = experiment_root / family_name
        if not family_root.exists():
            continue
        for trial_dir in sorted(family_root.glob("*/*/trial-*")):
            summary_path = trial_dir / "summary.json"
            keep = False
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    keep = isinstance(summary, dict) and summary.get("status") == "completed"
                except Exception:
                    keep = False
            if keep:
                continue
            shutil.rmtree(trial_dir)
            removed.append(
                {
                    "family": family_name,
                    "run_dir": portable_path(trial_dir),
                    "reason": "removed failed or incomplete trial before resume",
                }
            )
    return removed


def smoke(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
    require_model: bool,
    family_names: set[str] | None = None,
) -> dict[str, Any]:
    generated = _prepare_configs(model=model, testbed=testbed)
    selected = [
        (family, path)
        for family, path in generated
        if family_names is None or family.family_name in family_names
    ]
    if not selected:
        raise ConfigError("no family configs selected")
    checks = [
        _validate_generated_config(
            model=model,
            testbed=testbed,
            family=family,
            config_path=config_path,
            require_model=require_model,
        )
        for family, config_path in selected
    ]
    payload = {
        "status": "passed",
        "updated_utc": utc_now(),
        "mode": "smoke",
        "model": model.label,
        "model_id": model.model_id,
        "testbed": testbed.key,
        "seed": SEED,
        "family_count": len(checks),
        "expected_runs": len(checks) * testbed.expected_scenarios,
        "selected_families": [family.family_name for family, _ in selected],
        "checks": checks,
    }
    smoke_path = (
        _result_model_root(model)
        / testbed.result_experiment_name
        / "_progress"
        / "model_variant_smoke.json"
    )
    write_json(smoke_path, payload)
    payload["smoke_path"] = portable_path(smoke_path)
    return payload


def run_experiments(
    *,
    model: ModelVariant,
    testbed: TestbedVariant,
    family_names: set[str] | None = None,
    skip_environment_preflight: bool = False,
) -> dict[str, Any]:
    generated = _prepare_configs(model=model, testbed=testbed)
    selected = [
        (family, path)
        for family, path in generated
        if family_names is None or family.family_name in family_names
    ]
    if not selected:
        raise ConfigError("no family configs selected")

    model_status = _model_dir_status(model)
    if not (
        model_status["has_config"]
        and model_status["has_tokenizer"]
        and model_status["has_safetensors"]
    ):
        raise ConfigError(
            f"local model files are incomplete for {model.label}; run "
            f"`python scripts/download_local_model.py --model {model.key}` first"
        )
    backend_import_status = _validate_model_backend_imports(model)
    local_transport_cleanup = _cleanup_sphere_local_transports(
        [path for _, path in selected]
    )
    preflight = (
        {"required": False, "status": "skipped_by_user"}
        if skip_environment_preflight
        else _require_mrg_auth([path for _, path in selected])
    )
    removed_failed_runs = _purge_failed_or_incomplete_runs(
        model=model,
        testbed=testbed,
        family_names={family.family_name for family, _ in selected},
    )

    env_before = {key: os.environ.get(key) for key in _model_env(model)}
    os.environ.update(_model_env(model))
    try:
        suite_runs = []
        for family, config_path in selected:
            orchestrator = ExperimentOrchestrator(testbed_id=testbed.testbed_id)
            started = utc_now()
            summary = orchestrator.run(config_path)
            suite_runs.append(
                {
                    "family": family.family_name,
                    "config_path": portable_path(config_path),
                    "started_utc": started,
                    "completed_utc": utc_now(),
                    "status": summary.get("status"),
                    "results_root": summary.get("results_root"),
                }
            )
            if summary.get("status") != "completed":
                break
        payload = {
            "status": (
                "completed"
                if all(item.get("status") == "completed" for item in suite_runs)
                else "failed"
            ),
            "updated_utc": utc_now(),
            "mode": "run",
            "model": model.label,
            "model_id": model.model_id,
            "testbed": testbed.key,
            "seed": SEED,
            "model_status": model_status,
            "backend_import_status": backend_import_status,
            "local_transport_cleanup": local_transport_cleanup,
            "environment_preflight": preflight,
            "removed_failed_or_incomplete_runs": removed_failed_runs,
            "runs": suite_runs,
        }
        run_path = (
            _result_model_root(model)
            / testbed.result_experiment_name
            / "_progress"
            / "model_variant_run.json"
        )
        write_json(run_path, payload)
        payload["run_path"] = portable_path(run_path)
        return payload
    finally:
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-only robustness variants for current attacker/defender profiles."
    )
    parser.add_argument("--model", required=True, help="Model variant key or alias.")
    parser.add_argument(
        "--testbed",
        required=True,
        choices=sorted(TESTBED_VARIANTS),
        help="Testbed to run.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Generate configs and validate profiles/scenario counts without running experiments.",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="Let smoke validation pass even when local model weights are not downloaded yet.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Optional family name to run. Repeat to select multiple families.",
    )
    parser.add_argument(
        "--skip-environment-preflight",
        action="store_true",
        help="Skip the SPHERE/MRG auth preflight before full runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        model = _model_variant(args.model)
        testbed = _testbed_variant(args.testbed)
        if args.smoke_only:
            payload = smoke(
                model=model,
                testbed=testbed,
                require_model=not bool(args.allow_missing_model),
                family_names=set(args.family) if args.family else None,
            )
        else:
            payload = run_experiments(
                model=model,
                testbed=testbed,
                family_names=set(args.family) if args.family else None,
                skip_environment_preflight=bool(args.skip_environment_preflight),
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload.get("status") in {"passed", "completed"} else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def main_for(default_args: list[str]) -> int:
    return main([*default_args, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
