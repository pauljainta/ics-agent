from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from ics_agent.experiments import model_variant_runner as runner
from ics_agent.utils import load_yaml


def _write_summary(path: Path, *, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status}) + "\n")


def test_model_variant_resume_cleanup_removes_failed_gray_box_trial_only(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "WORKSPACE_ROOT", tmp_path)
    model = runner.ModelVariant(
        key="phi4_mini_instruct",
        label="Phi-4-mini-instruct",
        model_id="microsoft/Phi-4-mini-instruct",
        local_dir=Path("models/local_models/phi-4-mini-instruct"),
        quantization="bf16",
    )
    testbed = runner.TestbedVariant(
        key="water_distribution",
        testbed_id="water_distribution",
        result_experiment_name="water_distribution_openplc",
        expected_scenarios=11,
        family_configs=(),
    )
    root = (
        tmp_path
        / "Artifact"
        / "Results"
        / model.label
        / testbed.result_experiment_name
    )
    completed_trial = (
        root
        / "gray_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-completed"
        / "trial-001"
    )
    failed_trial = (
        root
        / "gray_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-grid-low-spoof"
        / "trial-001"
    )
    other_family_trial = (
        root
        / "black_box_attacker__agentic_defender"
        / "attack"
        / "wd-attack-completed"
        / "trial-001"
    )

    _write_summary(completed_trial / "summary.json", status="completed")
    _write_summary(failed_trial / "summary.json", status="failed")
    _write_summary(other_family_trial / "summary.json", status="completed")

    removed = runner._purge_failed_or_incomplete_runs(
        model=model,
        testbed=testbed,
        family_names={"gray_box_attacker__agentic_defender"},
    )

    assert completed_trial.exists()
    assert other_family_trial.exists()
    assert not failed_trial.exists()
    assert [item["family"] for item in removed] == [
        "gray_box_attacker__agentic_defender"
    ]


def test_model_variant_smoke_respects_single_selected_family(monkeypatch) -> None:
    model = runner.MODEL_VARIANTS["phi4_mini_instruct"]
    testbed = runner.TESTBED_VARIANTS["water_treatment"]
    selected_family = "black_box_high_cyber_high_domain__agentic_defender"

    generated_paths: list[str] = []

    def fake_prepare_configs(*, model, testbed):
        return [
            (
                family,
                runner.REPO_ROOT
                / "generated"
                / f"{family.family_name}.yaml",
            )
            for family in testbed.family_configs
        ]

    def fake_validate_generated_config(*, model, testbed, family, config_path, require_model):
        generated_paths.append(family.family_name)
        return {
            "family": family.family_name,
            "config_path": runner.portable_path(config_path),
            "scenario_count": testbed.expected_scenarios,
            "result_root": runner.portable_path(
                runner._result_model_root(model) / testbed.result_experiment_name
            ),
        }

    monkeypatch.setattr(runner, "_prepare_configs", fake_prepare_configs)
    monkeypatch.setattr(
        runner,
        "_validate_generated_config",
        fake_validate_generated_config,
    )

    payload = runner.smoke(
        model=model,
        testbed=testbed,
        require_model=True,
        family_names={selected_family},
    )

    assert payload["status"] == "passed"
    assert payload["family_count"] == 1
    assert payload["expected_runs"] == 18
    assert payload["selected_families"] == [selected_family]
    assert generated_paths == [selected_family]


def test_phi4_distribution_smoke_selects_black_and_gray_only(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_model_dir_status",
        lambda model: {
            "model_id": model.model_id,
            "local_dir": runner.portable_path(model.local_dir),
            "exists": True,
            "has_config": True,
            "has_tokenizer": True,
            "has_weight_index": True,
            "has_safetensors": True,
            "sample_files": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_dependency_status",
        lambda model: {
            "python": "python",
            "requires_bitsandbytes": model.requires_bitsandbytes,
            "bitsandbytes_available": False,
            "hf_cli_available": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_validate_model_backend_imports",
        lambda model: {
            "model_type": "phi3",
            "architectures": ["Phi3ForCausalLM"],
            "tokenizer_class": "GPT2TokenizerFast",
            "model_class": "Phi3ForCausalLM",
            "trust_remote_code": model.trust_remote_code,
        },
    )

    selected = {
        "black_box_attacker__agentic_defender",
        "gray_box_attacker__agentic_defender",
    }
    payload = runner.smoke(
        model=runner.MODEL_VARIANTS["phi4_mini_instruct"],
        testbed=runner.TESTBED_VARIANTS["water_distribution"],
        require_model=False,
        family_names=selected,
    )

    assert payload["status"] == "passed"
    assert payload["testbed"] == "water_distribution"
    assert payload["family_count"] == 2
    assert payload["expected_runs"] == 22
    assert set(payload["selected_families"]) == selected
    assert "white_box_attacker__agentic_defender" not in payload["selected_families"]
    assert (
        "black_box_high_cyber_high_domain__agentic_defender"
        not in payload["selected_families"]
    )
    assert {item["family"] for item in payload["checks"]} == selected
    assert all(item["scenario_count"] == 11 for item in payload["checks"])
    assert all(
        item["result_root"].endswith(
            "Artifact/Results/Phi-4-mini-instruct/water_distribution_openplc"
        )
        for item in payload["checks"]
    )


def test_distribution_white_box_model_variant_matches_seeded_backbone_guardrail() -> None:
    family = next(
        item
        for item in runner.TESTBED_VARIANTS["water_distribution"].family_configs
        if item.family_name == "white_box_attacker__agentic_defender"
    )
    raw_config = load_yaml(family.config_path)

    generated = runner._apply_model_override(
        raw_config=raw_config,
        family=family,
        testbed=runner.TESTBED_VARIANTS["water_distribution"],
        model=runner.MODEL_VARIANTS["phi4_mini_instruct"],
    )

    attacker = generated["conditions"][0]["profile_overrides"]["attacker"]
    assert attacker["white_box_commit_guardrail"] is False
    assert attacker["behavior_model"]["white_box_commit_guardrail"] is False
    assert attacker["llm"]["model_id"] == "microsoft/Phi-4-mini-instruct"


def test_water_treatment_model_variants_match_seeded_backbone_actions() -> None:
    testbed = runner.TESTBED_VARIANTS["water_treatment"]
    for family in testbed.family_configs:
        raw_config = load_yaml(family.config_path)
        generated = runner._apply_model_override(
            raw_config=raw_config,
            family=family,
            testbed=testbed,
            model=runner.MODEL_VARIANTS["qwen3_14b"],
        )
        overrides = generated["conditions"][0]["profile_overrides"]
        attacker = overrides["attacker"]
        defender = overrides["defender"]

        assert (
            attacker["allowed_action_types"]
            == runner.TREATMENT_HISTORICAL_ATTACKER_ACTIONS
        )
        assert (
            defender["allowed_action_types"]
            == runner.TREATMENT_HISTORICAL_DEFENDER_ACTIONS
        )
        if family.family_name == "white_box_attacker__agentic_defender":
            assert attacker["white_box_commit_guardrail"] is False
            assert attacker["behavior_model"]["white_box_commit_guardrail"] is False
        if family.family_name == "black_box_high_cyber_high_domain__agentic_defender":
            assert attacker["knowledge_files"] == [
                "configs/knowledge/phase35/black_box_constraints_manual.md",
                "configs/knowledge/phase35/high_cyber_ics_methodology_manual.md",
                "testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md",
            ]


def test_water_distribution_target_blind_matches_seeded_knowledge_files() -> None:
    testbed = runner.TESTBED_VARIANTS["water_distribution"]
    family = next(
        item
        for item in testbed.family_configs
        if item.family_name == "black_box_high_cyber_high_domain__agentic_defender"
    )
    raw_config = load_yaml(family.config_path)

    generated = runner._apply_model_override(
        raw_config=raw_config,
        family=family,
        testbed=testbed,
        model=runner.MODEL_VARIANTS["phi4_mini_instruct"],
    )

    attacker = generated["conditions"][0]["profile_overrides"]["attacker"]
    assert attacker["knowledge_files"] == [
        "configs/knowledge/phase35/black_box_constraints_manual.md",
        "configs/knowledge/phase35/high_cyber_ics_methodology_manual.md",
        "testbeds/water_distribution/knowledge/phase35/water_distribution_domain_primer.md",
    ]
    assert attacker["allowed_action_types"] == runner.DISTRIBUTION_HISTORICAL_ATTACKER_ACTIONS


def test_qwen3_14b_and_phi_configs_differ_only_by_model_fields() -> None:
    def normalized_profile_overrides(config):
        overrides = deepcopy(config["conditions"][0]["profile_overrides"])
        for role_override in overrides.values():
            llm = role_override.get("llm")
            if not isinstance(llm, dict):
                continue
            for key in (
                "model_id",
                "local_files_only",
                "quantization",
                "trust_remote_code",
            ):
                llm.pop(key, None)
        return overrides

    for testbed_key in ("water_treatment", "water_distribution"):
        testbed = runner.TESTBED_VARIANTS[testbed_key]
        for family in testbed.family_configs:
            raw_config = load_yaml(family.config_path)

            qwen3_config = runner._apply_model_override(
                raw_config=raw_config,
                family=family,
                testbed=testbed,
                model=runner.MODEL_VARIANTS["qwen3_14b"],
            )
            phi_config = runner._apply_model_override(
                raw_config=raw_config,
                family=family,
                testbed=testbed,
                model=runner.MODEL_VARIANTS["phi4_mini_instruct"],
            )

            assert normalized_profile_overrides(
                qwen3_config
            ) == normalized_profile_overrides(phi_config)
            assert (
                qwen3_config["conditions"][0]["profile_overrides"]["attacker"][
                    "llm"
                ]["model_id"]
                == "Qwen/Qwen3-14B"
            )
            assert (
                phi_config["conditions"][0]["profile_overrides"]["attacker"][
                    "llm"
                ]["model_id"]
                == "microsoft/Phi-4-mini-instruct"
            )


def test_model_variant_runner_rejects_historical_qwen35_anchor() -> None:
    try:
        runner._model_variant("qwen35_9b")
    except Exception as exc:
        assert "unknown model variant" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("historical Qwen3.5 anchor should not be runnable here")
