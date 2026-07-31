# Water-Distribution White-Box Alignment Pass

Date: `2026-07-31`

## Scope

This pass only changed the water-distribution white-box attacker path under `ics-agent`. The treatment white-box setup was left unchanged, the paper repo was not touched, and no secrets were committed.

## Treatment White-Box Knobs Standardized Into Distribution

Relative to the current treatment white-box attacker, the distribution white-box attacker is now aligned on these attacker-side knobs:

- `behavior_model.id: optionb_mitre_staged`
- `behavior_model.policy: agentic`
- `llm.backend: qwen_langgraph`
- `llm.use_langgraph: true`
- `llm.strict_backend: true`
- `llm.allow_langchain_fallback: true`
- `llm.max_new_tokens: 160`
- `action_cooldown_sec: 0.5`
- `note_cooldown_sec: 2.0`
- `reasoning_cooldown_sec: 1.0`
- `reasoning_refresh_sec: 2.0`
- `memory.attacker_candidate_window: 8`
- white-box seed discipline: `reproducibility.seed: 20260623`
- white-box knowledge policy expanded to include the distribution-grid and return-water process-model manuals in addition to the shared distribution white-box materials
- prompt wording updated to explicitly require the treatment-style staged, MITRE-inspired attack structure

## Code / Config / Prompt Changes

- Updated [water_distribution_white_box_vs_agentic_defender.yaml](/home/researcher/Agentic_ICS/ics-agent/configs/experiments/water_distribution_white_box_vs_agentic_defender.yaml) to align the single-family distribution white-box attacker with the treatment white-box runtime, seed, staged behavior model, and knowledge set.
- Updated [water_distribution_attack_families.yaml](/home/researcher/Agentic_ICS/ics-agent/configs/experiments/water_distribution_attack_families.yaml) so the distribution `white_box_attacker__agentic_defender` family uses the same aligned white-box attacker knobs as the single-family rerun config.
- Updated [white_box_attacker.md](/home/researcher/Agentic_ICS/ics-agent/testbeds/water_distribution/prompts/white_box_attacker.md) to mirror the treatment-side staged prompt intent while staying distribution-specific and bounded.
- Updated [test_water_distribution.py](/home/researcher/Agentic_ICS/ics-agent/tests/test_water_distribution.py) with focused regression coverage for the aligned distribution white-box config, seed, knowledge files, and staged prompt wording.

## Scenario Boundaries

No scenario catalog entries were changed and no authored scenario-level attacker constraints were removed. The only relaxation in this pass was implementation-level narrowing: the distribution white-box runtime, staged behavior-model wiring, prompt framing, and white-box knowledge bundle were aligned upward to the treatment white-box standard.

## Validation

Targeted validation passed before the rerun:

- `PYTHONPATH=. ./.venv/bin/pytest -q tests/test_water_distribution.py`

## Result Cleanup

Deleted only:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`

Preserved:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/black_box_attacker__agentic_defender`
- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/gray_box_attacker__agentic_defender`

## Rerun

Ran only the 12-scenario water-distribution `white_box_attacker__agentic_defender` family on the required deployed backend path:

- `backend: sphere_openplc`
- `project/experiment: openplcwater.lgarcia`
- `realization: example-realization-02`

New results were written to:

- `/home/researcher/Agentic_ICS/Artifact/Results/water_distribution_openplc/white_box_attacker__agentic_defender`

## Immediate Comparison Against The Prior Distribution White-Box Baseline

Comparison baseline: the immediately prior `water_distribution_openplc/white_box_attacker__agentic_defender` family present before deletion on `2026-07-31`.

| Metric | Prior distribution white-box | Refreshed distribution white-box |
| --- | ---: | ---: |
| Attack runs | 12 | 12 |
| Attacker safety-success rate | 0.0 | 0.0 |
| Attacker action count avg | 9.4167 | 20.5833 |
| Attacker recon action count avg | 0.6667 | 1.0 |
| Attacker execution success rate avg | 0.0 | 0.0 |
| Blocked attacker action count avg | 27.5 | 19.0833 |
| Time to detection from attack start avg (sec) | 1.1667 | 0.0 |
| Time to first defender action from attack start avg (sec) | 1.1667 | 0.0 |
| Defender action count avg | 8.1667 | 9.4167 |
| Contained-before-violation run count | 12 | 12 |
| Attacker cutoff activation rate | 1.0 | 1.0 |
| Phase-model-present run count | 0 | 12 |

The refreshed run now records MITRE-style attacker phase outputs in all `12/12` runs, with `11/12` runs ending at `manipulation` and `1/12` reaching `impact`.

White-box safety-success did not increase relative to the immediately prior distribution white-box baseline. It stayed at `0.0` (`0/12` successful safety violations).

## Secrets

No secrets or credentials were committed in this pass.
