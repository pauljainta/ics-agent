# Scripted Gray-Box Baseline Report

- Generated UTC: `2026-08-20T18:51:18.204353+00:00`
- Testbed: `water_treatment` only
- Attacker: `scripted_gray_box_treatment_attacker` deterministic scripted gray-box baseline
- Defender: `agentic_defender` current full containment defender
- Seed: `101` only
- Results root: `Artifact/Results/Water-treatment/scripted_baselines/gray_box_attacker_vs_full_containment_defender`
- Seed root: `Artifact/Results/Water-treatment/scripted_baselines/gray_box_attacker_vs_full_containment_defender/seed_101`
- This pass is a deterministic attacker-baseline comparison and does not replace the LLM attacker results.

## Outcome
- Completed scenarios: `18/18`
- Detected scenarios: `18/18`
- Defender-action scenarios: `18/18`
- Safety-violation scenarios: `0/18`
- Runtime containment cutoff activations: `18`
- Average wall-clock detection from attack start: `10.09 ± 4.21`
- Average attacker action count: `2.78 ± 1.00`
- Average downtime burden: `32.11 ± 46.48`

## Notes
- The attacker used deterministic fixed logic over the bounded gray-box runtime interface.
- The blue-team roles remained the current agentic observer and full containment defender.
- Resume ledgers were written under the result-root `_progress` directory and preserved per-scenario completion state.

## Repo Note
- This repo-local note summarizes the local scripted-baseline pass and references the local result root above.
- Raw experiment artifacts remain under `Artifact/Results` and are not intended for git commit.
