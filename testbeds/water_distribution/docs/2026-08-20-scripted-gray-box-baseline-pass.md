# Scripted Gray-Box Baseline Report

- Generated UTC: `2026-08-20T20:03:34.441933+00:00`
- Testbed: `water_distribution` only
- Attacker: `scripted_gray_box_distribution_attacker` deterministic scripted gray-box baseline
- Defender: `agentic_defender` current full containment defender
- Seed: `101` only
- Results root: `Artifact/Results/water_distribution_openplc/scripted_baselines/gray_box_attacker_vs_full_containment_defender`
- Seed root: `Artifact/Results/water_distribution_openplc/scripted_baselines/gray_box_attacker_vs_full_containment_defender/seed_101`
- This pass is a deterministic attacker-baseline comparison and does not replace the LLM attacker results.

## Outcome
- Completed scenarios: `11/11`
- Detected scenarios: `11/11`
- Defender-action scenarios: `11/11`
- Safety-violation scenarios: `0/11`
- Runtime containment cutoff activations: `11`
- Average wall-clock detection from attack start: `5.05 ± 1.77`
- Average attacker action count: `17.09 ± 4.81`
- Average downtime burden: `31.55 ± 25.93`

## Notes
- The attacker used deterministic fixed logic over the bounded gray-box runtime interface.
- The blue-team roles remained the current scripted observer and full containment defender.
- Resume ledgers were written under the result-root `_progress` directory and preserved per-scenario completion state.

## Repo Note
- This repo-local note summarizes the local scripted-baseline pass and references the local result root above.
- Raw experiment artifacts remain under `Artifact/Results` and are not intended for git commit.
