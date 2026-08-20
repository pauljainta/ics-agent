# Scripted Black-Box Baseline Report

- Generated UTC: `2026-08-20T05:14:20.156387+00:00`
- Testbed: `water_distribution` only
- Attacker: `scripted_black_box_distribution_attacker` deterministic scripted black-box baseline
- Defender: `agentic_defender` current full containment defender
- Seed: `101` only
- Results root: `Artifact/Results/water_distribution_openplc/scripted_baselines/black_box_attacker_vs_full_containment_defender`
- Seed root: `Artifact/Results/water_distribution_openplc/scripted_baselines/black_box_attacker_vs_full_containment_defender/seed_101`
- This pass is a deterministic attacker-baseline comparison and does not replace the LLM attacker results.

## Outcome
- Completed scenarios: `11/11`
- Detected scenarios: `11/11`
- Defender-action scenarios: `11/11`
- Safety-violation scenarios: `0/11`
- Runtime containment cutoff activations: `11`
- Average wall-clock detection from attack start: `5.32 ± 1.72`
- Average attacker action count: `7.36 ± 0.50`
- Average downtime burden: `110.91 ± 14.48`

## Notes
- The attacker used deterministic fixed logic over the black-box-visible runtime interface.
- The blue-team roles remained the current scripted observer and full containment defender.
- Resume ledgers were written under the result-root `_progress` directory and preserved per-scenario completion state.

## Repo Note
- This repo-local note summarizes the local scripted-baseline pass and references the local result root above.
- Raw experiment artifacts remain under `Artifact/Results` and are not intended for git commit.
