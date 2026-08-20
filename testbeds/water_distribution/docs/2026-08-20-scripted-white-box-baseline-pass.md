# Scripted White-Box Baseline Report

- Report date: `2026-08-20`
- Testbed: `water_distribution`
- Attacker: `scripted_white_box_distribution_attacker` deterministic scripted white-box baseline
- Defender: `agentic_defender` current full containment defender
- Seed: `101` only
- Results root: `Artifact/Results/water_distribution_openplc/scripted_baselines/white_box_attacker_vs_full_containment_defender`
- This pass is a deterministic attacker-baseline comparison and does not replace the LLM white-box attacker results.

## Status
- Scenario execution status: `0/11` completed.
- Run state: `not started`.
- Blocking order constraint: treatment remains the first unfinished testbed because its preflight failed on `2026-08-20`, so distribution was intentionally not launched.
- Resume state: once treatment completes or becomes runnable again, distribution should start from its first pending scenario under the result root above.

## Notes
- The scripted attacker profile resolves distinctly from the LLM white-box attacker while preserving the current bounded white-box distribution interface.
- No distribution result artifacts were created because the required execution order forbids starting distribution while treatment is still unfinished.
