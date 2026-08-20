# Scripted White-Box Baseline Report

- Report date: `2026-08-20`
- Testbed: `water_treatment`
- Attacker: `scripted_white_box_treatment_attacker` deterministic scripted white-box baseline
- Defender: `agentic_defender` current full containment defender
- Seed: `101` only
- Results root: `Artifact/Results/Water-treatment/scripted_baselines/white_box_attacker_vs_full_containment_defender`
- Seed root: `Artifact/Results/Water-treatment/scripted_baselines/white_box_attacker_vs_full_containment_defender/seed_101`
- This pass is a deterministic attacker-baseline comparison and does not replace the LLM white-box attacker results.

## Status
- Controller validation started on `2026-08-20`.
- Scenario execution status: `0/18` completed.
- Preflight status: `failed before scenario execution`.
- Failure point: live SPHERE/XDC attach for `example-realization-03.openplcwater.lgarcia`.
- Resume state: the next run resumes from `wt-attack-hmi-chatter` after treatment preflight is healthy again.

## Notes
- The scripted attacker profile resolves distinctly from the LLM white-box attacker while preserving the same bounded white-box treatment interface.
- The treatment result-root `_progress` ledger captured the preflight failure in `global_phase_status.json`, `optionb_environment_validation.json`, and `experiment_run.log`.
- No per-scenario metrics were fabricated because the run stopped before scenario execution.
