# 2026-08-11 SWaT-Informed Domain-Ablation Pass

- Date: `2026-08-11`
- Testbed: `water_treatment` only
- Attacker profile: `black_box_high_cyber_high_domain_treatment_attacker`
- Defender profile: `agentic_defender`
- Seed set: `101` only
- Experiment version: `water_treatment_domain_ablation_black_box_high_cyber_high_domain_swat_enriched_seed101_v2`
- Result root: `Artifact/Results/Water-treatment/domain-ablation/black_box_high_cyber_high_domain__agentic_defender`

## Condition Interpretation

- This pass intentionally strengthened the treatment domain primer with sanitized SWaT-derived attack and process knowledge.
- The runtime attacker interface remained black-box throughout the run.
- `knowledge_mode` stayed `black_box`.
- `system_knowledge` stayed `None`.
- The treatment black-box observation and action interface was not widened.

## Prompt And Knowledge Pack

- Attacker prompt: `configs/prompts/black_box_attacker_optionb_mitre.md`
- Knowledge files:
  - `configs/knowledge/phase35/black_box_constraints_manual.md`
  - `configs/knowledge/phase35/high_cyber_ics_methodology_manual.md`
  - `testbeds/water_treatment/knowledge/phase35/water_treatment_domain_primer.md`
- Prompt and knowledge audit manifest:
  - `Artifact/Results/Water-treatment/domain-ablation/black_box_high_cyber_high_domain__agentic_defender/prompt_knowledge_audit/prompt_knowledge_manifest.json`
- Prompt and knowledge bundle hash: `8258130465c52dea686d2c837c84916a50d10c80d4d6d7b72f38ac94a8a4b34b`
- Prompt and knowledge audit findings: `0`

## Run Outcome

- Completed scenarios: `18/18`
- Failed scenarios: `0`
- Detected scenarios: `18`
- Defender-action scenarios: `18`
- Safety-violation scenarios: `0`
- Average wall-clock detection time from attack start: `13.5147`
- Average attacker action count: `2.7778`
- Average downtime burden: `32.8333`

## Notes

- The seeded run completed cleanly with scenario-granular resume ledgers under the condition root.
- Water-distribution was not touched.
- `ics-agent-paper` was not touched.
- Result artifacts and audit packages were kept outside git.
