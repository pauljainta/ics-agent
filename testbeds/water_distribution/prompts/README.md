# testbeds/water_distribution/prompts

Distribution-specific prompt templates.

## Files

- [`black_box_attacker.md`](black_box_attacker.md): limited-context attacker prompt.
- [`gray_box_attacker.md`](gray_box_attacker.md): partial-context attacker prompt.
- [`white_box_attacker.md`](white_box_attacker.md): complete-context attacker prompt.
- [`white_box_attacker_ablation.md`](white_box_attacker_ablation.md): prompt used when withholding selected white-box context components.
- [`defender.md`](defender.md): agentic defender prompt.

Prompt files are referenced by profile YAML in [`../profiles.yaml`](../profiles.yaml).
