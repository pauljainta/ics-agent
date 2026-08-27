# configs/knowledge/shared

Shared manuals in this folder are reusable across testbeds.

## Files

- [`black_box_constraints_manual.md`](black_box_constraints_manual.md): constraints that keep black-box attackers from receiving target-specific semantics.
- [`high_cyber_ics_methodology_manual.md`](high_cyber_ics_methodology_manual.md): general ICS adversarial methodology for target-blind cyber skill.

These files are loaded through profile `knowledge_files` entries in [`../..//profiles/shared.yaml`](../../profiles/shared.yaml) and testbed-specific profile registries.

## Design Rule

Shared knowledge can describe general cyber or ICS concepts, but it should not reveal a specific testbed's tags, safety thresholds, authored scenario answers, private infrastructure, or hidden simulator state.
