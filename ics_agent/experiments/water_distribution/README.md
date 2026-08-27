# ics_agent/experiments/water_distribution

Water-distribution experiment controllers.

## Files

- [`seeded_backbone.py`](seeded_backbone.py): seeded attacker-knowledge comparison for distribution scenarios.
- [`domain_ablation.py`](domain_ablation.py): target-blind domain/cyber profile comparison.
- [`white_box_ablation.py`](white_box_ablation.py): white-box context-component ablation.
- [`defender_authority_ablation.py`](defender_authority_ablation.py): observe-only, delayed-containment, and full-containment defender comparisons.
- [`scripted_black_box_baseline.py`](scripted_black_box_baseline.py): deterministic black-box scripted attacker baseline.
- [`scripted_gray_box_baseline.py`](scripted_gray_box_baseline.py): deterministic gray-box scripted attacker baseline.
- [`scripted_white_box_baseline.py`](scripted_white_box_baseline.py): deterministic white-box scripted attacker baseline.
- [`__init__.py`](__init__.py): package marker.

## Related Entry Points

Use wrappers in [`../../../scripts/`](../../../scripts/README.md), especially:

- [`run_water_distribution_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py`](../../../scripts/run_water_distribution_agentic_black_gray_white_target_blind_attackers_vs_agentic_full_containment_defender_seeded.py)
- [`run_water_distribution_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py`](../../../scripts/run_water_distribution_agentic_white_box_context_ablation_attackers_vs_agentic_full_containment_defender.py)
- [`run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py`](../../../scripts/run_water_distribution_agentic_attackers_vs_agentic_observe_only_and_delayed_containment_defenders.py)
