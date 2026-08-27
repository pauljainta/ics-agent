# ics_agent/agents

This package implements actor policies and action-space construction.

## Files

- [`base.py`](base.py): common `RoleAgent` base class and decision helper.
- [`action_space.py`](action_space.py): admissible attacker and defender candidate-action builders.
- [`agentic.py`](agentic.py): LLM-backed attacker, defender, observer, and offline-analysis roles, including black-box, gray-box, white-box, target-blind, ablation, and defender-response behavior.
- [`scripted.py`](scripted.py): deterministic scripted policies for attacker and supporting roles.
- [`rule_based.py`](rule_based.py): deterministic rule-based attacker/defender policies.
- [`__init__.py`](__init__.py): package exports.

## Policy Types

- Agentic policies reason over bounded observations, profile context, memory, and candidate actions.
- Scripted policies follow deterministic scenario/action recipes.
- Rule-based policies observe state and choose from fixed conditions and response rules.

The orchestrator validates chosen actions against action semantics before applying them to the runtime.
