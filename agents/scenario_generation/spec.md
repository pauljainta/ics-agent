# ScenarioGeneration Agent — Spec

## Name
ScenarioGenerationAgent

## Purpose / role
Generate realistic ICS cyber-physical scenario injects for tabletop and simulation campaigns.

## Inputs
- Existing site asset inventory/KG
- Threat model library (MITRE ATT&CK for ICS, custom playbooks)
- Fidelity target (F0-F4)
- Constraints (duration, safety gates, allowed impacts)

## Outputs
- Scenario inject package (JSON/Markdown)
- Prerequisite checklist
- Expected observations and scoring rubric

## Tools required
- KG retriever
- Threat technique retriever
- Scenario template renderer
- Validation rules engine

## Actions
- `select_objective()`
- `choose_attack_path()`
- `generate_inject()`
- `validate_constraints()`
- `propose_human_review()`

## Constraints & safety
- No direct actuation commands in generated injects.
- Must include rollback/safe-stop notes for every scenario.
- Ambiguous effects require manual review.

## Autonomy level
- Start at L1 (assistive). Human approves publication of injects.

## Evaluation metrics
- Scenario realism score
- Coverage across tactics/techniques
- Reviewer acceptance rate

## Example API
- `POST /scenario` with context -> returns scenario id
- `GET /scenario/{id}` -> returns scenario package
