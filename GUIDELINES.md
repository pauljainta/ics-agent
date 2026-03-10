# Project guidelines (quick reference)

## Goals
- Build modular agents (`AssetMapping`, `ScenarioGeneration`, `DeploymentNodes`).
- Use LangChain for orchestration; use Qwen (local) as the LLM.
- Support fidelity ladder F0-F4 safely; start READ-ONLY.
- Provide reproducible experiments and structured reasoning artifacts for trust research.

## Folder responsibilities
- `agents/*`: agent implementations and their `spec.md`.
- `tools/*`: adapters (e.g., PLC, historian), retrievers, parsers.
- `models/*`: local model integration and quick demos.
- `configs/*`: configuration for different environments (sim, HIL, partner).
- `scripts/*`: developer helper scripts (setup, run).
- `docs/*`: architecture, experiment plans, meeting notes.
- `tests/*`: unit and integration tests.

## Branching & workflow
- Use feature branches: `feat/<short-description>`.
- PR template: short description, linked issue, test plan, security considerations.
- Require 1 reviewer and passing CI for merging to `main`.

## Naming & style rules
- Python: PEP8. Use `black` for formatting and `ruff` for linting.
- Docstrings: Google style.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).

## Issue & PR templates
- Issue template includes: summary, reproducible steps, expected behaviour, priority, related docs.
- PRs include: summary, test plan, backwards-incompatibility notes.

## Security & data
- Never commit credentials. Use `.env` and `.env.example`.
- Default mode for deployment nodes: `READ_ONLY=true`.
- Before enabling actuation: require manual approval and sign-off from PI.

## Testing & CI
- Write unit tests for parsing, schema validation, and model smoke tests.
- CI matrix: Python 3.10/3.11; run lint, unit tests, qwen inference smoke (mock or small model).
- Use GitHub Actions jobs: `lint`, `tests`, `smoke_model_test`.

## Minimal CI example (GitHub Actions)
- `ci.yml`:
  - jobs: lint -> tests -> smoke
  - run `python -m pytest -q`

## Onboarding checklist (for new dev)
1. Clone repo and install packages.
2. Run `python models/qwen_test.py` (note: may require CHPC).
3. Read `agents/*/spec.md`.
4. Run unit tests.
5. Join Slack group and link PRs there.
