# ICS-Agents — Prototype Platform

**Purpose:** a modular agent framework for cyber-physical wargaming and ICS resilience research.
This repository contains three primary agent classes: Asset Mapping, Scenario Generation, and Deployment Nodes. The system uses LangChain for orchestration and Qwen (local model) for reasoning.

## Quick start (developer)
1. Clone repo:
   ```bash
   git clone git@github.com:<org>/ics-agents.git
   cd ics-agents
   ```
2. Create venv and install:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Copy env template:
   ```bash
   cp .env.example .env
   # edit .env for local paths if needed
   ```
4. Run minimal Qwen test (local model required):
   ```bash
   python models/qwen_test.py
   ```
5. Run LangChain + Qwen demo:
   ```bash
   python models/langchain_qwen_demo.py
   ```

## Repo structure
See [GUIDELINES.md](./GUIDELINES.md) for full details.

## Contributing
- Branches: `feat/*`, `fix/*`, `chore/*`, `exp/*`.
- PRs require one approving review.
- All code must include tests for new functionality.

## Safety
- All agents default to READ-ONLY mode unless an explicit ACTUATE gate is enabled.
- Logs and full reasoning artifacts are recorded for every suggested action.
