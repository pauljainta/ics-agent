# Architecture Overview

The system is organized into three primary agent classes:
- Asset Mapping Agent
- Scenario Generation Agent
- Deployment Node Agent

LangChain orchestrates tool usage and reasoning. Qwen is used locally for model inference.
All deployment-related behavior defaults to read-only execution.
