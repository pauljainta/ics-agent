# models/local_models

This folder is intentionally empty in Git except for documentation and `.gitkeep`.

## Purpose

Store local model checkpoints here, for example:

```text
models/local_models/qwen3-14b/
models/local_models/phi-4-mini-instruct/
```

Weights and tokenizer files in this directory are ignored by Git. Do not commit model artifacts.

## Expected Usage

```bash
.venv/bin/python scripts/download_local_model.py --model qwen3_14b
.venv/bin/python scripts/download_local_model.py --model phi4_mini_instruct
```

The model-variant runner reads model paths from [`../../ics_agent/experiments/model_variant_runner.py`](../../ics_agent/experiments/model_variant_runner.py) and local environment variables from [`.env.example`](../../.env.example).
