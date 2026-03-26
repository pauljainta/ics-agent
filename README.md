# Local Qwen Runner

This project is trimmed to run a local/offline `Qwen/Qwen3.5-9B` model with one shared Python environment.

## Requirements
- Python module: `python/3.10.3`
- Single project venv: `.venv`
- Optional GPU modules:
  - `cuda/11.8.0`
  - `cudnn/8.6.0.163-11.8-gpu`

## Setup
```bash
bash scripts/setup_env.sh
cp -f .env.example .env
bash scripts/download_qwen_model.sh
```

## Run
```bash
bash scripts/run_local_qwen.sh
```

## CPU-only run
```bash
QWEN_DEVICE=cpu bash scripts/run_local_qwen.sh
```

## Custom prompt
```bash
source .venv/bin/activate
set -a; source .env; set +a
python models/qwen_test.py --offline --prompt "List three PLC tags for tank level control."
```

## Precision options
Use `QWEN_PRECISION` (or `--precision`) to control memory/speed:
- `auto` (default): fp16 on GPU, fp32 on CPU
- `bf16`: bfloat16 on GPU
- `fp16`: float16 on GPU
- `8bit`: 8-bit quantized load
- `4bit`: 4-bit quantized load

Example for tighter memory:
```bash
QWEN_PRECISION=4bit bash scripts/run_local_qwen.sh
```

## Notes
- Default model id: `Qwen/Qwen3.5-9B`
- Default local model folder: `./models/local_models/qwen3.5-9b`
- `scripts/run_local_qwen.sh` auto-falls back to CPU if GPU modules are unavailable.
