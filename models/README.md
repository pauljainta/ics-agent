# models

This directory documents local model storage.

## Contents

- [`local_models/`](local_models/README.md): ignored local checkpoint directory.

Model weights are too large for Git and may also carry license/access constraints. Keep them local.

Use:

```bash
.venv/bin/python scripts/download_local_model.py --model qwen3_14b
.venv/bin/python scripts/download_local_model.py --model phi4_mini_instruct
```

or place compatible checkpoints under `models/local_models/` manually.
