# Running the PLCBench model on CHPC (vLLM) with a local testbed

The chosen deployment is **Qwen2.5-32B-Instruct** served by **vLLM** on a CHPC
GPU node; the OpenPLC pair + broker + sandbox stay **local** and the agent reaches
the model over an SSH tunnel. Only inference runs on the GPU. Both comparison arms
(`plcbench_matched_*` and `gray_box_ollama_*`/comparator) use the one frozen
deployment.

Model choice rationale and alternatives: see `../plcbench_conformance.md`.

## 0. Confirm the allocation (do not inherit one)

Run `myallocation` on CHPC and use the account/partition it returns. The ecog
project uses a different allocation, which may not cover SPHERE/ics-agent — its own
runbook warns against inheriting a hard-coded account. Put the account, partition,
SSH alias, and any API key in a local **untracked** file (e.g. `.private/`), never
in Git.

## 1. VRAM math (pick the serving mode)

- **32B bf16** ≈ 65 GB weights → needs one 80 GB A100/H100, or two 40 GB A100s
  with `--tensor-parallel-size 2` (`--gres=gpu:a100:2`).
- **32B AWQ (4-bit)** ≈ 20 GB → fits **one 40 GB A100** with room for the KV
  cache at `--max-model-len 8192`. This is the Slurm script's default and the
  easiest to schedule. Quality drop vs bf16 is small.

## 2. Pre-stage the weights (compute nodes are often offline)

On a login node, with scratch as the HF cache:

```bash
export HF_HOME=/opt/research-runtime/hf  # confirm this path on your account
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ
```

Create the serving env once (conda + vLLM):

```bash
module load miniconda3/25.9.1
conda create -y -n vllm python=3.11 && conda activate vllm
pip install vllm
```

## 3. Launch the server

For the 50-pair WD supervisor, `serve_study_20261015.slurm` is a
**parameterized template**. Submit a private copy with explicit
`--account`/`--partition` and exported `PLCBENCH_STUDY_ROOT` and
`PLCBENCH_LOGIN_IP`; the latter must be the current login-node source IP
seen by the compute node. The local supervisor requires the matching
`PLCBENCH_CHPC_*` variables listed in the
[handoff guide](../plcbench_jp_handoff.md). The previous job's exact script
bytes are retained only in ignored local evidence, so a new deployment needs
its own script hash and identity record.

If CHPC provides a `vllm` module, inspect `module show vllm/current` and pin
the resolved Apptainer image by SHA-256 in the job script. This avoids creating
a separate conda environment. Keep the submitted script and its image hash in
the ignored local deployment evidence directory.
The [Apptainer template](serve_qwen_vllm_apptainer.slurm) uses a private key file
without a key CLI argument. Both launch templates bind to `127.0.0.1` by
default. If the login-node tunnel cannot reach that address on the compute
node, arrange cluster-side port restrictions first and set
`PLCBENCH_BIND_HOST` explicitly for that deployment.

```bash
HF_HOME=/opt/research-runtime/hf sbatch docs/baselines/chpc/serve_qwen_vllm.slurm
# watch the log for "serving ... on <node> port 8000"
tail -f logs/plcbench_vllm_*.out
```

Override via env if needed, e.g. bf16 on two GPUs:
`PLCBENCH_MODEL=Qwen/Qwen2.5-32B-Instruct PLCBENCH_QUANT= PLCBENCH_TP=2 sbatch ...`
(also change `--gres=gpu:a100:2`).

## 4. Tunnel from the machine running the testbed

```bash
# <node> is the compute node from the log; <chpc> your SSH alias (notchpeak).
ssh -N -L 8000:<node>:8000 <chpc>
# verify
curl -s http://localhost:8000/v1/models
```

When the service uses an API key, pass it in the `Authorization: Bearer` header
for the probe. Keep the key in an untracked file with mode `600` and provide its
path as `PLCBENCH_API_KEY_FILE` at job submission. For an Apptainer launch,
pass the key into the container as `APPTAINERENV_VLLM_API_KEY`; avoid a
`--api-key` argument because vLLM logs non-default CLI arguments. The API key
protects `/v1` endpoints, so also restrict access to the serving port on the
cluster network.

The `/v1/models` response has per-request timestamps and permission IDs. Do not
use a hash of that whole response as a deployment fingerprint. Freeze the stable
model ID, serving root, and context length from the health probe alongside the
serving-node weight revision and runtime digest. The batch runner checks the
stable listing and snapshot revision against a fresh probe before execution.
After capturing the serving-node metadata, use `scripts/plcbench_pin_deployment.py`
with `--record`, `--model-url`, `--api-key-file`, and `--freeze-output` to create a
new pinned JSON file. Run it again with `--record` pointing to that pinned file
and without `--freeze-output` to verify the live endpoint. The key stays out of
the record. The batch runner requires the pinned file as `--deployment-json`.

## 5. Point the agents at it

The agents already support an `openai`/`vllm` backend (vLLM speaks the OpenAI
API). Set the two comparison profiles' `llm` block to:

```yaml
    llm:
      backend: openai
      base_url: http://localhost:8000/v1
      model_id: plcbench          # matches --served-model-name
      # api_key: <only if you set PLCBENCH_API_KEY on the server>
      temperature: 0
      max_new_tokens: 512
      require_cuda: false
```

Keep the rest of each profile unchanged, so `plcbench_matched_*` (scaffold) and
its comparator remain a matched pair differing only in the scaffold. For the
native actor, build its generator with
`build_openai_generate("plcbench", base_url="http://localhost:8000/v1")`.
For `scripts/plcbench_batch.py`, set `PLCBENCH_API_KEY` when the server requires
it. `--model-url` accepts either the tunnel root or its `/v1` URL; the health
probe and chat client normalize those paths separately.

Dev parity: the local Ollama profiles (`llama3.1:latest`, swap to `qwen2.5`)
stay as the no-GPU smoke path; CHPC/vLLM is for measured runs.

## Freeze + record (both arms, identical)

HF checkpoint + revision, vLLM version, quantization (awq/none), tensor-parallel
size, `max-model-len`, temperature 0, `max_tokens` 512. Record these in the
run-bundle meta and in `../plcbench_conformance.md` when you freeze for measured
runs.

## Governance

No patient/PHI data is involved here (a public model + ICS telemetry prompts), so
this is lighter than the ecog MRI path — but still: no SSH aliases, accounts,
hostnames, or API keys in tracked Git; scratch is not backed up (keep the HF
cache reproducible via the model id); and confirm this project's own CHPC
allocation rather than reusing another project's.
