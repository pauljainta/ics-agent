# Full D3 on CHPC for the PLCBench and CTF1 offline trials

The primary decision-model candidate is the official
[`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3) checkpoint at
`cf15549a78fc52598ddd1951f321c58372bbd64f`. The package is about 52 GB.
Its own `d3_server.py` exposes `/v1/systemone`; this deployment uses that
reference server and the PyTorch kernels, not the separate local oMLX
quantization or the older Qwen/vLLM study server. The Slurm job fails if its
single GPU has less than 75,000 MiB. An H100 NVL 96 GB or A100 80 GB allocation
fits this gate; CUDA support must be confirmed on the allocated node.

Both client trials are synthetic and offline. The CTF1 adapter does not submit
to the challenge, and the PLCBench WT adapter does not contact a PLC. Keep the
model, runtime, and endpoint evidence separate from the Qwen 50-pair study.

## Prepare once with CHPC Slurm jobs

Authenticate interactively through CHPC. Run `myallocation` and verify that the
account and partition assigned to this project permit one 80 GB GPU. Do not
inherit the account from the older Qwen job. Check scratch quota and available
space; allow at least 70 GB for the checkpoint, cache, and environment.

Copy the D3 Slurm and preparation scripts in this directory, plus
`scripts/plcbench_systemone_relay.py`, to a private scratch directory on
CHPC. Run environment setup on a CPU compute node and the 52 GB download on a
data transfer node. From a login shell in that directory:

```bash
export D3_ROOT=/opt/research-runtime/plcbench-d3
mkdir -p logs
ENV_JOB=$(sbatch --parsable --account=<confirmed-cpu-account> \
  --partition=<confirmed-cpu-partition> --export=ALL,D3_ROOT="$D3_ROOT" \
  prepare_d3_env.slurm)
WEIGHTS_JOB=$(sbatch --parsable --dependency=afterok:"$ENV_JOB" \
  --account=<confirmed-transfer-account> --partition=<confirmed-transfer-partition> \
  --export=ALL,D3_ROOT="$D3_ROOT" download_d3_weights.slurm)
# Check the short GPU probe while the transfer runs:
sbatch --account=<confirmed-account> --partition=<confirmed-partition> \
  --gres=gpu:h100nvl:1 --export=ALL,D3_ROOT="$D3_ROOT" probe_d3_gpu.slurm
# After the transfer and probe succeed, submit the GPU server:
sbatch --account=<confirmed-account> --partition=<confirmed-partition> \
  --gres=gpu:h100nvl:1 \
  --export=ALL,D3_ROOT="$D3_ROOT",D3_API_KEY_FILE="$D3_ROOT/private/api_key",D3_LOGIN_IP=<verified-login-source-IP> \
  serve_d3_systemone.slurm
```

The preparation script pins the Hub revision, makes a dedicated Python 3.12
environment, and stores a bearer key with mode 600. It uses the upstream
Transformers 5.17.0 runtime requirements but omits the optional
`flash-linear-attention` accelerator for the first correctness trial. Preserve
the pip freeze file, job script, Slurm log, job ID, and
`runs/d3_<jobid>_identity.json`. The key must remain private and must not be
copied into a result bundle.

The D3 server serves only on compute-node loopback port 18080. A separate
relay binds the compute node's internal address on port 18081, accepts only
the login node's verified source IP, requires the bearer key even for health,
and forwards only `/health`, `/v1/models`, and `/v1/systemone`. Determine the
login source IP with `ip route get <allocated-compute-IP>` on the login node;
do not inherit it from another login session. Tunnel local port 18080 to
`<allocated-compute-node>:18081` through that same login node. The relay's
source-IP check is deliberately narrow, so a different login node must use a
new job with its own allowlist.

## Admission and trial

With a working tunnel to local `127.0.0.1:18080`, use the bearer key in a
private local file and check `/health` and authenticated `/v1/models`. The
model inventory must contain `d3`, and the Slurm identity file must report the
pinned revision, CUDA runtime, GPU, and server hash. An unauthenticated
`/v1/models` request must return 401. Copy the job identity JSON to a local
private location and give its path to each trial runner; both runners reject a
different revision.

```bash
python3 scripts/plcbench_wt_response_systemone.py \
  --sphere-root /path/to/pinned/sphere-usecases \
  --out-dir /private/tmp/plcbench-d3-chpc-preflight --mode preflight \
  --endpoint http://127.0.0.1:18080 --api-key-file /path/to/private/local/key \
  --deployment-json /path/to/private/local/d3_job_identity.json
```

Run the CTF1 ICS staff smoke separately from the CTF1 checkout:

```bash
python3 challenges/ctf1-ics/decision_model_eval.py \
  --out-dir /private/tmp/ctf1-d3-chpc-smoke --mode evaluate \
  --endpoint http://127.0.0.1:18080 --api-key-file /path/to/private/local/key \
  --deployment-json /path/to/private/local/d3_job_identity.json
```

Only after the one-step PLCBench responses pass the exact-action and
probability validation should the WT runner use `--mode full` in a new output
directory. Compare safety and service scores only among identical WT fixtures;
the CTF1 authored eight-case result is a separate smoke evaluation. A failed
admission stays `unassessable_model_not_admitted` and is not counted as a model
decision result.

The pinned full D3 checkpoint was served on CHPC and assessed on two fixed
offline WT fixture cases on 2026-10-21. See the
[trial record](d3_trial_20261021.md) for exact jobs, runtime identity, scores,
and limits. The server was stopped. This file remains the deployment recipe
for a **new** trial; it is not a 50-pair result or physical PLC experiment.
