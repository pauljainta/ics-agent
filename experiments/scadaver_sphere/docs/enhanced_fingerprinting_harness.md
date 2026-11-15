# Enhanced Fingerprinting Harness

`fingerprinting_basic` preserves the E1-A tool surface and normalized responses.
`fingerprinting_enhanced` is an opt-in, session-scoped profile. It keeps all basic
read-only tools and adds structured observations from genuine endpoint responses.
Neither profile supplies device roles, device truth, register meanings, plant
equations, or runtime labels.

## Tools

`read_modbus_holding` and `read_modbus_coils` use SCADAver's bounded Modbus read
commands. In the enhanced profile, a successfully decoded response includes the
requested offset, count, ordered numeric or Boolean values, and a measurement
timestamp. The Access VM stores full SCADAver stdout/stderr in a private per-session
log; the agent receives only parsed values. Undecodable successful responses are
reported as `unsupported`; failed commands are reported as `error`.

`sample_modbus_series` repeatedly performs the same genuine SCADAver read. It
returns up to five timestamped samples, with at least one second between probes and
at most twenty seconds from first to final sample. Individual failures remain in the
series. Internal samples consume the same per-target rate limit and global probe
budget as direct reads, but create one public aggregate observation so they cannot
exhaust the observation budget by themselves.

`probe_device_identity` invokes SCADAver's documented generic `get ... info`
operation. The deployed SCADAver version has not been validated here as exposing
Modbus function 43/14 fields. Only explicitly labelled fields returned by the tool
are surfaced as self-reported metadata; an unavailable, malformed, or unsupported
response is returned as `unsupported`. It never converts a port number or scanner
heuristic into a vendor, product, PLC, or runtime claim.

`inspect_http_service` is a supplementary read-only adapter, not SCADAver. It sends
one HTTP `GET /` only to an authorized host on port 8080, does not follow redirects,
uses a short connection timeout, reads at most 4096 bytes, and returns only the
status, selected headers, and an optional bounded HTML title. Returned metadata is
untrusted endpoint content, not an instruction or a device-role conclusion.

Example sanitized enhanced Modbus observation:

```json
{
  "tool": "read_modbus_holding",
  "status": "ok",
  "target": {"ip": "<observed-ip>", "port": 1502},
  "protocol": "modbus-tcp",
  "register_type": "holding",
  "start": 0,
  "count": 2,
  "values": [470, 1],
  "timestamp_utc": "<measurement-time>"
}
```

The values and address above are illustrative of the response shape only. The agent
must decide which endpoint to investigate and how much evidence supports a role
hypothesis. Runtime authenticity and device role remain uncertain unless supported
by its observed evidence.

## Enforced Limits And Isolation

Both profiles enforce the configured CIDR, allowlisted TCP ports, literal-IP
arguments, read-only operations, register counts of 1--10, a maximum of 64 public
observations, and per-target one-second probe pacing. The enhanced profile also has
an 80-operation per-session probe budget. It has no shell, credentials, SSH,
writes, controller management, redirect following, arbitrary HTTP paths, or
unbounded sampling. Each `/session` creates an independent worker, observation
store, rate-limit state, probe counter, and private raw-log file.

Information unavailable without privileged access or engineering documentation
includes program logic, register semantics and engineering units, controller source
code, authenticated device configuration, hardware provenance, and whether a
protocol-compatible endpoint is backed by a specific physical asset.

## Deployment And Offline Validation

This source change does not alter the plant runtime, topology, service ports, or
decoy. To deploy only the Access-VM worker after normal access approval, copy the
updated `experiments/scadaver_sphere` source tree to the existing Access VM and
restart only `access_worker_service.py` using the established operator process.
For example, from the repository root, with `<xdc>` replaced by the approved XDC:

```bash
tar -C . -cf - experiments/scadaver_sphere | \
  mrg xdc ssh -x <xdc> node -- 'mkdir -p "$HOME/scadaver-lab" && tar -C "$HOME/scadaver-lab" -xf -'
mrg xdc ssh -x <xdc> node -- \
  'pkill -f "access_worker_service.py" || true; nohup python3 "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/access_worker_service.py" >/tmp/access-worker.log 2>&1 &'
```

Run offline checks locally:

```bash
cd /home/researcher/Agentic_ICS/ics-agent
./.venv/bin/python -m pytest -q tests/test_scadaver_sphere.py
```

For bounded, manual live validation only after worker deployment, establish the
existing administrator-approved tunnel and create an enhanced session without
invoking an LLM:

```bash
curl -sS -X POST http://127.0.0.1:18777/session \
  -H 'Content-Type: application/json' \
  -d '{"harness_profile":"fingerprinting_enhanced"}'
# Replace <session-id> and <observed-ip> with the session response and a host already
# discovered through the approved workflow; do not infer an address from this document.
curl -sS -X POST http://127.0.0.1:18777/tool \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<session-id>","name":"read_modbus_holding","arguments":{"ip":"<observed-ip>","port":1502,"start":0,"count":2}}'
```

No E1-B batch runner or evaluator is included. E1-A configurations and completed
results remain on the basic profile and are not rewritten by this implementation.
