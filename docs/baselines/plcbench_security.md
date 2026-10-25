# PLCBench native actor — security posture & confinement gate

The **native-protocol** arm runs **model-generated code** (free-form Python/shell)
in a subprocess, by design — that is the condition being studied. This document
states what is and is not contained, the gate that enforces safe use, and the
controls required before the native actor runs anywhere but the isolated local
dev loop. (The **semantic-menu** arm runs *no* free-form code: it issues fixed
Modbus writes for named candidates, so it is out of scope here.)

## What the sandbox confines (`ics_agent/plcbench_sandbox.py`)

- **Environment allowlist** — the child starts from an empty env; only `PATH`,
  `HOME`/`TMPDIR` (→ scratch workdir), `LANG`, and the broker host/port are set.
  Control-plane/credential vars are absent by construction.
- **Workdir** — cwd, `HOME`, `TMPDIR` all resolve to a per-episode scratch dir.
- **Resource limits** — `RLIMIT_CPU`, `RLIMIT_FSIZE` (POSIX `setrlimit`).
- **Lifetime** — child runs in its own session; wall-clock timeout kills the
  whole process group, so a spawned server/`sleep` cannot linger.
- **Output cap** — stdout/stderr truncated to a byte ceiling.

## Legacy subprocess limitation

- **No OS network egress isolation.** `python -I` is *import/environment*
  isolation (ignores `PYTHONPATH`, user site, env config) — it is **not** a
  network, filesystem, or process sandbox. The child can open its own sockets to
  any reachable host. The localhost broker is the **intended** native path and
  the logging/rate-limit chokepoint, but it is **not enforced** as the *only*
  path. A localhost broker is not, by itself, proof other routes are blocked.
- This legacy mode remains available for isolated loopback regression tests. The
  versioned batch runner uses the Linux container mode below.

## Linux container mode (`ics_agent/plcbench_container.py`)

The versioned native batch starts each command in a Linux container with
`--network none`, no capabilities, a read-only root filesystem, process/memory/
CPU caps, and a non-root UID. A trusted host relay mounts a broker Unix socket;
the child has an internal `127.0.0.1` relay to that socket. On Docker Desktop,
the Unix socket is hosted by a trusted sidecar in a named volume because macOS
Unix sockets cannot be bind-mounted into the Linux VM. The model child remains
networkless in both cases.

The opt-in integration test confirms a Modbus read succeeds and appears in the
broker log while direct `1.1.1.1:80` egress fails. This is local verification,
not an independent security review.

## The gate (`plcbench_run.require_confinement`)

Block-by-default. The native entry (`run_native_episode`) refuses to execute
unless the caller names the environment, and the acknowledgement is recorded in
the episode bundle (`meta.confinement`):

- `confinement=None` → **refused** (default).
- `confinement="isolated-dev-loop"` → legacy subprocess allowed only for
  `127.0.0.1`/`localhost`/`::1`; **`0.0.0.0` is not loopback**.
- `confinement="linux-container"` → networkless child with broker Unix relay,
  still restricted to a loopback controller target.
- `confinement="external-reviewed"` → **refused**. A string is not proof of
  independent review or external-target controls.

Non-loopback execution has no enabled path in this code.

## Required before any non-loopback / shared / hardware use

1. Revalidate the networkless container and broker relay on the target Linux
   host, including Docker daemon access and the exact image digest.
2. **Broker as the sole, logged path** — every native transaction already passes
   the broker (allowlist + rate cap + transaction log); with (1) it also becomes
   the *only* reachable endpoint, so the log is complete.
3. **Independent review** of confinement and target admission, separate from
   functional tests, before any non-loopback path is added.
4. Keep the lab's physical/emergency controls and any real controller's
   protections **independent** of the evaluated policy.

## Dual-use posture (what this study is and is not)

- **Authority is an experimental condition, not an exploit.** The attacker/
  operator surface is the controller's genuine operator inputs (the `io_` coils
  it reads). Writing a legitimate operator input is **not** an initial compromise
  or a PLC vulnerability, and must not be described as one.
- **No guardrail removal, no offensive-capability expansion.** Model refusal is
  recorded as an *observed outcome*, never "removed" to force action and never
  equated with technical inability. No broader targeting, no interaction with
  public systems, no social engineering of real people.
- **Strategy freedom ≠ unrestricted authority.** A policy may choose its own
  approach *within* a declared, confined laboratory task without any external
  reach.
- The actor's own claim that it believes it is acting on a real utility is **not**
  a prerequisite and cannot be established from its words; state the authorized
  task consistently and report observable behavior.

## Current scope

Approved **only** on the isolated local dev loop (loopback OpenPLC pair, no
sensitive targets reachable from the host). Anything beyond that is gated above.
