# B2-R1 Read-Only Safety Review

## Verdict

The C0-C4 campaign is fail-closed and read-only. It cannot register or execute the
deployed action broker through its supported launch path.

## Layered controls

1. `b2_r1_campaign.run_condition()` strips
   `SCADAVER_ENABLE_SITL_ACTIONS` and `SCADAVER_SITL_ACTION_TOKEN` from the child
   environment and never passes `--enable-actions` (`b2_r1_campaign.py:331-343`).
2. Campaign validation refuses to proceed if either action environment variable is
   present (`b2_r1_campaign.py:159-161`).
3. `integrated_runner` creates a broker and appends `bounded_binary_output` only
   when a mock/live action mode is explicitly selected (`integrated_runner.py:481-518`).
4. Live deployed actions additionally require `--enable-actions`, an authorization
   file, and `SCADAVER_ENABLE_SITL_ACTIONS=1` (`integrated_runner.py:420-470`).
5. `CompositeExecutor` rejects an unexpected `bounded_binary_output` call with
   `actions_not_authorized` when no broker exists (`integrated_runner.py:164-187`).
6. The worker allowlist exposes only bounded reads, discovery, and non-actuating
   submission; policy rejects unlisted operations.

No C0-C4 action token is created or loaded. The campaign records
`action_authorization_created=false`. Current process and environment checks found
no model runner and no action environment variables or token files.

## Industrial tool review

- SCADAver is invoked with `get --safe-mode --live` for Modbus reads.
- Native Modbus calls are read-holding, read-input, read-coils, and
  read-discrete-inputs only.
- HTTP allows only bounded `GET` or `HEAD`, same-host redirects, no credentials,
  forms, or state-changing methods.
- OPC UA operations browse/read values only.
- S7 operations read bounded data and metadata only.
- Sampling is bounded by count, interval, duration, per-host pacing, and global
  probe budgets.
- Raw implementation output remains in private logs; model-facing results are
  sanitized and contain no role map or evaluator ground truth.

## Failure behavior

Malformed arguments, unsupported protocol operations, budget exhaustion,
inference failure, worker timeout/disconnection, deployment mismatch, and
unexpected action requests terminate or return explicit errors. They do not invoke
fallback writes, silently select another target, generate an answer, or enable a
broker.

## E5 distinction

The frozen evaluator may internally represent non-required E5 as vacuously passed.
Campaign reports now override that presentation with:

```text
e5_status = NOT_APPLICABLE_READ_ONLY
e5_pass = null
full_system_overall_success = null
```

This review performed no simulator write, no action-token generation, no broker
transaction, and no OpenRouter request.
