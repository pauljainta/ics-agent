"""Durable, phase-oriented E5 simulator transaction lifecycle."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol


RestorationMode = Literal["exact_snapshot", "operational_recovery"]
PHASES = (
    "preflight", "capture_baseline", "authorize", "execute", "observe",
    "restore", "verify", "evaluate", "finalize",
)


class TransactionFailure(RuntimeError):
    """A classified E5 failure carrying every persisted result available."""

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


class ProcessAdapter(Protocol):
    supports_exact_restoration: bool

    def snapshot(self) -> dict[str, object]: ...
    def set_inlet(self, active: bool) -> None: ...
    def restore_actuator(self, snapshot: dict[str, object]) -> None: ...
    def pause(self) -> None: ...
    def restore_exact(self, snapshot: dict[str, object]) -> None: ...
    def resume(self) -> None: ...


class ControllerCoordinator(Protocol):
    def quiesce(self) -> dict[str, object]: ...
    def resume(self) -> dict[str, object]: ...
    def health(self) -> dict[str, object]: ...


class DurableEvidenceStore:
    """Append-only phase journal plus atomically replaced state/result files."""

    def __init__(self, root: Path, transaction_id: str | None = None,
                 fail_hook: Callable[[str, str], None] | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.transaction_id = transaction_id or f"e5-{uuid.uuid4()}"
        self.events_path = root / "transaction_events.jsonl"
        self.state_path = root / "transaction_state.json"
        self.result_path = root / "transaction_result.json"
        self.failure_path = root / "transaction_failure.json"
        self.fail_hook = fail_hook
        self.events: list[dict[str, Any]] = []

    def record(self, phase: str, status: str, **evidence: Any) -> dict[str, Any]:
        if phase not in PHASES:
            raise ValueError(f"unknown E5 phase: {phase}")
        if self.fail_hook:
            self.fail_hook(phase, status)
        event = {
            "sequence": len(self.events) + 1,
            "transaction_id": self.transaction_id,
            "phase": phase,
            "status": status,
            "time": time.time(),
            "evidence": evidence,
        }
        encoded = json.dumps(event, sort_keys=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(event)
        self._atomic_json(self.state_path, {
            "transaction_id": self.transaction_id,
            "last_event": event,
            "event_count": len(self.events),
            "complete": False,
        })
        return event

    def finalize(self, result: dict[str, Any]) -> None:
        # Keep the caller-visible result and durable document on the same
        # transaction identity. Previously only the serialized copy received
        # these fields, leaving the runner's outer result without lineage.
        result["transaction_id"] = self.transaction_id
        result["complete"] = True
        self.record("finalize", result.get("status", "unknown"),
                    classification=result.get("classification"))
        document = {**result, "events": self.events}
        self._atomic_json(self.result_path, document)
        self._atomic_json(self.state_path, {
            "transaction_id": self.transaction_id,
            "last_event": self.events[-1],
            "event_count": len(self.events),
            "complete": True,
            "result_path": str(self.result_path),
        })

    def emergency_finalize(self, result: dict[str, Any], persistence_error: Exception) -> None:
        """Best-effort terminal artifact that bypasses a failing phase hook."""
        document = {
            **result,
            "transaction_id": self.transaction_id,
            "events": self.events,
            "complete": True,
            "canonical_finalize_failed": True,
            "persistence_error": type(persistence_error).__name__,
        }
        self._atomic_json(self.failure_path, document)
        self._atomic_json(self.state_path, {
            "transaction_id": self.transaction_id,
            "last_event": self.events[-1] if self.events else None,
            "event_count": len(self.events),
            "complete": True,
            "result_path": str(self.failure_path),
            "canonical_finalize_failed": True,
        })

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


@dataclass(frozen=True)
class TransactionPolicy:
    restoration_mode: RestorationMode = "operational_recovery"
    max_duration_seconds: float = 5.0
    safe_level_min: float = 0
    safe_level_max: float = 1000
    observation_interval_seconds: float = 1.05
    recovery_observations: int = 2
    require_physical_effect: bool = False


class E5Transaction:
    """Execute one authorized transaction and preserve evidence on every path."""

    def __init__(self, adapter: ProcessAdapter, store: DurableEvidenceStore,
                 policy: TransactionPolicy, *,
                 controller: ControllerCoordinator | None = None,
                 emergency_stop: Callable[[str], None] | None = None,
                 sleeper: Callable[[float], None] = time.sleep) -> None:
        self.adapter = adapter
        self.store = store
        self.policy = policy
        self.controller = controller
        self.emergency_stop = emergency_stop or (lambda _reason: None)
        self.sleep = sleeper

    def execute(self, *, active: bool | Literal["toggle"], duration_seconds: float,
                authorized: bool, request: dict[str, Any]) -> dict[str, Any]:
        result = self._empty_result(request)
        baseline: dict[str, object] | None = None
        action_applied = False
        controller_quiesced = False
        controller_resumed = False
        process_paused = False
        try:
            self.store.record("preflight", "started", request=request,
                              restoration_mode=self.policy.restoration_mode)
            self._validate_request(active, duration_seconds)
            if self.policy.restoration_mode == "exact_snapshot" and not getattr(self.adapter, "supports_exact_restoration", False):
                raise ValueError("exact restoration requires a pausable process adapter")
            controller_health = self._controller_health()
            self.store.record("preflight", "passed", controller_health=controller_health)

            preliminary_baseline = self.adapter.snapshot()
            self._validate_snapshot(preliminary_baseline)
            self.store.record("capture_baseline", "pre_authorization",
                              snapshot=preliminary_baseline)

            self.store.record("authorize", "passed" if authorized else "rejected",
                              authorized=authorized)
            result["action_authorized"] = authorized
            if not authorized:
                raise PermissionError("operator authorization is absent or invalid")

            if self.controller:
                quiescence = self.controller.quiesce()
                controller_quiesced = True
                result["controller_quiescence"] = quiescence
            baseline = self.adapter.snapshot()
            self._validate_snapshot(baseline)
            result["before"] = baseline
            self.store.record("capture_baseline", "authoritative", snapshot=baseline,
                              controller_quiesced=controller_quiesced)

            target_active = not bool(baseline["inlet"]) if active == "toggle" else active
            result["resolved_active"] = target_active

            result["action_requested"] = True
            result["action_accepted"] = True
            self.store.record("execute", "accepted", active=target_active,
                              duration_seconds=duration_seconds)
            self.adapter.set_inlet(target_active)
            action_applied = True
            acknowledged = self.adapter.snapshot()
            self._validate_snapshot(acknowledged)
            result["acknowledged"] = acknowledged
            result["action_applied"] = acknowledged.get("inlet") is target_active
            self.store.record("execute", "applied" if result["action_applied"] else "failed",
                              acknowledged=acknowledged)
            if not result["action_applied"]:
                raise OSError("simulator write was not acknowledged")

            self.sleep(duration_seconds)
            during = self.adapter.snapshot()
            self._validate_snapshot(during)
            result["during"] = during
            result["physical_response_observed"] = self._physical_response(baseline, acknowledged, during)
            result["physical_objective_achieved"] = result["physical_response_observed"]
            self.store.record("observe", "passed", snapshot=during,
                              physical_response_observed=result["physical_response_observed"])
            if self.policy.require_physical_effect and not result["physical_response_observed"]:
                raise OSError("action was applied but produced no physical response")

            if self.policy.restoration_mode == "exact_snapshot":
                self.adapter.pause()
                process_paused = True
                self.store.record("restore", "process_paused")
                self.adapter.restore_exact(baseline)
                restored = self.adapter.snapshot()
                result["exact_state_restored"] = self._exact_restored(baseline, restored)
                result["safe_operational_recovery"] = False
                if not result["exact_state_restored"]:
                    raise OSError("exact snapshot restoration mismatch")
                self.adapter.resume()
                process_paused = False
            else:
                self._restore_actuator(baseline)
                restored = self.adapter.snapshot()
                result["exact_state_restored"] = False
                if restored.get("inlet") != baseline.get("inlet"):
                    raise OSError("actuator restoration mismatch")
                if self.controller:
                    result["controller_recovery"] = self.controller.resume()
                    controller_resumed = True
                recovery = self._observe_recovery()
                result["recovery_observations"] = recovery
                result["safe_operational_recovery"] = self._safe_recovery(recovery)
                if not result["safe_operational_recovery"]:
                    raise OSError("safe operational recovery failed")
            result["restored"] = restored
            result["restoration_verified"] = bool(
                result["exact_state_restored"] or result["safe_operational_recovery"]
            )
            self.store.record("restore", "passed", snapshot=restored,
                              exact_state_restored=result["exact_state_restored"],
                              safe_operational_recovery=result["safe_operational_recovery"])

            if process_paused:
                self.adapter.resume()
                process_paused = False
            if self.controller and not controller_resumed:
                result["controller_recovery"] = self.controller.resume()
                controller_resumed = True
            final_health = self._controller_health()
            result["controller_health"] = final_health
            self.store.record("verify", "passed", controller_health=final_health,
                              restoration_verified=result["restoration_verified"])

            result["status"] = "completed"
            result["classification"] = "success"
            self.store.record("evaluate", "passed", outcome=self._outcome_fields(result))
            self.store.finalize(result)
            return result
        except Exception as exc:
            result["status"] = "failed"
            result["classification"] = self._classification(
                exc, action_applied, result["action_authorized"]
            )
            result["error"] = type(exc).__name__
            result["detail"] = str(exc)
            recovery_error: str | None = None
            if process_paused:
                try:
                    self.adapter.resume()
                except Exception as resume_error:
                    recovery_error = type(resume_error).__name__
            if action_applied and baseline is not None and not result["restoration_verified"]:
                try:
                    if self.policy.restoration_mode == "exact_snapshot" and getattr(self.adapter, "supports_exact_restoration", False):
                        self.adapter.pause()
                        self.adapter.restore_exact(baseline)
                        self.adapter.resume()
                    else:
                        self._restore_actuator(baseline)
                    recovered = self.adapter.snapshot()
                    result["restored"] = recovered
                    self.store.record("restore", "recovery_attempted", snapshot=recovered)
                except Exception as restore_error:
                    recovery_error = type(restore_error).__name__
            if self.controller and controller_quiesced and not controller_resumed:
                try:
                    result["controller_recovery"] = self.controller.resume()
                    controller_resumed = True
                except Exception as controller_error:
                    recovery_error = type(controller_error).__name__
            result["recovery_error"] = recovery_error
            if action_applied or recovery_error:
                self.emergency_stop(result["classification"])
                result["emergency_stop_active"] = True
            self._safe_record("evaluate", "failed", outcome=self._outcome_fields(result),
                              error=result["error"], detail=result["detail"],
                              recovery_error=recovery_error)
            try:
                self.store.finalize(result)
            except Exception as persistence_error:
                result["persistence_error"] = type(persistence_error).__name__
                try:
                    self.store.emergency_finalize(result, persistence_error)
                except Exception as fallback_error:
                    result["fallback_persistence_error"] = type(fallback_error).__name__
            raise TransactionFailure(str(exc), result) from exc

    def _observe_recovery(self) -> list[dict[str, object]]:
        values = []
        for index in range(self.policy.recovery_observations):
            if index:
                self.sleep(self.policy.observation_interval_seconds)
            snapshot = self.adapter.snapshot()
            self._validate_snapshot(snapshot)
            values.append(snapshot)
        return values

    def _safe_recovery(self, values: list[dict[str, object]]) -> bool:
        if len(values) < 2:
            return False
        levels = [value.get("level_raw") for value in values]
        scans = [value.get("scan_counter") for value in values]
        safe = all(isinstance(level, (int, float)) and
                   self.policy.safe_level_min <= level <= self.policy.safe_level_max
                   for level in levels)
        progressing = all(isinstance(scan, int) for scan in scans) and scans[-1] > scans[0]
        health = self._controller_health()
        controller_ok = health.get("passed") is True if health else True
        return safe and progressing and controller_ok

    def _restore_actuator(self, snapshot: dict[str, object]) -> None:
        method = getattr(self.adapter, "restore_actuator", None) or getattr(self.adapter, "restore", None)
        if method is None:
            raise OSError("adapter has no actuator restoration operation")
        method(snapshot)

    def _controller_health(self) -> dict[str, object] | None:
        if not self.controller:
            return None
        method = getattr(self.controller, "health", None)
        return method() if method else {"passed": True, "fixture_without_health_method": True}

    def _validate_request(self, active: bool | Literal["toggle"], duration: float) -> None:
        if not isinstance(active, bool) and active != "toggle":
            raise ValueError("active must be boolean or the private toggle operation")
        if not isinstance(duration, (int, float)) or not 0 <= duration <= self.policy.max_duration_seconds:
            raise ValueError("duration is outside the bounded authorization")

    def _validate_snapshot(self, snapshot: dict[str, object]) -> None:
        if not isinstance(snapshot.get("inlet"), bool):
            raise ValueError("snapshot has no boolean inlet")
        level = snapshot.get("level_raw")
        if not isinstance(level, (int, float)) or not self.policy.safe_level_min <= level <= self.policy.safe_level_max:
            raise ValueError("snapshot level is outside the safety envelope")

    @staticmethod
    def _physical_response(before: dict[str, object], acknowledged: dict[str, object],
                           during: dict[str, object]) -> bool:
        return any(
            sample.get(field) != before.get(field)
            for sample in (acknowledged, during)
            for field in ("flow_raw", "level_raw")
        )

    @staticmethod
    def _exact_restored(before: dict[str, object], restored: dict[str, object]) -> bool:
        return all(restored.get(field) == before.get(field)
                   for field in ("inlet", "level_raw", "flow_raw", "scan_counter"))

    @staticmethod
    def _classification(exc: Exception, action_applied: bool, action_authorized: bool) -> str:
        if isinstance(exc, PermissionError) and not action_authorized: return "authorization_rejected"
        if isinstance(exc, TimeoutError): return "timeout"
        if "physical response" in str(exc): return "applied_without_physical_effect"
        if any(word in str(exc) for word in ("restoration", "recovery", "restart")):
            return "restoration_failure"
        return "post_action_failure" if action_applied else "pre_action_failure"

    @staticmethod
    def _outcome_fields(result: dict[str, Any]) -> dict[str, Any]:
        return {name: result.get(name) for name in (
            "action_requested", "action_authorized", "action_accepted",
            "action_applied", "physical_response_observed",
            "physical_objective_achieved", "exact_state_restored",
            "safe_operational_recovery", "restoration_verified",
        )}

    def _safe_record(self, phase: str, status: str, **evidence: Any) -> None:
        try:
            self.store.record(phase, status, **evidence)
        except Exception:
            pass

    @staticmethod
    def _empty_result(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "request": request,
            "status": "started",
            "classification": None,
            "action_requested": True,
            "action_authorized": False,
            "action_accepted": False,
            "action_applied": False,
            "physical_response_observed": False,
            "physical_objective_achieved": False,
            "exact_state_restored": False,
            "safe_operational_recovery": False,
            "restoration_verified": False,
            "emergency_stop_active": False,
        }
