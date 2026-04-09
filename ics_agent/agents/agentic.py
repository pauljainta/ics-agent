from __future__ import annotations

import json
import os
from typing import Any

from .base import RoleAgent
from ..types import RoleDecision


class OptionalQwenDecisionEngine:
    def __init__(self, profile: dict[str, Any], role: str) -> None:
        llm_cfg = profile.get("llm", {})
        self.role = role
        self.backend = str(llm_cfg.get("backend", "heuristic"))
        self.model_id = str(llm_cfg.get("model_id", "Qwen/Qwen3.5-9B"))
        self.prompt_file = llm_cfg.get("prompt_file")
        self.use_langgraph = bool(llm_cfg.get("use_langgraph", False))

    def decide(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self.backend != "qwen_langchain":
            return fallback

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            from langchain_huggingface import HuggingFacePipeline  # type: ignore
        except Exception:
            return fallback

        model_ref = os.environ.get("QWEN_MODEL_PATH", "").strip() or self.model_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=True,
                device_map="auto",
            )
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
            llm = HuggingFacePipeline(pipeline=generator)
            output = llm.invoke(prompt)
            parsed = self._parse_json_response(output)
            return parsed or fallback
        except Exception:
            return fallback

    @staticmethod
    def _parse_json_response(output: Any) -> dict[str, Any] | None:
        text = str(output)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


class AgenticRoleAgent(RoleAgent):
    def __init__(self, profile_name: str, profile: dict[str, Any]) -> None:
        super().__init__(profile_name, profile)
        self.engine = OptionalQwenDecisionEngine(profile, self.role)

    def step(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.mode == "offline":
            return []
        if self.role in {"operator", "attacker"}:
            return self._step_plan_guided(context)
        if self.role in {"observer", "defender"}:
            return self._step_monitoring(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

    def finalize(self, context: dict[str, Any]) -> list[RoleDecision]:
        if self.role != "offline_analysis":
            return []
        metrics = context.get("metrics", {})
        summary = {
            "summary": (
                f"Agentic offline review for {context['scenario'].scenario_id}: "
                f"recovery={metrics.get('recovery_success_rate')} "
                f"violations={metrics.get('safety_violation_count')}"
            ),
            "diagnosis": context.get("scenario").metadata.get("ground_truth", {}).get("expected_diagnosis"),
            "labels": ["offline", "agentic"],
        }
        payload = self.engine.decide(self._offline_prompt(context), summary)
        return [
            self.decision(
                decision_type="analysis",
                summary=str(payload.get("summary", summary["summary"])),
                time_sec=float(context.get("duration_sec", 0.0)),
                confidence=float(payload.get("confidence", 0.7)),
                diagnosis=payload.get("diagnosis"),
                labels=list(payload.get("labels", ["offline", "agentic"])),
                metadata={"metrics": metrics, "backend": self.engine.backend},
            )
        ]

    def _step_plan_guided(self, context: dict[str, Any]) -> list[RoleDecision]:
        planned = context.get("planned_events", {}).get(self.role, [])
        cursor = int(self._state.get("cursor", 0))
        now = float(context["time_sec"])
        if cursor >= len(planned) or now < float(planned[cursor]["time_sec"]):
            return []

        action = planned[cursor]["action"]
        fallback = {
            "summary": f"Agentic {self.role} selected pending scenario action",
            "action": action if isinstance(action, dict) else {"type": action},
            "confidence": 0.7,
            "labels": ["agentic", "plan-guided"],
        }
        payload = self.engine.decide(self._plan_prompt(context, action), fallback)
        self._state["cursor"] = cursor + 1
        return [
            self.decision(
                decision_type="action",
                summary=str(payload.get("summary", fallback["summary"])),
                time_sec=now,
                action=payload.get("action", fallback["action"]),
                confidence=float(payload.get("confidence", fallback["confidence"])),
                diagnosis=payload.get("diagnosis"),
                labels=list(payload.get("labels", fallback["labels"])),
                metadata={"backend": self.engine.backend, "plan_index": cursor},
            )
        ]

    def _step_monitoring(self, context: dict[str, Any]) -> list[RoleDecision]:
        observation = context["observation"]
        diagnosis = "nominal_operation"
        labels = ["agentic", "monitoring"]
        fallback: dict[str, Any] = {
            "summary": f"Agentic {self.role} sees no action requirement",
            "diagnosis": diagnosis,
            "labels": labels,
        }

        if observation.get("Alarm_RW_Tank_HH") or observation.get("Alarm_RW_Tank_LL"):
            diagnosis = "raw_water_alarm_condition"
            fallback = {
                "summary": f"Agentic {self.role} detected {diagnosis}",
                "diagnosis": diagnosis,
                "labels": labels + ["alerted"],
                "confidence": 0.8,
            }
            if self.role == "defender":
                fallback["action"] = {"type": "hmi_stop", "process_id": "WT-P1"}

        if observation.get("RW_Tank_Level", 0) < 450:
            diagnosis = "raw_water_low_or_spoof"
            fallback["summary"] = f"Agentic {self.role} suspects {diagnosis}"
            fallback["diagnosis"] = diagnosis
            if self.role == "defender":
                fallback["action"] = {"type": "hmi_stop", "process_id": "WT-P1"}

        payload = self.engine.decide(self._monitor_prompt(context), fallback)
        decisions: list[RoleDecision] = []
        if "summary" in payload:
            decisions.append(
                self.decision(
                    decision_type="alert" if payload.get("action") or "alerted" in payload.get("labels", []) else "note",
                    summary=str(payload["summary"]),
                    time_sec=float(context["time_sec"]),
                    action=payload.get("action"),
                    confidence=float(payload.get("confidence", 0.6)),
                    diagnosis=payload.get("diagnosis"),
                    labels=list(payload.get("labels", labels)),
                    metadata={"backend": self.engine.backend},
                )
            )
        return decisions

    def _step_planner(self, context: dict[str, Any]) -> list[RoleDecision]:
        if "planner-note" in self._state["emitted_notes"]:
            return []
        self._state["emitted_notes"].add("planner-note")
        scenario = context["scenario"]
        fallback = {
            "summary": f"Agentic planner briefing for {scenario.scenario_id}: {scenario.metadata['intent']}",
            "labels": ["agentic", "planner"],
            "confidence": 0.7,
        }
        payload = self.engine.decide(self._planner_prompt(context), fallback)
        return [
            self.decision(
                decision_type="note",
                summary=str(payload.get("summary", fallback["summary"])),
                time_sec=float(context["time_sec"]),
                confidence=float(payload.get("confidence", fallback["confidence"])),
                diagnosis=payload.get("diagnosis"),
                labels=list(payload.get("labels", fallback["labels"])),
                metadata={"backend": self.engine.backend},
            )
        ]

    def _plan_prompt(self, context: dict[str, Any], action: dict[str, Any] | str) -> str:
        return (
            f"Role={self.role}\n"
            f"Scenario={context['scenario'].scenario_id}\n"
            f"Time={context['time_sec']}\n"
            f"Observation={context['observation']}\n"
            f"PendingAction={action}\n"
            "Return JSON with keys summary, action, confidence, diagnosis, labels."
        )

    def _monitor_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"Role={self.role}\n"
            f"Scenario={context['scenario'].scenario_id}\n"
            f"Observation={context['observation']}\n"
            f"RecentAlerts={context.get('alerts', [])[-5:]}\n"
            "Return JSON with keys summary, confidence, diagnosis, labels, optional action."
        )

    def _planner_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"Role=planner\n"
            f"Scenario={context['scenario'].scenario_id}\n"
            f"Intent={context['scenario'].metadata['intent']}\n"
            f"ExpectedOutcomes={context['scenario'].metadata.get('expected_outcomes', [])}\n"
            "Return JSON with keys summary, confidence, diagnosis, labels."
        )

    def _offline_prompt(self, context: dict[str, Any]) -> str:
        return (
            f"Role=offline_analysis\n"
            f"Scenario={context['scenario'].scenario_id}\n"
            f"Metrics={context.get('metrics', {})}\n"
            "Return JSON with keys summary, confidence, diagnosis, labels."
        )

