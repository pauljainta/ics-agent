from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import RoleAgent
from ..paths import REPO_ROOT


def _normalize_action(action: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, str):
        return {"type": action}
    return dict(action)


def _action_key(action: dict[str, Any] | str | None) -> str:
    normalized = _normalize_action(action)
    if normalized is None:
        return "null"
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


class CachedQwenDecisionEngine:
    _model_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
    _llm_cache: dict[tuple[Any, ...], Any] = {}
    _graph_cache: dict[tuple[str, str], Any] = {}

    def __init__(self, profile: dict[str, Any], role: str) -> None:
        llm_cfg = profile.get("llm", {})
        self.role = role
        self.backend = str(llm_cfg.get("backend", "qwen_langgraph"))
        self.model_id = str(llm_cfg.get("model_id", "Qwen/Qwen3.5-9B"))
        self.prompt_file = llm_cfg.get("prompt_file")
        self.use_langgraph = bool(llm_cfg.get("use_langgraph", self.backend == "qwen_langgraph"))
        self.strict_backend = bool(llm_cfg.get("strict_backend", False))
        self.allow_langchain_fallback = bool(llm_cfg.get("allow_langchain_fallback", True))
        self.local_files_only = bool(llm_cfg.get("local_files_only", True))
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 256))
        self.temperature = float(llm_cfg.get("temperature", 0.0))
        self.prompt_text = self._load_prompt()
        self.last_mode = "uninitialized"
        self.last_error: str | None = None

    def _load_prompt(self) -> str:
        if not self.prompt_file:
            return ""
        prompt_path = Path(self.prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = (REPO_ROOT / prompt_path).resolve()
        try:
            return prompt_path.read_text().strip()
        except FileNotFoundError:
            return ""

    def decide(self, context: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        self.last_error = None
        if self.backend == "heuristic":
            self.last_mode = "heuristic"
            return self._finalize_payload(fallback, context, fallback)

        if self.backend not in {"qwen_langchain", "qwen_langgraph"}:
            self.last_error = f"unsupported backend `{self.backend}`"
            if self.strict_backend:
                raise RuntimeError(self.last_error)
            self.last_mode = "heuristic_fallback"
            return self._finalize_payload(fallback, context, fallback)

        errors: list[str] = []
        try:
            if self.backend == "qwen_langgraph" and self.use_langgraph:
                try:
                    payload = self._decide_with_langgraph(context, fallback)
                    self.last_mode = "qwen_langgraph"
                    return payload
                except Exception as exc:
                    errors.append(f"langgraph: {exc}")
                    if not self.allow_langchain_fallback:
                        raise
            payload = self._decide_with_langchain(context, fallback)
            self.last_mode = "qwen_langchain"
            return payload
        except Exception as exc:  # pragma: no cover - depends on local model environment
            errors.append(f"langchain: {exc}")
            self.last_error = "; ".join(errors) if errors else str(exc)
            if self.strict_backend:
                raise RuntimeError(f"{self.role} backend `{self.backend}` unavailable: {self.last_error}") from exc
            self.last_mode = "heuristic_fallback"
            return self._finalize_payload(fallback, context, fallback)

    def _llm(self):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        from langchain_huggingface import HuggingFacePipeline  # type: ignore

        model_ref = os.environ.get("QWEN_MODEL_PATH", "").strip() or self.model_id
        model_cache_key = (
            model_ref,
            self.local_files_only,
            torch.cuda.is_available(),
        )
        if model_cache_key not in self._model_cache:
            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self._model_cache[model_cache_key] = (tokenizer, model)

        tokenizer, model = self._model_cache[model_cache_key]
        llm_cache_key = (
            model_cache_key,
            self.max_new_tokens,
            self.temperature,
        )
        if llm_cache_key in self._llm_cache:
            return self._llm_cache[llm_cache_key]

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 0.01),
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=generator)
        self._llm_cache[llm_cache_key] = llm
        return llm

    def _format_prompt(self, context: dict[str, Any]) -> str:
        prompt_template = self.prompt_text or "You are an ICS role agent. Reason carefully and answer with strict JSON."
        prompt = {
            "role": self.role,
            "scenario_id": context["scenario_id"],
            "suite": context["suite"],
            "intent": context["intent"],
            "process_id": context.get("process_id"),
            "time_sec": round(float(context["time_sec"]), 3),
            "state_name": context.get("state_name"),
            "candidate_actions": context.get("candidate_actions", []),
            "remaining_playbook": context.get("remaining_playbook", []),
            "recent_observations": context.get("recent_observations", []),
            "recent_actions": context.get("recent_actions", []),
            "recent_decisions": context.get("recent_decisions", []),
            "recent_alerts": context.get("recent_alerts", []),
            "role_goal": context.get("role_goal"),
            "role_constraints": context.get("role_constraints", []),
            "metrics": context.get("metrics"),
            "passed": context.get("passed"),
        }
        return (
            f"{prompt_template}\n\n"
            "Return strict JSON with keys: summary, candidate_id, confidence, diagnosis, labels, decision_type. "
            "`candidate_id` must match one of the listed candidates or be null.\n\n"
            f"Context:\n{json.dumps(prompt, indent=2)}"
        )

    def _decide_with_langchain(self, context: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.output_parsers import StrOutputParser  # type: ignore
        from langchain_core.prompts import PromptTemplate  # type: ignore

        prompt = PromptTemplate.from_template("{body}")
        chain = prompt | self._llm() | StrOutputParser()
        raw = chain.invoke({"body": self._format_prompt(context)})
        payload = self._parse_output(raw)
        return self._finalize_payload(payload, context, fallback)

    def _decide_with_langgraph(self, context: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        from langgraph.graph import END, StateGraph  # type: ignore

        cache_key = (self.role, self.backend)
        if cache_key not in self._graph_cache:
            graph = StateGraph(dict)
            graph.add_node("prepare", lambda state: {"prompt": self._format_prompt(state["context"])})
            graph.add_node("generate", lambda state: {"raw": self._llm().invoke(state["prompt"])})
            graph.add_node("parse", lambda state: {"payload": self._parse_output(state.get("raw", ""))})
            graph.set_entry_point("prepare")
            graph.add_edge("prepare", "generate")
            graph.add_edge("generate", "parse")
            graph.add_edge("parse", END)
            self._graph_cache[cache_key] = graph.compile()

        app = self._graph_cache[cache_key]
        result = app.invoke({"context": context})
        return self._finalize_payload(result.get("payload"), context, fallback)

    @staticmethod
    def _parse_output(raw: Any) -> dict[str, Any] | None:
        text = str(raw)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _finalize_payload(
        self,
        payload: dict[str, Any] | None,
        context: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_lookup = {item["id"]: item for item in context.get("candidate_actions", [])}
        selected = payload or {}
        candidate_id = selected.get("candidate_id")
        action = selected.get("action")
        if candidate_id in candidate_lookup:
            action = candidate_lookup[candidate_id]["action"]
            selected.setdefault("summary", candidate_lookup[candidate_id]["summary"])
        selected["action"] = _normalize_action(action)
        selected.setdefault("labels", fallback.get("labels", ["agentic"]))
        selected.setdefault("confidence", fallback.get("confidence", 0.65))
        selected.setdefault("decision_type", fallback.get("decision_type", "note"))
        selected.setdefault("summary", fallback.get("summary", f"Agentic {self.role} decision"))
        selected.setdefault("diagnosis", fallback.get("diagnosis"))
        if selected["action"] is None and fallback.get("action") is not None:
            selected["action"] = _normalize_action(fallback["action"])
        if selected["action"] is not None and selected.get("decision_type") == "note":
            selected["decision_type"] = "action"
        return selected


class AgenticRoleAgent(RoleAgent):
    def __init__(self, profile_name: str, profile: dict[str, Any]) -> None:
        super().__init__(profile_name, profile)
        self.engine = CachedQwenDecisionEngine(profile, self.role)
        memory = profile.get("memory", {})
        self.observation_window = int(memory.get("observation_window", 8))
        self.decision_window = int(memory.get("decision_window", 8))
        self.action_window = int(memory.get("action_window", 8))
        self.alert_window = int(memory.get("alert_window", 6))
        self.action_cooldown_sec = float(profile.get("action_cooldown_sec", 1.5))
        self.note_cooldown_sec = float(profile.get("note_cooldown_sec", 5.0))
        self.playbook_anticipation_sec = float(profile.get("playbook_anticipation_sec", 0.5))

    def step(self, context: dict[str, Any]) -> list[Any]:
        if self.mode == "offline":
            return []
        if self.role in {"operator", "attacker"}:
            return self._step_action_role(context)
        if self.role in {"observer", "defender"}:
            return self._step_monitoring_role(context)
        if self.role == "planner":
            return self._step_planner(context)
        return []

    def finalize(self, context: dict[str, Any]) -> list[Any]:
        if self.role != "offline_analysis":
            return []
        agent_context = self._build_offline_context(context)
        fallback = self._offline_fallback(agent_context)
        payload = self.engine.decide(agent_context, fallback)
        return [
            self.decision(
                decision_type="analysis",
                summary=str(payload.get("summary", fallback["summary"])),
                time_sec=float(context.get("duration_sec", 0.0)),
                confidence=float(payload.get("confidence", fallback.get("confidence", 0.7))),
                diagnosis=payload.get("diagnosis", fallback.get("diagnosis")),
                labels=list(payload.get("labels", fallback.get("labels", ["offline", "agentic"]))),
                metadata={
                    "backend": self.engine.last_mode,
                    "error": self.engine.last_error,
                    "metrics": context.get("metrics", {}),
                },
            )
        ]

    def _step_action_role(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        fallback = self._goal_fallback(agent_context)
        payload = self.engine.decide(agent_context, fallback)
        return self._payload_to_decisions(payload, agent_context)

    def _step_monitoring_role(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        fallback = self._monitoring_fallback(agent_context)
        payload = self.engine.decide(agent_context, fallback)
        return self._payload_to_decisions(payload, agent_context)

    def _step_planner(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        state_name = agent_context.get("state_name", "UNKNOWN")
        signature = f"planner:{state_name}"
        if not self._allow_signature(signature, float(context["time_sec"]), cooldown=self.note_cooldown_sec):
            return []
        fallback = {
            "summary": f"Planner phase update: plant state is {state_name} and objective is {agent_context['intent']}",
            "decision_type": "note",
            "confidence": 0.7,
            "labels": ["agentic", "planner"],
            "diagnosis": agent_context.get("expected_diagnosis"),
        }
        payload = self.engine.decide(agent_context, fallback)
        return self._payload_to_decisions(payload, agent_context)

    def _build_online_context(self, context: dict[str, Any]) -> dict[str, Any]:
        scenario = context["scenario"]
        recent_observations = [
            self._compact_observation(item)
            for item in context.get("observations", [])[-self.observation_window :]
        ]
        recent_actions = context.get("executed_actions", [])[-self.action_window :]
        recent_decisions = context.get("decisions", [])[-self.decision_window :]
        recent_alerts = context.get("alerts", [])[-self.alert_window :]
        observation = context["observation"]
        remaining_playbook = self._remaining_playbook(context)
        candidate_actions = self._candidate_actions(context, remaining_playbook)
        role_goal = {
            "attacker": scenario.metadata.get("attacker_actions", []),
            "defender": scenario.metadata.get("defender_responses", []),
            "observer": scenario.metadata.get("expected_outcomes", []),
            "planner": scenario.metadata.get("expected_outcomes", []),
            "operator": scenario.metadata.get("expected_outcomes", []),
        }.get(self.role, [])
        return {
            "role": self.role,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "process_id": scenario.execution.get("process_id"),
            "intent": scenario.metadata.get("intent", scenario.execution.get("description", "")),
            "expected_diagnosis": scenario.metadata.get("ground_truth", {}).get("expected_diagnosis"),
            "time_sec": float(context["time_sec"]),
            "state_name": observation.get("STATE_NAME") or observation.get("state_name"),
            "observation": observation,
            "recent_observations": recent_observations,
            "recent_actions": recent_actions,
            "recent_decisions": recent_decisions,
            "recent_alerts": recent_alerts,
            "candidate_actions": candidate_actions,
            "remaining_playbook": remaining_playbook,
            "role_goal": role_goal,
            "role_constraints": [
                "Only choose actions from candidate_actions.",
                "Prefer noop or note if no justified action is needed.",
                "Keep summaries concise and specific to the water-treatment process.",
            ],
        }

    def _build_offline_context(self, context: dict[str, Any]) -> dict[str, Any]:
        scenario = context["scenario"]
        return {
            "role": self.role,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "intent": scenario.metadata.get("intent", scenario.execution.get("description", "")),
            "time_sec": float(context.get("duration_sec", 0.0)),
            "candidate_actions": [],
            "remaining_playbook": [],
            "recent_observations": [self._compact_observation(item) for item in context.get("observations", [])[-self.observation_window :]],
            "recent_actions": context.get("executed_actions", [])[-self.action_window :],
            "recent_decisions": context.get("decisions", [])[-self.decision_window :],
            "recent_alerts": context.get("alerts", [])[-self.alert_window :],
            "role_goal": [
                "Summarize detection, mitigation, residual risk, and likely diagnosis from stored logs only."
            ],
            "role_constraints": ["Offline analysis may not issue live actions."],
            "metrics": context.get("metrics", {}),
            "passed": context.get("passed"),
            "expected_diagnosis": scenario.metadata.get("ground_truth", {}).get("expected_diagnosis"),
        }

    def _remaining_playbook(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        playbook = context.get("playbooks", {}).get(self.role, [])
        role_actions = [
            _normalize_action(item.get("action"))
            for item in context.get("executed_actions", [])
            if item.get("role") == self.role
        ]
        cursor = 0
        for planned in playbook:
            normalized = _normalize_action(planned.get("action"))
            if cursor < len(role_actions) and _action_key(role_actions[cursor]) == _action_key(normalized):
                cursor += 1
                continue
            break
        return playbook[cursor:]

    def _candidate_actions(self, context: dict[str, Any], remaining_playbook: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observation = context["observation"]
        now = float(context["time_sec"])
        process_id = context["scenario"].execution.get("process_id")
        candidates: list[dict[str, Any]] = []

        def add(summary: str, action: dict[str, Any] | str | None) -> None:
            normalized = _normalize_action(action)
            if normalized is None:
                return
            key = _action_key(normalized)
            if any(item["id"] == key for item in candidates):
                return
            candidates.append({"id": key, "summary": summary, "action": normalized})

        for item in remaining_playbook[:2]:
            if now + self.playbook_anticipation_sec >= float(item.get("time_sec", 0)):
                normalized = _normalize_action(item.get("action"))
                if normalized is not None:
                    add(f"Reference playbook action: {normalized.get('type')}", normalized)

        if self.role in {"operator", "attacker"} and not remaining_playbook:
            if observation.get("STATE_NAME") == "IDLE":
                add("Start the plant", {"type": "hmi_start", "process_id": "WT-P1"})
            if observation.get("STATE_NAME") == "RUNNING":
                add("Stop the plant", {"type": "hmi_stop", "process_id": "WT-P1"})

        if self.role == "defender":
            if observation.get("P2_Selected") == 0:
                add("Select Process 2 manual controls", {"process_id": "WT-P2", "type": "select_process", "active": True})
            if observation.get("P2_Selected") == 1:
                add("Deselect Process 2 manual controls", {"process_id": "WT-P2", "type": "select_process", "active": False})
        elif self.role in {"operator", "attacker"} and not remaining_playbook:
            if observation.get("P2_Selected") == 0:
                add("Select Process 2 manual controls", {"process_id": "WT-P2", "type": "select_process", "active": True})
            if observation.get("P2_Selected") == 1:
                add("Deselect Process 2 manual controls", {"process_id": "WT-P2", "type": "select_process", "active": False})

        diagnosis = context["scenario"].metadata.get("ground_truth", {}).get("expected_diagnosis", "")
        if self.role == "attacker" and process_id == "WT-P1" and not remaining_playbook:
            if "high" in diagnosis:
                add("Spoof raw-water tank high-high", {"type": "set_sensor", "name": "RW_Tank_Level", "value": 1210})
            if "low" in diagnosis or "spoof" in diagnosis:
                add("Spoof raw-water tank low", {"type": "set_sensor", "name": "RW_Tank_Level", "value": 390})
            add("Spoof UF tank high", {"type": "set_sensor", "name": "UF_UFFT_Tank_Level", "value": 920})
        if self.role in {"attacker", "operator"} and process_id == "WT-P2" and not remaining_playbook:
            for chemical in ("NaCl", "NaOCl", "HCl"):
                add(
                    f"Open {chemical} manual valve",
                    {"process_id": "WT-P2", "type": "set_manual_valve", "chemical": chemical, "open": True},
                )
                add(
                    f"Close {chemical} manual valve",
                    {"process_id": "WT-P2", "type": "set_manual_valve", "chemical": chemical, "open": False},
                )

        if self.role == "defender":
            if observation.get("Alarm_RW_Tank_HH") or int(observation.get("RW_Tank_Level", 0)) > 1180:
                add("Emergency stop due to high-high raw-water condition", {"type": "hmi_estop", "process_id": "WT-P1"})
            if observation.get("Alarm_RW_Tank_LL") or int(observation.get("RW_Tank_Level", 9999)) < 450:
                add("Stop plant due to low raw-water condition", {"type": "hmi_stop", "process_id": "WT-P1"})
            if observation.get("P2_Selected") == 1:
                add("Deselect Process 2 to constrain unauthorized dosing", {"process_id": "WT-P2", "type": "select_process", "active": False})

        add("Do nothing this cycle", {"type": "noop"})
        return candidates

    def _goal_fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        candidates = [item for item in context.get("candidate_actions", []) if item["action"]["type"] != "noop"]
        selected = candidates[0] if candidates else next(item for item in context["candidate_actions"] if item["action"]["type"] == "noop")
        decision_type = "action" if selected["action"]["type"] != "noop" else "note"
        return {
            "summary": selected["summary"],
            "candidate_id": selected["id"],
            "action": selected["action"],
            "decision_type": decision_type,
            "confidence": 0.72,
            "diagnosis": context.get("expected_diagnosis"),
            "labels": ["agentic", "goal-driven", self.role],
        }

    def _monitoring_fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        observation = context["observation"]
        diagnosis = "nominal_operation"
        summary = f"{self.role} sees nominal operation"
        decision_type = "note"
        candidate_id = next(item["id"] for item in context["candidate_actions"] if item["action"]["type"] == "noop")

        if observation.get("Alarm_RW_Tank_HH") or int(observation.get("RW_Tank_Level", 0)) > 1180:
            diagnosis = "raw_water_alarm_condition"
            summary = f"{self.role} detected a raw-water high-high condition"
            decision_type = "alert"
        elif observation.get("Alarm_RW_Tank_LL") or int(observation.get("RW_Tank_Level", 9999)) < 450:
            diagnosis = "raw_water_low_or_spoof"
            summary = f"{self.role} detected a raw-water low or spoof condition"
            decision_type = "alert"
        elif observation.get("P2_Selected") == 1 and float(context["time_sec"]) > 6:
            diagnosis = "unauthorized_chemical_dosing"
            summary = f"{self.role} detected sustained Process 2 actuation"
            decision_type = "alert"

        if self.role == "defender":
            if diagnosis == "raw_water_alarm_condition":
                candidate_id = _action_key({"type": "hmi_estop", "process_id": "WT-P1"})
            elif diagnosis == "raw_water_low_or_spoof":
                candidate_id = _action_key({"type": "hmi_stop", "process_id": "WT-P1"})
            elif diagnosis == "unauthorized_chemical_dosing":
                candidate_id = _action_key({"process_id": "WT-P2", "type": "select_process", "active": False})

        action = next((item["action"] for item in context["candidate_actions"] if item["id"] == candidate_id), None)

        return {
            "summary": summary,
            "candidate_id": candidate_id,
            "action": action,
            "decision_type": "action" if self.role == "defender" and decision_type == "alert" else decision_type,
            "confidence": 0.78 if diagnosis != "nominal_operation" else 0.6,
            "diagnosis": diagnosis,
            "labels": ["agentic", "monitoring", self.role],
        }

    def _offline_fallback(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics = context.get("metrics", {})
        summary = (
            f"Offline review: passed={context.get('passed')} "
            f"ttd={metrics.get('time_to_detection_sec')} "
            f"ttm={metrics.get('time_to_mitigation_sec')} "
            f"violations={metrics.get('safety_violation_count')}"
        )
        return {
            "summary": summary,
            "decision_type": "analysis",
            "confidence": 0.74,
            "diagnosis": context.get("expected_diagnosis"),
            "labels": ["offline", "agentic", "analysis"],
        }

    def _payload_to_decisions(self, payload: dict[str, Any], context: dict[str, Any]) -> list[Any]:
        summary = str(payload.get("summary", "")).strip()
        action = _normalize_action(payload.get("action"))
        decision_type = str(payload.get("decision_type", "note"))
        now = float(context["time_sec"])

        if action is not None and action.get("type") == "noop" and decision_type == "action":
            decision_type = "note"
            action = None

        signature = f"{decision_type}:{_action_key(action) if action else summary}:{payload.get('diagnosis')}"
        cooldown = self.action_cooldown_sec if action is not None else self.note_cooldown_sec
        if not self._allow_signature(signature, now, cooldown=cooldown):
            return []

        if not summary and action is None:
            return []

        return [
            self.decision(
                decision_type=decision_type,
                summary=summary or f"{self.role} decision",
                time_sec=now,
                action=action,
                confidence=float(payload.get("confidence", 0.65)),
                diagnosis=payload.get("diagnosis"),
                labels=list(payload.get("labels", ["agentic", self.role])),
                metadata={"backend": self.engine.last_mode, "error": self.engine.last_error},
            )
        ]

    def _allow_signature(self, signature: str, now: float, *, cooldown: float) -> bool:
        signatures = self._state.setdefault("signatures", {})
        last = signatures.get(signature)
        if last is not None and now - float(last) < cooldown:
            return False
        signatures[signature] = now
        return True

    @staticmethod
    def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "timestamp_utc",
            "elapsed_sec",
            "STATE_NAME",
            "RW_Tank_Level",
            "UF_UFFT_Tank_Level",
            "ChemTreat_NaCl_Level",
            "ChemTreat_NaOCl_Level",
            "ChemTreat_HCl_Level",
            "RW_Tank_PR_Valve_Sts",
            "ChemTreat_NaCl_Valve_Sts",
            "ChemTreat_NaOCl_Valve_Sts",
            "ChemTreat_HCl_Valve_Sts",
            "P2_Selected",
            "Alarm_RW_Tank_LL",
            "Alarm_RW_Tank_HH",
        ]
        return {field: observation.get(field) for field in fields if field in observation}
