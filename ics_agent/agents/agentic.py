from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from .action_space import action_key, build_candidate_actions, normalize_action
from .base import RoleAgent
from ..knowledge import load_manual_bundle
from ..paths import REPO_ROOT


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
        self.use_langgraph = bool(
            llm_cfg.get("use_langgraph", self.backend == "qwen_langgraph")
        )
        self.strict_backend = bool(llm_cfg.get("strict_backend", True))
        self.allow_langchain_fallback = bool(
            llm_cfg.get("allow_langchain_fallback", True)
        )
        self.local_files_only = bool(llm_cfg.get("local_files_only", True))
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 192))
        self.temperature = float(llm_cfg.get("temperature", 0.0))
        self.quantization = str(llm_cfg.get("quantization", "bf16")).strip().lower()
        self.require_cuda = bool(llm_cfg.get("require_cuda", True))
        self.knowledge_files = list(profile.get("knowledge_files", []))
        self.prompt_text = self._load_prompt()
        self.operational_knowledge = load_manual_bundle(self.knowledge_files)
        self.last_mode = "uninitialized"
        self.last_error: str | None = None
        self.last_raw_output: str | None = None

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

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        self.last_error = None
        if self.backend not in {"qwen_langchain", "qwen_langgraph"}:
            raise RuntimeError(
                f"unsupported backend `{self.backend}` for agentic role `{self.role}`"
            )

        errors: list[str] = []
        try:
            if self.backend == "qwen_langgraph" and self.use_langgraph:
                try:
                    payload = self._decide_with_langgraph(context)
                    self.last_mode = "qwen_langgraph"
                    return payload
                except Exception as exc:
                    errors.append(f"langgraph: {exc}")
                    if not self.allow_langchain_fallback:
                        raise
            payload = self._decide_with_langchain(context)
            self.last_mode = "qwen_langchain"
            return payload
        except (
            Exception
        ) as exc:  # pragma: no cover - depends on local model environment
            if not errors:
                errors.append(f"langchain: {exc}")
            elif "langchain:" not in errors[-1]:
                errors.append(f"langchain: {exc}")
            self.last_error = "; ".join(errors)
            raise RuntimeError(
                f"{self.role} backend `{self.backend}` unavailable: {self.last_error}"
            ) from exc

    def _llm(self):
        import torch  # type: ignore
        from langchain_huggingface import HuggingFacePipeline  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for agentic execution")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected for agentic execution")

        model_ref = os.environ.get("QWEN_MODEL_PATH", "").strip() or self.model_id
        model_cache_key = (
            model_ref,
            self.local_files_only,
            self.quantization,
        )
        if model_cache_key not in self._model_cache:
            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": self.local_files_only,
                "device_map": "auto",
            }
            if self.quantization == "4bit":
                load_kwargs["load_in_4bit"] = True
            elif self.quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
            else:
                if (
                    self.quantization == "bf16"
                    and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                ):
                    load_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
            self._model_cache[model_cache_key] = (tokenizer, model)

        tokenizer, model = self._model_cache[model_cache_key]
        llm_cache_key = (
            model_cache_key,
            self.max_new_tokens,
            self.temperature,
        )
        if llm_cache_key in self._llm_cache:
            return self._llm_cache[llm_cache_key]

        generator_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generator_kwargs["temperature"] = self.temperature

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generator_kwargs,
        )
        llm = HuggingFacePipeline(pipeline=generator)
        self._llm_cache[llm_cache_key] = llm
        return llm

    def _format_prompt(self, context: dict[str, Any]) -> str:
        prompt_template = (
            self.prompt_text
            or "You are an ICS role agent. Reason carefully and answer with strict JSON only."
        )
        payload = {
            "role": self.role,
            "scenario_id": context["scenario_id"],
            "suite": context["suite"],
            "intent": context["intent"],
            "process_id": context.get("process_id"),
            "time_sec": round(float(context["time_sec"]), 3),
            "state_name": context.get("state_name"),
            "observation": context.get("observation"),
            "candidate_actions": context.get("candidate_actions", []),
            "remaining_playbook": context.get("remaining_playbook", []),
            "recent_observations": context.get("recent_observations", []),
            "recent_actions": context.get("recent_actions", []),
            "recent_decisions": context.get("recent_decisions", []),
            "recent_alerts": context.get("recent_alerts", []),
            "active_alarms": context.get("active_alarms", []),
            "active_states": context.get("active_states", []),
            "active_commands": context.get("active_commands", []),
            "short_horizon_trends": context.get("short_horizon_trends", {}),
            "relationship_notes": context.get("relationship_notes", []),
            "risk_flags": context.get("risk_flags", []),
            "incident": context.get("incident"),
            "role_goal": context.get("role_goal"),
            "role_constraints": context.get("role_constraints", []),
            "expected_outcomes": context.get("expected_outcomes", []),
            "safety_bounds": context.get("safety_bounds", {}),
            "metrics": context.get("metrics"),
            "passed": context.get("passed"),
        }
        operational_knowledge = self.operational_knowledge.strip()
        knowledge_block = (
            f"Operational manuals:\n{operational_knowledge}\n\n"
            if operational_knowledge
            else ""
        )
        return (
            f"{prompt_template}\n\n"
            f"{knowledge_block}"
            "Return exactly one JSON object and no prose before or after it.\n"
            "Required keys: summary, candidate_id, confidence, diagnosis, labels, decision_type.\n"
            "Rules:\n"
            "- candidate_id must be either null or match one of candidate_actions[].id.\n"
            "- Never invent a new action or a new candidate id.\n"
            "- Use decision_type of note, alert, action, or analysis.\n"
            "- diagnosis must be null or a short snake_case style label, never a sentence.\n"
            "- Infer any diagnosis from the provided telemetry and history only.\n"
            "- If the evidence is weak, set diagnosis to null instead of guessing.\n"
            "- If you want to hold position, prefer the noop candidate when available.\n\n"
            "Example JSON:\n"
            '{"summary":"Monitor and hold.","candidate_id":null,"confidence":0.42,"diagnosis":null,"labels":["agentic"],"decision_type":"note"}\n\n'
            f"Context:\n{json.dumps(payload, indent=2)}"
        )

    def _decide_with_langchain(self, context: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.output_parsers import StrOutputParser  # type: ignore
        from langchain_core.prompts import PromptTemplate  # type: ignore

        prompt = PromptTemplate.from_template("{body}")
        chain = prompt | self._llm() | StrOutputParser()
        raw = chain.invoke({"body": self._format_prompt(context)})
        self.last_raw_output = self._raw_excerpt(raw)
        payload = self._parse_output(raw)
        if payload is None:
            raise RuntimeError(
                f"model output did not contain valid JSON: {self.last_raw_output}"
            )
        return self._finalize_payload(payload, context)

    def _decide_with_langgraph(self, context: dict[str, Any]) -> dict[str, Any]:
        from langgraph.graph import END, StateGraph  # type: ignore

        cache_key = (self.role, self.backend)
        if cache_key not in self._graph_cache:
            graph = StateGraph(dict)
            graph.add_node(
                "prepare",
                lambda state: {"prompt": self._format_prompt(state["context"])},
            )
            graph.add_node(
                "generate", lambda state: {"raw": self._llm().invoke(state["prompt"])}
            )
            graph.set_entry_point("prepare")
            graph.add_edge("prepare", "generate")
            graph.add_edge("generate", END)
            self._graph_cache[cache_key] = graph.compile()

        app = self._graph_cache[cache_key]
        result = app.invoke({"context": context})
        raw = result.get("raw", "")
        self.last_raw_output = self._raw_excerpt(raw)
        payload = self._parse_output(raw)
        if payload is None:
            raise RuntimeError(
                f"model output did not contain valid JSON: {self.last_raw_output}"
            )
        return self._finalize_payload(payload, context)

    @staticmethod
    def _parse_output(raw: Any) -> dict[str, Any] | None:
        import yaml

        text = textwrap.dedent(str(raw)).strip()
        if not text:
            return None

        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(candidate: str) -> None:
            candidate = textwrap.dedent(candidate).strip()
            if not candidate or candidate in seen:
                return
            seen.add(candidate)
            candidates.append(candidate)

        add_candidate(text)
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        cleaned = (
            cleaned.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        add_candidate(cleaned)

        for block in re.findall(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            add_candidate(block)

        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return payload[0]

            try:
                payload = yaml.safe_load(candidate)
            except yaml.YAMLError:
                payload = None
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return payload[0]

            try:
                payload = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                payload = None
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return payload[0]

            for index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
        return None

    @staticmethod
    def _raw_excerpt(raw: Any, *, limit: int = 320) -> str:
        text = re.sub(r"\s+", " ", str(raw)).strip()
        if len(text) <= limit:
            return text
        return f"{text[: limit - 3]}..."

    def _finalize_payload(
        self,
        payload: dict[str, Any] | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if payload is None:
            raise RuntimeError("model output did not contain valid JSON")

        selected = dict(payload)
        candidate_lookup = {
            item["id"]: item for item in context.get("candidate_actions", [])
        }
        action = normalize_action(selected.get("action"))
        candidate_id = selected.get("candidate_id")

        if candidate_id in candidate_lookup:
            matched = candidate_lookup[str(candidate_id)]
            action = matched["action"]
            selected["candidate_id"] = matched["id"]
            selected.setdefault("summary", matched["summary"])
        elif action is not None and candidate_lookup:
            key = action_key(action)
            matched = next(
                (item for item in candidate_lookup.values() if item["id"] == key), None
            )
            if matched is None:
                raise RuntimeError(
                    "model selected an action outside the candidate list"
                )
            action = matched["action"]
            selected["candidate_id"] = matched["id"]
            selected.setdefault("summary", matched["summary"])

        selected["action"] = action
        selected.setdefault("summary", f"Agentic {self.role} assessment")
        selected.setdefault("confidence", 0.55)
        selected["diagnosis"] = self._normalize_diagnosis(selected.get("diagnosis"))
        labels = selected.get("labels", ["agentic", self.role])
        if not isinstance(labels, list):
            labels = [str(labels)]
        selected["labels"] = [str(label) for label in labels]
        selected.setdefault(
            "decision_type",
            "analysis" if self.role == "offline_analysis" else "note",
        )
        if selected["decision_type"] not in {"note", "alert", "action", "analysis"}:
            selected["decision_type"] = (
                "analysis" if self.role == "offline_analysis" else "note"
            )

        if selected["action"] is not None and selected["action"].get("type") == "noop":
            selected["action"] = None
            if selected["decision_type"] == "action":
                selected["decision_type"] = "note"

        return selected

    @staticmethod
    def _normalize_diagnosis(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text or text in {"none", "null", "unknown"}:
            return None

        word_count = len(re.findall(r"[a-z0-9]+", text))
        if len(text) > 64 or word_count > 8 or any(char in text for char in ".!?;"):
            return None

        normalized = re.sub(r"[\s/-]+", "_", text)
        normalized = re.sub(r"[^a-z0-9_]", "", normalized).strip("_")
        if not normalized or len(normalized) > 64:
            return None
        return normalized


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
        self.playbook_anticipation_sec = float(
            profile.get("playbook_anticipation_sec", 0.5)
        )
        self.reasoning_cooldown_sec = float(profile.get("reasoning_cooldown_sec", 2.0))
        self.reasoning_refresh_sec = float(profile.get("reasoning_refresh_sec", 6.0))

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
        try:
            payload = self.engine.decide(agent_context)
        except Exception:
            return [
                self.decision(
                    decision_type="analysis",
                    summary="Agentic offline analysis unavailable; recorded metrics only.",
                    time_sec=float(context.get("duration_sec", 0.0)),
                    confidence=0.0,
                    diagnosis=None,
                    labels=["offline", "agentic", "backend-error"],
                    metadata={
                        "backend": self.engine.last_mode,
                        "error": self.engine.last_error,
                        "metrics": context.get("metrics", {}),
                    },
                )
            ]
        return [
            self.decision(
                decision_type="analysis",
                summary=str(payload.get("summary", "Agentic offline analysis")),
                time_sec=float(context.get("duration_sec", 0.0)),
                confidence=float(payload.get("confidence", 0.7)),
                diagnosis=payload.get("diagnosis"),
                labels=list(payload.get("labels", ["offline", "agentic"])),
                metadata={
                    "backend": self.engine.last_mode,
                    "error": self.engine.last_error,
                    "metrics": context.get("metrics", {}),
                },
            )
        ]

    def _step_action_role(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        if self._due_playbook_action(agent_context) is None and self._only_noop_candidates(
            agent_context
        ):
            return []
        if not self._should_reason(agent_context, channel="action"):
            return self._due_playbook_fallback(agent_context, reason="reasoning_cooldown")
        try:
            payload = self.engine.decide(agent_context)
        except Exception:
            return self._due_playbook_fallback(agent_context, reason="backend_error")
        decisions = self._payload_to_decisions(payload, agent_context)
        if any(decision.action is not None for decision in decisions):
            return decisions
        fallback = self._due_playbook_fallback(
            agent_context, reason="due_playbook_action"
        )
        return fallback or decisions

    def _step_monitoring_role(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        if not self._should_monitor(agent_context):
            return []
        if not self._should_reason(agent_context, channel="monitor"):
            return self._apply_defender_guardrail(
                agent_context,
                [],
                reason="reasoning_cooldown",
            )
        try:
            payload = self.engine.decide(agent_context)
        except Exception:
            return self._apply_defender_guardrail(
                agent_context,
                [],
                reason="backend_error",
            )
        decisions = self._payload_to_decisions(payload, agent_context)
        return self._apply_defender_guardrail(
            agent_context,
            decisions,
            reason="last_mile_containment",
        )

    def _step_planner(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        if not self._should_plan(agent_context):
            return []
        if not self._should_reason(agent_context, channel="planner"):
            return []
        try:
            payload = self.engine.decide(agent_context)
        except Exception:
            return []
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
        telemetry_summary = self._summarize_telemetry(
            observation=observation,
            recent_observations=context.get("observations", []),
        )
        incident = self._incident_context(
            context=context,
            telemetry_summary=telemetry_summary,
        )
        candidate_actions = self._candidate_actions(
            context,
            remaining_playbook,
            telemetry_summary=telemetry_summary,
            incident=incident,
        )
        return {
            "role": self.role,
            "scenario_id": self._visible_scenario_id(scenario),
            "suite": scenario.suite,
            "process_id": scenario.execution.get("process_id"),
            "intent": self._live_intent(scenario),
            "time_sec": float(context["time_sec"]),
            "state_name": observation.get("STATE_NAME")
            or observation.get("state_name"),
            "observation": self._compact_observation(observation),
            "recent_observations": recent_observations,
            "recent_actions": recent_actions,
            "recent_decisions": recent_decisions,
            "recent_alerts": recent_alerts,
            "active_alarms": telemetry_summary["active_alarms"],
            "active_states": telemetry_summary["active_states"],
            "active_commands": telemetry_summary["active_commands"],
            "short_horizon_trends": telemetry_summary["short_horizon_trends"],
            "relationship_notes": telemetry_summary["relationship_notes"],
            "risk_flags": telemetry_summary["risk_flags"],
            "incident": incident,
            "candidate_actions": candidate_actions,
            "remaining_playbook": remaining_playbook,
            "role_goal": self._live_role_goals(scenario),
            "role_constraints": [
                "Use recent telemetry and state to justify the decision.",
                "Choose only from candidate_actions or emit a note/alert with no live action.",
                "Do not assume hidden attack plans, ground truth labels, or future timeline steps.",
                "Keep summaries concise and specific to the water-treatment environment.",
                "Escalate incidents from suspected to confirmed to containing when evidence persists or worsens.",
            ],
            "expected_outcomes": self._live_expected_outcomes(scenario),
            "safety_bounds": scenario.metadata.get("safety_bounds", {}),
        }

    def _build_offline_context(self, context: dict[str, Any]) -> dict[str, Any]:
        scenario = context["scenario"]
        return {
            "role": self.role,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "intent": scenario.metadata.get(
                "intent", scenario.execution.get("description", "")
            ),
            "time_sec": float(context.get("duration_sec", 0.0)),
            "candidate_actions": [],
            "remaining_playbook": [],
            "recent_observations": [
                self._compact_observation(item)
                for item in context.get("observations", [])[-self.observation_window :]
            ],
            "recent_actions": context.get("executed_actions", [])[
                -self.action_window :
            ],
            "recent_decisions": context.get("decisions", [])[-self.decision_window :],
            "recent_alerts": context.get("alerts", [])[-self.alert_window :],
            "role_goal": [
                "Summarize detection, mitigation, residual risk, and likely diagnosis from stored logs only."
            ],
            "role_constraints": ["Offline analysis may not issue live actions."],
            "metrics": context.get("metrics", {}),
            "passed": context.get("passed"),
            "expected_outcomes": scenario.metadata.get("expected_outcomes", []),
            "safety_bounds": scenario.metadata.get("safety_bounds", {}),
        }

    def _visible_scenario_id(self, scenario: Any) -> str:
        if self.role in {"attacker", "offline_analysis"}:
            return scenario.scenario_id
        process_id = str(scenario.execution.get("process_id", "WT")).lower()
        return f"{scenario.suite}-{process_id}-live"

    def _live_intent(self, scenario: Any) -> str:
        process_id = scenario.execution.get("process_id", "WT-P1")
        role_intents = {
            "attacker": scenario.metadata.get(
                "intent", scenario.execution.get("description", "")
            ),
            "defender": (
                f"Monitor {process_id} telemetry for unsafe or suspicious behavior and "
                "apply the least disruptive effective mitigation when warranted."
            ),
            "observer": (
                f"Watch {process_id} telemetry for anomalies, explain likely causes, "
                "and raise timely evidence-based alerts."
            ),
            "planner": (
                f"Summarize the current {process_id} operating picture, plant risk, "
                "and useful near-term focus areas without predicting hidden events."
            ),
            "operator": (
                f"Operate {process_id} safely, follow the due playbook actions, and "
                "avoid unnecessary disruption."
            ),
        }
        return role_intents.get(self.role, "Monitor the plant safely.")

    def _live_role_goals(self, scenario: Any) -> list[str]:
        process_id = scenario.execution.get("process_id", "WT-P1")
        goals = {
            "attacker": scenario.metadata.get("attacker_actions", []),
            "defender": [
                f"Use current {process_id} telemetry, alarms, and action history to justify any intervention.",
                "Track incident state across turns and escalate from suspected to confirmed to containing.",
                "Prefer Process 2 deselection or a controlled stop before estop when both are viable.",
            ],
            "observer": [
                f"Describe anomalies in {process_id} using grounded plant evidence.",
                "Alert early when telemetry or command-state relationships look unsafe or implausible.",
            ],
            "planner": [
                "Summarize state transitions, alarms, and risk trends that matter for the next few seconds.",
            ],
            "operator": [
                "Execute due operator actions only when they are currently scheduled or clearly safe.",
                "Use hold position when the plant needs more evidence before acting.",
            ],
        }
        return goals.get(self.role, [])

    def _live_expected_outcomes(self, scenario: Any) -> list[str]:
        process_id = scenario.execution.get("process_id", "WT-P1")
        expected = {
            "attacker": scenario.metadata.get("expected_outcomes", []),
            "defender": [
                f"{process_id} remains in a safe operating envelope or is safely shut down.",
                "Unsafe manual dosing or raw-water anomalies are contained quickly once evidence persists or worsens.",
            ],
            "observer": [
                "Alerts and diagnoses stay tied to current telemetry and recent actions.",
            ],
            "planner": [
                "The plan notes near-term risks, active alarms, and likely consequences of recent commands.",
            ],
            "operator": [
                f"{process_id} follows the allowed playbook without unnecessary unsafe transitions.",
            ],
        }
        return expected.get(self.role, [])

    def _summarize_telemetry(
        self,
        *,
        observation: dict[str, Any],
        recent_observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        active_alarms = self._active_flags(observation, prefix="Alarm_")
        active_states = self._active_flags(observation, prefix="SYS_")
        active_commands = self._active_commands(observation)
        short_horizon_trends = self._short_horizon_trends(recent_observations)
        relationship_notes = self._derive_relationship_notes(
            observation,
            short_horizon_trends,
        )
        risk_flags = self._derive_risk_flags(
            observation,
            active_alarms,
            relationship_notes,
            short_horizon_trends,
        )
        return {
            "active_alarms": active_alarms,
            "active_states": active_states,
            "active_commands": active_commands,
            "short_horizon_trends": short_horizon_trends,
            "relationship_notes": relationship_notes,
            "risk_flags": risk_flags,
        }

    def _incident_context(
        self,
        *,
        context: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.role != "defender":
            return None

        now = float(context["time_sec"])
        recent_alerts = context.get("alerts", [])[-self.alert_window :]
        recent_actions = context.get("executed_actions", [])[-self.action_window :]
        diagnoses = [
            str(item.get("diagnosis"))
            for item in context.get("decisions", [])[-self.decision_window :]
            if item.get("role") in {"observer", "defender"} and item.get("diagnosis")
        ]
        incident_type = self._classify_incident(
            risk_flags=telemetry_summary["risk_flags"],
            active_alarms=telemetry_summary["active_alarms"],
            relationship_notes=telemetry_summary["relationship_notes"],
            diagnoses=diagnoses,
        )
        state = self._state.setdefault(
            "incident",
            {
                "type": "none",
                "status": "none",
                "severity": 0,
                "diagnosis": None,
                "recommended_action": None,
                "first_seen_time_sec": None,
                "last_seen_time_sec": None,
                "persistence_sec": 0.0,
                "evidence_count": 0,
                "observer_alert_count": 0,
                "attacker_action_count": 0,
                "defender_action_count": 0,
                "new_evidence": [],
                "evidence_tokens": [],
            },
        )

        if incident_type == "none":
            if state.get("status") in {"confirmed", "containing", "suspected"}:
                state.update(
                    {
                        "status": "stabilized",
                        "severity": 0,
                        "recommended_action": None,
                        "new_evidence": [],
                        "evidence_tokens": [],
                        "last_seen_time_sec": now,
                    }
                )
                return dict(state)
            if state.get("status") == "stabilized" and state.get("last_seen_time_sec") is not None:
                if now - float(state["last_seen_time_sec"]) <= 10.0:
                    return dict(state)
            self._state["incident"] = {
                "type": "none",
                "status": "none",
                "severity": 0,
                "diagnosis": None,
                "recommended_action": None,
                "first_seen_time_sec": None,
                "last_seen_time_sec": None,
                "persistence_sec": 0.0,
                "evidence_count": 0,
                "observer_alert_count": 0,
                "attacker_action_count": 0,
                "defender_action_count": 0,
                "new_evidence": [],
                "evidence_tokens": [],
            }
            return dict(self._state["incident"])

        observer_alert_count = sum(
            1
            for item in recent_alerts
            if item.get("role") == "observer"
            and self._incident_matches(
                incident_type,
                item.get("diagnosis"),
                item.get("summary"),
            )
        )
        attacker_action_count = sum(
            1
            for item in recent_actions
            if item.get("role") == "attacker"
            and self._coerce_number(item.get("time_sec")) is not None
            and now - float(item.get("time_sec", 0.0)) <= 12.0
        )
        defender_action_count = sum(
            1
            for item in recent_actions
            if item.get("role") == "defender"
            and self._is_relevant_defender_action(
                item.get("action"),
                incident_type=incident_type,
            )
            and self._coerce_number(item.get("time_sec")) is not None
            and now - float(item.get("time_sec", 0.0)) <= 12.0
        )
        severity = self._incident_severity(
            incident_type=incident_type,
            observation=context.get("observation", {}),
            telemetry_summary=telemetry_summary,
        )
        diagnosis = self._incident_diagnosis(incident_type=incident_type)
        recommended_action = self._incident_recommended_action(
            incident_type=incident_type,
            severity=severity,
            observation=context.get("observation", {}),
            telemetry_summary=telemetry_summary,
        )
        evidence_tokens = self._incident_evidence_tokens(
            incident_type=incident_type,
            telemetry_summary=telemetry_summary,
            observer_alert_count=observer_alert_count,
            attacker_action_count=attacker_action_count,
        )

        if state.get("type") != incident_type:
            first_seen_time_sec = now
            evidence_count = 1
            new_evidence = sorted(evidence_tokens)
        else:
            first_seen_time_sec = state.get("first_seen_time_sec")
            if first_seen_time_sec is None:
                first_seen_time_sec = now
            previous_tokens = set(state.get("evidence_tokens", []))
            new_evidence = sorted(set(evidence_tokens).difference(previous_tokens))
            evidence_count = int(state.get("evidence_count", 0)) + 1

        persistence_sec = round(now - float(first_seen_time_sec), 3)
        status = "suspected"
        if defender_action_count:
            status = "containing"
        elif (
            severity >= 3
            or observer_alert_count > 0
            or persistence_sec >= 6.0
            or (incident_type == "process2_manual_dosing" and attacker_action_count > 0)
        ):
            status = "confirmed"

        state.update(
            {
                "type": incident_type,
                "status": status,
                "severity": severity,
                "diagnosis": diagnosis,
                "recommended_action": recommended_action,
                "first_seen_time_sec": first_seen_time_sec,
                "last_seen_time_sec": now,
                "persistence_sec": persistence_sec,
                "evidence_count": evidence_count,
                "observer_alert_count": observer_alert_count,
                "attacker_action_count": attacker_action_count,
                "defender_action_count": defender_action_count,
                "new_evidence": new_evidence,
                "evidence_tokens": sorted(evidence_tokens),
            }
        )
        return dict(state)

    def _classify_incident(
        self,
        *,
        risk_flags: list[str],
        active_alarms: list[str],
        relationship_notes: list[str],
        diagnoses: list[str],
    ) -> str:
        diagnosis_set = set(diagnoses)
        if (
            "unauthorized_chemical_dosing" in diagnosis_set
            or "process2_manual_path_active" in risk_flags
            or "unauthorized_manual_dosing_possible" in risk_flags
            or any(flag.endswith("_depleting") for flag in risk_flags)
            or "process2_selected" in relationship_notes
            or "process2_manual_dosing_active" in relationship_notes
        ):
            return "process2_manual_dosing"
        if (
            "raw_water_alarm_condition" in diagnosis_set
            or "high_high_overflow_risk" in diagnosis_set
            or "overflow_risk" in risk_flags
            or "raw_water_high_high_band" in risk_flags
            or "Alarm_RW_Tank_HH" in active_alarms
        ):
            return "raw_water_overflow"
        if (
            "raw_water_low_or_spoof" in diagnosis_set
            or "raw_water_protection_risk" in risk_flags
            or "raw_water_low_band" in risk_flags
            or "Alarm_RW_Tank_LL" in active_alarms
        ):
            return "raw_water_low_protection"
        if active_alarms or "sensor_or_process_jump" in risk_flags:
            return "telemetry_anomaly"
        return "none"

    def _incident_severity(
        self,
        *,
        incident_type: str,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> int:
        risk_flags = telemetry_summary["risk_flags"]
        relationship_notes = telemetry_summary["relationship_notes"]
        pump_running = self._is_active_flag(observation.get("RW_Pump_Sts"))
        if incident_type == "process2_manual_dosing":
            if any(flag.endswith("_depleting") for flag in risk_flags):
                return 4 if sum(flag.endswith("_depleting") for flag in risk_flags) >= 2 else 3
            if "process2_manual_dosing_active" in relationship_notes:
                return 3
            if "process2_selected" in relationship_notes:
                return 2
            return 1
        if incident_type == "raw_water_overflow":
            if "command_status_divergence" in risk_flags and pump_running:
                return 4
            return 3 if pump_running else 2
        if incident_type == "raw_water_low_protection":
            return 3 if pump_running else 2
        if incident_type == "telemetry_anomaly":
            return 2 if telemetry_summary["active_alarms"] else 1
        return 0

    @staticmethod
    def _incident_diagnosis(*, incident_type: str) -> str | None:
        mapping = {
            "process2_manual_dosing": "unauthorized_chemical_dosing",
            "raw_water_overflow": "raw_water_alarm_condition",
            "raw_water_low_protection": "raw_water_low_or_spoof",
            "telemetry_anomaly": "sensor_or_process_jump",
        }
        return mapping.get(incident_type)

    def _incident_recommended_action(
        self,
        *,
        incident_type: str,
        severity: int,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> str | None:
        if incident_type == "process2_manual_dosing":
            if severity >= 3:
                return "select_process"
            return None
        if incident_type == "raw_water_overflow":
            if severity >= 4:
                return "hmi_estop"
            return "hmi_stop"
        if incident_type == "raw_water_low_protection":
            return "hmi_stop"
        if incident_type == "telemetry_anomaly" and telemetry_summary["active_alarms"]:
            return "hmi_stop"
        return None

    def _incident_evidence_tokens(
        self,
        *,
        incident_type: str,
        telemetry_summary: dict[str, Any],
        observer_alert_count: int,
        attacker_action_count: int,
    ) -> set[str]:
        tokens = {incident_type}
        tokens.update(telemetry_summary["active_alarms"])
        tokens.update(telemetry_summary["risk_flags"])
        if observer_alert_count:
            tokens.add("observer_alerted")
        if attacker_action_count:
            tokens.add("recent_attacker_action")
        return tokens

    def _incident_matches(
        self,
        incident_type: str,
        diagnosis: Any,
        summary: Any,
    ) -> bool:
        diagnosis_text = str(diagnosis or "")
        summary_text = str(summary or "").lower()
        if incident_type == "process2_manual_dosing":
            return diagnosis_text == "unauthorized_chemical_dosing" or "process 2" in summary_text
        if incident_type == "raw_water_overflow":
            return diagnosis_text in {"raw_water_alarm_condition", "high_high_overflow_risk"} or "high-high" in summary_text
        if incident_type == "raw_water_low_protection":
            return diagnosis_text == "raw_water_low_or_spoof" or "raw-water" in summary_text
        return bool(diagnosis_text)

    def _is_relevant_defender_action(
        self,
        action: Any,
        *,
        incident_type: str,
    ) -> bool:
        normalized = normalize_action(action)
        if normalized is None:
            return False
        action_type = str(normalized.get("type"))
        if incident_type == "process2_manual_dosing":
            return action_type == "select_process" and not normalized.get("active", True)
        if incident_type in {"raw_water_overflow", "raw_water_low_protection", "telemetry_anomaly"}:
            return action_type in {"hmi_stop", "hmi_estop"}
        return False

    def _active_flags(self, observation: dict[str, Any], *, prefix: str) -> list[str]:
        return sorted(
            key
            for key, value in observation.items()
            if key.startswith(prefix) and self._is_active_flag(value)
        )

    def _active_commands(self, observation: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        for key, value in observation.items():
            if key.endswith("_Cmd") and self._is_active_flag(value):
                keys.append(key)
            if key in {
                "HMI_Start_PB",
                "HMI_Stop_PB",
                "HMI_Start_Active",
                "HMI_Stop_Active",
                "HMI_Reset_PB",
                "P2_Selected",
            } and self._is_active_flag(value):
                keys.append(key)
        return sorted(keys)

    def _short_horizon_trends(
        self, observations: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        fields = [
            "RW_Tank_Level",
            "UF_UFFT_Tank_Level",
            "ChemTreat_NaCl_Level",
            "ChemTreat_NaOCl_Level",
            "ChemTreat_HCl_Level",
            "RW_Pump_Flow",
            "RW_Pump_Speed_Cmd",
        ]
        if not observations:
            return {}

        history = observations[-max(5, self.observation_window) :]
        trends: dict[str, dict[str, Any]] = {}
        for field in fields:
            points = [
                self._coerce_number(item.get(field))
                for item in history
                if self._coerce_number(item.get(field)) is not None
            ]
            if not points:
                continue
            current = points[-1]
            previous = points[-2] if len(points) >= 2 else current
            baseline = points[-4] if len(points) >= 4 else points[0]
            delta_1 = round(current - previous, 3)
            delta_window = round(current - baseline, 3)
            direction = "stable"
            if delta_window > 0.5:
                direction = "rising"
            elif delta_window < -0.5:
                direction = "falling"
            trends[field] = {
                "current": round(current, 3),
                "delta_1": delta_1,
                "delta_window": delta_window,
                "direction": direction,
            }
        return trends

    def _derive_relationship_notes(
        self,
        observation: dict[str, Any],
        trends: dict[str, dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if self._is_active_flag(observation.get("Alarm_RW_Tank_LL")):
            notes.append("raw_water_low_low_alarm_active")
        if self._is_active_flag(observation.get("Alarm_RW_Tank_HH")):
            notes.append("raw_water_high_high_alarm_active")

        if self._is_active_flag(observation.get("P2_Selected")):
            notes.append("process2_selected")

        chemical_status_fields = [
            "ChemTreat_NaCl_Valve_Sts",
            "ChemTreat_NaOCl_Valve_Sts",
            "ChemTreat_HCl_Valve_Sts",
        ]
        if any(self._is_active_flag(observation.get(field)) for field in chemical_status_fields):
            notes.append("process2_manual_dosing_active")

        for cmd_field, sts_field in (
            ("RW_Tank_PR_Valve_Cmd", "RW_Tank_PR_Valve_Sts"),
            ("RW_Tank_P_Valve_Cmd", "RW_Tank_P_Valve_Sts"),
            ("ChemTreat_NaCl_Valve_Cmd", "ChemTreat_NaCl_Valve_Sts"),
            ("ChemTreat_NaOCl_Valve_Cmd", "ChemTreat_NaOCl_Valve_Sts"),
            ("ChemTreat_HCl_Valve_Cmd", "ChemTreat_HCl_Valve_Sts"),
        ):
            if cmd_field not in observation or sts_field not in observation:
                continue
            if self._is_active_flag(observation.get(cmd_field)) != self._is_active_flag(
                observation.get(sts_field)
            ):
                notes.append(f"command_status_mismatch:{cmd_field}->{sts_field}")

        if (
            self._is_active_flag(observation.get("RW_Pump_Sts"))
            and self._coerce_number(observation.get("RW_Tank_Level")) is not None
            and self._coerce_number(observation.get("RW_Tank_Level")) <= 450.0
        ):
            notes.append("pump_running_while_raw_water_low")

        for field in (
            "RW_Tank_Level",
            "UF_UFFT_Tank_Level",
            "ChemTreat_NaCl_Level",
            "ChemTreat_NaOCl_Level",
            "ChemTreat_HCl_Level",
        ):
            trend = trends.get(field)
            if trend and abs(float(trend.get("delta_1", 0.0))) >= 80.0:
                notes.append(f"abrupt_change:{field}")

        return notes

    def _derive_risk_flags(
        self,
        observation: dict[str, Any],
        active_alarms: list[str],
        relationship_notes: list[str],
        trends: dict[str, dict[str, Any]],
    ) -> list[str]:
        risks: list[str] = []
        if active_alarms:
            risks.append("alarm_active")
        if "raw_water_low_low_alarm_active" in relationship_notes:
            risks.append("raw_water_protection_risk")
        if "raw_water_high_high_alarm_active" in relationship_notes:
            risks.append("overflow_risk")
        if "process2_selected" in relationship_notes:
            risks.append("process2_manual_path_active")
        if "process2_manual_dosing_active" in relationship_notes:
            risks.append("unauthorized_manual_dosing_possible")
        if any(note.startswith("command_status_mismatch:") for note in relationship_notes):
            risks.append("command_status_divergence")
        if any(note.startswith("abrupt_change:") for note in relationship_notes):
            risks.append("sensor_or_process_jump")

        rw_level = self._coerce_number(observation.get("RW_Tank_Level"))
        if rw_level is not None and rw_level <= 450.0:
            risks.append("raw_water_low_band")
        if rw_level is not None and rw_level >= 1180.0:
            risks.append("raw_water_high_high_band")

        for field in ("ChemTreat_NaCl_Level", "ChemTreat_NaOCl_Level", "ChemTreat_HCl_Level"):
            trend = trends.get(field)
            if trend and trend.get("direction") == "falling":
                risks.append(f"{field}_depleting")

        return sorted(set(risks))

    @staticmethod
    def _is_active_flag(value: Any) -> bool:
        numeric = AgenticRoleAgent._coerce_number(value)
        if numeric is not None:
            return numeric >= 1.0
        return str(value).strip().lower() in {"true", "on", "active", "running"}

    def _remaining_playbook(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        playbook = context.get("playbooks", {}).get(self.role, [])
        role_actions = [
            normalize_action(item.get("action"))
            for item in context.get("executed_actions", [])
            if item.get("role") == self.role
        ]
        now = float(context.get("time_sec", 0.0))
        cursor = 0
        for planned in playbook:
            normalized = normalize_action(planned.get("action"))
            if cursor < len(role_actions) and action_key(
                role_actions[cursor]
            ) == action_key(normalized):
                cursor += 1
                continue
            if float(planned.get("time_sec", 0.0)) <= now and self._action_satisfied(
                normalized,
                observation=context.get("observation", {}),
            ):
                cursor += 1
                continue
            break
        return playbook[cursor:]

    def _action_satisfied(
        self,
        action: dict[str, Any] | None,
        *,
        observation: dict[str, Any],
    ) -> bool:
        if action is None:
            return False

        action_type = str(action.get("type", ""))
        state_name = str(
            observation.get("STATE_NAME") or observation.get("state_name") or ""
        ).upper()

        if action_type == "hmi_start":
            return state_name == "RUNNING"
        if action_type in {"hmi_stop", "hmi_estop"}:
            return state_name == "SHUTDOWN"
        if action_type == "hmi_reset":
            return state_name == "IDLE"
        if action_type == "select_process":
            return self._is_active_flag(observation.get("P2_Selected")) == bool(
                action.get("active", True)
            )
        if action_type == "set_manual_valve":
            chemical = str(action.get("chemical", "")).strip()
            if not chemical:
                return False
            field = f"ChemTreat_{chemical}_Valve_Sts"
            if field not in observation:
                return False
            return self._is_active_flag(observation.get(field)) == bool(
                action.get("open", True)
            )
        return False

    def _candidate_actions(
        self,
        context: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        *,
        telemetry_summary: dict[str, Any],
        incident: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self.role in {"operator", "attacker"}:
            return self._strict_playbook_candidates(
                remaining_playbook, now=float(context["time_sec"])
            )
        if (
            self.role == "attacker"
            and not remaining_playbook
            and not context["scenario"].metadata.get("attacker_actions")
        ):
            noop_candidate = self._noop_candidate(
                "No attacker objective is active in this scenario."
            )
            return [noop_candidate] if noop_candidate is not None else []
        return build_candidate_actions(
            role=self.role,
            allowed_action_types=list(self.profile.get("allowed_action_types", [])),
            scenario=context["scenario"],
            observation=context["observation"],
            risk_flags=telemetry_summary["risk_flags"],
            active_alarms=telemetry_summary["active_alarms"],
            relationship_notes=telemetry_summary["relationship_notes"],
            recent_alerts=context.get("alerts", [])[-self.alert_window :],
            incident=incident,
            remaining_playbook=remaining_playbook,
            now=float(context["time_sec"]),
            playbook_anticipation_sec=self.playbook_anticipation_sec,
        )

    def _strict_playbook_candidates(
        self, remaining_playbook: list[dict[str, Any]], *, now: float
    ) -> list[dict[str, Any]]:
        allowed = set(self.profile.get("allowed_action_types", []))
        candidates: list[dict[str, Any]] = []

        for item in remaining_playbook[:3]:
            if float(item.get("time_sec", 0.0)) > now:
                break
            action = normalize_action(item.get("action"))
            if action is None or str(action.get("type")) not in allowed:
                continue
            candidates.append(
                {
                    "id": action_key(action),
                    "summary": f"Execute the due playbook action `{action.get('type', 'unknown')}`.",
                    "action": action,
                    "source": "playbook",
                }
            )

        noop_candidate = self._noop_candidate(
            "Wait until the next scheduled playbook action is due."
        )
        if noop_candidate is not None:
            candidates.append(noop_candidate)
        return candidates

    def _noop_candidate(self, summary: str) -> dict[str, Any] | None:
        allowed = set(self.profile.get("allowed_action_types", []))
        if "noop" not in allowed:
            return None
        return {
            "id": action_key({"type": "noop"}),
            "summary": summary,
            "action": {"type": "noop"},
            "source": "action_space",
        }

    def _due_playbook_action(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if self.role not in {"operator", "attacker"}:
            return None
        remaining_playbook = context.get("remaining_playbook", [])
        if not remaining_playbook:
            return None
        next_event = remaining_playbook[0]
        if float(next_event.get("time_sec", 0.0)) > float(context["time_sec"]):
            return None
        action = normalize_action(next_event.get("action"))
        if action is None:
            return None
        allowed = set(self.profile.get("allowed_action_types", []))
        if str(action.get("type")) not in allowed:
            return None
        return action

    def _due_playbook_fallback(
        self, context: dict[str, Any], *, reason: str
    ) -> list[Any]:
        action = self._due_playbook_action(context)
        if action is None:
            return []
        action_type = str(action.get("type", "unknown"))
        return [
            self.decision(
                decision_type="action",
                summary=f"Execute due playbook action `{action_type}` ({reason}).",
                time_sec=float(context["time_sec"]),
                action=action,
                confidence=0.25,
                labels=["agentic", self.role, "playbook-fallback"],
                metadata={
                    "backend": self.engine.last_mode,
                    "error": self.engine.last_error,
                    "fallback": reason,
                },
            )
        ]

    @staticmethod
    def _only_noop_candidates(context: dict[str, Any]) -> bool:
        candidates = context.get("candidate_actions", [])
        if not candidates:
            return False
        return all(
            normalize_action(item.get("action")) == {"type": "noop"}
            for item in candidates
        )

    def _payload_to_decisions(
        self, payload: dict[str, Any], context: dict[str, Any]
    ) -> list[Any]:
        summary = str(payload.get("summary", "")).strip()
        action = normalize_action(payload.get("action"))
        decision_type = str(payload.get("decision_type", "note"))
        now = float(context["time_sec"])

        if (
            action is not None
            and action.get("type") == "noop"
            and decision_type == "action"
        ):
            decision_type = "note"
            action = None

        if action is not None:
            if self._action_satisfied(action, observation=context.get("observation", {})):
                return []
            if self._recent_same_action(
                context,
                action=action,
                within_sec=max(self.reasoning_refresh_sec, 12.0),
            ):
                return []

        signature = f"{decision_type}:{action_key(action) if action else summary}:{payload.get('diagnosis')}"
        cooldown = (
            self.action_cooldown_sec if action is not None else self.note_cooldown_sec
        )
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
                metadata={
                    "backend": self.engine.last_mode,
                    "error": self.engine.last_error,
                },
            )
        ]

    def _apply_defender_guardrail(
        self,
        context: dict[str, Any],
        decisions: list[Any],
        *,
        reason: str,
    ) -> list[Any]:
        if self.role != "defender":
            return decisions
        if any(
            decision.action is not None and decision.action.get("type") != "noop"
            for decision in decisions
        ):
            return decisions
        incident = dict(context.get("incident") or {})
        if not self._guardrail_required(context, incident=incident):
            return decisions
        candidate = self._guardrail_candidate(context, incident=incident)
        if candidate is None:
            return decisions
        action = normalize_action(candidate.get("action"))
        if action is None or action.get("type") == "noop":
            return decisions
        if self._action_satisfied(action, observation=context.get("observation", {})):
            return decisions
        if self._recent_same_action(context, action=action, within_sec=12.0):
            return decisions
        now = float(context["time_sec"])
        signature = f"guardrail:{action_key(action)}:{incident.get('type')}"
        if not self._allow_signature(
            signature,
            now,
            cooldown=max(self.reasoning_refresh_sec, 12.0),
        ):
            return decisions
        decisions.append(
            self.decision(
                decision_type="action",
                summary=str(candidate.get("summary", "Contain the incident.")),
                time_sec=now,
                action=action,
                confidence=0.9,
                diagnosis=incident.get("diagnosis"),
                labels=["agentic", "defender", "safety-guardrail"],
                metadata={
                    "backend": self.engine.last_mode,
                    "error": self.engine.last_error,
                    "guardrail": reason,
                    "incident": incident,
                    "candidate_priority": candidate.get("priority"),
                    "candidate_source": candidate.get("source"),
                },
            )
        )
        return decisions

    def _guardrail_required(
        self,
        context: dict[str, Any],
        *,
        incident: dict[str, Any],
    ) -> bool:
        if str(incident.get("type", "none")) == "none":
            return False
        if str(incident.get("recommended_action") or "").strip() == "":
            return False
        if int(incident.get("severity", 0) or 0) >= 4:
            return True
        if str(incident.get("status")) in {"confirmed", "containing"}:
            return True
        if (
            str(incident.get("type")) == "raw_water_low_protection"
            and "Alarm_RW_Tank_LL" in context.get("active_alarms", [])
        ):
            return True
        return False

    def _guardrail_candidate(
        self,
        context: dict[str, Any],
        *,
        incident: dict[str, Any],
    ) -> dict[str, Any] | None:
        recommended_action = str(incident.get("recommended_action") or "")
        matching = [
            item
            for item in context.get("candidate_actions", [])
            if str(item.get("action", {}).get("type")) == recommended_action
        ]
        if not matching:
            matching = [
                item
                for item in context.get("candidate_actions", [])
                if item.get("guardrail_reason")
            ]
        if not matching:
            return None
        matching.sort(
            key=lambda item: (
                int(item.get("priority", 0)),
                str(item.get("guardrail_reason") or ""),
            ),
            reverse=True,
        )
        return matching[0]

    def _recent_same_action(
        self,
        context: dict[str, Any],
        *,
        action: dict[str, Any],
        within_sec: float,
    ) -> bool:
        now = float(context.get("time_sec", 0.0))
        target = action_key(action)
        for item in reversed(context.get("recent_actions", [])):
            if item.get("role") != "defender":
                continue
            previous_action = normalize_action(item.get("action"))
            if previous_action is None or action_key(previous_action) != target:
                continue
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is None:
                continue
            if now - time_sec <= within_sec:
                return True
        return False

    def _allow_signature(self, signature: str, now: float, *, cooldown: float) -> bool:
        signatures = self._state.setdefault("signatures", {})
        last = signatures.get(signature)
        if last is not None and now - float(last) < cooldown:
            return False
        signatures[signature] = now
        return True

    def _should_reason(self, context: dict[str, Any], *, channel: str) -> bool:
        now = float(context["time_sec"])
        state = self._state.setdefault("reasoning", {})
        signature = self._reasoning_signature(context, channel=channel)
        channel_state = state.get(channel, {})
        last_signature = channel_state.get("signature")
        last_time = float(channel_state.get("time_sec", -10_000.0))
        evidence = self._reasoning_evidence(context)

        if signature == last_signature and now - last_time < self.reasoning_refresh_sec:
            return False
        if (
            now - last_time < self.reasoning_cooldown_sec
            and not self._has_material_new_evidence(
                evidence,
                previous=channel_state.get("evidence", {}),
            )
        ):
            return False

        state[channel] = {"signature": signature, "time_sec": now, "evidence": evidence}
        return True

    def _should_monitor(self, context: dict[str, Any]) -> bool:
        if context.get("active_alarms") or context.get("risk_flags"):
            return True
        if self._recent_role_activity(
            context,
            roles={"attacker", "defender"},
            within_sec=8.0,
        ):
            return True
        if self._recent_alert_activity(context, within_sec=10.0):
            return True
        return False

    def _should_plan(self, context: dict[str, Any]) -> bool:
        if context.get("risk_flags") or context.get("active_alarms"):
            return True
        if self._recent_alert_activity(context, within_sec=10.0):
            return True
        if self._recent_role_activity(
            context,
            roles={"attacker", "defender"},
            within_sec=8.0,
        ):
            return True
        return False

    @staticmethod
    def _recent_state_change(context: dict[str, Any]) -> bool:
        observations = context.get("recent_observations", [])
        if len(observations) < 2:
            return False
        return observations[-1].get("STATE_NAME") != observations[-2].get("STATE_NAME")

    @staticmethod
    def _recent_role_activity(
        context: dict[str, Any], *, roles: set[str], within_sec: float
    ) -> bool:
        now = float(context.get("time_sec", 0.0))
        for item in context.get("recent_actions", []):
            if item.get("role") not in roles:
                continue
            time_sec = AgenticRoleAgent._coerce_number(item.get("time_sec"))
            if time_sec is None:
                continue
            if now - time_sec <= within_sec:
                return True
        return False

    @staticmethod
    def _recent_alert_activity(context: dict[str, Any], *, within_sec: float) -> bool:
        now = float(context.get("time_sec", 0.0))
        for item in context.get("recent_alerts", []):
            time_sec = AgenticRoleAgent._coerce_number(item.get("time_sec"))
            if time_sec is None:
                continue
            if now - time_sec <= within_sec:
                return True
        return False

    def _reasoning_signature(self, context: dict[str, Any], *, channel: str) -> str:
        safety_bounds = context.get("safety_bounds", {})
        observation = dict(context.get("observation") or {})
        signature_observation: dict[str, Any] = {}
        for key, value in observation.items():
            if key in {"timestamp_utc", "elapsed_sec", "sample_phase"}:
                continue
            numeric_value = self._coerce_number(value)
            bounds = safety_bounds.get(key)
            if numeric_value is not None and isinstance(bounds, dict):
                minimum = self._coerce_number(bounds.get("min"))
                maximum = self._coerce_number(bounds.get("max"))
                if minimum is not None and maximum is not None and maximum > minimum:
                    bucket = max(1.0, (maximum - minimum) / 20.0)
                    signature_observation[key] = round(numeric_value / bucket, 2)
                    continue
            signature_observation[key] = value

        payload = {
            "channel": channel,
            "state_name": context.get("state_name"),
            "observation": signature_observation,
            "candidate_ids": [
                item["id"] for item in context.get("candidate_actions", [])
            ],
            "playbook_head": [
                {
                    "time_sec": item.get("time_sec"),
                    "action": normalize_action(item.get("action")),
                }
                for item in context.get("remaining_playbook", [])[:2]
            ],
            "recent_alert_count": len(context.get("recent_alerts", [])),
            "recent_action_count": len(context.get("recent_actions", [])),
            "recent_decision_count": len(context.get("recent_decisions", [])),
            "active_alarms": context.get("active_alarms", []),
            "risk_flags": context.get("risk_flags", []),
            "incident": context.get("incident"),
            "relationship_notes": context.get("relationship_notes", [])[:8],
            "trend_directions": {
                field: item.get("direction")
                for field, item in context.get("short_horizon_trends", {}).items()
            },
            "metrics": context.get("metrics"),
            "passed": context.get("passed"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def _reasoning_evidence(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "active_alarms": tuple(context.get("active_alarms", [])),
            "risk_flags": tuple(context.get("risk_flags", [])),
            "observer_alert_count": sum(
                1 for item in context.get("recent_alerts", []) if item.get("role") == "observer"
            ),
            "attacker_action_count": sum(
                1 for item in context.get("recent_actions", []) if item.get("role") == "attacker"
            ),
            "process2_signal_count": sum(
                1
                for flag in context.get("risk_flags", [])
                if flag in {"process2_manual_path_active", "unauthorized_manual_dosing_possible"}
                or str(flag).endswith("_depleting")
            ),
            "incident_type": (context.get("incident") or {}).get("type"),
            "incident_status": (context.get("incident") or {}).get("status"),
            "incident_severity": int((context.get("incident") or {}).get("severity", 0) or 0),
        }

    def _has_material_new_evidence(
        self,
        current: dict[str, Any],
        *,
        previous: dict[str, Any],
    ) -> bool:
        if not previous:
            return True
        if current.get("observer_alert_count", 0) > previous.get("observer_alert_count", 0):
            return True
        if current.get("attacker_action_count", 0) > previous.get("attacker_action_count", 0):
            return True
        if current.get("process2_signal_count", 0) > previous.get("process2_signal_count", 0):
            return True
        if set(current.get("active_alarms", ())).difference(previous.get("active_alarms", ())):
            return True
        if set(current.get("risk_flags", ())).difference(previous.get("risk_flags", ())):
            return True
        if current.get("incident_severity", 0) > previous.get("incident_severity", 0):
            return True
        status_order = {
            "none": 0,
            "stabilized": 0,
            "suspected": 1,
            "confirmed": 2,
            "containing": 3,
        }
        return status_order.get(str(current.get("incident_status")), 0) > status_order.get(
            str(previous.get("incident_status")),
            0,
        )

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "timestamp_utc",
            "elapsed_sec",
            "sample_phase",
            "STATE_NAME",
            "RW_Tank_Level",
            "UF_UFFT_Tank_Level",
            "ChemTreat_NaCl_Level",
            "ChemTreat_NaOCl_Level",
            "ChemTreat_HCl_Level",
            "RW_Pump_Flow",
            "RW_Pump_Speed_Cmd",
            "RW_Tank_PR_Valve_Cmd",
            "RW_Tank_P_Valve_Cmd",
            "RW_Tank_PR_Valve_Sts",
            "RW_Tank_P_Valve_Sts",
            "RW_Pump_Sts",
            "ChemTreat_NaCl_Valve_Cmd",
            "ChemTreat_NaOCl_Valve_Cmd",
            "ChemTreat_HCl_Valve_Cmd",
            "ChemTreat_NaCl_Valve_Sts",
            "ChemTreat_NaOCl_Valve_Sts",
            "ChemTreat_HCl_Valve_Sts",
            "SYS_IDLE",
            "SYS_START",
            "SYS_RUNNING",
            "SYS_SHUTDOWN",
            "P2_Selected",
            "P2_NaCl_Cmd",
            "P2_NaOCl_Cmd",
            "P2_HCl_Cmd",
            "Alarm_RW_Tank_LL",
            "Alarm_RW_Tank_L",
            "Alarm_RW_Tank_H",
            "Alarm_RW_Tank_HH",
            "HMI_Start_Active",
            "HMI_Stop_Active",
        ]
        return {
            field: observation.get(field) for field in fields if field in observation
        }
