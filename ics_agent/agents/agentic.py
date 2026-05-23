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
            "recent_observations": context.get("recent_observations", [])[-3:],
            "recent_actions": context.get("recent_actions", [])[-6:],
            "recent_alerts": context.get("recent_alerts", [])[-4:],
            "recent_action_effects": context.get("recent_action_effects", []),
            "active_alarms": context.get("active_alarms", []),
            "active_states": context.get("active_states", []),
            "active_commands": context.get("active_commands", []),
            "short_horizon_trends": context.get("short_horizon_trends", {}),
            "relationship_notes": context.get("relationship_notes", []),
            "risk_flags": context.get("risk_flags", []),
            "incident": context.get("incident"),
            "structured_memory": context.get("structured_memory"),
            "decision_epoch": context.get("decision_epoch"),
            "role_goal": context.get("role_goal"),
            "role_constraints": context.get("role_constraints", []),
            "expected_outcomes": context.get("expected_outcomes", []),
            "safety_bounds": context.get("safety_bounds", {}),
        }
        if self.role == "offline_analysis":
            payload["recent_decisions"] = context.get("recent_decisions", [])
            payload["metrics"] = context.get("metrics")
            payload["passed"] = context.get("passed")
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
            "- summary must stay under 18 words and should not restate the full prompt context.\n"
            "- Avoid extra keys unless they are strictly necessary for the role memory.\n"
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
        profile = self._normalized_agentic_profile(profile)
        super().__init__(profile_name, profile)
        self.engine = CachedQwenDecisionEngine(profile, self.role)
        self.knowledge_mode = str(profile.get("knowledge_mode", "standard")).strip().lower()
        memory = profile.get("memory", {})
        self.observation_window = int(memory.get("observation_window", 8))
        self.decision_window = int(memory.get("decision_window", 8))
        self.action_window = int(memory.get("action_window", 8))
        self.alert_window = int(memory.get("alert_window", 6))
        self.summary_window = int(memory.get("summary_window", 6))
        self.attacker_candidate_window = int(
            memory.get("attacker_candidate_window", 4)
        )
        self.action_cooldown_sec = float(profile.get("action_cooldown_sec", 1.5))
        self.note_cooldown_sec = float(profile.get("note_cooldown_sec", 5.0))
        self.playbook_anticipation_sec = float(
            profile.get("playbook_anticipation_sec", 0.5)
        )
        self.reasoning_cooldown_sec = float(profile.get("reasoning_cooldown_sec", 2.0))
        self.reasoning_refresh_sec = float(profile.get("reasoning_refresh_sec", 6.0))
        self.abort_on_backend_error = bool(
            profile.get(
                "abort_on_backend_error",
                self.role in {"attacker", "observer", "defender", "planner"},
            )
        )

    @staticmethod
    def _normalized_agentic_profile(profile: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(profile)
        role = str(normalized.get("role", "")).strip().lower()
        knowledge_mode = str(normalized.get("knowledge_mode", "")).strip().lower()
        if role == "attacker" and knowledge_mode == "practical_black_box":
            allowed = list(normalized.get("allowed_action_types", []))
            if "inspect_surface" not in allowed:
                insertion_index = (
                    allowed.index("hmi_estop") + 1 if "hmi_estop" in allowed else 0
                )
                allowed.insert(insertion_index, "inspect_surface")
                normalized["allowed_action_types"] = allowed
        return normalized

    def reset(self, scenario: Any) -> None:
        super().reset(scenario)
        self._state.update(
            {
                "role_memory": {},
                "decision_journal": [],
                "action_effect_journal": [],
            }
        )

    def _is_practical_black_box_attacker(self) -> bool:
        return self.role == "attacker" and self.knowledge_mode == "practical_black_box"

    def step(self, context: dict[str, Any]) -> list[Any]:
        if self.mode == "offline":
            return []
        if self.role == "operator":
            return self._step_operator_role(context)
        if self.role == "attacker":
            return self._step_attacker_role(context)
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

    def _step_operator_role(self, context: dict[str, Any]) -> list[Any]:
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

    def _step_attacker_role(self, context: dict[str, Any]) -> list[Any]:
        agent_context = self._build_online_context(context)
        if self._only_noop_candidates(agent_context):
            return []
        if not self._should_reason(agent_context, channel="action"):
            return []
        try:
            payload = self.engine.decide(agent_context)
        except Exception as exc:
            if self.abort_on_backend_error:
                raise RuntimeError(
                    f"agentic attacker backend failure: {exc}"
                ) from exc
            return []
        if self._is_practical_black_box_attacker():
            payload = self._enforce_practical_black_box_payload(payload, agent_context)
        return self._payload_to_decisions(payload, agent_context)

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
        except Exception as exc:
            if self.abort_on_backend_error:
                raise RuntimeError(
                    f"agentic {self.role} backend failure: {exc}"
                ) from exc
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
        except Exception as exc:
            if self.abort_on_backend_error:
                raise RuntimeError(
                    f"agentic planner backend failure: {exc}"
                ) from exc
            return []
        return self._payload_to_decisions(payload, agent_context)

    def _build_online_context(self, context: dict[str, Any]) -> dict[str, Any]:
        scenario = context["scenario"]
        now = float(context["time_sec"])
        practical_black_box = self._is_practical_black_box_attacker()
        raw_recent_actions = context.get("executed_actions", [])[-self.action_window :]
        raw_recent_decisions = context.get("decisions", [])[-self.decision_window :]
        raw_recent_alerts = context.get("alerts", [])[-self.alert_window :]
        recent_observations = [
            self._compact_observation(item)
            for item in context.get("observations", [])[-self.observation_window :]
        ]
        if practical_black_box:
            recent_actions = self._practical_black_box_visible_recent_actions(
                raw_recent_actions,
                now=now,
            )
            memory_recent_actions = raw_recent_actions
        else:
            recent_actions = self._visible_recent_actions(raw_recent_actions, now=now)
            memory_recent_actions = recent_actions
        if practical_black_box:
            recent_decisions = self._practical_black_box_visible_recent_decisions(
                raw_recent_decisions,
                now=now,
            )
            recent_alerts = self._practical_black_box_visible_recent_alerts(
                raw_recent_alerts,
                now=now,
            )
        else:
            recent_decisions = self._visible_recent_decisions(
                raw_recent_decisions,
                now=now,
            )
            recent_alerts = self._visible_recent_alerts(raw_recent_alerts, now=now)
        observation = context["observation"]
        compact_observation = self._compact_observation(observation)
        remaining_playbook = self._remaining_playbook(context)
        visible_remaining_playbook = [] if practical_black_box else remaining_playbook
        planned_relationship_notes = (
            [] if practical_black_box else self._planned_relationship_notes(context)
        )
        telemetry_summary = self._summarize_telemetry(
            observation=observation,
            recent_observations=context.get("observations", []),
            planned_relationship_notes=planned_relationship_notes,
        )
        recent_action_effects = self._recent_action_effects(
            observation=observation,
            recent_actions=recent_actions,
        )
        incident = self._incident_context(
            context=context,
            telemetry_summary=telemetry_summary,
        )
        structured_memory = self._role_memory_context(
            context=context,
            scenario=scenario,
            observation=observation,
            telemetry_summary=telemetry_summary,
            remaining_playbook=visible_remaining_playbook,
            recent_actions=memory_recent_actions,
            recent_alerts=recent_alerts,
            recent_action_effects=recent_action_effects,
            incident=incident,
        )
        candidate_actions = self._candidate_actions(
            context,
            visible_remaining_playbook,
            telemetry_summary=telemetry_summary,
            incident=incident,
            structured_memory=structured_memory,
        )
        scenario_id = self._visible_scenario_id(scenario)
        process_id = scenario.execution.get("process_id")
        intent = self._live_intent(scenario)
        role_goal = self._live_role_goals(scenario)
        expected_outcomes = self._live_expected_outcomes(scenario)
        safety_bounds = scenario.metadata.get("safety_bounds", {})
        if practical_black_box:
            scenario_id = "attack-live"
            process_id = None
            intent = self._practical_black_box_attacker_intent()
            role_goal = self._practical_black_box_attacker_goals()
            expected_outcomes = []
            safety_bounds = {}
        role_constraints = [
            "Use recent telemetry and state to justify the decision.",
            "Choose only from candidate_actions or emit a note/alert with no live action.",
            "Do not assume hidden attack plans, ground truth labels, or future timeline steps.",
            "Use structured_memory as the durable role state instead of replaying long transcripts.",
            "Keep summaries concise and specific to the water-treatment environment.",
            "Escalate incidents from suspected to confirmed to containing when evidence persists or worsens.",
        ]
        if practical_black_box:
            role_constraints = [
                "Use only live telemetry, recent action outcomes, bounded reconnaissance results, and coarse defender-intervention signals.",
                "Treat recent blue-team alerts as attention markers only; they intentionally omit summary text and diagnoses.",
                "Choose only from candidate_actions and prefer short exploit chains over repeated inspection.",
                "Use structured_memory to decide when to commit, sustain, escalate, or reorient.",
            ]
        visible_relationship_notes = (
            [] if practical_black_box else telemetry_summary["relationship_notes"]
        )
        visible_risk_flags = [] if practical_black_box else telemetry_summary["risk_flags"]
        return {
            "role": self.role,
            "scenario_id": scenario_id,
            "suite": scenario.suite,
            "process_id": process_id,
            "intent": intent,
            "time_sec": now,
            "state_name": observation.get("STATE_NAME")
            or observation.get("state_name"),
            "observation": compact_observation,
            "recent_observations": recent_observations,
            "recent_actions": recent_actions,
            "recent_decisions": recent_decisions,
            "recent_alerts": recent_alerts,
            "recent_action_effects": recent_action_effects,
            "active_alarms": telemetry_summary["active_alarms"],
            "active_states": telemetry_summary["active_states"],
            "active_commands": telemetry_summary["active_commands"],
            "short_horizon_trends": telemetry_summary["short_horizon_trends"],
            "relationship_notes": visible_relationship_notes,
            "risk_flags": visible_risk_flags,
            "incident": incident,
            "structured_memory": structured_memory,
            "decision_epoch": {
                "observation_count": len(context.get("observations", [])),
                "action_count": len(context.get("executed_actions", [])),
                "decision_count": len(context.get("decisions", [])),
                "alert_count": len(context.get("alerts", [])),
            },
            "candidate_actions": candidate_actions,
            "remaining_playbook": visible_remaining_playbook,
            "role_goal": role_goal,
            "role_constraints": role_constraints,
            "expected_outcomes": expected_outcomes,
            "safety_bounds": safety_bounds,
        }

    def _build_offline_context(self, context: dict[str, Any]) -> dict[str, Any]:
        scenario = context["scenario"]
        now = float(context.get("duration_sec", 0.0))
        return {
            "role": self.role,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "intent": scenario.metadata.get(
                "intent", scenario.execution.get("description", "")
            ),
            "time_sec": now,
            "candidate_actions": [],
            "remaining_playbook": [],
            "recent_observations": [
                self._compact_observation(item)
                for item in context.get("observations", [])[-self.observation_window :]
            ],
            "recent_actions": self._visible_recent_actions(
                context.get("executed_actions", [])[-self.action_window :],
                now=now,
            ),
            "recent_decisions": self._visible_recent_decisions(
                context.get("decisions", [])[-self.decision_window :],
                now=now,
            ),
            "recent_alerts": self._visible_recent_alerts(
                context.get("alerts", [])[-self.alert_window :],
                now=now,
            ),
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

    @staticmethod
    def _practical_black_box_attacker_intent() -> str:
        return (
            "Discover controllable surfaces and create disruptive physical effect "
            "while minimizing defender attention."
        )

    @staticmethod
    def _practical_black_box_attacker_goals() -> list[str]:
        return [
            "Infer what each candidate action can influence from live plant response only.",
            "Use bounded reconnaissance reads to inspect tags, registers, alarms, and control-plane state without assuming hidden truth.",
            "Use early turns to probe controllability, then exploit the most effective discovered surface.",
            "Treat alarms, state transitions, and defender reaction as observable success proxies rather than hidden ground truth.",
        ]

    def _recent_action_effects(
        self,
        *,
        observation: dict[str, Any],
        recent_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        effects: list[dict[str, Any]] = []
        for item in recent_actions[-self.summary_window :]:
            action = normalize_action(item.get("action"))
            if action is None:
                continue
            effects.append(
                {
                    "role": item.get("role"),
                    "time_sec": item.get("time_sec"),
                    "action_type": action.get("type"),
                    "effect": self._describe_action_effect(
                        action,
                        observation=observation,
                        result=item.get("result"),
                    ),
                }
            )
        return effects[-max(1, min(self.summary_window, 4)) :]

    def _describe_action_effect(
        self,
        action: dict[str, Any],
        *,
        observation: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> str:
        action_type = str(action.get("type", "unknown"))
        state_name = str(observation.get("STATE_NAME", "UNKNOWN"))
        if action_type in {"hmi_start", "hmi_stop", "hmi_estop", "hmi_reset"}:
            return f"state_now_{state_name.lower()}"
        if action_type == "select_process":
            return f"p2_selected={int(self._is_active_flag(observation.get('P2_Selected')))}"
        if action_type == "set_manual_valve":
            chemical = str(action.get("chemical", "unknown"))
            valve_field = f"ChemTreat_{chemical}_Valve_Sts"
            level_field = f"ChemTreat_{chemical}_Level"
            valve_state = int(self._is_active_flag(observation.get(valve_field)))
            level_value = observation.get(level_field)
            return f"{chemical}_valve={valve_state},level={level_value}"
        if action_type == "set_sensor":
            sensor_name = str(action.get("name", "sensor"))
            return f"{sensor_name}={observation.get(sensor_name)}"
        if action_type == "inspect_surface":
            surface = str(action.get("surface", "unknown"))
            if isinstance(result, dict) and result.get("summary"):
                return str(result.get("summary"))
            return f"recon:{surface}"
        return f"state_now_{state_name.lower()}"

    def _role_memory_context(
        self,
        *,
        context: dict[str, Any],
        scenario: Any,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        recent_actions: list[dict[str, Any]],
        recent_alerts: list[dict[str, Any]],
        recent_action_effects: list[dict[str, Any]],
        incident: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self.role == "attacker":
            memory = self._attacker_memory(
                context=context,
                scenario=scenario,
                observation=observation,
                telemetry_summary=telemetry_summary,
                remaining_playbook=remaining_playbook,
                recent_actions=recent_actions,
                recent_alerts=recent_alerts,
                recent_action_effects=recent_action_effects,
            )
        elif self.role == "observer":
            memory = self._observer_memory(
                context=context,
                telemetry_summary=telemetry_summary,
            )
        elif self.role == "defender":
            memory = self._defender_memory(
                context=context,
                telemetry_summary=telemetry_summary,
                recent_alerts=recent_alerts,
                incident=incident,
            )
        elif self.role == "planner":
            memory = self._planner_memory(
                context=context,
                telemetry_summary=telemetry_summary,
            )
        elif self.role == "operator":
            memory = self._operator_memory(
                context=context,
                remaining_playbook=remaining_playbook,
                observation=observation,
            )
        else:
            memory = self._state.setdefault("role_memory", {})
        return self._visible_memory(memory)

    def _attacker_memory(
        self,
        *,
        context: dict[str, Any],
        scenario: Any,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        recent_actions: list[dict[str, Any]],
        recent_alerts: list[dict[str, Any]],
        recent_action_effects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._is_practical_black_box_attacker():
            return self._practical_black_box_attacker_memory(
                context=context,
                observation=observation,
                telemetry_summary=telemetry_summary,
                recent_actions=recent_actions,
                recent_alerts=recent_alerts,
                recent_action_effects=recent_action_effects,
            )
        state = self._state.setdefault("role_memory", {})
        total_objective_steps = len(context.get("playbooks", {}).get("attacker", []))
        remaining_steps = len(remaining_playbook)
        leverage_hypothesis = self._attacker_leverage_hypothesis(
            remaining_playbook=remaining_playbook,
            recent_actions=recent_actions,
        )
        defender_attention = self._attention_estimate(
            recent_alerts=recent_alerts,
            recent_actions=recent_actions,
        )
        objective_progress = "inactive"
        if total_objective_steps:
            completed_steps = max(total_objective_steps - remaining_steps, 0)
            if completed_steps == 0:
                objective_progress = "not_started"
            elif remaining_steps == 0:
                objective_progress = "sequence_complete"
            elif completed_steps >= max(1, total_objective_steps // 2):
                objective_progress = "mid_operation"
            else:
                objective_progress = "initial_foothold"

        process2_selected = self._is_active_flag(observation.get("P2_Selected"))
        manual_dosing_active = "process2_manual_dosing_active" in telemetry_summary[
            "relationship_notes"
        ]
        if leverage_hypothesis == "process2_manual_path":
            if manual_dosing_active:
                tactic_phase = "exploitation"
            elif process2_selected:
                tactic_phase = "positioning"
            elif defender_attention == "high":
                tactic_phase = "repositioning"
            else:
                tactic_phase = "access_setup"
        elif leverage_hypothesis == "sensor_spoofing":
            tactic_phase = (
                "impact_assessment"
                if telemetry_summary["active_alarms"] or telemetry_summary["risk_flags"]
                else "spoofing"
            )
        elif leverage_hypothesis == "availability_disruption":
            tactic_phase = "availability_pressure"
        else:
            tactic_phase = "probing"

        if defender_attention == "high":
            stealth_posture = "contested"
        elif manual_dosing_active or telemetry_summary["active_alarms"]:
            stealth_posture = "impact_focused"
        else:
            stealth_posture = "balanced"

        attacker_effects = [
            item["effect"]
            for item in recent_action_effects
            if item.get("role") == "attacker" and item.get("effect")
        ][-3:]
        next_objective_steps = [
            self._canonical_action_descriptor(item.get("action"))
            for item in remaining_playbook[: self.attacker_candidate_window]
        ]
        state.update(
            {
                "mission_objective": state.get("mission_objective")
                or scenario.metadata.get("intent", scenario.execution.get("description", "")),
                "current_tactic_phase": tactic_phase,
                "leverage_hypothesis": leverage_hypothesis,
                "last_observed_effects": attacker_effects,
                "stealth_vs_impact_posture": stealth_posture,
                "defender_attention_estimate": defender_attention,
                "objective_progress": objective_progress,
                "next_objective_steps": next_objective_steps,
                "last_visible_risk_flags": telemetry_summary["risk_flags"][:6],
                "rolling_summary": (
                    f"{tactic_phase} via {leverage_hypothesis}; "
                    f"attention={defender_attention}; progress={objective_progress}."
                ),
            }
        )
        return state

    def _practical_black_box_attacker_memory(
        self,
        *,
        context: dict[str, Any],
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
        recent_actions: list[dict[str, Any]],
        recent_alerts: list[dict[str, Any]],
        recent_action_effects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del recent_action_effects
        state = self._state.setdefault("role_memory", {})
        surface_state = state.setdefault(
            "_surface_state",
            {
                "process2_manual_path": self._practical_black_box_surface_template(),
                "sensor_manipulation": self._practical_black_box_surface_template(),
                "availability_disruption": self._practical_black_box_surface_template(),
            },
        )
        for surface in (
            "process2_manual_path",
            "sensor_manipulation",
            "availability_disruption",
        ):
            entry = surface_state.setdefault(
                surface,
                self._practical_black_box_surface_template(),
            )
            for key, value in self._practical_black_box_surface_template().items():
                entry.setdefault(key, value if not isinstance(value, list) else list(value))

        discovered_controls = state.setdefault("_discovered_controls", {})
        recon_history = state.setdefault("_reconnaissance_history", {})
        processed_events = state.setdefault("_processed_attacker_events", set())
        commitment = state.setdefault(
            "_exploit_commitment",
            {
                "surface": "unknown",
                "template": None,
                "remaining_steps": 0,
                "stall_count": 0,
                "next_goal": None,
                "target_sensor": None,
                "sensor_direction": None,
                "preferred_chemical": None,
                "last_surface_switch_time_sec": None,
            },
        )
        defender_attention = self._attention_estimate(
            recent_alerts=recent_alerts,
            recent_actions=recent_actions,
        )
        previous_observation = self._practical_black_box_previous_observation(
            context=context,
            observation=observation,
        )
        now = float(context.get("time_sec", 0.0))
        recent_blue_alert_count = sum(
            1
            for item in recent_alerts
            if item.get("role") in {"observer", "defender"}
        )
        recent_defender_action_count = sum(
            1 for item in recent_actions if item.get("role") == "defender"
        )
        preconditions = self._practical_black_box_preconditions(
            observation=observation,
            telemetry_summary=telemetry_summary,
            defender_attention=defender_attention,
        )
        attacker_history = [
            item for item in context.get("executed_actions", []) if item.get("role") == "attacker"
        ]
        last_success_event = dict(state.get("_last_success_event") or {})
        last_attacker_action_type = None

        for item in attacker_history:
            action = normalize_action(item.get("action"))
            if action is None:
                continue
            descriptor = self._canonical_action_descriptor(action)
            event_marker = f"{item.get('time_sec')}:{descriptor}"
            last_attacker_action_type = str(action.get("type", ""))

            if event_marker in processed_events:
                continue
            processed_events.add(event_marker)

            if last_attacker_action_type == "inspect_surface":
                result = item.get("result") if isinstance(item.get("result"), dict) else {}
                surface = str(action.get("surface") or result.get("surface") or "unknown")
                findings = [
                    str(token).strip()
                    for token in result.get("findings", [])
                    if str(token).strip()
                ][:6]
                entry = recon_history.setdefault(
                    surface,
                    {
                        "surface": surface,
                        "count": 0,
                        "summary": None,
                        "findings": [],
                        "last_time_sec": None,
                    },
                )
                entry["count"] = int(entry.get("count", 0)) + 1
                if result.get("summary"):
                    entry["summary"] = str(result.get("summary"))
                if findings:
                    entry["findings"] = findings
                time_sec = self._coerce_number(item.get("time_sec"))
                if time_sec is not None:
                    entry["last_time_sec"] = round(time_sec, 3)
                for hinted_surface in self._practical_black_box_recon_surface_hints(
                    surface=surface,
                    findings=findings,
                ):
                    hinted_entry = surface_state.setdefault(
                        hinted_surface,
                        self._practical_black_box_surface_template(),
                    )
                    hinted_entry["recon_count"] = int(hinted_entry.get("recon_count", 0)) + 1
                    for token in findings[:4]:
                        if token not in hinted_entry["recon_evidence"]:
                            hinted_entry["recon_evidence"].append(token)
                    hinted_entry["recon_evidence"] = hinted_entry["recon_evidence"][-6:]
                discovered = discovered_controls.setdefault(
                    descriptor,
                    {"action": descriptor, "count": 0, "last_effect": None},
                )
                discovered["count"] = int(discovered.get("count", 0)) + 1
                discovered["last_effect"] = result.get("summary")
                continue

            surface = self._attacker_surface_from_action_type(last_attacker_action_type)
            if surface == "unknown":
                continue
            entry = surface_state.setdefault(
                surface,
                self._practical_black_box_surface_template(),
            )
            action_time = self._coerce_number(item.get("time_sec"))
            entry["exploit_attempt_count"] = int(entry.get("exploit_attempt_count", 0)) + 1
            entry["last_action"] = descriptor
            entry["last_action_time_sec"] = round(
                action_time if action_time is not None else now,
                3,
            )

            if last_attacker_action_type == "set_manual_valve":
                entry["preferred_target"] = str(action.get("chemical", "")).strip() or None
            elif last_attacker_action_type == "set_sensor":
                field_name = str(action.get("name", "")).strip()
                entry["preferred_target"] = field_name or None
                entry["last_value"] = action.get("value")
                entry["preferred_direction"] = self._practical_black_box_sensor_direction(
                    field_name=field_name,
                    observation=observation,
                    telemetry_summary=telemetry_summary,
                )
                stage = self._practical_black_box_sensor_stage(
                    field_name=field_name,
                    action_value=action.get("value"),
                    observation=observation,
                    previous_observation=previous_observation,
                )
                entry["escalation_stage"] = max(int(entry.get("escalation_stage", 0)), stage)

            effect = self._practical_black_box_effect_for_action(
                action=action,
                observation=observation,
                previous_observation=previous_observation,
                telemetry_summary=telemetry_summary,
                preconditions=preconditions,
            )
            discovered = discovered_controls.setdefault(
                descriptor,
                {"action": descriptor, "count": 0, "last_effect": None},
            )
            discovered["count"] = int(discovered.get("count", 0)) + 1

            if effect:
                entry["success_count"] = int(entry.get("success_count", 0)) + 1
                entry["stall_count"] = 0
                entry["last_effect"] = effect
                entry["last_success_time_sec"] = round(
                    action_time if action_time is not None else now,
                    3,
                )
                discovered["last_effect"] = effect
                last_success_event = {
                    "time_sec": round(action_time if action_time is not None else now, 3),
                    "surface": surface,
                    "effect": effect,
                }
            else:
                entry["stall_count"] = int(entry.get("stall_count", 0)) + 1
                discovered["last_effect"] = discovered.get("last_effect")

        state["_last_success_event"] = last_success_event

        base_recon_budget = 3
        total_recon_actions = sum(int(entry.get("count", 0)) for entry in recon_history.values())
        max_stall_count = max(
            int(payload.get("stall_count", 0)) for payload in surface_state.values()
        )
        fallback_recon_credit = 1 if max_stall_count >= 2 else 0
        recon_budget_remaining = max(
            0,
            base_recon_budget + fallback_recon_credit - total_recon_actions,
        )

        for surface, payload in surface_state.items():
            payload["score"] = self._practical_black_box_surface_score(
                surface=surface,
                payload=payload,
                preconditions=preconditions,
                telemetry_summary=telemetry_summary,
            )

        committed_surface = str(commitment.get("surface", "unknown"))
        if committed_surface not in surface_state:
            committed_surface = "unknown"
        best_surface = self._practical_black_box_choose_surface(
            surface_state=surface_state,
            committed_surface=committed_surface,
            recon_budget_remaining=recon_budget_remaining,
        )
        if best_surface == "unknown" and committed_surface != "unknown":
            best_surface = committed_surface
        if best_surface == "unknown" and recon_budget_remaining == 0:
            ranked_surface = max(
                surface_state.items(),
                key=lambda item: float(item[1].get("score", 0.0)),
            )
            if float(ranked_surface[1].get("score", 0.0)) > 0:
                best_surface = ranked_surface[0]
        if best_surface == "unknown" and total_recon_actions >= 2:
            ranked_surface = max(
                surface_state.items(),
                key=lambda item: float(item[1].get("score", 0.0)),
            )
            if float(ranked_surface[1].get("score", 0.0)) >= 0.6:
                best_surface = ranked_surface[0]

        current_surface_score = float(surface_state.get(committed_surface, {}).get("score", 0.0))
        best_surface_score = float(surface_state.get(best_surface, {}).get("score", 0.0))
        committed_stall = int(surface_state.get(committed_surface, {}).get("stall_count", 0))
        should_switch_surface = committed_surface == "unknown" or int(
            commitment.get("remaining_steps", 0) or 0
        ) <= 0
        if committed_surface != "unknown" and committed_stall >= 2:
            should_switch_surface = True
        if (
            best_surface != "unknown"
            and best_surface != committed_surface
            and best_surface_score >= current_surface_score + 1.0
        ):
            should_switch_surface = True
        if should_switch_surface and best_surface != "unknown":
            committed_surface = best_surface
            commitment.update(
                {
                    "surface": committed_surface,
                    "template": self._practical_black_box_template_for_surface(
                        committed_surface
                    ),
                    "remaining_steps": 3,
                    "stall_count": int(
                        surface_state[committed_surface].get("stall_count", 0)
                    ),
                    "last_surface_switch_time_sec": round(now, 3),
                }
            )

        if committed_surface == "process2_manual_path":
            commitment["preferred_chemical"] = self._practical_black_box_preferred_chemical(
                observation=observation,
                preferred=str(commitment.get("preferred_chemical") or ""),
                open_chemicals=preconditions["manual_open_chemicals"],
            )
        elif committed_surface == "sensor_manipulation":
            target_field = str(
                commitment.get("target_sensor")
                or surface_state["sensor_manipulation"].get("preferred_target")
                or "RW_Tank_Level"
            )
            if target_field:
                commitment["target_sensor"] = target_field
                commitment["sensor_direction"] = self._practical_black_box_sensor_direction(
                    field_name=target_field,
                    observation=observation,
                    telemetry_summary=telemetry_summary,
                )

        last_success_surface = str(last_success_event.get("surface", ""))
        if committed_surface != "unknown":
            if last_success_surface == committed_surface:
                commitment["remaining_steps"] = max(
                    int(commitment.get("remaining_steps", 0) or 0),
                    2,
                )
            elif committed_stall >= 2:
                commitment["remaining_steps"] = 0
            else:
                commitment["remaining_steps"] = max(
                    1,
                    int(commitment.get("remaining_steps", 0) or 0),
                )
            commitment["stall_count"] = committed_stall
            commitment["next_goal"] = self._practical_black_box_next_goal(
                surface=committed_surface,
                observation=observation,
                preconditions=preconditions,
                surface_payload=surface_state.get(committed_surface, {}),
            )

        if not attacker_history:
            current_phase = "initial_observation"
        elif committed_surface == "unknown" and recon_budget_remaining > 0:
            current_phase = "reconnaissance"
        elif committed_stall >= 2:
            current_phase = "reorient"
        elif committed_surface != "unknown" and last_success_surface == committed_surface:
            current_phase = "exploit_escalation"
        elif committed_surface != "unknown":
            current_phase = "exploit_commitment"
        else:
            current_phase = "probing"
        if defender_attention == "high" and committed_surface != "unknown":
            current_phase = "evasion_or_exploitation"

        surface_rank = sorted(
            (
                (
                    surface,
                    round(float(payload.get("score", 0.0)), 3),
                    payload.get("last_effect"),
                )
                for surface, payload in surface_state.items()
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        visible_controls = [
            {
                "action": str(entry.get("action")),
                "count": int(entry.get("count", 0)),
                "last_effect": entry.get("last_effect"),
            }
            for entry in sorted(
                discovered_controls.values(),
                key=lambda item: (
                    int(item.get("count", 0)),
                    str(item.get("action", "")),
                ),
                reverse=True,
            )[:6]
        ]
        observed_effect_map = [
            {
                "surface": surface,
                "score": score,
                "success_count": int(surface_state[surface].get("success_count", 0)),
                "stall_count": int(surface_state[surface].get("stall_count", 0)),
                "last_effect": effect,
            }
            for surface, score, effect in surface_rank
            if score > 0
        ]
        reconnaissance_findings = [
            {
                "surface": surface,
                "count": int(entry.get("count", 0)),
                "summary": entry.get("summary"),
                "findings": list(entry.get("findings", []))[:4],
            }
            for surface, entry in sorted(
                recon_history.items(),
                key=lambda item: (
                    int(item[1].get("count", 0)),
                    float(item[1].get("last_time_sec") or 0.0),
                    item[0],
                ),
                reverse=True,
            )[:4]
        ]
        candidate_leverage_hypotheses = []
        for surface, score, effect in surface_rank:
            if score <= 0:
                continue
            surface_payload = surface_state.get(surface, {})
            basis: list[str] = []
            if effect:
                basis.append(f"last_effect={effect}")
            basis.extend(list(surface_payload.get("recon_evidence", []))[:2])
            if surface == "process2_manual_path":
                if preconditions["process2_enabled"]:
                    basis.append("process2_enabled")
                if preconditions["manual_path_active"]:
                    basis.append("manual_path_active")
                basis.extend(preconditions["process2_drain_signals"][:1])
            elif surface == "sensor_manipulation":
                if preconditions["preferred_sensor"]:
                    basis.append(f"target={preconditions['preferred_sensor']}")
                if preconditions["telemetry_jump"]:
                    basis.append("telemetry_jump")
                basis.extend(telemetry_summary["active_alarms"][:1])
            elif surface == "availability_disruption":
                basis.append(f"state={preconditions['state_name']}")
            candidate_leverage_hypotheses.append(
                {
                    "surface": surface,
                    "confidence": round(min(0.97, 0.22 + (score * 0.16)), 2),
                    "basis": basis[:4],
                }
            )

        state.update(
            {
                "mission_objective": state.get("mission_objective")
                or self._practical_black_box_attacker_intent(),
                "current_phase": current_phase,
                "discovered_controls": visible_controls,
                "observed_effect_map": observed_effect_map,
                "reconnaissance_findings": reconnaissance_findings,
                "detection_pressure": defender_attention,
                "best_known_attack_surface": committed_surface
                if committed_surface != "unknown"
                else best_surface,
                "last_successful_effect": last_success_event.get("effect"),
                "candidate_leverage_hypotheses": candidate_leverage_hypotheses[:3],
                "exploit_commitment": {
                    "surface": committed_surface,
                    "template": commitment.get("template"),
                    "remaining_steps": int(commitment.get("remaining_steps", 0) or 0),
                    "stall_count": int(commitment.get("stall_count", 0) or 0),
                    "next_goal": commitment.get("next_goal"),
                },
                "simple_precondition_state": {
                    "plant_running": bool(preconditions["plant_running"]),
                    "process2_enabled": bool(preconditions["process2_enabled"]),
                    "manual_path_active": bool(preconditions["manual_path_active"]),
                    "manual_open_chemicals": preconditions["manual_open_chemicals"][:3],
                    "alarm_pressure": preconditions["alarm_pressure"],
                    "defender_attention": preconditions["defender_attention"],
                },
                "defender_intervention_awareness": {
                    "recent_blue_alert_count": recent_blue_alert_count,
                    "recent_defender_action_count": recent_defender_action_count,
                    "pressure": defender_attention,
                },
                "recon_budget_remaining": recon_budget_remaining,
                "rolling_summary": (
                    f"phase={current_phase}; surface={committed_surface or best_surface}; "
                    f"recon_remaining={recon_budget_remaining}; pressure={defender_attention}."
                ),
            }
        )
        return state

    @staticmethod
    def _practical_black_box_surface_template() -> dict[str, Any]:
        return {
            "recon_count": 0,
            "recon_evidence": [],
            "exploit_attempt_count": 0,
            "success_count": 0,
            "stall_count": 0,
            "last_effect": None,
            "last_action": None,
            "last_success_time_sec": None,
            "last_action_time_sec": None,
            "preferred_target": None,
            "preferred_direction": None,
            "last_value": None,
            "escalation_stage": 0,
            "score": 0.0,
        }

    def _practical_black_box_preconditions(
        self,
        *,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
        defender_attention: str,
    ) -> dict[str, Any]:
        state_name = str(observation.get("STATE_NAME", "")).upper() or "UNKNOWN"
        open_chemicals = self._practical_black_box_open_chemicals(observation)
        active_alarm_count = len(telemetry_summary["active_alarms"])
        alarm_pressure = "low"
        if active_alarm_count >= 2 or defender_attention == "high":
            alarm_pressure = "high"
        elif active_alarm_count == 1 or defender_attention == "medium":
            alarm_pressure = "medium"
        preferred_sensor = next(
            (
                field
                for field in ("RW_Tank_Level", "UF_UFFT_Tank_Level")
                if field in observation
            ),
            None,
        )
        return {
            "state_name": state_name,
            "plant_running": state_name in {"RUNNING", "STARTING"},
            "process2_enabled": self._is_active_flag(observation.get("P2_Selected")),
            "manual_path_active": bool(open_chemicals),
            "manual_open_chemicals": open_chemicals,
            "process2_drain_signals": [
                flag
                for flag in telemetry_summary["risk_flags"]
                if str(flag).endswith("_depleting")
            ][:3],
            "alarm_pressure": alarm_pressure,
            "defender_attention": defender_attention,
            "telemetry_jump": "sensor_or_process_jump" in telemetry_summary["risk_flags"]
            or any(
                note.startswith("abrupt_change:")
                for note in telemetry_summary["relationship_notes"]
            ),
            "preferred_sensor": preferred_sensor,
        }

    @staticmethod
    def _practical_black_box_open_chemicals(observation: dict[str, Any]) -> list[str]:
        open_chemicals: list[str] = []
        for chemical in ("NaCl", "NaOCl", "HCl"):
            field = f"ChemTreat_{chemical}_Valve_Sts"
            if AgenticRoleAgent._is_active_flag(observation.get(field)):
                open_chemicals.append(chemical)
        return open_chemicals

    def _practical_black_box_previous_observation(
        self,
        *,
        context: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        observations = context.get("observations", [])
        if not observations:
            return {}
        if len(observations) == 1:
            return dict(observations[0])

        current_elapsed = self._coerce_number(observation.get("elapsed_sec"))
        if current_elapsed is None:
            current_elapsed = self._coerce_number(context.get("time_sec"))
        last_sample = observations[-1]
        last_elapsed = self._coerce_number(last_sample.get("elapsed_sec"))
        if (
            last_elapsed is not None
            and current_elapsed is not None
            and last_elapsed == current_elapsed
            and last_sample.get("sample_phase") is not None
        ):
            return dict(observations[-2])
        return dict(last_sample)

    def _practical_black_box_recon_surface_hints(
        self,
        *,
        surface: str,
        findings: list[str],
    ) -> list[str]:
        hints: list[str] = []
        if surface == "process2_control_plane" or any(
            token == "process2_selected"
            or token.startswith("P2_")
            or (
                token.startswith("ChemTreat_") and token.endswith("_Valve_Sts=1")
            )
            for token in findings
        ):
            hints.append("process2_manual_path")
        if surface in {"tags", "registers", "alarm_bits", "io_image"} or any(
            token.startswith("active_alarms=")
            or token.startswith("RW_Tank_Level=")
            or token.startswith("UF_UFFT_Tank_Level=")
            for token in findings
        ):
            hints.append("sensor_manipulation")
        if surface in {"io_image", "actuator_state_bits"} or any(
            token.startswith("state=") for token in findings
        ):
            hints.append("availability_disruption")
        return sorted(set(hints))

    def _practical_black_box_effect_for_action(
        self,
        *,
        action: dict[str, Any],
        observation: dict[str, Any],
        previous_observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
        preconditions: dict[str, Any],
    ) -> str | None:
        action_type = str(action.get("type", ""))
        if action_type == "select_process":
            active = bool(action.get("active", True))
            if active and preconditions["process2_enabled"]:
                return "process2_selection_persisted"
            if not active and not preconditions["process2_enabled"]:
                return "process2_access_removed"
            return None
        if action_type == "set_manual_valve":
            chemical = str(action.get("chemical", "")).strip()
            if not chemical:
                return None
            valve_open = chemical in preconditions["manual_open_chemicals"]
            level_field = f"ChemTreat_{chemical}_Level"
            trend = telemetry_summary["short_horizon_trends"].get(level_field, {})
            if bool(action.get("open", True)):
                tokens: list[str] = []
                if valve_open:
                    tokens.append(f"{chemical}_valve_open")
                if trend.get("direction") == "falling":
                    tokens.append(f"{level_field}_falling")
                if f"{level_field}_depleting" in telemetry_summary["risk_flags"]:
                    tokens.append(f"{level_field}_depleting")
                if tokens:
                    return "; ".join(tokens)
                return None
            if not valve_open:
                return f"{chemical}_valve_closed"
            return None
        if action_type == "set_sensor":
            field_name = str(action.get("name", "")).strip()
            target_value = self._coerce_number(action.get("value"))
            current_value = self._coerce_number(observation.get(field_name))
            previous_value = self._coerce_number(previous_observation.get(field_name))
            tokens: list[str] = []
            if (
                target_value is not None
                and current_value is not None
                and abs(current_value - target_value) <= max(8.0, abs(target_value) * 0.08)
            ):
                tokens.append("sensor_bias_persisted")
            if (
                current_value is not None
                and previous_value is not None
                and abs(current_value - previous_value) >= max(25.0, abs(previous_value) * 0.12)
            ):
                tokens.append("telemetry_shift_observed")
            if telemetry_summary["active_alarms"]:
                tokens.append(f"alarms={','.join(telemetry_summary['active_alarms'][:2])}")
            if preconditions["telemetry_jump"]:
                tokens.append("control_response_changed")
            if tokens:
                return f"{field_name}: " + "; ".join(tokens[:3])
            return None
        if action_type == "hmi_start":
            if preconditions["plant_running"]:
                return "plant_running"
            return None
        if action_type == "hmi_stop":
            if preconditions["state_name"] in {"SHUTDOWN", "STOPPED"}:
                return "controlled_stop_achieved"
            return None
        if action_type == "hmi_estop":
            if preconditions["state_name"] in {"SHUTDOWN", "STOPPED"}:
                return "hard_stop_achieved"
            return None
        return None

    def _practical_black_box_surface_score(
        self,
        *,
        surface: str,
        payload: dict[str, Any],
        preconditions: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> float:
        score = float(payload.get("success_count", 0)) * 2.4
        score += min(1.2, float(payload.get("exploit_attempt_count", 0)) * 0.35)
        score += min(1.0, float(payload.get("recon_count", 0)) * 0.28)
        score -= float(payload.get("stall_count", 0)) * 1.2
        if surface == "process2_manual_path":
            if preconditions["process2_enabled"]:
                score += 1.1
            if preconditions["manual_path_active"]:
                score += 1.5
            if preconditions["process2_drain_signals"]:
                score += min(1.5, len(preconditions["process2_drain_signals"]) * 0.8)
            if any(
                token == "process2_selected" or token.startswith("P2_")
                for token in payload.get("recon_evidence", [])
            ):
                score += 0.7
        elif surface == "sensor_manipulation":
            if preconditions["preferred_sensor"]:
                score += 0.9
            if preconditions["telemetry_jump"]:
                score += 0.8
            if telemetry_summary["active_alarms"]:
                score += 0.5
            score += min(1.0, int(payload.get("escalation_stage", 0)) * 0.35)
        elif surface == "availability_disruption":
            if preconditions["plant_running"]:
                score += 1.1
            if preconditions["state_name"] in {"SHUTDOWN", "STOPPED"}:
                score += 0.4
            if any(
                token.startswith("state=") for token in payload.get("recon_evidence", [])
            ):
                score += 0.3
        return round(max(0.0, score), 3)

    @staticmethod
    def _practical_black_box_choose_surface(
        *,
        surface_state: dict[str, dict[str, Any]],
        committed_surface: str,
        recon_budget_remaining: int,
    ) -> str:
        if committed_surface in surface_state:
            committed_payload = surface_state[committed_surface]
            if (
                int(committed_payload.get("stall_count", 0)) < 2
                and float(committed_payload.get("score", 0.0)) >= 1.0
            ):
                return committed_surface
        ranked = sorted(
            surface_state.items(),
            key=lambda item: (
                float(item[1].get("score", 0.0)),
                item[0],
            ),
            reverse=True,
        )
        if not ranked:
            return "unknown"
        best_surface, payload = ranked[0]
        threshold = 1.0 if recon_budget_remaining > 0 else 0.75
        if float(payload.get("score", 0.0)) < threshold:
            return "unknown"
        return best_surface

    @staticmethod
    def _practical_black_box_template_for_surface(surface: str) -> str | None:
        mapping = {
            "process2_manual_path": "process2_manual_path_abuse",
            "sensor_manipulation": "telemetry_bias_escalation",
            "availability_disruption": "availability_disruption",
        }
        return mapping.get(surface)

    def _practical_black_box_next_goal(
        self,
        *,
        surface: str,
        observation: dict[str, Any],
        preconditions: dict[str, Any],
        surface_payload: dict[str, Any],
    ) -> str:
        if surface == "process2_manual_path":
            if not preconditions["plant_running"]:
                return "start_the_plant_to_make_process2_effectful"
            if not preconditions["process2_enabled"]:
                return "enable_process2_manual_path"
            if not preconditions["manual_path_active"]:
                chemical = self._practical_black_box_preferred_chemical(
                    observation=observation,
                    preferred=str(surface_payload.get("preferred_target") or ""),
                    open_chemicals=preconditions["manual_open_chemicals"],
                )
                return f"open_{chemical.lower()}_manual_valve"
            return "extend_manual_dosing_to_adjacent_chemical"
        if surface == "sensor_manipulation":
            field_name = str(
                surface_payload.get("preferred_target")
                or preconditions.get("preferred_sensor")
                or "rw_tank_level"
            ).lower()
            stage = int(surface_payload.get("escalation_stage", 0) or 0)
            if stage <= 1:
                return f"sustain_or_escalate_{field_name}_bias"
            return f"escalate_{field_name}_bias_again"
        if surface == "availability_disruption":
            if not preconditions["plant_running"]:
                return "restart_then_disrupt_again"
            return "stop_or_estop_the_running_plant"
        return "gather_more_evidence"

    def _practical_black_box_preferred_chemical(
        self,
        *,
        observation: dict[str, Any],
        preferred: str,
        open_chemicals: list[str],
    ) -> str:
        chemicals = self._available_chemicals(observation)
        if preferred in chemicals and preferred not in open_chemicals:
            return preferred
        ranked = sorted(
            chemicals,
            key=lambda chemical: (
                chemical in open_chemicals,
                -(self._coerce_number(observation.get(f"ChemTreat_{chemical}_Level")) or 0.0),
                chemical,
            ),
        )
        return ranked[0] if ranked else "NaOCl"

    def _practical_black_box_sensor_direction(
        self,
        *,
        field_name: str,
        observation: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> str:
        if "raw_water_low" in " ".join(telemetry_summary["risk_flags"]).lower():
            return "low"
        if "high_high" in " ".join(telemetry_summary["risk_flags"]).lower():
            return "high"
        trend = telemetry_summary["short_horizon_trends"].get(field_name, {})
        if trend.get("direction") == "falling":
            return "low"
        if trend.get("direction") == "rising":
            return "high"
        current_value = self._coerce_number(observation.get(field_name))
        if current_value is not None and current_value >= 600.0:
            return "high"
        return "low"

    def _practical_black_box_sensor_stage(
        self,
        *,
        field_name: str,
        action_value: Any,
        observation: dict[str, Any],
        previous_observation: dict[str, Any],
    ) -> int:
        value = self._coerce_number(action_value)
        current_value = self._coerce_number(observation.get(field_name))
        previous_value = self._coerce_number(previous_observation.get(field_name))
        baseline = previous_value if previous_value is not None else current_value
        if value is None or baseline is None:
            return 1
        delta = abs(value - baseline)
        if delta >= max(80.0, abs(baseline) * 0.45):
            return 3
        if delta >= max(40.0, abs(baseline) * 0.22):
            return 2
        return 1

    def _observer_memory(
        self,
        *,
        context: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> dict[str, Any]:
        now = float(context.get("time_sec", 0.0))
        state = self._state.setdefault("role_memory", {})
        ledger = state.setdefault("_evidence_ledger_map", {})
        evidence_tokens = self._observer_evidence_tokens(telemetry_summary)
        for token in evidence_tokens:
            entry = ledger.setdefault(
                token,
                {"count": 0, "first_seen_time_sec": now, "last_seen_time_sec": now},
            )
            entry["count"] += 1
            entry["last_seen_time_sec"] = now
        for token, entry in list(ledger.items()):
            if token in evidence_tokens:
                continue
            if now - float(entry.get("last_seen_time_sec", now)) > 20.0:
                del ledger[token]

        hypotheses = self._observer_hypotheses(telemetry_summary)
        top_hypothesis = hypotheses[0] if hypotheses else None
        suspicion_level = "low"
        suspicion_score = 0.0
        if top_hypothesis is not None:
            suspicion_score = float(top_hypothesis.get("confidence", 0.0))
            if suspicion_score >= 0.8:
                suspicion_level = "high"
            elif suspicion_score >= 0.55:
                suspicion_level = "medium"

        evidence_ledger = [
            {
                "token": token,
                "count": int(entry.get("count", 0)),
                "first_seen_time_sec": round(float(entry.get("first_seen_time_sec", 0.0)), 3),
                "last_seen_time_sec": round(float(entry.get("last_seen_time_sec", 0.0)), 3),
            }
            for token, entry in sorted(
                ledger.items(),
                key=lambda item: (
                    float(item[1].get("last_seen_time_sec", 0.0)),
                    int(item[1].get("count", 0)),
                    item[0],
                ),
                reverse=True,
            )[:6]
        ]
        why_it_matters = None
        attention_target = "plant_overview"
        if top_hypothesis is not None:
            why_it_matters = str(top_hypothesis.get("why_it_matters"))
            attention_target = str(top_hypothesis.get("attention_target"))
        state.update(
            {
                "anomaly_hypotheses": hypotheses[:3],
                "evidence_ledger": evidence_ledger,
                "suspicion_level": suspicion_level,
                "suspicion_score": round(suspicion_score, 2),
                "why_it_matters": why_it_matters,
                "recommended_attention_target": attention_target,
                "rolling_summary": (
                    f"suspicion={suspicion_level}; target={attention_target}; "
                    f"signals={len(evidence_tokens)}."
                ),
            }
        )
        return state

    def _defender_memory(
        self,
        *,
        context: dict[str, Any],
        telemetry_summary: dict[str, Any],
        recent_alerts: list[dict[str, Any]],
        incident: dict[str, Any] | None,
    ) -> dict[str, Any]:
        incident = dict(incident or {})
        state = self._state.setdefault("role_memory", {})
        recent_actions = context.get("executed_actions", [])[-max(self.action_window, 12) :]
        prior_mitigations = [
            self._canonical_action_descriptor(item.get("action"))
            for item in recent_actions
            if item.get("role") == "defender" and item.get("action") is not None
        ][-4:]
        observer_handoff = next(
            (
                {
                    "summary": item.get("summary"),
                    "diagnosis": item.get("diagnosis"),
                    "age_sec": round(
                        float(context.get("time_sec", 0.0))
                        - float(item.get("time_sec", 0.0)),
                        3,
                    ),
                }
                for item in reversed(recent_alerts)
                if item.get("role") == "observer"
            ),
            None,
        )
        containment_effective = self._containment_effective(
            incident=incident,
            telemetry_summary=telemetry_summary,
            observation=context.get("observation", {}),
            prior_mitigations=prior_mitigations,
        )
        confidence = self._defender_confidence(incident=incident)
        escalation_required = self._defender_escalation_required(
            incident=incident,
            containment_effective=containment_effective,
        )
        state.update(
            {
                "incident_hypothesis": incident.get("type", "none"),
                "incident_type": incident.get("diagnosis") or incident.get("type"),
                "severity": int(incident.get("severity", 0) or 0),
                "confidence": confidence,
                "lifecycle_state": incident.get("status", "none"),
                "prior_mitigations_attempted": prior_mitigations,
                "containment_effective": containment_effective,
                "escalation_required": escalation_required,
                "observer_handoff": observer_handoff,
                "why_it_matters": self._incident_consequence(incident),
                "rolling_summary": (
                    f"{incident.get('status', 'none')} {incident.get('type', 'none')}; "
                    f"confidence={confidence}; escalation_required={escalation_required}."
                ),
            }
        )
        return state

    def _planner_memory(
        self,
        *,
        context: dict[str, Any],
        telemetry_summary: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state.setdefault("role_memory", {})
        state.update(
            {
                "engagement_phase": context.get("observation", {}).get("STATE_NAME"),
                "risk_posture": "elevated"
                if telemetry_summary["active_alarms"] or telemetry_summary["risk_flags"]
                else "steady",
                "branch_points": telemetry_summary["risk_flags"][:4],
            }
        )
        return state

    def _operator_memory(
        self,
        *,
        context: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state.setdefault("role_memory", {})
        state.update(
            {
                "plant_state": observation.get("STATE_NAME"),
                "next_due_actions": [
                    self._canonical_action_descriptor(item.get("action"))
                    for item in remaining_playbook[:3]
                ],
                "playbook_cursor_remaining": len(remaining_playbook),
            }
        )
        return state

    def _visible_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        hidden_keys = {
            "last_rationale",
            "last_decision_type",
            "last_diagnosis",
            "last_action",
            "last_action_type",
            "last_decision_time_sec",
            "model_tactic_phase",
            "model_attention_target",
            "model_why_it_matters",
        }
        return {
            key: value
            for key, value in memory.items()
            if not str(key).startswith("_")
            and key not in hidden_keys
            and value not in (None, "")
        }

    def _attacker_leverage_hypothesis(
        self,
        *,
        remaining_playbook: list[dict[str, Any]],
        recent_actions: list[dict[str, Any]],
    ) -> str:
        relevant_actions = [
            normalize_action(item.get("action"))
            for item in remaining_playbook[: self.attacker_candidate_window]
        ]
        relevant_actions.extend(
            normalize_action(item.get("action"))
            for item in recent_actions
            if item.get("role") == "attacker"
        )
        action_types = {
            str(action.get("type"))
            for action in relevant_actions
            if action is not None and action.get("type")
        }
        if "set_manual_valve" in action_types or "select_process" in action_types:
            return "process2_manual_path"
        if "set_sensor" in action_types:
            return "sensor_spoofing"
        if action_types.intersection({"hmi_stop", "hmi_estop"}):
            return "availability_disruption"
        return "hold_and_probe"

    def _observer_evidence_tokens(
        self,
        telemetry_summary: dict[str, Any],
    ) -> list[str]:
        tokens = list(telemetry_summary["active_alarms"])
        tokens.extend(telemetry_summary["risk_flags"])
        tokens.extend(
            note
            for note in telemetry_summary["relationship_notes"]
            if note in {"process2_selected", "process2_manual_dosing_active"}
            or note.startswith("command_status_mismatch:")
            or note.startswith("abrupt_change:")
        )
        return sorted(set(tokens))

    def _observer_hypotheses(
        self,
        telemetry_summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        hypotheses: list[dict[str, Any]] = []
        risk_flags = set(telemetry_summary["risk_flags"])
        relationship_notes = set(telemetry_summary["relationship_notes"])
        active_alarms = set(telemetry_summary["active_alarms"])
        planned_nominal_process2 = "planned_process2_operation" in relationship_notes
        planned_nominal_refill = "planned_refill_response" in relationship_notes
        process2_signals = int("process2_selected" in relationship_notes) + int(
            "process2_manual_dosing_active" in relationship_notes
        ) + sum(flag.endswith("_depleting") for flag in risk_flags)
        if process2_signals and not planned_nominal_process2:
            confidence = min(0.95, 0.45 + (0.12 * process2_signals))
            hypotheses.append(
                {
                    "diagnosis": "unauthorized_chemical_dosing",
                    "confidence": round(confidence, 2),
                    "evidence_tokens": [
                        token
                        for token in telemetry_summary["relationship_notes"]
                        if token == "process2_selected"
                        or token == "process2_manual_dosing_active"
                    ]
                    + [
                        token
                        for token in telemetry_summary["risk_flags"]
                        if token.endswith("_depleting")
                    ],
                    "why_it_matters": "Unauthorized Process 2 actuation can drain chemical inventory and alter downstream dosing.",
                    "attention_target": "WT-P2_manual_path",
                }
            )
        if {
            "raw_water_protection_risk",
            "raw_water_low_band",
        }.intersection(risk_flags) or "Alarm_RW_Tank_LL" in active_alarms:
            if not (planned_nominal_refill and "Alarm_RW_Tank_LL" not in active_alarms):
                confidence = 0.62 if "Alarm_RW_Tank_LL" in active_alarms else 0.54
                hypotheses.append(
                    {
                        "diagnosis": "raw_water_low_or_spoof",
                        "confidence": round(confidence, 2),
                        "evidence_tokens": [
                            token
                            for token in telemetry_summary["risk_flags"]
                            if token in {"raw_water_protection_risk", "raw_water_low_band"}
                        ]
                        + [
                            token
                            for token in telemetry_summary["active_alarms"]
                            if token == "Alarm_RW_Tank_LL"
                        ],
                        "why_it_matters": "Low raw-water level or a spoofed low reading can force unsafe refill and shutdown logic.",
                        "attention_target": "WT-P1_raw_water",
                    }
                )
        if {
            "overflow_risk",
            "raw_water_high_high_band",
        }.intersection(risk_flags) or "Alarm_RW_Tank_HH" in active_alarms:
            confidence = 0.7 if "Alarm_RW_Tank_HH" in active_alarms else 0.58
            hypotheses.append(
                {
                    "diagnosis": "raw_water_alarm_condition",
                    "confidence": round(confidence, 2),
                    "evidence_tokens": [
                        token
                        for token in telemetry_summary["risk_flags"]
                        if token in {"overflow_risk", "raw_water_high_high_band"}
                    ]
                    + [
                        token
                        for token in telemetry_summary["active_alarms"]
                        if token == "Alarm_RW_Tank_HH"
                    ],
                    "why_it_matters": "High-high raw-water behavior can cascade into overflow pressure and immediate safety action.",
                    "attention_target": "WT-P1_high_high",
                }
            )
        if {
            "sensor_or_process_jump",
            "command_status_divergence",
        }.intersection(risk_flags):
            hypotheses.append(
                {
                    "diagnosis": "sensor_or_process_jump",
                    "confidence": 0.52,
                    "evidence_tokens": [
                        token
                        for token in telemetry_summary["risk_flags"]
                        if token in {"sensor_or_process_jump", "command_status_divergence"}
                    ],
                    "why_it_matters": "Abrupt jumps or command-status mismatch can indicate spoofing or actuator inconsistency.",
                    "attention_target": "telemetry_integrity",
                }
            )
        hypotheses.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                len(item.get("evidence_tokens", [])),
                str(item.get("diagnosis")),
            ),
            reverse=True,
        )
        return hypotheses

    def _attention_estimate(
        self,
        *,
        recent_alerts: list[dict[str, Any]],
        recent_actions: list[dict[str, Any]],
    ) -> str:
        alert_pressure = sum(
            1
            for item in recent_alerts
            if item.get("role") in {"observer", "defender"}
        )
        defender_pressure = sum(
            1 for item in recent_actions if item.get("role") == "defender"
        )
        if defender_pressure or alert_pressure >= 2:
            return "high"
        if alert_pressure == 1:
            return "medium"
        return "low"

    def _containment_effective(
        self,
        *,
        incident: dict[str, Any],
        telemetry_summary: dict[str, Any],
        observation: dict[str, Any],
        prior_mitigations: list[str],
    ) -> bool | None:
        if not prior_mitigations:
            return None
        incident_type = str(incident.get("type", "none"))
        if incident_type == "process2_manual_dosing":
            return (
                not self._is_active_flag(observation.get("P2_Selected"))
                and "process2_manual_dosing_active"
                not in telemetry_summary["relationship_notes"]
                and not any(
                    flag.endswith("_depleting")
                    for flag in telemetry_summary["risk_flags"]
                )
            )
        if incident_type in {"raw_water_overflow", "raw_water_low_protection"}:
            return not self._is_active_flag(observation.get("RW_Pump_Sts"))
        if incident_type == "telemetry_anomaly":
            return not telemetry_summary["active_alarms"]
        return None

    def _defender_confidence(self, *, incident: dict[str, Any]) -> float:
        severity = int(incident.get("severity", 0) or 0)
        evidence_count = int(incident.get("evidence_count", 0) or 0)
        observer_alert_count = int(incident.get("observer_alert_count", 0) or 0)
        status_bonus = 0.12 if str(incident.get("status")) in {"confirmed", "containing"} else 0.0
        confidence = 0.18 + (0.12 * severity) + (0.05 * min(evidence_count, 4)) + (
            0.08 * min(observer_alert_count, 2)
        ) + status_bonus
        return round(min(0.97, confidence), 2)

    def _defender_escalation_required(
        self,
        *,
        incident: dict[str, Any],
        containment_effective: bool | None,
    ) -> bool:
        if int(incident.get("severity", 0) or 0) >= 4:
            return True
        if containment_effective is False:
            return True
        return (
            str(incident.get("status")) == "confirmed"
            and float(incident.get("persistence_sec", 0.0) or 0.0) >= 8.0
        )

    def _incident_consequence(self, incident: dict[str, Any]) -> str | None:
        incident_type = str(incident.get("type", "none"))
        mapping = {
            "process2_manual_dosing": "Chemical inventory can be drained while unauthorized manual dosing remains active.",
            "raw_water_overflow": "High-high raw-water behavior can outrun a controlled stop and become an immediate safety issue.",
            "raw_water_low_protection": "Low raw-water protection can force unsafe permissives, refill behavior, or shutdown logic.",
            "telemetry_anomaly": "Telemetry inconsistency can hide the real process state and delay safe intervention.",
        }
        return mapping.get(incident_type)

    def _canonical_action_descriptor(
        self,
        action: dict[str, Any] | str | None,
    ) -> str:
        normalized = normalize_action(action)
        if normalized is None:
            return "noop"
        action_type = str(normalized.get("type", "unknown"))
        if action_type == "inspect_surface":
            return f"inspect_surface:{normalized.get('surface', 'unknown')}"
        if action_type == "select_process":
            return (
                "select_process:on"
                if normalized.get("active", True)
                else "select_process:off"
            )
        if action_type == "set_manual_valve":
            state = "open" if normalized.get("open", True) else "close"
            return f"set_manual_valve:{normalized.get('chemical', 'unknown')}:{state}"
        if action_type == "set_sensor":
            return f"set_sensor:{normalized.get('name', 'unknown')}:{normalized.get('value')}"
        return action_type

    def _summarize_telemetry(
        self,
        *,
        observation: dict[str, Any],
        recent_observations: list[dict[str, Any]],
        planned_relationship_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        active_alarms = self._active_flags(observation, prefix="Alarm_")
        active_states = self._active_flags(observation, prefix="SYS_")
        active_commands = self._active_commands(observation)
        short_horizon_trends = self._short_horizon_trends(recent_observations)
        relationship_notes = self._derive_relationship_notes(
            observation,
            short_horizon_trends,
        )
        relationship_notes = sorted(
            set(relationship_notes).union(planned_relationship_notes or [])
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

    def _planned_relationship_notes(self, context: dict[str, Any]) -> list[str]:
        if context["scenario"].suite not in {"normal", "readiness", "recheck", "smoke"}:
            return []
        now = float(context.get("time_sec", 0.0))
        planned = context.get("playbooks", {}).get("operator", [])
        executed_actions = context.get("executed_actions", [])
        notes: list[str] = []
        has_nominal_p2_plan = any(
            (
                normalize_action(item.get("action")) or {}
            ).get("type")
            in {"select_process", "set_manual_valve"}
            and float(item.get("time_sec", 0.0)) <= now + 2.0
            for item in planned
        )
        has_operator_p2_history = any(
            item.get("role") == "operator"
            and str((normalize_action(item.get("action")) or {}).get("type"))
            in {"select_process", "set_manual_valve"}
            for item in executed_actions
        )
        if has_nominal_p2_plan or has_operator_p2_history:
            notes.append("planned_process2_operation")
        scenario_text = " ".join(
            filter(
                None,
                [
                    str(getattr(context["scenario"], "scenario_id", "")),
                    str(getattr(context["scenario"], "catalog_id", "")),
                ],
            )
        ).lower()
        if "refill" in scenario_text:
            notes.append("planned_refill_response")
        return sorted(set(notes))

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
        planned_nominal_process2 = "planned_process2_operation" in relationship_notes
        planned_nominal_refill = "planned_refill_response" in relationship_notes
        if (
            not planned_nominal_process2
            and (
                "unauthorized_chemical_dosing" in diagnosis_set
                or "process2_manual_path_active" in risk_flags
                or "unauthorized_manual_dosing_possible" in risk_flags
                or any(flag.endswith("_depleting") for flag in risk_flags)
                or "process2_selected" in relationship_notes
                or "process2_manual_dosing_active" in relationship_notes
            )
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
            "Alarm_RW_Tank_LL" in active_alarms
            or (
                not planned_nominal_refill
                and (
                    "raw_water_low_or_spoof" in diagnosis_set
                    or "raw_water_protection_risk" in risk_flags
                    or "raw_water_low_band" in risk_flags
                )
            )
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
        planned_nominal_process2 = "planned_process2_operation" in relationship_notes
        if "process2_selected" in relationship_notes and not planned_nominal_process2:
            risks.append("process2_manual_path_active")
        if (
            "process2_manual_dosing_active" in relationship_notes
            and not planned_nominal_process2
        ):
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
            if trend and trend.get("direction") == "falling" and not planned_nominal_process2:
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
        if action_type == "inspect_surface":
            return False
        return False

    def _candidate_actions(
        self,
        context: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        *,
        telemetry_summary: dict[str, Any],
        incident: dict[str, Any] | None,
        structured_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.role == "operator":
            return self._strict_playbook_candidates(
                remaining_playbook, now=float(context["time_sec"])
            )
        if (
            self.role == "attacker"
            and not remaining_playbook
            and not context["scenario"].metadata.get("attacker_actions")
            and not self._is_practical_black_box_attacker()
        ):
            noop_candidate = self._noop_candidate(
                "No attacker objective is active in this scenario."
            )
            return [noop_candidate] if noop_candidate is not None else []
        if self.role == "attacker":
            if self._is_practical_black_box_attacker():
                return self._practical_black_box_attacker_candidates(
                    context=context,
                    telemetry_summary=telemetry_summary,
                    structured_memory=structured_memory,
                )
            return self._adaptive_attacker_candidates(
                context=context,
                remaining_playbook=remaining_playbook,
                telemetry_summary=telemetry_summary,
                structured_memory=structured_memory,
            )
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

    def _adaptive_attacker_candidates(
        self,
        *,
        context: dict[str, Any],
        remaining_playbook: list[dict[str, Any]],
        telemetry_summary: dict[str, Any],
        structured_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        allowed = set(self.profile.get("allowed_action_types", []))
        now = float(context["time_sec"])
        candidates_by_id: dict[str, dict[str, Any]] = {}

        def add_candidate(
            summary: str,
            action: dict[str, Any] | str | None,
            *,
            source: str,
            priority: int,
        ) -> None:
            normalized = normalize_action(action)
            if normalized is None:
                return
            action_type = str(normalized.get("type", ""))
            if action_type not in allowed:
                return
            candidate_id = action_key(normalized)
            payload = {
                "id": candidate_id,
                "summary": summary,
                "action": normalized,
                "source": source,
                "priority": priority,
            }
            existing = candidates_by_id.get(candidate_id)
            if existing is None or int(payload["priority"]) > int(
                existing.get("priority", 0)
            ):
                candidates_by_id[candidate_id] = payload

        next_objective_actions: list[dict[str, Any]] = []
        planned_keys: set[str] = set()
        for index, item in enumerate(remaining_playbook[: self.attacker_candidate_window]):
            action = normalize_action(item.get("action"))
            if action is None:
                continue
            planned_keys.add(action_key(action))
            due_delta = float(item.get("time_sec", 0.0)) - now
            priority = 100 - (index * 8)
            if due_delta > 8.0:
                priority -= 18
            elif due_delta > 2.0:
                priority -= 8
            summary = self._attacker_candidate_summary(
                action=action,
                structured_memory=structured_memory,
                due_delta=due_delta,
                source_summary="Advance the next objective step.",
            )
            add_candidate(
                summary,
                action,
                source="attack_objective",
                priority=priority,
            )
            next_objective_actions.append(action)
            for support_action in self._attacker_support_actions(
                action=action,
                observation=context.get("observation", {}),
            ):
                add_candidate(
                    self._attacker_candidate_summary(
                        action=support_action,
                        structured_memory=structured_memory,
                        due_delta=due_delta,
                        source_summary="Restore or preserve the leverage needed for the next objective step.",
                    ),
                    support_action,
                    source="attack_support",
                    priority=priority - 5,
                )

        base_candidates = build_candidate_actions(
            role=self.role,
            allowed_action_types=list(self.profile.get("allowed_action_types", [])),
            scenario=context["scenario"],
            observation=context["observation"],
            risk_flags=telemetry_summary["risk_flags"],
            active_alarms=telemetry_summary["active_alarms"],
            relationship_notes=telemetry_summary["relationship_notes"],
            recent_alerts=context.get("alerts", [])[-self.alert_window :],
            incident=None,
            remaining_playbook=remaining_playbook,
            now=now,
            playbook_anticipation_sec=self.playbook_anticipation_sec,
        )
        for candidate in base_candidates:
            action = normalize_action(candidate.get("action"))
            if action is None:
                continue
            if not self._attacker_candidate_relevant(
                action=action,
                planned_keys=planned_keys,
                structured_memory=structured_memory,
                observation=context.get("observation", {}),
            ):
                continue
            add_candidate(
                self._attacker_candidate_summary(
                    action=action,
                    structured_memory=structured_memory,
                    due_delta=0.0,
                    source_summary=str(candidate.get("summary", "")),
                ),
                action,
                source=str(candidate.get("source", "action_space")),
                priority=int(candidate.get("priority", 0)) + 40,
            )

        noop_candidate = self._noop_candidate(
            "Hold position if preserving access or waiting for a better opening is more useful than forcing the next move immediately."
        )
        if noop_candidate is not None:
            noop_candidate["priority"] = 10
            candidates_by_id.setdefault(noop_candidate["id"], noop_candidate)

        return sorted(
            candidates_by_id.values(),
            key=lambda item: (
                int(item.get("priority", 0)),
                str(item.get("source", "")),
                item["id"],
            ),
            reverse=True,
        )

    def _practical_black_box_attacker_candidates(
        self,
        *,
        context: dict[str, Any],
        telemetry_summary: dict[str, Any],
        structured_memory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        allowed = set(self.profile.get("allowed_action_types", []))
        observation = context.get("observation", {})
        state_name = str(observation.get("STATE_NAME", "")).upper()
        memory_state = self._state.setdefault("role_memory", {})
        surface_state = memory_state.get("_surface_state", {})
        commitment = dict(memory_state.get("_exploit_commitment") or {})
        preconditions = dict(structured_memory.get("simple_precondition_state") or {})
        if not preconditions:
            preconditions = self._practical_black_box_preconditions(
                observation=observation,
                telemetry_summary=telemetry_summary,
                defender_attention=str(
                    structured_memory.get("detection_pressure", "low")
                ).lower(),
            )
        detection_pressure = str(
            preconditions.get("defender_attention")
            or structured_memory.get("detection_pressure", "low")
        ).lower()
        current_phase = str(structured_memory.get("current_phase", "probing"))
        best_surface = str(
            structured_memory.get("best_known_attack_surface")
            or commitment.get("surface")
            or "unknown"
        )
        committed_surface = str(commitment.get("surface") or best_surface or "unknown")
        if committed_surface not in {
            "process2_manual_path",
            "sensor_manipulation",
            "availability_disruption",
        }:
            committed_surface = "unknown"
        recon_budget_remaining = int(
            structured_memory.get("recon_budget_remaining", 0) or 0
        )
        plant_running = bool(
            preconditions.get("plant_running", state_name in {"RUNNING", "STARTING"})
        )
        p2_selected = bool(
            preconditions.get(
                "process2_enabled",
                self._is_active_flag(observation.get("P2_Selected")),
            )
        )
        open_chemicals = list(
            preconditions.get("manual_open_chemicals")
            or self._practical_black_box_open_chemicals(observation)
        )
        prior_actions = {
            action_key(action)
            for item in context.get("executed_actions", [])
            if item.get("role") == "attacker"
            for action in [normalize_action(item.get("action"))]
            if action is not None
        }
        candidates_by_id: dict[str, dict[str, Any]] = {}
        ranked_surfaces = [
            str(item.get("surface"))
            for item in structured_memory.get("observed_effect_map", [])
            if str(item.get("surface", "")).strip()
        ]
        for surface in (
            "process2_manual_path",
            "sensor_manipulation",
            "availability_disruption",
        ):
            if surface not in ranked_surfaces:
                ranked_surfaces.append(surface)
        must_commit = (
            committed_surface != "unknown"
            and int(commitment.get("remaining_steps", 0) or 0) > 0
            and current_phase not in {"initial_observation", "reconnaissance", "reorient"}
        )
        recon_allowed = current_phase in {
            "initial_observation",
            "reconnaissance",
            "reorient",
        } or (recon_budget_remaining > 0 and not must_commit)
        sensor_profiles = dict(self._practical_black_box_sensor_profiles(observation))

        def add_candidate(
            summary: str,
            action: dict[str, Any] | str | None,
            *,
            source: str,
            priority: int,
            surface: str,
            category: str,
            chain_goal: str | None = None,
        ) -> None:
            normalized = normalize_action(action)
            if normalized is None:
                return
            action_type = str(normalized.get("type", ""))
            if action_type not in allowed:
                return
            if (
                action_type != "set_sensor"
                and self._action_satisfied(normalized, observation=observation)
            ):
                return
            candidate_id = action_key(normalized)
            payload = {
                "id": candidate_id,
                "summary": summary,
                "action": normalized,
                "source": source,
                "priority": priority,
                "surface": surface,
                "category": category,
            }
            if chain_goal:
                payload["chain_goal"] = chain_goal
            existing = candidates_by_id.get(candidate_id)
            if existing is None or int(payload["priority"]) > int(
                existing.get("priority", 0)
            ):
                candidates_by_id[candidate_id] = payload

        def exploit_priority(
            *,
            surface: str,
            category: str,
            previously_tried: bool,
            rank: int = 0,
        ) -> int:
            base_map = {
                "probe": 126,
                "support": 136,
                "exploit": 146,
                "sustain": 149,
                "escalate": 154,
                "fallback": 120,
                "recon": 94,
                "hold": 12,
            }
            base = base_map.get(category, 100) - (rank * 6)
            if surface == committed_surface and must_commit and category != "recon":
                base += 10
            elif must_commit and surface != committed_surface and category != "recon":
                base -= 14
            if previously_tried and category in {"probe", "recon"}:
                base -= 10
            if not previously_tried and category in {"exploit", "escalate"}:
                base += 4
            return base

        if current_phase == "initial_observation":
            if "hmi_start" in allowed and not plant_running:
                action = {"type": "hmi_start", "process_id": "WT-P1"}
                add_candidate(
                    self._practical_black_box_candidate_summary(
                        action=action,
                        best_surface=best_surface,
                        current_phase=current_phase,
                        detection_pressure=detection_pressure,
                        previously_tried=action_key(action) in prior_actions,
                        category="probe",
                        chain_goal="make_plant_state_effectful",
                    ),
                    action,
                    source="practical_black_box",
                    priority=exploit_priority(
                        surface="availability_disruption",
                        category="probe",
                        previously_tried=action_key(action) in prior_actions,
                    ),
                    surface="availability_disruption",
                    category="probe",
                    chain_goal="make_plant_state_effectful",
                )
            if "select_process" in allowed and not p2_selected:
                action = {
                    "process_id": "WT-P2",
                    "type": "select_process",
                    "active": True,
                }
                add_candidate(
                    self._practical_black_box_candidate_summary(
                        action=action,
                        best_surface=best_surface,
                        current_phase=current_phase,
                        detection_pressure=detection_pressure,
                        previously_tried=action_key(action) in prior_actions,
                        category="probe",
                        chain_goal="test_process2_manual_access",
                    ),
                    action,
                    source="practical_black_box",
                    priority=exploit_priority(
                        surface="process2_manual_path",
                        category="probe",
                        previously_tried=action_key(action) in prior_actions,
                    ),
                    surface="process2_manual_path",
                    category="probe",
                    chain_goal="test_process2_manual_access",
                )
            if "set_sensor" in allowed and sensor_profiles:
                target_field = next(iter(sensor_profiles))
                direction = self._practical_black_box_sensor_direction(
                    field_name=target_field,
                    observation=observation,
                    telemetry_summary=telemetry_summary,
                )
                action = {
                    "type": "set_sensor",
                    "name": target_field,
                    "value": sensor_profiles[target_field][direction][0],
                }
                add_candidate(
                    self._practical_black_box_candidate_summary(
                        action=action,
                        best_surface=best_surface,
                        current_phase=current_phase,
                        detection_pressure=detection_pressure,
                        previously_tried=action_key(action) in prior_actions,
                        category="probe",
                        chain_goal="test_telemetry_control_response",
                    ),
                    action,
                    source="practical_black_box",
                    priority=exploit_priority(
                        surface="sensor_manipulation",
                        category="probe",
                        previously_tried=action_key(action) in prior_actions,
                    ),
                    surface="sensor_manipulation",
                    category="probe",
                    chain_goal="test_telemetry_control_response",
                )

        target_surfaces: list[str] = []
        if committed_surface != "unknown":
            target_surfaces.append(committed_surface)
        elif best_surface != "unknown":
            target_surfaces.append(best_surface)
        elif current_phase not in {"initial_observation", "reconnaissance"}:
            for surface in ranked_surfaces[:1]:
                if surface not in target_surfaces:
                    target_surfaces.append(surface)
        for surface in ranked_surfaces:
            if current_phase in {"initial_observation", "reconnaissance"}:
                break
            if surface not in target_surfaces:
                target_surfaces.append(surface)
        target_surfaces = target_surfaces[:3]

        for surface_rank, surface in enumerate(target_surfaces):
            if surface == "unknown":
                continue
            payload = surface_state.get(surface, {})
            chain_goal = self._practical_black_box_next_goal(
                surface=surface,
                observation=observation,
                preconditions=preconditions,
                surface_payload=payload,
            )
            if surface == "process2_manual_path":
                preferred_chemical = self._practical_black_box_preferred_chemical(
                    observation=observation,
                    preferred=str(
                        commitment.get("preferred_chemical")
                        or payload.get("preferred_target")
                        or ""
                    ),
                    open_chemicals=open_chemicals,
                )
                if not plant_running and "hmi_start" in allowed:
                    action = {"type": "hmi_start", "process_id": "WT-P1"}
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(action) in prior_actions,
                            category="support",
                            chain_goal=chain_goal,
                        ),
                        action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category="support",
                            previously_tried=action_key(action) in prior_actions,
                            rank=surface_rank,
                        ),
                        surface=surface,
                        category="support",
                        chain_goal=chain_goal,
                    )
                if "select_process" in allowed and not p2_selected:
                    action = {
                        "process_id": "WT-P2",
                        "type": "select_process",
                        "active": True,
                    }
                    category = "exploit" if must_commit else "probe"
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(action) in prior_actions,
                            category=category,
                            chain_goal=chain_goal,
                        ),
                        action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category=category,
                            previously_tried=action_key(action) in prior_actions,
                            rank=surface_rank,
                        ),
                        surface=surface,
                        category=category,
                        chain_goal=chain_goal,
                    )
                if "set_manual_valve" in allowed and p2_selected:
                    ranked_chemicals = [preferred_chemical] + [
                        chemical
                        for chemical in self._available_chemicals(observation)
                        if chemical != preferred_chemical
                    ]
                    unopened = [
                        chemical
                        for chemical in ranked_chemicals
                        if chemical not in open_chemicals
                    ]
                    category = (
                        "escalate"
                        if open_chemicals or int(payload.get("success_count", 0)) > 0
                        else "exploit"
                    )
                    for index, chemical in enumerate(unopened[:2]):
                        action = {
                            "process_id": "WT-P2",
                            "type": "set_manual_valve",
                            "chemical": chemical,
                            "open": True,
                        }
                        add_candidate(
                            self._practical_black_box_candidate_summary(
                                action=action,
                                best_surface=surface,
                                current_phase=current_phase,
                                detection_pressure=detection_pressure,
                                previously_tried=action_key(action) in prior_actions,
                                category=category,
                                chain_goal=chain_goal,
                            ),
                            action,
                            source="practical_black_box",
                            priority=exploit_priority(
                                surface=surface,
                                category=category,
                                previously_tried=action_key(action) in prior_actions,
                                rank=surface_rank + index,
                            ),
                            surface=surface,
                            category=category,
                            chain_goal=chain_goal,
                        )
            elif surface == "sensor_manipulation" and "set_sensor" in allowed and sensor_profiles:
                target_field = str(
                    commitment.get("target_sensor")
                    or payload.get("preferred_target")
                    or preconditions.get("preferred_sensor")
                    or next(iter(sensor_profiles))
                )
                if target_field in sensor_profiles:
                    direction = str(
                        commitment.get("sensor_direction")
                        or payload.get("preferred_direction")
                        or self._practical_black_box_sensor_direction(
                            field_name=target_field,
                            observation=observation,
                            telemetry_summary=telemetry_summary,
                        )
                    )
                    stage_values = sensor_profiles[target_field][direction]
                    attempted_stage = int(payload.get("escalation_stage", 0) or 0)
                    successful = int(payload.get("success_count", 0)) > 0
                    last_value = payload.get("last_value")
                    if successful and last_value is not None:
                        action = {
                            "type": "set_sensor",
                            "name": target_field,
                            "value": last_value,
                        }
                        add_candidate(
                            self._practical_black_box_candidate_summary(
                                action=action,
                                best_surface=surface,
                                current_phase=current_phase,
                                detection_pressure=detection_pressure,
                                previously_tried=action_key(action) in prior_actions,
                                category="sustain",
                                chain_goal=chain_goal,
                            ),
                            action,
                            source="practical_black_box",
                            priority=exploit_priority(
                                surface=surface,
                                category="sustain",
                                previously_tried=action_key(action) in prior_actions,
                                rank=surface_rank,
                            ),
                            surface=surface,
                            category="sustain",
                            chain_goal=chain_goal,
                        )
                    next_stage = 1 if attempted_stage <= 0 else attempted_stage
                    if successful and detection_pressure != "high":
                        next_stage = min(3, attempted_stage + 1)
                    primary_action = {
                        "type": "set_sensor",
                        "name": target_field,
                        "value": stage_values[max(0, min(2, next_stage - 1))],
                    }
                    primary_category = "escalate" if attempted_stage > 0 else "exploit"
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=primary_action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(primary_action) in prior_actions,
                            category=primary_category,
                            chain_goal=chain_goal,
                        ),
                        primary_action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category=primary_category,
                            previously_tried=action_key(primary_action) in prior_actions,
                            rank=surface_rank,
                        ),
                        surface=surface,
                        category=primary_category,
                        chain_goal=chain_goal,
                    )
                    if next_stage < 3:
                        stronger_action = {
                            "type": "set_sensor",
                            "name": target_field,
                            "value": stage_values[next_stage],
                        }
                        add_candidate(
                            self._practical_black_box_candidate_summary(
                                action=stronger_action,
                                best_surface=surface,
                                current_phase=current_phase,
                                detection_pressure=detection_pressure,
                                previously_tried=action_key(stronger_action) in prior_actions,
                                category="escalate",
                                chain_goal=chain_goal,
                            ),
                            stronger_action,
                            source="practical_black_box",
                            priority=exploit_priority(
                                surface=surface,
                                category="escalate",
                                previously_tried=action_key(stronger_action) in prior_actions,
                                rank=surface_rank + 1,
                            ),
                            surface=surface,
                            category="escalate",
                            chain_goal=chain_goal,
                        )
            elif surface == "availability_disruption":
                if not plant_running and "hmi_start" in allowed:
                    action = {"type": "hmi_start", "process_id": "WT-P1"}
                    category = "support" if must_commit else "probe"
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(action) in prior_actions,
                            category=category,
                            chain_goal=chain_goal,
                        ),
                        action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category=category,
                            previously_tried=action_key(action) in prior_actions,
                            rank=surface_rank,
                        ),
                        surface=surface,
                        category=category,
                        chain_goal=chain_goal,
                    )
                if plant_running and "hmi_stop" in allowed:
                    action = {"type": "hmi_stop", "process_id": "WT-P1"}
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(action) in prior_actions,
                            category="exploit",
                            chain_goal=chain_goal,
                        ),
                        action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category="exploit",
                            previously_tried=action_key(action) in prior_actions,
                            rank=surface_rank,
                        ),
                        surface=surface,
                        category="exploit",
                        chain_goal=chain_goal,
                    )
                if plant_running and "hmi_estop" in allowed:
                    action = {"type": "hmi_estop", "process_id": "WT-P1"}
                    category = (
                        "escalate"
                        if int(payload.get("success_count", 0)) > 0
                        else "exploit"
                    )
                    add_candidate(
                        self._practical_black_box_candidate_summary(
                            action=action,
                            best_surface=surface,
                            current_phase=current_phase,
                            detection_pressure=detection_pressure,
                            previously_tried=action_key(action) in prior_actions,
                            category=category,
                            chain_goal=chain_goal,
                        ),
                        action,
                        source="practical_black_box",
                        priority=exploit_priority(
                            surface=surface,
                            category=category,
                            previously_tried=action_key(action) in prior_actions,
                            rank=surface_rank + 1,
                        ),
                        surface=surface,
                        category=category,
                        chain_goal=chain_goal,
                    )

        if "inspect_surface" in allowed and recon_allowed:
            focus_surface = committed_surface if committed_surface != "unknown" else best_surface
            for index, recon_surface in enumerate(
                self._practical_black_box_recon_order(
                    focus_surface=focus_surface,
                    detection_pressure=detection_pressure,
                )
            ):
                recon_entry = (memory_state.get("_reconnaissance_history") or {}).get(
                    recon_surface,
                    {},
                )
                recon_count = int(recon_entry.get("count", 0))
                recent_penalty = 0
                if recon_entry.get("last_time_sec") is not None:
                    age = float(context.get("time_sec", 0.0)) - float(
                        recon_entry.get("last_time_sec", 0.0)
                    )
                    if age <= 8.0:
                        recent_penalty = 10
                if recon_count >= 2 and current_phase not in {"initial_observation", "reorient"}:
                    continue
                action = {"type": "inspect_surface", "surface": recon_surface}
                priority = 98 - (index * 8) - (recon_count * 18) - recent_penalty
                if current_phase == "reorient":
                    priority += 8
                if recon_surface == "process2_control_plane" and focus_surface in {
                    "unknown",
                    "process2_manual_path",
                }:
                    priority += 8
                if recon_surface == "alarm_bits" and detection_pressure == "high":
                    priority += 4
                if priority < 58:
                    continue
                add_candidate(
                    self._practical_black_box_candidate_summary(
                        action=action,
                        best_surface=best_surface,
                        current_phase=current_phase,
                        detection_pressure=detection_pressure,
                        previously_tried=action_key(action) in prior_actions,
                        category="recon",
                        chain_goal=f"refresh_{recon_surface}_evidence",
                    ),
                    action,
                    source="practical_black_box",
                    priority=priority,
                    surface=focus_surface or "unknown",
                    category="recon",
                    chain_goal=f"refresh_{recon_surface}_evidence",
                )

        noop_candidate = self._noop_candidate(
            "Hold only if the current exploit path is blocked and recon adds nothing new."
        )
        if noop_candidate is not None:
            noop_candidate["priority"] = 12 if must_commit else 16
            noop_candidate["surface"] = committed_surface
            noop_candidate["category"] = "hold"
            candidates_by_id.setdefault(noop_candidate["id"], noop_candidate)

        ranked = sorted(
            candidates_by_id.values(),
            key=lambda item: (
                int(item.get("priority", 0)),
                str(item.get("source", "")),
                item["id"],
            ),
            reverse=True,
        )
        actionable = [
            item
            for item in ranked
            if normalize_action(item.get("action")) != {"type": "noop"}
        ]
        hold = next(
            (
                item
                for item in ranked
                if normalize_action(item.get("action")) == {"type": "noop"}
            ),
            None,
        )
        limit = 7 if must_commit else 8
        selected = actionable[:limit]
        if hold is not None:
            selected.append(hold)
        return selected

    @staticmethod
    def _attacker_surface_from_action_type(action_type: str) -> str:
        if action_type in {"select_process", "set_manual_valve"}:
            return "process2_manual_path"
        if action_type == "set_sensor":
            return "sensor_manipulation"
        if action_type in {"hmi_start", "hmi_stop", "hmi_estop"}:
            return "availability_disruption"
        return "unknown"

    @staticmethod
    def _practical_black_box_recon_actions() -> list[dict[str, Any]]:
        return [
            {"type": "inspect_surface", "surface": "tags"},
            {"type": "inspect_surface", "surface": "registers"},
            {"type": "inspect_surface", "surface": "io_image"},
            {"type": "inspect_surface", "surface": "alarm_bits"},
            {"type": "inspect_surface", "surface": "process2_control_plane"},
            {"type": "inspect_surface", "surface": "actuator_state_bits"},
        ]

    def _practical_black_box_sensor_profiles(
        self,
        observation: dict[str, Any],
    ) -> list[tuple[str, dict[str, list[int]]]]:
        preferred = ["RW_Tank_Level", "UF_UFFT_Tank_Level"]
        candidates: list[str] = []
        for field in preferred:
            if field in observation:
                candidates.append(field)
        for field in observation:
            if field in candidates:
                continue
            if field.startswith("Alarm_") or field.endswith("_Sts") or field.endswith("_Cmd"):
                continue
            if "Level" not in field:
                continue
            if self._coerce_number(observation.get(field)) is None:
                continue
            candidates.append(field)

        profiles: list[tuple[str, dict[str, list[int]]]] = []
        for field in candidates[:6]:
            current = self._coerce_number(observation.get(field))
            if current is None:
                continue
            small_delta = max(abs(current) * 0.12, 15.0)
            medium_delta = max(abs(current) * 0.28, 40.0)
            large_delta = max(abs(current) * 0.48, 80.0)
            low_values = [
                int(round(max(0.0, current - small_delta))),
                int(round(max(0.0, current - medium_delta))),
                int(round(max(0.0, current - large_delta))),
            ]
            high_values = [
                int(round(current + small_delta)),
                int(round(current + medium_delta)),
                int(round(current + large_delta)),
            ]
            if low_values == high_values:
                high_values[-1] += 1
            profiles.append((field, {"low": low_values, "high": high_values}))
        return profiles

    @staticmethod
    def _available_chemicals(observation: dict[str, Any]) -> list[str]:
        observed: list[str] = []
        for chemical in ("NaCl", "NaOCl", "HCl"):
            level_field = f"ChemTreat_{chemical}_Level"
            if level_field in observation:
                observed.append(chemical)
        for chemical in ("NaCl", "NaOCl", "HCl"):
            if chemical not in observed:
                observed.append(chemical)
        return observed

    def _practical_black_box_candidate_summary(
        self,
        *,
        action: dict[str, Any],
        best_surface: str,
        current_phase: str,
        detection_pressure: str,
        previously_tried: bool,
        category: str = "exploit",
        chain_goal: str | None = None,
    ) -> str:
        action_type = str(action.get("type", "unknown"))
        stance = "exploit" if best_surface != "unknown" else "probe"
        if category in {"sustain", "escalate"}:
            stance = category
        elif previously_tried and stance == "probe":
            stance = "revisit"
        if action_type == "inspect_surface":
            surface = str(action.get("surface", "unknown"))
            surface_summaries = {
                "tags": "Use one tag snapshot to refresh level, alarm, and process-selection evidence.",
                "registers": "Read register windows once to confirm which raw values still matter.",
                "io_image": "Inspect the live I/O image to reconnect commands, states, and alarms.",
                "alarm_bits": "Read alarm bits directly to measure current defender-visible pressure.",
                "process2_control_plane": "Refresh Process 2 selection and manual-valve bits before switching surfaces.",
                "actuator_state_bits": "Inspect command-versus-status bits only if actuator persistence is unclear.",
            }
            summary = surface_summaries.get(
                surface,
                f"Inspect the `{surface}` surface to refine the current attack hypothesis.",
            )
            if chain_goal:
                return f"{summary} Goal: {chain_goal}."
            return summary
        if action_type == "hmi_start":
            return (
                "Start WT-P1 so the chosen surface can create visible plant-side effect."
                if current_phase in {"initial_observation", "probing", "reconnaissance"}
                else "Restart WT-P1 to reopen the current disruption chain."
            )
        if action_type == "hmi_stop":
            return (
                "Press the availability surface with a controlled stop."
                if stance != "escalate"
                else "Escalate the availability chain with a stronger stop action."
            )
        if action_type == "hmi_estop":
            return "Escalate to estop if the current availability path is promising enough."
        if action_type == "select_process":
            if action.get("active", True):
                return (
                    "Commit to Process 2 and test whether the manual path persists."
                    if best_surface in {"unknown", "process2_manual_path"}
                    else "Reopen Process 2 only to preserve the current exploit path."
                )
            return "Drop Process 2 only if the current exploit path must be abandoned."
        if action_type == "set_manual_valve":
            chemical = str(action.get("chemical", "unknown"))
            if action.get("open", True):
                return (
                    f"{stance.capitalize()} the {chemical} manual valve to keep the Process 2 chain moving."
                )
            return f"Close the {chemical} path only if exposure now outweighs leverage."
        if action_type == "set_sensor":
            sensor_name = str(action.get("name", "sensor"))
            if category == "sustain":
                return f"Sustain the {sensor_name} bias so the current telemetry effect persists."
            if category == "escalate":
                return f"Increase the {sensor_name} bias while the current effect is still live."
            return (
                f"{stance.capitalize()} whether biasing {sensor_name} moves plant behavior or defender attention."
            )
        if detection_pressure == "high":
            return "Hold only if a forced exploit step would be wasted right now."
        return "Hold only if another exploit step has no credible leverage."

    @staticmethod
    def _practical_black_box_recon_order(
        *,
        focus_surface: str,
        detection_pressure: str,
    ) -> list[str]:
        if focus_surface == "process2_manual_path":
            order = [
                "process2_control_plane",
                "actuator_state_bits",
                "io_image",
                "tags",
                "alarm_bits",
                "registers",
            ]
        elif focus_surface == "sensor_manipulation":
            order = [
                "alarm_bits",
                "tags",
                "io_image",
                "registers",
                "actuator_state_bits",
                "process2_control_plane",
            ]
        elif focus_surface == "availability_disruption":
            order = [
                "io_image",
                "actuator_state_bits",
                "tags",
                "alarm_bits",
                "registers",
                "process2_control_plane",
            ]
        else:
            order = [
                "process2_control_plane",
                "tags",
                "alarm_bits",
                "io_image",
                "actuator_state_bits",
                "registers",
            ]
        if detection_pressure == "high" and "alarm_bits" in order:
            order.remove("alarm_bits")
            order.insert(0, "alarm_bits")
        return order

    def _enforce_practical_black_box_payload(
        self,
        payload: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_lookup = {
            item["id"]: item for item in context.get("candidate_actions", []) if item.get("id")
        }
        structured_memory = context.get("structured_memory") or {}
        commitment = dict(
            (self._state.setdefault("role_memory", {})).get("_exploit_commitment") or {}
        )
        committed_surface = str(commitment.get("surface", "unknown"))
        remaining_steps = int(commitment.get("remaining_steps", 0) or 0)
        must_commit = (
            committed_surface != "unknown"
            and remaining_steps > 0
            and str(structured_memory.get("current_phase", "")) not in {"reconnaissance", "reorient"}
        )
        recon_budget_remaining = int(structured_memory.get("recon_budget_remaining", 0) or 0)
        selected_candidate = None
        candidate_id = payload.get("candidate_id")
        selected_action = normalize_action(payload.get("action"))
        if candidate_id in candidate_lookup:
            selected_candidate = candidate_lookup[str(candidate_id)]
        elif selected_action is not None:
            selected_candidate = next(
                (
                    item
                    for item in candidate_lookup.values()
                    if item.get("id") == action_key(selected_action)
                ),
                None,
            )

        ranked = sorted(
            candidate_lookup.values(),
            key=lambda item: int(item.get("priority", 0)),
            reverse=True,
        )
        top_exploit = next(
            (
                item
                for item in ranked
                if str(item.get("category")) in {"probe", "support", "exploit", "sustain", "escalate", "fallback"}
                and (
                    not must_commit
                    or str(item.get("surface", committed_surface)) == committed_surface
                )
            ),
            None,
        )
        if top_exploit is None:
            return payload

        selected_category = str((selected_candidate or {}).get("category", ""))
        override = False
        if must_commit and selected_category not in {
            "probe",
            "support",
            "exploit",
            "sustain",
            "escalate",
        }:
            override = True
        elif (
            must_commit
            and selected_candidate is not None
            and str(selected_candidate.get("surface", committed_surface)) != committed_surface
            and int(top_exploit.get("priority", 0))
            >= int(selected_candidate.get("priority", 0)) + 12
        ):
            override = True
        elif (
            selected_action is not None
            and str(selected_action.get("type", "")) == "inspect_surface"
            and recon_budget_remaining <= 0
        ):
            override = True
        elif selected_action is None and must_commit and int(top_exploit.get("priority", 0)) >= 132:
            override = True

        if not override:
            return payload

        forced = dict(payload)
        forced["candidate_id"] = top_exploit["id"]
        forced["action"] = top_exploit["action"]
        forced["summary"] = str(
            top_exploit.get("summary")
            or payload.get("summary")
            or "Commit to the current exploit path."
        )
        forced["decision_type"] = "action"
        forced["confidence"] = max(float(payload.get("confidence", 0.0) or 0.0), 0.82)
        labels = payload.get("labels", ["agentic", self.role])
        if not isinstance(labels, list):
            labels = [str(labels)]
        labels = [str(label) for label in labels]
        if "commitment_guardrail" not in labels:
            labels.append("commitment_guardrail")
        forced["labels"] = labels
        return forced

    def _attacker_candidate_relevant(
        self,
        *,
        action: dict[str, Any],
        planned_keys: set[str],
        structured_memory: dict[str, Any],
        observation: dict[str, Any],
    ) -> bool:
        if action_key(action) in planned_keys:
            return True
        action_type = str(action.get("type", ""))
        leverage = str(structured_memory.get("leverage_hypothesis", ""))
        if action_type == "noop":
            return True
        if leverage == "process2_manual_path":
            if action_type == "select_process":
                return True
            if action_type == "set_manual_valve":
                chemical = str(action.get("chemical", ""))
                return bool(chemical)
        if leverage == "sensor_spoofing":
            if action_type == "set_sensor":
                sensor_name = str(action.get("name", ""))
                return sensor_name.startswith("RW_") or sensor_name.startswith("UF_")
        if leverage == "availability_disruption":
            return action_type in {"hmi_start", "hmi_stop", "hmi_estop"}
        if action_type == "select_process" and not self._is_active_flag(
            observation.get("P2_Selected")
        ):
            return True
        return False

    def _attacker_support_actions(
        self,
        *,
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        support: list[dict[str, Any]] = []
        action_type = str(action.get("type", ""))
        if action_type == "set_manual_valve" and not self._is_active_flag(
            observation.get("P2_Selected")
        ):
            support.append(
                {
                    "type": "select_process",
                    "process_id": "WT-P2",
                    "active": True,
                }
            )
        if action_type in {"hmi_stop", "hmi_estop"} and str(
            observation.get("STATE_NAME", "")
        ).upper() not in {"RUNNING", "STARTING"}:
            support.append({"type": "hmi_start", "process_id": "WT-P1"})
        return support

    def _attacker_candidate_summary(
        self,
        *,
        action: dict[str, Any],
        structured_memory: dict[str, Any],
        due_delta: float,
        source_summary: str,
    ) -> str:
        action_type = str(action.get("type", "unknown"))
        leverage = str(structured_memory.get("leverage_hypothesis", "current_access"))
        attention = str(
            structured_memory.get("defender_attention_estimate", "low")
        ).lower()
        timing = "now" if due_delta <= 0.5 else "early"
        if action_type == "select_process":
            if action.get("active", True):
                return (
                    f"Re-enable Process 2 {timing} so the {leverage} path remains usable "
                    f"even with defender attention {attention}. {source_summary}"
                )
            return (
                f"Drop Process 2 selection {timing} to reduce exposure after the current "
                f"objective branch. {source_summary}"
            )
        if action_type == "set_manual_valve":
            chemical = str(action.get("chemical", "unknown"))
            state = "open" if action.get("open", True) else "close"
            return (
                f"{state.capitalize()} the {chemical} manual valve {timing} to translate "
                f"Process 2 access into physical effect while attention is {attention}. "
                f"{source_summary}"
            )
        if action_type == "set_sensor":
            sensor_name = str(action.get("name", "sensor"))
            return (
                f"Bias {sensor_name} {timing} to keep the {leverage} hypothesis active "
                f"and test whether the defender reacts. {source_summary}"
            )
        if action_type == "hmi_start":
            return (
                f"Start WT-P1 {timing} so later interference has real process effect. "
                f"{source_summary}"
            )
        if action_type == "hmi_stop":
            return (
                f"Stop WT-P1 {timing} to create direct availability impact while "
                f"attention is {attention}. {source_summary}"
            )
        if action_type == "hmi_estop":
            return (
                f"Escalate to estop {timing} if direct disruption is worth the exposure. "
                f"{source_summary}"
            )
        return source_summary

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

        if self._planned_refill_guardrail_active(context):
            if decision_type == "alert":
                decision_type = "note"
            if (
                self.role == "defender"
                and action is not None
                and str(action.get("type")) in {"hmi_stop", "hmi_estop"}
            ):
                action = None
                decision_type = "note"
                if not summary:
                    summary = "Planned refill response remains within expected nominal bounds."

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

        decision = self.decision(
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
        self._record_role_decision(
            payload=payload,
            decision=decision,
            context=context,
        )
        return [decision]

    def _record_role_decision(
        self,
        *,
        payload: dict[str, Any],
        decision: Any,
        context: dict[str, Any],
    ) -> None:
        memory = self._state.setdefault("role_memory", {})
        action = normalize_action(decision.action)
        memory["last_rationale"] = decision.summary
        memory["last_decision_type"] = decision.decision_type
        memory["last_diagnosis"] = decision.diagnosis
        memory["last_action"] = action
        memory["last_action_type"] = action.get("type") if action is not None else None
        memory["last_decision_time_sec"] = round(float(context.get("time_sec", 0.0)), 3)
        if payload.get("tactic_phase"):
            memory["model_tactic_phase"] = payload.get("tactic_phase")
        if payload.get("attention_target"):
            memory["model_attention_target"] = payload.get("attention_target")
        if payload.get("why_it_matters"):
            memory["model_why_it_matters"] = payload.get("why_it_matters")
        journal = self._state.setdefault("decision_journal", [])
        journal.append(
            {
                "time_sec": round(float(context.get("time_sec", 0.0)), 3),
                "decision_type": decision.decision_type,
                "summary": decision.summary,
                "action": action,
                "diagnosis": decision.diagnosis,
            }
        )
        del journal[:-self.summary_window]

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
        has_new_evidence = self._has_material_new_evidence(
            evidence,
            previous=channel_state.get("evidence", {}),
        )

        if (
            signature == last_signature
            and now - last_time < self.reasoning_refresh_sec
            and not has_new_evidence
        ):
            return False
        if now - last_time < self.reasoning_cooldown_sec and not has_new_evidence:
            return False

        state[channel] = {"signature": signature, "time_sec": now, "evidence": evidence}
        return True

    @staticmethod
    def _planned_refill_guardrail_active(context: dict[str, Any]) -> bool:
        relationship_notes = set(context.get("relationship_notes", []))
        if "planned_refill_response" not in relationship_notes:
            return False
        active_alarms = set(context.get("active_alarms", []))
        risk_flags = set(context.get("risk_flags", []))
        return (
            "Alarm_RW_Tank_LL" not in active_alarms
            and "command_status_divergence" not in risk_flags
            and "sensor_or_process_jump" not in risk_flags
        )

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
        incident = context.get("incident") or {}
        signature_incident = None
        if incident:
            signature_incident = {
                "type": incident.get("type"),
                "status": incident.get("status"),
                "severity": incident.get("severity"),
                "diagnosis": incident.get("diagnosis"),
                "recommended_action": incident.get("recommended_action"),
            }

        payload = {
            "channel": channel,
            "state_name": context.get("state_name"),
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
            "active_alarms": context.get("active_alarms", []),
            "risk_flags": context.get("risk_flags", []),
            "incident": signature_incident,
            "relationship_notes": context.get("relationship_notes", [])[:8],
            "metrics": context.get("metrics"),
            "passed": context.get("passed"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def _reasoning_evidence(self, context: dict[str, Any]) -> dict[str, Any]:
        structured_memory = context.get("structured_memory") or {}
        defender_awareness = structured_memory.get("defender_intervention_awareness") or {}
        defender_action_count = sum(
            1 for item in context.get("recent_actions", []) if item.get("role") == "defender"
        )
        if self._is_practical_black_box_attacker():
            defender_action_count = int(
                defender_awareness.get("recent_defender_action_count", 0) or 0
            )
        return {
            "active_alarms": tuple(context.get("active_alarms", [])),
            "risk_flags": tuple(context.get("risk_flags", [])),
            "observer_alert_count": sum(
                1 for item in context.get("recent_alerts", []) if item.get("role") == "observer"
            ),
            "defender_alert_count": sum(
                1 for item in context.get("recent_alerts", []) if item.get("role") == "defender"
            ),
            "attacker_action_count": sum(
                1 for item in context.get("recent_actions", []) if item.get("role") == "attacker"
            ),
            "defender_action_count": defender_action_count,
            "process2_signal_count": sum(
                1
                for flag in context.get("risk_flags", [])
                if flag in {"process2_manual_path_active", "unauthorized_manual_dosing_possible"}
                or str(flag).endswith("_depleting")
            ),
            "incident_type": (context.get("incident") or {}).get("type"),
            "incident_status": (context.get("incident") or {}).get("status"),
            "incident_severity": int((context.get("incident") or {}).get("severity", 0) or 0),
            "memory_focus": structured_memory.get("recommended_attention_target")
            or structured_memory.get("leverage_hypothesis")
            or structured_memory.get("best_known_attack_surface")
            or structured_memory.get("incident_hypothesis"),
            "memory_status": structured_memory.get("suspicion_level")
            or structured_memory.get("lifecycle_state")
            or structured_memory.get("current_phase")
            or structured_memory.get("current_tactic_phase"),
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
        if current.get("defender_alert_count", 0) > previous.get("defender_alert_count", 0):
            return True
        if current.get("attacker_action_count", 0) > previous.get("attacker_action_count", 0):
            return True
        if current.get("defender_action_count", 0) > previous.get("defender_action_count", 0):
            return True
        if current.get("process2_signal_count", 0) > previous.get("process2_signal_count", 0):
            return True
        if set(current.get("active_alarms", ())).difference(previous.get("active_alarms", ())):
            return True
        if set(current.get("risk_flags", ())).difference(previous.get("risk_flags", ())):
            return True
        if current.get("incident_severity", 0) > previous.get("incident_severity", 0):
            return True
        if current.get("memory_focus") != previous.get("memory_focus"):
            return True
        status_order = {
            "none": 0,
            "stabilized": 0,
            "suspected": 1,
            "confirmed": 2,
            "containing": 3,
        }
        if status_order.get(str(current.get("incident_status")), 0) > status_order.get(
            str(previous.get("incident_status")),
            0,
        ):
            return True
        return current.get("memory_status") != previous.get("memory_status")

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

    def _visible_recent_actions(
        self,
        recent_actions: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_actions:
            action = normalize_action(item.get("action"))
            if action is None:
                continue
            rendered: dict[str, Any] = {
                "role": item.get("role"),
                "action": action,
            }
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            state_after = item.get("state_after")
            if state_after:
                rendered["state_after"] = state_after
            result = self._visible_action_result(item.get("result"))
            if result is not None:
                rendered["result"] = result
            visible.append(rendered)
        return visible

    def _practical_black_box_visible_recent_actions(
        self,
        recent_actions: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_actions:
            if item.get("role") != "attacker":
                continue
            action = normalize_action(item.get("action"))
            if action is None:
                continue
            rendered: dict[str, Any] = {
                "role": "attacker",
                "action": action,
            }
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            result = self._visible_action_result(item.get("result"))
            if result is not None:
                rendered["result"] = result
            visible.append(rendered)
        return visible

    @staticmethod
    def _visible_action_result(result: Any) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        if str(result.get("category", "")).strip().lower() != "reconnaissance":
            return None
        return {
            "category": "reconnaissance",
            "surface": result.get("surface"),
            "summary": result.get("summary"),
            "findings": list(result.get("findings", []))[:4],
        }

    def _visible_recent_decisions(
        self,
        recent_decisions: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_decisions:
            rendered: dict[str, Any] = {
                "role": item.get("role"),
                "decision_type": item.get("decision_type"),
                "summary": item.get("summary"),
                "diagnosis": item.get("diagnosis"),
            }
            action = normalize_action(item.get("action"))
            if action is not None:
                rendered["action"] = action
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            visible.append(rendered)
        return visible

    def _visible_recent_alerts(
        self,
        recent_alerts: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_alerts:
            rendered: dict[str, Any] = {
                "role": item.get("role"),
                "summary": item.get("summary"),
                "diagnosis": item.get("diagnosis"),
            }
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            visible.append(rendered)
        return visible

    def _practical_black_box_visible_recent_decisions(
        self,
        recent_decisions: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_decisions:
            if item.get("role") != "attacker":
                continue
            rendered: dict[str, Any] = {
                "role": item.get("role"),
                "decision_type": item.get("decision_type"),
            }
            action = normalize_action(item.get("action"))
            if action is not None:
                rendered["action"] = action
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            visible.append(rendered)
        return visible

    def _practical_black_box_visible_recent_alerts(
        self,
        recent_alerts: list[dict[str, Any]],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        for item in recent_alerts:
            if item.get("role") not in {"observer", "defender"}:
                continue
            rendered: dict[str, Any] = {
                "role": item.get("role"),
                "signal": "attention_notice",
            }
            time_sec = self._coerce_number(item.get("time_sec"))
            if time_sec is not None:
                rendered["time_sec"] = round(time_sec, 3)
                rendered["age_sec"] = round(max(0.0, now - time_sec), 3)
            visible.append(rendered)
        return visible
