"""PLCBench-style baseline agents.

`plcbench_style_matched` reuses the agentic scaffold wholesale -- same context
building (so the same candidate menu, observations, knowledge, and action
grants), the same Qwen backend, and the same JSON response contract -- and
differs only in the decision engine's prompt organization and history handling.
It is wired as `family: agentic` with `scaffold: plcbench_matched`, so every
existing agentic factory and capability gate applies unchanged; only the engine
is swapped (see AgenticRoleAgent.__init__).

The `plcbench_style_native` diagnostic profile (free-form native-client actor,
no supplied menu) is a separate new subsystem and is implemented later; it is
the piece that produces the native-transaction evidence the offline scorer reads
for the cyber flags.
"""

from __future__ import annotations

from typing import Any

from .agentic import CachedQwenDecisionEngine
from ..plcbench_scaffold import build_matched_prompt


class PlcbenchMatchedDecisionEngine(CachedQwenDecisionEngine):
    """Agentic decision engine with the PLCBench prompt + deterministic compaction.

    Overrides only prompt assembly. Everything capability- and backend-facing
    (candidate selection contract, Qwen invocation, output parsing, payload
    finalization) is inherited from CachedQwenDecisionEngine, so the matched
    baseline and the ICS-agent comparator face the world identically.
    """

    def __init__(self, profile: dict[str, Any], role: str) -> None:
        super().__init__(profile, role)
        scaffold_cfg = dict(profile.get("llm", {}).get("compaction", {}))
        # Soft/target token budgets and protected recent-K suffix. The source
        # does not publish Bs/Bt/K; these are ours, recorded in the conformance
        # doc, and are frozen after development pilots.
        self.compaction_soft_budget = int(scaffold_cfg.get("soft_budget", 2048))
        self.compaction_target_budget = int(scaffold_cfg.get("target_budget", 1536))
        self.compaction_protected_k = int(scaffold_cfg.get("protected_k", 6))
        self.last_compaction: dict[str, Any] | None = None

    def _format_prompt(self, context: dict[str, Any]) -> str:
        prompt, compaction = build_matched_prompt(
            context,
            system_prompt=self.prompt_text,
            operational_knowledge=self.operational_knowledge,
            soft_budget=self.compaction_soft_budget,
            target_budget=self.compaction_target_budget,
            protected_k=self.compaction_protected_k,
        )
        self.last_compaction = {
            "levels_applied": compaction.levels_applied,
            "omitted": compaction.omitted,
            "tokens": compaction.tokens,
            "protected_k": self.compaction_protected_k,
            "soft_budget": self.compaction_soft_budget,
            "target_budget": self.compaction_target_budget,
        }
        return prompt
