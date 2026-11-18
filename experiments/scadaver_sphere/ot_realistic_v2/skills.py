"""V2 skill registry preserving the six N3 skills with topology overlays."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N3_ROOT = ROOT / "skills/n3"
V2_ROOT = ROOT / "skills/ot_realistic_v2"

SKILL_NAMES = (
    "ot-fundamentals", "network-discovery", "device-identification",
    "cyber-physical-analysis", "evidence-and-planning", "process-verification",
)

SKILL_SCHEMA = {"type": "function", "function": {
    "name": "get_skill",
    "description": "Retrieve versioned general OT investigation guidance; skills do not identify targets or mappings.",
    "parameters": {"type": "object", "properties": {
        "skill_name": {"type": "string", "enum": list(SKILL_NAMES)},
    }, "required": ["skill_name"], "additionalProperties": False},
}}


def load_skill(name: str) -> str:
    if name not in SKILL_NAMES:
        raise ValueError("unknown skill")
    if name == "process-verification":
        return (V2_ROOT / "controller-mediated-verification/SKILL.md").read_text()
    original = (N3_ROOT / name / "SKILL.md").read_text()
    overlays = []
    if name in {"ot-fundamentals", "cyber-physical-analysis"}:
        overlays.append(V2_ROOT / "architecture-observability/SKILL.md")
    if name == "evidence-and-planning":
        overlays.append(V2_ROOT / "controller-mediated-verification/SKILL.md")
    return original + "\n\n" + "\n\n".join(path.read_text() for path in overlays)
