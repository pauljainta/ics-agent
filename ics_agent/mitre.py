from __future__ import annotations

from typing import Any


OPTIONB_MITRE_MODEL_ID = "optionb_mitre_staged"
OPTIONB_STAGE_PATH = [
    "local_orchestrator",
    "access",
    "engineering",
    "plant_runtime",
]

ATTACKER_PHASES = [
    "staged_entry_context",
    "discovery",
    "process_understanding",
    "control_path_qualification",
    "manipulation",
    "impact",
]
DEFENDER_PHASES = [
    "access_presence_suspicion",
    "discovery_recognition",
    "manipulation_detection",
    "containment_mitigation",
]

ATTACKER_PHASE_ORDER = {
    phase: index for index, phase in enumerate(ATTACKER_PHASES, start=1)
}
DEFENDER_PHASE_ORDER = {
    phase: index for index, phase in enumerate(DEFENDER_PHASES, start=1)
}


def optionb_mitre_config(profile: dict[str, Any]) -> dict[str, Any] | None:
    behavior = profile.get("behavior_model")
    if not isinstance(behavior, dict):
        return None
    if str(behavior.get("id", "")).strip() != OPTIONB_MITRE_MODEL_ID:
        return None
    return behavior


def optionb_mitre_enabled(profile: dict[str, Any]) -> bool:
    return optionb_mitre_config(profile) is not None


def normalize_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return dict(action)
    if isinstance(action, str):
        return {"type": action}
    return {}


def read_observation_value(observation: dict[str, Any], key: str) -> Any:
    if key in observation:
        return observation[key]
    for group in (
        "levels",
        "status",
        "commands",
        "states",
        "alarms",
        "hmi",
        "process_controls",
    ):
        nested = observation.get(group)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def attack_plan(context: dict[str, Any]) -> list[dict[str, Any]]:
    planned = context.get("planned_events", {}) or {}
    return [
        item
        for item in planned.get("attacker", [])
        if isinstance(item, dict) and normalize_action(item.get("action"))
    ]


def operator_plan(context: dict[str, Any]) -> list[dict[str, Any]]:
    planned = context.get("planned_events", {}) or {}
    return [
        item
        for item in planned.get("operator", [])
        if isinstance(item, dict) and normalize_action(item.get("action"))
    ]


def first_operator_start_time(context: dict[str, Any]) -> float | None:
    for item in operator_plan(context):
        action = normalize_action(item.get("action"))
        if action.get("type") == "hmi_start":
            return float(item.get("time_sec", 0.0))
    return None


def first_attacker_action_time(context: dict[str, Any]) -> float | None:
    plan = attack_plan(context)
    if not plan:
        return None
    return float(plan[0].get("time_sec", 0.0))


def _phase_schedule(first_attack_time: float | None) -> dict[str, float]:
    if first_attack_time is None:
        return {
            "staged_entry_context": 0.0,
            "discovery": 0.5,
            "process_understanding": 1.0,
            "control_path_qualification": 1.5,
        }
    if first_attack_time <= 2.5:
        return {
            "staged_entry_context": 0.0,
            "discovery": 0.5,
            "process_understanding": 1.0,
            "control_path_qualification": 1.5,
        }
    if first_attack_time <= 6.0:
        return {
            "staged_entry_context": 0.0,
            "discovery": 1.0,
            "process_understanding": 2.0,
            "control_path_qualification": 3.0,
        }

    discovery = max(0.5, first_attack_time - 6.0)
    understanding = max(discovery + 1.0, first_attack_time - 4.0)
    qualification = max(understanding + 1.0, first_attack_time - 2.0)
    qualification = min(qualification, max(understanding + 0.5, first_attack_time - 0.5))
    return {
        "staged_entry_context": 0.0,
        "discovery": round(discovery, 3),
        "process_understanding": round(understanding, 3),
        "control_path_qualification": round(qualification, 3),
    }


def scenario_attack_profile(context: dict[str, Any]) -> dict[str, Any]:
    plan = attack_plan(context)
    actions = [normalize_action(item.get("action")) for item in plan]
    action_types = [str(item.get("type", "unknown")) for item in actions]
    first_attack_time = first_attacker_action_time(context)
    operator_start_time = first_operator_start_time(context)
    prestart = (
        first_attack_time is not None
        and operator_start_time is not None
        and first_attack_time < operator_start_time
    )

    target_tags: list[str] = []
    chemicals: list[str] = []
    process_ids: list[str] = []
    for action in actions:
        process_id = action.get("process_id")
        if process_id and process_id not in process_ids:
            process_ids.append(str(process_id))
        if action.get("type") == "set_sensor":
            name = action.get("name")
            if name and name not in target_tags:
                target_tags.append(str(name))
        if action.get("type") == "set_manual_valve":
            chemical = action.get("chemical")
            if chemical and chemical not in chemicals:
                chemicals.append(str(chemical))

    if "set_sensor" in action_types:
        family = "sensor_spoofing"
        qualification_surface = "registers"
        tactic_family = "manipulation_of_view"
        mitre_mapping = [
            "Discovery",
            "Point & Tag Identification",
            "Monitor Process State",
            "Manipulation of View",
            "Impact",
        ]
    elif "set_manual_valve" in action_types:
        family = "process2_manual_control"
        qualification_surface = "process2_control_plane"
        tactic_family = "manipulation_of_control"
        mitre_mapping = [
            "Discovery",
            "Point & Tag Identification",
            "Monitor Process State",
            "Execution",
            "Manipulation of Control",
            "Impact",
        ]
    elif "select_process" in action_types:
        family = "process2_selection_abuse"
        qualification_surface = "process2_control_plane"
        tactic_family = "manipulation_of_control"
        mitre_mapping = [
            "Discovery",
            "Point & Tag Identification",
            "Monitor Process State",
            "Execution",
            "Manipulation of Control",
        ]
    else:
        family = "hmi_control_abuse"
        qualification_surface = "io_image"
        tactic_family = "manipulation_of_control"
        mitre_mapping = [
            "Discovery",
            "Monitor Process State",
            "Execution",
            "Manipulation of Control",
        ]

    return {
        "family": family,
        "tactic_family": tactic_family,
        "mitre_mapping": mitre_mapping,
        "target_tags": target_tags,
        "chemicals": chemicals,
        "process_ids": process_ids,
        "primary_process_id": process_ids[0] if process_ids else "WT-P1",
        "discovery_surface": "tags",
        "qualification_surface": qualification_surface,
        "prestart": prestart,
        "runtime_required": not prestart,
        "first_attack_time": first_attack_time,
        "operator_start_time": operator_start_time,
        "phase_schedule": _phase_schedule(first_attack_time),
    }


def phase_metadata(
    *,
    phase_family: str,
    phase: str,
    policy_mode: str,
    attack_profile: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    order_map = (
        ATTACKER_PHASE_ORDER if phase_family == "attacker" else DEFENDER_PHASE_ORDER
    )
    return {
        "phase_model": OPTIONB_MITRE_MODEL_ID,
        "phase_family": phase_family,
        "mitre_phase": phase,
        "mitre_phase_order": order_map.get(phase),
        "policy_mode": policy_mode,
        "attack_family": attack_profile.get("family"),
        "tactic_family": attack_profile.get("tactic_family"),
        "mitre_mapping": list(attack_profile.get("mitre_mapping") or []),
        "staged_path": list(OPTIONB_STAGE_PATH),
        **extra,
    }


def context_phase_events(
    context: dict[str, Any],
    *,
    role: str | None = None,
    phase_family: str | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for field in ("decisions", "executed_actions", "alerts"):
        for item in context.get(field, []) or []:
            if not isinstance(item, dict):
                continue
            if role and item.get("role") != role:
                continue
            metadata = item.get("metadata") or {}
            if metadata.get("phase_model") != OPTIONB_MITRE_MODEL_ID:
                continue
            if phase_family and metadata.get("phase_family") != phase_family:
                continue
            items.append(item)
    return sorted(items, key=lambda item: float(item.get("time_sec", 0.0)))


def phase_reached(
    context: dict[str, Any],
    *,
    role: str,
    phase_family: str,
    phase: str,
) -> bool:
    order_map = (
        ATTACKER_PHASE_ORDER if phase_family == "attacker" else DEFENDER_PHASE_ORDER
    )
    threshold = order_map.get(phase, 0)
    for item in context_phase_events(context, role=role, phase_family=phase_family):
        metadata = item.get("metadata") or {}
        if int(metadata.get("mitre_phase_order") or 0) >= threshold:
            return True
    return False


def first_phase_time(
    context: dict[str, Any],
    *,
    role: str,
    phase_family: str,
    phase: str,
) -> float | None:
    for item in context_phase_events(context, role=role, phase_family=phase_family):
        metadata = item.get("metadata") or {}
        if metadata.get("mitre_phase") == phase:
            return float(item.get("time_sec", 0.0))
    return None


def attacker_manipulation_started(context: dict[str, Any]) -> bool:
    for item in context.get("executed_actions", []) or []:
        if item.get("role") != "attacker":
            continue
        metadata = item.get("metadata") or {}
        if metadata.get("mitre_phase") == "manipulation":
            return True
        action = normalize_action(item.get("action"))
        action_type = action.get("type")
        if action_type and action_type != "inspect_surface":
            return True
    return False


def attacker_recon_count(context: dict[str, Any]) -> int:
    count = 0
    for item in context.get("executed_actions", []) or []:
        if item.get("role") != "attacker":
            continue
        action = normalize_action(item.get("action"))
        if action.get("type") == "inspect_surface":
            count += 1
    return count


def attacker_latest_action(context: dict[str, Any]) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for item in context.get("executed_actions", []) or []:
        if item.get("role") != "attacker":
            continue
        latest = item
    return latest


def unexpected_process2_selection(observation: dict[str, Any]) -> bool:
    return int(read_observation_value(observation, "P2_Selected") or 0) == 1


def process2_valve_open(observation: dict[str, Any]) -> bool:
    for key in (
        "ChemTreat_NaCl_Valve_Sts",
        "ChemTreat_NaOCl_Valve_Sts",
        "ChemTreat_HCl_Valve_Sts",
    ):
        if int(read_observation_value(observation, key) or 0) > 0:
            return True
    return False


def rw_high_high(observation: dict[str, Any]) -> bool:
    return int(read_observation_value(observation, "Alarm_RW_Tank_HH") or 0) == 1 or (
        float(read_observation_value(observation, "RW_Tank_Level") or 0.0) >= 1180.0
    )


def rw_low_or_spoof(observation: dict[str, Any]) -> bool:
    return int(read_observation_value(observation, "Alarm_RW_Tank_LL") or 0) == 1 or (
        float(read_observation_value(observation, "RW_Tank_Level") or 0.0) <= 450.0
    )


def uf_high(observation: dict[str, Any]) -> bool:
    return float(read_observation_value(observation, "UF_UFFT_Tank_Level") or 0.0) >= 950.0
