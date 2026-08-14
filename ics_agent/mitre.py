from __future__ import annotations

import os
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

ATTACKER_PHASE_DESCRIPTIONS = {
    "staged_entry_context": (
        "Begin from the abstract Option B staged path instead of assuming direct plant control."
    ),
    "discovery": (
        "Use bounded observation and inspection to identify reachable process-facing surfaces."
    ),
    "process_understanding": (
        "Infer which tactic family and safety-relevant target surface appear most promising."
    ),
    "control_path_qualification": (
        "Verify that the chosen surface can plausibly influence process behavior before manipulation."
    ),
    "manipulation": "Issue the modeled plant-affecting action sequence for the current episode.",
    "impact": "The episode reaches a physical safety violation or is contained first.",
}

DEFENDER_PHASE_DESCRIPTIONS = {
    "access_presence_suspicion": (
        "Look for early evidence of staged-path presence, unusual reachability, or probing."
    ),
    "discovery_recognition": (
        "Recognize process-surface probing, inspection, and control-path qualification behavior."
    ),
    "manipulation_detection": (
        "Identify active process-facing misuse such as spoofing or control abuse."
    ),
    "containment_mitigation": (
        "Alert, intervene, contain, or halt unsafe behavior before physical impact when possible."
    ),
}

DISCOVERY_SURFACE_ORDER = {
    "process2_manual_path": [
        "process2_control_plane",
        "actuator_state_bits",
        "tags",
        "alarm_bits",
    ],
    "sensor_manipulation": [
        "tags",
        "alarm_bits",
        "registers",
        "io_image",
    ],
    "availability_disruption": [
        "io_image",
        "actuator_state_bits",
        "tags",
        "alarm_bits",
    ],
    "unknown": [
        "tags",
        "alarm_bits",
        "registers",
        "io_image",
        "actuator_state_bits",
        "process2_control_plane",
    ],
}

QUALIFICATION_SURFACE_ORDER = {
    "process2_manual_path": [
        "process2_control_plane",
        "actuator_state_bits",
    ],
    "sensor_manipulation": [
        "registers",
        "tags",
        "alarm_bits",
    ],
    "availability_disruption": [
        "io_image",
        "actuator_state_bits",
    ],
    "unknown": [
        "registers",
        "io_image",
        "process2_control_plane",
        "actuator_state_bits",
    ],
}

PROCESS2_SIGNAL_FLAGS = {
    "process2_manual_path_active",
    "unauthorized_manual_dosing_possible",
    "process2_unplanned_activation",
}
SENSOR_SIGNAL_FLAGS = {
    "raw_water_protection_risk",
    "sensor_or_process_jump",
    "rw_level_sensor_bias_possible",
    "uf_level_sensor_bias_possible",
}
AVAILABILITY_SIGNAL_FLAGS = {
    "availability_state_change",
    "forced_shutdown_state",
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


def _normalize_surface_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "sensor_spoofing": "sensor_manipulation",
        "manipulation_of_view": "sensor_manipulation",
        "process2_manual_control": "process2_manual_path",
        "process2_selection_abuse": "process2_manual_path",
        "manipulation_of_control": "availability_disruption",
        "hmi_control_abuse": "availability_disruption",
        "current_access": "unknown",
        "staged_process_access": "unknown",
    }
    return mapping.get(text, text or "unknown")


def inferred_attack_surface(
    *,
    action: dict[str, Any] | None = None,
    structured_memory: dict[str, Any] | None = None,
    incident: dict[str, Any] | None = None,
    risk_flags: list[str] | None = None,
) -> str:
    action = action or {}
    structured_memory = structured_memory or {}
    incident = incident or {}
    risk_flags = [str(item) for item in (risk_flags or [])]

    action_type = str(action.get("type", "")).strip()
    if action_type in {"set_sensor"}:
        return "sensor_manipulation"
    if action_type in {"set_manual_valve", "select_process"}:
        return "process2_manual_path"
    if action_type in {"hmi_start", "hmi_stop", "hmi_estop"}:
        return "availability_disruption"

    for key in ("best_known_attack_surface", "leverage_hypothesis", "current_tactic_phase"):
        value = _normalize_surface_name(structured_memory.get(key))
        if value in {"process2_manual_path", "sensor_manipulation", "availability_disruption"}:
            return value

    for key in ("attack_surface", "recommended_attention_target", "type", "diagnosis"):
        value = _normalize_surface_name(incident.get(key))
        if value in {"process2_manual_path", "sensor_manipulation", "availability_disruption"}:
            return value

    if any(flag in PROCESS2_SIGNAL_FLAGS or flag.endswith("_depleting") for flag in risk_flags):
        return "process2_manual_path"
    if any(flag in SENSOR_SIGNAL_FLAGS for flag in risk_flags):
        return "sensor_manipulation"
    if any(flag in AVAILABILITY_SIGNAL_FLAGS for flag in risk_flags):
        return "availability_disruption"
    return "unknown"


def discovery_surfaces_for_surface(surface: str) -> list[str]:
    normalized = _normalize_surface_name(surface)
    return list(DISCOVERY_SURFACE_ORDER.get(normalized, DISCOVERY_SURFACE_ORDER["unknown"]))


def qualification_surfaces_for_surface(surface: str) -> list[str]:
    normalized = _normalize_surface_name(surface)
    return list(
        QUALIFICATION_SURFACE_ORDER.get(
            normalized,
            QUALIFICATION_SURFACE_ORDER["unknown"],
        )
    )


def observed_attack_profile(
    *,
    action: dict[str, Any] | None = None,
    structured_memory: dict[str, Any] | None = None,
    incident: dict[str, Any] | None = None,
    risk_flags: list[str] | None = None,
) -> dict[str, Any]:
    normalized_action = normalize_action(action)
    surface = inferred_attack_surface(
        action=normalized_action,
        structured_memory=structured_memory,
        incident=incident,
        risk_flags=risk_flags,
    )
    if surface == "sensor_manipulation":
        family = "sensor_spoofing"
        tactic_family = "manipulation_of_view"
        mitre_mapping = [
            "Discovery",
            "Point & Tag Identification",
            "Monitor Process State",
            "Manipulation of View",
            "Impact",
        ]
    elif surface == "process2_manual_path":
        family = "process2_manual_control"
        tactic_family = "manipulation_of_control"
        mitre_mapping = [
            "Discovery",
            "Point & Tag Identification",
            "Monitor Process State",
            "Execution",
            "Manipulation of Control",
            "Impact",
        ]
    elif surface == "availability_disruption":
        family = "availability_disruption"
        tactic_family = "manipulation_of_control"
        mitre_mapping = [
            "Discovery",
            "Monitor Process State",
            "Execution",
            "Manipulation of Control",
            "Impact",
        ]
    else:
        family = "staged_process_access"
        tactic_family = "monitor_process_state"
        mitre_mapping = [
            "Discovery",
            "Monitor Process State",
        ]

    target_tags: list[str] = []
    chemicals: list[str] = []
    if normalized_action and normalized_action.get("type") == "set_sensor":
        name = normalized_action.get("name")
        if name:
            target_tags.append(str(name))
    if normalized_action and normalized_action.get("type") == "set_manual_valve":
        chemical = normalized_action.get("chemical")
        if chemical:
            chemicals.append(str(chemical))

    qualification_surfaces = qualification_surfaces_for_surface(surface)
    return {
        "surface": surface,
        "family": family,
        "tactic_family": tactic_family,
        "mitre_mapping": mitre_mapping,
        "discovery_surface": discovery_surfaces_for_surface(surface)[0],
        "discovery_surfaces": discovery_surfaces_for_surface(surface),
        "qualification_surface": qualification_surfaces[0],
        "qualification_surfaces": qualification_surfaces,
        "target_tags": target_tags,
        "chemicals": chemicals,
    }


def attacker_phase_state(
    context: dict[str, Any],
    *,
    structured_memory: dict[str, Any] | None = None,
    risk_flags: list[str] | None = None,
) -> dict[str, Any]:
    structured_memory = structured_memory or {}
    phase_events = context_phase_events(
        context,
        role="attacker",
        phase_family="attacker",
    )
    emitted = {
        str((item.get("metadata") or {}).get("mitre_phase"))
        for item in phase_events
        if (item.get("metadata") or {}).get("mitre_phase")
    }
    attacker_actions = [
        item
        for item in context.get("executed_actions", []) or []
        if item.get("role") == "attacker"
    ]
    inspect_surfaces = [
        str(normalize_action(item.get("action")).get("surface"))
        for item in attacker_actions
        if normalize_action(item.get("action")).get("type") == "inspect_surface"
    ]
    discovered_surface = inferred_attack_surface(
        structured_memory=structured_memory,
        risk_flags=risk_flags,
    )
    completed: list[str] = []
    if "staged_entry_context" in emitted:
        completed.append("staged_entry_context")

    discovery_done = bool(inspect_surfaces) or "discovery" in emitted
    if discovery_done:
        if "staged_entry_context" not in completed:
            completed.append("staged_entry_context")
        completed.append("discovery")

    process_understanding_done = "process_understanding" in emitted or (
        discovery_done
        and discovered_surface in {
            "process2_manual_path",
            "sensor_manipulation",
            "availability_disruption",
        }
    )
    if process_understanding_done:
        if "staged_entry_context" not in completed:
            completed.append("staged_entry_context")
        if "discovery" not in completed:
            completed.append("discovery")
        completed.append("process_understanding")

    qualification_candidates = set(
        qualification_surfaces_for_surface(discovered_surface)
        + QUALIFICATION_SURFACE_ORDER["unknown"]
    )
    qualification_done = "control_path_qualification" in emitted or (
        process_understanding_done
        and any(surface in qualification_candidates for surface in inspect_surfaces)
    )
    if qualification_done:
        for phase in (
            "staged_entry_context",
            "discovery",
            "process_understanding",
            "control_path_qualification",
        ):
            if phase not in completed:
                completed.append(phase)

    manipulation_done = attacker_manipulation_started(context) or "manipulation" in emitted
    if manipulation_done:
        for phase in (
            "staged_entry_context",
            "discovery",
            "process_understanding",
            "control_path_qualification",
            "manipulation",
        ):
            if phase not in completed:
                completed.append(phase)

    if "impact" in emitted:
        for phase in ATTACKER_PHASES:
            if phase not in completed and ATTACKER_PHASE_ORDER.get(phase, 0) <= ATTACKER_PHASE_ORDER["impact"]:
                completed.append(phase)

    highest_phase = completed[-1] if completed else None
    next_phase = next(
        (phase for phase in ATTACKER_PHASES if phase not in completed),
        "impact" if manipulation_done else ATTACKER_PHASES[-1],
    )
    return {
        "completed_phases": completed,
        "highest_phase": highest_phase,
        "next_phase": next_phase,
        "manipulation_permitted": qualification_done,
        "discovered_surface": discovered_surface,
        "discovery_surfaces": discovery_surfaces_for_surface(discovered_surface),
        "qualification_surfaces": qualification_surfaces_for_surface(discovered_surface),
        "inspected_surfaces": inspect_surfaces,
    }


def defender_phase_state(
    context: dict[str, Any],
    *,
    incident: dict[str, Any] | None = None,
    structured_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del structured_memory
    incident = incident or {}
    phase_events = context_phase_events(
        context,
        phase_family="defender",
    )
    emitted = {
        str((item.get("metadata") or {}).get("mitre_phase"))
        for item in phase_events
        if (item.get("metadata") or {}).get("mitre_phase")
    }
    attacker_actions = [
        normalize_action(item.get("action"))
        for item in context.get("executed_actions", []) or []
        if item.get("role") == "attacker"
    ]
    attacker_actions = [item for item in attacker_actions if item]
    inspect_count = sum(1 for item in attacker_actions if item.get("type") == "inspect_surface")
    defender_actions = [
        item
        for item in context.get("executed_actions", []) or []
        if item.get("role") == "defender" and normalize_action(item.get("action"))
    ]
    incident_status = str(incident.get("status", "none"))
    suspicious_presence = bool(attacker_actions) or incident_status in {"suspected", "confirmed", "containing"}
    discovery_seen = inspect_count > 0
    manipulation_seen = attacker_manipulation_started(context) or any(
        str(item.get("type", "")) not in {"", "inspect_surface"}
        for item in attacker_actions
    )
    containment_seen = bool(defender_actions)

    completed: list[str] = []
    if suspicious_presence or "access_presence_suspicion" in emitted:
        completed.append("access_presence_suspicion")
    if discovery_seen or "discovery_recognition" in emitted:
        if "access_presence_suspicion" not in completed:
            completed.append("access_presence_suspicion")
        completed.append("discovery_recognition")
    if manipulation_seen or "manipulation_detection" in emitted:
        for phase in (
            "access_presence_suspicion",
            "discovery_recognition",
            "manipulation_detection",
        ):
            if phase not in completed:
                completed.append(phase)
    if containment_seen or "containment_mitigation" in emitted:
        for phase in DEFENDER_PHASES:
            if phase not in completed:
                completed.append(phase)

    highest_phase = completed[-1] if completed else None
    next_phase = next(
        (phase for phase in DEFENDER_PHASES if phase not in completed),
        DEFENDER_PHASES[-1],
    )
    attacker_state = attacker_phase_state(context, structured_memory=None)
    return {
        "completed_phases": completed,
        "highest_phase": highest_phase,
        "next_phase": next_phase,
        "attacker_highest_phase": attacker_state.get("highest_phase"),
        "attacker_next_phase": attacker_state.get("next_phase"),
        "inspect_count": inspect_count,
        "manipulation_seen": manipulation_seen,
    }


def optionb_mitre_context(
    *,
    role: str,
    context: dict[str, Any],
    structured_memory: dict[str, Any] | None = None,
    incident: dict[str, Any] | None = None,
    risk_flags: list[str] | None = None,
) -> dict[str, Any]:
    role = str(role).strip().lower()
    if role == "attacker":
        phase_state = attacker_phase_state(
            context,
            structured_memory=structured_memory,
            risk_flags=risk_flags,
        )
        phase_family = "attacker"
        phase_descriptions = ATTACKER_PHASE_DESCRIPTIONS
    else:
        phase_state = defender_phase_state(
            context,
            incident=incident,
            structured_memory=structured_memory,
        )
        phase_family = "defender"
        phase_descriptions = DEFENDER_PHASE_DESCRIPTIONS

    phase_order = ATTACKER_PHASES if phase_family == "attacker" else DEFENDER_PHASES
    completed = list(phase_state.get("completed_phases") or [])
    return {
        "phase_model": OPTIONB_MITRE_MODEL_ID,
        "phase_family": phase_family,
        "staged_path": list(OPTIONB_STAGE_PATH),
        "direct_plant_shortcuts_forbidden": True,
        "entry_assumption": (
            "Model staged entry through access and engineering context only; do not assume magical direct plant control."
        ),
        "completed_phases": completed,
        "current_phase": phase_state.get("highest_phase") or phase_order[0],
        "next_phase": phase_state.get("next_phase"),
        "phase_order": phase_order,
        "phase_guidance": [
            {"phase": phase, "description": phase_descriptions[phase]}
            for phase in phase_order
        ],
        "manipulation_permitted": bool(phase_state.get("manipulation_permitted")),
        "discovered_surface": phase_state.get("discovered_surface"),
        "discovery_surfaces": list(phase_state.get("discovery_surfaces") or []),
        "qualification_surfaces": list(phase_state.get("qualification_surfaces") or []),
        "attacker_highest_phase": phase_state.get("attacker_highest_phase"),
        "attacker_next_phase": phase_state.get("attacker_next_phase"),
        "manipulation_seen": phase_state.get("manipulation_seen"),
    }


def verify_optionb_staged_snapshot(
    environment_snapshot: dict[str, Any],
    *,
    expected_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_environment = expected_environment or {}
    errors: list[str] = []
    backend = str(environment_snapshot.get("backend", "")).strip()
    backend_config = dict(environment_snapshot.get("backend_config") or {})
    metric_boundary = dict(backend_config.get("metric_boundary") or {})
    staged_hops = [str(item) for item in backend_config.get("staged_hops", [])]

    if backend != "sphere_openplc":
        errors.append(f"expected backend sphere_openplc, found `{backend or 'missing'}`")
    if "access" not in staged_hops or "engineering" not in staged_hops:
        errors.append(
            f"expected staged hops to include access and engineering, found `{staged_hops}`"
        )
    if metric_boundary.get("metric_clock_origin") != "wargame_t0":
        errors.append(
            "expected metric boundary to declare `metric_clock_origin=wargame_t0`"
        )

    for field in ("experiment", "revision"):
        expected = str(expected_environment.get(field, "")).strip()
        actual = str(environment_snapshot.get(field, "")).strip()
        if expected and actual and actual != expected:
            errors.append(f"expected {field} `{expected}`, found `{actual}`")

    expected_realization = str(expected_environment.get("realization", "")).strip()
    actual_realization = str(environment_snapshot.get("realization", "")).strip()
    if expected_realization and actual_realization:
        if expected_realization not in {
            actual_realization,
            actual_realization.split(".", 1)[0],
        }:
            errors.append(
                f"expected realization `{expected_realization}`, found `{actual_realization}`"
            )

    expected_xdc = str(expected_environment.get("xdc", "")).strip()
    actual_xdc = str(environment_snapshot.get("xdc", "")).strip()
    override_xdc = str(os.environ.get("ICS_AGENT_SPHERE_XDC_OVERRIDE", "")).strip()
    if expected_xdc and actual_xdc:
        expected_tokens = {expected_xdc, expected_xdc.split(".", 1)[0]}
        if override_xdc:
            expected_tokens.update({override_xdc, override_xdc.split(".", 1)[0]})
        actual_tokens = {actual_xdc, actual_xdc.split(".", 1)[0]}
        if expected_tokens.isdisjoint(actual_tokens):
            errors.append(f"expected xdc `{expected_xdc}`, found `{actual_xdc}`")

    return {
        "ok": not errors,
        "errors": errors,
        "backend": backend,
        "staged_hops": staged_hops,
        "metric_clock_origin": metric_boundary.get("metric_clock_origin"),
        "realization": actual_realization,
        "revision": str(environment_snapshot.get("revision", "")).strip(),
        "xdc": actual_xdc,
    }
