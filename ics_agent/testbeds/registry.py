from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import OPENPLC_SUPPORT_DIR, REPO_ROOT, RUNTIME_SCENARIO_ROOT


@dataclass(frozen=True)
class TestbedSpec:
    testbed_id: str
    display_name: str
    scenario_root: Path
    support_root: Path
    prompt_root: Path | None = None
    knowledge_root: Path | None = None
    canonical_sphere_realization: str | None = None
    canonical_sphere_deployment_name: str | None = None
    primary_process_id: str | None = None


def normalize_testbed_id(value: str | None) -> str:
    normalized = str(value or "water_treatment").strip().lower().replace("-", "_")
    aliases = {
        "wt": "water_treatment",
        "water_treatment": "water_treatment",
        "water_treatment_openplc": "water_treatment",
        "wd": "water_distribution",
        "water_distribution": "water_distribution",
        "water_distribution_openplc": "water_distribution",
    }
    return aliases.get(normalized, normalized or "water_treatment")


TESTBEDS: dict[str, TestbedSpec] = {
    "water_treatment": TestbedSpec(
        testbed_id="water_treatment",
        display_name="Water Treatment",
        scenario_root=RUNTIME_SCENARIO_ROOT,
        support_root=OPENPLC_SUPPORT_DIR,
        primary_process_id="WT-P1",
    ),
    "water_distribution": TestbedSpec(
        testbed_id="water_distribution",
        display_name="Water Distribution",
        scenario_root=REPO_ROOT / "testbeds" / "water_distribution" / "scenarios",
        support_root=REPO_ROOT / "testbeds" / "water_distribution" / "openplc",
        prompt_root=REPO_ROOT / "testbeds" / "water_distribution" / "prompts",
        knowledge_root=REPO_ROOT / "testbeds" / "water_distribution" / "knowledge",
        canonical_sphere_realization="example-realization-01",
        canonical_sphere_deployment_name="example-realization-01",
        primary_process_id="WD-UC0",
    ),
}


def get_testbed_spec(testbed_id: str | None) -> TestbedSpec:
    normalized = normalize_testbed_id(testbed_id)
    try:
        return TESTBEDS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown testbed `{testbed_id}`") from exc


def observation_focus_fields(testbed_id: str | None) -> list[str]:
    normalized = normalize_testbed_id(testbed_id)
    if normalized == "water_distribution":
        return [
            "STATE_NAME",
            "Supply_Tank_Level",
            "Supply_Pump_Flow",
            "Supply_Pump_Speed_Cmd",
            "Supply_Pump_Sts",
            "Supply_Tank_Valve_Cmd",
            "Supply_Tank_Valve_Sts",
            "Grid_Elev_Tank_Level",
            "Grid_Consum_Tank_Level",
            "Grid_Elev_Valve_Cmd",
            "Grid_Elev_Valve_Sts",
            "RWS_Tank_Level",
            "RWS_Pump_Speed_Cmd",
            "RWS_Pump_Sts",
            "RWS_Tank_Valve_Cmd",
            "RWS_Tank_Valve_Sts",
            "Alarm_Supply_Tank_HH",
            "HMI_Start_PB",
            "HMI_Stop_PB",
        ]
    return [
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


def trend_fields(testbed_id: str | None) -> list[str]:
    normalized = normalize_testbed_id(testbed_id)
    if normalized == "water_distribution":
        return [
            "Supply_Tank_Level",
            "Grid_Elev_Tank_Level",
            "Grid_Consum_Tank_Level",
            "RWS_Tank_Level",
            "Supply_Pump_Flow",
            "Supply_Pump_Speed_Cmd",
            "RWS_Pump_Speed_Cmd",
        ]
    return [
        "RW_Tank_Level",
        "UF_UFFT_Tank_Level",
        "ChemTreat_NaCl_Level",
        "ChemTreat_NaOCl_Level",
        "ChemTreat_HCl_Level",
        "RW_Pump_Flow",
        "RW_Pump_Speed_Cmd",
    ]
