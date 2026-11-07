from ics_agent.plcbench_sitl import (
    IMPACT_HOLD_SCANS,
    S1LevelPlant,
    S2TemperaturePlant,
    S3ThermalMixingPlant,
    S4QuadTankPlant,
)


def test_s1_uses_the_specified_100_ms_mass_balance() -> None:
    plant = S1LevelPlant()
    result = plant.step(manual=True, commands={"inlet": True, "drain": False})
    assert result.state["L"] == 50.45


def test_s2_overheat_latch_forces_cooling() -> None:
    plant = S2TemperaturePlant(temperature=136.0)
    result = plant.step(manual=True, commands={"heater_enable": True, "heater_power": 1.0})
    assert result.safety_active
    assert result.outputs["heater_enable"] is False
    assert result.outputs["fan"] is True


def test_s3_low_level_guard_turns_drain_and_heater_off() -> None:
    plant = S3ThermalMixingPlant(level=4.0, temperature=50.0)
    result = plant.step(manual=True, commands={"drain": True, "heater": True})
    assert result.outputs["drain"] is False
    assert result.outputs["heater"] is False


def test_s4_upper_tank_cutoff_disables_the_related_pump() -> None:
    plant = S4QuadTankPlant(h1=40.0, h4=66.0)
    result = plant.step()
    assert result.outputs["pump1_enable"] is False


def test_threshold_controllers_retain_state_inside_their_deadbands() -> None:
    s1 = S1LevelPlant(level=29.0)
    assert s1.step().outputs["inlet"] is True
    s1.level = 50.0
    assert s1.step().outputs["inlet"] is True

    s4 = S4QuadTankPlant(h1=41.0)
    assert s4.step().outputs["pump1_enable"] is True
    s4.h1 = 45.0
    assert s4.step().outputs["pump1_enable"] is True


def test_impact_requires_30_seconds_of_continuous_predicate() -> None:
    plant = S1LevelPlant(level=86.0)
    for _ in range(IMPACT_HOLD_SCANS - 1):
        plant._impact(True)
    assert plant._impact(True)
