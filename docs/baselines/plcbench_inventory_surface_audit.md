# WT high-inventory operator-surface audit

Result: **grounded selective lever not found** in the current full-system OpenPLC
realization. The proposed high-inventory cell stays `unqualified`.

The controller's UF feed-tank loop stops the raw-water pump when
`in_UF_UFFT_Tank_Level >= UFFT_PUMP_STOP` and restarts it only below
`UFFT_PUMP_START` (`wt_controller.st`, raw-water pump control). The downstream
UF transfer and valve outputs are controller-computed each scan. The mapped HMI
inputs include plant stop/start and P2 manual dosing, but the inspected command
surface has no operator input that selectively blocks UF outflow while leaving
inflow under normal control. Writing an output register externally is not a
grounded operator lever because controller logic recomputes it.

This is a source-level finding, not a proof that high inventory is impossible.
Qualification needs a named writable input, a simulator trace showing its
effect through the controller, an admission/reset contract, and warning/impact
thresholds from the realized tank dynamics. None is inferred from the pilot
dosing result.
