# Experimental WT membrane state

The full-system WT simulator already contains `membrane_damage`. For each
simulator scan, RO-feed chlorine above `WQ_RO_CL_MAX` increases damage by
`WQ_DAMAGE_RATE × excess`; damage is capped at 1. Salt rejection is
`WQ_RO_REJECTION × (1 − damage)`, and modeled permeate conductivity follows
from the remaining fraction, clipped to the AIT504 display range.

`ics_agent/plcbench_membrane.py` extracts the ST constants and checks this
equation's source fragments before mirroring a step. Offline tests cover
monotonic damage, persistence through a chemistry flush, full-restart reset,
and conductivity response. These tests qualify code behavior only. The model
has **no physical validation** and its threshold/rate are not vendor limits.
WT chemistry episodes require an external full-simulator restart proof with a
new generation ID and zero damage before their clean-state flush.
