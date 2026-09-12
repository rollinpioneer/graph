# L2RAR2 R16 V2 Physical Reference Repair Campaign

Status: `STOPPED_CALIBRATION_FAILED`.

The parent A2/B/C certificate was reused without rerunning those stages. The repaired V2 geometry and contact reference executed real MuJoCo traces for primary control, backup control, and I1/I2/I3. All five instances reached model construction and physics, so they consume five new physical instances. Both controls failed the frozen pre-hold requirement (negative measured height rise); no intervention level was selected. Development, generator-gate, and evaluation were correctly not dispatched after the calibration hard stop.

The legacy V1.1 calibration remains invalid for scientific level selection. No confirmation or L3 entry occurred and `selected_candidate_id` remains null.
