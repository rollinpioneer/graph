# Stage 0A — Environment, Interface and Time Binding (Plan v1.1 / Method 2.1.1)

Status: `PASS`.

This report is the new-profile Stage 0A ledger. Historical v2.1 experiments/stage_status/stage_0a.json (BLOCKED) was snapshotted and not overwritten.

## Calibration

| family | attempts | legal successes | T_ref median (s) | d_ref median (s) | success cases |
|---|---:|---:|---:|---:|---|
| D0 | 5 | 5 | 15.050000000001983 | 4.5500000000015195 | D0_dev_00,D0_dev_01,D0_dev_02,D0_dev_03,D0_dev_04 |
| T_B | 5 | 5 | 23.09999999999752 | 4.199999999997672 | T_B_dev_00,T_B_dev_01,T_B_dev_02,T_B_dev_03,T_B_dev_04 |
| T_C | 5 | 5 | 23.09999999999752 | 4.399999999997561 | T_C_dev_00,T_C_dev_01,T_C_dev_02,T_C_dev_03,T_C_dev_04 |

- Suite H = max T_ref = **23.09999999999752 seconds**.
- Discount: Gamma = 2^(-d_seconds/H) with H from this measurement.
- Tcap(D0 smoke) = 16384 * d_ref(D0) = **74547.2000000249 s**.
- Tcap(T_B core) = 65536 * d_ref(T_B) = **275251.19999984745 s**.
- Tcap(T_C core) = 65536 * d_ref(T_C) = **288358.3999998402 s**.

## Evidence files

- runs/stage_0a/reference_attempts.jsonl sha256=3fb2f984f17fff6d78e2c29ba57f1e77d2273e234ac0e2d8ef4218d0a82c50fd
- runs/stage_0a/reference_execution_manifest.json sha256=fce3d21fa839ae8a83222cb73f4e33b556f2ec45e4cfb1b775259183374f36d2
- runs/stage_0a/reference_run.log sha256=f77109b23c0eb22305a6d71ee06fff2cf48c7cbbdfdd617f3c0688a6d1cc9434

## Binding

Reused verified LIBERO/robosuite/MuJoCo stack, DashScope Beijing endpoint binding, and D0/T_C splits. New T_B split: configs/splits/T_B_stage_0a.json (train seeds 4100+, dev seeds 8100+).

New-profile RL training runs this stage: **0**.

