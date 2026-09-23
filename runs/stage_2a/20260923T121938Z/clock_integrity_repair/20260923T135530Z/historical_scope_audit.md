# Historical scope audit (readonly)

Defect is in the shared baseline used by 0A/0D/1A and Stage 2A. This is **not** a declaration that every past number is invalid, and **not** a declaration that everything is safe.

## Stage 2A stamp 20260923T121938Z

- T_B B0 and T_B B1-K: **affected**. Both complete PPO updates contain padded zero-physics transitions. Weights/Adam cannot be reused. Incomplete third rollout never entered PPO.
- Six unstarted jobs: no weights. Must use `clock_integrity_r1` when later authorized.

## Stage 1A (20260922T164616Z_1a2768d8) readonly

B2: 8192 transitions, 5 with duration ? 0.05, **0 INTERRUPT**.
Full: 8192 transitions, 4 with duration ? 0.05, **0 INTERRUPT**.

All nine 0.05 rows are `controller_exit=TASK_DEADLINE` with duration `0.05000000000006111` (IEEE residue), not exact `0.05` from `1/20`. They match a real last 20 Hz tick at deadline, not the zero-physics pad.

1A success/fail records and smoke PASS remain historical. Training metrics that used real positive durations are not shown to be pad-contaminated. Residual risk: other zero-step rejects could have been padded; none appear in these logs as INTERRUPT. **No auto retrain/re-eval.**

## Stage 0A reference / H / d_ref

H and d_ref come from verified-success skill duration statistics (typically seconds, not 0.05 interrupt pads). There is **insufficient evidence** that calibration used padded zero-physics rows. **Do not silently retune H.** Recalibrate only if later audit of the calibration traces shows padded durations entered the statistic.

## Stage 0D

Empty-patch / prior identity work is orthogonal. Clock fallback existed, but 0D is not an RL T/Gamma consumer in the same way. Cannot prove every 0D duration field; no auto rerun.

## VLM cache

Task/image/ID/contract/payload identity unchanged. Clock repair is not a reason to regenerate VLM caches.
