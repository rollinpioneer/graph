# Claim/evidence table

| Claim | Evidence | Interpretation |
|---|---|---|
| Full durable boundary | `runs/v13_r3_recovery/full_checkpoint_verification.json` | N=2048, T=9771.400000000713, 2 complete updates; hashes verified |
| Full stall phase | `stall_phase_reconciliation.md`, `stall_timeline.json`, `last_log_records.json` | PPO internal stall at attempted N=3072; not a completed update |
| B2 actual endpoint | `runs/v13_r3_recovery/D0/B2/.../job_summary.json` | Tcap-first N=1935, 1 complete + 1 fragment; no N=2048 claim |
| B2 first update | `b2_first_update_selfcheck.json`, `train_metrics.csv` | 1024 transitions, four epoch rows, finite losses, parameter change |
| Full prior switch | `full_absent_dev10.json`, `full_s_train_original.csv`, `full_s_train_absent.csv` | Same Full checkpoint, prior-only intervention; Full@Absent is not B2 |
