# CP-DISR v1.3 R3 D0 prior comparison — execution summary

Status: **FAILED_STALLED**.

- Execution identity: `xushijie2@gpu03`, Python `/home/xushijie2/envs/lerobotpi0-xfs/bin/python`, isolated worktree `/home/xushijie2/graph_cp_disr_v13_r3`.
- Authorized jobs: `v13_R3_D0_Full_s0_E8` and `v13_R3_D0_B2_s0_E8`; only Full started. B2 was not started.
- Full observed transition log: N=3072, T=14758.050000000949. Last persisted boundary: N=2048, T=9771.400000000713, 2 complete PPO updates.
- Full stalled during the PPO update attempt at N=3072 for over 30 minutes without progress; process was stopped with evidence preserved and no restart.
- Dev10 completed: N=0 = 0/10; N=2048 = 10/10, mean discounted return 0.6350014581. N=4096/N=8192 and all B2 points were not reached.
- Natural non-empty prior cases: S_train=['D0_train_13', 'D0_train_31', 'D0_train_32', 'D0_train_62']; S_dev=['D0_dev_00'].
- VLM/test-ID/extra seed/A_CAT/A_Q/A_B/T_C: 0. R2/B0 untouched.
- No Full-vs-B2 or Full@Absent conclusion is made because the paired run was not completed.

See `Results_v1.md`, `checkpoint_index.json`, `duration_audit.md`, `cost_ledger.csv`, and `unresolved_items.md`. Checkpoint binaries remain on the XFS host and are intentionally not committed.
