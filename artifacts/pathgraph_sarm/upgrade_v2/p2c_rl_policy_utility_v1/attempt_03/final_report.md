# P2C-RL campaign Attempt-03 final report

Status: `CAMPAIGN_COMPLETE_NO_CONFIRMATORY_CLAIM`

Scientific decision: `POLICY_UTILITY_NOT_ESTABLISHED_OR_MIXED`

## Execution

- Release: `P2CRL_V1_ATTEMPT_03`
- Release SHA256: `8dca27d063c0985c5b3a14695dc7b2f27c0c24fe43cbc87f473621253356f4ef`
- Frozen runner commit: `8e6aa25507b97bef0988049789fe0b6dfd02e5d8`
- Runtime: `/home/xushijie/.conda/envs/pathgraph-p2crl-frozen-v1`
- Smoke: 5/5 complete, 2,560 environment steps, 100 optimizer updates.
- Formal: 60/60 complete, 31,457,280 environment steps, 1,228,800 optimizer updates.
- Real backend was used. The job ledger contains no failed or incomplete jobs.

## Final outputs

- Main test panel: 512 contracts, 60 jobs, 30,720 expected rows.
- Stochastic panel: 15,360 expected rows using four registered sampling seeds.
- Critical-state panel: policy inputs were observation and action mask; the oracle process was analysis-only.
- `RW_V2 - GEOM`: point 0.0, 95% CI [0.0, 0.0], margin gate false, CI gate false.
- `RW_V2 - FLAT`: point 0.0, 95% CI [0.0, 0.0], margin gate false, CI gate false.
- Bootstrap count: 20,000 with seed 2026091806.
- Global confirmation passed: false.

Attempt-03 completed the registered workload, but it did not establish the preregistered policy-utility claim. No confirmation claim is made.

## Recovery disclosure

The campaign was not uninterrupted. The original SSH-attached run ended, after which the ledger was recovered without rerunning jobs already marked complete. Partial `C_s331_RW2` evidence was quarantined under `interrupted_recovery`. A later `C_s359_GEOM` validation subprocess exited 120 because a disowned process retained a dead PTY; its partial training and validation evidence was also quarantined. The final five jobs were then run under true `nohup` with redirected output and `/dev/null` stdin. The final log terminates with:

`EXECUTE_OK P2CRL_CAMPAIGN_RESULT_V1 learn_called None`

The campaign database still records `status=RESUMING`. This is a runner bookkeeping residue: the runner does not update the campaign row after successful completion. This result package does not rewrite that state. Completion is supported by all 65 job rows being `COMPLETE`, the exact step totals, the four final outputs, and the terminal log marker.

## Artifact policy

Raw training, validation, recovery evidence, SQLite state, logs, and checkpoint binaries remain outside Git at:

`/home/__compress_data/xushijie/graph_pathgraph_p2c_rl_data/run_v1/campaign_v1_attempt_03`

Git contains only lightweight final outputs and audit indexes. `checkpoint_manifest.tsv` records the already-produced hashes from checkpoint metadata without copying checkpoint binaries. `external_artifacts.tsv` records hashes for the database, terminal log, and final outputs.

This result is isolated on the Attempt-03 results branch. `main` was not modified or merged.
