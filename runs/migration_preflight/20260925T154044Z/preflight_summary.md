# CP-DISR v1.2 XFS Migration Preflight

- Status: **BLOCKED**
- phase_a_e16_ready: **false**
- Account/host: `xushijie2@gpu03`
- Repository: `/home/xushijie2/graph_cp_disr_v2_1`
- Branch: `codex/cp-disr-v1.2-xfs-preflight`
- Trusted source commit: `163dbe6bdacdcc2b294bc514348d798ad33fd6c0`
- Active filesystem: `/home` XFS (`/dev/mapper/rl-home`)

Hard-gate failures: target environment is missing `torch_geometric` (2.6.1), so default `cp_disr` import, R-GCN forward/backward, Full/B2 probes, and complete production test collection cannot run; the same T_C case produced different RGB hashes across two resets; production checkpoint publication does not implement temp+flush/fsync+atomic latest-last publication or failure injection recovery.

Passed evidence: pip check; required package versions except missing PyG; CUDA availability; T_C cache identity (train 64/8, dev 20/0, test 30/3); bounded XFS fsync write; MuJoCo RGB/depth dimensions and finite depth; one real 0.05s control tick; clean close; P0 runner no-learning resume probe. No formal RL, PPO, VLM, final test, or old Stage 2A resume occurred.

The Phase A pair is fully specified in `phase_a_pair_manifest.json` and remains `NOT_STARTED`.
