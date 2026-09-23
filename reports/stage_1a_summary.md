# Stage 1A — One-task Smoke (Plan v1.1 / Method 2.1.1)

Status: `PASS`.

This report is the new-profile ledger. It does not overwrite `experiments/part_1_smoke/stage_1a/`.

## Startup repairs

- 0D empty-patch identity: /home/__compress_data/xushijie/graph_cp_disr_v2_1/runs/stage_0d/empty_patch_erratum.json
- deadline production tests: tests/test_deadline_semantics.py
- runner: `python -m cp_disr.cli stage-1a-v11-run` using `runtime_manifest_v211.yaml`

## Frozen profile

- H = 23.09999999999752
- D0 d_ref = 4.5500000000015195
- Tcap = Ncap * d_ref = 74547.2000000249
- actor_episode_discount_weight = false
- Gamma = 2**(-duration_seconds/H)

## Jobs

### B2

- planned_id: v11_1A_D0_B2_s0
- run_id: v11_1A_D0_B2_s0_20260922T164616Z_1a2768d8
- N/T: 8192 / 39602.150000001166
- complete/fragment updates: 8 / 0
- train_success_episodes: 1112
- step0_dev_success: 0.0
- selected_checkpoint: /home/__compress_data/xushijie/graph_cp_disr_v2_1/runs/stage_1a/D0/B2/seed_0/20260922T164616Z_1a2768d8/checkpoints/n_008192.pt
- test-ID: NOT_CREATED

### Full

- planned_id: v11_1A_D0_Full_s0
- run_id: v11_1A_D0_Full_s0_20260922T164616Z_1a2768d8
- N/T: 8192 / 40727.25000000116
- complete/fragment updates: 8 / 0
- train_success_episodes: 1470
- step0_dev_success: 0.0
- selected_checkpoint: /home/__compress_data/xushijie/graph_cp_disr_v2_1/runs/stage_1a/D0/Full/seed_0/20260922T164616Z_1a2768d8/checkpoints/n_008192.pt
- test-ID: NOT_CREATED

## Smoke gate

- passed: True
- reasons: []
- resource_cap: None

## Cost

{
  "new_vlm_calls": 0,
  "new_rl_jobs": 2,
  "startup_diagnostics": true,
  "api_cost": 0,
  "note": "reused existing D0 cache; no new VLM calls"
}

Next stage suggestion only: 1B if smoke incomplete; otherwise wait for explicit Stage 1B/2A authorization.

## Evidence appendix

- Live stamp: `20260922T164616Z_1a2768d8`
- Baseline / git_commit recorded in run hashes: `4323f4d6d14e1bca671e47695be9bffc5b0c59ac`
- Runtime: `experiments/manifests/runtime_manifest_v211.yaml`
- Split: `configs/splits/D0_stage_1a.json` (train 64, dev 10, test_isolated=true, test-IDs none)
- B2 DP/Delta: all 8 complete updates strictly 0; `dp_opportunity_n=0` so conditioned rates are N/A
- Full DP/Delta: opportunity_n summed across 8 updates = 374; every complete update had DP and Delta witnesses
- B2 prior: all Absent / empty effective prior by method
- Full prior: episode-level 80/20 Original/Absent; eval Original
- Selected checkpoints use last-3-window mean discounted return, then worst window, then earlier step
- test-ID: NOT_CREATED because no registered test list; dev was not substituted
- Old Stage 1A dir `experiments/part_1_smoke/stage_1a` was not overwritten
- Old Stage 2A STOP guard was not lifted
- Superseded collection attempts retained: serial `20260922T142041Z`, failed parallel `20260922T151725Z` / `20260922T153039Z`, slow parallel `20260922T153738Z`
- No 1B / 2A / 2B / 3A executed
