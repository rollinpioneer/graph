# Frozen main-evaluation model list for the later unified T_B evaluation

Frozen at 2026-10-02T13:08:18Z (UTC), repository head 477de5dec2d59e22eaa0f6a840bb0d1b80f0a687. This freezes the candidate list only; no test is released.

## Rules

- R1 Main-evaluation model of every run = its last valid post-update checkpoint (the final_n_*_u* generation), the registered primary endpoint.
- R2 The checkpoint selected by the job rule (max_mean3_then_max_worst_then_earlier_N) and any best-dev / common-dev20 selection are listed in separate columns; they are never a substitute for the final model. A method whose selected checkpoint is n_000000 is evaluated at its final model, never at N = 0.
- R3 Intermediate evaluation points (N = 4096, 8192) are dev learning-process evidence, not members of this list.
- R4 Existing hash verification is reused: intermediate checkpoint hashes are cited from the runs' own generation HASHES.json and the earlier checkpoint_identity records; no intermediate model is loaded or re-hashed. Only the final checkpoint files are re-hashed (bytes only, no model load) as the freeze-time stat check.
- R5 A model whose usability cannot be shown stays listed with its gap; no sidecar stands in for it and no automatic retraining follows (Plan v3 12.2).
- R6 The list is frozen as a candidate list for the later release; it does not release any test, and the test_release_manifest is not built here.

## Main-evaluation checkpoints (last valid post-update generation)

| slot | run | method | seed | list | final checkpoint | N | T (s) | update | sha256 (recomputed now) | recorded = recomputed | file/hash checks | load under 2.1.1 code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TB-M1 | R2-B0-s0 | B0 | 0 | BASE_6x30 | final_n_014705_u14.pt | 14705 | 68814.80 | 14 | f50a857ce9b589e65b9b7b4d9e983b7e3babb1b9e93d54875f3e383e99b837f0 | True | OK | NOT_CHECKED (see UC1) |
| TB-M2 | R1-B1K-s0 | B1-K | 0 | BASE_6x30 | final_n_013623_u13.pt | 13623 | 68813.40 | 13 | 2c93d96f3999b5192f5ceb7c817e9a00a94937dfa1a22db9e76ac02cf7d0c99a | True | OK | NOT_CHECKED (see UC1) |
| TB-M3 | R-TB-K-1 | B1-K | 1 | BASE_6x30 | final_n_014531_u14.pt | 14531 | 68813.45 | 14 | fa6ca4cf20689aba50c81b52ae2b202772e97cf0a886d74ae58cc22fcb91f385 | True | OK | TRAINED_AND_FRESH_LOADED_ON_THE_2.1.1_PATH |
| TB-M4 | R1-B2-s0 | B2 | 0 | BASE_6x30 | final_n_014464_u14.pt | 14464 | 68818.75 | 14 | 976f70102ee993a639edd4572eebe2053ad8ac6d7b450bd43dc3576a367b413e | True | OK | NOT_CHECKED (see UC1) |
| TB-M5 | R-TB-DK-1 | B2 | 1 | BASE_6x30 | final_n_014170_u13.pt | 14170 | 68817.10 | 13 | b1e331c98d4937b919242912cb3011bb079ec6971a3f8b7b51c682546b7476ca | True | OK | TRAINED_AND_FRESH_LOADED_ON_THE_2.1.1_PATH |
| TB-M6 | R-TB-E-0 | B1-K+E | 0 | BASE_6x30 | final_n_014512_u14.pt | 14512 | 68813.30 | 14 | f9db3e4ff6dc877dbd4b1b433ac31789e43f6fd45404c35e5103674e4ed4aaba | True | OK | TRAINED_AND_FRESH_LOADED_ON_THE_2.1.1_PATH |
| TB-C1 | R-TB-E-1 | B1-K+E | 1 | CONDITIONAL_OUTSIDE_628 | final_n_014309_u13.pt | 14309 | 68820.25 | 13 | 62099c437dabd361412ae2fe3b5b6fadcf785bc88661e32ac7d35564f79522e7 | True | OK | TRAINED_AND_FRESH_LOADED_ON_THE_2.1.1_PATH |

## Selected / best-dev checkpoints (separate columns; not substitutes for final)

| run | job-rule selected checkpoint | equals final | rule | common-dev20 selection (historical runs) |
|---|---|---|---|---|
| R2-B0-s0 | n_000000.pt | False | max_mean3_then_max_worst_then_earlier_N | n_000000.pt 0/10 (return 0.0000) |
| R1-B1K-s0 | n_000000.pt | False | max_mean3_then_max_worst_then_earlier_N | n_000000.pt 0/20 (return 0.0000) |
| R-TB-K-1 | n_000000.pt | False | max_mean3_then_max_worst_then_earlier_N | not recorded |
| R1-B2-s0 | final_n_014464_u14.pt | True | max_mean3_then_max_worst_then_earlier_N | final_n_014464_u14.pt 20/20 (return 0.5003) |
| R-TB-DK-1 | final_n_014170_u13.pt | True | max_mean3_then_max_worst_then_earlier_N | not recorded |
| R-TB-E-0 | final_n_014512_u14.pt | True | max_mean3_then_max_worst_then_earlier_N | not recorded |
| R-TB-E-1 | final_n_014309_u13.pt | True | max_mean3_then_max_worst_then_earlier_N | not recorded |

Runs whose selected checkpoint is n_000000 are evaluated at their final model, never at N = 0.

## Counts

Base: 6 models x 30 = 180 episodes if released (inside the Plan v3 cap of 628). Conditional: B1-K+E seed 1 adds 30 (outside the cap, Plan v3 12.2).

## Not in the list

- R2 B0 first start folder 20260927T065500Z_ac37e5e4: n_000000 only, no job_summary (UC6)
- all n_000000 / n_004096 / n_008192 checkpoints and all update_complete_* / update_fragment_1 generations: R3, not main-evaluation models

## Identity gaps carried with the historical entries

See `02_historical_identity_and_compatibility.md` (UC1-UC7). A gap is carried, not repaired.

