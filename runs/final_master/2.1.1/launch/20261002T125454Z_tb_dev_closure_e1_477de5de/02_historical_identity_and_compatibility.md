# Historical T_B seed-0 runs: identity, compatibility evidence and unconfirmed items

Read-only inspection of the three runs found under `graph_cp_disr_v2_1/runs` (v13_r1 and v13_r2). Nothing was retrained, resumed, converted or copied; no checkpoint was loaded; no episode was run. They keep their original execution identity and are listed apart from the new runs (no pooled mean).

## Identity as recorded

| slot | run | planned_id | N / T (s) | updates | stop | recorded job_dir differs from local copy | source commits recorded in manifest |
|---|---|---|---|---|---|---|---|
| TB-M2 | R1-B1K-s0 | v13_R1_T_B_B1K_s0_E16 | 13623 / 68813.40 | 13 + 1 | Tcap | False | {"git_commit": "0eb3776d143d2142eb2893e2a12a4d85f2d14fc1", "base_commit": "66614acb443dcacee2b16c5472d4ecbd697bf561"} |
| TB-M4 | R1-B2-s0 | v13_R1_T_B_B2_s0_E16 | 14464 / 68818.75 | 14 + 1 | Tcap | False | {"git_commit": "0eb3776d143d2142eb2893e2a12a4d85f2d14fc1", "base_commit": "66614acb443dcacee2b16c5472d4ecbd697bf561"} |
| TB-M1 | R2-B0-s0 | v13_R2_T_B_B0_s0_E16 | 14705 / 68814.80 | 14 + 1 | Tcap | True | {"git_commit": "228517ed214e18bd75ab11b84f3edff6b07f14b7", "baseline_commit": "55a3b7ce35edebbbb8a587fe1e3a98ca1967db07"} |

Plan v3 section 1 values reproduced from the run files (N and complete + fragment updates): R1-B1K-s0 True, R1-B2-s0 True, R2-B0-s0 True.

## File-level comparison with the current execution path (E-1 manifest)

| run | equal | differ | H / d_ref / Tcap equal | dev10 split hash equal | dev10 case ids equal |
|---|---|---|---|---|---|
| R1-B1K-s0 | collector, persistence | neural, runtime_factory, stage2a_v11, torch_rl | True | True | True |
| R1-B2-s0 | collector, persistence | neural, runtime_factory, stage2a_v11, torch_rl | True | True | True |
| R2-B0-s0 | collector, persistence | neural, runtime_factory, stage2a_v11, torch_rl | True | True | True |

## Internal consistency of each historical final checkpoint (bytes re-hashed, not loaded)

| run | final file | sha256 | recorded model.pt sha equals recomputed | latest = final | N and T equal job_summary | eval_final uses it | fresh-load record |
|---|---|---|---|---|---|---|---|
| R1-B1K-s0 | final_n_013623_u13.pt | 2c93d96f3999b5192f5ceb7c817e9a00a94937dfa1a22db9e76ac02cf7d0c99a | True | True | True | True | True |
| R1-B2-s0 | final_n_014464_u14.pt | 976f70102ee993a639edd4572eebe2053ad8ac6d7b450bd43dc3576a367b413e | True | True | True | True | True |
| R2-B0-s0 | final_n_014705_u14.pt | f50a857ce9b589e65b9b7b4d9e983b7e3babb1b9e93d54875f3e383e99b837f0 | True | True | True | True | True |

## Existing compatibility evidence cited

- CE1. docs/authoritative/CP_DISR_Final_Experimental_Plan_v3.md section 1 (rows T_B R1 B1-K/B2, T_B R2 B0) - keep original N/T and model identity; B0 keeps the actual old-account deviation; do not write it as an xushijie2 run, do not discard or auto-retrain
- CE2. Plan v3 section 1.2 and section 12.2 - no automatic resumption of historical obligations; when historical weights/inputs cannot be shown usable, no sidecar and no automatic retraining; the gap stays recorded and goes to the elastic / manual decision
- CE3. docs/authoritative/CP_DISR_Final_Research_Content_v5.md lines 824-825 ([E1], [E2]) - R1 ref 55a3b7ce..., runs/v13_r1/20260926T112613Z/; R2 ref eaa43e97..., runs/v13_r2/20260927T070500Z/, actual old-account run
- CE4. R-TB-E-1 launch dir: smoke_compatibility_rebind.json and e1_run_record.json (same_profile_check_vs_E0) - the new runs R-TB-E-0/K-1/DK-1 (prep cac2fa5b) and R-TB-E-1 (prep e4dc34ae) share one execution path; E-1 differs from E-0 only in the train-split path line of the resolved manifest (normalized hash identical)
- CE5. this record (file-level, no model loaded) - historical and current runs share H, d_ref, Tcap, Ncap, the dev10 split hash and the dev10 case ids, and the collector and persistence source hashes; they differ in neural, torch_rl, runtime_factory and stage2a_v11 source hashes

## Unconfirmed items (listed apart; no pooled mean; no automatic retraining)

| id | item | why | status | handling |
|---|---|---|---|---|
| UC1 | Whether the three historical final weights load and run under the current (2.1.1) neural/torch_rl/runtime code | source hashes of neural, torch_rl, runtime_factory and stage2a_v11 differ from the current execution path; each historical run only has its own fresh-load record from the code of its time | NOT_CHECKED | listed apart; not averaged with new runs; no retraining; a load-only check (no episode) is requested in the sub-card, not run here |
| UC2 | Whether the historical evaluation function (stage2a_v11 543cb10c...) and the current one (451fdbd9...) score identically | different source hash; no differential audit exists and a full-repository audit is out of scope | UNVERIFIED | historical dev numbers stay as recorded under their own identity; test would use the single unified evaluator, which is the point of the unified evaluation card |
| UC3 | R2 B0 s0 original location and byte-level provenance of the local copy | job_summary records job_dir under /home/__compress_data/xushijie/graph_cp_disr_v2_1_r2_b0/ (old account); the files read here are a copy under graph_cp_disr_v2_1/runs/v13_r2; the original was not readable from this account when this record was built (probe result in the JSON: permission denied) | ORIGINAL_NOT_ACCESSIBLE | internal consistency only (generation HASHES.json vs recomputed final hash, fresh-load record); recorded as an actual old-account run, not rewritten as an xushijie2 run |
| UC4 | Commit pointers differ between run manifests and the ledger text | R1 manifests record git_commit 0eb3776d... (base 66614acb...), R2 records git_commit 228517ed... (baseline 55a3b7ce...); Research Content v5 cites 55a3b7ce... for R1 and eaa43e97... for R2 | NOT_RECONCILED | each run keeps the identity recorded in its own manifest; no re-derivation |
| UC5 | Weights-to-input chain for test release (Adam/model index, input hashes) of the historical runs | generation folders hold model.pt/model.json/episode.pkl/rng.json with HASHES.json, but the test_release_manifest (Plan 12.1 item 3) has not been built | NOT_BUILT | built at release time from the frozen list; any item that cannot be proven stays a recorded gap |
| UC6 | R2 B0 first start folder 20260927T065500Z_ac37e5e4 | contains only n_000000 and no job_summary: an aborted start, not the run that produced N14705 | EXCLUDED | not part of any list; named only so it is not mistaken for a model |
| UC7 | R2 B0 s0 `common_dev20.json` holds 10 cases, not 20 | the file is labelled common_dev20 but records n = 10 (the same ten frozen dev10 case ids as eval_final.json), whereas the R1 B1-K / B2 files record n = 20 and Plan v3 section 1 reports a selected dev20 of 0/20 for B0 | RECORDED_DIFFERENCE_NOT_RECONCILED | the separate selected column shows the file as recorded (0/10 for B0); it is not used for any main-evaluation decision and no number is rewritten |

Correction to an earlier note: an earlier message said these historical artifacts could not be found; that came from a depth-limited file search. They are present, and this record is based on the files.

