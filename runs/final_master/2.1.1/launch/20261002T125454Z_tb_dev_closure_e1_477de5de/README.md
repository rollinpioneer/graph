# T_B development-evidence closure with R-TB-E-1 and unified-evaluation preparation

Directory: `20261002T125454Z_tb_dev_closure_e1_477de5de`. New summary material only; no original run record is overwritten (R-TB-E-1 results stay in commit 477de5dec2d59e22eaa0f6a840bb0d1b80f0a687).

| file | content |
|---|---|
| `01_comparison_with_E1.md/.json/_points.csv` | comparison table with R-TB-E-1: success, discounted return, actual N / T at 0 / 4096 / 8192 / final; permitted statements; what is not said |
| `02_historical_identity_and_compatibility.md/.json` | R1 B1-K / B2 seed 0 and R2 B0 seed 0: original identity, cited compatibility evidence, unconfirmed items listed apart |
| `03_frozen_model_list.md/.json` | frozen main-evaluation list (last valid post-update checkpoint), selected / best-dev in separate columns, hash reuse |
| `04_evaluation_record_revision_design.md` | design only: episode start to first independent confirmed success, action sequence, evaluated-model hash; old success_seconds kept as FINAL_SKILL_DURATION_ONLY |
| `05_tb_independent_eval_subcard_request.md/.json` | T_B independent-evaluation sub-card request: models, data source, episode budget, prerequisites, unresolved items, decisions requested |
| `06_test_source_provenance_check.json` | test source evidence from file names, hashes and config declarations only |
| `build_summary.json`, `MANIFEST_SHA256.json`, `scripts/build_tb_dev_closure.py` | build checks, file hashes, the zero-environment builder |

Not done in this step: no RL, environment, episode, provider or optimizer was added or used; no formal test was started; no test score, cache or relation truth was read; no model was loaded; no scene was generated; ELASTIC-02..04 are not allocated; Family B and RoboCasa are not resumed; no method or training configuration was changed.

Build checks: {"base_csv_crosscheck_all_equal": true, "e1_matches_committed_record": true, "all_final_checkpoint_checks_true": {"R2-B0-s0": true, "R1-B1K-s0": true, "R-TB-K-1": true, "R1-B2-s0": true, "R-TB-DK-1": true, "R-TB-E-0": true, "R-TB-E-1": true}, "head": "477de5dec2d59e22eaa0f6a840bb0d1b80f0a687", "forbidden_wording_hits": [], "protected_inputs_unchanged": true, "created": "2026-10-02T13:08:20Z", "zero_env": {"torch_imported": false, "episodes_run": 0, "models_loaded": 0, "test_result_files_opened": 0, "test_cache_or_relation_truth_opened": 0, "test_scene_capture_files_byte_hashed_not_parsed": 120}}
