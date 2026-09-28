# Source diff evidence

## Active source

- Selected worktree: `/home/xushijie2/graph_cp_disr_final_s0`
- Branch: `codex/cp-disr-final-s0-audit`
- Base commit: `cd95646ee0502419052210590b280750374a0401`
- Fixed read-only evidence ref: `55a3b7ce35edebbbb8a587fe1e3a98ca1967db07`
- S0 working-tree source hashes are recorded in `source_identity.json`.

## Historical evidence comparison

The following is a read-only comparison from the fixed evidence ref to the selected base commit. It is historical evidence only; no historical run was re-executed.

```text
experiments/manifests/runtime_manifest_v211.yaml   |    6 +-
 reports/v13_r3/Results_v1.md                       |   14 +
 reports/v13_r3/checkpoint_index.json               |   34 +
 reports/v13_r3/cost_ledger.csv                     |   11 +
 reports/v13_r3/duration_audit.md                   |    3 +
 reports/v13_r3/full_eval_metrics.csv               |    3 +
 reports/v13_r3/full_eval_n_000000.json             |  134 +
 reports/v13_r3/full_eval_n_002048.json             |  134 +
 reports/v13_r3/full_train_metrics.csv              |    3 +
 reports/v13_r3/hard_error_full_stall.json          |   22 +
 reports/v13_r3/input_audit.json                    |   51 +
 reports/v13_r3/r3_summary.md                       |   14 +
 reports/v13_r3/unresolved_items.md                 |    7 +
 reports/v13_r3_recovery/Results_R3_Recovery.md     |   25 +
 reports/v13_r3_recovery/claim_evidence_table.md    |    9 +
 reports/v13_r3_recovery/cost_ledger.csv            |   11 +
 .../v13_r3_recovery/full_original_absent_dev10.csv |   21 +
 reports/v13_r3_recovery/full_prior_conditional.csv |    9 +
 .../full_prior_conditional_train.csv               |    9 +
 reports/v13_r3_recovery/matched_comparison.csv     |    5 +
 .../v13_r3_recovery/matched_n2048_comparison.csv   |    5 +
 reports/v13_r3_recovery/unresolved_items.csv       |    4 +
 reports/v13_r3_recovery/unresolved_items.md        |    5 +
 runs/v13_r3/hard_error_full_stall.json             |   22 +
 runs/v13_r3/input_audit_corrected.json             |   51 +
 .../20260928T032333Z_4e75091f/README.md            |    3 +
 .../recovery_manifest.json                         |   10 +
 .../checkpoint_selection.json                      |   53 +
 .../checkpoints/final.json                         |    1 +
 .../checkpoints/final.rng.json                     | 7289 ++++++++++++++++++++
 .../checkpoints/n_000000.json                      |    1 +
 .../checkpoints/n_000000.rng.json                  | 6376 +++++++++++++++++
 .../checkpoints/update_complete_1.json             |    1 +
 .../checkpoints/update_complete_1.rng.json         | 6809 ++++++++++++++++++
 .../checkpoints/update_fragment_1.json             |    1 +
 .../checkpoints/update_fragment_1.rng.json         | 7289 ++++++++++++++++++++
 .../20260928T032333Z_4e75091f/decision_log.jsonl   | 2104 ++++++
 .../environment_snapshot.json                      |   13 +
 .../20260928T032333Z_4e75091f/episode_log.jsonl    |  264 +
 .../20260928T032333Z_4e75091f/episode_priors.jsonl |  265 +
 .../20260928T032333Z_4e75091f/eval_final.json      |  149 +
 .../20260928T032333Z_4e75091f/eval_metrics.csv     |    3 +
 .../20260928T032333Z_4e75091f/eval_n_000000.json   |  149 +
 .../20260928T032333Z_4e75091f/job_summary.json     |  117 +
 .../seed_0/20260928T032333Z_4e75091f/manifest.json |   25 +
 .../recovery_endpoint.json                         |  119 +
 .../20260928T032333Z_4e75091f/resolved_config.json |  116 +
 .../20260928T032333Z_4e75091f/resolved_config.yaml |  110 +
 .../seed_0/20260928T032333Z_4e75091f/resume.json   |   13 +
 .../20260928T032333Z_4e75091f/run_summary.md       |    8 +
 .../software_snapshot.json                         |    7 +
 .../20260928T032333Z_4e75091f/train_metrics.csv    |    3 +
 .../20260928T032333Z_4e75091f/transition_log.jsonl | 1935 ++++++
 .../20260928T032333Z_4e75091f/tuning_history.jsonl |    0
 runs/v13_r3_recovery/b2_eval_n_000000.json         |  149 +
 runs/v13_r3_recovery/b2_eval_n_002048.json         |  152 +
 .../v13_r3_recovery/b2_first_update_selfcheck.json |   17 +
 runs/v13_r3_recovery/b2_train.log                  |    8 +
 runs/v13_r3_recovery/b2_training_summary.json      |  120 +
 runs/v13_r3_recovery/execution_identity.json       |   10 +
 runs/v13_r3_recovery/full_absent_dev10.json        |  256 +
 .../full_absent_dev10_remaining.json               |  229 +
 runs/v13_r3_recovery/full_absent_remaining.log     |    5 +
 .../full_checkpoint_verification.json              |   17 +
 .../full_eval_execution_identity.json              |   10 +
 .../full_original_dev10_reused.json                |  149 +
 runs/v13_r3_recovery/full_prior_eval.log           |    5 +
 runs/v13_r3_recovery/full_prior_eval_status.json   |   11 +
 runs/v13_r3_recovery/full_prior_switch_rows.json   |  432 ++
 runs/v13_r3_recovery/full_s_train_absent.csv       |    5 +
 runs/v13_r3_recovery/full_s_train_original.csv     |    5 +
 runs/v13_r3_recovery/last_log_records.json         |   38 +
 runs/v13_r3_recovery/recovery_authorization.json   |  796 +++
 .../recovery_authorization_snapshot.json           |   17 +
 runs/v13_r3_recovery/shadow_forward_summary.json   |    4 +
 .../source_input_checkpoint_hashes.json            |   13 +
 runs/v13_r3_recovery/stall_phase_reconciliation.md |   16 +
 runs/v13_r3_recovery/stall_timeline.json           |   21 +
 src/cp_disr/phase_a_v13_r3.py                      | 1089 +++
 status/v13_r3.json                                 |   40 +
 status/v13_r3_recovery.json                        |   35 +
 tools/postprocess_r3.py                            |  155 +
 tools/recovery_alias_outputs.py                    |   21 +
 tools/recovery_b2_train.py                         |   70 +
 tools/recovery_correct_budget.py                   |   16 +
 tools/recovery_finalize.py                         |   33 +
 tools/recovery_full_absent_remaining.py            |   24 +
 tools/recovery_full_prior_eval.py                  |   62 +
 tools/recovery_prepare_evidence.py                 |   24 +
 tools/run_r3_d0.py                                 |  103 +
 90 files changed, 37999 insertions(+), 3 deletions(-)
```

## S0 working-tree changes

```text
src/cp_disr/neural.py   | 103 ++++++++++++++++++++++++++++++++++++++++++++++--
 src/cp_disr/torch_rl.py |  13 +++++-
 2 files changed, 111 insertions(+), 5 deletions(-)
```

Tracked historical worktrees were not modified. No reset, clean, force push, PR, provider call, or historical-run overwrite was performed.
