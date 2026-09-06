# Run Manifest

- round_id: `u3g3_validation_selection_and_test`
- purpose: Preserve validation-only graph selection and record the single final test evaluation after metric correction.
- generated_at: `2026-09-06`
- repo_root: `/home/__compress_data/xushijie/graph_github_upload`
- code_baseline: `9b33044` plus this protocol-conformance rerun
- gpu_query: `direct_fallback_noninteractive`; no GPU computation was started
- api_calls: `0`; API keys were not read
- commands: `evaluate-graphs --split val`; `select-graphs`; `evaluate-graphs --split test` once after evaluator correction
- result: selection remains validation-locked to `data_only_transition_graph`; test is final-evaluation-only
- test_protocol: `test_not_used_for_selection=true`; `test_run_once=true`; final test metrics replaced by one corrected final evaluation
- public_zip_policy: update only `downloads/upgrade_v2/U0_U1_complete.zip`
- internal_zip_output: `/tmp/pathgraph_u3g_stage_zips/u3g3_validation_selection_and_test.zip`
- omitted_payloads: raw JSONL/NPZ/checkpoints/logs and secrets remain excluded or represented by placeholders
