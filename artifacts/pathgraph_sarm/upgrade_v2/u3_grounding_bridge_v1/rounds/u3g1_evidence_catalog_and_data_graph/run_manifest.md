# Run Manifest

- round_id: `u3g1_evidence_catalog_and_data_graph`
- purpose: Build the train-only finite evidence catalog and observed data-only transition graph.
- generated_at: `2026-09-06`
- repo_root: `/home/__compress_data/xushijie/graph_github_upload`
- code_baseline: `c455d993a76231059b14b06ab667ae387edccef2` plus maintenance branch edits
- gpu_query: `direct_fallback_noninteractive`; CPU-only deterministic derivation, no GPU computation
- api_calls: `0`; API keys were not read
- commands: rebuild train cluster profiles with segment counts; rebuild data-only graph; retain rare edges
- result: `11` nodes, `30` transitions, start `C03`, success `C02`, input split `train`
- public_zip_policy: update only `downloads/upgrade_v2/U0_U1_complete.zip`
- internal_zip_output: `/tmp/pathgraph_u3g_stage_zips/u3g1_evidence_catalog_and_data_graph.zip`
- omitted_payloads: raw JSONL/NPZ/checkpoints/logs and secrets remain excluded or represented by placeholders
