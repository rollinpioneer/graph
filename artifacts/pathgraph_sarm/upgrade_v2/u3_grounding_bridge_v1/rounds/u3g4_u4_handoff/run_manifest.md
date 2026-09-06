# Run Manifest

- round_id: `u3g4_u4_handoff`
- purpose: Produce the U4 data-only handoff with truthful element-level support counts and package provenance.
- generated_at: `2026-09-06`
- repo_root: `/home/__compress_data/xushijie/graph_github_upload`
- code_baseline: `c455d993a76231059b14b06ab667ae387edccef2` plus maintenance branch edits
- gpu_query: `direct_fallback_noninteractive`; CPU-only deterministic derivation, no GPU computation
- api_calls: `0`; API keys were not read
- commands: compute episode-local node/edge support counts from cached segments; rebuild U4 handoff; validate package policy
- result: `GO_U4_DATA_ONLY`; `41` elements; `2` queued queries; U3 remains `U3_INCONCLUSIVE`
- public_zip_policy: update only `downloads/upgrade_v2/U0_U1_complete.zip`
- internal_zip_output: `/tmp/pathgraph_u3g_stage_zips/u3g4_u4_handoff.zip`
- omitted_payloads: raw JSONL/NPZ/checkpoints/logs and secrets remain excluded or represented by placeholders
