# Run Manifest

- round_id: `u3g2_ground_existing_llm_candidates`
- purpose: Rebuild deterministic grounding metadata for the two existing semantic candidates.
- generated_at: `2026-09-06`
- repo_root: `/home/__compress_data/xushijie/graph_github_upload`
- code_baseline: `c455d993a76231059b14b06ab667ae387edccef2` plus maintenance branch edits
- gpu_query: `direct_fallback_noninteractive`; CPU-only deterministic derivation, no GPU computation
- api_calls: `0`; API keys were not read
- commands: reuse normalized candidate JSON; rebuild node and edge grounding; assemble six thresholds; validate grounding gate
- result: `6` grounded variants, source SHA checks pass, hallucinated raw evidence IDs `0`
- public_zip_policy: update only `downloads/upgrade_v2/U0_U1_complete.zip`
- internal_zip_output: `/tmp/pathgraph_u3g_stage_zips/u3g2_ground_existing_llm_candidates.zip`
- omitted_payloads: raw JSONL/NPZ/checkpoints/logs and secrets remain excluded or represented by placeholders
