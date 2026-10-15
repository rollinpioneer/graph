# U2 handoff correction run manifest

- round_id: `u2_handoff_patch_v1`
- purpose: episode-local boundary metrics, incoming-transition reward attribution, and U3 train-only handoff
- scope: same explicit-state stochastic simulator family only
- baseline_commit: `4fd1daa373f7f7cdf99604476c18af2644682a63`
- execution_date: `2026-09-05`
- successful_execution_source_commit: `67622e7a9314db645f1dae70d66ca5c9a720fd79`
- boundary_command: `python -m upgrade_v2.u2_handoff_patch.cli recompute-boundaries --splits val,test --tolerances 1,2`
- reward_command: `python -m upgrade_v2.u2_handoff_patch.cli recompute-reward --split test --bootstrap 5000 --seed 20260957`
- compute: CPU cache recomputation; no model retraining
- gpu: not used for this CPU-only correction pass
- external_llm: not called
- outputs: `boundaries_v2/`, `reward_v2/`
- boundary_cache_status: `BOUNDARY_CACHE_RECOMPUTED`; 0 missing predictions; input inventory SHA256 `322ee3ea9f9a5d6030e874e223b176d7f69980a1a6ce053883d4511853d0b8af`
- reward_cache_status: `REWARD_CACHE_RECOMPUTED`; 0 missing items; input inventory SHA256 `60389719b0ebfdf69b397a54b13831b0d4986c4222a03929112dae69d434bd72`
- omitted payloads: NPZ, Parquet, JSONL, checkpoints, and logs remain represented by repository manifests/placeholders
