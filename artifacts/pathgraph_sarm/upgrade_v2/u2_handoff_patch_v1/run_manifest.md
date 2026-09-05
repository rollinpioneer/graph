# U2 handoff correction run manifest

- round_id: `u2_handoff_patch_v1`
- purpose: episode-local boundary metrics, incoming-transition reward attribution, and U3 train-only handoff
- scope: same explicit-state stochastic simulator family only
- baseline_commit: `4fd1daa373f7f7cdf99604476c18af2644682a63`
- execution_date: `2026-09-05`
- boundary_command: `python -m upgrade_v2.u2_handoff_patch.cli recompute-boundaries --splits val,test --tolerances 1,2`
- reward_command: `python -m upgrade_v2.u2_handoff_patch.cli recompute-reward --split test --bootstrap 5000 --seed 20260957`
- compute: CPU cache recomputation; no model retraining
- gpu: not used for this CPU-only correction pass
- external_llm: not called
- outputs: `boundaries_v2/`, `reward_v2/`
- omitted payloads: NPZ, Parquet, JSONL, checkpoints, and logs remain represented by repository manifests/placeholders

