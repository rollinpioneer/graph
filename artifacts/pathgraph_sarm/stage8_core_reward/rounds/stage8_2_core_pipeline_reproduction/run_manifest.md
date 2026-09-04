# Run Manifest

- round_id: stage8_2_core_pipeline_reproduction
- generated_at: 2026-09-03T16:29:11.342490+00:00
- purpose: Independent immutable-checkpoint inference and reward pipeline reproduction.
- gpu_ids: 4,5
- code_commit: 2941eba9427a3398d3cbe26b10c01e3ece56bd18

## Executed Commands
- `run_frozen_reward_inference.py x 9`
- `merge_final_ensemble_predictions.py`
