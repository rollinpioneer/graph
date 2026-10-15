# Run Manifest

- round_id: stage7_4_scaling_and_coverage
- mode: reward_only
- gpu_query: sudo_noninteractive_failed; direct_fallback_noninteractive
- gpu_ids: 0,1,3,4,5
- coverage_training_jobs: 6
- coverage_inference_jobs: 18
- controlled_symbolic_stress_settings: 144
- provenance: real_or_environment_rollout; controlled_symbolic_stress; derived_counterfactual_with_empirical_noise
