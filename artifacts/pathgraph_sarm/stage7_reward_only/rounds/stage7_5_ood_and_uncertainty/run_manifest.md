# Run Manifest

- round_id: stage7_5_ood_and_uncertainty
- mode: reward_only
- gpu_query: sudo_noninteractive_failed; direct_fallback_noninteractive
- gpu_ids: 0,1,3,4,5
- order_holdout_training_jobs: 6
- order_holdout_inference_jobs: 12
- perturbation_inference_jobs: 60
- beta_lcb: post_hoc_only
