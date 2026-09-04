# Run Manifest

- round_id: stage7_6_auto_graph_exploration
- mode: reward_only
- entry_gate: alternative_structural_support=true; recovery_structural_support=true
- gpu_query: sudo_noninteractive_failed; direct_fallback_noninteractive
- embedding_jobs: 9
- selection_jobs: 12
- stability_jobs: 3
- selected_K: 7
- selected_change_point_quantile: 0.90
- manual_graph_remains_main: true
