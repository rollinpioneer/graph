# Run Manifest

- round_id: stage1_4_g0_decision
- generated_at: 2026-09-02
- repo_root: /home/xushijie/CUPID
- code_commit: unavailable (CUPID root is not a git worktree)
- gpu_ids: none (CPU-only scoring and gate)
- command: python tools/stage1/select_graph_tasks.py --config configs/stage1/stage1.yaml --task-summary artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/task_structure_summary.csv --coverage artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/coverage_matrix.csv --tags artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/trajectory_tags.csv --manifest artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl --output-dir artifacts/pathgraph_sarm/stage1/1.4_g0_decision
