# Run Manifest

- round_id: stage1_2_trajectory_coverage
- generated_at: 2026-09-02
- repo_root: /home/xushijie/CUPID
- code_commit: unavailable (CUPID root is not a git worktree)
- gpu_ids: none (metadata/event scan is CPU-only)
- command: python tools/stage1/analyze_trajectory_structure.py --config stage1_project/configs/stage1/stage1.yaml --inventory artifacts/pathgraph_sarm/stage1/1.1_asset_inventory/asset_inventory.csv --output-dir artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage --reuse-manual-labels
