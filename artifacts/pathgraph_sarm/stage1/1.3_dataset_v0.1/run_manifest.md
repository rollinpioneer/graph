# Run Manifest

- round_id: stage1_3_dataset_v0_1
- generated_at: 2026-09-02
- repo_root: /home/xushijie/CUPID
- code_commit: unavailable (CUPID root is not a git worktree)
- gpu_ids: none (CPU-only manifest/split construction)
- command: python tools/stage1/build_dataset_v01.py --config stage1_project/configs/stage1/stage1.yaml --inventory artifacts/pathgraph_sarm/stage1/1.1_asset_inventory/asset_inventory.csv --tags artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/trajectory_tags.csv --path-signatures artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/path_signatures.jsonl --output-dir artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1
