# Run Manifest

- round_id: stage3_1_runtime_input_patch
- generated_at_utc: 2026-09-03T01:55:43+00:00
- repo_root: /home/__compress_data/xushijie/CUPID
- m1_freeze_root: /home/__compress_data/xushijie/CUPID/artifacts/pathgraph_sarm/stage2/m1_freeze_v1
- GPU IDs: none (CPU-only input repair; no training, inference, or feature extraction)
- command: python3 tools/stage3/build_runtime_input_patch.py --repo-root /home/xushijie/CUPID --m1-root artifacts/pathgraph_sarm/stage2/m1_freeze_v1 --output-dir artifacts/pathgraph_sarm/stage3/rounds/stage3_1_runtime_input_patch
- M1 write policy: immutable/read-only
