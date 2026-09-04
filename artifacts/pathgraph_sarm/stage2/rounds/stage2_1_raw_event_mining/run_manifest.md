# Run Manifest

- round_id: stage2_1_raw_event_mining
- purpose: decode raw rollouts and mine evidence-backed events
- started_at: 2026-09-02T16:55:45.536305+00:00
- finished_at: 2026-09-02T16:55:45.536320+00:00
- repo_root: /home/xushijie/CUPID
- git_commit: unavailable (CUPID root is not a git worktree)
- python: python 3.x
- gpu_ids: none (CPU-only evidence extraction and deterministic scripted collection)
- command: python tools/stage2/stage2_pipeline.py --mode all
- input_manifest: /home/xushijie/CUPID/artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl
- forbidden_gt_input: /home/xushijie/CUPID/artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/episode_events.jsonl
