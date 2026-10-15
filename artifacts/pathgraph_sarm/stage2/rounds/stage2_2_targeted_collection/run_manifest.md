# Run Manifest

- round_id: stage2_2_targeted_collection
- purpose: transparent scripted-oracle evidence collection and final task selection
- started_at: 2026-09-02T16:55:45.809577+00:00
- finished_at: 2026-09-02T16:55:45.809590+00:00
- repo_root: /home/xushijie/CUPID
- git_commit: unavailable (CUPID root is not a git worktree)
- python: python 3.x
- gpu_ids: none (CPU-only evidence extraction and deterministic scripted collection)
- command: python tools/stage2/stage2_pipeline.py --mode all
- gpu_ids: none (no GPU rollout; deterministic CPU scripted controller)
- controller_source: scripted_oracle
