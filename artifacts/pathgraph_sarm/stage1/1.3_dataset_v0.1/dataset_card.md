# Dataset card: pathgraph_stage1_dataset_v0.1

- Episode unit: one complete CUPID rollout manifest row; source files are referenced read-only.
- Tasks: 2; episodes: 200.
- Labels: outcome from existing `success` field; semantic subgoal/recovery labels are not present.
- Split: deterministic seed 20260831, per-task shuffle with ratios train/val/test={'train': 0.8, 'val': 0.1, 'test': 0.1}; group_id=episode_id prevents cross-split episode leakage.
- Known gap: no observed alternative-order paths or explicit recovery events; G0 must gate further collection/annotation.
