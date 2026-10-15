# PathGraph-SARM Stage 1 complete manifest

- Completed sub-stages: 1.1 asset inventory, 1.2 trajectory coverage, 1.3 dataset v0.1, 1.4 G0 decision.
- Dataset version: `pathgraph_stage1_dataset_v0.1`.
- Source data is read-only and remains at the paths recorded in `source_fingerprint.csv` and `episode_manifest.jsonl`.
- GPU: not used; all work was metadata/event/split processing and did not require privileged GPU inspection.
- G0 status: `SWITCH` because current evidence has no verifiable alternative-order or episode-internal recovery events. The targeted collection plan is the actionable next input.
- Direct Stage 2 inputs: `dataset_v0.1/episode_manifest.jsonl`, `dataset_v0.1/splits.csv`, `trajectory_tags.csv`, `coverage_matrix.csv`, `1.4_g0_decision/selected_tasks.yaml`, and `1.4_g0_decision/stage2_handoff.md`.
