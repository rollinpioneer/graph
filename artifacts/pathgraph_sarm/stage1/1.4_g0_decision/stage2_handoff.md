# Stage 2 handoff

G0 status: **SWITCH**.

Direct inputs:
- `artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl`
- `artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/trajectory_tags.csv`
- `artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/coverage_matrix.csv`

Candidate semantic nodes (not frozen):
- `square`: reach_object -> grasp_object -> place_object -> release_object -> success_terminal
- `transport`: grasp_source -> lift_object -> transport_object -> place_target -> release_object -> success_terminal

First command after evidence is available: create Graph spec v1 and the annotation manual, then annotate failure_onset/recovery_complete and alternative path signatures on complete episodes.
