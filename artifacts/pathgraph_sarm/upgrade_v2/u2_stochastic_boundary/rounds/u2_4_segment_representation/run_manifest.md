# u2_4_segment_representation

- round_id: `u2_4_segment_representation`
- purpose: Build unknown-aware segments, history embeddings, clusters, and observed transition summaries.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T17:50:01.609772+08:00`
- evidence_end: `2026-09-05T18:48:00.655741+08:00`
- manifest_generated_at: `2026-09-05T18:51:27.411228+08:00`
- code_commit: `8886b0731b1792fb98dcb1ced5a0115537728c90`
- gpu_ids: `none`
- selection_split: `val` where model/config selection applies
- test_used_for_selection: `False`
- archive_policy: `single cumulative ZIP only; no per-round ZIP emitted`

## Reproduction commands

```bash
source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli build-segments --dataset "$U2_DATA/formal" --boundary-source "$U2_SEGMENTS/configs/boundary_source_lock.json" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli encode-segments --segments "$U2_SEGMENTS/segments" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli cluster-segments --methods raw_observable_kmeans,history_embedding_kmeans,history_plus_event_posterior_kmeans --clusters 5,7,9,11 --seeds 631,632,633 ...
```
