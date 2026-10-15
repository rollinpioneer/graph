# u2_3_causal_boundary_models

- round_id: `u2_3_causal_boundary_models`
- purpose: Train three frozen boundary variants, select on validation only, and run val/test inference.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T17:43:59.750345+08:00`
- evidence_end: `2026-09-05T18:48:00.541372+08:00`
- manifest_generated_at: `2026-09-05T18:51:27.411228+08:00`
- code_commit: `8886b0731b1792fb98dcb1ced5a0115537728c90`
- gpu_ids: `1,4,6`
- selection_split: `val` where model/config selection applies
- test_used_for_selection: `False`
- archive_policy: `single cumulative ZIP only; no per-round ZIP emitted`

## Reproduction commands

```bash
source "$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/u2_env.sh"
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli build-boundary-jobs --mode formal --dataset "$U2_DATA/formal" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-jobs --job-table "$U2_ROUNDS/u2_3_causal_boundary_models/tables/u2_boundary_formal_jobs.tsv" --gpu-ids 1,4,6 ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli select-boundary-checkpoints --job-root "$U2_MODELS/formal" --job-table ... --split val ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-inference-jobs --gpu-ids 1,4,6 ...
```
