# u2_5_budgeted_correction

- round_id: `u2_5_budgeted_correction`
- purpose: Compare zero-label, fixed oracle-budget, and query-strategy correction under a frozen protocol.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T17:53:30.755641+08:00`
- evidence_end: `2026-09-05T18:48:00.770933+08:00`
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
"$PYTHON_BIN" -m upgrade_v2.u2.cli build-query-queues --dataset "$U2_DATA/formal" --weak-posteriors "$U2_WEAK/posteriors" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli reveal-oracle-clips --queue-root "$U2_BUDGET/queues" --dataset "$U2_DATA/formal" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-value-jobs --gpu-ids 1,4,6 ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-budgeted-correction --job-root "$U2_BUDGET/models" --dataset "$U2_DATA/formal" ...
```
