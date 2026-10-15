# u2_6_reward_impact_and_gate

- round_id: `u2_6_reward_impact_and_gate`
- purpose: Fit independent continuation q/D references, aggregate reward by boundary source, and finalize U2 gate.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T17:35:44.560576+08:00`
- evidence_end: `2026-09-05T18:48:00.886751+08:00`
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
"$PYTHON_BIN" -m upgrade_v2.u2.cli collect-boundary-value-continuations --dataset "$U2_DATA/formal" --anchors-per-family 4 --continuations-per-anchor 3 --horizon 32 ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli launch-value-jobs --gpu-ids 1,4,6 ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli aggregate-reward-by-boundary --dataset "$U2_DATA/formal" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-boundary-reward-impact --bootstrap 5000 ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli finalize-u2 --repo-root "$REPO_ROOT" --u2-root "$U2_ROOT" --final-root "$U2_FINAL"
```
