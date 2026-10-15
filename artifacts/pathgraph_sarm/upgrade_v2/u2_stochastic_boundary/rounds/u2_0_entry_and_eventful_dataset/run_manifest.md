# u2_0_entry_and_eventful_dataset

- round_id: `u2_0_entry_and_eventful_dataset`
- purpose: Freeze simulator-only entry, eventful trajectories, family split, and exact state restore.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T16:39:34.258731+08:00`
- evidence_end: `2026-09-05T18:48:00.197464+08:00`
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
"$PYTHON_BIN" -m upgrade_v2.u2.cli collect-eventful-dataset --mode formal --root-families 120 --rollouts-per-family 6 --scenarios nominal_success,grazing_contact,slip_recovery,obstacle_detour,terminal_collision,stagnation --split-ratios 0.70,0.15,0.15 --split-unit root_family_id --seed 20260953 --output-root "$U2_DATA/formal"
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli validate-eventful-dataset --dataset "$U2_DATA/formal" ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli audit-observation-action-alignment --dataset "$U2_DATA/formal" ...
```
