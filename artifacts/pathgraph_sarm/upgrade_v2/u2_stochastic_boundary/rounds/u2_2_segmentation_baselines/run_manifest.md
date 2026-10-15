# u2_2_segmentation_baselines

- round_id: `u2_2_segmentation_baselines`
- purpose: Fit and evaluate uniform, causal hysteresis, and offline multivariate change-point baselines.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T16:45:27.349737+08:00`
- evidence_end: `2026-09-05T18:48:00.426109+08:00`
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
"$PYTHON_BIN" -m upgrade_v2.u2.cli run-segmentation-baselines --dataset "$U2_DATA/formal" --weak-posteriors "$U2_WEAK/posteriors" --methods uniform,sensor_hysteresis,multivariate_change_point --selection-split val ...
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli evaluate-segmentation-baselines --dataset "$U2_DATA/formal" --prediction-root "$U2_BASELINES/predictions" --selection "$U2_BASELINES/selection/baseline_selection.csv" --split test ...
```
