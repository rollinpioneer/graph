# u2_1_event_candidates_and_weak_labels

- round_id: `u2_1_event_candidates_and_weak_labels`
- purpose: Generate causal sensor-change candidates, calibrated weak posteriors, and unknown states.
- execution_status: `PASS` (preserved result audit; no payload rerun required)
- evidence_start: `2026-09-05T16:43:00.146351+08:00`
- evidence_end: `2026-09-05T18:48:00.311902+08:00`
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
"$PYTHON_BIN" -m upgrade_v2.u2.cli write-weak-rules --output "$U2_WEAK/configs/weak_rules.yaml"
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli extract-event-candidates --dataset "$U2_DATA/formal" --rules "$U2_WEAK/configs/weak_rules.yaml" --output-root "$U2_WEAK/candidates" --manifest "$U2_WEAK/manifests/event_candidate_manifest.csv"
```
```bash
"$PYTHON_BIN" -m upgrade_v2.u2.cli aggregate-weak-events --candidate-root "$U2_WEAK/candidates" --dataset "$U2_DATA/formal" ...
```
