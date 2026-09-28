# CP-DISR S0 — Scientific Code Audit

## Status

- `execution_state`: **STOPPED**
- `authorized`: true for S0 only
- `master_id`: `CP-DISR-FINAL-EXEC-3.0`
- `configsha8`: `44fc2b9b`
- `active source`: `/home/xushijie2/graph_cp_disr_final_s0` at `cd95646ee0502419052210590b280750374a0401`
- `RL training attempts`: 0
- `elastic attempts`: 0
- `provider/API requests`: 0
- `test evidence`: 54/54 selected production/unit regressions completed with positive results

The authoritative Research Content v5 and Experimental Plan v3 attachments are now supplied and hash-recorded. S0 remains STOPPED because the active source and candidate worktrees contain no B_PLAN planner/development entry point or bound legal development-scene executor, so the required up-to-10-episode B_PLAN validation cannot be honestly run.

## Thirteen audit items

1. **B1-K effect permission — COMPLETE / engineering.** Existing B1-K uses `GraphEncoder(G_K)` and therefore can see current graph topology, including ADD/DEL edges and proposition values. It does not have an explicit candidate-local effect-token input; the old `phi_K(c, 0)` path remains successor-free.
2. **B1-K+E — COMPLETE / positive.** Added `EffectTokenReadout` and `effect_fuse`. It consumes PRE_POS, PRE_NEG, ADD, DEL, UNKNOWN, conditional guards, ordered typed arguments and grounded bindings. It does not call `nominal_overlay`, `successor`, `four_views`, reward, evaluator, future state, or oracle geometry.
3. **A_STAT — COMPLETE / positive.** Added `A_STAT` as the Full-shaped path with only the prior input replaced by `S_R = Z_H(F) - Z_K(F)`, implemented as `delta.zh - delta.zk`; it reuses the same candidate context, `phi_K`, `phi_P`/AnchoredPrior, Actor/V/Q, residual bound and optimizer interface.
4. **Encoder/global DAG — COMPLETE / engineering.** Four shared `RGCNConv` layers use 128 channels, 4 bases, mean aggregation and root/self transform, followed by ReLU and node-wise LayerNorm. Goal rows and a nonlinear global row are produced by `GoalReadout`; policy context reads pooled/global rows, so strict end-to-end locality is not claimed.
5. **Graph/hub/distance — COMPLETE / engineering.** Synthetic fixture graph metrics are in `graph_structure_report.json`. `GRAPH_REMOTE` is `NOT_APPLICABLE`: no pre-registered legal remote distance bin exists in active inputs; no fake nodes, predicates or relations were added.
6. **Four-view same-patch/same-R alignment — COMPLETE / positive.** `four_views()` uses the same template/facts and one contract for K/H/current/nominal views. Existing alignment, nominal-isolation and new null tests passed.
7. **Q — COMPLETE / positive.** `Policy` produces all-candidate Q values; PPO selects only the actually executed candidate. Q/V targets are detached in production helpers, and no hidden truth, R* or Q_ref loader was found.
8. **Gradient clipping — COMPLETE / positive.** Formal `PPO.update` now measures distinct pre/post global norms around the existing threshold 0.5 and records trigger, optimizer step, method/seed through the update log context. No no-clip run was added.
9. **B_PLAN development — STOPPED / NOT_RUN.** Static discovery of the active source (`src`, `scripts`, `tools`) and four candidate worktrees found no `B_PLAN` planner/development entry point, legal development-scene binding, or max-node/depth search implementation. See `b_plan_dev_report.md`.
10. **Relation truth/utility/opportunity — COMPLETE / UNBOUND.** Relation schema/validation is present, but T_P truth, utility, opportunity and Q_ref are not bound into the active training loader. They remain analysis-only/UNBOUND; no values were invented.
11. **A_CAT capacity — COMPLETE / engineering.** Allocated/active/finite-gradient counts, parameter-shape hash, forward calls and CPU fixture timing are in `parameter_capacity_report.csv`. Full/A_STAT shape and active counts match; A_CAT is reported as a distinct wider multi-view path, not a strict single-variable comparison.
12. **B1-H — COMPLETE / engineering.** Existing implementation remains the old `phi_K(c, Z_K)` slot and is not silently relabeled as matched-v1; it remains optional and was not trained or allocated an elastic slot.
13. **Budget/checkpoint/counters — COMPLETE / engineering.** No S0 run changed an experiment envelope or checkpoint. Existing PPO/update and persistence paths were audited; unknown historical counters remain NOT_MEASURED rather than zero.

## Property/regression evidence

See `property_tests.json`, `property_tests.md`, and the raw pytest command result summarized in the stage manifest.
