# GFP-TASK4-GLOBAL-DIAGNOSIS-V1

Date: 2026-10-16. Authorized: existing data/results and frozen inference only.

Root: `/home/xushijie3/work/gfp_task4_global_diagnosis_v1_20261016`.
Branch: `codex/gfp-task4-global-diagnosis-v1-20261016`.
Branch link: https://github.com/rollinpioneer/graph/tree/codex/gfp-task4-global-diagnosis-v1-20261016
Starting archive commit: `1ccb0d283dbc2a31eeebdbbf849c78fb3bb6be82`.

Question: does evidence independently favor insufficient behavior support (H-data), topological value mismatch (H-value), or behavior/policy learning mismatch (H-policy) for task4 versus task1?

## Frozen bounds and order

1. Read the full existing OGBench 1.1.0 HumanoidMaze-medium navigate training and validation data. Count task1/task4 support with identical definitions: start-cell visit followed by goal-radius 0.5 m hit within an original episode; also report start-radius 1 m sensitivity. Do not call these changing-collection-goal paths expert demonstrations. Count directed edges and consecutive cell-turn triples on all shortest routes, unique contributing episodes, durations and posture/velocity summaries.
2. Reuse existing task1 Z01 and task4 Z2 frozen inference and evaluation records. Keep original random batches and normalization; never renormalize Guidance by distance/behavior group.
3. Additional pure inference: task1/task4, seeds 0/1, checkpoints 200k/600k/1M. Select states before any new model scoring. On the union of shortest-route cells, classify observed next-100-step motion into detour progress (BFS improves >=1 cell, Euclidean distance grows >0.5 m), aligned progress (BFS improves >=1 cell, Euclidean distance decreases >0.5 m), regress (BFS worsens >=1 cell), or stationary (same BFS cell-distance and displacement <0.5 m). All windows stay in one original episode. Per task/split/group select at most 256 states and at most 8 per original episode, RNG seed 20261016. Thus <=4,096 unique selected task-state entries, <=24,576 model-state evaluations, eight fixed noise/time trials each (<=196,608 model-state-noise trials). All seeds/checkpoints see the same selected states and noises.
4. Measure Actor/VaBC data-action RMSE (single, average and best of eight), paired Actor/VaBC output discrepancy, unweighted VaBC flow-matching residual, target ensemble Q scores and data-versus-proposal ranks. These are fit/support and scoring diagnostics, not simulated action effects or true action advantages. No direction is inferred from joint torque signs. Any additional scoring on stratified batches must not be represented as training-normalized Guidance.
5. Aggregate by original trajectory, report train/held-out validation separately and each training seed/checkpoint. Use trajectory resampling intervals for differences; controls for position, posture and speed are associative. Reuse evaluations for performance; no new evaluation episodes.

## Permission and stop rules

Environment construction/reset/step, training/update, hyperparameter changes, new modules and new weights are prohibited. Parameter/optimizer/Agent RNG hashes must stay unchanged. Data, prior weights/results remain read-only. No extrapolation of task1 Q to task4 reward semantics. Missing long-term action-ground-truth remains unknown. If no independent mechanism evidence separates the hypotheses, stop this diagnosis and report unresolved explanations rather than expanding experiments.

## Deliverable

One joint report with a hypothesis decision table, compact data/inference tables and necessary scientific figures. Explicit lightweight code/config/card/results/receipts archived on this branch; full arrays, data, weights and logs remain on server. GPU inference resource cost recorded separately from prior reused computation. Zero new environment steps and zero optimizer updates.
