# EG-POSE-RECOVERABILITY-A-V1
Status: REGISTERED_BEFORE_BRANCH_RANKING_RESULTS
Branch: [codex/eg-pose-recoverability-a-v1-20261018](https://github.com/rollinpioneer/graph/tree/codex/eg-pose-recoverability-a-v1-20261018)
Parent: 56381890d6f77b854e97d4ba10b982067415e9b5
Authorization: user requested execution, no training.

## Frozen population and provenance
Use the prior subset-model development Top-1 cohort: scenes 8,36,41,44,58,65,73,75,80,86, frames 0,4,...,252. Total 640 cases: 187 original failures and all 453 original successes at native friction mu<=0.4 (positive quality, noncollision). Prior decoder-lattice cache reports 170/187 recoverable. Revalidate case identity/counts and the original action. Do not replace the cohort, model, input clouds or sampled points. Freeze input/prediction/cache/checkpoint hashes before ranking analysis. Use cached native geometry quality where identical; success cases need up to 453*48 additional native calls. Failure cache retains 187*48 values.
Frozen subset checkpoint SHA256: 9eb793cfdcaf18cf6dfd1a41294a97db9eddf2b3e97cc208ab21009ad1d71950.

## Geometry and candidate rules
Keep original center, approach, width (official clamp), height and object association. Grid: 12 rotations relative to original rotation, k*15 degrees, and depths .01,.02,.03,.04m. Map offset k to class (original_angle_class+k)%12; parallel-jaw pi symmetry must be checked against decoder matrices. Original action is reused exactly. Do not use label-coordinate rotations or label widths.
Recover original 12 angle and 4 depth logits by frozen inference on saved inputs (no retraining). Replay original batches of four contiguous frames for matching selected stride-4 outputs; require original raw predictions allclose rtol=2e-4, atol=2e-5, selected Top-1 unique association and unchanged angle/depth class. Stop affected stage on mismatch, do not silently rematch different geometry.
1 EG original top angle x top depth.
2 Cartesian top3 angle x top2 depth.
3 Cartesian top2 angle x top3 depth.
4 Joint top6 sorted by softmax(angle[:12])*softmax(depth[:4]); equivalently sum of corresponding raw logits; excluded background classes, no temperature tuning.
5 Uniform random6 without replacement.
6 Keep EG original plus uniform5 from remaining47.
7 Joint top12.
8 All48.
Additional matched-budget controls: uniform12 and keep-original-plus11. Ties: class index ascending, then angle-major/depth-minor flat index. Deterministic alternatives must contain original (assert; stop on problematic ties). Random coverage is exact hypergeometric expectation, not a lucky random seed. Report per-case random selection risk exactly.

## Outcomes
On failures report recoveries /187 and /170, per scene; repeat /170 denominator only after cache count verification. Define pass by native positive mu<=0.4000001. Best feasible means minimum positive mu; list all equally-best feasible cells, ranks, and a canonical one by joint-score rank then flat index. Also report ranks of any feasible cell to avoid arbitrary tie attribution.
Angle-only, depth-only, either-single, both-single, joint-only, no-recovery categories: feasibility counterfactuals, not causal head errors.
On all453 successes report candidate oracle retention and fraction of invalid alternatives, forced-change top-ranked-alternative failure, uniform-selection failure within each set; unchanged joint argmax must reproduce baseline. These are risks, not performance of a trained selector or robot success.
No AP improvement claim; no quality-aware selector deployed.

## Statistical decision and stopping
Analysis unit is scene (10 clusters); primary rates are case-weighted on the frozen cohort. Paired cluster bootstrap 20,000 resamples of scenes with replacement, seed20261018. Report nominal95% percentile intervals.
Four predeclared candidate-rule tests: top3x2, top2x3, joint6 vs keep-original random6; joint12 vs keep-original random12. Bonferroni one-sided lower 1.25th percentile (family alpha .05 /4) must be strictly above zero for a reliable ranking advantage. Random expectation computed exactly from good-cell counts. Compare to plain random as descriptive secondary; all48 is only offline upper bound.
If no strategy passes, or no feasible recovery exists, pause training that depends on EG class ranking. If one passes, this supports only retaining class rankings for further investigation, not automatic training or practical benefit. No arbitrary minimum recovery percentage is chosen after results. Report absolute oracle gain, opportunity missed, success risks and uncertainty; do not conclude joint modeling is required from either-single recoverability alone.
Known historical187/170 totals informed this card; new ranked-strategy outcomes were not inspected before registration. All work is exploratory development material, not independent test confirmation.

## Resources / integrity
Zero network training. GPU only for frozen replay on currently free device; CPU native scoring per scene parallel up to10 single-thread low-priority workers, below user32 ceiling. Reuse existing environment, no shared modifications. Preserve all source data and earlier results. No scheduled monitor. Validate source/code identities and a small original-action/decoder correspondence check. Stop on material correctness failure; publish partial/blocked results honestly. Archive explicit code/config/card/light results on this branch, leave large data on server.
