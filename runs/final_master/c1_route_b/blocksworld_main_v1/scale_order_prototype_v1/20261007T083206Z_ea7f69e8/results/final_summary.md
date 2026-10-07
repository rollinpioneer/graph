# C1-BW-SCALE-ORDER-PROTOTYPE-V1 — final summary

Scope run: tier 1 of the plan only (step-0 error taxonomy, ONE new training `R-C1-SO-MG-S8-PAD-0`, development evaluation, branch recommendation). No second training, no external-method run.
verify.json: PASS — 292/292 new episodes + 48 planner references, 0 technical gaps, weights unchanged across evaluation, old result trees and the M1-GOAL reference checkpoint unchanged.
Registration commit `1f603112fabcfc9e5484736fb214d1713fa40596` was pushed before the training and before any new evaluation. All evaluation here is DEVELOPMENT material, not a final confirmation.

## 1. Step 0: error taxonomy (from saved logs, nothing re-queried)
The six-way classification (plan 3.3) was implemented here (audit script not available) and reproduces the audit's MG_C3 count on the scorer-control suite exactly: **102 raw non-optimal decisions** (B2+C3: 418).

MG_C3, raw first choices, scorer-control suite: neutral-preparation wrong 38 (+6 OTHER_AMBIGUOUS, which the audit appears to fold into its 44), wrong destruction choice 18, missed direct progress 17, refuse necessary destruction 12, non-optimal direct progress 8, avoidable destruction 3. By goal-tower height: h=4 49, h=3 48, h=2 5. B2+C3 raw errors are 360/418 missed direct progress.
Denominators (MG_C3): 1528 decisions; raw non-optimal 102; executed non-optimal 89; raw-optimal rewritten to non-optimal by C3 33 (single-step rewrites, not a trajectory-level causal effect); C3 left no optimal action 26; episodes with a raw error 37/112; unique (problem,state) raw errors 102. Full tables for 9 conditions on Confirm112-v2 and 4 on the scorer-control suite: `results/error_taxonomy.csv`, `error_denominators.csv` (the two suites are never pooled).

## 2. Training MG-S8-PAD
Padding of every parent trajectory to 6/7/8 blocks with table blocks (goal OnTable), exact planner relabelling of 22,338 decisions: remaining optimal length mismatches 0, original optimal actions missing from the padded optimal set 0, illegal original actions 0, exception trajectories 0; 1,476 decisions (6.6%) gained extra optimal actions (mean |A*| 1.078 -> 1.256). Label cost: 2,700 unique relabelled states, 139k planner nodes, ~1 s wall on 48 processes (negligible).
Training: same model/trainer/seed/hyper-parameters as M1-GOAL, init A02 B2 + the original head init; 100 epochs, 4,100 steps, 744,600 samples, NaN 0; each epoch 50% original / 50% pad (6/7/8 equally). Compute is not equal to M1-GOAL: 57.9 s/epoch (5,793 s) versus 35.1 s/epoch (3,508 s; GPUs were shared with other users in both runs, so wall times are only indicative); the graphs are larger. Final train mass 0.99998.

## 3. Board48 (factorial development board: n x k x h, 6 problems per cell) — success / all-steps-optimal
| condition | ALL | n6k1h2 | n6k1h4 | n6k2h2 | n6k2h4 | n8k1h2 | n8k1h4 | n8k2h2 | n8k2h4 |
|---|---|---|---|---|---|---|---|---|---|
| MG_C3 | 47 / 30 (48) | 6/6 | 6/3 | 6/5 | 6/4 | 6/5 | 5/2 | 6/5 | 6/0 |
| PAD_C3 | 47 / 30 (48) | 6/6 | 6/3 | 6/5 | 6/4 | 6/5 | 6/1 | 6/5 | 5/1 |
| B_G1C3 | 43 / 32 (48) | 6/6 | 3/2 | 6/6 | 5/3 | 6/6 | 5/1 | 6/6 | 6/2 |

PAD vs MG on the same 48 problems: success both 46, only PAD 1, only MG 1; all-steps-optimal both 29, 1 / 1. **No net difference.** MG itself already scales: success 24/24 at n=6, 23/24 at n=8; the drop is in optimality and it follows height, not object count: all-steps-optimal h=2 21/24, h=4 9/24 (n=6 18/24, n=8 12/24). MG_C3 vs B_G1C3: success +4 for MG (5 B_G1C3 step-cap failures, all in h=4 cells), all-steps-optimal -2 (3 vs 5 only-one).

## 4. Scorer-control suite, 112 problems (development material from now on)
| condition | ALL | T-MONO | T-BREAK | H4 | COLOR (16 pairs) |
|---|---|---|---|---|---|
| B_G1C3 | 109 / 88 | 32 / 25 | 32 / 27 | 14 / 7 | 31 / 29 |
| MG_C3 | 108 / 82 | 32 / 23 | 31 / 26 | 14 / 4 | 31 / 29 |
| PAD_C3 | 108 / 85 | 32 / 28 | 32 / 30 | 13 / 5 | 31 / 22 |

PAD vs MG: success 2 / 2 (net 0); all-steps-optimal 11 / 8 (net +3). T-MONO optimal 28 vs 23 and T-BREAK 30 vs 26 favour PAD; COLOR optimal 22 vs 29 favours MG: colour-twin all-steps-optimal PAD both 6 / RED-only 10 / BLUE-only 0 (MG 13 / 3 / 0; B_G1C3 14 / 1 / 0), i.e. the RED-over-BLUE asymmetry in optimality returns for PAD while success stays 15 / 1 / 0. PAD vs B_G1C3: success 2 / 3, optimal 10 / 13. dev36 (PAD raw argmax, no controller): 36 success, 36 decision-perfect.
PAD raw errors by class on this suite (93 vs MG 102): neutral preparation 50 (MG 38), wrong destruction 9 (18), missed direct progress 4 (17), refuse necessary destruction 12 (12), avoidable destruction 7 (3). Post-hoc description: `results/post_error_classes_new_runs.csv`, `post_colour_twins_fresh112.csv`.

## 5. Common offline state bank (240 states: init, ~1/3, ~2/3, pre-final, one fixed deviation per board problem; chosen without any model score)
Top-1 in the full optimal set: MG 229/240, PAD 230/240; mean optimal-action probability 0.949 vs 0.958; repaired 2, newly wrong 1. All-optimal-neutral preparation states (106): 100 vs 100 (repaired 1, newly wrong 1). n=8 states: 112 vs 111. h=4 states: 112 vs 113. Differences are one or two states.

## 6. Reading against plan section 6 and the recommendation
- The original M1-GOAL does NOT degrade when table blocks are added (board48 success 47/48 at n=6 and n=8, offline top-1 optimal 95% of bank states); S8-PAD changes nothing on the board and moves the 112-suite optimality by +3 with a -7 colour-twin optimality loss. Plan row "S8-PAD gives no improvement" applies: distractor-object scale coverage is not the bottleneck here (this does not exclude other scale factors such as more objects that must actually be moved).
- Residual errors concentrate in preparation choices (neutral-preparation wrong is the largest class: 38 of 102 for MG, 50 of 93 for PAD) and in optimality at tower height 4 (n8k2h4 all-steps-optimal 0/6 and 1/6), not in object count and not in missed direct progress.
- Recommended branch: **NEXT_SUBGOAL** (plan section 7: joint next-sub-goal / action prediction with the matched SG-BASE control), per the plan's default order. The pair-interaction branch stays an open alternative: wrong-destruction + refused-destruction + avoidable destruction are 33 of 102 MG raw errors, but they shrank under PAD training data (9+12+7 = 28 of 93) without any interaction module, so they are not yet evidence for one.
- No additional training was started.
