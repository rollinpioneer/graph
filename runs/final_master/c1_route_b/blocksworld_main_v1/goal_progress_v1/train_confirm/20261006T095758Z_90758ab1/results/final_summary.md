# C1-BW-GOAL-PROGRESS-V1 — phase-2 summary

State: **C1_BW_GOAL_PROGRESS_COMPLETE**. Three trainings (M0, M1, M2; cap 3 respected, no retry), 672/672 Confirm112 episodes + 108 dev36 episodes + 112 planner references, each started and completed once, 0 technical gaps
(`verify.json`: PASS). No score gate was used. Earlier states are unchanged (`PPO_ID_GATE_FAIL`, `IMITATION_ID_GATE_FAIL`, A03 state, REP label `NOT_ISSUED`).
Training: 100 epochs each from the A02 B2 final checkpoint, 4,100 optimizer steps, 744,600 decision samples shown, no NaN, identical D_train (1,296 trajectories / 7,446 decisions = A02 D2 + C0/C3 train rollouts), identical rank labels;
M1/M2 have identical encoder starting weights, identical 82,688-parameter heads (equal initial tensors) and 513,472 trainable parameters each (M0: 701,889). Only the final (epoch 100) checkpoints were evaluated.

## Confirm112 (success / decision-perfect, 8 problems per slice; independent of Pilot80; counts are integers out of 8)

| slice | M0 B2-continue | M1 global | M2 additive | C0 B2 | C3 visited-succ. | G1 goal-progress rule |
|---|---|---|---|---|---|---|
| A1-4 (colour reversed) | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| A1-5 (colour reversed, h=3) | 8 / 8 | **3 / 2** | **3 / 3** | 8 / 8 | 8 / 8 | 4 / 4 |
| C5-1 single tower | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| **C5-2** (2,2,1) | 1 / 1 | 8 / 8 | 8 / 8 | 1 / 1 | 8 / 3 | 8 / 8 |
| **C5-23** (3,2) | 1 / 1 | 7 / 7 | 7 / 6 | 1 / 1 | 8 / 3 | 6 / 6 |
| **S6-2** | 3 / 3 | 7 / 5 | 7 / 5 | 2 / 2 | 8 / 3 | 7 / 7 |
| **S7-2** | 0 / 0 | 7 / 7 | 8 / 7 | 0 / 0 | 6 / 1 | 7 / 7 |
| **S8-2** | 0 / 0 | 7 / 7 | 7 / 7 | 0 / 0 | 7 / 0 | 6 / 6 |
| K6-1 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 7 / 7 |
| **K6-3** (three towers) | 0 / 0 | 8 / 8 | 8 / 8 | 0 / 0 | 8 / 0 | 8 / 8 |
| H6-13 | 8 / 7 | 8 / 8 | 8 / 8 | 8 / 8 | 8 / 8 | 5 / 5 |
| H6-14 (h=4) | 3 / 2 | 6 / 4 | 4 / 4 | 3 / 2 | 8 / 2 | 4 / 4 |
| H6-23 (3,3) | 0 / 0 | 4 / 4 | 6 / 6 | 0 / 0 | 8 / 1 | 2 / 2 |
| H6-24 (4,2) | 0 / 0 | 3 / 2 | 2 / 2 | 0 / 0 | 4 / 0 | 1 / 1 |
| **D_core** (decision-perfect, 6 core slices, equal weight) | 0.104 | **0.875** | **0.854** | 0.083 | 0.208 | **0.875** |
| **S_core** (success) | 0.104 | 0.917 | 0.938 | 0.083 | 0.938 | 0.875 |
| ID dev36 success / perfect | 36 / 36 | 36 / 36 | 36 / 36 | (A02: 36 / 35) | - | - |

Exact planner: 112/112 optimal plans, maximum 4,403 expanded nodes. Every M1/M2 failure (20 each) ended at the step cap inside a repeated state-action loop.

## Registered direction labels (computed after all six conditions were complete)

`NO_CLEAR_ADDITIVE_ADVANTAGE`, `G1_CLOSE_OR_STRONGER_THAN_M2`, `COMPOSITION_REVERSAL_TRADEOFF`. Not issued: `DECOMPOSITION_PROMISING` (M2 − M1 on D_core = −0.021, needs ≥ +0.10), `NEW_MODELS_NOT_SUPPORTED`.

## Reading

1. **Multi-tower and larger-n failure was removed for M1 and M2, not for M0.** On the six core slices B2 and its continued training (M0) stay at D_core 0.08–0.10; M1/M2 reach 0.85–0.88 with no n=6–8 training. Extra B2 training on the same data (M0) did not help, so training budget is not the explanation of the gain.
2. **Aggregation order did not matter.** M1 (global) vs M2 (additive): D_core 0.875 vs 0.854; on the 48 core problems both perfect 40, only M1 2, only M2 1. The additive constraint is not shown to be what helps; the evidence cannot say it is harmful or useless either.
3. **What changed from M0 to M1/M2 is a bundle** (goal-free state encoding, score = value difference over nominal successors, exact pairwise rank loss, no GRU/time input). This run has no ablation separating the parts (a fourth training was not authorised), so none of them can be credited alone.
4. **Simple rules are as strong on the core slices.** The training-free G1 rule matches M1 on D_core (0.875; the rule is not free: in 27 of its 31 failures B2's own top choice was already optimal and the rule replaced it) and C3 matches M2 on S_core (0.938, but D_core only 0.208: detours).
5. **Colour-reversal trade-off.** A1-5 (colour-reversed, tower height 3): M1 3/8 and M2 3/8 successes against 8/8 for B2, M0 and C3 (G1 4/8); A1-4 stays 8/8. Success on A1 in total: C0 16, M2 11 (the registered tolerance was one episode). Failures here are loops with an early first error (step 2–7).
6. **Height boundary.** M1 and M2 handle height-≤2 multi-tower problems up to n = 8; towers of height 3–4 are mixed (H6-23: M1 4/8, M2 6/8; H6-14: 6/8 vs 4/8; H6-24: 3/8, 2/8). Not a significance test; the cells are 8 problems each.
7. First-error type (own trajectories): the dominant B2 error "put down instead of goal-stack" is rare for M1/M2 (4 and 6 of 26 and 24 first errors) against 60 of 66 for M0.

## Answer for the algorithm mainline (nothing beyond this runbook was run)

The audited hypothesis "scoring each goal separately lets single-tower training transfer to multi-tower goals" is **partly supported only as "goal-free per-goal progress scoring transfers"**: both aggregation orders work and are indistinguishable, so additivity itself is not the contribution. The honest contribution claim is the bundle (goal-free encoding + successor value-difference + exact rank supervision), with the explicit open question which component carries it, the A1-5 colour-reversal regression it introduces, and the strong training-free G1/C3 baselines. A single next control would be an ablation of that bundle (e.g. the B2 policy trained with the rank loss and with the goal-free encoder switched on/off) on a freshly generated suite; it is not started.
