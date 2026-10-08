# C1-BW-METHOD-SERIAL-SUITE-V3 — final summary

Status: **C1_METHOD_SERIAL_SUITE_COMPLETE** (driver: 43/43 stages DONE; no technical gap, no mathematical alias, no skipped family). Base `2d008431e`, branch `codex/cp-disr-c1-bw-method-serial-v3`, registration commit `6c12b1f33` (pushed before any training), storage owner xushijie3.
Executed: 5 families, **9 trainings** (cap 9, one seed each, 100 epochs, 4,100 updates, 744,600 decision exposures, NaN 0, one attempt each), 2-step look-ahead (0 training), Board48 selection, one locked common confirmation set **Confirm128** (n in {6,8} x goal towers in {1,2} x goal height in {2,4}, 16 per cell, no shortfall, 852 earlier problem classes excluded, 0 overlap), **17 conditions x 128 = 2,176 episodes**, 128 exact-planner references (all optimal).
Training scope: n in {3,4,5}, goal-tower height in {2,3}, one non-trivial goal tower, no padding, no height-4 gradient data. Height 4 is therefore *goal-height extrapolation*; n=8, 2 towers and the colour structure are further shifts.
Sources of numbers: success / all-steps-optimal (n). "Observed best" below means best on this one confirmation set only.

## 1. Confirm128 (one checkpoint per condition; one training seed; deterministic episodes are not independent trials)

| condition | ALL (128) | h=2 seen (64) | h=4 unseen (64) | must-destroy (34) |
|---|---|---|---|---|
| MG_C3 (frozen reference) | 124 / 79 | 64 / 56 | 60 / 23 | 30 / 16 |
| B_G1C3 (frozen reference) | 115 / 83 | 64 / 64 | 51 / 19 | 28 / 15 |
| LOOK2_MG | 127 / 96 | 64 / 62 | 63 / 34 | 34 / 22 |
| LOOK2_COUNT | 71 / 43 | 52 / 39 | 19 / 4 | 9 / 2 |
| SG_BASE | 124 / 74 | 64 / 56 | 60 / 18 | 30 / 15 |
| SG_FACT | 90 / 60 | 64 / 58 | 26 / 2 | 17 / 8 |
| SG_JOINT | 104 / 64 | 64 / 56 | 40 / 8 | 24 / 10 |
| CAL_ABS_C3 | 126 / 84 | 64 / 57 | 62 / 27 | 32 / 19 |
| CAL_REL_C3 | 124 / 84 | 64 / 58 | 60 / 26 | 30 / 18 |
| CAL_ABS_LOOK2 | 128 / 90 | 64 / 59 | 64 / 31 | 34 / 22 |
| CAL_REL_LOOK2 | 128 / 96 | 64 / 62 | 64 / 34 | 34 / 23 |
| REC_SELF_T4 / T8 | 123 / 86 (both) | 64 / 55 | 59 / 31 | 31 / 20 |
| REC_REL_T4 / T8 | 123 / 87 (both) | 64 / 55 | 59 / 32 | 31 / 20 |
| GOAL_DENSE | 120 / 83 | 64 / 55 | 56 / 28 | 27 / 18 |
| GOAL_REL | 122 / 95 | 64 / 57 | 58 / 38 | 29 / 21 |

Observed best on this set: CAL_REL_LOOK2 (128 / 96) and LOOK2_MG (127 / 96); among one-step controllers GOAL_REL has the most all-steps-optimal episodes (95) and CAL_ABS_C3 the most successes (126). The system chosen from development information alone *before* the confirmation set existed was **CAL_ABS** (one-step + C3; Board48 key): 126 / 84 vs 124 / 79 for MG_C3 (success 2 vs 0 discordant, optimal 9 vs 4) — a small, not clearly separable gain: preselection **inconclusive**.

## 2. Family answers (fact / matched comparison / supported / not supported)

### Two-step look-ahead (0 training)
- Fact: LOOK2_MG 127 / 96 vs MG_C3 124 / 79 (success 4 vs 1 discordant; optimal 29 vs 12; h4 optimal 22 vs 11). LOOK2_COUNT collapses to 71 / 43. The Board48/fresh112 h4 panels and the shared-tree bank gave the ordering `RECURRENT_THEN_CALIBRATION` (neural leaf was *not* weaker than the goal-count leaf: h4 completions 24 vs 9 on Board48, 12 vs 5 on fresh112; cross-root optimal choice 0.979 vs 0.892).
- Matched: the same tree (same roots, contracts, de-duplication, tie-break) with learned value vs unmet-goal-count leaves. LOOK2_MG vs LOOK2_COUNT: +56 successes (56 vs 0 discordant), h4 +44.
- Supported: the learned leaf value is substantially more useful than the goal count, and two-step expansion with the learned value improves all-steps-optimality (esp. h4) over the one-step controller at the same weights. Cost: 28,588 leaf states expanded in 1,988 decisions (24,430 unique), 1,734 batched model calls; episode wall 79 s vs 47 s for MG_C3 (+70%); it also uses contract expansion that one-step controllers do not.
- Not supported: that the gain is a model-capability gain (extra search compute and contract access are part of it); anything about deeper look-ahead.

### Next-event organisation (3 trainings)
- Labels (900 unique goal-state pairs, 7,446 decisions, all complete, 0 unknown): event selection informative (Q != U0) in 63.4% of states; **pair-specific information (Y != Q x A) only 3.0% of states (27; 2.5% of decisions, 185)**; Y = Q x A in 97%; events are OnTable-only 48.1%, On-only 49.8%, mixed 2.1%. FACT and JOINT therefore differ in a small slice (not mathematically equal, so all three were trained).
- Fact: SG_BASE 124 / 74 (new-module gradient ~3e-4 -> 5e-7: the unsupervised head barely moved, as the runbook predicted). SG_FACT 90 / 60 and SG_JOINT 104 / 64: h4 success falls to 26/64 and 40/64 (BASE 60/64) while h2 stays 64/64. On the 240 development bank states the event heads predict much better (event top-1 hit BASE 87, FACT 206, JOINT 215; mean P_Y 0.35 / 0.83 / 0.88) but the action policy is worse at h4.
- Matched: JOINT vs FACT (same architecture, same budget): success +14 net (20 vs 6 discordant, all at h4), optimal +4; both far below BASE.
- Supported: event prediction is learnable from these labels and the supervision changes behaviour; in this configuration organising by next event *hurt* goal-height-4 execution.
- Not supported: that pairing information is what helps — only 2.5% of decisions carry it, the selected checkpoints differ (FACT epoch 80, JOINT epoch 20), and both lose to BASE; "events help preparation" is not supported by these results.

### Value calibration (2 trainings)
- Fact: constants from training states only: median positive margin 60.1 (2,009 pairs), a0 = 0.0333, c0 = 1.659, beta0 = 30.06; learned beta 30.55 (ABS) and 30.94 (REL), no clamp. One-step: CAL_ABS 126 / 84, CAL_REL 124 / 84 vs MG_C3 124 / 79. Under the same look-ahead: CAL_ABS 128 / 90, CAL_REL 128 / 96 vs LOOK2_MG 127 / 96.
- Matched: REL vs ABS one-step: success 0 vs 2, optimal 2 vs 2 (no difference); under look-ahead: success equal (128), optimal +6 for REL (7 vs 1 discordant; h4 +3). Look-ahead vs one-step within each model: ABS +2 / +6, REL +4 / +12 (success / optimal net).
- Supported: temperature and step scale stay stable and the initial policy was preserved; relative calibration is not worse than absolute and is somewhat better as a look-ahead leaf value (+6 optimal on 128 problems).
- Not supported: that calibration adds anything beyond the uncalibrated MG leaf (CAL_REL_LOOK2 = LOOK2_MG in optimal count, +1 success); that the value error improved (value error was not tabulated here); a general advantage of relative over absolute objectives.

### Recurrent relational reasoning (2 trainings)
- Fact: T4 = T8 for both SELF and REL on all 128 episodes (0 discordant); REL vs SELF 0 discordant in success, 1 vs 0 in optimal. Logits do change with the iteration count (mean max |dlogit| T4 vs T8 on 240 bank states: 0.67 SELF, 1.42 REL) but top choices do not change on Confirm128. REL_T4 vs MG_C3: success -1 net, optimal +8 net (9 vs 1; h4 +9) — the same for SELF, so it comes from the fine-tuned encoder/value heads rather than from neighbour information.
- Supported: no loss of simple-case ability (h2 64/64).
- Not supported: any benefit of relational neighbours (REL = SELF), any test-time-compute extrapolation (T8 = T4), or that recurrence cannot help: the action loss was ~0 on the training data (new-module gradient ~1e-5), so the module was barely supervised; no shared/unshared-parameter control exists.

### Goal-relation attention (2 trainings)
- Fact: GOAL_REL 122 / 95, GOAL_DENSE 120 / 83, MG_C3 124 / 79. Mask density (non-self goal pairs readable): DENSE 1.00; REL 0.60 on training initial states, 0.47 Board48, 0.56 fresh112, 0.48 Confirm128. On the 240-state bank GOAL_REL has 236 first choices optimal (MG 229), neutral-preparation states 104/106 (MG 100), 7 repaired and 0 newly wrong vs MG.
- Matched: REL vs DENSE (same module, same pair features, mask only): success +2 net (3 vs 1), optimal +12 (15 vs 3), h4 optimal +10 (12 vs 2). REL vs MG_C3: success -2 (1 vs 3), optimal +16 (18 vs 2), h4 optimal +15 (15 vs 0), h4 success -2.
- Supported: with this mask the relation-restricted attention is clearly better than dense attention in optimality, with no cost in search.
- Not supported: a relational mechanism: REL hides 44-53% of the non-self goal pairs in the new problems vs 40% in training, so the gain may be distribution-shift suppression of unseen cross-tower pairs; no third mask ablation was run, and the contract-threat term of the mask is empty in Blocksworld.

## 3. Cost and training accounting
Each training: 4,100 updates, 744,600 exposures, 49-68 min wall (SG 3,122-3,166 s, CAL 3,401/3,644 s, REC 3,599/4,051 s, GOAL 2,948/2,970 s; total about 8.3 GPU-hours); extra cost reported per family (endpoints, recurrence, pair tensors) in `compute_costs.csv`. Development: 9 conditions x 5 checkpoints on Board48, support panels (fresh112, dev36, 240-state bank) only for the selected checkpoint; selection could not be changed by the support panels.

## 4. Execution notes
Training and evaluation of independent tasks were run on several GPUs after the user's instruction of 2026-10-08 (GOAL_REL ran concurrently with GOAL_DENSE; 16 confirmation conditions ran concurrently on GPUs 1-7); earlier stages ran serially under the single driver. See `technical_gaps.md` for the exact deviations.
