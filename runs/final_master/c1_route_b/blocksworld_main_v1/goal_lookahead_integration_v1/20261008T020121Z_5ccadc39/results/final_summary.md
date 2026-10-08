# C1-BW-GOAL-LOOKAHEAD-INTEGRATION-V1 — final summary

Plan: `CP_DISR_C1_Five_Methods_Closeout_and_Integration_Plan_20261008.md` (sections 5 and 6). Base `225cbae0c`, branch `codex/cp-disr-c1-bw-goal-lookahead-integration-v1`, registration commit `f65fc99c4` pushed before any run.
Executed: the zero-training completion of the 3x2 matrix (DENSE_LOOK2, REL_LOOK2: 256 episodes) and the ONE budgeted training of section 6 (GOAL_RAND, sparsity-matched random-mask control; 4,100 updates, 744,600 exposures, NaN 0, one attempt, 3,752 s), plus its Board48 selection and support panels, and its one-step and look-ahead evaluations (256 episodes). Nothing else was trained, no depth other than 2, no extra module (no G1, calibration or sub-goal).
All episodes ran on the Confirm128 problems of the method-serial v3 card, which are **development material from this card on** (no independent confirmation claim). One training seed per learned scorer; deterministic episodes; success / all-steps-optimal (n).

## 1. The 3x2 matrix (Confirm128)

| scorer | one-step + C3 | fixed two-step + C3 |
|---|---|---|
| MG (frozen reference) | 124 / 79 | 127 / 96 |
| GOAL_DENSE | 120 / 83 | **128 / 95** (DENSE_LOOK2) |
| GOAL_REL | 122 / 95 | **128 / 97** (REL_LOOK2) |
| GOAL_RAND (sparsity-matched control) | 124 / 81 | 127 / 99 |

Layers (success / optimal): REL_LOOK2 h2 64/63, h4 64/34, must-destroy 34/23; DENSE_LOOK2 h2 64/62, h4 64/33, must-destroy 34/23; LOOK2_MG h2 64/62, h4 63/34, must-destroy 34/22; RAND_LOOK2 h2 64/62, h4 63/37, must-destroy 34/24.

## 2. Reading the plan's section 5.3 mapping
- **REL_LOOK2 vs LOOK2_MG:** success 128 vs 127 (1 vs 0 discordant), optimal 97 vs 96 (3 vs 2), h4 optimal 34 vs 34 (2 vs 2). It keeps the completion ability of the learned-value look-ahead; the hard-layer plan quality is not measurably improved. No complementary gain.
- **REL_LOOK2 vs DENSE_LOOK2:** success identical (128), optimal 97 vs 95 (3 vs 1 discordant): close, so the combination's performance cannot be attributed to the relation mask.
- **REL_LOOK2 vs GOAL_REL (same scorer, look-ahead added):** +6 successes (6 vs 0, all at h4), optimal +2 (18 vs 16 discordant); at h4 the optimal count falls 38 -> 34 (11 vs 15). Look-ahead adds completions to GOAL_REL, not optimality. The one-step optimality edge of GOAL_REL over MG (+16 net) disappears under look-ahead: all four scorers land at 95-99 optimal under the same two-step tree.
- Outcome class: **"combination differences are small"** — no depth, fallback or tie-break tuning follows; the working execution baseline stays "learned state value + fixed two-step contract look-ahead"; relation-restricted attention remains the best one-step candidate.
- Cost of the two-step layer (all four scorers): about 28,100-28,400 leaf states in ~2,000 decisions (24,000-24,300 unique), ~1,700-1,740 batched model calls, 34 s model time; episode wall 83-84 s vs 47-50 s one-step (about +70%).

## 3. Sparsity-matched random-mask control (section 6)
Definition (registered before training): same module, parameters, pair features, bias and initialisation as GOAL_DENSE / GOAL_REL, same 100-epoch budget, same 5 checkpoints and Board48 selection key; per query row keep the self entry plus k randomly ranked other goals, k = number of non-self targets of the relation mask in that row; the ranking is a fixed integer hash of (scored-state proposition truth vector, row, column): a pure function of the state, identical on every call, CPU = GPU (tested), not invariant to block renaming, no relation / label / history. Mask density matches by construction: 0.60 (training initial states), 0.47 Board48, 0.56 fresh112, 0.48 Confirm128.
- Selected checkpoint: epoch 20 (Board48 48 success / 30 optimal / 11 h4-optimal; the five checkpoints differ by 1-2 problems). Offline bank: 229/240 first choices optimal, 1 repaired / 1 newly wrong vs MG.
- **One-step:** RAND_C3 124 / 81 vs GOAL_REL 122 / 95: optimal -14 net (5 vs 19 discordant), h4 optimal 24 vs 38 (2 vs 16), must-destroy 16 vs 21; success +2 (3 vs 1). RAND_C3 vs MG_C3: success identical, optimal 81 vs 79 (4 vs 2) — the control behaves like MG. RAND_C3 vs GOAL_DENSE: optimal 81 vs 83, success 124 vs 120.
- **Two-step:** RAND_LOOK2 127 / 99 vs REL_LOOK2 128 / 97 (success 0 vs 1, optimal 4 vs 2): no difference; so under look-ahead the scorer's mask semantics did not matter.
- Per section 6: for the one-step scorer, "REL better than the sparsity-matched control" -> the choice of *which* goals interact is not reproduced by generic sparsification with the same row budget. This does not prove that the true optimal dependencies were recovered.

## 4. Verification
See `verify.json`: all 7 stages DONE, every episode ledgered once, 0 technical gaps, Confirm128 split hash unchanged, v3 selected checkpoints unchanged, old trees unchanged.
