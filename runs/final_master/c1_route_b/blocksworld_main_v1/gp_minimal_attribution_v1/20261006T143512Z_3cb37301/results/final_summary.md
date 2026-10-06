# C1-BW-GP-MINIMAL-ATTRIBUTION-V1 — final summary

State: **C1_BW_GP_MINIMAL_ATTRIBUTION_COMPLETE**. Two trainings only (B2-RANK, M1-GOAL; no third, no M1-NLL), 1,008 confirmation episodes (9 conditions x 112) + 72 dev36 episodes + 112 planner references, each started and completed once, 0 technical gaps (`verify.json`: PASS).
Both trainings were restarted once from epoch 4 / 8 on faster GPUs with the user's approval and an unchanged frozen config (the aborted partial outputs were kept outside git); each final run is a complete 100-epoch, 4,100-step, 744,600-sample training with no NaN.
Earlier states and labels are unchanged. Pre-training fixtures: B2-RANK with rank_weight 0 reproduces the old M0 step (max gradient difference 7e-9); M1-GOAL with marks off equals the old M1 and its head initial SHA matches. Old destruction labels: train 19/144, dev 5/36, 0 unknown. G1 on dev36: 30/36 success and 30/36 optimal, failing all 5 `D*=True` dev cases.

## Confirmation (success S / all-steps-optimal D; layers by "an optimal plan exists that never destroys a satisfied goal" = MONO, "every optimal plan destroys one" = BREAK)

| condition | dev36 S/D | T-MONO (32) S/D | T-BREAK (32) S/D | H4-MONO (8) | H4-BREAK (8) | COLOR-MONO (16) | COLOR-BREAK (16) | all 112 S/D |
|---|---|---|---|---|---|---|---|---|
| C0 A02 B2 | 36/35 | 1/1 | 3/2 | 0/0 | 1/1 | 16/16 | 16/16 | 37/36 |
| M0 old B2-continue | 36/36 | 1/1 | 1/0 | 0/0 | 0/0 | 16/16 | 16/16 | 34/33 |
| M1 old | 36/36 | 28/26 | 18/17 | 1/1 | 0/0 | 13/12 | 6/6 | 66/62 |
| **B2-RANK** (new) | 36/36 | 10/9 | 4/3 | 1/1 | 0/0 | 16/16 | 14/14 | 45/43 |
| **M1-GOAL** (new) | 36/36 | 27/26 | 19/19 | 1/1 | 0/0 | 16/16 | 9/9 | 72/71 |
| G1 | 30/30 | 27/27 | 0/0 | 1/1 | 0/0 | 12/12 | 0/0 | 40/40 |
| C3 | NA | 22/2 | 25/6 | 6/0 | 3/1 | 16/16 | 16/16 | 88/41 |
| **G1+C3** | NA | 32/28 | 32/29 | 8/1 | 7/2 | 16/12 | 16/16 | **111/88** |
| M1+C3 | NA | 32/27 | 30/19 | 7/1 | 7/2 | 15/12 | 15/10 | 106/71 |

Planner: 112/112 optimal plans. Layer sizes are fixed by registration (no shortfall); T-MONO starts with fewer goals already satisfied than T-BREAK (0.8 vs 2.1 on average) and optimal lengths average 14 vs 15, so the layers are not an orthogonal design.

## Matched comparisons (net = only-first-correct minus only-second-correct; direction by the registered threshold, N = 32 -> 5)

- **B2-RANK vs M0:** T-MONO S +9 / D +8 (`A_AHEAD`); T-BREAK +3 / +3 and H4 +1 (no clear difference); colour -2 (no clear difference). Adding the rank loss alone to the B2 path helps the no-destruction multi-tower problems (1 -> 10 of 32) but reaches far less than M1 (28) and does nothing for the must-destroy layer (4 of 32).
- **M1-GOAL vs M1:** T-MONO S -1 / D 0, T-BREAK S +1 / D +2, H4 0 (no clear difference). Restoring the encoder's goal marks **did not remove the multi-tower or the must-destroy gain** (goal-free encoding is not what carries it in this implementation). On the colour-twin layer M1-GOAL is clearly ahead of M1: S +6, D +7 (`A_AHEAD`).

## Colour twins (16 skeleton pairs, same names / initial state / goal supports, all colours flipped; net = RED-only minus BLUE-only, threshold 3 for 16 pairs)

- C0, M0 and B2-RANK: no RED/BLUE difference (16/16, 16/16 and 15/16 pairs both correct).
- **M1: RED-only 7, BLUE-only 0, both 6, neither 3 (success; optimal 8 / 0 / 5 / 3) -> clear colour sensitivity of the old M1.**
- **M1-GOAL: RED-only 1, BLUE-only 0, both 12, neither 3 -> no clear difference.** Its remaining failures are 3 pairs that fail in both colours, i.e. not repaired by swapping colours; this does not rule out colour acting together with structure.
- G1: both-colour failures dominate (10 pairs fail both); C3 and G1+C3: no difference.

## Against the simple rules

- **G1** (zero training) is as good as M1 / M1-GOAL on T-MONO (27) but solves 0 of 32 must-destroy problems; M1-GOAL gets 19 of them optimal (only-model 19, only-G1 0). So the learned models do things a G1 that only prefers goal-adding moves cannot.
- **G1+C3 is stronger than every learned condition in every layer** (111/112 success, 88/112 optimal; T-BREAK 32 and 29; mean 14-15 steps and 0 loops in the T layers). The learned conditions lose to it on T-MONO (26-27 vs 28), T-BREAK (17-19 vs 29), H4 and overall (71 vs 88 optimal). It is also not complete: it fails optimality on H4 (3 of 16), on 4 colour-MONO problems and on a few T cells. C3 alone gets success (88/112) with many detours (41 optimal). M1+C3 repairs success (106/112) but with detours (71 optimal) and keeps a RED-over-BLUE gap (optimal +4, `A_AHEAD`).

## Answers to the five questions

1. **New vs matched reference.** B2-RANK: clear gain on T-MONO (+9), none elsewhere. M1-GOAL: no change in multi-tower or must-destroy, clear improvement in colour twins versus M1.
2. **Problems that need a step back.** M1-GOAL solves 19 / 32 optimally, M1 17, B2-RANK 3, C0 2, G1 0 — more than B2 and G1, but G1+C3 solves 29 and C3 alone solves 25 (6 optimal).
3. **Versus C3 / G1+C3 / M1+C3.** The learned models do not add to the rule combinations on this suite: G1+C3 completes and optimises more, with fewer steps.
4. **Colour isolated?** For the old M1 the colour-flip effect is isolated at the pair level (7 vs 0); the other conditions show none (B2 family, rules) or a small one (M1-GOAL +1 pair). Heights: H4 is weak for every learned condition (0-1 of 8 per stratum) and for the rule combinations on optimality; since H4 shares n and k with T6-2 but not its goal edges or initial layout, tower height is not isolated from the other differences.
5. **Next working base.** See the recommendation.

## Recommendation (nothing beyond this card was run)

- **Learned base:** M1-GOAL (goal-conditioned encoder + the M1 scoring head) is the better learned candidate: it keeps the multi-tower and must-destroy gains of M1, removes M1's colour asymmetry, ID dev36 36/36. Goal-free encoding is not needed for the composition gain; rank loss alone on B2 is not enough.
- **Reference that has to stay in every future table:** G1+C3. On this suite it beats all learned models, so a learned-method claim needs a task slice where G1+C3 fails (here only H4 optimality and a few colour-MONO / T cells, all where the learned models are also weak), or a result that matches it without the visited-set bookkeeping.
- **M1-NLL is not triggered** (the rule requires B2-RANK to show no clear benefit; it did on T-MONO). It is listed as a candidate only and not started.
- Any new design based on these results needs a fresh confirmation set; this one is consumed.

Stop: next action is `WAIT_FOR_USER_MAINLINE_DECISION`.
