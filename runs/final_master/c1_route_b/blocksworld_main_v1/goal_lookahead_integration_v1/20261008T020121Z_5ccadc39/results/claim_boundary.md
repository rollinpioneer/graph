# C1-BW-GOAL-LOOKAHEAD-INTEGRATION-V1 — claim boundary

## Supported (development material; one training seed per learned scorer; fixed checkpoints; deterministic episodes)
1. Adding the fixed two-step contract look-ahead to any of the four goal-attention variants / MG scorers brings completions to 127-128 of 128 and all-steps-optimal counts to 95-99 of 128 (one-step: 79-95). The tree, tie-break and fall-back are identical across scorers.
2. The one-step optimality advantage of the relation-restricted attention (GOAL_REL 95 vs MG 79 vs dense 83) is **not additive** with look-ahead: REL_LOOK2 97 vs LOOK2_MG 96 vs DENSE_LOOK2 95.
3. A sparsity-matched random mask trained under the same recipe does not reproduce GOAL_REL's one-step gain (RAND_C3 81 optimal, MG-like), so for the one-step scorer the effect is not explained by "fewer pairs read" alone in this experiment.

## Not supported / must not be claimed
- That REL_LOOK2 is a better system than LOOK2_MG or DENSE_LOOK2 (97 / 96 / 95 optimal, 0-1 success discordant): differences are 1-3 problems. That the combination "composes" or is complementary (it is not, here).
- That the relation mask learned conflict / contract-threat / dependency reasoning: the control only rules out one generic-sparsity explanation. The control is not relation-structure matched (the relation mask is symmetric and follows object bindings; the random mask is neither), its training barely moved the policy away from MG (new-module gradients ~1e-5, selected epoch 20), and a random mask may simply make the attention unusable rather than "uninformative". The contract-threat term is empty in Blocksworld.
- That look-ahead makes any scorer better *as a model*: it adds search compute (about +70% episode time) and contract access; the learned values are used through it.
- Any statement about other seeds, depths other than 2, other tie-breaks, other domains, external methods, or a general ranking of REL vs DENSE vs RAND under look-ahead (RAND_LOOK2 had the highest optimal count, 99, by 2-4 problems).
- Confirmation: Confirm128 is development material here. The results do not independently confirm the planned combination; a fresh problem set would be needed for a confirmation claim.

## Disclosures
- The plan (section 6) defers the control's executable details to a later method card; they were fixed here before training (see `plan/registration.json`) and are the only definition used. The control was additionally evaluated with the two-step layer (RAND_LOOK2), which the plan did not request, as the execution-matched partner of REL_LOOK2.
- Authorisation: the user's message "执行" naming the plan file; recorded in `plan/registration.json`. Training and evaluations ran in parallel on several GPUs (user instruction of 2026-10-08).
- `model.py` / `common.py` were extended (a third attention mode `rand`, condition `GOAL_RAND`); the dense / rel modes, all v3 weights and all earlier results are unchanged (model-equivalence fixtures pass).
- MG_C3, LOOK2_MG, GOAL_DENSE, GOAL_REL and B_G1C3 rows are reused from the method-serial v3 card (same code, problems, weights).
