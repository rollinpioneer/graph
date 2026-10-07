# C1-BW-SCALE-ORDER-PROTOTYPE-V1 — claim boundary

## Supported (development evidence; one training seed; deterministic argmax)
- Step 0: with the plan's priority rules implemented in this repository, the raw-choice error count of M1-GOAL+C3 on the scorer-control suite is 102 and of B2+C3 418, as in the user audit; the class split matches the audit up to one grouping (our 38 neutral + 6 OTHER_AMBIGUOUS vs the audit's 44). The two suites' tables are reported separately.
- Adding table distractor blocks to the single-tower training trajectories (MG-S8-PAD, 50% original / 50% padded, identical budget) did not change success on the 48-problem factorial board (47/48 both, 1 / 1 discordant) and changed all-steps-optimal on the 112-suite by +3 (11 / 8 discordant) while reducing colour-twin optimality (PAD 6 both / 10 RED-only vs MG 13 / 3).
- The unpadded M1-GOAL already handles n=6 and n=8 problems with one or two goal towers on this board (success 47/48); its optimality losses follow goal-tower height (h=4 all-steps-optimal 9/24 vs h=2 21/24), not object count.
- On 240 offline states chosen without any model score, MG and PAD differ in whether the first choice is optimal at only 3 states (repaired 2, newly wrong 1).

## Not supported / must not be claimed
- "Scale / object count is not a factor at all": only table-distractor padding of the training problems was tested; the plan itself says it does not cover training in which more objects must actually be moved, taller towers, or other layouts. The negative result is for this augmentation.
- "Interaction models are (un)necessary": no interaction or sub-goal model was trained here. The error classes are descriptive; 'neutral preparation wrong' does not mean the model has no relational information.
- That the sub-goal branch will work: the recommendation follows the plan's default order and the residual-error profile; the plan itself notes the sub-goal label does not supervise the whole goal priority or the fallback decisions.
- Any ranking of MG, PAD and B2+G1+C3 as final systems: the 112-suite and the 48-problem board are development material (the 112-suite was already used for design in the previous card); each scorer is one checkpoint; deterministic episodes are not independent trials; the board has 6 problems per cell and several differences are 1–3 problems; no significance or equivalence is claimed and no threshold label is issued.
- Compute-matched comparison: MG-S8-PAD used the same steps and samples but larger graphs (57.9 vs 35.1 s per epoch under shared GPUs).
- The colour-optimality asymmetry of PAD as a mechanism: it is one suite (16 pairs); padding colours were hash-balanced but the cause is not isolated.

## Disclosures
- The Desktop plan's header says `execution_authorized: false`; execution was authorized by the user's chat message (recorded in prep/registration.json), which also allowed parallel multi-GPU execution (training GPU 6; baseline and new evaluations on other GPUs) and lab-xushijie3 disk space (not needed).
- The extra checkpoints (epochs 20/40/60/80) were saved as the plan allows but not evaluated; the reported number uses epoch 100 only. One training run, no checkpoint or seed selection.
- `scripts/c1_bw_scale_order_post.py` (class split of the new runs and PAD colour twins) was written after the results were seen: descriptive, not part of the registration.
- The audit's classification script was not available; our implementation is plan 3.3 and matches the audit's totals but is not digit-for-digit a reuse.
- MG_C3, B_G1C3 numbers on the 112-suite are reused from the previous card (same code, same problems, same weights); no number is recomputed twice. KR 2023 / GSP / AIW-AD / RPN were not run (plan tier 3, separately authorised).
