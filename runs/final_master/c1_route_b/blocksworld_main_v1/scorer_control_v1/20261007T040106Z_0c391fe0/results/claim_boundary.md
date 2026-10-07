# C1-BW-SCORER-CONTROL-V1 — claim boundary

## Supported (this suite, these two frozen checkpoints, deterministic argmax)
- With the same visited-successor rule (C3) and the same fresh problems, the frozen M1-GOAL scorer succeeds on 108/112 and is optimal on 82/112, against 86/112 and 42/112 for the frozen A02 B2 scorer.
- Adding the strict-goal-progress preference (G1) on top of C3 improves the old scorer (109/112, 88/112 optimal) and does not improve the new scorer (102/112, 78/112).
- The best execution system in this card is still B2 + G1 + C3. M1-GOAL + C3 matches it on success (108 vs 109, no clear difference) but not on all-steps-optimal (82 vs 88).
- The result is consistent in direction with the previous card (B2 + G1 + C3 111/112 and 88/112 on Confirm112-v2; 109/112 and 88/112 here), but the two suites are not independent replications of one estimate.
- Zero new training, zero optimizer steps, zero dev reruns; no policy was queried to choose any problem.

## Not supported / must not be claimed
- "The learned scorer adds value on this task family": under G1+C3 it does not, and under C3 alone it only recovers what G1 already supplies to the old scorer.
- "M1-GOAL is the best system", "learning is useless", or "the rules need no learned scores": B2 + G1 + C3 still reads the learned B2 score for every choice (the rule combination is zero NEW training, not zero learning).
- Any single-component mechanism for the M1 bundle; any sample-efficiency, seed-variance, other-task or search-cost claim. Each scorer is one checkpoint from one training seed, and the 112 deterministic episodes per condition are not 112 independent trials.
- Equality of conditions with labels "no clear difference": the thresholds (5 of 32, ceil(5N/32) for 16 <= N < 32, no label below 16) are decision aids, not significance or equivalence bounds. The H4 and colour-pair counts (8 or 16 items) are too small for most labels.
- That G1 "hurts" the new scorer in general: the S difference is 6 problems of 112 (0 vs 6 one-sided), on one suite; the optimal-action difference vs MG_C3 is not labelled.
- B-vs-MG contrasts as pure scorer causes: the two scorers differ in architecture, training objective, rank loss and history, so the contrast identifies fixed scoring systems, not a single module.

## Disclosures
- The plan's fixture "original two conditions equal the original implementation" was met by using the identical callables (`a04p_controls.choose('C3')`, `gp_attribution.combo_choose`) plus an independent restatement of plan section 4.1 compared on 117 random decisions (ties and illegal-high logits included), a C3-before-G1 case, the empty-candidate case and an optimal-trajectory-preservation replay on train-split cases. No earlier suite and no dev case was rerun.
- B_C3 and B_G1C3 were run again on the fresh suite as paired references; no number from the earlier card is reused in any table here.
- Earlier training history (two approved migration restarts of the M1-GOAL / B2-RANK trainings) is unchanged by this card.
- The Desktop plan file has `execution_authorized: false` in its own header; execution was authorized by the user's chat message ("执行") naming that file, recorded in prep/registration.json.

## Exit under plan section 9
New combinations do not exceed B_G1C3 -> B_G1C3 stays the best execution system on this task family and the local read-out / loss / rule tuning loop on Blocksworld ends here. No new training, seeds or task selection follows automatically.
