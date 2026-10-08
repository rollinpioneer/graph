# C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3: method recommendation (one priority, plan section 15 Q8)

**Recommendation: the next card is a topology-coverage training card (proposal only; nothing has been started, generated or trained, and it needs the user's explicit authorization).** Not exact cost reduction, not an independent-confirmation-only card, and not another search-recipe variant.

## Why this one
- The remaining IPC failures are not a cost problem for four of the problems: with FAST, DENSE explores 28,913 / 29,127 / 21,787 / 13,559 expansions on p06 / p11 / p12 / p17 without a plan, while WL needs 593 / 1,556 / 978 / 588. FAST has passed WL's solving expansion count after 6-16 s. More speed does not change that.
- For p21 / p22 (WL needs 25,037 / 48,652 expansions, FAST reached 1,804 / 512) the gap is 14x / 95x. The exact reductions found here (1.34-2.2x per call, relational message passing still 487 of 492 ms at p22 B=48) cannot close it; only an exact shared / incremental computation could, which is a research item of its own and would help only those two problems.
- The 1:1 alternation did not turn the frozen DENSE score into a better search (IPC +2 / -1, both repairs solved faster by h_add alone, Joint quality line missed), so combining scorers is not the next lever.
- The same-family strength is retained (Struct / Joint ratio 1.02 / 1.09 against WL 1.17 / 1.22), which is the situation the plan's decision table assigns to topology coverage. That the cause is topology is a hypothesis; the card below is designed so that its result separates topology from the other factors.

## Proposed shape (plan section 12; not authorised, not started)
1. **Label-cost estimate first, no training.** Count reachable states and the cost of the exact labels for candidate topologies (1-3 vehicles, 2-6 places, matching hoists) with single-pile non-trivial goals of height 2-3 and few crates. The old BFS cost cannot be assumed; if exact labels explode, the supervision has to be redesigned and stated (plans that are not proved optimal are not exact distances, and the complete optimal-action set may not be available) with the same label authority for the control. Report this estimate and stop for a decision.
2. **New problems from the public generator only**, generated independently of the IPC trajectories; record which worlds / goals are seen. Separate *world topology seen vs unseen* from *goal structure seen vs unseen*; do not build one set where everything is larger and call it goal-structure transfer.
3. **Three trainings, shared initialisation seed and optimisation pipeline**: (a) DENSE on the original topology, matched to the new budget; (b) DENSE on the multi-topology data with the same goal restriction, same updates and sample budget; (c) one WL fit on the same new training problems. Multiple seeds only after a candidate is shown to work; architecture differences on IPC are not a seed-variance measurement.
4. **Evaluation**: the same shared engine, FAST for DENSE, 300 s / 100,000 expansions / 8 GiB, on new independent test problems (topology seen / unseen x goal seen / unseen), IPC22 only as the already-consumed reference. No ALT, no new scorer.
5. **Pre-register the readout before looking**: node-level (expansions at solution relative to WL on commonly solved problems, and the p06 / p11 / p12 / p17 family), coverage at 300 s, plan quality, with failures kept as truncated where the time limit hits before the node limit.
6. A negative result would show that this training intervention is insufficient; representation, supervision, optimisation, scale and time stay separate and no conclusion about the encoder follows.

## What would change the recommendation
- If the label-cost estimate (step 1) shows that no tractable exact supervision exists, the card should be re-scoped before any training.
- If a REF repeat on the IPC problems shows the trajectory divergences seen here are inside the reference's own noise (cheap, bounded: 22 problems x one run), nothing in this recommendation changes; if it showed that FAST changes search behaviour beyond the reference noise, a FAST correction would come first.
