# C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3: claim boundary

Scope of every statement below: frozen DENSE-G update 4100, WL-GOOSE author model (typed / IPC fits of the previous card), h_add; Struct32, Joint128 and the 22 public-IPC Depots problems, which are all development / comparison material (no sealed test); one training seed per learned scorer; one search round per condition on a shared machine; 300 s / 100,000 expansions / 8 GiB per search; the shared eager GBFS engine of the previous card (plus the ALT variant with two alternated open lists, 1:1).

## What this card supports (development evidence)
1. **FAST is an implementation speed-up of the frozen DENSE-G value with unchanged judgements up to the reference's own GPU nondeterminism.** Tensors are bitwise identical; values are within the plan tolerance on all 6,230 compared values (REF repeat: 6,228) and within the reference's own repeat noise; IPC coverage is identical (the same 14 problems, same expansion counts on all 14). Per call 1.34-2.2x cheaper, 1.74x more scored states per second in the search, 1.5-2.1x more expansions in 300 s. This is an engineering result, not a new planning method.
2. **Speed alone does not explain the IPC failures of the DENSE scorer.** On p06 / p11 / p12 / p17 the DENSE search explores 13k-29k expansions without a plan while WL needs 588-1,556 expansions. These four are a node-level guidance disadvantage relative to WL in this engine. Conversely DENSE solves p14 and p15 where WL and h_add exhaust 100,000 nodes.
3. **On the training family (Struct / Joint) the DENSE scorer remains better than WL and h_add inside the same engine, also when combined with h_add**: ALT_DENSE_ADD is shorter than ALT_WL_ADD on Joint (78 shorter / 32 longer, 123 of 128 problems with fewer expansions) and has far smaller ratio than h_add.
4. **The 1:1 alternation does not improve the frozen DENSE search in a way that survives the pre-registered protection line**: Joint completion is kept (128/128) but the mean length ratio rises by 0.0578 (> 0.05) and the number of optimal plans falls 72 -> 40; IPC coverage changes by +2 / -1.

## What is only a hypothesis
- That the IPC guidance gap comes from limited **world-topology** coverage of the training problems (it could equally be supervision, optimisation, scale or the representation); this card tested none of these.
- That ALT has genuine problem-specific complementarity (p14: 83 expansions vs 1,289 for DENSE alone while h_add and WL fail) - one problem, no mechanism test.
- That p21 / p22 are guidance problems: they are still truncated (FAST reached 1,804 / 512 expansions, WL needs 25,037 / 48,652) and p19 / p20 have no WL reference.
- That FAST and REF trajectories coincide: 16 of 22 IPC rolling hashes differ (the reading is revoked for those rows); how much of this is the reference's own run-to-run nondeterminism was not measured in the IPC matrix (no REF repeat there).

## What must stop
- The 1:1 alternating recipe (ALT_DENSE_ADD, ALT_WL_ADD) as a candidate improvement; no automatic 3:1, 8:1, third scorer or learned scheduler.
- The claims "FAST improves coverage" (it does not here) and "IPC is only a speed problem" (not for the four problems above).
- Any claim that ALT added value from the neural score on IPC: both repairs (p11, p21) are solved faster by h_add alone.

## What cannot be concluded
- A new attention principle, a new search algorithm, public-domain state of the art, or that all IPC problems differ only by speed; that ALT proves the model has learned task-dependent behaviour; that one failed recipe shows a model class can never work.
- That the Joint protection line is a statistical non-inferiority result (it is an engineering trade-off line); it is not met.
- Generalisation to other domains or to unseen IPC problems: the 22 IPC problems and the Struct / Joint sets were consumed by earlier cards; the sign tests in `paired_quality.csv` are uncorrected, single-seed, on development problems.
- Wall-clock rankings between GPU rows (REF / FAST / ALT_DENSE, same-GPU adjacent timing blocks) and CPU rows (WL / h_add / ALT_WL, different tasks, 4 threads) are indicative; the shared machine had a 1-min load average of 2-11.
- Counts versus time: fewer expansions or fewer scored states are not a speed result (ALT_DENSE vs EAGER_HADD: 10.5x fewer expansions, 4.2x more wall time).
