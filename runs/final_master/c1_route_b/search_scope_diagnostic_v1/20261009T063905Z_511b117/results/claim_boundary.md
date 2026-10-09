# CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1: claim boundary

Scope of every statement: frozen DENSE-G (update 4100, sha256 `ffaad0f4...5cd38`) and the author WL-GOOSE models of the earlier cards; 6 IPC Depots problems (p06, p11, p12, p17 = group F, p14, p15 = group R) and 4 Train96 control problems; prefixes of at most 4,096 expansions / 60 s (IPC) or 128 expansions / 15 s (controls) in the shared eager GBFS; one round, one seed per learned scorer, a shared machine; bounds = LM-cut of the installed planner (lower), saved valid plans / bounded 5 s plans (upper) on IPC, exact distances on the controls; all problems are development material (no sealed test).

## What this card supports
1. **The ACTION loss has no explicit parent-child and no explicit cross-parent term**: the shared V(s) cancels in the softmax over one parent's actions and in every pairwise difference (code reading plus a formula fixture). Indirect constraints through shared parameters and shared successors remain and were not measured.
2. **The diagnostic can detect cross-group misorders when the labels are exact**: on four small Train96 problems the WL-driven prefixes contain certified cross-group witnesses (10 / 5 / 10 / 2, clearly inverted in 3 / 1 / 7 / 2 events), the DENSE-driven prefixes contain none in 10 / 6 / 8 / 11 resolved events. This validates the machinery (observer, groups, anchors, certificate direction) and shows nothing about IPC.
3. **On the four IPC problems where DENSE loses to WL, DENSE's frontier does not follow the saved reference plans**: the deepest reference state ever in OPEN at a snapshot is position 1 / 6 / 4 / 1 of 58 / 50 / 63 / 25 steps. (Those plans come from WL, the native WL planner and LAMA, not from DENSE.)
4. **With one reference plan per problem and LM-cut as the only lower bound, the cross-scope question cannot be answered on these problems**: 449 of 449 candidate pairs in the DENSE-driven F prefixes are unresolved; the best margin L(popped) - U(anchor) is -30 / -20 / -29 / -8.

## What is only a hypothesis
- That the IPC guidance failures of DENSE come from the missing cross-scope training signal: neither confirmed nor refuted here.
- That DENSE's local rankings are worse on the F problems than elsewhere: 14 of 85 orderable successor pairs inverted on F versus 0 of 42 on p14 / p15 and 2 of 246 on the controls is an exploratory observation outside the registered thresholds (decision-level errors: 1 / 1 / 1 / 2 of 16 parents, threshold 6); the pair sets are selected by the bounds (reference child versus successors with large lower bound), the counts are small, nothing was tested.
- That a differently anchored diagnostic (anchors from the scorer's own frontier or its own solution, tighter bounds) would find witnesses: untested.

## What stops
- Entering layer 2 (matched short fine-tuning) on the basis of this card: the label is not `CROSS_SCOPE_SIGNAL`. Plan 17.3: the current neural-training proposal is rejected for now; this is a measurement failure, not a refutation of the hypothesis.
- Reading `NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX` (the mechanical output of the registered quantity) as "no cross-scope problem": it was produced by counting events whose bounds exist, although all bounds overlapped; the corrected label is `INCONCLUSIVE_CERTIFICATES`.

## What cannot be concluded
- That all front-sorting or cross-group supervision methods are useless, or that the relational encoder is at fault.
- Anything about the causes of the IPC failures, topology effects, or the value of the WL exact-successor backup (not run).
- Any rate of misordering in the OPEN: certified witnesses are not an unbiased sample (the upper bounds cover only reference-path states and 28 bounded plans), and the unresolved share is 100% on F.
- Equivalence of the DENSE trajectories across GPU runs (the p14 prefix solved at 1,290 expansions, the earlier FAST run at 1,289), or any wall-clock statement (shared machine, load 11-16).
- Any statement about layer-2 or backup success; both are unstarted and unauthorised.
