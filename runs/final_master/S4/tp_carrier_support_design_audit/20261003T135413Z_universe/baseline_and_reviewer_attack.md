# Baseline and reviewer attack

Pre-registered future baselines: B_PLAN, B_PLAN+R, B2, Full, A_STAT, A_CAT, A_Q, simple-rule baseline, LLM-Planner, VLM-Planner (if conditions allow)

Comparison axes: J, learning samples, helpful/neutral/harmful strata, online API calls, inference latency, sensitivity to a wrong prior, generalisation to unseen conditions. The claim is a comparison, not that only Full can solve it.

## A. Can one hand-written rule solve it?

A fixed rule suffices: R1, R2, R3, R6 remain unrefuted.

## B. Can B_PLAN derive it from the contract?

No: the contract-only plan uses 5 skills in S1 against 1 for the best route, so the contract does not reveal the S1 winner; in S2 both orders tie.

## C. Can B2 memorise scene ids or action frequencies?

Yes for the winner label: a single carrier-first action wins or ties in every scenario, so B2 can learn 'carrier first when A is on B' from frequency alone.

## D. Is a static prior alone enough for A_STAT?

Yes: the relation never points to object-first as the strictly better action, so a static prior that always points to carrier-first is sufficient.

## E. Can an LLM/VLM planner answer it directly?

Likely yes: 'move the tray with the object on it' is a commonsense answer; it would need neutral, reversed and goal-dependent sub-conditions to be a diagnostic, and no reversed condition is established.

## F. If Full wins, can it be attributed to more parameters?

Not separable here: with one dominant action the comparison would measure parameter count rather than prior use.

## G. Does the task rely only on weak perception?

The task needs support and carrier geometry from public markers; weak perception is not the lever.

## H. Does it hold only in a self-built scene?

It is a self-built derived scene (LIBERO-derived assets, D0 host), so it cannot be reported as a LIBERO benchmark result.
