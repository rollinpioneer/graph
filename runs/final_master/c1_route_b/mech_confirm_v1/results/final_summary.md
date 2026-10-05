# CP-DISR-C1-MECH-CONFIRM-V1 - final summary

Base commit `5a2b3d18d21cbab19b9d09c3c430b3203e873730`; prep commit `4848ace4e9e1fa5e1b8a8807bad4e8936e3915ea` (QMARK frozen in `5e7c4ff00`). State: `C1_MECHANISM_CONFIRMATION_COMPLETE`, `NEXT_ACTION = WAIT_FOR_USER_MAINLINE_DECISION`.

## Headline (frozen rules, one QMARK seed, 32 fresh cases per model)

| model | total | IN_S | BUF_T | BUF_T+BUF_S | IN_S+BUF_T |
|---|---|---|---|---|---|
| B2 (seed 0) | 27/32 = 0.844 | 8/8 | 7/8 | 4/8 | 8/8 |
| ABS (seed 0) | 23/32 = 0.719 | 8/8 | 7/8 | **0/8** | 8/8 |
| +E (seed 0) | 0/32 = 0.000 | 0/8 | 0/8 | 0/8 | 0/8 |
| NC (seed 0) | 8/32 = 0.250 | 8/8 | 0/8 | 0/8 | 0/8 |
| **QMARK (seed 0)** | **27/32 = 0.844** | 8/8 | 7/8 | 4/8 | 8/8 |
| B_PLAN (reference) | 26/32 = 0.813 | 8/8 | 7/8 | 3/8 | 8/8 |

* **Main class: M3 (RELATIONAL_QUERY_IS_SUFFICIENT).** T(QMARK) - T(B2) = 0.000, T(QMARK) - T(ABS) = +0.125, QMARK trails B2 by more than 0.20 in 0 of 4 cells. M1 and M2 need T(B2) - T(QMARK) >= 0.20, so they are excluded.
* **B2 vs ABS: UNRESOLVED.** T(B2) - T(ABS) = 0.125 < 0.15 (DIFFERENCE_STABILITY_SIGNAL needs 0.15), and ABS trails B2 by more than 0.20 in 1 cell (the BUF_T+BUF_S cell, 0/8 vs 4/8).
* **Route B recommendation: NARROW_AND_CONTINUE (Decision B).** QMARK is ID-qualified (final dev12 12/12, 0 NaN, no hard failure, stop = Tcap, N = 14532).
* No conditional seed was triggered by the frozen M4 rule. QMARK seeds 1 and 2 were pre-trained at the user's request (see the disclosure below) and are **trained-and-unevaluated: they are not results**.
* `verify.json` = PASS (192 formal episodes, each model x case once; all hashes equal the recorded ones).

## What can and cannot be said

M3 licenses: *explicitly generating the successor state is one effective implementation, but it is not necessary for this structural generalization; a strong candidate-contract-state relational computation (QMARK: same relational graph encoder, one marker, no successor) reached the same total as B2.* It does not license: that QMARK is a method, that difference-based or absolute representations are equivalent, or any claim about other task families.

Limits that bound every statement above: one QMARK seed and one seed per old model; 8 cases per cell; the policies are deterministic argmax so the evidence unit is seed x cell, not 32 independent episodes (QMARK and B2 have the same total but solve different BUF_T+BUF_S cases: B2 {02,14,26,30}, QMARK {06,18,22,30}); 4/8 vs 4/8 vs 0/8 in one cell carries most of the discrimination.

## Answers to the runbook's 10 questions

1. **Were the two test bindings seen in old training?** Never as goals (goal exposure = 0 for second_object->container and target->buffer). But they were executed as actions of other goals: summed over the 3 seeds per method, target->buffer 2,185-4,308 executions and second_object->container 300-679, with the public effect observed TRUE after 99% of normally terminated executions that have a stored successor snapshot (`prep/training_binding_exposure.json`). So "unseen" means unseen goal binding / goal composition, not unseen execution. The old test30 per-action mechanism is NOT_RECOVERABLE (episode outcomes only; it was not re-run).
2. **+E on Fresh Confirm.** 0/32. Every episode diverges from the public-optimal first action: at decision 0 in 16, decision 1 in 8, decision 2 in 8, always by a PICK that is not optimal; 31/32 episodes repeat an action, 8/32 destroy a completed goal, 8 end in NO_CANDIDATE_SAFE_TERMINATION after an earlier divergence, 24 run out of time. Among its selected placements 192 are second_object->buffer (the training-goal placement) and 7 target->container; the metric 'train-default-binding error at a non-optimal decision' is 0 because the divergence label was already set by the earlier wrong PICK. +E therefore keeps executing its training goal pattern; it does not compose the new goal.
3. **ABS double-AtBuffer failure.** All 8 ABS failures in BUF_T+BUF_S are first divergences before any goal is completed, and all 8 are the same action: an unnecessary `OPEN container` first (the lid is irrelevant for buffer goals). The time budget is then exhausted (INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE). It is neither a destroyed completed goal nor a failure after the first goal. ABS is otherwise equal to B2 (BUF_T 7/8, IN_S 8/8, IN_S+BUF_T 8/8).
4. **Does QMARK learn dev12?** Yes: seed 0 dev12 4 -> 12 -> 12 -> 12 of 12 (0 / 4096 / 8192 / final). Seeds 1 and 2 (pre-trained, unevaluated): 0 -> 12 -> 12 -> 11 and 3 -> 12 -> 12 -> 12; seed 1 would not have been ID-qualified at final (11/12).
5. **Does QMARK catch B2/ABS?** It equals B2 (27/32; cell-wise 8/7/4/8) and exceeds ABS by 4/32; the whole difference sits in BUF_T+BUF_S (4/8 vs 0/8), the other three cells are equal.
6. **Which mechanism is best supported?** Strong relational query itself (M3). Not supported by this run: B2-specific before/after difference (M2), shared explicit-application advantage (M1). Absolute-successor vs difference stays UNRESOLVED.
7. **B_PLAN.** 26/32 (0.813), 6 failures all NO_CANDIDATE_SAFE_TERMINATION (public observation loss: held object observed UNKNOWN so the hard mask leaves no legal action; 2 of the 6 had NO_PLAN statuses). Cost: 114 planning calls (82 replans after the first decision), mean 2.5 / max 6 expanded nodes, mean CPU 0.96 ms, max 2.5 ms per decision, no search timeout, 0 plan changes between decisions. In the sense of runbook 10.3 this is close to 'high success at very low search cost' (frozen threshold 0.9 not reached, so the automatic class is MIXED): contract search matches the best learned totals on this suite at about a millisecond per decision. The paper therefore cannot say contract search fails here; route B has to rest on amortised decisions, learned representation, or conditions beyond this suite.
8. **Does the old test30 conclusion survive?** Partly. The direction of the old result is reproduced for +E (0.000 -> 0.000), NC (0.222 -> 0.250), ABS (0.733 -> 0.719) and B2 (0.922 -> 0.844). What does not survive as an interpretation is that B2's advantage identifies explicit successor application: the new strong control equals B2.
9. **Route B.** NARROW_AND_CONTINUE: the contribution becomes 'candidate-contract-state relational computation supports structural generalization; explicit successor construction is one effective implementation, not a necessary one'. Whether to enter a second task family is the user's decision.
10. **External method?** Not required to settle this card. If one is installed next, the most dangerous neighbour is a learned policy that also uses relational structure of the same action schemas (ASNet-style), because B2, ABS and QMARK all use that kind of computation and B_PLAN shows the symbolic reference is cheap. This is a suggestion only; nothing was installed or reproduced.

## What the suite and qualification are (frozen disclosures)

* Fresh Confirm v1: 8 qualification + 32 final cases; cells IN_S, BUF_T, BUF_T+BUF_S, IN_S+BUF_T (BUF_T is the one new goal set); only goal set, object-destination binding and pose seed differ from frozen T_B; no pose or seed equals any old train60/dev12/test30/qualification case; construction attempt 1 (attempt 0 failed a design audit on the qualification subset, a geometry-balance check, before any model saw the suite).
* Physical qualification: **raw FAIL preserved (7/8 cases)**; status is only `PASS_UNDER_DISCLOSED_AMENDMENT`. `C1_qual_02` (BUF_T+BUF_S) ended with no legal action after a normal PICK (the held object was observed UNKNOWN, so the hard mask left no candidate): a public-observation loss, not a controller failure; every cell has a passing case, so the per-cell stop rule did not trigger. We do not claim that the nominal successor equals the real observed state. Observation-loss endings do occur in the final set (NO_CANDIDATE_SAFE_TERMINATION: B_PLAN 6, B2 2, QMARK 1, ABS 1, +E 8, NC 8 episodes; B_PLAN fails the BUF_T+BUF_S cell mainly this way).
* QMARK is a control, not a method: +128 parameters (0.015% of B2), 3.2 vs 5.4 mean encoder forwards per decision measured on Fresh Confirm (offline 5 vs 9 for 4 legal candidates), same 4 relational layers, not strictly equal-compute; its logging path still calls the successor routine for a log field (policy forward does not; traced in Q01).
* Analysis fields (public-optimal first actions, completed goals, divergence) are post-hoc and never policy inputs.

## Concurrency / deviation disclosure

QMARK seeds 1 and 2 were trained concurrently with seed 0 on the user's explicit request (runbook 6.4 normally releases them one at a time after an M4 result). The deviation is recorded in `conditional_release.json` in the registration directory. They were not evaluated on Fresh Confirm because the frozen M4 rule was not triggered; total new training = 3 (cap 3). Wall-clock is not sample efficiency.

## Not done (by design)

No second task family, no further seeds, no new main method, no external reproduction, no VLM / prior / C2 / C3, no paper-line rewrite.
