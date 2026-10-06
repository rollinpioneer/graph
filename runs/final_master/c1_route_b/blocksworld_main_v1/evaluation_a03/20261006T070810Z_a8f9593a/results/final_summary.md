# CP-DISR-C1-BW-FROZEN-EVAL-A03 — final summary

State: **C1_BW_A03_EVALUATION_COMPLETE_WITH_ID_GAP**. Evaluation only: 0 new training runs, 0 optimizer steps. The original states are unchanged and still hold: `PPO_ID_GATE_FAIL`, `IMITATION_ID_GATE_FAIL`
(ASNET-READOUT 32/36 < 33/36; 33 was not changed to 32 and ASNET was not singled out), original REP label `NOT_ISSUED`, `receptive_field_status = NOT_TESTABLE_STRUCTURALLY_EMPTY`.

## Protocol changes made by A03 (decided after the ID result was known)

- **Release rule changed:** the A02 rule "all three pass the ID gate before A0/A1/A2/B" was replaced by "the three A02 final checkpoints are evaluated once, whatever the ID gap". The ID gap is reported next to every test number.
- **A0 is diagnostic only**; it never blocked A1/A2/B (`a0_performance_stop_removed_for_all = true`). No replacement score gate exists.
- Models: the three A02 `final.pt` files only (SHA-256 and byte sizes verified before and after; sidecars match). Splits: frozen files, hashes verified. Frozen sources (models, environment, planner, metrics, trainer) are byte-identical to the base commit; the new A03 entry/tests/spec are registered separately.
- Registration commit `0fd4b2dc` / identity-check fix `64709c79` (the clean-tree check excludes only A03's own run root because its tracked ledger grows during the run); no episode had been evaluated at either commit (ledger empty).
- **Test exposure (kept as disclosed):** `PREMATERIALIZED_SUITE_WITH_DISCLOSED_SMOKE_EXPOSURE`; `registered_A02_checkpoint_test_exposure_before_A03 = false`; `protocol_selection_used_ID_results = true`. An earlier unregistered three-update smoke checkpoint was dry-run on the A0/A1/A2/B files before registration (PPO stop-point record, not independently re-checked). This is not a fully blind suite, and it is now consumed: any method designed from these results needs a fresh confirmatory suite.

Completeness: 408/408 learned episodes and 136/136 planner references completed once (ledger: each case started and completed exactly once), 0 technical-incomplete, 0 unavailable reference labels, 0 non-finite probabilities, 0 "decision-perfect but not successful" episodes (`results/verify.json`: PASS).

## Main table (success / decision-perfect; same cases for all methods)

| method | ID dev (36) | A0 (24, diagnostic) | A1 colour reversal (32) | A2 non-isomorphic (32) | B n=6 (16) | B n=7 (16) | B n=8 (16) |
|---|---|---|---|---|---|---|---|
| B2-CACHED | 36 / 35 | 24 / 24 | 32 / 32 | 8 / 8 | 2 / 2 | 0 / 0 | 0 / 0 |
| QMARK | 34 / 33 | 23 / 22 | 18 / 18 | 15 / 11 | 5 / 3 | 0 / 0 | 0 / 0 |
| ASNET-READOUT | 36 / 32 | 24 / 21 | 17 / 17 | 5 / 3 | 2 / 2 | 0 / 0 | 0 / 0 |

A1/A2 equal-weight decision-perfect mean (A0 excluded): B2 0.625, QMARK 0.453, ASNET 0.313. Exact planner: 136/136 optimal plans (plan length = optimal length everywhere).
All numbers are single-seed, deterministic-policy results on fixed suites; they are descriptive, not significance tests, and the test-minus-ID differences are not training-quality-corrected.

## What the results show

- **A0 (diagnostic):** each model's renamed-problem trajectory is identical to its saved dev-problem trajectory in 24/24 pairs (selected action sequences equal under a block relabelling; `results/a0_pairs.csv`). The A0 scores of QMARK (22/24) and ASNET (21/24) therefore reproduce their own ID behaviour; there is no sign of a renaming/binding fault.
- **A1 colour reversal:** B2 stays perfect (32/32). QMARK and ASNET fail the same 14–15 cases (paired: 17 both perfect, 1 only QMARK, 0 only ASNET); their first wrong decision comes early (mean step ≈ 2) and ends in a state-action loop. This is a condition-specific pattern for two of the three models; it is not shown to be caused by the architecture.
- **A2 non-isomorphic goals:** every model is weak (B2 8, QMARK 11, ASNET 3 of 32 decision-perfect). QMARK has the best raw count and B2 does not dominate it (paired: both 5, only B2 3, only QMARK 6). ASNET is lowest, but its ID gap (32/36) and the A2 gap are not separable here.
- **B scale:** at n = 7 and 8 no model solves a single case (0/16 each); at n = 6, 2–5 of 16. Failures are loop-terminated at the step cap (46/46 B2 failures, 43/43 QMARK, 46/46 ASNET contain a repeated state-action pair), first divergence around step 6–7. Avoidable destruction of a satisfied goal is rare (0–3 failed episodes per split and method), so failures are not mainly "undoing finished goals".
- **Search is easy here:** the exact planner solves all 136 problems optimally; mean expanded nodes 36–153 (A0/A1), 127–279 (A2), 945 / 2,724 / 7,656 at n = 6 / 7 / 8 (max 46,716; mean wall time 154 ms per problem at n = 8, cold cache per case). A learned policy has no search-cost argument on this task family. Measured wall time per decision (GPU-synchronised forward calls on a shared GPU): B2 ≈ 29 ms, QMARK/ASNET ≈ 17–18 ms; no multiplicative or sample-efficiency claim is made from these.

## Answers to the four decision questions

1. **B2 vs QMARK:** different patterns, no consistent winner. B2 is clearly better on colour reversal (+14 cases), QMARK is nominally better on A2 (+3) and n=6 (+1) and slightly worse ID (−2); both collapse at n ≥ 7. Not a repeatable single-direction advantage.
2. **ASNET:** it matches QMARK on A1 (same failing cases) and is lowest on A2/ID decision quality. Its ID detours (3 cycle episodes) continue as the lowest A2 score, but the evidence does not separate "ID detours carried forward" from a new condition-specific failure, and it cannot be attributed to architecture alone.
3. **Search:** simple and strong; learning claims should rest on transfer/representation, not search cost.
4. **Next minimal algorithmic change:** see the recommendation.

## Recommendation for the algorithm mainline (no training or implementation started)

**Observed dominant problem:** all three frozen policies fail by *getting trapped in a state-action loop after a first wrong decision in an unfamiliar configuration* (A2 and B), and two of three additionally fail early under colour reversal (A1). Representation differences between B2/QMARK/ASNET are small, mixed in sign, one-seed and confounded with ID gaps; they do not support continuing a B2-vs-QMARK-vs-ASNET ranking as the mainline.

**Recommended mainline:** keep B2-CACHED (the only model robust to colour reversal and best on the A1/A2 mean) as the base, and make the next algorithmic step **recovery from loops, not another representation**: a loop-aware decision rule (a visited-state-aware mask/penalty applied at decision time and in the aggregation rollouts), expected behaviour change: fewer step-cap loops and more successes in A2/B without changing the learned scoring on states it already solves. **Necessary control:** the identical A02 B2 final checkpoint without the loop-aware rule (same cases, same cap), so any gain is attributable to the rule and not to extra training or data; a second, separate question (training on n = 6 to test the scale gap) should not be mixed into the same experiment. Because this test suite is now consumed, a freshly generated confirmatory suite must be registered before the next confirmatory claim. Nothing in this recommendation has been implemented or trained.

Stop here: next action is `WAIT_FOR_USER_ALGORITHM_MAINLINE_DECISION`.
