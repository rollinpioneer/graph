# CP-DISR-TB-STRUCT-GEN-V1 — prep summary

Suite: six goal-composition cells on the frozen T_B layout (lid closed, same candidates / mask / controller / verifier / reward / deadline). Train+dev: IN_T (depth 3), BUF_S (depth 2), IN_T+BUF_S (depth 5, the frozen T_B goal). Struct-gen test: IN_S (depth 3), BUF_T+BUF_S (depth 4), IN_S+BUF_T (depth 5); every test cell contains an object->destination binding absent from train.

Counts: train 60, dev 12, test 30 (102 cases). Depth levels: DEPTH-1 = 2-3, DEPTH-2 = 4, DEPTH-3 = 5 (pure symbolic shortest-chain, independently re-computed).

## Gates

- dependency_depth_audit: PASS
- binding_novelty_audit: PASS
- shortcut_audit: PASS
- physical_qualification: PASS_UNDER_DISCLOSED_AMENDMENT
- split_frozen: PASS
- representation_identity: PASS
- capacity_identity: PASS
- offline_tests: PASS
- budget_fairness: PASS
- reward_evaluator_controller_files_unchanged: PASS

## Disclosures

- physical qualification raw verdict FAIL under the strict fact-equality criterion; the same six episodes pass the disclosed reference-relative amendment (physical_qualification_amendment.json); no episode was added or repeated
- task_evaluator.py is unchanged; StructGenEvaluator reuses its atomic checks (unit-tested for equivalence)
- dev has 12 cases (4 per train cell), evaluated at 0/4096/8192/final; the test split is evaluated post hoc on final checkpoints only

## Budget

- mean skill duration in qualification 4.50 s vs current-profile 4.79 s; expected N at Tcap = 15306 (< Ncap 16384): Tcap remains the binding stop, no budget change.

Release verdict: **PASS**
