# CP-DISR-TB-EVAL-PREP-LITE-01 - T_B unified evaluation preparation (mainline first)

Baseline `241159c1c`, branch `codex/cp-disr-tb-eval-prep`. Preparation only: no formal holdout, no test episode, no test score, no RL / provider / optimizer / training.
`GLOBAL_S4 = NOT_COMPLETE`, `GLOBAL_S5 = NOT_COMPLETE`; this is T_B-specific evaluation preparation, not the project-wide S6.

## Frozen priority (disclosed revision)

CORE (MUST): B1-K seed1 (R-TB-K-1), B2 seed1 (R-TB-DK-1), B1-K+E seed0 (R-TB-E-0), B1-K+E seed1 (R-TB-E-1): 4 x 30 = **120** episodes.
HISTORICAL_EXTENSION: B1-K s0 and B2 s0, up to +60, decided here by compatibility, not by test results. SYSTEM_REFERENCE: B0 s0, up to +30, only because it needs no new engineering. Maximum 210; never reached by retraining, conversion or a complex adapter.
Original Plan v3 was 6 x 30 base + conditional E1 x 30. The CORE-120-first ordering is a **local priority revision made in this card**, not what Plan v3 said.
Main-evaluation model = each run's last valid post-update final checkpoint; selected / best-dev only in a separate column. Test label: `PREMATERIALIZED_INDEPENDENT_HOLDOUT` (not a strict blind test).

## Phase A - historical model compatibility (0 environment, 0 episode, 0 optimizer)

All seven final checkpoints re-hashed equal to the frozen list and their sidecars. The four CORE models load strictly under the current code (class A). The three historical models fail strict load under the current Policy (14 `effect_readout.*` keys absent) and were **not** converted; they load strictly under the unmodified archived source whose hashes equal their run manifests (class B). B1-K s0, B2 s0 (0eb3776d) and B0 s0 (228517ed) share identical execution-file hashes, so there is one archived execution class and no third class. No model fell into class C.

## Phase B - recorder `eval_row_v2` (offline)

New evaluation-only module `src/cp_disr/final_tb_eval_record.py`; stage2a_v11 / collector / neural / torch_rl / runtime_factory / controller / Verifier / Evaluator / reward / deadline / termination / checkpoint files are byte-identical to the baseline. 20/20 offline tests pass, including an old-field comparison against the real `stage2a_v11.eval_episodes`, and all 5 injected recorder defects are caught.

## Phase C - consistency on frozen dev case T_B_dev_00 (3 episodes, cap 3)

| class | model | old fields (success, G, steps, reason, success_seconds) | internal consistency |
|---|---|---|---|
| current 2.1.1 | R-TB-K-1 (fails: 11 actions, deadline-capped) | identical | pass |
| archived v13 | R1 B2 seed0 (succeeds: 5 actions) | identical (G = 0.5015025717815832) | pass |

On the success case the old `success_seconds` is 2.90 s (final skill only) while the episode took 23.00 s simulated to the first independently confirmed success, the quantity the old record could not give.
Budget disclosure: the first archived attempt ran its episode but its result file was lost (relative output path after `chdir`); nothing from it was used, it counts against the cap, and the re-run was the third and last episode. The worker now resolves paths first and probes the output directory before building an environment. The recorder itself does not pre-check its output path; adding that would change its hash and need re-testing, so it is a release-card hardening item.
Caveats: the current-class episode is a failing case (no real-environment success-path timing on the current class); the archived-class evidence is one B2 episode, and B1-K / B0 share the class by recorded hashes.

## Decision (made before any test result)

Both execution classes are consistent, so HISTORICAL_EXTENSION (+60) and SYSTEM_REFERENCE (+30) are **included** in the candidate manifest, reported per execution class. Gaps carried, not repaired: UC2 (two evaluator versions; one-case differential only), UC3 (B0 original location inaccessible), UC4, UC5, UC7.

## Test source (frozen, nothing opened)

30 active ids in fixed order, reset seeds and row hashes recorded from the committed split configs; scene-set sha `db01aa572d9f8529...` from committed provenance. No test RGB/depth, VLM cache, relation truth or test result was opened. The future runner must build a no-prior split from `T_B_stage_2a_test30.json` rows (no cache-pointer keys) and never use `T_B_stage_2a_v11.json` test rows.

## Not done / next

No formal holdout was authorised or started. A separate holdout release card must state the S4 / S5 status for T_B, bind the candidate manifest to a committed source, add the small frozen-list runner around `evaluate_checkpoint_recorded`, and build the no-prior test split.
