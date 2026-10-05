# CP-DISR-TB-STRUCT-GEN-V1 - final summary

Card: T_B structural generalization, 12 training runs (+E, B2, ABS, NC x seeds 0/1/2), frozen release, prep commit `0fa123fd200fa7aa90f543aed63a54ff80c5faf3`, branch `codex/cp-disr-tb-structural-generalization-v1`.
All 12 runs stopped at Tcap (68812.8 s), 0 NaN, 0 hard fail, `verify.json` = PASS. Test30 was evaluated post hoc, once, on each run's final checkpoint only.

## Headline

* **OUTCOME_A** by the frozen rules (`STRUCTURAL_GROUNDING_HYPOTHESIS_STRENGTHENED`); paper effect label **STRONGLY_STRENGTHENED**.
* Final test30 success, mean over seeds: **B2 0.922, ABS 0.733, NC 0.222, +E 0.000**.
* Delta = mean(B2,ABS) - mean(+E,NC) = 0.717; per-seed delta 0.617 / 0.733 / 0.800; successor family leads by >= 0.10 in 3/3 test cells.
* Rule trace: D not met (T(NC) 0.222 < 0.5); B not met (max T 0.922); C not met (T(B2)-T(ABS) = 0.189 < 0.25); A met.
* This is 3 seeds on one task (T_B) and one suite. Within a cell the evaluated policy is deterministic (argmax), and several runs score 10/10 or 0/10 in a cell, so the effective unit of evidence is seed x cell, not the 10 episodes of a cell.

## What the suite changes (and what it does not)

Only the goal composition (which known action consequences must be combined) and which object is bound to which destination change. Contracts, predicates, candidate IDs, hard mask, controller, perception, verifier, reward, deadline, layout and the evaluator's atomic checks are the frozen T_B ones. No new skill, predicate or object type. Train/dev cells: IN_T, BUF_S, IN_T+BUF_S (60 train, 12 dev cases). Test cells (30 cases, separate file never opened by the training loader): IN_S, BUF_T+BUF_S, IN_S+BUF_T. Dev12 is a new dataset identity and is never mixed with old dev10 counts.

## Axes (not a full factorial)

* Main generalization variable: unseen binding / effect composition (every test cell contains a binding absent from train: second_object->container or target->buffer).
* Dependency depth (shortest legal nominal-apply chain) is a preregistered **descriptive stratification**: IN_T 3, BUF_S 2, IN_T+BUF_S 5 (train/dev); IN_S 3, BUF_T+BUF_S 4, IN_S+BUF_T 5 (test). Test depth 4 mixes novel depth, composition and binding, so no independent depth causal claim is made.
* Matched-depth comparisons: depth 3 = dev IN_T vs test IN_S; depth 5 = dev IN_T+BUF_S vs test IN_S+BUF_T.

## Shortcut audit and qualification

* Shortcut audit S1-S6 passed before any run (`prep/shortcut_audit.json`, `prep/binding_novelty_audit.json`, `prep/dependency_depth_audit.json`).
* Physical qualification (<= 6 episodes, no training): the strict criterion (verified public facts == nominal successor facts at every step) **FAILED on all 6 cases; that raw FAIL is preserved** (`prep/physical_qualification_results.json`). Status carried forward is only **PASS_UNDER_DISCLOSED_AMENDMENT** (`prep/physical_qualification_amendment.json`). Basis: 6/6 TASK_SUCCESS, normal controller exits, zero TRUE/FALSE contradiction, and discrepancies only of the kinds UNKNOWN propagation, partial observation, or frozen-contract OnTable semantics, all within frozen-T_B reference categories. We do not claim the nominal successor equals the real observed state.
* Design note: lid-open depth variation was infeasible under the unchanged evaluator, so difficulty is varied through goal-composition cells instead.

## A. Learning dynamics (dev12 successes out of 12, seeds 0/1/2)

| family | N=0 | 4096 | 8192 | final |
|---|---|---|---|---|
| +E  | 0 / 4 / 4 | 4 / 9 / 12 | 12 / 12 / 12 | 12 / 12 / 12 |
| B2  | 3 / 0 / 4 | 12 / 12 / 12 | 12 / 12 / 12 | 12 / 12 / 12 |
| ABS | 3 / 0 / 4 | 12 / 8 / 12 | 12 / 12 / 12 | 12 / 12 / 12 |
| NC  | 4 / 0 / 4 | 8 / 4 / 4 | 8 / 4 / 12 | 4 / 8 / 8 |

Early-learning labels (dev full success at 4096/8192): +E 3 EARLY (first effective 8192, 8192, 4096); B2 3 EARLY (4096 x3); ABS 3 EARLY (4096, 8192, 4096); NC 1 EARLY (seed 2, 8192) and 2 NEVER. NC-0 and NC-2 regressed from their dev best by the final checkpoint (8 -> 4 and 12 -> 8).

## B. In-distribution dev12 at final

+E 12/12 in every seed, B2 12/12, ABS 12/12; NC 4, 8, 8 of 12. By dev cell: NC-0 passes BUF_S only (4/4), NC-1 and NC-2 fail IN_T+BUF_S (0/4); all other runs 4/4 in all three dev cells.

## C. Structural-generalization test30 at final (success n/30, per-cell n/10 as IN_S, BUF_T+BUF_S, IN_S+BUF_T)

| run | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| +E  | 0/30 | 0/30 | 0/30 | 0.000 |
| B2  | 27/30 (10,8,9) | 27/30 (10,8,9) | 29/30 (10,9,10) | 0.922 |
| ABS | 20/30 (10,0,10) | 27/30 (10,8,9) | 19/30 (10,0,9) | 0.733 |
| NC  | 10/30 (10,0,0) | 10/30 (0,0,10) | 0/30 | 0.222 |

Pooled test rate by cell (mean over seeds): +E 0 / 0 / 0; B2 1.00 / 0.83 / 0.93; ABS 1.00 / 0.27 / 0.93; NC 0.33 / 0.00 / 0.33. By binding-novelty group (second_object->container involved / target->buffer involved): +E 0 / 0; B2 0.97 / 0.88; ABS 0.97 / 0.60; NC 0.33 / 0.17.
Test failures are all time/no-safe-termination failures (INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE or NO_CANDIDATE_SAFE_TERMINATION), never a TASK_SUCCESS reversal.

## D. Matched-depth unseen-binding comparison (final; seen-binding dev cell vs unseen-binding test cell at equal depth)

| family | depth 3 dev IN_T (12 cases) | depth 3 test IN_S (30) | depth 5 dev IN_T+BUF_S (12) | depth 5 test IN_S+BUF_T (30) |
|---|---|---|---|---|
| +E  | 12/12 | 0/30  | 12/12 | 0/30 |
| B2  | 12/12 | 30/30 | 12/12 | 28/30 |
| ABS | 12/12 | 30/30 | 12/12 | 28/30 |
| NC  | 8/12 (0,4,4) | 10/30 (10,0,0) | 0/12 | 10/30 (0,10,0) |

At matched depth the successor-difference families (B2, ABS) keep dev performance on the unseen binding (<= 2/30 loss at depth 5), +E drops from full dev success to zero in both comparisons, and NC is seed-dependent and below the successor families in both ID and test.

## E. Depth strata (descriptive only, no independent depth claim)

Test by depth: depth 3 (IN_S) B2 1.00, ABS 1.00, NC 0.33, +E 0; depth 4 (BUF_T+BUF_S, novel depth + composition + binding) B2 0.83, ABS 0.27, NC 0, +E 0; depth 5 (IN_S+BUF_T) B2 0.93, ABS 0.93, NC 0.33, +E 0. Depth 4 is the only stratum where B2 and ABS differ; it confounds three novelties and the ABS result is bimodal by seed (0/10, 8/10, 0/10), so we do not attribute it to depth or to any single factor.

## F. Generalization gap (dev12 final minus test30 final)

| family | s0 | s1 | s2 |
|---|---|---|---|
| +E  | 1.000 | 1.000 | 1.000 |
| B2  | 0.100 | 0.100 | 0.033 |
| ABS | 0.333 | 0.100 | 0.367 |
| NC  | 0.000 | 0.333 | 0.667 |

NC-0's zero gap is because its dev score was already low (4/12), not because it generalizes.

## G. Outcome, paper effect, need for a new method

* **OUTCOME_A**, effect **STRONGLY_STRENGTHENED**, as defined by the frozen rules written before any run. These labels are rule outputs on 3 seeds, one task, one suite.
* Reading, within the frozen boundaries: representations built on the nominal successor difference (B2, ABS) transferred to unseen binding/composition far better than the direct grounded-effect (+E) and state-effect-interaction (NC) representations, which both fit dev12 early but (+E) fail completely or (NC) inconsistently on test.
* B2 is near ceiling (0.92) and ABS reaches 0.73; the remaining B2/ABS gap is concentrated in the confounded depth-4 cell. This card gives no evidence that a new method is needed; the open question is the ABS depth-4 seed variability, which would need new seeds or a cleaner depth/binding design, neither of which was authorized here.

## Parameter / computation / wall-clock identity

Effective trainable parameters: B2 and ABS 849,987; NC 884,291; +E 2,695,363 (`prep/capacity_identity.json`, equal to the representation-control audit). Same Tcap/Ncap, PPO settings and eval points for all runs. Actual N at stop: +E 14491/14399/14591; B2 14409/14384/14532; ABS 15250/14374/14281; NC 14562/14964/14632 (`training_accounting.json`). Wall-clock time is not sample efficiency and is not used as evidence.

## Concurrency amendment (12-wide)

On user request (with rollback condition), the plan ran 12-wide instead of 6-wide: six extra workers shared GPUs 1-6 with the first six, started through the unchanged CLI under in-process overrides (MAX_WORKERS=12, shared-GPU refusal skipped); tracked source, HEAD and release unchanged. A reversible SIGSTOP/SIGCONT guard was active (load1 > 75, MemAvailable < 30 GiB, or GPU memory thresholds); its first version paused the extra group once at 16:52Z on 2026-10-04 (foreign GPU job); it was replaced by a per-worker version (own-GPU threshold) and the workers were resumed in place at 16:59Z; after that nothing was paused. Simulated-time budgets are independent of wall time, so scientific content is unchanged (`concurrency_amendment.json`). Foreign jobs by other users occupied GPUs 1 and 6 intermittently; nothing was killed or restarted.
