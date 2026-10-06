# C1-BW-A04P — phase-1 summary (zero training)

State: **C1_BW_A04P_COMPLETE**. Probe state (registered rule, thresholds fixed before any output): **`GOAL_PROGRESS_READY`**. 0 training runs, 0 optimizer steps. All earlier states are unchanged
(`PPO_ID_GATE_FAIL`, `IMITATION_ID_GATE_FAIL`, `C1_BW_A03_EVALUATION_COMPLETE_WITH_ID_GAP`, REP label `NOT_ISSUED`, `NOT_TESTABLE_STRUCTURALLY_EMPTY`).
Completeness: 320/320 controller episodes (4 controllers x Pilot80) and 80/80 planner references, each started and completed once, 0 technical gaps; B2 checkpoint hash unchanged; old result trees unchanged (`verify.json`: PASS).
Original A04 did not exist locally or remotely (checked), so the four runbook controllers are the whole phase.

## 1. First-error table with explicit denominators (A03 logs, `first_error_denominator_table.csv/_summary.json`)

The audit's "failed-only" and "all with a first error" denominators were separated (groups: all episodes / failed / successful detours).
B2 A2: 24 of 32 episodes have a first error, **all 24 are failures and all 24 are "PUT_DOWN(x) instead of the goal STACK"**; the best optimal action is ranked 2nd in 21/24 (3rd–5th in 3).
B2 B slice: 46/48 have a first error (all failures); 29/46 are put-down-instead-of-goal-stack, best optimal rank 2 in 21/46. QMARK A2: 17 failures + 4 successful detours (21 episodes with a first error; 20 put-down-instead-of-stack, 16 of them in failures) — the audit's "20 vs 17" is not a failure share.
The first repeated state-action is an optimal action in 19/24 (B2 A2) and 29/46 (B2 B) of the failures: loops usually repeat a correct move after the first wrong one.

## 2. G0 goal/time paired forward check (consumed A03 states, development diagnostic; `probe_*`)

70 B2 failures in A2/B -> 46 valid samples (24 excluded: first error was not a PUT_DOWN; 0 replay mismatches, 0 local-order changes). Reference time tau = 0.25 (median over 196 D0 expert stacking steps).

| condition | n | argmax becomes the goal STACK | margin up | median margin change |
|---|---|---|---|---|
| G10 other goal towers flattened | 46 | 33 (72%) | 44 (96%) | +40.0 logits (median margin -16.1 -> +16.9) |
| G01 reference time input | 46 | 0 | 19 (41%) | -0.0006 |
| G11 both | 46 | 33 (72%) | 44 (96%) | +40.0 |

Reading: B2's wrong put-down is highly sensitive to the *other tower's goal relations*, with the same history and hidden; the time input has no measurable effect on this decision (the history stored in the GRU is not excluded).
This supports "score depends on the whole goal set" as sensitivity evidence only: flattening also turns the goal into a training-like single-tower goal; it does not prove that additive scoring is the cure or that coverage is not the cause.
Registered rule outcome: G10 meets the support condition, G01 does not -> `GOAL_PROGRESS_READY`.

## 3. Pilot80 (development evidence, success / decision-perfect; same 80 cases for all controllers)

| controller | A1 (16) | A2 (16) | B6 (16) | B7 (16) | B8 (16) | all (80) | rescued / harmed success vs C0 |
|---|---|---|---|---|---|---|---|
| C0 original B2 | 16/16 | 4/4 | 1/1 | 0/0 | 0/0 | 21/21 | - |
| C1 attempt counts | 16/16 | 9/4 | 3/1 | 0/0 | 0/0 | 28/21 | 7 / 0 |
| C3 visited-successor exclusion | 16/16 | 16/10 | 14/1 | 12/0 | 4/1 | **62/28** | 41 / 0 |
| G1 direct small-goal progress | 12/12 | 14/14 | 4/4 | 5/5 | 3/3 | 38/38 | 23 / 6 |

- C3 turns almost every loop into a success without any training (A2 16/16, B6 14/16, B7 12/16, B8 4/16) but with detours: decision-perfect stays 28/80 (it cannot be called an optimal-decision gain). One C3 episode ended `NO_UNVISITED_SUCCESSOR`.
- G1 improves A2 and every B slice in *optimal* decisions (38/38 overall) but harms colour-reversed A1 (12/16, 4 harmed): the direct rule is not free.
- C1 helps a little; it only acts after a repeat, so it cannot repair the first error (decision-perfect unchanged at 21).
- Pilot80 now belongs to development evidence; the headroom warning `LOW_HEADROOM_FOR_SUCCESS_GAIN` was not triggered (C3 is 100% on A2 but 87.5% on B6).

## Consequence under the runbook

The registered decision is `GOAL_PROGRESS_READY` and the whole runbook was authorised as a conditional execution, so phase 2 (at most 3 trainings M0/M1/M2, independent Confirm112) proceeds. The pilot also records that simple untrained rules (C3 for success, G1 for optimality) are strong baselines that phase 2 must be compared against; they are re-evaluated on the independent Confirm112 set, not carried over from here.
