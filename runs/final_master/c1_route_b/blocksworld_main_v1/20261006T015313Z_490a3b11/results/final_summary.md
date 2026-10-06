# CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 - stopped at PPO_ID_GATE_FAIL

State: `PPO_ID_GATE_FAIL` / `NEXT_ACTION = WAIT_FOR_USER_DECISION_ON_PLANNER_IMITATION` (runbook 12.3 and 16 E2). Amendment A01 in force (hop > 4 structurally empty; no receptive-field label).
Base `1f7b4f6ca`; freeze commit `490a3b11d49f6155baf2a6b6c430edac6609d2af`; three trainings run once each, in parallel, no rerun, no restart, no checkpoint selection.

## What ran

| run | method | updates / N | stop | NaN | wall (s) | final ID dev36 success | final ID dev36 decision-perfect | ID gate (>= 33 and >= 33) |
|---|---|---|---|---|---|---|---|---|
| R-C1-BW-B2-0 | B2-CACHED | 128 / 131072 | Ncap | 0 | 9436 | 34 | 34 | PASS |
| R-C1-BW-QMARK-0 | QMARK (BW adapter) | 128 / 131072 | Ncap | 0 | 9188 | 35 | 35 | PASS |
| R-C1-BW-ASNET-0 | ASNET-READOUT | 128 / 131072 | Ncap | 0 | 7482 | 34 | **26** | **FAIL** |

ID dev36 trajectory (success / decision-perfect) at step 0, N = 16384, 32768, 65536, 131072:
B2 0/0, 33/33, 34/34, 35/35, 34/34; QMARK 0/0, 30/29, 33/33, 35/34, 35/35; ASNET 0/0, 29/29, 34/33, 36/31, 34/26.

By the frozen rule the card stops: the three runs are kept, the A0 / A1 / A2 / B formal evaluations and the planner baseline were NOT run (`eval/` is empty), and no OOD interpretation is made.

## Why ASNET-READOUT fails the gate (descriptive, in-distribution only)

It reaches the goal in 34/36 dev cases but not optimally: only 26/36 episodes are decision-perfect. By block count (cases, success, perfect): n=3 12/12/6, n=4 12/11/10, n=5 12/11/10.
Ten of its dev episodes contain a repeated (state, action) cycle and eight succeed with extra steps (mean +3.5 over the optimum); first divergences are PICK_UP (5), STACK (4), UNSTACK (1). B2: 2 divergences (both n=3 failures), QMARK: 1 (one n=4 failure); neither succeeds with detours.
The gate asks for optimal action choice, not just success; the discounted sparse reward in PPO does not enforce it. ASNET's decision-perfect count peaked at 33 (N=32768), then fell to 31 (N=65536) and 26 (N=131072) while success stayed at 34-36.

## Disclosures

* One seed per method; the unit is the case, not independent trainings. 34/34, 35/35 vs 34/26 on 36 dev cases is a small, single-seed comparison.
* ASNET-READOUT is an ASNet-STYLE adapted readout on the same encoder and PPO (no teacher imitation, no alternating layers, no heuristic features); it is not a reproduction of the original ASNet system.
* The evaluation split files (A0 / A1 / A2 / B) were generated before the freeze and, before registration, opened once by a dry run of the evaluation / results scripts on UNREGISTERED 3-update smoke checkpoints (kept only in ~/tmp/bw, never in the repo) to test the code path. No registered checkpoint touched them; no formal evaluation file exists.
* n=3 multiplicity: 24 problem-isomorphism classes each used twice in the 48 n=3 train cases; the 12 n=3 dev cases are distinct classes with initial-state classes disjoint from train.
* A01: on the unchanged production graph every candidate is 1 or 3 hops from some unmet goal (the HandEmpty proposition is a hub); hop > 4 is structurally empty, so nothing here speaks to receptive-field limits.

## Decision requested

The runbook allows only one continuation: whether to authorize ONE unified planner-imitation training for all three methods (the default stays 3 trainings; nothing was switched automatically). Alternatives you may choose instead: accept the gate failure as the result of this card and stop; or relax the decision-perfect gate for ASNET only (this would change a frozen rule and needs an explicit amendment). No new seeds, no new algorithm and no evaluation will start without your instruction.
