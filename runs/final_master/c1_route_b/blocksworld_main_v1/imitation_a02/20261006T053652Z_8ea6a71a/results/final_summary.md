# CP-DISR-C1-BW-IMITATION-A02 — final summary

State: **IMITATION_ID_GATE_FAIL** (the PPO state `PPO_ID_GATE_FAIL` is preserved and unchanged). `limitation_label = LIMIT_TRAINING`; no representation (REP) label is issued;
`receptive_field_status = NOT_TESTABLE_STRUCTURALLY_EMPTY`. A0 / A1 / A2 / B and the planner baseline were NOT run (formal evaluation condition not met).

Protocol (as authorized): three methods trained from scratch (init seed 0) on identical planner-labelled datasets, loss `-log sum_{a in A*} pi(a|s)` over the FULL optimal-action set,
100 + 50 + 50 epochs with two shared aggregation rounds, last epoch only, no checkpoint selection. D0 / D1 / D2 hashes are identical across the three runs
(D0 6b053f16…, D1 8d542bf7…, D2 d4c9fae9…; 144 / 576 / 1008 trajectories, 810 / 3384 / 5826 decisions). No NaN. Frozen V/Q heads unchanged.

ID gate on dev36 (final imitation checkpoint, evaluated once; requires success >= 33/36 AND decision-perfect >= 33/36):

| method | success | decision-perfect | gate |
|---|---|---|---|
| B2-CACHED | 36/36 | 35/36 | PASS |
| QMARK | 34/36 | 33/36 | PASS |
| ASNET-READOUT | 36/36 | 32/36 | **FAIL** (decision-perfect 32 < 33) |

ASNET-READOUT's four non-decision-perfect dev episodes: BW_dev_n4_002, BW_dev_n4_005, BW_dev_n4_010, BW_dev_n5_004 (three contain a repeated state-action cycle; all four still reach the goal).
Training-set fit of the final epoch (optimal-action mass / argmax-in-A*): B2 1.000 / 1.000, QMARK 1.000 / 1.000, ASNET 0.996 / 0.997.

Because the pre-registered rule requires ALL THREE methods to pass, the gate fails; per A02 the run stops here (no extra rounds or epochs, no mid checkpoint, no single-method change, no other supervision,
no formal evaluation set opened). Next action: WAIT_FOR_USER_MAINLINE_DECISION.
