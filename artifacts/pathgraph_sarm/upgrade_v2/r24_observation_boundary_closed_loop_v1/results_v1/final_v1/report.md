# R24 observation-boundary and closed-loop robustness

Date: 2026-09-13

R24 is based on `429efbb3fca0572f7afe4bc6d535433a54d7058f` and executed from frozen runner commit
`ea9823e0b63a679918aa2f6b54b1fc50a4211a4d`. `O_C3_CLP3_CANONICAL_TIME`, its canonical-time adapter,
the visual detector, contact proxy, and the R23 closed-loop controller remained
blob-identical. R22 and R23 stayed read-only; `main` was not modified.

The collection completed 96/96 new physical rollouts across four new families,
eight robustness cases, and three paired methods. Collection passed without an
execution exception. The frozen evaluation decision is **FAIL**.

## Primary results

- CLP3 final task success: 17/32 (53.1%).
- Raw O_C3 final task success: 16/32 (50.0%).
- Recovery-disabled final task success: 8/32 (25.0%).
- CLP3 signal exposure among physical-loss rollouts: 20/28 (71.4%); among recovery-required rollouts: 16/24 (66.7%).
- CLP3 recovery execution: 14 successful and 4 failed outer executions, with 18 outer cycles and 26 inner loops.
- CLP3 task duration: median 6.70 s, mean 6.81 s, p90 10.76 s.

## Observation-boundary attribution

C2 and C4 account for eight CLP3 rollouts in which physical loss existed but
the frozen online trigger signal did not form. All eight are recorded as
`SIGNAL_NOT_OBSERVED`; all have `candidate_error=false`. C4 can contain later
`contact=false` samples while still lacking the required raw O_C3 true-to-false
event edge. These are signal-exposure failures, not candidate classification
errors.

CLP3 produced zero false executed actions in partial-slip and active-release
conflict cases. Raw O_C3 executed four unnecessary recoveries on the four
partial-slip cases.

## Failed robustness gates

- C1 delayed contact loss: 3/4 final task success. Family seed 896002 completed
  relocation, regrasp, and 0.50 s hold verification, but final placement did
  not satisfy the goal-stable predicate; attribution is `TASK_RECOVERY_ERROR`.
- C8 secondary loss: 2/4 final task success. Families 896000 and 896003 formed
  and executed two recovery cycles. Families 896001 and 896002 executed one
  successful recovery, but no second recovery action formed after the secondary
  loss during resumed transport; both remain `TASK_RECOVERY_ERROR`.
- C6 relocation error: 4/4 correctly localized as `RELOCATION_ERROR`.
- C7 regrasp disturbance: 4/4 final success, requiring two inner loops per
  rollout and demonstrating successful bounded regrasp retry.

No result was rerun or replaced after the FAIL decision. Complete RGB, online
streams, decisions, and physics traces remain in the external data directory;
Git contains locks, collection/evaluation summaries, paired metrics, failure
localization, and integrity provenance.
