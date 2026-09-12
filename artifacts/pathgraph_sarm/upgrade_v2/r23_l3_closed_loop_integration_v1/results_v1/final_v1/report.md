# R23 L3 closed-loop integration

Date: 2026-09-12

The experiment freezes `O_C3_CLP3_CANONICAL_TIME` at base commit
`cb6ed23f88e44336f17197c300bc72c7fe19df49`. No L2R candidate,
threshold, temporal scoring, or physical-reference source was modified.

R23 V0 retained all 60 rollouts and failed because one family did not expose
`contact=false` within the fixed 0.20 s scene window. Both CLP3 and raw O_C3
therefore had no recovery trigger. This was classified as a triggering-stage
orchestration error, not a candidate miss.

R23R1 used four different families and seeds, and waited for a real observable
contact-loss event within a bounded 1.0 s scene window. The frozen runner then
executed 60 paired rollouts across CLP3, raw O_C3, and recovery-disabled arms.

- CLP3: 20/20 final task success, 8/8 actual retry/recover success, zero false
  executed actions, and one recovery loop per positive rollout.
- Raw O_C3: 20/20 final task success, but four unnecessary recoveries on the
  four single-frame contact faults; overall success 16/20.
- Recovery-disabled: 12/20 final task success and 0/8 positive recovery
  success.

Failure attribution is recorded as `TRIGGERING_ERROR`, `RELOCATION_ERROR`,
`REGRASP_ERROR`, or `TASK_RECOVERY_ERROR`. R23R1 CLP3 produced none of these.
Raw RGB and complete physical traces remain in the external confirmation
directories; the Git result package contains locks, summaries, paired metrics,
failure attribution, and external-data provenance.
