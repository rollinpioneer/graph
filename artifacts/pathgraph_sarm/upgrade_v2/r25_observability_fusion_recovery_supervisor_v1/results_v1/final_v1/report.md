# R25 observability fusion × recovery supervisor factorial confirmation

Date: 2026-09-12

R25 started from `17898e29d73cce33eb594bd1a0496500273b3053` and ran from frozen runner
commit `6d8b1012b3f6d8689bff5c4239d098528238fa74`. R22, R23, R24,
`O_C3_CLP3_CANONICAL_TIME`, the canonical-time adapter, visual detector,
contact proxy, and R23 controller remained read-only and blob-identical.

## R24 zero-physics development

The eight R24 `SIGNAL_NOT_OBSERVED` episodes were separated from recovery
execution failures. C2 masked contact as true; C4 used leading missing values
that removed the raw O_C3 true-to-false edge. `LossObservabilityFusionV1`
formed one diagnostic proposal in 8/8 and zero proposals in the eight
partial-slip/release controls. The three `TASK_RECOVERY_ERROR` episodes were
separated from four C6 `RELOCATION_ERROR` comparators.

`RecoverySupervisorV2` passed fixed development tests for bounded relocation
retry, post-recovery goal verification, second-loss rearming, and bounded
permanent failure. At this point the fallback thresholds were frozen.

## Independent 2×2 confirmation

Four new families and new seeds produced 112/112 physical rollouts. Collection
passed without execution errors. Final task success by factor cell was:

- CLP3 baseline: 8/28.
- Fusion only: 12/28.
- Supervisor V2 only: 20/28.
- Fusion + Supervisor V2: 26/28.

Supervisor V2 passed all three targeted confirmation gates: transient
relocation 8/8, goal-verification retry 8/8, and secondary-loss recovery 8/8.
Both safety controls had zero false executed actions.

The overall decision is **FAIL** because Fusion exposed 12/16 observability
factor outcomes rather than 16/16, and the full factor cell reached 26/28.
These four arm outcomes represent two underlying family/case conditions in
family 898003. Physical traces show weld off and `inside_capture=false`, but
the bounded candidate stream contained only one distinct usable contact-false
timestamp. The maximum real-RGB relative displacement was approximately
5.84 px, below the frozen 6.0 px threshold.

All such outcomes remain `SIGNAL_NOT_OBSERVED` with `candidate_error=false`.
No confirmation rollout was replaced or rerun, and neither CLP3 nor any
fallback threshold was changed using confirmation data.
