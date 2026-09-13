# R26 normalized observability fusion — final report

## Decision

`CURRENT_SENSOR_REPRESENTATION_NOT_SEPARABLE`. The R24/R25 zero-physics
family-wise audit did not meet the required exposure and false-proposal gates.
No R26 confirmation rollout was started and no frozen L2R candidate was changed.

## Evidence

The audit processed 208 historical episodes and 27,350 deduplicated frames.
Evidence identity was `(physical_time_ns, jpeg_sha256)`; duplicate IDs were
never counted as a second visual observation. The selected LOFO thresholds
were evaluated without reading reference data during online processing.

Observed gate values were R24 missing-signal exposure 5/8, R25 observability
exposure 4/16, held-out negative false proposals 108, overall positive
exposure 0.375, and minimum per-family exposure 0.0. Input provenance and
prefix-causality audits passed, but the hard separability gate therefore failed.

The failure is a sensor-representation limitation, not a candidate error. The
R22, R23, R24, R25 results and `O_C3_CLP3_CANONICAL_TIME` remain read-only.
`LossObservabilityFusionV2_NormalizedEvidence` was not selected; confirmation
parameters were not tuned and no physical budget was consumed.
