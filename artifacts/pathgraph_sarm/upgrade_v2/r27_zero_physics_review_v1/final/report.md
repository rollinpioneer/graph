# R27 Zero-Physics Review Report

Formal R27 decision retained: **R27_INSUFFICIENT_OBSERVATION_HORIZON**

This review is additive. It does not modify the original R27 result directory, report, commit, selected online candidate, or recovery supervisor. It does not claim confirmation passed.

## Executed-Control Censoring

Episodes: 208; proposal episodes: 112; logged recovery-control executions: 96; proposal without execution: 16.

Proposal-only episodes preserved through their last candidate observation: 16/16. A proposal alone is never an observation cutoff.

The observation end is the last row strictly before the earliest logged recovery-control start, release, or attempt change. Recovery start requires both a non-empty execution record and a later active lifecycle row from a new attempt.

## Observation Coverage

| horizon | all loss full/total | all coverage | target loss full/total | target coverage |
|---:|---:|---:|---:|---:|
| 100 ms | 98/116 | 84.48% | 16/16 | 100.00% |
| 250 ms | 40/116 | 34.48% | 12/16 | 75.00% |
| 500 ms | 36/116 | 31.03% | 8/16 | 50.00% |
| 750 ms | 36/116 | 31.03% | 8/16 | 50.00% |
| 1000 ms | 24/116 | 20.69% | 4/16 | 25.00% |

## B1/B2/B3 Counts

Counts are per horizon over all reference-loss episodes. `late` is retained separately for the 1000 ms diagnostic window; the formal on-time deadline remains 750 ms.

| method | horizon | on time | early | late | missed | right censored | false loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 100 ms | 0 | 0 | 0 | 98 | 18 | 0 |
| B1 | 250 ms | 0 | 0 | 0 | 40 | 76 | 0 |
| B1 | 500 ms | 0 | 0 | 0 | 36 | 80 | 0 |
| B1 | 750 ms | 0 | 0 | 0 | 36 | 80 | 0 |
| B1 | 1000 ms | 0 | 0 | 0 | 24 | 92 | 0 |
| B2 | 100 ms | 38 | 18 | 0 | 42 | 18 | 0 |
| B2 | 250 ms | 38 | 18 | 0 | 24 | 36 | 0 |
| B2 | 500 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B2 | 750 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B2 | 1000 ms | 38 | 18 | 0 | 8 | 52 | 0 |
| B3 | 100 ms | 38 | 18 | 0 | 42 | 18 | 0 |
| B3 | 250 ms | 38 | 18 | 0 | 24 | 36 | 0 |
| B3 | 500 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B3 | 750 ms | 38 | 18 | 0 | 20 | 40 | 0 |
| B3 | 1000 ms | 38 | 18 | 0 | 8 | 52 | 0 |

Per-family counts for every horizon are in `outer_validation/by_family_metrics.csv`.

## Frozen CLP3 Interface Shadow

The test uses the frozen `INTERCEPT_REASON` value `historical_hold_established_then_observed_non_release_contact_loss`.

The first exact-reason proposal with measured `contact=false` becomes pending and is suppressed while waiting; same-time duplicates remain pending; a strictly later same-attempt `contact=false` row at 100 ms confirms. `PASS_THROUGH` preserves the input proposal/action but is not a CLP3 confirmation. `CLEAR` resets pending and is also not confirmation. `contact=true`, `contact=null`, release, attempt change, and a different reason are never converted into fabricated contact.

## Development Replenishment

Status: `DEVELOPMENT_COLLECTION_COMPLETE`; physical development rollouts: 24.

At 750 ms, actual-loss coverage is 8/8 (100.00%). The intended slow-detachment cases that did not become physical loss remain no-loss outcomes.

| method | on time | early | late | missed | right censored | false loss |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0 | 0 | 0 | 8 | 0 | 0 |
| B2 | 2 | 0 | 0 | 6 | 0 | 0 |
| B3 | 2 | 0 | 0 | 6 | 0 | 0 |

All 24 development rollouts retained exactly 1.50 s after intervention end. Candidate recovery-control executions and candidate-triggered early terminations were both zero.

Any replenishment comparison is development-only and is not an independent confirmation.

## Conclusion

Corrected cache-only route result: **R27_INSUFFICIENT_OBSERVATION_HORIZON**.

Formal R27 remains unchanged. Confirmation is not passed.
