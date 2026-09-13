# R27 Temporal Loss Decision Report

Final decision: **R27_INSUFFICIENT_OBSERVATION_HORIZON**

## Scope

This is a zero-physics replay of the available R24/R25 caches. R26 historical results are preserved and not replaced.

## Observation horizons

Episodes processed: 208
Target loss episodes: 16
Target episodes with an un-intervened 750 ms prefix: 8 (0.500)

## Candidate comparison

```json
{
  "B1": {
    "method": "B1",
    "horizon_ns": "750000000",
    "episodes": "208",
    "reference_loss_episodes": "116",
    "fully_observed_loss_episodes": "36",
    "on_time": "0",
    "early": "0",
    "late": "0",
    "missed": "36",
    "right_censored": "80",
    "false_loss_events": "0"
  },
  "B2": {
    "method": "B2",
    "horizon_ns": "750000000",
    "episodes": "208",
    "reference_loss_episodes": "116",
    "fully_observed_loss_episodes": "36",
    "on_time": "38",
    "early": "18",
    "late": "0",
    "missed": "20",
    "right_censored": "40",
    "false_loss_events": "0"
  },
  "B3": {
    "method": "B3",
    "horizon_ns": "750000000",
    "episodes": "208",
    "reference_loss_episodes": "116",
    "fully_observed_loss_episodes": "36",
    "on_time": "38",
    "early": "18",
    "late": "0",
    "missed": "20",
    "right_censored": "40",
    "false_loss_events": "0"
  }
}
```

## Interface shadow

The three contact semantics are tested without forcing contact=false and without executing recovery.

## Route choice

R27_INSUFFICIENT_OBSERVATION_HORIZON

No physical confirmation or downstream training was started.
