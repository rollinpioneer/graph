# CP-DISR-TB-REP-CONTROLS-01 — frozen-dev results

Frozen dev10, deterministic argmax, H = 23.09999999999752 s, Tcap = 68812.79999996186 s, Ncap = 16384. Primary development endpoints: dev10 success, mean discounted return, actual N, actual T at N = 0 / 4096 / 8192 / final. No test30, no holdout, no provider. Source of the new runs: prep commit `014191037319f60bcd985d995996b2ee9a603dea`.

## dev10 success (n of 10)

| group | run | method | seed | 0 | 4096 | 8192 | final |
|---|---|---|---|---|---|---|---|
| existing | R-TB-E-0 | B1-K+E | 0 | 0 | 0 | 0 | 10 |
| existing | R-TB-E-1 | B1-K+E | 1 | 0 | 0 | 0 | 10 |
| existing | R-TB-DK-1 | B2 | 1 | 0 | 10 | 10 | 10 |
| existing | R-TB-K-1 | B1-K | 1 | 0 | 0 | 0 | 0 |
| new | R-TB-ABS-0 | B2-ABS | 0 | 0 | 0 | 0 | 10 |
| new | R-TB-NC-0 | B1-K+NC | 0 | 0 | 0 | 0 | 0 |
| new | R-TB-ABS-1 | B2-ABS | 1 | 0 | 10 | 10 | 10 |
| new | R-TB-NC-1 | B1-K+NC | 1 | 0 | 0 | 0 | 10 |
| new | R-TB-ABS-2 | B2-ABS | 2 | 0 | 0 | 10 | 10 |
| new | R-TB-NC-2 | B1-K+NC | 2 | 0 | 0 | 0 | 0 |

## mean discounted return

| group | run | method | seed | 0 | 4096 | 8192 | final |
|---|---|---|---|---|---|---|---|
| existing | R-TB-E-0 | B1-K+E | 0 | 0.000000 | 0.000000 | 0.000000 | 0.489340 |
| existing | R-TB-E-1 | B1-K+E | 1 | 0.000000 | 0.000000 | 0.000000 | 0.499977 |
| existing | R-TB-DK-1 | B2 | 1 | 0.000000 | 0.498849 | 0.498849 | 0.498849 |
| existing | R-TB-K-1 | B1-K | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| new | R-TB-ABS-0 | B2-ABS | 0 | 0.000000 | 0.000000 | 0.000000 | 0.498849 |
| new | R-TB-NC-0 | B1-K+NC | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| new | R-TB-ABS-1 | B2-ABS | 1 | 0.000000 | 0.498849 | 0.498849 | 0.498849 |
| new | R-TB-NC-1 | B1-K+NC | 1 | 0.000000 | 0.000000 | 0.000000 | 0.499977 |
| new | R-TB-ABS-2 | B2-ABS | 2 | 0.000000 | 0.000000 | 0.498849 | 0.498849 |
| new | R-TB-NC-2 | B1-K+NC | 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Actual N / T at each evaluation point

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R-TB-E-0 | u0: N 0 / T 0.00 | u4: N 4096 / T 18784.95 | u8: N 8192 / T 38777.10 | u14: N 14512 / T 68813.30 |
| R-TB-E-1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20583.30 | u8: N 8192 / T 40920.85 | u13: N 14309 / T 68820.25 |
| R-TB-DK-1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20565.60 | u8: N 8192 / T 40370.00 | u13: N 14170 / T 68817.10 |
| R-TB-K-1 | u0: N 0 / T 0.00 | u4: N 4096 / T 18022.50 | u8: N 8192 / T 37574.70 | u14: N 14531 / T 68813.45 |
| R-TB-ABS-0 | u0: N 0 / T 0.00 | u4: N 4096 / T 18972.50 | u8: N 8192 / T 38794.70 | u13: N 14070 / T 68814.00 |
| R-TB-NC-0 | u0: N 0 / T 0.00 | u4: N 4096 / T 18954.45 | u8: N 8192 / T 39034.10 | u14: N 14446 / T 68815.75 |
| R-TB-ABS-1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20540.55 | u8: N 8192 / T 40787.70 | u13: N 14048 / T 68820.10 |
| R-TB-NC-1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20608.65 | u8: N 8192 / T 40485.20 | u14: N 14526 / T 68817.20 |
| R-TB-ABS-2 | u0: N 0 / T 0.00 | u4: N 4096 / T 17942.75 | u8: N 8192 / T 37332.35 | u14: N 15093 / T 68813.85 |
| R-TB-NC-2 | u0: N 0 / T 0.00 | u4: N 4096 / T 19952.60 | u8: N 8192 / T 36150.40 | u15: N 15499 / T 68815.10 |

## Failure-reason families (not-success episodes)

| run | point | families |
|---|---|---|
| R-TB-E-0 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-E-0 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-E-0 | 8192 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 7, "NO_CANDIDATE_SAFE_TERMINATION": 3} |
| R-TB-E-1 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-E-1 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-E-1 | 8192 | {"NO_CANDIDATE_SAFE_TERMINATION": 10} |
| R-TB-DK-1 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-K-1 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-K-1 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-K-1 | 8192 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-K-1 | final | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-ABS-0 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-ABS-0 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-ABS-0 | 8192 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-0 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-0 | 4096 | {"NO_CANDIDATE_SAFE_TERMINATION": 10} |
| R-TB-NC-0 | 8192 | {"NO_CANDIDATE_SAFE_TERMINATION": 10} |
| R-TB-NC-0 | final | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-ABS-1 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-1 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-1 | 4096 | {"NO_CANDIDATE_SAFE_TERMINATION": 10} |
| R-TB-NC-1 | 8192 | {"NO_CANDIDATE_SAFE_TERMINATION": 10} |
| R-TB-ABS-2 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-ABS-2 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-2 | 0 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-2 | 4096 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-2 | 8192 | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |
| R-TB-NC-2 | final | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} |

## Training budget actually used (new runs)

| run | N | T (s) | stop | complete+fragment updates | optimizer steps | NaN | hard_fail |
|---|---|---|---|---|---|---|---|
| R-TB-ABS-0 | 14070 | 68814.00 | Tcap | 13+1 | 860 | 0 | None |
| R-TB-NC-0 | 14446 | 68815.75 | Tcap | 14+1 | 880 | 0 | None |
| R-TB-ABS-1 | 14048 | 68820.10 | Tcap | 13+1 | 868 | 0 | None |
| R-TB-NC-1 | 14526 | 68817.20 | Tcap | 14+1 | 896 | 0 | None |
| R-TB-ABS-2 | 15093 | 68813.85 | Tcap | 14+1 | 924 | 0 | None |
| R-TB-NC-2 | 15499 | 68815.10 | Tcap | 15+1 | 948 | 0 | None |

## Seed consistency and rule-based case candidates

```json
{
 "existing_reference": {
  "R-TB-DK-1": {
   "early": true,
   "pattern": {
    "0": 0,
    "4096": 10,
    "8192": 10,
    "final": 10
   }
  },
  "R-TB-E-0": {
   "early": false,
   "pattern": {
    "0": 0,
    "4096": 0,
    "8192": 0,
    "final": 10
   }
  },
  "R-TB-E-1": {
   "early": false,
   "pattern": {
    "0": 0,
    "4096": 0,
    "8192": 0,
    "final": 10
   }
  },
  "R-TB-K-1": {
   "early": false,
   "pattern": {
    "0": 0,
    "4096": 0,
    "8192": 0,
    "final": 0
   }
  }
 },
 "new": {
  "ABS": {
   "R-TB-ABS-0": {
    "early": false,
    "pattern": {
     "0": 0,
     "4096": 0,
     "8192": 0,
     "final": 10
    }
   },
   "R-TB-ABS-1": {
    "early": true,
    "pattern": {
     "0": 0,
     "4096": 10,
     "8192": 10,
     "final": 10
    }
   },
   "R-TB-ABS-2": {
    "early": true,
    "pattern": {
     "0": 0,
     "4096": 0,
     "8192": 10,
     "final": 10
    }
   }
  },
  "NC": {
   "R-TB-NC-0": {
    "early": false,
    "pattern": {
     "0": 0,
     "4096": 0,
     "8192": 0,
     "final": 0
    }
   },
   "R-TB-NC-1": {
    "early": false,
    "pattern": {
     "0": 0,
     "4096": 0,
     "8192": 0,
     "final": 10
    }
   },
   "R-TB-NC-2": {
    "early": false,
    "pattern": {
     "0": 0,
     "4096": 0,
     "8192": 0,
     "final": 0
    }
   }
  }
 },
 "rule": "early = dev10 10/10 at N=4096 or N=8192 (pre-registered); a control is 'B2-like' when a strict majority of its seeds is early",
 "rule_based_candidates": [
  "CASE_A",
  "CASE_D_flag"
 ]
}
```

## Identity notes

- Existing +E / B2 / B1-K rows are the current-profile runs R-TB-E-0, R-TB-E-1, R-TB-DK-1, R-TB-K-1 (prep cac2fa5b / e4dc34ae; 03_frozen_model_list.md); they are read-only and unchanged.
- The historical seed-0 B2 run (R1, original execution identity) is not in this table; it was 0/10 at 4096 and 10/10 at 8192 and final.
- Training seeds: ABS and NC use seeds 0/1/2 (the existing +E has seeds 0,1; B2 has seed 1 in the current profile). Seeds are not balanced across all arms; this is reported, not corrected.

