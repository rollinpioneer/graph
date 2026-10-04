# CP-DISR-TB-SEED-BALANCE-01 — matched-seed results (frozen dev10 only)

Early-learning labels (frozen before any run): EARLY = dev10 10/10 at N=4096 or 8192 (first_effective_checkpoint 4096 / 8192); FINAL_ONLY = not early, final 10/10; NEVER = final not 10/10.

## Matched table (first_effective_checkpoint)

| family | seed0 | seed1 | seed2 | EARLY | FINAL_ONLY | NEVER |
|---|---|---|---|---|---|---|
| +E | FINAL_ONLY | FINAL_ONLY | 4096 | 1/3 | 2/3 | 0/3 |
| B2 | 4096 | 4096 | 4096 | 3/3 | 0/3 | 0/3 |
| ABS | FINAL_ONLY | 4096 | 8192 | 2/3 | 1/3 | 0/3 |
| NC | NEVER | FINAL_ONLY | NEVER | 0/3 | 1/3 | 2/3 |

## Same-seed comparison

| seed | B2 | +E | ABS | NC |
|---|---|---|---|---|
| 0 | 4096 | FINAL_ONLY | FINAL_ONLY | NEVER |
| 1 | 4096 | FINAL_ONLY | 4096 | FINAL_ONLY |
| 2 | 4096 | 4096 | 8192 | NEVER |

## Conclusion class (frozen rules, in order)

- if B2 EARLY seeds <= 1 of 3 -> SEED_SENSITIVITY_HIGH
- elif +E EARLY seeds >= 1 of 3 -> MIXED_REPRESENTATION_PATTERN
- elif B2 >= 2/3 EARLY and ABS >= 2/3 EARLY and +E 0/3 EARLY and NC 0/3 EARLY -> MECHANISM_PATTERN_REPLICATED
- else -> INCONCLUSIVE

**MIXED_REPRESENTATION_PATTERN** — +E has an EARLY seed

## Per-point values (frozen existing rows and the three new runs)

| group | run | method | seed | point | success | return | N | T |
|---|---|---|---|---|---|---|---|---|
| frozen_existing | R-TB-E-0 | B1-K+E | 0 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_existing | R-TB-E-0 | B1-K+E | 0 | 4096 | 0/10 | 0.000000 | 4096 | 18784.95 |
| frozen_existing | R-TB-E-0 | B1-K+E | 0 | 8192 | 0/10 | 0.000000 | 8192 | 38777.10 |
| frozen_existing | R-TB-E-0 | B1-K+E | 0 | final | 10/10 | 0.489340 | 14512 | 68813.30 |
| frozen_existing | R-TB-E-1 | B1-K+E | 1 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_existing | R-TB-E-1 | B1-K+E | 1 | 4096 | 0/10 | 0.000000 | 4096 | 20583.30 |
| frozen_existing | R-TB-E-1 | B1-K+E | 1 | 8192 | 0/10 | 0.000000 | 8192 | 40920.85 |
| frozen_existing | R-TB-E-1 | B1-K+E | 1 | final | 10/10 | 0.499977 | 14309 | 68820.25 |
| frozen_existing | R-TB-DK-1 | B2 | 1 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_existing | R-TB-DK-1 | B2 | 1 | 4096 | 10/10 | 0.498849 | 4096 | 20565.60 |
| frozen_existing | R-TB-DK-1 | B2 | 1 | 8192 | 10/10 | 0.498849 | 8192 | 40370.00 |
| frozen_existing | R-TB-DK-1 | B2 | 1 | final | 10/10 | 0.498849 | 14170 | 68817.10 |
| frozen_new | R-TB-ABS-0 | B2-ABS | 0 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-ABS-0 | B2-ABS | 0 | 4096 | 0/10 | 0.000000 | 4096 | 18972.50 |
| frozen_new | R-TB-ABS-0 | B2-ABS | 0 | 8192 | 0/10 | 0.000000 | 8192 | 38794.70 |
| frozen_new | R-TB-ABS-0 | B2-ABS | 0 | final | 10/10 | 0.498849 | 14070 | 68814.00 |
| frozen_new | R-TB-NC-0 | B1-K+NC | 0 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-NC-0 | B1-K+NC | 0 | 4096 | 0/10 | 0.000000 | 4096 | 18954.45 |
| frozen_new | R-TB-NC-0 | B1-K+NC | 0 | 8192 | 0/10 | 0.000000 | 8192 | 39034.10 |
| frozen_new | R-TB-NC-0 | B1-K+NC | 0 | final | 0/10 | 0.000000 | 14446 | 68815.75 |
| frozen_new | R-TB-ABS-1 | B2-ABS | 1 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-ABS-1 | B2-ABS | 1 | 4096 | 10/10 | 0.498849 | 4096 | 20540.55 |
| frozen_new | R-TB-ABS-1 | B2-ABS | 1 | 8192 | 10/10 | 0.498849 | 8192 | 40787.70 |
| frozen_new | R-TB-ABS-1 | B2-ABS | 1 | final | 10/10 | 0.498849 | 14048 | 68820.10 |
| frozen_new | R-TB-NC-1 | B1-K+NC | 1 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-NC-1 | B1-K+NC | 1 | 4096 | 0/10 | 0.000000 | 4096 | 20608.65 |
| frozen_new | R-TB-NC-1 | B1-K+NC | 1 | 8192 | 0/10 | 0.000000 | 8192 | 40485.20 |
| frozen_new | R-TB-NC-1 | B1-K+NC | 1 | final | 10/10 | 0.499977 | 14526 | 68817.20 |
| frozen_new | R-TB-ABS-2 | B2-ABS | 2 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-ABS-2 | B2-ABS | 2 | 4096 | 0/10 | 0.000000 | 4096 | 17942.75 |
| frozen_new | R-TB-ABS-2 | B2-ABS | 2 | 8192 | 10/10 | 0.498849 | 8192 | 37332.35 |
| frozen_new | R-TB-ABS-2 | B2-ABS | 2 | final | 10/10 | 0.498849 | 15093 | 68813.85 |
| frozen_new | R-TB-NC-2 | B1-K+NC | 2 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| frozen_new | R-TB-NC-2 | B1-K+NC | 2 | 4096 | 0/10 | 0.000000 | 4096 | 19952.60 |
| frozen_new | R-TB-NC-2 | B1-K+NC | 2 | 8192 | 0/10 | 0.000000 | 8192 | 36150.40 |
| frozen_new | R-TB-NC-2 | B1-K+NC | 2 | final | 0/10 | 0.000000 | 15499 | 68815.10 |
| new_this_card | R-TB-B2-0-CURRENT | B2 | 0 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| new_this_card | R-TB-B2-0-CURRENT | B2 | 0 | 4096 | 10/10 | 0.499977 | 4096 | 19219.55 |
| new_this_card | R-TB-B2-0-CURRENT | B2 | 0 | 8192 | 10/10 | 0.498849 | 8192 | 39346.00 |
| new_this_card | R-TB-B2-0-CURRENT | B2 | 0 | final | 10/10 | 0.498849 | 14451 | 68812.90 |
| new_this_card | R-TB-B2-2-CURRENT | B2 | 2 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| new_this_card | R-TB-B2-2-CURRENT | B2 | 2 | 4096 | 10/10 | 0.499977 | 4096 | 18175.50 |
| new_this_card | R-TB-B2-2-CURRENT | B2 | 2 | 8192 | 10/10 | 0.498849 | 8192 | 34001.70 |
| new_this_card | R-TB-B2-2-CURRENT | B2 | 2 | final | 10/10 | 0.499977 | 15583 | 68814.45 |
| new_this_card | R-TB-E-2-CURRENT | B1-K+E | 2 | 0 | 0/10 | 0.000000 | 0 | 0.00 |
| new_this_card | R-TB-E-2-CURRENT | B1-K+E | 2 | 4096 | 10/10 | 0.412617 | 4096 | 18682.90 |
| new_this_card | R-TB-E-2-CURRENT | B1-K+E | 2 | 8192 | 9/10 | 0.288375 | 8192 | 38839.95 |
| new_this_card | R-TB-E-2-CURRENT | B1-K+E | 2 | final | 10/10 | 0.400300 | 14845 | 68815.65 |

## Training budget actually used (new runs)

| run | N | T (s) | stop | complete+fragment | optimizer steps | NaN | hard_fail |
|---|---|---|---|---|---|---|---|
| R-TB-B2-0-CURRENT | 14451 | 68812.90 | Tcap | 14+1 | 892 | 0 | None |
| R-TB-B2-2-CURRENT | 15583 | 68814.45 | Tcap | 15+1 | 976 | 0 | None |
| R-TB-E-2-CURRENT | 14845 | 68815.65 | Tcap | 14+1 | 920 | 0 | None |
