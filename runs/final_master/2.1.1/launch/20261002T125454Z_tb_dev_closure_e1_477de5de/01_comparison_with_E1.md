# T_B development evidence: comparison table with R-TB-E-1

Built from existing files only (no episode, no environment, no model load, no test). The comparison base is the earlier three-run table (R-TB-E-0, R-TB-DK-1, R-TB-K-1); R-TB-E-1 is added without re-auditing the repository. Re-read values of the three base runs equal the base CSV (success, return, N, checkpoint hash): True (12 rows).

Frozen dev10, deterministic argmax, frozen H = 23.1 s, empty R. **Final is the registered primary endpoint**; the 0 / 4096 / 8192 points are pre-registered learning-process evidence only.

## Part 1. Current-profile runs (execution path of prep cac2fa5b / e4dc34ae)

**Current runs: success (n of 10, frozen dev10, deterministic argmax)**

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 0/10 | 0/10 | 0/10 | 10/10 |
| R-TB-E-1 B1-K+E s1 | 0/10 | 0/10 | 0/10 | 10/10 |
| R-TB-DK-1 B2 s1 | 0/10 | 10/10 | 10/10 | 10/10 |
| R-TB-K-1 B1-K s1 | 0/10 | 0/10 | 0/10 | 0/10 |

**Current runs: mean discounted return (frozen H = 23.1 s)**

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 0.000000 | 0.000000 | 0.000000 | 0.489340 |
| R-TB-E-1 B1-K+E s1 | 0.000000 | 0.000000 | 0.000000 | 0.499977 |
| R-TB-DK-1 B2 s1 | 0.000000 | 0.498849 | 0.498849 | 0.498849 |
| R-TB-K-1 B1-K s1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Current runs: actual N / T at each evaluation point** (N = valid skill transitions, T = interaction seconds recorded in the generation manifest; update = update index of the evaluated generation)

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | u0: N 0 / T 0.00 | u4: N 4096 / T 18784.95 | u8: N 8192 / T 38777.10 | u14: N 14512 / T 68813.30 |
| R-TB-E-1 B1-K+E s1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20583.30 | u8: N 8192 / T 40920.85 | u13: N 14309 / T 68820.25 |
| R-TB-DK-1 B2 s1 | u0: N 0 / T 0.00 | u4: N 4096 / T 20565.60 | u8: N 8192 / T 40370.00 | u13: N 14170 / T 68817.10 |
| R-TB-K-1 B1-K s1 | u0: N 0 / T 0.00 | u4: N 4096 / T 18022.50 | u8: N 8192 / T 37574.70 | u14: N 14531 / T 68813.45 |

T at the same N differs between runs because T is the simulated interaction time actually consumed (skill durations differ); N and T are the recorded values, not the nominal 4096 / 8192.

### Training budget actually used

| run | N actual (Ncap 16384) | T actual s (Tcap 68812.8) | stop | complete + fragment updates | optimizer steps | train success episodes | NaN_n | hard_fail |
|---|---|---|---|---|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 14512 | 68813.30 | Tcap | 14 + 1 | 872 | 981 | 0 | None |
| R-TB-E-1 B1-K+E s1 | 14309 | 68820.25 | Tcap | 13 + 1 | 876 | 950 | 0 | None |
| R-TB-DK-1 B2 s1 | 14170 | 68817.10 | Tcap | 13 + 1 | 876 | 1644 | 0 | None |
| R-TB-K-1 B1-K s1 | 14531 | 68813.45 | Tcap | 14 + 1 | 864 | 194 | 0 | None |

### Every evaluation point (long form)

| run | point | update | N actual | T actual (s) | success | mean disc. return | recorded success_seconds mean [FINAL_SKILL_DURATION_ONLY] | exit-reason families | checkpoint (sha256 first 12, recorded) |
|---|---|---|---|---|---|---|---|---|---|
| R-TB-E-0 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (1882e819ac5d) |
| R-TB-E-0 | 4096 | 4 | 4096 | 18784.95 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (9260b4b66522) |
| R-TB-E-0 | 8192 | 8 | 8192 | 38777.10 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 7, "NO_CANDIDATE_SAFE_TERMINATION": 3} | n_008192.pt (461e9b65bd2f) |
| R-TB-E-0 | final | 14 | 14512 | 68813.30 | 10/10 | 0.489340 | 2.915 | {"TASK_SUCCESS": 10} | final_n_014512_u14.pt (f9db3e4ff6dc) |
| R-TB-E-1 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (a83c64bef762) |
| R-TB-E-1 | 4096 | 4 | 4096 | 20583.30 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (c606ebeae8fe) |
| R-TB-E-1 | 8192 | 8 | 8192 | 40920.85 | 0/10 | 0.000000 | NA | {"NO_CANDIDATE_SAFE_TERMINATION": 10} | n_008192.pt (04238a0b116c) |
| R-TB-E-1 | final | 13 | 14309 | 68820.25 | 10/10 | 0.499977 | 2.920 | {"TASK_SUCCESS": 10} | final_n_014309_u13.pt (62099c437dab) |
| R-TB-DK-1 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (916a08b08c29) |
| R-TB-DK-1 | 4096 | 4 | 4096 | 20565.60 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | n_004096.pt (0c8da7c604d4) |
| R-TB-DK-1 | 8192 | 8 | 8192 | 40370.00 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | n_008192.pt (ffb932a0131b) |
| R-TB-DK-1 | final | 13 | 14170 | 68817.10 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | final_n_014170_u13.pt (b1e331c98d49) |
| R-TB-K-1 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (a8e43efd5546) |
| R-TB-K-1 | 4096 | 4 | 4096 | 18022.50 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (3ccec569b000) |
| R-TB-K-1 | 8192 | 8 | 8192 | 37574.70 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_008192.pt (796f28feafca) |
| R-TB-K-1 | final | 14 | 14531 | 68813.45 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | final_n_014531_u14.pt (fa6ca4cf2068) |

## Part 2. Historical seed-0 runs (original execution identity, listed apart, not pooled with Part 1)

R1 B1-K / B2 and R2 B0 keep the N, T and identity they were run with. They are not averaged with, and not ranked against, the Part 1 runs; the unconfirmed items are in `02_historical_identity_and_compatibility.md`.

**Historical runs: success (n of 10, frozen dev10, deterministic argmax)**

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R1-B1K-s0 B1-K s0 | 0/10 | 0/10 | 0/10 | 0/10 |
| R1-B2-s0 B2 s0 | 0/10 | 0/10 | 10/10 | 10/10 |
| R2-B0-s0 B0 s0 | 0/10 | 0/10 | 0/10 | 0/10 |

**Historical runs: mean discounted return (frozen H = 23.1 s)**

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R1-B1K-s0 B1-K s0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| R1-B2-s0 B2 s0 | 0.000000 | 0.000000 | 0.499977 | 0.499977 |
| R2-B0-s0 B0 s0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

**Historical runs: actual N / T at each evaluation point** (N = valid skill transitions, T = interaction seconds recorded in the generation manifest; update = update index of the evaluated generation)

| run | 0 | 4096 | 8192 | final |
|---|---|---|---|---|
| R1-B1K-s0 B1-K s0 | u0: N 0 / T 0.00 | u4: N 4096 / T 20543.40 | u8: N 8192 / T 41451.20 | u13: N 13623 / T 68813.40 |
| R1-B2-s0 B2 s0 | u0: N 0 / T 0.00 | u4: N 4096 / T 19430.05 | u8: N 8192 / T 38697.95 | u14: N 14464 / T 68818.75 |
| R2-B0-s0 B0 s0 | u0: N 0 / T 0.00 | u4: N 4096 / T 18867.30 | u8: N 8192 / T 39081.25 | u14: N 14705 / T 68814.80 |

### Training budget actually used

| run | N actual (Ncap 16384) | T actual s (Tcap 68812.8) | stop | complete + fragment updates | optimizer steps | train success episodes | NaN_n | hard_fail |
|---|---|---|---|---|---|---|---|---|
| R1-B1K-s0 B1-K s0 | 13623 | 68813.40 | Tcap | 13 + 1 | 812 | 206 | 0 | None |
| R1-B2-s0 B2 s0 | 14464 | 68818.75 | Tcap | 14 + 1 | 892 | 1379 | 0 | None |
| R2-B0-s0 B0 s0 | 14705 | 68814.80 | Tcap | 14 + 1 | 876 | 187 | 0 | None |

### Every evaluation point (long form)

| run | point | update | N actual | T actual (s) | success | mean disc. return | recorded success_seconds mean [FINAL_SKILL_DURATION_ONLY] | exit-reason families | checkpoint (sha256 first 12, recorded) |
|---|---|---|---|---|---|---|---|---|---|
| R1-B1K-s0 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (bc2aaa3425b2) |
| R1-B1K-s0 | 4096 | 4 | 4096 | 20543.40 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (44311905d0de) |
| R1-B1K-s0 | 8192 | 8 | 8192 | 41451.20 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_008192.pt (f313b73f653a) |
| R1-B1K-s0 | final | 13 | 13623 | 68813.40 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | final_n_013623_u13.pt (2c93d96f3999) |
| R1-B2-s0 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (8f4d0b6048db) |
| R1-B2-s0 | 4096 | 4 | 4096 | 19430.05 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 8, "NO_CANDIDATE_SAFE_TERMINATION": 2} | n_004096.pt (3e376237f3e5) |
| R1-B2-s0 | 8192 | 8 | 8192 | 38697.95 | 10/10 | 0.499977 | 2.920 | {"TASK_SUCCESS": 10} | n_008192.pt (f7c62ff104cd) |
| R1-B2-s0 | final | 14 | 14464 | 68818.75 | 10/10 | 0.499977 | 2.920 | {"TASK_SUCCESS": 10} | final_n_014464_u14.pt (976f70102ee9) |
| R2-B0-s0 | 0 | 0 | 0 | 0.00 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (e8b463aec197) |
| R2-B0-s0 | 4096 | 4 | 4096 | 18867.30 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (7b18c9b66361) |
| R2-B0-s0 | 8192 | 8 | 8192 | 39081.25 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_008192.pt (4b856fc7b95a) |
| R2-B0-s0 | final | 14 | 14705 | 68814.80 | 0/10 | 0.000000 | NA | {"NO_CANDIDATE_SAFE_TERMINATION": 10} | final_n_014705_u14.pt (f50a857ce9b5) |

## Part 3. What may be said

- B2 seed 1 (R-TB-DK-1) was 10/10 on frozen dev10 at N = 4096 and N = 8192 (and at final).
- B1-K+E seed 0 (R-TB-E-0) and seed 1 (R-TB-E-1) were 0/10 at N = 4096 and N = 8192 and 10/10 at final.
- B1-K seed 1 (R-TB-K-1) was 0/10 at all four evaluation points.
- Historical B2 seed 0 (R1, original execution identity, listed apart) was 0/10 at N = 4096 and 10/10 at N = 8192 and at final.
- Historical B1-K seed 0 (R1) and B0 seed 0 (R2) were 0/10 at all four evaluation points.

## Part 4. What is not said

<!-- NOT_CLAIMED_BEGIN -->
- No learning-speed ratio or multiple is stated.
- No statistical-significance statement is made (one or two seeds per method, ten dev cases).
- No statement of method equivalence and none of general superiority of any method is made.
- The 4096 / 8192 points are pre-registered learning-process evidence on dev10; the registered primary endpoint stays the final point.
- No change to Method 2.1.1, the training configuration, the profile or any budget follows from these numbers.
<!-- NOT_CLAIMED_END -->

## Caveats

- C1 `success_seconds` is the duration of the final successful skill only (label FINAL_SKILL_DURATION_ONLY); the episode-level time to success is UNVERIFIED in all existing eval rows (code: `stage2a_v11.py` line 1044 falls back to `t.duration` because `rl.Snapshot` has `clock_seconds`, not `elapsed_seconds`). The frozen discounted return is the time-sensitive quantity here. The revision design is in `04_evaluation_record_revision_design.md`.
- C2 Scope: dev10 only; one or two training seeds per method; the three Part 1 base runs and R-TB-E-1 ran on a shared server with 1-3 concurrent workers, so wall-clock is not compared here.
- C3 Selection columns (job-rule selected checkpoint, common-dev20 selection) are kept in `03_frozen_model_list.md` and are not used in any table above.
- C4 Exit-reason families are copied from the eval rows (reason text before the first colon); no mechanism is inferred.
