## Table A — one row per run (existing evaluations only; no episode was run for this table)

Eval points are the pre-registered 0 / 4096 / 8192 / final (Plan v3 §7.1). Success is n of 10 dev10 cases, deterministic argmax. "Frozen discounted return" = recorded `mean_discounted_return` of the eval file (frozen H = 23.1 s). "Final-skill duration" = the recorded `success_seconds` field (see caveat C1); NA = no successful case.

| run | success n/10 at 0 / 4096 / 8192 / final | mean discounted return at 0 / 4096 / 8192 / final | recorded success_seconds (mean over successes) at 0 / 4096 / 8192 / final |
|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 0 / 0 / 0 / 10 | 0.0000 / 0.0000 / 0.0000 / 0.4893 | NA / NA / NA / 2.915 |
| R-TB-DK-1 B2 s1 | 0 / 10 / 10 / 10 | 0.0000 / 0.4988 / 0.4988 / 0.4988 | NA / 3.540 / 3.540 / 3.540 |
| R-TB-K-1 B1-K s1 | 0 / 0 / 0 / 0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | NA / NA / NA / NA |

## Table B — actual budget and compute (job_summary / launch_state; training only)

| run | N actual (Ncap 16384) | T actual s (Tcap 68812.8) | stop | complete + fragment updates | optimizer steps | train success episodes | wall h (start -> finish UTC) | PPO min (sum, complete updates) | collection min (sum, between updates) | first-update profile (ppo s / transitions per wall s) |
|---|---|---|---|---|---|---|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 14512 | 68813.30 | Tcap | 14 + 1 | 872 | 981 | 8.89 (10-01T12:01 -> 10-01T20:55) | 84.5 | 439.4 | 351 / 0.529 |
| R-TB-DK-1 B2 s1 | 14170 | 68817.10 | Tcap | 13 + 1 | 876 | 1644 | 11.88 (10-01T12:40 -> 10-02T00:33) | 230.5 | 444.6 | 1624 / 0.278 |
| R-TB-K-1 B1-K s1 | 14531 | 68813.45 | Tcap | 14 + 1 | 864 | 194 | 7.93 (10-01T13:59 -> 10-01T21:54) | 52.2 | 414.0 | 250 / 0.415 |

All three: NaN_n 0, hard_fail null, HEAD cac2fa5b3. Compute columns are wall clock on a shared 112-core server with shared GPUs and 1-3 concurrent workers (caveat C2).

## Table C — every eval point (from `eval_n_*.json` / `eval_final.json`)

| run | eval file | update / N | success | mean disc. return | recorded success_seconds mean | failure reasons | checkpoint (sha256 first 12) |
|---|---|---|---|---|---|---|---|
| R-TB-E-0 | eval_n_000000.json | 0 / 0 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (1882e819ac5d) |
| R-TB-E-0 | eval_n_004096.json | 4 / 4096 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (9260b4b66522) |
| R-TB-E-0 | eval_n_008192.json | 8 / 8192 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 7, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-2": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-4": 1, "NO_CANDIDATE_SAFE_TERMINATION:d0-env-0:ep-6": 1} | n_008192.pt (461e9b65bd2f) |
| R-TB-E-0 | eval_final.json | 14 / 14512 | 10/10 | 0.489340 | 2.915 | {"TASK_SUCCESS": 10} | final_n_014512_u14.pt (f9db3e4ff6dc) |
| R-TB-DK-1 | eval_n_000000.json | 0 / 0 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (916a08b08c29) |
| R-TB-DK-1 | eval_n_004096.json | 4 / 4096 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | n_004096.pt (0c8da7c604d4) |
| R-TB-DK-1 | eval_n_008192.json | 8 / 8192 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | n_008192.pt (ffb932a0131b) |
| R-TB-DK-1 | eval_final.json | 13 / 14170 | 10/10 | 0.498849 | 3.540 | {"TASK_SUCCESS": 10} | final_n_014170_u13.pt (b1e331c98d49) |
| R-TB-K-1 | eval_n_000000.json | 0 / 0 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_000000.pt (a8e43efd5546) |
| R-TB-K-1 | eval_n_004096.json | 4 / 4096 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_004096.pt (3ccec569b000) |
| R-TB-K-1 | eval_n_008192.json | 8 / 8192 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | n_008192.pt (796f28feafca) |
| R-TB-K-1 | eval_final.json | 14 / 14531 | 0/10 | 0.000000 | NA | {"INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE": 10} | final_n_014531_u14.pt (fa6ca4cf2068) |

### Caveats
- **C1 `success_seconds`.** In `src/cp_disr/stage2a_v11.py` line 1044 the field is `snapshot.elapsed_seconds + t.duration` if the snapshot has `elapsed_seconds`, else `t.duration`. `rl.Snapshot` has `clock_seconds` and no `elapsed_seconds`, so the recorded value is the duration of the final (successful) skill only (it equals single-skill durations such as 2.90 s / 3.55 s). Total episode time to success is NOT recorded in the eval rows -> UNVERIFIED; it is not reconstructed here. The discounted return G is recorded and frozen (H = 23.1 s) and is the time-sensitive quantity.
- **C2 compute.** Not a controlled compute-cost measurement: concurrency differed (E-0 alone ~38 min, then 2, then 3 workers until E-0 finished at 20:55Z; DK-1 ran with 1-3 workers) and GPU 0 hosted other users' jobs.
- **C3 scope.** dev10 only (10 cases, seed per run as registered: E-0 seed 0, DK-1 seed 1, K-1 seed 1), the registered final point is the primary endpoint; no best-dev checkpoint substitutes for it. The job_summary `selected_checkpoint` (E-0 final, DK-1 final, K-1 n_000000 by the max_mean3 -> max_worst -> earlier-N rule) is listed in the JSON only and is not used here.
- **C4 failure reasons** are copied from the eval rows: all zero-success points of E-0 (0/4096) and K-1 (all four) end with `INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE`; E-0 at 8192 has 7 of that and 3 `NO_CANDIDATE_SAFE_TERMINATION`. No mechanism is inferred.
