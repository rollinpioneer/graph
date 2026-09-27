# CP-DISR v1.3 R1: first development comparison at N=4096

Status: BOTH_DEV10_COMPLETE. This is a single-seed development observation, not an independent test result.

Training source: `0eb3776d143d2142eb2893e2a12a4d85f2d14fc1`.
Frozen common dev10 split SHA-256: `b16130d209c107a200149cc5a62e4e300f21a9cd962a69f65834926630a9fa079`.
Case order: `T_B_dev_00, T_B_dev_03, T_B_dev_04, T_B_dev_08, T_B_dev_09, T_B_dev_11, T_B_dev_12, T_B_dev_14, T_B_dev_15, T_B_dev_19`.

| Method | N=0 dev10 success | N=4096 dev10 success | N=4096 mean discounted return | Train T at N=4096 (s) | PPO optimizer steps | DK RMS | DH RMS | DP RMS | Delta RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1-K | 0/10 | 0/10 | 0.0 | 20543.40000000682 | 244 | 0.0 | 0.0 | 0.0 | 0.0 |
| B2 | 0/10 | 0/10 | 0.0 | 19430.050000006013 | 248 | 0.10480744529704678 | 0.10480744529704678 | 0.0 | 0.0 |

At this checkpoint, dev10 success and mean discounted return are tied at zero. This does not establish method equivalence. B2 shows nonzero DK/DH diagnostic activity; DP/Delta are exactly zero for both methods as required by the no-prior profile. The DP opportunity denominator is zero for both, so conditioned rates are N/A.

The two evaluations each contain 10 episodes in the same frozen case order, with isolated evaluation RNG and zero evaluation optimizer steps. Training continues to the original N/T/update caps.

Evidence:

- `runs/v13_r1/T_B/B1-K/seed_0/20260926T112613Z_9e64dd09/eval_n_004096.json`
- `runs/v13_r1/T_B/B2/seed_0/20260926T112613Z_9e64dd09/eval_n_004096.json`
