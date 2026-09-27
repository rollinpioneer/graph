# CP-DISR v1.3 Results draft

## Scope
R2 adds exactly one new formal RL run, T_B B0 seed 0. R1 B1-K/B2 training and evaluations are reused read-only. No VLM, final test, extra seed, T_C prototype, or R1 re-evaluation was added.

## Training boundary
B0 stopped at the first authorized boundary: Tcap=68812.8 simulation seconds, with actual N=14705 transitions, T=68814.8, 14 complete PPO updates plus one fragment update, and 876 optimizer steps. It did not reach N=16384.

## Development evaluation
B0 dev10 was evaluated at N=0, 4096, 8192, and the actual final N=14705; all were 0/10 with mean starting discounted return 0.0. The frozen B0 checkpoint rule selected N=0 because all observed B0 dev10 windows tied at zero (`max_mean3_then_max_worst_then_earlier_N`). Exactly one B0 common dev20 was then run: 0/20, return 0.0. This is a single-seed, post-selection development result, not a test result.

## Three-method context
At common measured N=8192, reused R1 results are B1-K 0/10 and B2 10/10 (mean return 0.4999766845), while B0 is 0/10. Endpoint N differs across methods: B1-K 13623, B2 14464, B0 14705. These are not equal-budget or test-set claims.

All complete-task durations that could not be reconstructed from independent episode-start-to-first-success intervals are recorded as NA; terminal skill duration is never substituted.
