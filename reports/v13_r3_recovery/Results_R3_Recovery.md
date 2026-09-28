# CP-DISR v1.3 R3-Recovery: matched N=2048 and Full prior switch

## Scope and identity

This recovery ran only as `xushijie2@gpu03` with `/home/xushijie2/envs/lerobotpi0-xfs/bin/python` in the isolated recovery worktree. It did not resume or retrain Full, and it did not touch B0/R2.

## Full durable boundary

The preserved Full checkpoint `n_002048.pt` was verified against SHA-256 `4b5f2a873fed2052db30ade97c4bc08e06f298fec8674db3d02a7f7f3309021c`; `update_complete_2.pt` was verified as `f9ff429ffd3db955b38d2b9e47a5b83800238609957a1f496953e721465cd4be`. The authoritative Full boundary is N=2048, T=9771.400000000713, two complete updates.

## Matched B2 recovery

B2 was initialized from fresh model/Adam/RNG/environment state with empty effective prior and strict DP/uP/Delta zero. Real Tcap-first stopping occurred at N=1935 and T=9319.000000000771, so N=2048 and a second complete update were not reached. There was one complete 1024-transition update and one terminal 911-transition fragment; no extension was attempted. N=0 and the actual final-N dev10 evaluations are persisted. The first-update self-check passed (1024 transitions, four epoch log rows, finite metrics, parameter L2 change 3.462288303750047, generation sidecars present).

## Full prior switch

The same Full N=2048 checkpoint was evaluated with an empty prior for the frozen `S_dev=[D0_dev_00]` case and the four frozen S_train cases under both `Full@Original` and `Full@Absent`. Full@Original dev10 was reused from the original persisted evaluation; it was not rerun. Full@Absent covered all 10 dev10 cases (10/10 success; mean start discounted return 0.6350014580965666). The frozen nonempty-prior stratum remains D0_dev_00. S_train success was 3/4 under both conditions; DP/Delta opportunity and nonzero witnesses were zero in these four cases. Optional same-state shadow forward was not run, therefore action-flip denominator and TV are explicitly NA. Full@Absent is an inference-time switch, not B2 training.

## Stall reconciliation

The old Full tail is classified `PPO_INTERNAL_STALL` with high confidence because the frozen training log contains the PPO update line at N=3072 immediately before the no-transition window. N=3072 is not treated as a completed update; the durable endpoint remains N=2048. No long reproduction or Full resume was performed.

## Claims and limits

This is a matched-boundary recovery record, not an E8 completion. No test/VLM/extra seed/T_C/A_CAT/A_Q/A_B activity occurred.
