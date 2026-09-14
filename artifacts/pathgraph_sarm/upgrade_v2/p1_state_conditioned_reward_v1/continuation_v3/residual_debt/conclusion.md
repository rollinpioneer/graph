# Residual debt

Normalized no-progress cycles accumulate D_k ≈ k/6 inside one episode.
Matched lawful suffix `in_transit→dropped→recovery→grasped→in_transit→placed→success` has unchanged per-step signed rewards for preload k=0/1/3/8 under FULL_FROZEN.
Conclusion: residual debt exists, but the tested lawful suffix reward did not change.
This is not proof that residual debt is harmless in general, and it is not a reason to zero debt at recovery.
matched_suffix_changed=False

Extra reachable skeletons (not scored as new physics): partial recovery stop, recover-then-terminal-failure, recover-then-fail-again, success-then-hold.
