# Stage 6R1 reporting patch note

1. The evidence field `reward_retuned_after_test=false` is the authoritative value.
2. The decision check named `reward_retuned_after_test=true` means that the no-post-test-retuning check passed; Stage 7 reports use `no_post_test_reward_retune=true` to remove this ambiguity.
3. The original M4 checksum list contains an omitted large rollout table. Stage 7 creates a separate portable manifest and does not rewrite the frozen source list.
4. These reporting corrections do not change experiment results, thresholds, or decisions.
