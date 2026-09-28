# S0 implementation changes

## Production source

- `src/cp_disr/neural.py`: registered `B1-K+E` and `A_STAT`; added candidate-local `EffectTokenReadout`; preserved B1-K successor-free behavior; implemented A_STAT as `delta.zh - delta.zk`; added diagnostics for effect-token count and static-relation input.
- `src/cp_disr/torch_rl.py`: added non-mutating global gradient norm measurement and formal PPO pre/post clipping log fields.

## Tests

- `tests/test_s0_final_variants.py`: added production tests for B1-K+E permissions, A_STAT/Full null equivalence, Full empty-patch null, bounds, candidate reorder, A_STAT/A_CAT shared properties and clipping logs.

No training entrypoint, Stage 2A registry, method version, provider configuration, or historical run was changed. The new variants are implemented in the production neural component but are not added to the frozen Stage 2A job matrix; no S0 authorization was inferred for training them.

## B_PLAN binding and development

- src/cp_disr/baselines/b_plan.py: added the production finite-cost symbolic search and replanning baseline with frozen S0 budgets, canonical tie-break, independent reference median skill cost, layered relaxed-step heuristic, explicit search outcome counters, relation-cache exclusion, and public-input exclusions.
- src/cp_disr/cli.py: added b-plan-dev, bound to the shared T_B RuntimeFactory and development split.
- tests/test_b_plan.py: added 5 unit tests for chain planning, canonical tie-break, negative goals, UNKNOWN facts, depth and node budgets.
- runs/final_master/S0/60c26621/b_plan_dev/: recorded the 10 authorized T_B development episodes and raw decisions.
