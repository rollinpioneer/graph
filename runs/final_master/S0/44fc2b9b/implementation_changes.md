# S0 implementation changes

## Production source

- `src/cp_disr/neural.py`: registered `B1-K+E` and `A_STAT`; added candidate-local `EffectTokenReadout`; preserved B1-K successor-free behavior; implemented A_STAT as `delta.zh - delta.zk`; added diagnostics for effect-token count and static-relation input.
- `src/cp_disr/torch_rl.py`: added non-mutating global gradient norm measurement and formal PPO pre/post clipping log fields.

## Tests

- `tests/test_s0_final_variants.py`: added production tests for B1-K+E permissions, A_STAT/Full null equivalence, Full empty-patch null, bounds, candidate reorder, A_STAT/A_CAT shared properties and clipping logs.

No training entrypoint, Stage 2A registry, method version, provider configuration, or historical run was changed. The new variants are implemented in the production neural component but are not added to the frozen Stage 2A job matrix; no S0 authorization was inferred for training them.
