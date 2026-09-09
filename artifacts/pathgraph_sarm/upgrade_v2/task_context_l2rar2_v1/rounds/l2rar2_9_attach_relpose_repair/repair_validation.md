## Material Passport

- Artifact type: versioned attach-relpose repair collection validation
- Verification status: `ANALYZED`; development validation only
- Current boundary: `L2RAR2_PARTIAL_KEEP_G1`; L3 closed

# Attach-relpose repair validation

- Collection: `4` new root families x `8` frozen cases = `32` rollouts.
- Reference result: `32/32` labeled; unresolved rows retained: `0`.
- Frozen reference limit remains `0.02 m`; no label or threshold change was applied.
- The only simulator change is attach-time weld relative pose initialization: body2=`object` is stored in body1=`gripper`'s local frame as `R_gripper_world.T @ (p_object_world - p_gripper_world)` and `q_gripper_world^-1 * q_object_world`; snapshots restore `model.eq_data`.

## Fixed candidate comparison

- Compared only `B_count2` and existing `C3_vector_rho035` on this new collection through `M1_requested_effect_gate`; no parameter search or rename was used.
- Decision: `PROCEED_TO_ONLINE_INTERFACE_REPAIR`.
- If collection passes, the next independently gated step is the online interface repair: distinguish data missing/invalid, no current hold evidence, and established historical hold evidence. This validation does not apply that repair.

## Accounting

- New data rollouts are counted as development validation, not confirmation. Training jobs: `0`; API calls: `0`; API keys read: `false`; L3: closed.
