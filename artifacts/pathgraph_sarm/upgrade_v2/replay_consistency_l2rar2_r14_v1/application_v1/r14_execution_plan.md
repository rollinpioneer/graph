# R14 minimal replay execution plan (not authorised)

## Execution 1 - ordinary replay

Reuse the locked case program, variant, simulator version and callback
semantics.  Persist state at: initial-after-forward, before-action,
control callback, action-end callback, after-perform-return.  Every
sample keeps its own capture_order and sequence.

Fields: qpos, qvel, qacc_warmstart, mocap_pos, mocap_quat, eq_active,
model.eq_data, RNG state/hash, contact pair summary, renderer callback
count, action/control/event rows, object/gripper geometry.

Compare: cache actions/controls/events vs ordinary, and cache
action-end vs ordinary action-end callback state, with rtol=0 and the
frozen atolerances.  Any main-gate failure stops the run
(`STOP_AFTER_EXECUTION_1`); execution 2 is then not run.

## Execution 2 - instrumented replay

Only after execution 1 passes and one instance remains.  Same renderer,
actions and callbacks; adds pre-reviewed read-only recording only.

## Explicit non-goals

R14 verifies replay consistency only.  It does not verify K4/K5/K6
physical loss, does not rank recovery candidates and does not open L3.
