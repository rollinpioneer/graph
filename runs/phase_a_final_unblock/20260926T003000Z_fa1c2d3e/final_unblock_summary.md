# CP-DISR v1.2 Phase A final unblock

`phase_a_e16_ready=true` and `phase_a_status=NOT_STARTED`.

DashScope 1.27.6 was installed in the target Python with fixed dependencies and `pip check` is clean. The no-key provider dispatch test passed without reading an API key or making a request.

Production candidate features now use versioned SHA-256 encoding `cp_disr_candidate_feature_v1`; built-in process-random hashing was removed from SnapshotBuilder. Two same-process constructions and three fresh processes under different `PYTHONHASHSEED` values matched candidate IDs/order/mask/features/base input/graph/prior. Older training runs are not claimed bitwise comparable.

T_C semantic reset trials covered non-empty-prior train case `T_C_train_01` and legal empty-prior dev case `T_C_dev_00`: 10 same-process plus 3 fresh-process trials each. State, camera and depth were exact. Raw RGB was intentionally not relabeled exact: hashes varied with a bounded one-LSB raster difference, at most 1 pixel in this capture set (prior evidence retained 1-3 pixels); perception masks/measurements, verifier facts, last_perception, snapshots, Full/B2 outputs and deterministic selected actions were identical. Classification: `RAW_RGB_LSB_VARIANCE_NO_DOWNSTREAM_EFFECT`.

Targeted clock/deadline passed 31, pure/torch_runtime passed 86 with no critical skips, and the complete suite passed 121 with zero failures/collection errors. Persistence regression passed for two generations, fresh-process loading, raw event/CSV preservation and before_latest failure recovery.

One bounded real no-learning decision attempt was completed for each fresh Full and B2 policy on `T_C_train_01`; both had finite valid outputs and normal controller/evaluator records. B2 effective prior, DP, uP and Delta were exactly zero. No optimizer/PPO update occurred.

No E16, formal RL, VLM, test-ID or final-test episode was started.
