# Stage 1A-P0 summary

status: PASS

Stage 1A training status remains NOT_STARTED.
readiness.stage_1a_runtime_ready: True

## Binding result

- D0 Action registry: OPEN, PICK, PLACE, PLACE_BUFFER
- MOVE: REJECTED_FOR_D0; not a PLACE_BUFFER synonym; SafetyManager rejects MOVE
- Controller qualification: {"a:OPEN:container:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PICK:target:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PICK:second_object:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PLACE:target:container:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PLACE:second_object:container:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PLACE_BUFFER:target:buffer:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}, "a:PLACE_BUFFER:second_object:buffer:v1": {"verified_success": 5, "attempts": 5, "wrong_object": 0, "nan": 0, "pass": true}}
- Verifier: allowed RGB-D / gripper / EEF / public_layout only; hidden-state leakage=false
- TaskEvaluator: independent hidden-truth goal check; evaluator unit status=PASS
- Time unit token: seconds
- Clock source: mujoco_sim_data_time
- skill timeouts (s): {"OPEN": 16.0, "PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}
- task deadline (s): 43.0
- d_ref (median verified-success duration, s): 3.5500000000006953
- D0 split: train 64 / dev 10
- D0 caches: train 64 / extra Stage 1A dev 10
- Stage 0C frozen dev caches: 24, fingerprint unchanged
- scripted dev10: all 10 succeeded, no repeat reward
- B2 dry-run transition: {"duration": 4.749999999999588, "reward": 0.0, "terminated": false, "truncated": false, "reason": "CONTINUE", "action": null}
- Full dry-run transition: {"duration": 4.2999999999996374, "reward": 0.0, "terminated": false, "truncated": false, "reason": "CONTINUE", "action": null}
- PPO updates: none
- Stage 0B regression: returncode 0 (40 passed)

## Still unbound

T_A, T_B, T_C, T_D, T_E remain MUST_BIND. This round bound D0 runtime only.

## Notes

- runtime.actual_interaction_time_unit is the required token seconds. MuJoCo sim.data.time is recorded separately as actual_clock_source=mujoco_sim_data_time.
- Stage 0C 24 caches were not regenerated or edited.
- New D0 caches live under new keys in experiments/vlm_cache/{train,dev}.
