# CP-DISR-TP-EF-POST-OPEN-RESTORE-R2 — final summary

Outcome: **ALL R2 CONDITIONS PASS** (stop: none; caps exceeded: none)

Renderer contract applicable: True (problems: [])

## Stage A dev_03 restore
- status PASS; failures []
  - camera_live_vs_frozen: pass=True
  - clock: pass=True
  - controller_gripper_state: pass=True
  - facts_cached_obs: pass=True
  - goal_candidates_mask: pass=True
  - no_skill_executed: pass=True
  - post_open_requirements: pass=True
  - public_obs_cached_exact: pass=True
  - render_A: pass=True
  - render_B: pass=True
  - rng: pass=True
  - sim_arrays: pass=True
  - sim_arrays_raw: pass=True
  - A_saved_forced_vs_restore_render_1: rgb max_abs=1 changed_px=2 (0.0001220703125), depth_raw max=0.0, depth_metric max=0.0, proprio max=0.0, perception pass=True, facts={'restore1_vs_manifest_forced_equal': True, 'saved_vs_restore1_equal': True}
  - B_restore_render_1_vs_render_2: rgb max_abs=1 changed_px=2 (0.0001220703125), depth_raw max=0.0, depth_metric max=0.0, proprio max=0.0, perception pass=True, facts={'restore1_vs_restore2_equal': True}

## dev_05 restore
- status PASS; failures []
  - camera_live_vs_frozen: pass=True
  - clock: pass=True
  - controller_gripper_state: pass=True
  - facts_cached_obs: pass=True
  - goal_candidates_mask: pass=True
  - no_skill_executed: pass=True
  - post_open_requirements: pass=True
  - public_obs_cached_exact: pass=True
  - render_A: pass=True
  - render_B: pass=True
  - rng: pass=True
  - sim_arrays: pass=True
  - sim_arrays_raw: pass=True
  - A_saved_forced_vs_restore_render_1: rgb max_abs=0 changed_px=0 (0.0), depth_raw max=0.0, depth_metric max=0.0, proprio max=0.0, perception pass=True, facts={'restore1_vs_manifest_forced_equal': True, 'saved_vs_restore1_equal': True}
  - B_restore_render_1_vs_render_2: rgb max_abs=0 changed_px=0 (0.0), depth_raw max=0.0, depth_metric max=0.0, proprio max=0.0, perception pass=True, facts={'restore1_vs_restore2_equal': True}

## dev_05 capture
- PASS; facts {'p:AtBuffer:second_object:buffer': 'FALSE', 'p:AtBuffer:target:buffer': 'FALSE', 'p:GripperEmpty': 'TRUE', 'p:Held:second_object': 'FALSE', 'p:Held:target': 'FALSE', 'p:Inside:second_object:container': 'FALSE', 'p:Inside:target:container': 'FALSE', 'p:OnTable:second_object': 'TRUE', 'p:OnTable:target': 'TRUE', 'p:Open:container': 'TRUE'}; masks {'a:PICK:second_object:v1': True, 'a:PICK:target:v1': True}; failures []

## Budget
- used: {'continuation_skills': 0, 'dev03_restore_validations': 1, 'dev05_open_skill_calls': 1, 'dev05_restore_validations': 1, 'dev05_setup_episodes': 1, 'environment_constructions': 3, 'optimizer_steps': 0, 'provider_requests': 0, 'rl_transitions': 0, 'skill_retries': 0, 'start_case_calls': 3, 'test_episodes': 0}
- env.reset() invocations incl. bootstrap resets: 6
- processes: [{'constructions': 1, 'env_reset_calls': 2, 'kind': 'dev03_restore', 'skills': 0, 'start_case': 1, 'status': 'PASS'}, {'constructions': 1, 'env_reset_calls': 2, 'kind': 'dev05_setup', 'skills': 1, 'start_case': 1, 'status': 'PASS'}, {'constructions': 1, 'env_reset_calls': 2, 'kind': 'dev05_restore', 'skills': 0, 'start_case': 1, 'status': 'PASS'}]
