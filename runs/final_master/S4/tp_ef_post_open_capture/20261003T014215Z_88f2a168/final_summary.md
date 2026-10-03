# CP-DISR-TP-EF-POST-OPEN-CAPTURE-1 — final summary

Outcome: **STOPPED / FAIL**  (stop reason: STOPPED_RESTORE_FAIL:T_B_dev_03; caps exceeded: none)

## T_B_dev_03
- setup: PASS; OPEN exit NORMAL_TERMINATION; facts {'p:AtBuffer:second_object:buffer': 'FALSE', 'p:AtBuffer:target:buffer': 'FALSE', 'p:GripperEmpty': 'TRUE', 'p:Held:second_object': 'FALSE', 'p:Held:target': 'FALSE', 'p:Inside:second_object:container': 'FALSE', 'p:Inside:target:container': 'FALSE', 'p:OnTable:second_object': 'TRUE', 'p:OnTable:target': 'TRUE', 'p:Open:container': 'TRUE'}; masks {'a:PICK:second_object:v1': True, 'a:PICK:target:v1': True}; failures []
- fresh-process restore: FAIL; failures ['forced_render_obs']
  - clock: pass=True
  - controller_gripper_state: pass=True
  - facts: pass=True
  - facts_forced_render: pass=True
  - forced_render_obs: pass=False
  - goal_candidates_mask: pass=True
  - identity: pass=None
  - no_skill_executed: pass=True
  - post_open_requirements: pass=True
  - public_obs_cached: pass=True
  - rng: pass=True
  - sim_arrays: pass=True
  - sim_arrays_raw: pass=True

## T_B_dev_05
- setup: NOT_ATTEMPTED

## Budget (as used)

- used: {'continuation_skills': 0, 'environment_constructions': 2, 'fresh_restore_validations': 1, 'open_skill_calls': 1, 'optimizer_steps': 0, 'provider_requests': 0, 'rl_transitions': 0, 'setup_episodes': 1, 'start_case_calls': 2, 'test_episodes': 0}
- low-level env.reset() invocations including create_task_runtime bootstrap resets: 4 (start_case-level resets equal the start_case count; each process makes one real construction and reuses it for start_case)
- processes: [{'case': 'T_B_dev_03', 'cmd': 'setup', 'constructions': 1, 'env_reset_calls': 2, 'exit': 0, 'skills': 1, 'start_case': 1, 'status': 'PASS'}, {'case': 'T_B_dev_03', 'cmd': 'restore', 'constructions': 1, 'env_reset_calls': 2, 'exit': 0, 'skills': 0, 'start_case': 1, 'status': 'FAIL'}]
