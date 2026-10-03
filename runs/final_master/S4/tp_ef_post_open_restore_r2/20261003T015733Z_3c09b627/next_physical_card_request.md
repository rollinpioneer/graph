# Next physical card request — CP-DISR-TP-EF-POST-OPEN-RESTORE-R2

dev_03 restore, dev_05 capture and dev_05 restore all passed under the frozen R2 criteria. Requested next card: **8 fixed-script physical branches** = 2 states x 2 candidates x 2 repeats. Not started by this card.

## Frozen protocol (no stepwise B_PLAN replanning)

- States: T_B_dev_03 (snapshot /home/xushijie2/graph_cp_disr_snapshots/tp_ef_post_open_capture/20261003T014215Z/T_B_dev_03) and T_B_dev_05 (R2 capture; paths in post_open snapshot records). Each branch restores in a fresh process on the same physical GPU; OPEN is never replayed.
- T: PICK(target) -> PLACE(target, container) -> PICK(second_object) -> PLACE_BUFFER(second_object, buffer).
- S: PICK(second_object) -> PLACE_BUFFER(second_object, buffer) -> PICK(target) -> PLACE(target, container).
- Repeats: 2 per (state, candidate), each in its own fresh process from identical snapshot bytes; branch ids pre-registered.

## To freeze before the first branch

- Thresholds: the R2 restore criteria (rgb <= 3 uint8, depth 1e-6, proprio 1e-9, sim arrays 1e-12, facts/mask/goal/controller/RNG/clock exact) apply to the restore gate of every branch; a failed restore gate stops the card.
- epsilon rule: max(floor, within-candidate repeat range), floors dG = 0.02, dT = 2.1 s, computed from repeats before cross-candidate comparison; G = 2^(-tau/H), H = 23.1 s, deadline 60 s on the restored clock.
- Logging: facts, mask, controller exit after every skill; per-branch restore metrics.
- Budget: 8 branches, 8 fresh-process constructions, <= 4 skills per branch, provider 0, RL/optimizer 0, test 0.

## Evidence in this card

- budget used: {'continuation_skills': 0, 'dev03_restore_validations': 1, 'dev05_open_skill_calls': 1, 'dev05_restore_validations': 1, 'dev05_setup_episodes': 1, 'environment_constructions': 3, 'optimizer_steps': 0, 'provider_requests': 0, 'rl_transitions': 0, 'skill_retries': 0, 'start_case_calls': 3, 'test_episodes': 0}
