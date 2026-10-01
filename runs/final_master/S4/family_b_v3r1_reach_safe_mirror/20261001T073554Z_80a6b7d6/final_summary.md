# Family B V3-R1 reach-safe mirror pilot

- Card CP-DISR-S4-FAMILY-B-V3-R1-REACH-SAFE-MIRROR-PILOT-1; frozen profile `FAMILY_B_PUBLIC_OBS_V2` 3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917 (unchanged).
- Gate 0: GATE0_PASS. Waves: A=NOT_EVALUATED, B=NOT_EVALUATED, C=NOT_EVALUATED; run state STOPPED (step A1 did not meet its technical/qualification gate).
- Attempts reserved/started/terminal: 1/1/1 (cap 8); constructions 1; explicit resets 1; internal resets 2; live skill calls 1/48.
- Provider / representation / RL / optimizer / elastic / formal test = 0; S2/S3/TP training = false.
- Mechanism: **FAMILY_B_V3_R1_REACH_SAFE_MECHANISM_PILOT_NOT_ESTABLISHED**; technical: STOPPED_BEFORE_ALL_BRANCHES_RAN; next action `STOP_TP_TASK_SEARCH_AND_ADJUST_CLAIMS`.

| pair | C(U) | C(V) | delta U-V | PLACE_BUFFER U/V (s) | steps U/V | match |
|---|---|---|---|---|---|---|
| layout_0/B_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_0/C_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_1/B_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_1/C_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |

Conditions: 10_layout_0_preference_reversal=False, 11_layout_1_preference_reversal=False, 12_same_object_best_pad_changes_with_layout=False, 13_neither_always_u_nor_always_v_reaches_4_of_4=False, 14_provider_representation_rl_optimizer_zero=True, 15_old_evidence_and_frozen_config_unchanged=True, 1_8_of_8_task_success=False, 2_8_of_8_complete_records=False, 3_four_pair_restores_pass=False, 4_place_buffer_duration_diff_le_0.15=False, 5_controller_steps_diff_le_3=False, 6_public_depth_support_ge_12_initial_and_post_candidate=False, 7_no_noplan_timeout_exception_or_recorder_error=False, 8_candidates_equally_legal=False, 9_plan_length_and_skill_multiset_match=False
- Measured winners: {}; predicted: {'layout_0/B_PENDING': 'pad_u', 'layout_0/C_PENDING': 'pad_v', 'layout_1/B_PENDING': 'pad_v', 'layout_1/C_PENDING': 'pad_u'}.
- Always-U 0/4, always-V 0/4; state-conditioned reversal False.
- No significance or generalization claim; speedup NOT_MEASURED.

## Stop reason (A1)

- The reach fix worked: the first setup action `a:PICK:obj_c` (obj_c at A_V, 0.610 m from the base) ran to NORMAL_TERMINATION and the arm reached the show pose; the reset-time public depth support was carrier 40 and setup target 52 (agentview), above the new >= 12 gate.
- The run stopped at the A1 decision boundary: the next scripted setup action `a:PLACE:obj_c:receiver:v1` was masked off (`SCRIPT_DIVERGENCE_MASK`, public `p:Held:obj_c` UNKNOWN, `p:OnTable:obj_c` FALSE, `p:AtBuffer:obj_c:pad_v` TRUE).
- Cause from the controller trace: during DESCEND and PRESS the right gripper finger (`gripper0_finger2_collision`) was in contact with `pad_v` (up to ~480 N), the PRESS phase used its whole 4 s because the hand could not descend, and during CLOSE obj_c was dragged onto `pad_v` (`obj_c_g0`/`pad_v` contacts). The frozen target anchor sits 0.0202 m (AABB) from its near pad, inside the gripper-finger envelope, although it satisfies the card's 0.008 m AABB clearance. The cube therefore ends up on the pad, not in the gripper.
- Per the card: stop, no coordinate change, no controller change, no V3-R2/V4/Family C. 1 of 8 attempts used, 1 of 48 skill calls used; A2 and Waves B/C never released. Nothing about the isochronous mechanism was measured.
