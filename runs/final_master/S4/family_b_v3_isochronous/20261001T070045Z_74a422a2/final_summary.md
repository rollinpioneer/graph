# Family B V3 isochronous cross-staging pilot

- Card CP-DISR-S4-FAMILY-B-V3-ISOCHRONOUS-CROSS-STAGING-PILOT-1; frozen profile `FAMILY_B_PUBLIC_OBS_V2` 3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917 (unchanged).
- Gate 0: GATE0_PASS. Waves: A=NOT_EVALUATED, B=NOT_EVALUATED, C=NOT_EVALUATED; run state STOPPED (step A1 did not meet its technical/qualification gate).
- Attempts reserved/started/terminal: 1/1/1 (cap 8); constructions 1; explicit resets 1; internal resets 2; live skill calls 1/48.
- Provider / representation / RL / optimizer / elastic / formal test = 0; S2/S3/TP training = false.
- Mechanism: **FAMILY_B_V3_ISOCHRONOUS_MECHANISM_PILOT_NOT_ESTABLISHED**; technical: STOPPED_BEFORE_ALL_BRANCHES_RAN; next action `S4_CLAIM_ADJUSTMENT_DECISION`.

| pair | C(U) | C(V) | delta U-V | PLACE_BUFFER U/V (s) | steps U/V | match |
|---|---|---|---|---|---|---|
| layout_0/B_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_0/C_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_1/B_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |
| layout_1/C_PENDING | None | None | None | None (abs diff) | None (abs diff) | False |

Conditions: 10_layout_0_preference_reversal=False, 11_layout_1_preference_reversal=False, 12_same_object_best_pad_changes_with_layout=False, 13_neither_always_u_nor_always_v_reaches_4_of_4=False, 14_provider_representation_rl_optimizer_zero=True, 15_old_evidence_and_frozen_config_unchanged=True, 1_8_of_8_task_success=False, 2_8_of_8_complete_records=False, 3_four_pair_restores_pass=False, 4_place_buffer_duration_diff_le_0.15=False, 5_controller_steps_diff_le_3=False, 6_key_target_support_ge_12=False, 7_no_noplan_timeout_exception_or_recorder_error=False, 8_candidates_equally_legal=False, 9_plan_length_and_skill_multiset_match=False
- Measured winners: {}; predicted: {'layout_0/B_PENDING': 'pad_u', 'layout_0/C_PENDING': 'pad_v', 'layout_1/B_PENDING': 'pad_v', 'layout_1/C_PENDING': 'pad_u'}.
- Always-U 0/4, always-V 0/4; state-conditioned reversal False.
- No significance or generalization claim; speedup NOT_MEASURED.

## Stop reason (A1)

- Wave A canary A1 (layout_0 / B_PENDING / pad_u) ended `SKILL_FAILURE:TIMEOUT` in the first setup action `a:PICK:obj_c:v1`. obj_c sits at target anchor V (0.22, -0.18), 0.80 m from the robot base, beyond the previously demonstrated reach (0.76 m; recorded as a Gate 0 risk).
- The PICK itself closed on the cube (HOVER, DESCEND, PRESS, CLOSE all completed, 19 gripper contacts with obj_c), but `LIFT_SHOW` toward the frozen show pose (0.02, -0.08) stalled with the end effector at x = 0.200 m (command saturated at dpos_x = -1, no collision with another object) until the 9 s skill timeout.
- Per the card: stop, no coordinate change, no lowered gate, no ninth attempt. A2 and Waves B/C were never released; 1 of 8 attempts used, 1 of 48 skill calls used. Nothing about the isochronous mechanism was measured.
