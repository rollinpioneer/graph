# CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1 - final design audit summary

Verdict: **TASK_FAIL_NO_STABLE_SUPPORT_PAIR**  (label DESIGN_ONLY_NOT_AUTHORIZED, family T_P_CARRIER_SUPPORT_V1)

All failing categories (each evaluated independently; the verdict is the first in the declared order): TASK_FAIL_NO_STABLE_SUPPORT_PAIR, TASK_FAIL_NO_STABILITY_CONTRAST, TASK_FAIL_COMMUTATIVE, TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT, TASK_FAIL_STATIC_PRIOR_SUFFICIENT, TASK_FAIL_CANARY_NOT_BOUNDED

## Category evidence

- ENGINEERING_UNRESOLVED: ok - {"assets_without_box_collision_proxy": []}
- TASK_FAIL_CANARY_NOT_BOUNDED: FAIL - {"bounded": false, "episodes": ["C1 stable centre MOVE_CARRIER", "C2 stable centre repeat", "C3 edge configuration MOVE_CARRIER", "C4 edge repeat"], "max_episodes": 4, "name": "CP-DISR-TP-CARRIER-SUPPORT-MECHANISM-CANARY-1", "questions": {"Q1 carrier moves stably": "C1, C2", "Q2 co-transport in the 
- TASK_FAIL_COMMUTATIVE: FAIL - {"reversed_witness_not_path_only": false, "s2": "COMMUTATIVE", "s3_harmful_possible": false}
- TASK_FAIL_CONTRACT_REVEALED: ok - {"contract_plans": {"S1_CO_TRANSPORT_STABLE": 5, "S2_SEPARATE_DESTINATIONS": 3, "S3_CO_TRANSPORT_UNSTABLE": 5, "S4_NO_SUPPORT_NEUTRAL": 3}, "optimal_skills": {"S1_CO_TRANSPORT_STABLE": 1, "S2_SEPARATE_DESTINATIONS": 3, "S3_CO_TRANSPORT_UNSTABLE": 1, "S4_NO_SUPPORT_NEUTRAL": 3}}
- TASK_FAIL_EVALUATOR_SEMANTIC_CHANGE: ok - {"existing_evaluators_modified": false}
- TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT: FAIL - {"counterexamples": {"R1": [], "R2": [], "R3": [], "R4": ["S3_CO_TRANSPORT_UNSTABLE"], "R5": ["S1_CO_TRANSPORT_STABLE", "S3_CO_TRANSPORT_UNSTABLE"], "R6": []}, "unrefuted_rules": ["R1", "R2", "R3", "R6"]}
- TASK_FAIL_NO_CARRIED_OBJECT: ok - {"unusable": {"akita_black_bowl": "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED", "alphabet_soup": "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED", "black_book": "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED", "butter": "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED", "chefmate_8_frypan": "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED", "cookies": "NEW_LOW_LEVEL_BEH
- TASK_FAIL_NO_GENERIC_CARRIER: ok - {"classes": {"akita_black_bowl": "ASSET_SPECIFIC_CONTROLLER_REQUIRED", "basket": "GENERIC_BINDING_REQUIRED", "cookies": "GENERIC_MOVE_CARRIER_COMPATIBLE", "desk_caddy": "NOT_MOVABLE", "flat_stove": "NOT_MOVABLE", "glazed_rim_porcelain_ramekin": "ASSET_SPECIFIC_CONTROLLER_REQUIRED", "microwave": "NOT
- TASK_FAIL_NO_STABILITY_CONTRAST: FAIL - {"contrast_pairs": [], "edge_class_counts": {"UNKNOWN_NEEDS_PHYSICAL_CANARY": 6}, "marginal_edge_pairs": []}
- TASK_FAIL_NO_STABLE_SUPPORT_PAIR: FAIL - {"centre_class_counts": {"UNKNOWN_NEEDS_PHYSICAL_CANARY": 6}, "pairs_evaluated": 6, "stable_center_pairs": []}
- TASK_FAIL_RELATION_NOT_EXPRESSIBLE: ok - [{"id": "rel_carrier_goal", "issues": []}, {"id": "rel_pick_supports_move", "issues": []}]
- TASK_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR: ok - {"new_states": [], "spec_status": "SPECIFIED"}
- TASK_FAIL_REQUIRES_NEW_PERCEPTION_MODEL: ok - {"A_marker_pixels_on_carrier": 65, "B_marker_pixels_with_A_present": 222, "evaluated_pair": "basket|bbq_sauce", "status": "PASS"}
- TASK_FAIL_SNAPSHOT_REWRITE: ok - {"capture_sim_reads_env_sim_data": true, "free_joint_objects_set_by_qpos": true, "object_set_defined_by_case": true, "python_state_dump_covers_robot_controller_gripper_rng": true, "restore_builds_env_from_case_first": true, "sim_fields_saved": ["\"qpos\", \"qvel\", \"act\", \"ctrl\", \"qacc_warmstar
- TASK_FAIL_STATIC_PRIOR_SUFFICIENT: FAIL - {"best_first_action_by_scenario": {"S1_CO_TRANSPORT_STABLE": "carrier_first", "S2_SEPARATE_DESTINATIONS": "tie", "S3_CO_TRANSPORT_UNSTABLE": "carrier_first", "S4_NO_SUPPORT_NEUTRAL": "tie"}}

## PASS conditions

- 10_relation_expressible_non_redundant: True
- 11_verifier_evaluator_definable: True
- 12_snapshot_no_rewrite: True
- 13_no_new_models_or_low_level: True
- 14_canary_bounded: False
- 1_one_carrier_generic: True
- 2_one_carried_object: True
- 3_stable_centre_pair: False
- 4_edge_pair: False
- 5_neutral_configuration: True
- 6_s1_skill_advantage: True
- 7_reversed_not_path_only: False
- 8_r1_to_r6_all_refuted: False
- 9_contract_cannot_order: True

Budget: environment constructions 0, env.reset 0, start_case 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test 0, demo replay 0.
