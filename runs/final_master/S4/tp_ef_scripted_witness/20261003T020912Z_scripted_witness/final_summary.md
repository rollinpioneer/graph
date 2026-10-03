# CP-DISR-TP-EF-SCRIPTED-WITNESS-1 — final summary

Mechanism gate: **INSUFFICIENT_FIRST_WITNESS**
Stopped: no; branches run: 4; caps exceeded: none

## T_B_dev_03: NOT_RELIABLE  winner=NONE
- conditions: {'1_both_routes_2of2_task_success': True, '2_exactly_four_skills_each': True, '3_no_anomaly': True, '4_repeat_winner_direction_consistent': True, '5_abs_delta_T_gt_epsilon_T': False, '6_abs_delta_G_gt_epsilon_G': False, '7_shorter_time_and_higher_Q_same_route': True}
- delta_T=0.04999999999997229 delta_G=-0.0009151407425168623 per-repeat winners=['S', 'S']
- mean decision-relative time: {'T': 16.499999999997428, 'S': 16.449999999997456}; mean Q_ref_decision: {'T': 0.6095068271022848, 'S': 0.6104219678448016}

## Branches

- A1 T_B_dev_03 route T rep0: PASS success=True skills=4 decision_rel=16.499999999997428 Q=0.6095068271022848 anomalies=[]
- A2 T_B_dev_03 route S rep0: PASS success=True skills=4 decision_rel=16.449999999997456 Q=0.6104219678448016 anomalies=[]
- A3 T_B_dev_03 route T rep1: PASS success=True skills=4 decision_rel=16.499999999997428 Q=0.6095068271022848 anomalies=[]
- A4 T_B_dev_03 route S rep1: PASS success=True skills=4 decision_rel=16.449999999997456 Q=0.6104219678448016 anomalies=[]

epsilon_G=0.02 epsilon_T=2.1

Budget used: {'elastic': 0, 'environment_constructions': 4, 'open_skill_calls': 0, 'optimizer_steps': 0, 'physical_branch_attempts': 4, 'provider_requests': 0, 'rl_transitions': 0, 'scripted_skill_calls': 16, 'skill_retries': 0, 'start_case_calls': 4, 'test_episodes': 0}
env.reset() invocations (incl. bootstrap resets): 8
