# B_PLAN development report

execution_state=COMPLETE; measurement_status=MEASURED; task=T_B; split=configs/splits/T_B_phase_a_v13_r1_dev10.json.

The production binding is src/cp_disr/baselines/b_plan.py and the CLI entry point is b-plan-dev. It uses the shared RuntimeFactory, SkillExecutor, FactVerifier, SnapshotBuilder, TaskEvaluator, FactStore and registered SkillContract set. Each decision searches nominal contract successors and executes only the first action; the real Verifier then rebuilds public facts before the next decision.

Frozen search contract: depth=6, max_nodes=4096, CPU time=2.0 seconds per decision, canonical tie-break, positive independent-reference median skill cost d_ref=4.199999999997672 seconds from runs/stage_0a/calibration.json. The heuristic is unmet goal count plus layered contract shortest relaxed-step count; it is not claimed admissible. The planner records PLAN_FOUND, NO_PLAN and SEARCH_TIMEOUT, while the runner records skill failure and Verifier failure separately.

Development result: 10/10 episodes completed; success=6; episode statuses={'NO_PLAN': 4, 'SUCCESS': 6}; search statuses={'GOAL_ALREADY_SATISFIED': 0, 'NO_PLAN': 4, 'PLAN_FOUND': 34, 'SEARCH_TIMEOUT': 0}; skill_failure_count=0; verifier_failure_count=0. All 10 cases were from the dev split and no test rows were used.

Planner inputs are limited to public T/F/U FactStore facts, goal, complete registered SkillContract, and public remaining deadline. Hidden simulator truth, future outcomes, VLM relation, and RL Q are excluded from the planner. The runtime split removes relation-cache bindings, and relation_cache_loaded=false.

Evidence files are under b_plan_dev/: b_plan_dev_results.json, decisions.jsonl, episodes.json, b_plan_config.json, b_plan_runtime_manifest.yaml, and b_plan_runtime_split.json. No provider, RL training, B_PLAN+R, B_PLAN+R*, S1, or formal test was run.
