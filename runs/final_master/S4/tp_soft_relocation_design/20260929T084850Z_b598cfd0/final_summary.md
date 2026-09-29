# S4 T_P_SOFT_RELOCATION_V1 design feasibility (CP-DISR-S4-TASK-REDESIGN-1)

Curated mechanism demonstration; not an unbiased benchmark. method_version 2.1.1; no method upgrade; S2 not authorized.

**Result: task_family_feasibility = NOT_ESTABLISHED; recommended_next_action = EXPLICIT_RESEARCH_DECISION (card rule on gate failure: REDESIGN_GEOMETRY_OR_CONTROLLER).**

## Pool and static screen

64 generated configs (16 per occlusion bin); 64/64 static resets used; 58 PASS (per bin {'0': 13, '1': 15, '2': 15, '3': 15}). Two resets (pool_00, pool_01) were lost to a perception colour-mask crosstalk found before the fix; four further configs failed the reset-fact checks.

## Physical probe

24 scenes, 96 paired branch episodes (direct vs relocation, 2 repeats, shared restore seed, B_PLAN continuation): direct success 46/48, relocation success 26/48. Scene opportunity: STRONG_HELPFUL 1, COST_HELPFUL 0, NEUTRAL 0, HARMFUL 20, UNKNOWN 3. Gate checks: {'both_reachable_ge_12': True, 'engineering_failure_le_20pct': True, 'helpful_bins_ge_2': False, 'helpful_ge_8': False, 'neutral_or_harmful_ge_4': True}; engineering-failure rate 0.062.

The direct route rarely fails at these separations (no helpful headroom), and when both routes succeed relocation takes about 2x the simulated time. Relocation-route failures after PICK(interferer) repeat identically in both repeats and are probably a verifier/perception reading issue, not evidence about the physical benefit; that cause was not verified.

## Not run

Provider calls, representation probe, RL/optimizer/training, S2/S3, formal test: none (all ledger entries 0 where capped at 0; provider first calls/retries 0/12, 0/12; representation forwards 0/12).

## Throughput

Physical workers: 2 then 4 (max 4). Aggregate 0.219/s at 2 workers, 0.427/s at 4 workers; decision {'reason': 'FOUR_KEPT', 'workers_kept': 4}. 4-vs-2 ratio 1.77x on different branch subsets; single-worker baseline not measured directly (best-single estimate 0.166/s from min wall).
