# Negative evidence - TASK_FAIL_NO_EXISTING_SOFT_STAGING_VALUE

## What gate 1 required

A public, recordable, non-hard-precondition downstream mechanism by which using W removes an existing high-level operation, entering existing skill count / rework / recovery / formal cost, not a new reward, hidden truth or 'must pass W' rule, and not merely 'W is closer to the goal'.

## What the platform has

- Skills in every registry: {"configs/contracts/d0_runtime_skills.yaml": ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"], "configs/contracts/skills.yaml": ["MOVE", "OPEN", "PICK", "PLACE", "PLACE_BUFFER"], "configs/runtime/stage_2a_contract_registry.yaml": ["OPEN", "PICK", "PLACE", "PLACE_BUFFER"], "configs/runtime/tp_fb_contract_registry.yaml": ["PICK", "PLACE", "PLACE_BUFFER"], "configs/runtime/tp_so_mvp_contract_registry.yaml": ["PICK", "PLACE"], "configs/runtime/tp_sr_contract_registry.yaml": ["PICK", "PLACE", "PLACE_BUFFER"], "configs/runtime/tp_sr_v2_contract_registry.yaml": ["PICK", "PLACE", "PLACE_BUFFER"]}
- Executor dispatch: ['MOVE', 'OPEN', 'PICK', 'PLACE', 'PLACE_BUFFER']; evaluator task ids: ['T_A', 'T_B']
- PLACE_BUFFER contract: PRE ['Held']; ADD ['AtBuffer', 'GripperEmpty']; DEL ['Held'] (no conditional effects).
- Skills whose precondition mentions a buffer or region: none
- Recovery-like skill names: none; recovery-like function definitions in src: ['src/cp_disr/analysis/tp_bsi_preflight.py:798:def recoverability_audit(scenes, geom):']
  (terms searched: RESEAT, RE_SEAT, REGRASP, RE_GRASP, REPOSITION, REHANDLE, REPICK, RE_PICK, RELOCATE, RECOVER; the only hits in src are module, path or audit-function names, not skills)

## Criteria

- existing_downstream_operation_that_staging_can_remove: met=False. skills in every registry and in the executor are PICK, PLACE, PLACE_BUFFER, OPEN (and the unused MOVE); none is a re-seat, re-grasp, reposition or recovery skill; recovery-like names found only in analysis/module/path names
- not_hard_precondition: met=None. not reached: there is no mechanism to classify
- benefit_enters_existing_cost_or_skill_count: met=False. TaskEvaluator goals depend only on Inside/AtBuffer of fixed regions (T_A, T_B); no location-quality term exists; PLACE_BUFFER only ADDs AtBuffer and DELs Held; no PRE of any skill mentions a buffer or region
- not_new_reward_hidden_truth_or_must_pass_W: met=False. a 'premium' W is only meaningful if some operation is cheaper for an object staged there; creating that is a new Evaluator/reward condition, a hard PRE, a controller feature or a hidden label
- not_merely_closer_to_goal: met=None. not reached

## Earlier physical evidence on buffer staging and relocation

- soft_relocation_design: {"file": "runs/final_master/S4/tp_soft_relocation_design/20260929T084850Z_b598cfd0/final_summary.md", "direct_success": "46/48", "relocation_success": "26/48", "scene_opportunity": "STRONG_HELPFUL 1, COST_HELPFUL 0, NEUTRAL 0, HARMFUL 20, UNKNOWN 3", "feasibility": "NOT_ESTABLISHED", "reading": "the direct route rarely fails and relocation takes about twice the simulated time: no discrete rework to save"}
- soft_relocation_pilot_v2: {"file": "runs/final_master/S4/tp_soft_relocation_pilot_v2/20260929T095821Z_efe8aa02/final_summary.md", "mechanism": "NOT_ESTABLISHED", "next_action": "ABANDON_OR_REDESIGN_RELOCATION_MECHANISM"}
- family_b_staging_v2: {"file": "runs/final_master/S4/family_b_staging_v2/20261001T053917Z_3064ee8b/final_summary.md", "physical_mechanism": "NOT_ESTABLISHED", "reading": "pad_u and pad_v differ by seconds only (all 24 branches succeed with matching skill multisets): a path/time difference, not a discrete rework"}
- family_b_v3r1: {"file": "runs/final_master/S4/family_b_v3r1_reach_safe_mirror/20261001T073554Z_80a6b7d6/final_summary.md", "mechanism": "FAMILY_B_V3_R1_REACH_SAFE_MECHANISM_PILOT_NOT_ESTABLISHED", "next_action": "STOP_TP_TASK_SEARCH_AND_ADJUST_CLAIMS"}

## Always-direct (static, not physically tested)

- FINISH(p,G_p) + FINISH(q,G_q) = PICK,PLACE,PICK,PLACE: 4 skills, about 15.5 s; staging routes: {'one_object_staged': 6, 'both_objects_staged': 8} skills; each staged object adds about 8.4 s.

## What would have to be added for W to matter

A location-dependent cost or a premium, i.e. a new Evaluator/reward term, a hard precondition tied to W, a controller feature (a re-seat or re-grasp that works only at W) or a hidden preferred-object label. Each is forbidden by the card, so the candidate stops here.

No coordinates were chosen, no reset was spent and no episode was run; nothing was tuned.
