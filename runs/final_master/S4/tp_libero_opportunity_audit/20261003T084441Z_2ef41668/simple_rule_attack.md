# Attacks 1, 3, 4 and 5

| card | A1_fixed_rule | A3_static_prior | A4_llm_vlm_direct | A5_memorisation |
|---|---|---|---|---|
| M1_BASKET_TWO_ITEMS | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | MEMORISABLE_FIXED_SCENES |
| M2_STOVE_TWO_POTS | FIXED_HEURISTIC_SOLVABLE | NOT_APPLICABLE | NOT_APPLICABLE | MEMORISABLE_FIXED_SCENES |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | FIXED_HEURISTIC_SOLVABLE | NOT_APPLICABLE | NOT_APPLICABLE | MEMORISABLE_FIXED_SCENES |
| M4A_ARTICULATED_PLUS_PLACE | FIXED_HEURISTIC_SOLVABLE | NOT_APPLICABLE | NOT_APPLICABLE | MEMORISABLE_FIXED_SCENES |
| M4B_ARTICULATED_HARD_PRECONDITION | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | MEMORISABLE_FIXED_SCENES |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | MEMORISABLE_FIXED_SCENES |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | MEMORISABLE_FIXED_SCENES |
| M7_SINGLE_GOAL_OTHER | FIXED_HEURISTIC_SOLVABLE | NOT_APPLICABLE | NOT_APPLICABLE | MEMORISABLE_FIXED_SCENES |
| DV1_STACK_ORDER_IN_SHARED_BASKET | NOT_SOLVED_BY_ONE_RULE | NOT_APPLICABLE | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |
| DV2_SUPPORT_REMOVE_TOP_FIRST | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |
| DV4_SHARED_DESTINATION_CAPACITY | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | FIXED_HEURISTIC_SOLVABLE | STATIC_PRIOR_SUFFICIENT | KEEP_AS_LLM_VLM_BASELINE | NEEDS_STATE_AND_GOAL_VARIATION |

### M1_BASKET_TWO_ITEMS
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. free-site placement or 'larger first' removes the effect in every member task (pessimistic fill 0.636 < 1.0)
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: neutral,helpful
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M2_STOVE_TWO_POTS
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. symmetric objects: any fixed order, with Turnon first, is as good as any other
- A3_static_prior: NOT_APPLICABLE. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: NOT_APPLICABLE. would require neutral, reversed, harmful and goal-dependent sub-conditions: neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M3_TWO_TARGETS_DISTINCT_DESTINATIONS
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. commutative: any fixed order
- A3_static_prior: NOT_APPLICABLE. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: NOT_APPLICABLE. would require neutral, reversed, harmful and goal-dependent sub-conditions: neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M4A_ARTICULATED_PLUS_PLACE
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. turn on first, then place: a single ordering rule
- A3_static_prior: NOT_APPLICABLE. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: NOT_APPLICABLE. would require neutral, reversed, harmful and goal-dependent sub-conditions: neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M4B_ARTICULATED_HARD_PRECONDITION
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. open first, close last
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M5_SPATIAL_SINGLE_GOAL_DISTRACTORS
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. act on the goal object directly: one rule solves every task
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M6_OBJECT_SINGLE_GOAL_SAME_SCENE
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. act on the goal object directly: one rule solves every task
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### M7_SINGLE_GOAL_OTHER
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. act on the goal object directly: one rule solves every task
- A3_static_prior: NOT_APPLICABLE. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: NOT_APPLICABLE. would require neutral, reversed, harmful and goal-dependent sub-conditions: neutral
- A5_memorisation: MEMORISABLE_FIXED_SCENES. public tasks have fixed scene ids and fixed initial-state files; B2 can memorise per-task winners

### DV1_STACK_ORDER_IN_SHARED_BASKET
- A1_fixed_rule: NOT_SOLVED_BY_ONE_RULE. a 'larger first' rule fails on pairs where size and fragility disagree; this can be designed in, but the real effect size is unmeasured
- A3_static_prior: NOT_APPLICABLE. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral,harmful
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak

### DV2_SUPPORT_REMOVE_TOP_FIRST
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. remove whatever rests on a needed object
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak

### DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. clear any neighbour within the envelope
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak

### DV4_SHARED_DESTINATION_CAPACITY
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. never place a non-goal item on the destination
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,harmful
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak

### DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. act on the goal object
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,harmful
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak

### DV6_ARTICULATED_OPTIONAL_PREPARATION
- A1_fixed_rule: FIXED_HEURISTIC_SOLVABLE. open iff a pending goal needs the drawer: a one-line rule on the goal text
- A3_static_prior: STATIC_PRIOR_SUFFICIENT. relation without consequence information is enough when one rule decides every state
- A4_llm_vlm_direct: KEEP_AS_LLM_VLM_BASELINE. would require neutral, reversed, harmful and goal-dependent sub-conditions: helpful,neutral
- A5_memorisation: NEEDS_STATE_AND_GOAL_VARIATION. a derived family must randomise layout and goal so labels do not leak
