# Planner attack (Attack 2)

B_PLAN: uniform-cost search over the contract state space with a relaxed goal-distance heuristic; cost is the reference skill duration (src/cp_disr/baselines/b_plan.py); ties break by contract id

A card survives this attack only when the contract gives NO winner (CONTRACT_TIE_NO_WINNER) and the order is not a hard precondition.

| card | verdict | reason |
|---|---|---|
| M1_BASKET_TWO_ITEMS | CONTRACT_TIE_NO_WINNER | PICK/PLACE contracts have no PRE/ADD/DEL that distinguishes the two orders; B_PLAN ties and breaks by id |
| M2_STOVE_TWO_POTS | CONTRACT_TIE_NO_WINNER | no contract relation orders the two pots; Turnon is outside the current predicate vocabulary |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | CONTRACT_TIE_NO_WINNER | independent PICK/PLACE pairs; B_PLAN ties |
| M4A_ARTICULATED_PLUS_PLACE | CONTRACT_TIE_NO_WINNER | Turnon has no contract; no current relation orders it against the placement |
| M4B_ARTICULATED_HARD_PRECONDITION | CONTRACT_SOLVABLE | OPEN/CLOSE with Open(container) as PRE of In/PLACE gives the winner |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | CONTRACT_TIE_NO_WINNER | a single goal chain; B_PLAN returns it directly |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | CONTRACT_TIE_NO_WINNER | a single goal chain; B_PLAN returns it directly |
| M7_SINGLE_GOAL_OTHER | CONTRACT_TIE_NO_WINNER | a single goal chain; B_PLAN returns it directly |
| DV1_STACK_ORDER_IN_SHARED_BASKET | CONTRACT_TIE_NO_WINNER | no contract relation orders the two items (the drop site occupancy is not a predicate) |
| DV2_SUPPORT_REMOVE_TOP_FIRST | CONTRACT_SOLVABLE | a support predicate in PRE of PICK(B) determines the order |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | CONTRACT_SOLVABLE | Safety hard mask already removes the colliding candidate |
| DV4_SHARED_DESTINATION_CAPACITY | CONTRACT_SOLVABLE | the goal contract only wants the goal item at the destination; B_PLAN never plans the distractor |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | CONTRACT_SOLVABLE | the goal contract names the object |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | CONTRACT_TIE_NO_WINNER | optional preparation has no contract winner |
