# Reviewer objection matrix

| card | objection | evidence in this audit | survives |
|---|---|---|---|
| M1_BASKET_TWO_ITEMS | a one-line rule solves every state | free-site placement or 'larger first' removes the effect in every member task (pessimistic fill 0.636 < 1.0) | no |
| M1_BASKET_TWO_ITEMS | the effect size is unmeasured | unverifiable statically: G4, G8, G11 | no |
| M1_BASKET_TWO_ITEMS | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x3; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob  | no |
| M2_STOVE_TWO_POTS | a one-line rule solves every state | symmetric objects: any fixed order, with Turnon first, is as good as any other | no |
| M2_STOVE_TWO_POTS | the effect size is unmeasured | unverifiable statically: G4, G11 | no |
| M2_STOVE_TWO_POTS | the current skills cannot execute it | task skill coverage: NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour | no |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | a one-line rule solves every state | commutative: any fixed order | no |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | the orders differ only by path | commutative / path-only | no |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | the effect size is unmeasured | unverifiable statically: G11 | no |
| M3_TWO_TARGETS_DISTINCT_DESTINATIONS | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x2; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob  | no |
| M4A_ARTICULATED_PLUS_PLACE | a one-line rule solves every state | turn on first, then place: a single ordering rule | no |
| M4A_ARTICULATED_PLUS_PLACE | the effect size is unmeasured | unverifiable statically: G3, G4, G11 | no |
| M4A_ARTICULATED_PLUS_PLACE | the current skills cannot execute it | task skill coverage: NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour | no |
| M4B_ARTICULATED_HARD_PRECONDITION | a one-line rule solves every state | open first, close last | no |
| M4B_ARTICULATED_HARD_PRECONDITION | the contract already decides it | OPEN/CLOSE with Open(container) as PRE of In/PLACE gives the winner | no |
| M4B_ARTICULATED_HARD_PRECONDITION | the effect size is unmeasured | unverifiable statically: G11 | no |
| M4B_ARTICULATED_HARD_PRECONDITION | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x2, NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its | no |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | a one-line rule solves every state | act on the goal object directly: one rule solves every task | no |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | the orders differ only by path | commutative / path-only | no |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | the effect size is unmeasured | unverifiable statically: G11 | no |
| M5_SPATIAL_SINGLE_GOAL_DISTRACTORS | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x10; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob | no |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | a one-line rule solves every state | act on the goal object directly: one rule solves every task | no |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | the orders differ only by path | commutative / path-only | no |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | the effect size is unmeasured | unverifiable statically: G11 | no |
| M6_OBJECT_SINGLE_GOAL_SAME_SCENE | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x10; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob | no |
| M7_SINGLE_GOAL_OTHER | a one-line rule solves every state | act on the goal object directly: one rule solves every task | no |
| M7_SINGLE_GOAL_OTHER | the orders differ only by path | commutative / path-only | no |
| M7_SINGLE_GOAL_OTHER | the effect size is unmeasured | unverifiable statically: G11 | no |
| M7_SINGLE_GOAL_OTHER | the current skills cannot execute it | task skill coverage: NEW_SCRIPTED_SKILL x9, NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its | no |
| DV1_STACK_ORDER_IN_SHARED_BASKET | the effect size is unmeasured | unverifiable statically: G3, G4, G8, G11, G13 | no |
| DV1_STACK_ORDER_IN_SHARED_BASKET | the current skills cannot execute it | only 0 of 10 grocery assets are orientation-agnostic pinchable (none); 4 more would be pinchable if spawn orientation were verified (bbq_sauce,butter,chocolate_ | no |
| DV1_STACK_ORDER_IN_SHARED_BASKET | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
| DV2_SUPPORT_REMOVE_TOP_FIRST | a one-line rule solves every state | remove whatever rests on a needed object | no |
| DV2_SUPPORT_REMOVE_TOP_FIRST | the contract already decides it | a support predicate in PRE of PICK(B) determines the order | no |
| DV2_SUPPORT_REMOVE_TOP_FIRST | the effect size is unmeasured | unverifiable statically: G11, G13 | no |
| DV2_SUPPORT_REMOVE_TOP_FIRST | the current skills cannot execute it | stacked starts violate the OnTable contract and the Verifier OnTable bound; hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIB | no |
| DV2_SUPPORT_REMOVE_TOP_FIRST | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | a one-line rule solves every state | clear any neighbour within the envelope | no |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | the contract already decides it | Safety hard mask already removes the colliding candidate | no |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | the effect size is unmeasured | unverifiable statically: G11, G13 | no |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | the current skills cannot execute it | PICK and PLACE_BUFFER would suffice but hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cub | no |
| DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
| DV4_SHARED_DESTINATION_CAPACITY | a one-line rule solves every state | never place a non-goal item on the destination | no |
| DV4_SHARED_DESTINATION_CAPACITY | the contract already decides it | the goal contract only wants the goal item at the destination; B_PLAN never plans the distractor | no |
| DV4_SHARED_DESTINATION_CAPACITY | the effect size is unmeasured | unverifiable statically: G11, G13 | no |
| DV4_SHARED_DESTINATION_CAPACITY | the current skills cannot execute it | hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per | no |
| DV4_SHARED_DESTINATION_CAPACITY | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | a one-line rule solves every state | act on the goal object | no |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | the contract already decides it | the goal contract names the object | no |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | the effect size is unmeasured | unverifiable statically: G11, G13 | no |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | the current skills cannot execute it | the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator a | no |
| DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | a one-line rule solves every state | open iff a pending goal needs the drawer: a one-line rule on the goal text | no |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | the effect size is unmeasured | unverifiable statically: G11, G13 | no |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | the current skills cannot execute it | sliding drawer needs a handle-pull skill; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob mo | no |
| DV6_ARTICULATED_OPTIONAL_PREPARATION | a modified task is not the LIBERO benchmark | must be labelled LIBERO_DERIVED_DIAGNOSTIC_TASK and kept out of the public-validation table | no |
