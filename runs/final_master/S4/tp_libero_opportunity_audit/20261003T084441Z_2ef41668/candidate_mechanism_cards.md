# Candidate mechanism cards

Statuses per hard gate: PASS, FAIL, UNVERIFIABLE_STATICALLY (needs a physical measurement that this card is forbidden to take). A candidate needs all G1-G15 = PASS.

## M1_BASKET_TWO_ITEMS

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#0; libero_10#1; libero_10#7

- state: two items on the table, basket empty
- candidate_A: PICK/PLACE the first item into the basket
- candidate_B: PICK/PLACE the second item first
- both_legal: 2 legal first actions: PICK item one or PICK item two; both are on the table
- persistent_effect: first item inside the basket
- downstream_effect: second item may land on the first if the drop site is fixed
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: packing order (bulky or flat item last) is a commonsense relation, but the official tasks fix no relation, no label and no goal variation
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: neutral where the fill is small; helpful only if the pile effect is real, which needs a physical probe (not established statically)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: free-site placement or 'larger first' removes the effect in every member task (pessimistic fill 0.636 < 1.0)

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: PICK item one or PICK item two; both are on the table |
| G2 | PASS | PICK/PLACE contracts have no PRE/ADD/DEL that distinguishes the two orders; B_PLAN ties and breaks by id |
| G3 | PASS | pessimistic fill ratio of the basket contain_region by the two items: 0.636, 0.434, 0.551 (material threshold 0.50); material in 2 of 3 tasks |
| G4 | UNVERIFIABLE_STATICALLY | basket occupancy matters only if both items are dropped at one fixed site; with a free-site destination binding (absent from the current PLACE) the orders are interchangeable and a fill below 1.0 leaves room for both |
| G5 | PASS | difference is not a pure path distance: occupancy of the basket, not only path, if the drop site is fixed |
| G6 | PASS | packing order (bulky or flat item last) is a commonsense relation, but the official tasks fix no relation, no label and no goal variation |
| G7 | PASS | site occupancy is not a contract predicate |
| G8 | UNVERIFIABLE_STATICALLY | neutral where the fill is small; helpful only if the pile effect is real, which needs a physical probe |
| G9 | FAIL | free-site placement or 'larger first' removes the effect in every member task (pessimistic fill 0.636 < 1.0) |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x3; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G9, G10; unverifiable: G4, G8, G11

## M2_STOVE_TWO_POTS

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#8

- state: libero_10#8
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 2 legal first actions: place pot one or pot two first; the goal also contains Turnon, which is a separate action
- persistent_effect: pessimistic fill ratio of the flat_stove cook_region by the two items: 1.991 (material threshold 0.50); material in 1 of 1 tasks; a ratio above 1.0 means the two pots cannot both fit inside one cook_region footprint, so the On predicate semantics decide feasibility
- downstream_effect: overlap, knob access and pot collision need a simulator
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: the two pots play identical roles; the only candidate relation is geometric access
- helpful_case: absent
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: symmetric pots give only the neutral case (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: symmetric objects: any fixed order, with Turnon first, is as good as any other

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: place pot one or pot two first; the goal also contains Turnon, which is a separate action |
| G2 | PASS | no contract relation orders the two pots; Turnon is outside the current predicate vocabulary |
| G3 | PASS | pessimistic fill ratio of the flat_stove cook_region by the two items: 1.991 (material threshold 0.50); material in 1 of 1 tasks; a ratio above 1.0 means the two pots cannot both fit inside one cook_region footprint, so the On predicate semantics decide feasibility |
| G4 | UNVERIFIABLE_STATICALLY | overlap, knob access and pot collision need a simulator |
| G5 | PASS | difference is not a pure path distance: Turnon and the two placements interact through physical access |
| G6 | FAIL | the two pots play identical roles; the only candidate relation is geometric access |
| G7 | PASS | geometric access would be a safety/contract precondition if it existed |
| G8 | FAIL | symmetric pots give only the neutral case |
| G9 | FAIL | symmetric objects: any fixed order, with Turnon first, is as good as any other |
| G10 | FAIL | task skill coverage: NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding; Turnon needs a rotary knob skill (NEW_LOW_LEVEL_CONTROLLER_REQUIRED) |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G6, G8, G9, G10; unverifiable: G4, G11

## M3_TWO_TARGETS_DISTINCT_DESTINATIONS

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#4; libero_10#6

- state: libero_10#4; libero_10#6
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 2 legal first actions: either object can be picked first
- persistent_effect: the two destinations are different plates or regions; placing one object does not change the other's destination
- downstream_effect: orders are interchangeable up to path
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: no relation to express
- helpful_case: absent
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: independent subtasks: neutral only (established)
- expected_extra_skills: 0: the orders differ only by path
- expected_effect_on_J: path time only
- simple_rule_attack: commutative: any fixed order

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: either object can be picked first |
| G2 | PASS | independent PICK/PLACE pairs; B_PLAN ties |
| G3 | FAIL | the two destinations are different plates or regions; placing one object does not change the other's destination |
| G4 | FAIL | orders are interchangeable up to path |
| G5 | FAIL | the difference between the orders is only low-level path |
| G6 | FAIL | no relation to express |
| G7 | PASS | no relation |
| G8 | FAIL | independent subtasks: neutral only |
| G9 | FAIL | commutative: any fixed order |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x2; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G3, G4, G5, G6, G8, G9, G10; unverifiable: G11

## M4A_ARTICULATED_PLUS_PLACE

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#2

- state: libero_10#2
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 2 legal first actions: turn the stove on or place the pot first
- persistent_effect: stove state and pot position persist; their physical interaction (knob blocked by the pot) cannot be established statically
- downstream_effect: needs physical measurement
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: at most a geometric access relation
- helpful_case: absent
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: no goal variation exists in the public tasks (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: turn on first, then place: a single ordering rule

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: turn the stove on or place the pot first |
| G2 | PASS | Turnon has no contract; no current relation orders it against the placement |
| G3 | UNVERIFIABLE_STATICALLY | stove state and pot position persist; their physical interaction (knob blocked by the pot) cannot be established statically |
| G4 | UNVERIFIABLE_STATICALLY | needs physical measurement |
| G5 | PASS | difference is not a pure path distance: knob access |
| G6 | FAIL | at most a geometric access relation |
| G7 | PASS | geometric |
| G8 | FAIL | no goal variation exists in the public tasks |
| G9 | FAIL | turn on first, then place: a single ordering rule |
| G10 | FAIL | task skill coverage: NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding; the rotary knob needs rotation actions that the executor zeroes |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G6, G8, G9, G10; unverifiable: G3, G4, G11

## M4B_ARTICULATED_HARD_PRECONDITION

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#3; libero_10#9; libero_goal#3

- state: libero_10#3; libero_10#9; libero_goal#3
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 1 legal first actions: the container must be opened before anything can go in; closing must follow insertion
- persistent_effect: container state persists, but it is a hard precondition, not a prior
- downstream_effect: container state gates the next action
- hard_precondition_or_not: HARD: articulated CLOSE on microwave_1 gates a placement into its interior; articulated CLOSE on white_cabinet_1_bottom_region gates a placement into its interior; articulated OPEN on microwave_1_heating_region gates a placement into its interior; articulated OPEN on wooden_cabinet_1_top_region gates a pl
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: container-open relation
- helpful_case: present
- neutral_case: absent
- reversed_or_harmful_case: absent
- case_coverage_note: always required (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: open first, close last

| gate | status | reason |
|---|---|---|
| G1 | FAIL | 1 legal first high-level actions in the decision state: the container must be opened before anything can go in; closing must follow insertion |
| G2 | FAIL | hard precondition decides the order (articulated CLOSE on microwave_1 gates a placement into its interior; articulated CLOSE on white_cabinet_1_bottom_region gates a placement into its interior; articulated OPEN on microwave_1_heating_region gates a placement into its interior; articulated OPEN on wooden_cabinet_1_top_region gates a pl): belongs in PRE/ADD/DEL, the contract planner returns the winner |
| G3 | PASS | container state persists, but it is a hard precondition, not a prior |
| G4 | PASS | container state gates the next action |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | container-open relation |
| G7 | FAIL | Open(container) is already in the current contract vocabulary |
| G8 | FAIL | always required |
| G9 | FAIL | open first, close last |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x2, NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G1, G2, G7, G8, G9, G10; unverifiable: G11

## M5_SPATIAL_SINGLE_GOAL_DISTRACTORS

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_spatial#0; libero_spatial#1; libero_spatial#2; libero_spatial#3; libero_spatial#4; libero_spatial#5; libero_spatial#6; libero_spatial#7; libero_spatial#8; libero_spatial#9

- state: libero_spatial#0; libero_spatial#1; libero_spatial#2
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 1 legal first actions: one required action chain; moving a distractor is legal but never needed (3 of 10 tasks have a distractor that may enter the finger envelope, proxy)
- persistent_effect: no second goal for the first action to affect
- downstream_effect: single goal
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: the language relation identifies WHICH object to move (grounding), not WHICH order to use
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: same-scene goal switching only changes the target object (established)
- expected_extra_skills: 0: the orders differ only by path
- expected_effect_on_J: path time only
- simple_rule_attack: act on the goal object directly: one rule solves every task

| gate | status | reason |
|---|---|---|
| G1 | FAIL | 1 legal first high-level actions in the decision state: one required action chain; moving a distractor is legal but never needed (3 of 10 tasks have a distractor that may enter the finger envelope, proxy) |
| G2 | PASS | a single goal chain; B_PLAN returns it directly |
| G3 | FAIL | no second goal for the first action to affect |
| G4 | FAIL | single goal |
| G5 | FAIL | the difference between the orders is only low-level path |
| G6 | PASS | the language relation identifies WHICH object to move (grounding), not WHICH order to use |
| G7 | PASS | goal grounding |
| G8 | PASS | same-scene goal switching only changes the target object |
| G9 | FAIL | act on the goal object directly: one rule solves every task |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x10; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G1, G3, G4, G5, G9, G10; unverifiable: G11

## M6_OBJECT_SINGLE_GOAL_SAME_SCENE

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_object#0; libero_object#1; libero_object#2; libero_object#3; libero_object#4; libero_object#5; libero_object#6; libero_object#7; libero_object#8; libero_object#9

- state: libero_object#0; libero_object#1; libero_object#2
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 1 legal first actions: one required action chain; moving a distractor is legal but never needed (7 of 10 tasks have a distractor that may enter the finger envelope, proxy)
- persistent_effect: no second goal for the first action to affect
- downstream_effect: single goal
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: the language relation identifies WHICH object to move (grounding), not WHICH order to use
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: same-scene goal switching only changes the target object (established)
- expected_extra_skills: 0: the orders differ only by path
- expected_effect_on_J: path time only
- simple_rule_attack: act on the goal object directly: one rule solves every task

| gate | status | reason |
|---|---|---|
| G1 | FAIL | 1 legal first high-level actions in the decision state: one required action chain; moving a distractor is legal but never needed (7 of 10 tasks have a distractor that may enter the finger envelope, proxy) |
| G2 | PASS | a single goal chain; B_PLAN returns it directly |
| G3 | FAIL | no second goal for the first action to affect |
| G4 | FAIL | single goal |
| G5 | FAIL | the difference between the orders is only low-level path |
| G6 | PASS | the language relation identifies WHICH object to move (grounding), not WHICH order to use |
| G7 | PASS | goal grounding |
| G8 | PASS | same-scene goal switching only changes the target object |
| G9 | FAIL | act on the goal object directly: one rule solves every task |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x10; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G1, G3, G4, G5, G9, G10; unverifiable: G11

## M7_SINGLE_GOAL_OTHER

origin: UNMODIFIED_PUBLIC_TASK, host: LIBERO_ENV
tasks: libero_10#5; libero_goal#0; libero_goal#1; libero_goal#2; libero_goal#4; libero_goal#5; libero_goal#6; libero_goal#7; libero_goal#8; libero_goal#9

- state: libero_10#5; libero_goal#0; libero_goal#1
- candidate_A: first required action chain
- candidate_B: alternative first action
- both_legal: 1 legal first actions: one required action chain; moving a distractor is legal but never needed (2 of 10 tasks have a distractor that may enter the finger envelope, proxy)
- persistent_effect: no second goal for the first action to affect
- downstream_effect: single goal
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: the language relation identifies WHICH object to move (grounding), not WHICH order to use
- helpful_case: absent
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: same-scene goal switching only changes the target object (established)
- expected_extra_skills: 0: the orders differ only by path
- expected_effect_on_J: path time only
- simple_rule_attack: act on the goal object directly: one rule solves every task

| gate | status | reason |
|---|---|---|
| G1 | FAIL | 1 legal first high-level actions in the decision state: one required action chain; moving a distractor is legal but never needed (2 of 10 tasks have a distractor that may enter the finger envelope, proxy) |
| G2 | PASS | a single goal chain; B_PLAN returns it directly |
| G3 | FAIL | no second goal for the first action to affect |
| G4 | FAIL | single goal |
| G5 | FAIL | the difference between the orders is only low-level path |
| G6 | FAIL | the language relation identifies WHICH object to move (grounding), not WHICH order to use |
| G7 | PASS | goal grounding |
| G8 | FAIL | same-scene goal switching only changes the target object |
| G9 | FAIL | act on the goal object directly: one rule solves every task |
| G10 | FAIL | task skill coverage: NEW_SCRIPTED_SKILL x9, NEW_LOW_LEVEL_CONTROLLER x1; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | BDDL predicates are public and evaluable, but the RGB-D Verifier needs segmentation of textured objects that the project does not have |
| G12 | PASS | BDDL goal predicates (On/In/Open/Close/Turnon) on simulator state define an independent Evaluator |
| G13 | PASS | official fixed initial-state files give a deterministic reset (hashed, not unpickled, in this audit); snapshot restore would be new |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G1, G3, G4, G5, G6, G8, G9, G10; unverifiable: G11

## DV1_STACK_ORDER_IN_SHARED_BASKET

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: D0_ENV_WITH_LIBERO_ASSETS
tasks: derived: two grocery items, one public basket

- state: two items on the table, empty basket; PLACE drops at a fixed basket site
- candidate_A: PLACE the sturdy/heavy item first
- candidate_B: PLACE the fragile/light item first
- both_legal: 2 legal first actions: either item can be placed first
- persistent_effect: pile order in the basket: the second item lands on the first
- downstream_effect: tipping or ejection of the top item would force re-pick (extra skills)
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: needed: no single rule decides every state
- semantic_relation: sturdy/heavy/large first is a natural commonsense relation readable from the image and task language
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: present
- case_coverage_note: helpful/neutral/reversed can be designed by choosing item pairs, but the physical existence of the effect is not shown (not established statically)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: a 'larger first' rule fails on pairs where size and fragility disagree; this can be designed in, but the real effect size is unmeasured

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: either item can be placed first |
| G2 | PASS | no contract relation orders the two items (the drop site occupancy is not a predicate) |
| G3 | UNVERIFIABLE_STATICALLY | pile heights per item (asset z extents, orientation unverified): {"alphabet_soup": {"min_height": 0.062, "max_height": 0.076}, "bbq_sauce": {"min_height": 0.029, "max_height": 0.107}, "butter": {"min_height": 0.017, "max_height": 0.076}, "chocolate_pudding": {"min_height": 0.027, "max_height": 0.08}}; basket contain_region depth 0.139 m. Whether the second drop topples or ejects an item is a contact-dynamics question. |
| G4 | UNVERIFIABLE_STATICALLY | success of In(basket) and the rework rate need a physical probe; the static asset data cannot establish them |
| G5 | PASS | difference is not a pure path distance: an ejected item needs a re-pick, which is extra skills not path |
| G6 | PASS | sturdy/heavy/large first is a natural commonsense relation readable from the image and task language |
| G7 | PASS | weight/fragility are not in the contract vocabulary |
| G8 | UNVERIFIABLE_STATICALLY | helpful/neutral/reversed can be designed by choosing item pairs, but the physical existence of the effect is not shown |
| G9 | PASS | a 'larger first' rule fails on pairs where size and fragility disagree; this can be designed in, but the real effect size is unmeasured |
| G10 | FAIL | only 0 of 10 grocery assets are orientation-agnostic pinchable (none); 4 more would be pinchable if spawn orientation were verified (bbq_sauce,butter,chocolate_pudding,cream_cheese). hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per-object grasp heights and container binding are new |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G10; unverifiable: G3, G4, G8, G11, G13

## DV2_SUPPORT_REMOVE_TOP_FIRST

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: D0_ENV_WITH_LIBERO_ASSETS
tasks: derived: item A resting on item B, goals vary over A, B or both

- state: A rests on B
- candidate_A: move A first
- candidate_B: pick B directly
- both_legal: 1 legal first actions: picking B is blocked while A rests on it (hard); the only legal first move for a B-goal is A
- persistent_effect: A removed from the support of B
- downstream_effect: B cannot be picked while A rests on it
- hard_precondition_or_not: HARD: Clear(B) / not supported-by(A, B)
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: support relation is visible
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: A-only goals are neutral, B goals are forced (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: remove whatever rests on a needed object

| gate | status | reason |
|---|---|---|
| G1 | FAIL | 1 legal first high-level actions in the decision state: picking B is blocked while A rests on it (hard); the only legal first move for a B-goal is A |
| G2 | FAIL | hard precondition decides the order (Clear(B) / not supported-by(A, B)): belongs in PRE/ADD/DEL, the contract planner returns the winner |
| G3 | PASS | support relation persists, but it is a hard precondition |
| G4 | PASS | gates the second action |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | support relation is visible |
| G7 | FAIL | Support is a precondition predicate, i.e. a contract fact |
| G8 | PASS | A-only goals are neutral, B goals are forced |
| G9 | FAIL | remove whatever rests on a needed object |
| G10 | FAIL | stacked starts violate the OnTable contract and the Verifier OnTable bound; hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per-object grasp heights and container binding are new |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G1, G2, G7, G9, G10; unverifiable: G11, G13

## DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: D0_ENV_WITH_LIBERO_ASSETS
tasks: derived: a neighbour placed inside the pinch envelope of the goal object

- state: neighbour next to the goal object
- candidate_A: clear the neighbour first
- candidate_B: grasp directly
- both_legal: 2 legal first actions: clearing is legal; grasping directly is legal only if the envelope is free (safety mask)
- persistent_effect: neighbour moved away
- downstream_effect: the grasp collides with the neighbour or not
- hard_precondition_or_not: HARD: finger-envelope collision is a Safety/contract-level hard mask; the BSI, OCC and VIS preflights on the same Panda gripper found no safe soft band
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: adjacency is visible
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: far neighbours are neutral (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: clear any neighbour within the envelope

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: clearing is legal; grasping directly is legal only if the envelope is free (safety mask) |
| G2 | FAIL | hard precondition decides the order (finger-envelope collision is a Safety/contract-level hard mask; the BSI, OCC and VIS preflights on the same Panda gripper found no safe soft band): belongs in PRE/ADD/DEL, the contract planner returns the winner |
| G3 | PASS | neighbour position persists |
| G4 | PASS | collision or not |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | adjacency is visible |
| G7 | FAIL | adjacency/collision is a geometric precondition the Safety module encodes |
| G8 | PASS | far neighbours are neutral |
| G9 | FAIL | clear any neighbour within the envelope |
| G10 | FAIL | PICK and PLACE_BUFFER would suffice but hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per-object grasp heights and container binding are new |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G2, G7, G9, G10; unverifiable: G11, G13

## DV4_SHARED_DESTINATION_CAPACITY

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: D0_ENV_WITH_LIBERO_ASSETS
tasks: derived: a plate that takes one item, a goal-relevant item and a goal-irrelevant item

- state: goal item and distractor both on the table, small destination
- candidate_A: place the goal item first
- candidate_B: place the distractor first
- both_legal: 2 legal first actions: placing the distractor on the destination is legal
- persistent_effect: destination occupied
- downstream_effect: the goal item no longer fits
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: goal-relevance of an item is visible in language
- helpful_case: present
- neutral_case: absent
- reversed_or_harmful_case: present
- case_coverage_note: distractor-first is only ever harmful (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: never place a non-goal item on the destination

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: placing the distractor on the destination is legal |
| G2 | FAIL | the goal contract only wants the goal item at the destination; B_PLAN never plans the distractor |
| G3 | PASS | occupied destination |
| G4 | PASS | blocks the goal item |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | goal-relevance of an item is visible in language |
| G7 | FAIL | relevance is the goal itself |
| G8 | PASS | distractor-first is only ever harmful |
| G9 | FAIL | never place a non-goal item on the destination |
| G10 | FAIL | hosting LIBERO assets inside the project's D0 platform keeps the controller, but LIBERO objects are textured and non-cubic, so perception (colour override), per-object grasp heights and container binding are new |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G2, G7, G9, G10; unverifiable: G11, G13

## DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: LIBERO_ENV
tasks: derived: official scene, goal switched between objects

- state: same scene, goal names one of several objects
- candidate_A: act on the named object
- candidate_B: act on another object
- both_legal: 2 legal first actions: acting on a non-goal object is legal
- persistent_effect: none
- downstream_effect: none beyond cost
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: language names the object
- helpful_case: present
- neutral_case: absent
- reversed_or_harmful_case: present
- case_coverage_note: wrong-object-first is only harmful (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: act on the goal object

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: acting on a non-goal object is legal |
| G2 | FAIL | the goal contract names the object |
| G3 | FAIL | no persistent effect on the goal chain |
| G4 | FAIL | only extra cost for the wrong object |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | language names the object |
| G7 | FAIL | goal grounding |
| G8 | PASS | wrong-object-first is only harmful |
| G9 | FAIL | act on the goal object |
| G10 | FAIL | the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G2, G3, G4, G7, G9, G10; unverifiable: G11, G13

## DV6_ARTICULATED_OPTIONAL_PREPARATION

origin: LIBERO_DERIVED_DIAGNOSTIC_TASK, host: LIBERO_ENV
tasks: derived: drawer opened early to stage an item that a later goal may need

- state: closed drawer, goal may or may not need it
- candidate_A: open the drawer early
- candidate_B: wait
- both_legal: 2 legal first actions: opening is legal at any time
- persistent_effect: drawer open
- downstream_effect: saves or wastes a skill
- hard_precondition_or_not: NOT_HARD
- candidate_consequence_role: none: a fixed rule already decides
- semantic_relation: whether a later goal needs the drawer is expressible
- helpful_case: present
- neutral_case: present
- reversed_or_harmful_case: absent
- case_coverage_note: harmful when the drawer blocks nothing but costs a skill (established)
- expected_extra_skills: not established statically
- expected_effect_on_J: not established statically
- simple_rule_attack: open iff a pending goal needs the drawer: a one-line rule on the goal text

| gate | status | reason |
|---|---|---|
| G1 | PASS | 2 legal first high-level actions in the decision state: opening is legal at any time |
| G2 | PASS | optional preparation has no contract winner |
| G3 | PASS | drawer state persists |
| G4 | PASS | saves the later OPEN when a later goal needs the drawer |
| G5 | PASS | difference is not a pure path distance:  |
| G6 | PASS | whether a later goal needs the drawer is expressible |
| G7 | FAIL | a later In(drawer) goal makes Open(drawer) a hard PRE of that goal; the 'optional' reading disappears once the goal is stated |
| G8 | PASS | harmful when the drawer blocks nothing but costs a skill |
| G9 | FAIL | open iff a pending goal needs the drawer: a one-line rule on the goal text |
| G10 | FAIL | sliding drawer needs a handle-pull skill; the project has no LIBERO environment binding (0 references to libero/BDDL in src), its perception is a colour-blob model for D0 cubes, its Verifier/Evaluator are D0-specific and its snapshot module is D0-only; every LIBERO-hosted task needs a new platform binding that is not a limited per-object binding |
| G11 | UNVERIFIABLE_STATICALLY | postconditions (Held, In basket) need segmentation of the chosen LIBERO assets; colour override of materials is a design choice, not an existing component |
| G12 | PASS | BDDL-style predicates on simulator state give an independent Evaluator |
| G13 | UNVERIFIABLE_STATICALLY | the project's snapshot restore is D0-only; LIBERO assets inside D0 would need the restore contract re-validated in a physical card |
| G14 | PASS | first mechanism canary needs about 4 episodes |
| G15 | PASS | task selection uses BDDL structure and asset geometry only; no score, outcome or demonstration was read |

failed: G7, G9, G10; unverifiable: G11, G13
