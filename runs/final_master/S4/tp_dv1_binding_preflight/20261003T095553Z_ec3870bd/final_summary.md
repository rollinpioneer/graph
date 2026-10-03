# CP-DISR-TP-DV1-BINDING-PREFLIGHT-1 - final summary

Verdict: **DV1_FAIL_BASKET_GEOMETRY**  (label LIBERO_DERIVED_DIAGNOSTIC_TASK)

All failing categories (evaluated independently; the verdict is the first in the declared order): DV1_FAIL_BASKET_GEOMETRY, DV1_FAIL_FIXED_HEURISTIC_SUFFICIENT, DV1_FAIL_NO_HELPFUL_NEUTRAL_REVERSED_SET, ENGINEERING_UNRESOLVED

## Pass conditions

- 10_no_drawer_knob_door_controller: True
- 1_two_objects_one_generic_pick: True
- 2_one_pair_enters_basket: True
- 3_no_new_perception_model: True
- 4_verifier_public: True
- 5_evaluator_semantics_unchanged: True
- 6_snapshot_no_rewrite: True
- 7_no_order_specific_drop: False
- 8_two_pairs_refute_one_rule: False
- 9_helpful_neutral_reversed: False

## Pinchability (official spawn orientation, spread axis world y)

| object | status | pinch width yaw0 (m) | yaw90 (m) | contact height (m) | palm clearance (m) |
|---|---|---|---|---|---|
| bbq_sauce | GENERIC_PICK_YAW90 | 0.1071 | 0.0470 | 0.0160 | 0.0159 |
| butter | NEW_LOW_LEVEL_BEHAVIOR_REQUIRED | 0.0395 | 0.0762 | 0.0078 | 0.0276 |
| chocolate_pudding | GENERIC_PICK_YAW0 | 0.0463 | 0.0802 | 0.0160 | 0.0176 |
| cream_cheese | GENERIC_PICK_YAW0 | 0.0427 | 0.0812 | 0.0083 | 0.0272 |

## Marker pixel support (minimum 8)

| object | at A | at B | held (hand box) | in basket at drop |
|---|---|---|---|---|
| bbq_sauce | 84 | 172 | 23 | 165 |
| butter | 44 | 95 | 41 | 68 |
| chocolate_pudding | 59 | 120 | 40 | 102 |
| cream_cheese | 53 | 111 | 54 | 83 |

Best marker pixels inside the basket over the pre-registered basket grid: 217

## Basket

every order of every pair ends inside the cavity below the rim; the Evaluator is TRUE for both objects in all 12 ordered pairs, so the order cannot change success or create a public re-pick
- cavity depth 0.124 m, worst pile top 0.057 m, pessimistic pair floor fill 0.584
- pathways: {"ejection": false, "first_item_evicted": false, "tipping_out_of_cavity": false}

Budget: environment constructions 0, env.reset 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test 0, demo replay 0.
