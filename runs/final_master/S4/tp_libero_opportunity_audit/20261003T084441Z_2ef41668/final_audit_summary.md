# CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1 - final audit summary

Verdict: **NO_LOW_COST_TP_CANDIDATE_FOUND**

Upstream: https://github.com/Lifelong-Robot-Learning/LIBERO.git @ 8f1084e3132a39270c3a13ebe37270a43ece2a01 (2025-03-15T20:13:56+08:00); task map sha256 0c950df0a785aa55; BDDL root (all suites) sha256 6d619a4065c50779; audit date 2026-10-03

## Mechanism cards

- DV1_STACK_ORDER_IN_SHARED_BASKET (LIBERO_DERIVED_DIAGNOSTIC_TASK, D0_ENV_WITH_LIBERO_ASSETS): failed G10; unverifiable G3,G4,G8,G11,G13
- DV2_SUPPORT_REMOVE_TOP_FIRST (LIBERO_DERIVED_DIAGNOSTIC_TASK, D0_ENV_WITH_LIBERO_ASSETS): failed G1,G2,G7,G9,G10; unverifiable G11,G13
- DV3_NEIGHBOUR_CLEARING_IN_FINGER_ENVELOPE (LIBERO_DERIVED_DIAGNOSTIC_TASK, D0_ENV_WITH_LIBERO_ASSETS): failed G2,G7,G9,G10; unverifiable G11,G13
- DV4_SHARED_DESTINATION_CAPACITY (LIBERO_DERIVED_DIAGNOSTIC_TASK, D0_ENV_WITH_LIBERO_ASSETS): failed G2,G7,G9,G10; unverifiable G11,G13
- DV5_GOAL_CONDITIONED_DISTRACTOR_RELEVANCE (LIBERO_DERIVED_DIAGNOSTIC_TASK, LIBERO_ENV): failed G2,G3,G4,G7,G9,G10; unverifiable G11,G13
- DV6_ARTICULATED_OPTIONAL_PREPARATION (LIBERO_DERIVED_DIAGNOSTIC_TASK, LIBERO_ENV): failed G7,G9,G10; unverifiable G11,G13
- M1_BASKET_TWO_ITEMS (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G9,G10; unverifiable G4,G8,G11
- M2_STOVE_TWO_POTS (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G6,G8,G9,G10; unverifiable G4,G11
- M3_TWO_TARGETS_DISTINCT_DESTINATIONS (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G3,G4,G5,G6,G8,G9,G10; unverifiable G11
- M4A_ARTICULATED_PLUS_PLACE (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G6,G8,G9,G10; unverifiable G3,G4,G11
- M4B_ARTICULATED_HARD_PRECONDITION (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G1,G2,G7,G8,G9,G10; unverifiable G11
- M5_SPATIAL_SINGLE_GOAL_DISTRACTORS (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G1,G3,G4,G5,G9,G10; unverifiable G11
- M6_OBJECT_SINGLE_GOAL_SAME_SCENE (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G1,G3,G4,G5,G9,G10; unverifiable G11
- M7_SINGLE_GOAL_OTHER (UNMODIFIED_PUBLIC_TASK, LIBERO_ENV): failed G1,G3,G4,G5,G6,G8,G9,G10; unverifiable G11

## Public validation coverage

- strict coverage 0.000 (exact 0, binding 0 of 40); upper bound 0.150; label LIBERO_COMPATIBLE_SUBSET

## Nearest misses

- DV1_STACK_ORDER_IN_SHARED_BASKET: failed G10; unverifiable G3,G4,G8,G11,G13
- M1_BASKET_TWO_ITEMS: failed G9,G10; unverifiable G4,G8,G11
- DV6_ARTICULATED_OPTIONAL_PREPARATION: failed G7,G9,G10; unverifiable G11,G13

## Decisions that would change the outcome

- authorise a physical probe (environment construction) so that the unverifiable persistent-effect gates of DV1 and M1 can be measured; this card is forbidden to do so
- authorise new scripted skills (yaw-aligned grasp, rim grasp, drawer pull, knob) and a LIBERO environment binding; without them G10 fails for every LIBERO-hosted task in the four audited suites
- authorise hosting a derived family inside the D0 platform with LIBERO assets (still needs the physical probe)
- authorise a LIBERO-90 audit: it was not audited here, so nothing is claimed about it; the failures found in the four suites are skill- and mechanism-level, which more tasks of the same kind would not remove by themselves

Budget: environment constructions 0, env.reset 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, score reads 0, demo replay 0.
