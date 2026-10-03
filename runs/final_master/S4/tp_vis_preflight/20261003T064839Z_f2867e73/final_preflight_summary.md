# CP-DISR-TP-VIS-PREFLIGHT-1 鈥?final preflight summary

Verdict: **PREFLIGHT_FAIL_NO_VIS_A_SOFT_BAND**  (stopped at phase H)

Phases: {'A': 'PASS', 'B': 'PASS', 'C': 'PASS', 'D': 'PASS', 'E': 'PASS', 'F': 'PASS', 'G': 'PASS', 'H': 'FAIL'}

## A. OCC evidence binding

- status PASS; calibration reproduced exactly: True; production sources unchanged: True

## D. tall heights under the unmodified production code

- H=0.06 m: survives=True []
- H=0.08 m: survives=False ['REQUIRES_VERIFIER_CHANGE_ONTABLE']
- H=0.10 m: survives=False ['REQUIRES_VERIFIER_CHANGE_ONTABLE']
- H=0.12 m: survives=False ['REQUIRES_VERIFIER_CHANGE_ONTABLE']

## E. colour candidates

- COLOR_CANDIDATE_STATIC_PASS: selected c1_interferer_cyan; passing ['c1_interferer_cyan']

## F. projection regression

- equal heights reproduce the OCC zero: True over 2411 layouts; a 0.12 m occluder reaches a top-face occlusion of 1.00 (machinery can occlude)

## G/H. VIS-A layouts and soft band

- H=0.06: total 16875, world-safe 2818, camera-visible 2813, partial occlusion 165, soft band 0, classes {'CLEAR': 165}; top rejection reasons {'TALL_OUTSIDE_PROVEN_WORKSPACE': 7758, 'CLOSED_FINGER_HITS_OTHER': 2193, 'OBJECTS_TOUCH_OR_OVERLAP': 2187, 'HAND_BODY_HITS_OTHER': 1251}
- H=0.08: total 16875, world-safe 2239, camera-visible 2226, partial occlusion 269, soft band 0, classes {'CLEAR': 269}; top rejection reasons {'TALL_OUTSIDE_PROVEN_WORKSPACE': 7758, 'CLOSED_FINGER_HITS_OTHER': 2945, 'OBJECTS_TOUCH_OR_OVERLAP': 2187, 'HAND_BODY_HITS_OTHER': 1103}
- H=0.1: total 16875, world-safe 1993, camera-visible 1974, partial occlusion 326, soft band 8, classes {'CLEAR': 316, 'HARD_DETECTION_LOSS': 1, 'SOFT_OCCLUSION_CANDIDATE': 8, 'UNKNOWN': 1}; top rejection reasons {'TALL_OUTSIDE_PROVEN_WORKSPACE': 7758, 'CLOSED_FINGER_HITS_OTHER': 3259, 'OBJECTS_TOUCH_OR_OVERLAP': 2187, 'HAND_BODY_HITS_OTHER': 1048}
- H=0.12: total 16875, world-safe 1806, camera-visible 1774, partial occlusion 342, soft band 27, classes {'CLEAR': 296, 'HARD_DETECTION_LOSS': 15, 'SOFT_OCCLUSION_CANDIDATE': 27, 'UNKNOWN': 4}; top rejection reasons {'TALL_OUTSIDE_PROVEN_WORKSPACE': 7758, 'CLOSED_FINGER_HITS_OTHER': 3462, 'OBJECTS_TOUCH_OR_OVERLAP': 2187, 'HAND_BODY_HITS_OTHER': 988}

- ceiling with EVERY safety constraint removed (overlap forbidden only), max centroid shift while the cube is still detected: {'0.06': 0.0173, '0.08': 0.0199, '0.1': 0.0212, '0.12': 0.0212}
- zone [0.0119, 0.0279) m; gating heights [0.06]; WEAK/MODERATE bins {}
- max centroid shift per height (informational included): {'0.06': 0.0, '0.08': 0.0065, '0.1': 0.0199, '0.12': 0.0281}

Budget: environment constructions 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test reads 0.
