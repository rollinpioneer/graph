# CP-DISR-TP-BSI-PREFLIGHT-1 — final preflight summary

Verdict: **PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND**  (stopped at phase F)

Phases: {'A': 'PASS', 'B': 'PASS', 'C': 'PASS', 'D': 'PASS', 'E': 'PASS', 'F': 'FAIL'}

## A. real skill durations (clean logs)

- PICK_second: n=6 mean=4.425 median=4.400 std=0.240 min=4.20 max=4.85
- PICK_target: n=6 mean=5.250 median=5.600 std=0.747 min=4.20 max=5.85
- PLACE_BUFFER_second: n=6 mean=3.550 median=3.550 std=0.000 min=3.55 max=3.55
- PLACE_target: n=6 mean=2.900 median=2.900 std=0.000 min=2.90 max=2.90

## B. discrete rework bound

- BLOCK_BUFFER: extra two skills = 7.97 s (range 7.75-8.40); p_min for 2.1 s = 0.263; E[dT] at failure p=0.3 / 0.8 = 2.39 / 6.38
- BLOCK_CONTAINER: extra two skills = 8.15 s (range 7.10-8.75); p_min for 2.1 s = 0.258; E[dT] at failure p=0.3 / 0.8 = 2.44 / 6.52
- median-gate note: the 40-episode gate 'median delta_T > 2.1 s' per scene is passed only if >= 3 of 5 bad-order seeds fail their first placement; with failure probability p inside the design band this is not guaranteed

## F. static sweep / soft-band audit

- BLOCK_BUFFER: valid interfering candidates 49, WEAK/MODERATE 48, SAFE 0; causes {'UNCERTIFIED_FINGER_CLEARANCE_BELOW_PROVEN_ENVELOPE': 48}
- BLOCK_CONTAINER: valid interfering candidates 9, WEAK/MODERATE 8, SAFE 0; causes {'COLLISION_FINGERS_WITH_container_wall': 8, 'COLLISION_HELD_BLOCKER_LIFT_WITH_container_wall': 8}
- category: PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND
- proven finger-clearance envelope: 0.0098 m

Budget: environment constructions 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test reads 0.
