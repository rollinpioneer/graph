# CP-DISR-TP-OCC-PREFLIGHT-1 鈥?final preflight summary

Verdict: **PREFLIGHT_FAIL_NO_SOFT_OCCLUSION_BAND**  (stopped at phase F)

Phases: {'A': 'PASS', 'B': 'PASS', 'C': 'PASS', 'D': 'PASS', 'E': 'PASS', 'F': 'FAIL'}

## C/D. camera model vs real clean dev blobs

- blobs 20/20; centroid error mean 0.19 px, max 0.56 px; area ratio real/pred 0.82-1.23; mean mask IoU 0.93
- metric depth fit (informational, not gating): model form ok True, median error 0.00019 m, all-pixel RMS 0.0070 m (edge pixels)
- the colour mask captures the top face of each cube only

## E/F. world-safe projected overlap and soft band

- OCCLUDE_TARGET: world-safe 4143/16875; with projected overlap 77; classes {'CLEAR': 77}; max centroid shift 0.0000 m; min top-face visible fraction 1.00
- OCCLUDE_SECOND: world-safe 4143/16875; with projected overlap 77; classes {'CLEAR': 77}; max centroid shift 0.0000 m; min top-face visible fraction 1.00
- structural check (all safety constraints ignored, touching layouts included): max top-face occluded fraction 0.0 over 2411 layouts
- explanation: a cube hides another cube's top face only if the line of sight to a point at the top height passes through it; that line is above the top height everywhere between the target point and the camera, so an equal-height cube can hide side faces but never the top face, which is the only face the colour mask sees (calibrated).
- spread axis (from saved proprioception): world index 1 (alignment 1.000); grasp-affecting shift lower bound 0.0119 m

Budget: environment constructions 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test reads 0.
