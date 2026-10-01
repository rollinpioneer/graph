# Family B gripper-clearance evidence note (B line, offline, read-only)

Verdict: `INSUFFICIENT_FOR_SWEPT_CLEARANCE`
(The V3-R1 failure mechanism IS directly recorded; the swept clearance of any layout, including the failed one and any
alternative, can NOT be established from the existing records. Nothing below is a safe/verified coordinate.)

Scope: existing JSON/JSONL/CSV/source/XML text only. No simulator, GPU, env construction, git write, or test-split data.
Frozen numbers are quoted as recorded; "derived" = arithmetic on recorded numbers; "NOT AVAILABLE" = not in any file I read.

Paths (abbreviated):
- R1  = /home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/S4/family_b_v3r1_reach_safe_mirror/20261001T073554Z_80a6b7d6
- V3  = .../S4/family_b_v3_isochronous/20261001T070045Z_74a422a2
- STG = .../S4/family_b_staging_v2/20261001T053917Z_3064ee8b ; OBS2 = .../S4/family_b_observation_v2_r2/20261001T045653Z_3c98a096
- SRC = /home/xushijie2/graph_cp_disr_final_tb_launch/src/cp_disr/platforms/libero
- XML = /home/xushijie2/envs/lerobotpi0-xfs/lib/python3.10/site-packages/robosuite/models/assets/grippers/panda_gripper.xml

## 1. What collided, in which phase (recorded)

Only ONE skill call ran in each of V3 and V3-R1 (both: layout_0 / B_PENDING / candidate pad_u, step A1, first setup action
`a:PICK:obj_c:v1`; R1/branch_results.csv, V3/branch_results.csv; "live skill calls 1/48" in both final_summary.md).
So PLACE_BUFFER, release, retreat and carry-back were never executed in V3 or V3-R1.
Phases that ran: HOVER, DESCEND, PRESS, CLOSE, LIFT_SHOW (PICK only). Sim step = 0.05 s (SRC/clock.py:9, CONTROL_DT=1/20).

### 1a. V3-R1 (R1/captures/21b32e647802715c/action_00/contacts.jsonl + controller_trace.jsonl; status SCRIPT_DIVERGENCE_MASK)
obj_c at anchor A_V = (-0.00350163, -0.25038683); pad_v = (0.00824918, -0.16519342) (R1/captures/.../layout.json).
Contact rows (aggregated by me from contacts.jsonl; "step" = rows' `step` field; distance<0 = penetration):

| phase | steps (sim s) | geom pair (geom1 | geom2) | n rows | min dist | max normal force |
|---|---|---|---|---|---|---|
| DESCEND | 36-46 (2.80-3.30) | `pad_v` | `gripper0_finger2_collision` (body gripper0_rightfinger) | 11 | -2.52 mm | 480.8 N |
| PRESS | 50-126 (3.50-7.30) | same pair | 76 | -1.99 mm | 439.4 N |
| CLOSE | 127-159 (7.35-8.95) | same pair | 17 | -0.47 mm | 25.9 N |
| CLOSE | 132-142 | `obj_c_g0` | `gripper0_finger1_collision` | 11 | -7.97 mm | 5.9 N |
| CLOSE | 135-162 (7.75-9.10) | `pad_v` | `obj_c_g0` (cube dragged against pad) | 71 | -3.47 mm | 2.3 N |
| CLOSE / LIFT_SHOW | 143-180 | finger1_pad_collision | finger2_pad_collision (pads touching each other = nothing between them) | 64 / 51 | -1e-5 m | 0.8 N |
| LIFT_SHOW | 163-180 | `pad_v` | `obj_c_g0` | 22 | -1.60 mm | 1.0 N |

Controller trace (phase_begin/phase_end events, same dir):
- First finger2-pad_v contact at step 36: eef (-0.0045,-0.2498,0.8417); pad_v top z = 0.829+0.004 = 0.833 (qa_state.jsonl pad size [0.045,0.045,0.004]).
  DESCEND ended at eef z 0.8355 (target 0.8291). PRESS target z 0.8211 but eef z stayed 0.832-0.848; dpos_z = -1.0 (saturated) the whole phase;
  PRESS consumed all 80 max_steps = 4.0 s (3.30 -> 7.30 s). Hand could not descend (blocked on the pad rim).
- CLOSE (gripper_qpos at end [0.0013, 0.0003] = fully closed, cube NOT between the fingers); eef drifted (-0.0004,-0.2467,0.8479) -> (-0.0324,-0.1901,0.8441)
  (dy = +0.057 m toward pad_v) while closing.
- Final cube pose (qa_state.jsonl row 1 hidden_truth): obj_c = (-0.0063,-0.2384,0.8288) vs rest (-0.0035,-0.2504,0.8199): moved 0.012 m toward pad_v, +8.9 mm z.
  Cube never held; public `p:AtBuffer:obj_c:pad_v` = TRUE and `p:Held:obj_c` UNKNOWN (R1/final_summary.md; captures/.../snapshot.json fact_values).
  The AtBuffer predicate is xy-proximity only: |dx|,|dy| <= PAD_HALF 0.045 + buffer_margin 0.03 and z < table_z+0.10 (SRC/family_b_adapters.py:138-142);
  final |dy| = 0.0732 <= 0.075, so a cube lying on the table 8 mm beside the pad satisfies it.
- Result: next scripted action `a:PLACE:obj_c:receiver:v1` masked off -> SCRIPT_DIVERGENCE_MASK; run stopped (R1/mechanism_gate.json: stop.step A1).
No contact with carrier, obj_b, pad_u, receiver, or table appears in the recorded pairs for this action.

Failure phase / geom pair (V3-R1): DESCEND->PRESS (persisting into CLOSE), right finger collision mesh (finger2_collision) vs the static pad_v box rim.

### 1b. V3 (V3/captures/dc61f9bb75eaf9ab/action_00; status SKILL_FAILURE:TIMEOUT)
obj_c at (0.22,-0.18); pad_v at (0.12,-0.18) (V3/captures/.../layout.json).
- Recorded gripper contacts: only obj_c (palm `gripper0_hand_collision`, `finger1/2_collision`, both finger pads) and
  `table_collision` vs `gripper0_finger2_collision` (CLOSE steps 53-87, max 10.6 N, min dist -0.04 mm). No pad_u/pad_v/obj_b/carrier row exists.
- HOVER/DESCEND/PRESS/CLOSE completed (19 contacts with obj_c). LIFT_SHOW (steps 88-180) toward (0.02,-0.08,0.985) stalled at eef
  (0.2001,-0.0810,0.9906), dpos_x saturated at -1.0 (trace step 178: dpos [-1.0, 0.12, -0.67]); the last 19 contacts are all cube-in-gripper
  (contact_count 19). So the V3 failure is reach/kinematic stall with cube held (base radius 0.80 m, V3/final_summary.md), NOT a gripper collision
  with a neighbour. The skill then ended at sim t=10.0 s (started 1.0 s; 9.0 s PICK timeout in the runtime manifest skill_timeouts).
- Palm-cube (`hand_collision`) contact (-0.82 mm) and finger-table contact appeared during CLOSE; both also occur in nominal successful runs (section 2e), so they are not by themselves the V3 cause.

## 2. Swept space: what is recorded / derived / not available

### 2a. Open-gripper footprint (recorded poses of collision geoms; V3 and R1 qa_state.jsonl row 0 are identical, eef (-0.1028,0.0000,1.0110), qpos [0.0392,-0.0392])
Source: R1/captures/21b32e647802715c/action_00/qa_state.jsonl `gripper_geoms` (geom xpos/xmat/size). For MESH geoms (type 7) `size` is the mesh
bounding-box half extent in the geom frame, NOT the true hull. World half-extent below = |R|*size (conservative upper bound, computed by me).

| geom (open) | centre rel. eef (x,y,z) m | world bbox half-extent (x,y,z) m | status |
|---|---|---|---|
| hand_collision (palm) | (-0.0093, +0.0006, +0.0725) | (0.038, 0.105, 0.055) | bbox bound; width ~0.21 m along y |
| finger1_collision (mesh) | (-0.0026, -0.0512, +0.0207) | (0.015, 0.019, 0.0345) | bbox bound |
| finger2_collision (mesh) | (-0.0026, +0.0512, +0.0206) | (0.0146, 0.019, 0.0346) | bbox bound |
| finger1/2_pad_collision (box, XML half 0.008x0.004x0.008) | (-0.0004/-0.0005, -/+0.0427, +0.0036) | (0.009, 0.004, 0.009) | directly recorded box |

Derived from these (and XML finger_joint range 0..0.04 / -0.04..0, XML pad box pos/size):
- Opening axis = world y (finger centres at y = +-0.0512 about the eef; same axis on which the pads/anchors are collinear).
- Gap between pad inner faces when open = 2*(0.0427-0.004) = 0.0774 m (XML max 0.079); cube width 0.040 => 0.0187 m slack per side if perfectly centred.
- Finger mesh y-extent from the grip axis: inner face between 0.032 and 0.0405, outer face between 0.062 (using local half 0.0107, qa_state `size`[0]) and 0.0702 (bbox bound).
  Lowest point of the finger mesh: about eef_z - 0.010 (from the step-36 contact: eef 0.8417, pad top 0.833, penetration 1.4 mm) to eef_z - 0.0138 (bbox bound).
  Cross-check: V3 eef z reached 0.8099-0.8121 and finger2-table contact appears; cube rest z 0.8199 implies table surface ~0.7999 (derived, not recorded) = eef - 0.010..0.012. Consistent.
- Palm lower surface >= eef_z + 0.0176 (bbox bound). Palm-cube contact is recorded, so the palm does reach cube-top height at PRESS.
- The nominal `table_top_z` value is 0.825 (qa_state hidden_truth), but cube centres rest at z=0.8199 (cube bottom ~0.7999): pads/receiver are built at
  z0 = table_top_z (SRC/family_b_env.py:~60-73: pad spans z 0.825-0.833), i.e. the pad top is ~0.033 m above the cube-support plane, cube top 0.040 m.
  (Table surface z itself is not a recorded field; inferred.)

### 2b. Descent envelope vs the layout (derived)
- Design only enforces planar AABB gaps between cube/pad/receiver boxes: min 0.020193 m (R1/aabb_clearance_audit.json `min_gap_m`; required 0.008).
  No gripper footprint enters that audit (aabb_clearance_audit.json `half_sizes` lists cube 0.02, pad 0.045, receiver only).
- Anchor-pad geometry is built collinear with the finger-opening axis: obj_c centre to pad_v nearest edge along y = 0.08519 - 0.045 = 0.0402 m
  (the 0.0202 AABB gap + cube half 0.02). The open finger2 inner face sits at 0.032-0.0405 from the grip axis and its outer face at 0.062-0.070,
  i.e. the finger2 mesh footprint lies at/over the pad rim (pad y-range starts 0.0402 from the cube centre). x offset obj_c-pad_v is only 0.0118 m,
  pad x half 0.045, finger x half 0.015: overlap in plan for any eef xy within a few mm of the cube centre.
- Vertical: grasp target z = min(perc_z-0.012, table_top_z+0.02+0.002) (SRC/skill_executor.py:297-300) -> recorded DESCEND target 0.8291 / PRESS 0.8211
  => finger bottom ~0.819 / 0.811 (derived) vs pad top 0.833: 14-22 mm vertical overlap. Hence a vertical descent through the pad rim is expected; recorded contact agrees.
- Nominal runs for contrast (STG, OBS2 layouts differ: pad_u (-0.2,0.12), pad_v (0.2,-0.02), layout.json): closest cube-pad planar edge gap 0.036 m
  (obj_c (0.133,-0.121) vs pad_v: dy 0.101, dx 0.067), with x-edge offset ~0.022 from the cube centre vs finger x half 0.015 -> ~7 mm plan-view x gap (derived, bbox).
  Across all 144 recorded actions (STG 120 + OBS2 24) there is ZERO gripper-geom contact with pad_u/pad_v (contacts.jsonl aggregation, see 2e).

### 2c. Closing sweep
- Recorded: gripper_qpos only at phase_begin/phase_end events (controller_trace.jsonl). CLOSE hold = 36 control steps = 1.8 s (SRC/skill_executor.py:340-341 `_hold(...,36)`).
  R1 CLOSE end qpos [0.0013,0.0003] (closed on nothing).
- Derived: each finger travels ~0.0187 m to meet a centred 0.040 m cube; closing is symmetric only if the hand is centred; in R1 eef moved dy=+0.057 m during CLOSE.
- NOT AVAILABLE: per-step finger qpos, per-step finger geom poses, cube yaw/tilt, so the actual closing sweep path cannot be reconstructed.

### 2d. Carry-back / retreat / PLACE_BUFFER / release (target generation + sparse records)
Skill law (SRC/skill_executor.py): `_toward` gain 6, clip 0.06, scale 20 per axis (lines 58-62) -> every axis saturates independently at |err|>=0.05 m,
so moves are not straight lines; `step_osc` commands rotation = 0 (SRC/d0_env.py:267) so yaw is only held by the OSC controller (not recorded per step).
- PICK (330-349): hover z = table_top+0.22 = 1.045 over target; grasp; press = grasp-0.008; close 36 steps; LIFT_SHOW to (0.02,-0.08,table_top+0.16=0.985), tol 0.03; fallback to hover.
- PLACE_BUFFER (370-387): hover over buffer at z 1.045 with gripper closed; drop z = table_top+0.02+0.05 = 0.895; open 16 steps; retreat vertically to hover with gripper open.
- Derived at release over pad_v (0.00825,-0.16519): finger bottom ~0.885 vs pad top 0.833 and a resting neighbour cube top 0.840 => ~0.045 m vertical separation.
  Open finger1 outer y-extent = -0.1652 - (0.062..0.0702) = -0.2272..-0.2354 vs obj_c +y face at -0.2304 (if still at A_V): plan-view result ranges from +3 mm clear to
  5 mm overlap depending on which half-extent is true. Vertical separation decides; plan-view alone is undecided.
- Derived LIFT_SHOW from A_V (hypothetical clean grasp): R1 trace steps 163-170 show y rising 0.0786 m while z rises 0.0549 m (z/y ~0.70, one run, started over the pad).
  Applying that to a 0.040 m y-travel to the pad_v rim (y = -0.2102) gives eef z ~0.857, finger/cube bottom ~0.843-0.847 (cube centre is ~0.008 ABOVE eef when held, V3 qa_state row 1),
  i.e. +10..+14 mm above pad top 0.833. Single-run extrapolation, OSC transient not modelled: sign probably positive but NOT established.
- Recorded evidence of PLACE_BUFFER/release/retreat exists only in STG/OBS2 (different layout): no gripper-pad contact, finger-vs-cube contacts only (2e).
- NOT AVAILABLE for V3/V3-R1 layouts: any PLACE_BUFFER, release, retreat or carry-back record; any pad_u side (A_U/pad_u) record; any layout_1 record.

### 2e. Nominal-run contact calibration (STG + OBS2, all TASK_SUCCESS runs: STG final_summary.md, OBS2 final_summary.md "TASK_SUCCESS" x4; my aggregation of contacts.jsonl)
| skill / phase | gripper geom vs other | branches | min dist | max force |
|---|---|---|---|---|
| PICK CLOSE | finger1/2 + palm vs picked cube | 20+4 | -3.0 mm | 2.1 N |
| PICK CLOSE | finger1/2_collision vs table_collision | 20+4 | -0.5 mm | 43.1 N |
| PLACE_BUFFER DROP_DESCEND/RELEASE | finger1/2 + palm vs carrier cube | 20+4 | -2.6 mm | 2.1 N |
| PLACE RELEASE | finger2 vs `receiver_back` / finger1 vs `receiver_front` | 9 of 20 (+3 of 4) | -0.2 mm | 20.9 / 19.7 N |
| PLACE RETREAT | finger vs receiver wall | 3-4 of 20 | -0.1 mm | 21.2 N |
| any | gripper vs pad_u/pad_v | 0 of 144 actions | - | - |
Reading: palm/finger contact with the grasped object and finger-table contact are routine; finger contact with the receiver wall (wall inner half-width 0.035 m
along y, SRC/family_b_env.py:23 RECEIVER_INNER) also occurs in successful runs, i.e. open-finger footprint vs obstacle overlap of mm scale is not rare.
Recorder caveat: contacts.jsonl keeps only pairs whose category is finger/palm/target/interferer/lid (SRC/tp_sr_instrumentation.py:389-391, max 80 pairs); static pads are
tagged "finger" (cat1 of `pad_v` in contacts rows), so gripper-neighbour contacts are included but cube-cube/other contacts are not; no contact position or normal vector is stored.

### 2f. Quantity status summary
| quantity | status |
|---|---|
| open finger/palm geom centres, mesh bbox half-extents, pad box sizes at t0 and t_end | directly recorded (qa_state.jsonl) |
| finger pad box size/pos, joint ranges, eef offset 0.097 | XML text (panda_gripper.xml) |
| finger/palm TRUE hull (STL), per-step geom poses, per-step finger qpos, yaw/tilt, contact points | NOT AVAILABLE (STL not read; qpos only hashed; contacts have no pos) |
| eef xyz at every control step, phase targets/tolerances | directly recorded (controller_trace.jsonl) |
| finger outer/inner y-extent, lowest finger z, slack, derived clearances | derived (+-2-8 mm uncertainty from bbox vs hull) |
| object pose before/after (xyz), cube orientation | xyz recorded at 2 instants only; orientation NOT AVAILABLE |
| PLACE_BUFFER/release/retreat/carry-back swept path in V3/V3-R1 layouts | NOT AVAILABLE (never executed) |

## 3. Why the verdict is INSUFFICIENT_FOR_SWEPT_CLEARANCE
Missing measurements (listed only; not proposing to run them):
1. True collision-hull extents of finger1/2_collision and hand_collision along y, x, z (mesh vertices), to replace the +-8 mm bbox/hull ambiguity (e.g. 0.062 vs 0.0702 outer reach).
2. Per-step gripper geom poses (or at least finger qpos, hand yaw/tilt) across HOVER/DESCEND/PRESS/CLOSE/LIFT/retreat to bound the swept volume instead of two snapshots.
3. Executed PLACE_BUFFER, release and retreat/carry-back records in the V3/V3-R1 geometry (pad_u and pad_v sides, both layouts), with neighbour (obj_b/obj_c/carrier/receiver) poses over time.
4. Contact positions/normals (to localise rim vs top contact) and cube orientation (to explain the +8.9 mm z / tilt at the end of R1).
5. The real table-surface z and the pad/receiver bottom z relative to it (inferred 0.025 m offset, not recorded).
6. Perception-offset statistics for the grasp xy (recorded: DESCEND end offset ~2.7 mm in y in R1) to size the lateral margin that must be added to any footprint.
7. A grasp pose set for the A_U side (mirror of A_V) to check whether finger1-vs-pad_u shows the same behaviour.

## 4. Two layout/direction hypotheses (hypotheses only; not verified, no coordinate claimed safe)
Failure basis: finger2 mesh rim contact with pad_v during DESCEND/PRESS because the pad edge is 0.0402 m from the cube centre along the opening axis (y).

H1 (rotate the anchor-pad axis away from the opening axis, e.g. place the pad edge ~0.04 m from the cube along x, the finger-thickness axis).
- Plausibly removes: the finger-rim/pad contact in DESCEND/PRESS (finger x half 0.015 vs 0.040 gap leaves ~0.025 m bbox slack; V3 had the pad 0.055 m edge-distance along x
  and recorded no pad contact, V3/captures/.../contacts.jsonl). Also removes cube being dragged toward pad during CLOSE (a consequence of the rim contact).
- Leaves/uncertain: palm half-width in x (0.038, bbox) over a pad edge 0.0402 away at z where the palm bottom (>= eef+0.0176) is near cube top; finger-table and palm-cube contacts (routine);
  the diagonal LIFT_SHOW crossing (2d, +10..+14 mm estimate, unestablished); yaw drift; the base-radius fairness the R1 design relied on
  (R1/reach_envelope_audit.json: collinear-normal layout gives equal U/V radii 0.5918 / 0.6102; an x-offset breaks that and V3's 0.80 m radius stalled the arm).
- Unknown: PLACE_BUFFER release/retreat path relative to the moved neighbour (never executed).

H2 (keep the y-axis layout but increase the cube-to-pad edge distance beyond the open-finger outer reach, i.e. > ~0.070 m plus margin vs 0.0402 m now;
  centre-to-centre > ~0.115 m vs 0.0852 m now).
- Plausibly removes: finger-over-pad-rim in plan view during descent (pad edge outside the 0.062-0.070 outer reach), independent of z.
- Leaves/uncertain: anchors move outward in radius (A_V radius 0.6102 m, cap 0.65 m, margin 0.0398, R1/reach_envelope_audit.json), so reach margin shrinks; pad-to-pad gap
  (0.0804 m now) or anchor-to-receiver gaps shrink if pads move inward; finger-table/palm-cube contacts and carry-back remain unmeasured; the 0.0187 m per-side slack still
  must absorb grasp-xy offset; the point-symmetry about the show pose (R1/geometry_invariants.json) would need re-derivation.
Comparison: both target the same recorded mechanism (finger2/pad_v rim); H1 changes the failing axis, H2 changes the margin on the same axis. Neither addresses what the
record cannot show (release/retreat/carry-back). Changing the show pose (0.02,-0.08) is not implied by the evidence: the recorded failure happened before LIFT_SHOW (the lift was already compromised).

## 5. What this does NOT show
- It does not show that any coordinate, layout or direction is collision-free; no swept clearance number is established.
- It does not show V3 failed by collision (V3 = reach stall with cube held) or that V3-R1's reach fix fails; the R1 reach worked (arm reached the show pose).
- It does not show how PLACE_BUFFER, release, retreat or carry-back behave in the V3/V3-R1 geometry; they were never run, and pad_u side / layout_1 were never run.
- It does not show the failure is repeatable beyond one attempt (A1 only, 1 of 8 attempts; R1/final_summary.md); no variance, seeds, or perception-noise sensitivity.
- It does not establish true finger/palm hull dimensions (bbox bound used) or orientation drift.
- The recorded `AtBuffer` TRUE for a cube lying beside the pad is a predicate property (margin 0.03), not evidence the cube was placed.
- It does not use or reflect any test-split data.
