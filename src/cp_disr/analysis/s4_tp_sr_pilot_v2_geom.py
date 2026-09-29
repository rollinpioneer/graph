"""Measured gripper geometry, predicted sweep model and bar-interferer design for the T_P_SR V2 prototype. Pure computation on saved Wave D QA snapshots:
no environment is constructed, nothing is reset, no provider is called."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cp_disr.platforms.libero.tp_sr_generator import (BUFFER_HALF, CONTAINER_OUTER_HALF, FOOTPRINT_MARGIN, LID_HALF, LID_OFFSET_X, OBJECT_HALF, TABLE_SKIN_HALF, TABLE_TOP_Z)

HOVER_DZ = 0.22             # skill_executor._hover_z
GRASP_Z = TABLE_TOP_Z + OBJECT_HALF + 0.002     # skill_executor._grasp_z (perceived z is biased high so the table-based term is the minimum)
PRESS_DZ = 0.008
SHOW_POSE = (0.02, -0.08, TABLE_TOP_Z + 0.16)
OBJ_ZRANGE = (TABLE_TOP_Z, TABLE_TOP_Z + 2 * OBJECT_HALF)
PHASES = ("approach", "descend", "press", "close", "lift_show")
COLLISION_GEOMS = ("hand_collision", "finger1_collision", "finger2_collision", "finger1_pad_collision", "finger2_pad_collision")
BAR = {"hx": 0.050, "hy": 0.030, "hz": OBJECT_HALF, "density": 1000.0, "friction": [1.2, 0.005, 0.0001]}     # fixed a priori, never tuned on outcomes
D_STEP = 0.0005
X_STEP = 0.001
X_MAX = 0.060
Z_GRAZE_MARGIN = 0.004      # vertices within 4 mm of the slab top only graze an object edge (soft-contact scale) and do not count as sweep overlap
MIN_INITIAL_GAP = 0.002
OWN_GRASP_MIN_CLEARANCE = 0.001
CONTROL_MIN_CLEARANCE = 0.020


GRIPPER_XML = "robosuite/models/assets/grippers/panda_gripper.xml"
MESH_GEOMS = ("hand_collision", "finger1_collision", "finger2_collision")
BOX_GEOMS = ("finger1_pad_collision", "finger2_pad_collision")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _gripper_model():
    import mujoco
    import robosuite
    return mujoco.MjModel.from_xml_path(str(Path(robosuite.__file__).parent / "models/assets/grippers/panda_gripper.xml"))


def _vertices(model, rec, eef):
    """World vertices of every gripper collision geom relative to the eef, from the saved QA pose (geom xpos/xmat) and the real collision mesh / box corners."""
    import mujoco
    out = {}
    for g in rec["gripper_geoms"]:
        name = g["geom"].replace("gripper0_", "")
        R, pos = np.asarray(g["xmat"], dtype=float).reshape(3, 3), np.asarray(g["xpos"], dtype=float)
        if name in MESH_GEOMS:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            mid = int(model.geom_dataid[gid])
            v = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]], dtype=float)
        elif name in BOX_GEOMS:
            sz = np.asarray(g["size"], dtype=float)
            v = np.array([[sx * sz[0], sy * sz[1], sz_ * sz[2]] for sx in (-1, 1) for sy in (-1, 1) for sz_ in (-1, 1)])
        else:
            continue
        out[name] = (v @ R.T + pos) - eef
    return out


def measure_gripper(out):
    """Open pose from the first QA record of a wave-D branch (gripper opened by the reset recipe); closed pose from a record where the target is held after PICK.
    Geometry is the REAL collision mesh / pad boxes placed with the saved geom poses (an earlier AABB model was 12 mm too tall at the palm and was replaced)."""
    out = Path(out)
    open_p = out / "captures/094286b08fcf5904/action_00/qa_state.jsonl"                # T_P_SR_pool_33 direct, 'before' record: reset pose, gripper open
    held_p = out / "captures/3c4f588df6e91b04/action_00/qa_state.jsonl"                # T_P_SR_pool_57 direct, 'after' record: target held at the show pose
    rec_o = [json.loads(x) for x in open_p.read_text().splitlines()][0]
    rec_c = [json.loads(x) for x in held_p.read_text().splitlines()][-1]
    model = _gripper_model()
    eef_o, eef_c = np.asarray(rec_o["eef_pos"], dtype=float), np.asarray(rec_c["eef_pos"], dtype=float)
    vo, vc = _vertices(model, rec_o, eef_o), _vertices(model, rec_c, eef_c)
    if rec_o["tag"] != "before" or rec_c["tag"] != "after" or set(vo) != set(MESH_GEOMS + BOX_GEOMS) or set(vc) != set(vo):
        raise RuntimeError("wave D QA snapshots do not carry the expected gripper collision geoms")
    for k in vo:
        if vo[k].shape != vc[k].shape:
            raise RuntimeError("open/closed vertex sets differ")
    p1o, p2o, p1c, p2c = vo["finger1_pad_collision"], vo["finger2_pad_collision"], vc["finger1_pad_collision"], vc["finger2_pad_collision"]
    inner_open, inner_closed = float(p2o[:, 1].min() - p1o[:, 1].max()), float(p2c[:, 1].min() - p1c[:, 1].max())
    ext = lambda v: {"x": [float(v[:, 0].min()), float(v[:, 0].max())], "y": [float(v[:, 1].min()), float(v[:, 1].max())], "z": [float(v[:, 2].min()), float(v[:, 2].max())]}
    doc = {"source": {"open_state": {"file": str(open_p.relative_to(out)), "sha256": _sha(open_p), "record": "before"},
                      "closed_state": {"file": str(held_p.relative_to(out)), "sha256": _sha(held_p), "record": "after"}, "gripper_xml": GRIPPER_XML},
           "eef_start": eef_o.tolist(), "finger_axis_world": "y (pads separate along world Y at the fixed top-down orientation)",
           "usable_opening_open_m": inner_open, "usable_opening_closed_on_cube_m": inner_closed,
           "pad_centre_spacing_open_m": float(abs(p2o[:, 1].mean() - p1o[:, 1].mean())), "pad_thickness_along_finger_axis_m": float(p1o[:, 1].max() - p1o[:, 1].min()),
           "geoms": {k: {"vertices": int(len(vo[k])), "offset_extent_open": ext(vo[k]), "offset_extent_closed": ext(vc[k])} for k in vo},
           "palm_bottom_above_eef_m": float(vo["hand_collision"][:, 2].min()),
           "note": "clearance uses vertices below the object-slab top; the palm bottom stays above the slab at every skill pose so only pads/fingers can meet an object"}
    doc["_verts"] = {k: (vo[k], vc[k]) for k in vo}
    return doc


def _seg(a, b, n, o0, o1):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    return a[None] + t[:, None] * (b - a)[None], np.linspace(o0, o1, n)


def sweep_points(eef_start, xy, n=21, o_end=1.0):
    """Piecewise-linear eef path of the unchanged PICK skill aimed at xy, with the opening fraction (0 open .. 1 closed on a cube; o_end < 1 when the closing
    stops on a wider object). Rows: phase, k, x, y, z, opening."""
    x, y = float(xy[0]), float(xy[1])
    hover = (x, y, TABLE_TOP_Z + HOVER_DZ)
    grasp = (x, y, GRASP_Z)
    press = (x, y, GRASP_Z - PRESS_DZ)
    rows = []
    for pi, (a, b, o0, o1) in enumerate([(eef_start, hover, 0, 0), (hover, grasp, 0, 0), (grasp, press, 0, 0), (press, press, 0, o_end), (press, SHOW_POSE, o_end, o_end)]):
        p, o = _seg(a, b, n, o0, o1)
        for k in range(n):
            rows.append([pi, k, *p[k], o[k]])
    return np.asarray(rows, dtype=float)


def _rel_footprints(geo, pts, zrange=OBJ_ZRANGE):
    """Per geom and sample: xy bounding box (relative to the eef) of the collision vertices that lie inside the slab and deeper than the graze margin."""
    rel = {}
    for name, (vo, vc) in geo["_verts"].items():
        lo, hi, ok = np.zeros((len(pts), 2)), np.zeros((len(pts), 2)), np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            o = pts[i, 5]
            v = (1 - o) * vo + o * vc
            z = pts[i, 4] + v[:, 2]
            m = (z < zrange[1] - Z_GRAZE_MARGIN) & (z > zrange[0] - 0.05)
            if m.any():
                lo[i], hi[i], ok[i] = v[m, :2].min(axis=0), v[m, :2].max(axis=0), True
        rel[name] = (lo, hi, ok)
    return rel


def clearance(geo, pts, box_xy, box_half_xy, o_end=1.0):
    """Signed planar clearance between the gripper collision vertices lying inside the object slab (deeper than the graze margin) and an axis-aligned box.
    Negative = predicted overlap. Returns (min clearance, argmin info, number of overlapping samples)."""
    cache = geo.setdefault("_rel", {})
    key = round(float(o_end), 6)
    if key not in cache:
        canon = sweep_points(geo["eef_start"], (0.0, 0.0), int(round(len(pts) / len(PHASES))), o_end)
        cache[key] = (canon, _rel_footprints(geo, canon))
    canon, rel = cache[key]
    if pts.shape != canon.shape or not np.allclose(pts[:, [0, 1, 4, 5]], canon[:, [0, 1, 4, 5]]):
        raise RuntimeError("sweep profile does not match the cached footprint profile")
    bx, by = float(box_xy[0]), float(box_xy[1])
    hbx, hby = float(box_half_xy[0]), float(box_half_xy[1])
    best, info, over = 1.0, None, 0
    for name, (lo, hi, ok) in rel.items():
        l, h = pts[:, 2:4] + lo, pts[:, 2:4] + hi
        gx = np.maximum(l[:, 0] - (bx + hbx), (bx - hbx) - h[:, 0])
        gy = np.maximum(l[:, 1] - (by + hby), (by - hby) - h[:, 1])
        cl = np.where((gx > 0) & (gy > 0), np.hypot(gx, gy), np.maximum(gx, gy))
        cl = np.where(ok, cl, np.inf)
        over += int(np.sum(ok & (cl <= 0)))
        i = int(np.argmin(cl))
        if cl[i] < best:
            best, info = float(cl[i]), {"geom": name, "phase": PHASES[int(pts[i, 0])], "sample": int(pts[i, 1]), "eef": pts[i, 2:5].tolist(), "opening": float(pts[i, 5])}
    return best, info, over


def _rect_ok(c, half, statics):
    cx, cy = c
    if max(abs(cx) + half[0], abs(cy) + half[1]) > TABLE_SKIN_HALF - 0.01:
        return False
    for _n, sc, sh in statics:
        if abs(cx - sc[0]) < half[0] + sh and abs(cy - sc[1]) < half[1] + sh:
            return False
    return True


def statics_for(container_xy, buffer_xy):
    lid = (container_xy[0] + LID_OFFSET_X, container_xy[1])
    return [("container", tuple(container_xy), CONTAINER_OUTER_HALF + FOOTPRINT_MARGIN), ("buffer", tuple(buffer_xy), BUFFER_HALF + FOOTPRINT_MARGIN), ("lid", lid, LID_HALF + FOOTPRINT_MARGIN)]


def design_scene(geo, layout, side, n=21, bar=BAR):
    """Deterministic design of one interference config and its offset control (no outcome data is used). The bar is a graspable slab of the cube height whose
    finger-axis width is <= 0.8 x the measured usable opening; its long axis runs along world X. Search grid: bar x-offset from the target (dx) and near-face
    offset along the finger axis (d). An interference config needs predicted target-sweep overlap (<= 0), a clear own grasp sweep (> 1 mm, evaluated with
    the fingers closing on the bar width), no initial penetration (>= 2 mm gap) and no intrusion once relocated to the buffer. The choice maximises
    min(target-sweep overlap depth, own-grasp clearance), ties to the smaller |dx|, then the smaller d."""
    t = np.asarray(layout["target_xy"], dtype=float)
    bh = (bar["hx"], bar["hy"])
    eef0 = np.asarray(geo["eef_start"], dtype=float)
    opening = geo["usable_opening_open_m"]
    if 2 * bar["hy"] > 0.8 * opening + 1e-12:
        return {"status": "STOPPED_ASYMMETRIC_INTERFERER_NOT_GRASPABLE", "reason": "bar short width exceeds 0.8 x usable opening"}
    o_bar = float(np.clip((opening - 2 * bar["hy"]) / (opening - geo["usable_opening_closed_on_cube_m"]), 0.0, 1.0))
    statics = statics_for(layout["container_xy"], layout["buffer_xy"])
    pts_t = sweep_points(eef0, t, n, 1.0)
    cube_half = (OBJECT_HALF, OBJECT_HALF)

    def bar_centre(dx, d):
        return np.array([t[0] + dx, t[1] + side * (d + bar["hy"])])

    def initial_gap(c):
        gx = abs(c[0] - t[0]) - (OBJECT_HALF + bar["hx"])
        gy = abs(c[1] - t[1]) - (OBJECT_HALF + bar["hy"])
        return float(np.hypot(gx, gy) if (gx > 0 and gy > 0) else max(gx, gy))

    def probe(dx, d):
        c = bar_centre(dx, d)
        c_t, i_t, ov_t = clearance(geo, pts_t, c, bh, 1.0)
        c_own, i_own, _ = clearance(geo, sweep_points(eef0, c, n, o_bar), t, cube_half, o_bar)
        c_after, _i, _o = clearance(geo, pts_t, layout["buffer_xy"], bh, 1.0)
        return c, c_t, i_t, ov_t, c_own, i_own, c_after

    cands = []
    nx = int(round(X_MAX / X_STEP))
    for kx in range(-nx, nx + 1):
        dx = kx * X_STEP
        for kd in range(int(round(0.020 / D_STEP)), int(round(0.080 / D_STEP)) + 1):
            d = kd * D_STEP
            c = bar_centre(dx, d)
            if initial_gap(c) < MIN_INITIAL_GAP or not _rect_ok(c, bh, statics):
                continue
            _c, c_t, _i, _ov, c_own, _io, c_after = probe(dx, d)
            if c_t <= 0 and c_own >= OWN_GRASP_MIN_CLEARANCE and c_after > 0:
                cands.append((min(-c_t, c_own), -abs(dx), -d, dx, d))
    if not cands:
        return {"status": "STOPPED_ASYMMETRIC_INTERFERER_NOT_GRASPABLE", "reason": "no bar placement gives predicted target-sweep overlap and a clear own grasp sweep"}
    best = max(cands)
    dx, d = best[3], best[4]
    c, c_t, i_t, ov_t, c_own, i_own, c_after = probe(dx, d)
    ctrl = None
    for kd in range(int(round(d / D_STEP)), int(round(0.30 / D_STEP)) + 1):
        dc = kd * D_STEP
        cc = bar_centre(dx, dc)
        cl, il, _ = clearance(geo, pts_t, cc, bh, 1.0)
        if cl >= CONTROL_MIN_CLEARANCE and _rect_ok(cc, bh, statics):
            ctrl = (dc, cc, cl, il)
            break
    if ctrl is None:
        return {"status": "STOPPED_ASYMMETRIC_INTERFERER_NOT_GRASPABLE", "reason": "no lateral offset gives the control clearance"}
    dc, cc, cl, il = ctrl
    return {"status": "OK", "side": side, "bar_half": [bar["hx"], bar["hy"], bar["hz"]], "own_grasp_opening_fraction": o_bar, "x_offset_from_target_m": dx,
            "interference": {"near_face_offset_m": d, "bar_xy": c.tolist(), "target_sweep_clearance_m": c_t, "target_sweep_overlap_samples": ov_t, "closest": i_t,
                             "own_grasp_sweep_clearance_m": c_own, "own_grasp_closest": i_own, "target_sweep_clearance_after_relocation_m": c_after,
                             "initial_gap_to_target_m": initial_gap(c)},
            "control": {"near_face_offset_m": dc, "bar_xy": cc.tolist(), "lateral_shift_m": dc - d, "target_sweep_clearance_m": cl, "closest": il,
                        "initial_gap_to_target_m": initial_gap(cc)},
            "search": {"candidates": len(cands), "objective": "max min(-target_sweep_clearance, own_grasp_clearance)"}}
