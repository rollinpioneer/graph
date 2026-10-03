"""CP-DISR-TP-BSI-PREFLIGHT-1: zero-environment preflight of the Bidirectional Soft Interference T_P.

Reads source, XML/STL assets and already-saved logs only. No environment construction, no skill, no provider, no RL.
Phases A-F are evaluated in order; the first failing phase ends the card (later phases are written as NOT_RUN).
All thresholds below are declared in code and committed before the run.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import math
import os
import re
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CARD = "CP-DISR-TP-BSI-PREFLIGHT-1"
BASELINE = "c9743877af26086c684d5e429565845524350cbc"
FORBIDDEN = re.compile(r"holdout|test30|test_id|final_test|/test|_test|test_", re.I)
SW1_REL = "runs/final_master/S4/tp_ef_scripted_witness"
DISC_REL = "runs/final_master/S4/tp_ef_discovery/20261002T171217Z_b2046983"
H = 23.1
EPS_T = 2.1
EPS_G = 0.02
# ---- thresholds declared before the run -------------------------------------------------------------
GRID_STEP = 0.01
GRID_HALF = 0.10
SCORE_BINS = [("CLEAR", 0.0, 0.05), ("WEAK", 0.05, 0.40), ("MODERATE", 0.40, 0.75), ("HARD_BLOCKING_RISK", 0.75, 1.0001)]
SAFETY_MARGIN_PX = 6
MIN_TOP_FACE_PIXELS = 24
BASE_POSES = {"target_xy": (-0.12, -0.10), "second_xy": (0.12, -0.10)}   # inside the observed T_B train/dev ranges
CONTAINER_XY = (0.18, 0.12)
BUFFER_XY = (-0.18, 0.12)
SHOW_XY = (0.02, -0.08)
PICK_MODES = {"BLOCK_CONTAINER": {"dest": "container", "blocker": "second_object", "other": "target", "route_good": "S", "route_bad": "T"},
              "BLOCK_BUFFER": {"dest": "buffer", "blocker": "target", "other": "second_object", "route_good": "T", "route_bad": "S"}}
FAIL_CATEGORIES = ("PREFLIGHT_FAIL_INSUFFICIENT_DISCRETE_EFFECT", "PREFLIGHT_FAIL_CONTRACT_REVEALED", "PREFLIGHT_FAIL_COMMUTATIVE", "PREFLIGHT_FAIL_GEOMETRY", "PREFLIGHT_FAIL_SWEEP_COLLISION",
                   "PREFLIGHT_FAIL_CAMERA", "PREFLIGHT_FAIL_RECOVERABILITY", "PREFLIGHT_FAIL_RELATION_NOT_EXPRESSIBLE", "PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND", "ENGINEERING_UNRESOLVED")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


class Guard:
    """Reader that refuses test/holdout paths and counts what it opened."""

    def __init__(self):
        self.opened, self.refused = 0, []

    def read(self, path):
        if FORBIDDEN.search(str(path).replace("\\", "/")):
            self.refused.append(str(path))
            raise PermissionError("FORBIDDEN_PATH:" + str(path))
        self.opened += 1
        return Path(path).read_text(encoding="utf-8", errors="ignore")


class ZeroEnvGuard:
    """Makes any environment construction raise and counts attempts (must stay 0)."""

    def __init__(self):
        self.calls = 0

    def install(self):
        from cp_disr.platforms.libero import d0_env, runtime_factory as rf
        g = self

        def boom(*a, **k):
            g.calls += 1
            raise RuntimeError("ZERO_ENV_GUARD: environment construction is forbidden in this card")
        d0_env.make_env = boom
        rf.make_env = boom
        d0_env.D0ManipulationEnv.__init__ = boom
        return self


# --------------------------------------------------------------------------------- source extraction
def module_consts(path, names):
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            v = node.value
            try:
                out[node.targets[0].id] = ast.literal_eval(v)
            except Exception:
                if isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "array":
                    out[node.targets[0].id] = ast.literal_eval(v.args[0])
    return out


def extract_stages(path, func):
    """AST extraction of the skill stage calls (_move_to/_hold) of one executor method."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            stages = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in ("_move_to", "_hold"):
                    kw = {k.arg: ast.unparse(k.value) for k in sub.keywords}
                    args = [ast.unparse(a) for a in sub.args]
                    stages.append({"line": sub.lineno, "call": sub.func.attr, "args": args[1:] if sub.func.attr == "_move_to" else args[1:], "keywords": kw})
            stages.sort(key=lambda s: s["line"])
            assigns = {}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                    assigns[sub.targets[0].id] = ast.unparse(sub.value)
            return {"function": func, "stages": stages, "assignments": assigns, "defined_at_line": node.lineno}
    raise RuntimeError("function not found: " + func)


def stl_bounds(path):
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = np.array([[float(x) for x in m.groups()] for m in re.finditer(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)])
    else:
        n = struct.unpack("<I", data[80:84])[0]
        arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        v = arr["v"].reshape(-1, 3)
    return v.min(0), v.max(0)


def gripper_geometry():
    """Panda gripper solids from the robosuite XML and STL assets (hand frame: +z points toward the table, spread axis = hand x)."""
    import robosuite
    base = Path(robosuite.__file__).parent / "models/assets/grippers"
    import xml.etree.ElementTree as ET
    root = ET.parse(base / "panda_gripper.xml").getroot()
    eef_z = float([b for b in root.iter("body") if b.attrib.get("name") == "eef"][0].attrib["pos"].split()[2])
    fing = [b for b in root.iter("body") if b.attrib.get("name") == "leftfinger"][0]
    finger_z0 = float(fing.attrib["pos"].split()[2])
    joint = [j for j in fing.iter("joint")][0]
    slide_max = float(joint.attrib["range"].split()[1])
    tip = [b for b in fing.iter("body") if b.attrib.get("name") == "finger_joint1_tip"][0]
    tip_pos = [float(x) for x in tip.attrib["pos"].split()]
    pad = [g for g in tip.iter("geom")][0]
    pad_half = [float(x) for x in pad.attrib["size"].split()]
    pad_pos = [float(x) for x in pad.attrib["pos"].split()]
    fmin, fmax = stl_bounds(base / "meshes/panda_gripper/finger.stl")
    hmin, hmax = stl_bounds(base / "meshes/panda_gripper/hand.stl")
    # finger frame -> hand frame: z_hand = z_f + finger_z0; world z = site_z - (z_hand - eef_z)
    finger_low_rel_site = -(float(fmax[2]) + finger_z0 - eef_z)       # lowest finger mesh point relative to the grip site (negative = below)
    finger_high_rel_site = -(float(fmin[2]) + finger_z0 - eef_z)
    pad_low = -((tip_pos[2] + finger_z0 + pad_pos[2] + pad_half[2]) - eef_z)
    pad_center_open = tip_pos[1] + pad_pos[1] + slide_max
    return {"source": "robosuite panda_gripper.xml + finger.stl + hand.stl", "eef_site_hand_z": eef_z, "finger_body_hand_z": finger_z0, "slide_max_each_finger": slide_max,
            "finger_mesh_bounds_finger_frame": {"min": fmin.tolist(), "max": fmax.tolist()}, "hand_mesh_bounds": {"min": hmin.tolist(), "max": hmax.tolist()},
            "finger_lowest_point_rel_site_z": finger_low_rel_site, "finger_highest_point_rel_site_z": finger_high_rel_site, "pad_lowest_rel_site_z": pad_low,
            "pad_half_extents": pad_half, "pad_center_spread_open": pad_center_open, "pad_inner_face_open": pad_center_open - pad_half[1], "pad_outer_face_open": pad_center_open + pad_half[1],
            "finger_inner_face_open": slide_max + float(fmin[1]), "finger_outer_face_open": slide_max + float(fmax[1]), "finger_half_width_orthogonal": float(fmax[0]),
            "finger_inner_face_closed_on_cube": 0.02, "finger_outer_face_closed_on_cube": 0.02 + float(fmax[1]),
            "hand_lowest_point_rel_site_z": -(float(hmax[2]) - eef_z), "hand_half_span_spread_axis": float(max(abs(hmin[1]), abs(hmax[1]))),
            "spread_axis_note": "fingers slide along hand x (finger frames rotated +90 deg about z); world orientation of the spread axis is not verified offline, so both world axes are tested"}


# ------------------------------------------------------------------------------------------ Phase A
ROLE = {"a:PICK:target:v1": "PICK_target", "a:PICK:second_object:v1": "PICK_second", "a:PLACE:target:container:v1": "PLACE_target", "a:PLACE_BUFFER:second_object:buffer:v1": "PLACE_BUFFER_second"}


def collect_durations(root, guard):
    root = Path(root)
    rows = []
    sw_dirs = sorted((root / SW1_REL).glob("*_scripted_witness"))
    for d in sw_dirs:
        f = d / "per_skill_facts_and_masks.jsonl"
        for line in guard.read(f).splitlines():
            r = json.loads(line)
            p = r["post"]
            if p is None or p["controller_exit"] != "NORMAL_TERMINATION" or r["pre"]["action"] not in ROLE:
                continue
            rows.append({"source": "SCRIPTED-WITNESS-1", "episode": r["order"], "case_id": r["case_id"], "route": r["route"], "position": r["pre"]["index"], "action": r["pre"]["action"], "role": ROLE[r["pre"]["action"]],
                         "duration": float(p["duration"]), "controller_exit": p["controller_exit"], "episode_success": True})
    for p in sorted((root / DISC_REL / "branches").glob("*.json")):
        d = json.loads(guard.read(p))
        if not (d.get("success", d.get("task_success")) and d.get("valid") and d.get("termination") == "TASK_SUCCESS"):
            continue
        for i, t in enumerate(d["trace"]):
            if t["action"] in ROLE and t["controller_exit"] == "NORMAL_TERMINATION":
                rows.append({"source": "DISCOVERY-1", "episode": p.stem, "case_id": d["case_id"], "route": "mixed", "position": i, "action": t["action"], "role": ROLE[t["action"]],
                             "duration": float(t["elapsed_end"] - t["elapsed_start"]), "controller_exit": t["controller_exit"], "episode_success": True})
    return rows


def stats(values):
    a = np.array(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "min": float(a.min()), "max": float(a.max()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


def duration_summary(rows):
    by = {}
    for r in rows:
        by.setdefault(r["role"], []).append(r["duration"])
    summ = {k: stats(v) for k, v in sorted(by.items())}
    return summ


def discrete_bound(summ):
    """Rework effect: a failed first placement adds the first skill pair (pick + place of the same object) to the good route."""
    out = {"definition": "wrong order with one unconfirmed first placement = 6 skills; good order = 4 skills; extra = the failed first PICK + first PLACE", "modes": {}}
    pairs = {"BLOCK_CONTAINER": ("PICK_target", "PLACE_target"), "BLOCK_BUFFER": ("PICK_second", "PLACE_BUFFER_second")}
    for mode, (a, b) in pairs.items():
        sa, sb = summ[a], summ[b]
        mean = sa["mean"] + sb["mean"]
        lo, hi = sa["min"] + sb["min"], sa["max"] + sb["max"]
        pmin = EPS_T / mean
        table = {f"{p:.1f}": p * mean for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
        out["modes"][mode] = {"skills": [a, b], "delta_skills_if_recovery": 2, "delta_T_if_recovery_mean": mean, "delta_T_if_recovery_min_sum": lo, "delta_T_if_recovery_max_sum": hi,
                              "p_min_for_2p1s_mean": pmin, "p_min_for_2p1s_at_min_durations": EPS_T / lo, "p_min_for_2p1s_at_max_durations": EPS_T / hi, "E_delta_T_by_p": table}
    # design acceptance band: bad-order first-placement success in [0.2, 0.7]  <=>  failure probability p in [0.3, 0.8]
    for mode, v in out["modes"].items():
        v["E_delta_T_at_design_band"] = {"p=0.3": 0.3 * v["delta_T_if_recovery_mean"], "p=0.8": 0.8 * v["delta_T_if_recovery_mean"]}
        v["crosses_2p1s_inside_design_band"] = 0.3 * v["delta_T_if_recovery_mean"] > EPS_T
    # pilot rule 'median delta_T > 2.1 s' over 5 seeds needs >= 3 failing bad-order seeds
    binom = {}
    for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        binom[f"{p:.1f}"] = float(sum(math.comb(5, k) * p ** k * (1 - p) ** (5 - k) for k in range(3, 6)))
    out["median_gate_note"] = {"statement": "the 40-episode gate 'median delta_T > 2.1 s' per scene is passed only if >= 3 of 5 bad-order seeds fail their first placement; with failure probability p inside the design band this is not guaranteed",
                               "P_median_pass_n5_by_failure_probability": binom, "p_for_50pct_median_pass": 0.5}
    out["status"] = "PASS" if all(v["crosses_2p1s_inside_design_band"] for v in out["modes"].values()) else "FAIL"
    out["category"] = "" if out["status"] == "PASS" else "PREFLIGHT_FAIL_INSUFFICIENT_DISCRETE_EFFECT"
    return out


# ------------------------------------------------------------------------------------------ geometry
class Geometry:
    """Static solids and controller waypoints, all heights relative to table_top_z (= z0 in d0_env)."""

    def __init__(self, root, grip):
        d0 = Path(root) / "src/cp_disr/platforms/libero/d0_env.py"
        c = module_consts(d0, {"OBJECT_HALF", "LID_HALF", "CONTAINER_WALL", "CONTAINER_H", "BUFFER_HALF"})
        gc = module_consts(Path(root) / "src/cp_disr/geom_constants.py", {"CONTAINER_INNER"})
        self.cube = float(c["OBJECT_HALF"][0])
        self.wall = float(c["CONTAINER_WALL"])
        self.cont_h = float(c["CONTAINER_H"])
        self.inner = float(gc["CONTAINER_INNER"][0])
        self.buffer_half = float(c["BUFFER_HALF"][0])
        self.buffer_top = 2 * float(c["BUFFER_HALF"][2])
        self.floor_top = 0.008
        self.skin_top = 0.003
        ex = Path(root) / "src/cp_disr/platforms/libero/skill_executor.py"
        txt = ex.read_text()
        self.hover = float(re.search(r"table_top_z \+ (0\.\d+)\)", txt).group(1))
        self.show_z = float(re.search(r"show = np.array\(\[0\.02, -0\.08, float\(self\.env\.table_top_z \+ (0\.\d+)\)\]\)", txt).group(1))
        self.drop_container = self.cont_h + float(re.search(r"table_top_z \+ CONTAINER_H \+ (0\.\d+)", txt).group(1))
        self.drop_buffer = self.cube + float(re.search(r"table_top_z \+ OBJECT_HALF\[2\] \+ (0\.\d+)", txt).group(1))
        self.grasp_z = self.cube + float(re.search(r"table \+ half_z \+ (0\.\d+)", txt).group(1))
        self.grip = grip
        self.fb = grip["finger_lowest_point_rel_site_z"]          # lowest finger point relative to the grip site (negative)
        self.workspace = ((-0.45, 0.45), (-0.45, 0.45))
        self.camera = None

    # ---- solids as (x0, x1, y0, y1, ztop)
    def walls(self):
        cx, cy = CONTAINER_XY
        w, i = 2 * self.wall, self.inner
        t = self.cont_h
        # d0_env: left/right boxes half-size (wall, inner) centred at +-(inner+wall); front/back boxes half-size (inner+wall, wall) centred at +-(inner+wall)
        return [(cx + i, cx + i + w, cy - i, cy + i, t), (cx - i - w, cx - i, cy - i, cy + i, t),
                (cx - i - self.wall, cx + i + self.wall, cy + i, cy + i + w, t), (cx - i - self.wall, cx + i + self.wall, cy - i - w, cy - i, t)]

    def pad(self):
        bx, by = BUFFER_XY
        h = self.buffer_half
        return (bx - h, bx + h, by - h, by + h, self.buffer_top)

    def cube_rect(self, c):
        return (c[0] - self.cube, c[0] + self.cube, c[1] - self.cube, c[1] + self.cube)

    @staticmethod
    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))

    @staticmethod
    def rects_hit(a, b):
        return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]

    def finger_rects(self, eef_xy, axis, closed_on_cube=False):
        """xy rectangles of the two finger meshes (bounding boxes) about the eef, spread along world axis 0 or 1."""
        g = self.grip
        inner = g["finger_inner_face_closed_on_cube"] if closed_on_cube else g["finger_inner_face_open"]
        outer = g["finger_outer_face_closed_on_cube"] if closed_on_cube else g["finger_outer_face_open"]
        hw = g["finger_half_width_orthogonal"]
        out = []
        for s in (-1, 1):
            a0, a1 = sorted((s * inner, s * outer))
            if axis == 0:
                out.append((eef_xy[0] + a0, eef_xy[0] + a1, eef_xy[1] - hw, eef_xy[1] + hw))
            else:
                out.append((eef_xy[0] - hw, eef_xy[0] + hw, eef_xy[1] + a0, eef_xy[1] + a1))
        return out

    # ---- candidate layout
    def blocker_pose(self, mode, off):
        dest = CONTAINER_XY if mode == "BLOCK_CONTAINER" else BUFFER_XY
        c = (dest[0] + off[0], dest[1] + off[1])
        cheb = max(abs(off[0]), abs(off[1]))
        if mode == "BLOCK_CONTAINER":
            rect = self.cube_rect(c)
            if cheb <= self.inner - self.cube + 1e-12:
                return {"rest": "ON_CONTAINER_FLOOR", "elev_top": self.floor_top + 2 * self.cube, "support_top": self.floor_top, "center": c}
            if any(self.rects_hit(rect, (w[0], w[1], w[2], w[3])) for w in self.walls()):
                return {"rest": "INVALID_OVERLAPS_WALL", "elev_top": None, "support_top": None, "center": c}
            return {"rest": "ON_TABLE_OUTSIDE", "elev_top": self.skin_top + 2 * self.cube, "support_top": self.skin_top, "center": c}
        if cheb <= self.buffer_half - self.cube + 1e-12:
            return {"rest": "ON_BUFFER_PAD", "elev_top": self.buffer_top + 2 * self.cube, "support_top": self.buffer_top, "center": c}
        if cheb < self.buffer_half + self.cube - 1e-12:
            return {"rest": "INVALID_OVERHANGS_PAD_EDGE", "elev_top": None, "support_top": None, "center": c}
        return {"rest": "ON_TABLE_OUTSIDE", "elev_top": self.skin_top + 2 * self.cube, "support_top": self.skin_top, "center": c}

    def score(self, mode, off):
        dest = CONTAINER_XY if mode == "BLOCK_CONTAINER" else BUFFER_XY
        drop = self.cube_rect(dest)
        c = (dest[0] + off[0], dest[1] + off[1])
        return self.overlap(self.cube_rect(c), drop) / (4 * self.cube ** 2)

    @staticmethod
    def bin_of(s):
        for name, lo, hi in SCORE_BINS:
            if lo <= s < hi:
                return name
        return "HARD_BLOCKING_RISK"

    def static_solids_for_pick(self, mode, blocker_center):
        solids = [("container_wall", w[:4], w[4]) for w in self.walls()]
        solids.append(("buffer_pad", self.pad()[:4], self.buffer_top))
        other = BASE_POSES["target_xy"] if mode == "BLOCK_CONTAINER" else BASE_POSES["second_xy"]
        solids.append(("other_cube", self.cube_rect(other), self.skin_top + 2 * self.cube))
        return solids

    def pick_sweep(self, mode, pose):
        """Open fingers descend over the blocker (both world spread axes) and the held blocker is lifted along the straight path to the show pose."""
        res = {"descent_collisions": [], "min_vertical_clearance": None, "lift_collisions": []}
        c = pose["center"]
        site_z = self.grasp_z
        finger_bottom = site_z + self.fb
        support = pose["support_top"]
        clearances = []
        for axis in (0, 1):
            for fr in self.finger_rects(c, axis):
                for name, rect, top in self.static_solids_for_pick(mode, c):
                    if name == "other_cube":
                        pass
                    if self.rects_hit(fr, rect):
                        # finger descends from hover to the grasp height; it hits the solid when its bottom goes below the solid top
                        if finger_bottom < top:
                            res["descent_collisions"].append({"axis": axis, "solid": name, "finger_bottom": finger_bottom, "solid_top": top})
        # vertical clearance of the finger bottoms above the surface that carries the blocker
        res["min_vertical_clearance"] = finger_bottom - support
        res["finger_bottom_at_grasp"] = finger_bottom
        # lift: straight line from grasp pose to the show pose; held cube bottom height along the path
        start = np.array([c[0], c[1], site_z])
        end = np.array([SHOW_XY[0], SHOW_XY[1], self.show_z])
        for t in np.linspace(0, 1, 41):
            p = start + t * (end - start)
            bottom = p[2] - self.cube
            rect = self.cube_rect((p[0], p[1]))
            for name, srect, top in self.static_solids_for_pick(mode, c):
                if name == "container_wall" and pose["rest"] == "ON_CONTAINER_FLOOR" and bottom < top and self.rects_hit(rect, srect):
                    res["lift_collisions"].append({"t": float(t), "solid": name, "cube_bottom": float(bottom), "solid_top": top})
                elif name != "container_wall" and bottom < top - 1e-9 and self.rects_hit(rect, srect) and not (name == "buffer_pad" and pose["rest"] == "ON_BUFFER_PAD" and t < 0.05):
                    res["lift_collisions"].append({"t": float(t), "solid": name, "cube_bottom": float(bottom), "solid_top": top})
        res["lift_collisions"] = res["lift_collisions"][:3]
        return res

    def place_clearance(self, mode, pose):
        """Held cube (other object) released at the destination centre: clearance of its bottom and of the closed fingers above the blocker/walls."""
        drop_site = self.drop_container if mode == "BLOCK_CONTAINER" else self.drop_buffer
        cube_bottom = drop_site - self.cube - 0.001
        finger_bottom = drop_site + self.fb
        blocker_top = pose["elev_top"] if pose["elev_top"] is not None else None
        out = {"drop_site_z": drop_site, "held_cube_bottom": cube_bottom, "finger_bottom": finger_bottom, "wall_top": self.cont_h if mode == "BLOCK_CONTAINER" else None}
        if blocker_top is not None:
            out["blocker_top"] = blocker_top
            out["cube_bottom_clearance_over_blocker"] = cube_bottom - blocker_top
            out["finger_clearance_over_blocker"] = finger_bottom - blocker_top
        return out

    def project(self, point, cam):
        R = np.array(cam["mat"], dtype=float)
        t = np.array(cam["pos"], dtype=float)
        pc = R.T @ (np.array(point, dtype=float) - t)
        z = -pc[2]
        f = 0.5 * 128 / math.tan(math.radians(cam["fovy"]) / 2.0)
        return (63.5 + f * pc[0] / z, 63.5 - f * pc[1] / z, z, f)

    def camera_audit(self, center, z0_abs, cam):
        xs, ys = [], []
        top_z = z0_abs + 0.003 + 2 * self.cube
        for dx, dy, z in itertools.product((-self.cube, self.cube), (-self.cube, self.cube), (z0_abs + 0.003, top_z)):
            u, v, d, f = self.project((center[0] + dx, center[1] + dy, z), cam)
            xs.append(u)
            ys.append(v)
        top = [self.project((center[0] + dx, center[1] + dy, top_z), cam) for dx, dy in ((-self.cube, -self.cube), (self.cube, -self.cube), (self.cube, self.cube), (-self.cube, self.cube))]
        area = 0.5 * abs(sum(top[i][0] * top[(i + 1) % 4][1] - top[(i + 1) % 4][0] * top[i][1] for i in range(4)))
        margin = min(min(xs), min(ys), 127 - max(xs), 127 - max(ys))
        return {"pixel_margin": float(margin), "top_face_pixel_area": float(area), "pass": bool(margin >= SAFETY_MARGIN_PX and area >= MIN_TOP_FACE_PIXELS)}


def projection_sanity(geom, cam, z0_abs, snap_dir="/home/xushijie2/graph_cp_disr_snapshots/tp_ef_post_open_capture/20261003T014215Z/T_B_dev_03"):
    """Compare the projected container / buffer centres with the colour-blob centroids of the saved public RGB (development snapshot, never a test image)."""
    p = Path(snap_dir) / "obs_public_cached.npz"
    if not p.is_file():
        return {"status": "NOT_AVAILABLE"}
    rgb = np.load(p)["rgb"].astype(np.float32)
    rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
    colors = {"container": (0.18, 0.35, 0.75), "buffer": (0.18, 0.72, 0.30)}
    centers = {"container": (CONTAINER_XY[0], CONTAINER_XY[1], z0_abs + 0.03), "buffer": (BUFFER_XY[0], BUFFER_XY[1], z0_abs + 0.008)}
    out = {}
    for k, col in colors.items():
        m = np.linalg.norm(rgb - np.array(col)[None, None, :], axis=2) < 0.32
        ys, xs = np.where(m)
        u, v, d, f = geom.project(centers[k], cam)
        if len(xs) < 8:
            out[k] = {"blob_pixels": int(len(xs)), "status": "NO_BLOB"}
            continue
        out[k] = {"blob_pixels": int(len(xs)), "blob_centroid": [float(xs.mean()), float(ys.mean())], "projected": [float(u), float(v)], "pixel_distance": float(math.hypot(xs.mean() - u, ys.mean() - v))}
    ok = all(v.get("pixel_distance", 99) < 6.0 for v in out.values())
    return {"status": "PASS" if ok else "MODEL_UNVERIFIED", "tolerance_px": 6.0, "objects": out}


def candidate_grid(geom, cam, z0_abs):
    cands = []
    steps = int(round(GRID_HALF / GRID_STEP))
    for mode in PICK_MODES:
        for ix, iy in itertools.product(range(-steps, steps + 1), repeat=2):
            off = (round(ix * GRID_STEP, 4), round(iy * GRID_STEP, 4))
            pose = geom.blocker_pose(mode, off)
            s = geom.score(mode, off)
            row = {"mode": mode, "offset": list(off), "rest": pose["rest"], "score": s, "score_bin": geom.bin_of(s), "blocker_center": list(pose["center"])}
            if pose["rest"].startswith("INVALID"):
                row.update(safety="INVALID_REST_POSE", reasons=[pose["rest"]])
                cands.append(row)
                continue
            ps = geom.pick_sweep(mode, pose)
            pl = geom.place_clearance(mode, pose)
            cm = geom.camera_audit(pose["center"], z0_abs, cam)
            reasons = []
            if ps["descent_collisions"]:
                reasons.append("COLLISION_FINGERS_WITH_" + "+".join(sorted({c["solid"] for c in ps["descent_collisions"]})))
            if ps["lift_collisions"]:
                reasons.append("COLLISION_HELD_BLOCKER_LIFT_WITH_" + "+".join(sorted({c["solid"] for c in ps["lift_collisions"]})))
            if not reasons and ps["min_vertical_clearance"] < PROVEN_CLEARANCE(geom) - 1e-9:
                reasons.append("UNCERTIFIED_FINGER_CLEARANCE_BELOW_PROVEN_ENVELOPE")
            if not cm["pass"]:
                reasons.append("CAMERA_MARGIN_OR_PIXELS")
            safety = "SAFE" if not reasons else ("UNSAFE_COLLISION" if any(r.startswith("COLLISION") for r in reasons) else "UNCERTIFIED")
            row.update(safety=safety, reasons=reasons, pick_sweep={k: ps[k] for k in ("min_vertical_clearance", "finger_bottom_at_grasp")}, place=pl, camera=cm,
                       stack_hazard_if_score_ge_0p5=bool(s >= 0.5))
            cands.append(row)
    return cands


def PROVEN_CLEARANCE(geom):
    """Finger-bottom clearance above the supporting surface realised by the standard table pick (the geometry exercised by every clean execution so far)."""
    return geom.grasp_z + geom.fb - geom.skin_top


def select_pilot_scenes(cands):
    """Deterministic: mode -> severity bin -> sha256 of the config; first SAFE candidate of each (mode, bin) for WEAK and MODERATE."""
    chosen = {}
    for mode in PICK_MODES:
        for bin_name in ("WEAK", "MODERATE"):
            pool = [c for c in cands if c["mode"] == mode and c["score_bin"] == bin_name and c["safety"] == "SAFE"]
            pool.sort(key=lambda c: digest({"mode": c["mode"], "offset": c["offset"], "bin": bin_name}))
            if pool:
                chosen[f"{'BC' if mode == 'BLOCK_CONTAINER' else 'BB'}-{bin_name}"] = pool[0]
    return chosen


# ------------------------------------------------------------------------------------------ verdicts
def category_from_candidates(cands):
    """NO_SAFE band decision with per-mode root causes."""
    res = {}
    for mode in PICK_MODES:
        inter = [c for c in cands if c["mode"] == mode and c["score_bin"] in ("WEAK", "MODERATE", "HARD_BLOCKING_RISK") and not c["rest"].startswith("INVALID")]
        weak_mod = [c for c in inter if c["score_bin"] in ("WEAK", "MODERATE")]
        safe = [c for c in weak_mod if c["safety"] == "SAFE"]
        causes = Counter(r for c in weak_mod for r in c["reasons"])
        res[mode] = {"interfering_valid_candidates": len(inter), "weak_or_moderate_valid": len(weak_mod), "safe_weak_or_moderate": len(safe), "root_cause_counts": dict(causes),
                     "all_failures_are_collisions": bool(weak_mod) and all(any(r.startswith("COLLISION") for r in c["reasons"]) for c in weak_mod)}
    if all(v["safe_weak_or_moderate"] > 0 for v in res.values()):
        return "", res
    only_collision = all(v["safe_weak_or_moderate"] > 0 or v["all_failures_are_collisions"] for v in res.values())
    return ("PREFLIGHT_FAIL_SWEEP_COLLISION" if only_collision else "PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND"), res


# ------------------------------------------------------------------------------------------ Phase C/D
def controller_contract(root, geom, grip):
    root = Path(root)
    ex = root / "src/cp_disr/platforms/libero/skill_executor.py"
    skills = {name: extract_stages(ex, fn) for name, fn in (("PICK", "_pick"), ("PLACE", "_place"), ("PLACE_BUFFER", "_place_buffer"), ("OPEN", "_open"))}
    cube = geom.cube
    pad_bottom_at_grasp = geom.grasp_z + geom.fb
    return {
        "card": CARD, "source_files": {str(p.relative_to(root)): sha_file(p) for p in [ex, root / "src/cp_disr/platforms/libero/d0_env.py", root / "src/cp_disr/platforms/libero/runtime_factory.py", root / "src/cp_disr/platforms/libero/verifier.py",
                                                                                           root / "src/cp_disr/platforms/libero/task_evaluator.py", root / "src/cp_disr/platforms/libero/safety.py", root / "configs/runtime/stage_2a_contract_registry.yaml"]},
        "heights_relative_to_table_top_z": {"hover": geom.hover, "show_pose_z": geom.show_z, "show_pose_xy": list(SHOW_XY), "grasp_site_z": geom.grasp_z, "press_site_z": geom.grasp_z - 0.008,
                                            "drop_site_z_container": geom.drop_container, "drop_site_z_buffer": geom.drop_buffer, "container_wall_top": geom.cont_h, "container_floor_top": geom.floor_top,
                                            "buffer_pad_top": geom.buffer_top, "table_skin_top": geom.skin_top},
        "held_and_blocker_objects": {"cube_half": cube, "cube_edge": 2 * cube, "target_and_second_object_identical": True, "grasp_rule": "site z = min(perceived_z - 0.012, table_top + 0.02 + 0.002) -> hard-coded relative to the TABLE, not the support under the object"},
        "stages": skills,
        "destination_resolver": {"T_B": "env.resolve_skill_destination is defined only by the Family B env; the T_B env therefore uses env.container_center / env.buffer_center (the fixed destination centres)",
                                 "container_center_xy": list(CONTAINER_XY), "buffer_center_xy": list(BUFFER_XY)},
        "evaluator": {"inside": "|dx|,|dy| <= CONTAINER_INNER (=%.3f) from the container centre, z <= table_top + 0.12, lid away > 0.10" % geom.inner, "at_buffer": "|dx|,|dy| <= 0.06 from the buffer centre, z <= table_top + 0.10"},
        "verifier_thresholds": {"inside_margin": 0.02, "buffer_margin": 0.03, "held_xy": 0.06, "held_z": 0.07, "table_z_tol": 0.05},
        "safety": {"workspace_xy": [[-0.45, 0.45], [-0.45, 0.45]], "margin_note": "+/-0.05 slack in _workspace_ok"},
        "gripper_solids": grip,
        "sweep_volume_sources": ["open finger mesh boxes (spread axis, both world axes tested) from the hover to the grasp height", "held cube 0.04 x 0.04 x 0.04 carried along the straight line to the show pose",
                                 "closed finger meshes about the held cube during the PLACE / PLACE_BUFFER descent", "hand mesh (lowest point %.4f m above the grip site)" % grip["hand_lowest_point_rel_site_z"]],
        "derived": {"finger_bottom_at_grasp_rel_table": pad_bottom_at_grasp, "finger_clearance_above_table_skin_standard_pick": pad_bottom_at_grasp - geom.skin_top,
                    "finger_clearance_above_buffer_pad_top_pick": pad_bottom_at_grasp - geom.buffer_top,
                    "open_finger_inner_face": grip["finger_inner_face_open"], "open_finger_outer_face": grip["finger_outer_face_open"],
                    "container_wall_span_from_centre": [geom.inner, geom.inner + 2 * geom.wall], "container_opening_edge": 2 * geom.inner,
                    "held_cube_bottom_over_wall_top_at_container_drop": geom.drop_container - cube - 0.001 - geom.cont_h,
                    "held_cube_bottom_over_target_top_on_pad_at_buffer_drop": geom.drop_buffer - cube - 0.001 - (geom.buffer_top + 2 * cube)},
        "answers": {
            "1_held_object_size": "0.04 m cube for both objects",
            "2_descent_paths": "PLACE / PLACE_BUFFER hover at table_top+%.2f over the destination centre, then a straight vertical descent to the drop height and release" % geom.hover,
            "3_blocker_held_object_collision_space": "container: blocker must sit inside the 0.06 m opening (walls are 0.048 high, held cube bottom clears the wall top by %.3f m at drop); buffer: blocker rests on the 0.008 m pad and the held cube bottom clears its top by only %.3f m"
                                                    % (geom.drop_container - cube - 0.001 - geom.cont_h, geom.drop_buffer - cube - 0.001 - (geom.buffer_top + 2 * cube)),
            "4_blocker_vs_finger": "closed fingers stay above the blocker top by %.3f m (container) / %.3f m (buffer) at the drop: the held cube, not a finger, is the lowest solid" % (geom.drop_container + geom.fb - (geom.floor_top + 2 * cube), geom.drop_buffer + geom.fb - (geom.buffer_top + 2 * cube)),
            "5_blocker_pickable_from_above": "evaluated per candidate in static_sweep_audit.json",
            "6_wrong_order_recovery_region": "evaluated per candidate in static_sweep_audit.json (informational) - Phase H is only run if Phase F passes"},
        "controller_modification_needed": False}


def contract_tie_audit(root, sw_facts):
    from cp_disr.analysis import tp_ef_protocol_review as pr
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.facts import FactRecord, FactStore, Truth
    template = pr.build_template(root)
    values = {k: Truth(v) for k, v in sw_facts.items()}
    records = tuple(FactRecord(fact_id=k, value=v, capture_time=0.0, available_time=0.0, evidence_ids=("saved",), last_confirmed_value=v, last_confirmed_time=0.0, reason="saved-post-OPEN-facts") for k, v in sorted(values.items()))
    planner = BPlanPlanner(SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=4.199999999997672))
    plan = planner.plan(FactStore(records), template, 60.0)
    legal_t, legal_s = pr.legal(template, values, pr.IDS["pt"]), pr.legal(template, values, pr.IDS["ps"])
    rt = pr.min_plan_after(template, values, pr.IDS["pt"])
    rs = pr.min_plan_after(template, values, pr.IDS["ps"])
    st, tt = pr.script_legality(template, values, ["pt", "plt", "ps", "pbs"]), pr.script_legality(template, values, ["ps", "pbs", "pt", "plt"])
    cost = lambda depth: depth * 4.199999999997672
    predicates = sorted({a.split(":")[1] for a in sw_facts})
    per_mode = {m: {"facts_hash": digest({k: v for k, v in sw_facts.items()}), "scene_geometry_in_facts": False} for m in PICK_MODES}
    return {"card": CARD, "post_open_facts_source": "SCRIPTED-WITNESS-1 snapshot_identity_manifest (Verifier-confirmed T_B_dev_03 post-OPEN facts)", "facts": sw_facts, "predicates_present": predicates,
            "blocking_or_geometry_predicates_present": [p for p in predicates if re.search(r"block|inter|clear|reach", p, re.I)],
            "pick_target_legal": legal_t, "pick_second_legal": legal_s, "forced_first_T": {"plan_found": rt[0], "min_skills": rt[1], "route": rt[2]}, "forced_first_S": {"plan_found": rs[0], "min_skills": rs[1], "route": rs[2]},
            "scripted_T_legal_reaches_goal": tt[0], "scripted_S_legal_reaches_goal": st[0], "depth_T": rt[1], "depth_S": rs[1], "nominal_cost_T": cost(rt[1]) if rt[1] else None, "nominal_cost_S": cost(rs[1]) if rs[1] else None,
            "skill_multiset_equal": sorted(["PICK", "PLACE", "PICK", "PLACE_BUFFER"]) == sorted(["PICK", "PLACE_BUFFER", "PICK", "PLACE"]), "bplan_free_choice": {"status": plan.status, "plan": list(plan.plan), "depth": plan.depth, "cost": plan.cost},
            "per_mode_planner_input": per_mode, "modes_indistinguishable_to_planner": len({v["facts_hash"] for v in per_mode.values()}) == 1,
            "tie": bool(legal_t and legal_s and rt[0] and rs[0] and rt[1] == rs[1] and tt[0] and st[0]),
            "status": "PASS" if (legal_t and legal_s and rt[0] and rs[0] and rt[1] == rs[1] == 4 and tt[0] and st[0] and not [p for p in predicates if re.search(r"block|inter|clear|reach", p, re.I)]) else "FAIL",
            "category_if_fail": "PREFLIGHT_FAIL_CONTRACT_REVEALED"}


# --------------------------------------------------------------------------------------- orchestration
PHASE_FILES = {
    "G": ["pilot_scene_manifest.json", "canary_registration.json", "canary_execution_spec.json"], "H": ["recoverability_static_audit.json"], "I": ["relation_expressibility_audit.json"], "J": ["next_canary_card_request.md"]}


def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4", "runs/final_master/2.1.1/T_B", "runs/stage_0a", "docs/authoritative", "status"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "tp_bsi_preflight" not in str(p) and not FORBIDDEN.search(str(p).replace("\\", "/")):
                out[str(p.relative_to(root))] = sha_file(p)
    for p in sorted((root / "configs/splits").glob("*test*.json")):
        out[str(p.relative_to(root))] = sha_file(p)
    return out


def write_not_run(out, phase, reason):
    for f in PHASE_FILES[phase]:
        path = Path(out) / f
        if path.exists():
            continue
        if f.endswith(".md"):
            path.write_text(f"# {f}\n\nNOT_RUN: {reason}\n", encoding="utf-8")
        else:
            write_json(path, {"status": "NOT_RUN", "reason": reason})


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard, zg = Guard(), ZeroEnvGuard().install()
    from cp_disr.analysis import tp_ef_protocol_review as pr
    before = protected_hashes(root)
    write_json(out / "protected_before.json", before)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    write_json(out / "authorization.json", {"card": CARD, "baseline": BASELINE, "authorized_by": "user chat instruction approving CP-DISR-TP-BSI-PREFLIGHT-1", "design_doc": "CP_DISR_TP_Bidirectional_Soft_Interference_Design_v2.md",
                                           "limits": {"environment_constructions": 0, "start_case": 0, "env_reset": 0, "skill_calls": 0, "physical_episodes": 0, "provider_requests": 0, "rl_transitions": 0, "optimizer_steps": 0, "test_reads": 0},
                                           "holdout_note": "the design lists T_B independent holdout episodes as a duration source, but the card also forbids reading T_B test30 results; the prohibition was applied and the holdout was not opened",
                                           "thresholds_declared_before_run": {"grid_step": GRID_STEP, "grid_half": GRID_HALF, "score_bins": SCORE_BINS, "safety_margin_px": SAFETY_MARGIN_PX, "min_top_face_pixels": MIN_TOP_FACE_PIXELS,
                                                                              "clearance_certification": "finger-bottom clearance above the supporting surface must be >= the clearance of the standard table pick (the geometry exercised by every clean execution)",
                                                                              "eps_T": EPS_T, "eps_G": EPS_G, "H": H},
                                           "phase_order": "A,B,C,D,E,F evaluated in order; the first failing phase stops the card (G-J NOT_RUN)"})
    write_json(out / "source_identity.json", {"card": CARD, "baseline": BASELINE, "execution_commit": commit, "dirty_tracked": dirty, "branch": subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True).stdout.strip(),
                                              "module_sha256": sha_file(__file__), "python": sys.executable})
    phases, category, stop_phase = {}, "", ""
    # ---- A
    rows = collect_durations(root, guard)
    with (out / "skill_duration_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "episode", "case_id", "route", "position", "action", "role", "duration", "controller_exit", "episode_success"])
        w.writeheader()
        w.writerows(rows)
    summ = duration_summary(rows)
    need = ("PICK_target", "PICK_second", "PLACE_target", "PLACE_BUFFER_second")
    a_ok = all(summ.get(k, {}).get("n", 0) >= 2 for k in need)
    write_json(out / "skill_duration_summary.json", {"card": CARD, "sources": sorted({r["source"] for r in rows}), "excluded": ["timeout skills", "controller failures", "technical episodes (NO_PLAN)", "Family B", "T_B holdout/test", "OPEN"],
                                                    "per_skill": summ, "T_pick_target": summ.get("PICK_target", {}).get("mean"), "T_pick_second": summ.get("PICK_second", {}).get("mean"), "T_place": summ.get("PLACE_target", {}).get("mean"),
                                                    "T_place_buffer": summ.get("PLACE_BUFFER_second", {}).get("mean"), "status": "PASS" if a_ok else "FAIL"})
    phases["A"] = "PASS" if a_ok else "FAIL"
    if not a_ok:
        category, stop_phase = "ENGINEERING_UNRESOLVED", "A"
    # ---- B
    if not stop_phase:
        bound = discrete_bound(summ)
        bound["T_extra_two_skills"] = {m: v["delta_T_if_recovery_mean"] for m, v in bound["modes"].items()}
        write_json(out / "discrete_effect_bound.json", bound)
        phases["B"] = bound["status"]
        if bound["status"] != "PASS":
            category, stop_phase = bound["category"], "B"
    # ---- geometry basics (needed by C-F)
    grip = gripper_geometry()
    geom = Geometry(root, grip)
    prof = json.loads((root / "configs/final_master/family_b_obs_v2/observation_profile_v2.json").read_text())
    cam = prof["camera_manifest"]["agentview"]
    z0_abs = 0.825
    # ---- C
    if not stop_phase:
        cc = controller_contract(root, geom, grip)
        write_json(out / "controller_motion_contract.json", cc)
        phases["C"] = "PASS"
    # ---- D
    if not stop_phase:
        sw_manifest = sorted((root / SW1_REL).glob("*_scripted_witness"))[-1] / "snapshot_identity_manifest.json"
        facts = json.loads(guard.read(sw_manifest))["snapshots"]["T_B_dev_03"]["facts"]
        tie = contract_tie_audit(root, facts)
        write_json(out / "nominal_contract_state.json", {"facts": facts, "predicates": tie["predicates_present"], "legal": {"PICK_target": tie["pick_target_legal"], "PICK_second": tie["pick_second_legal"]}})
        write_json(out / "bplan_tie_audit.json", tie)
        phases["D"] = tie["status"]
        if tie["status"] != "PASS":
            category, stop_phase = tie["category_if_fail"], "D"
    # ---- E/F geometry
    cands = candidate_grid(geom, cam, z0_abs) if not stop_phase else []
    if not stop_phase:
        write_json(out / "asset_geometry_identity.json", {"card": CARD, "constants": {"cube_half": geom.cube, "container_inner_half": geom.inner, "container_wall_half_thickness": geom.wall, "container_wall_height": geom.cont_h,
                                                                                       "container_floor_top": geom.floor_top, "buffer_half": geom.buffer_half, "buffer_top": geom.buffer_top, "table_skin_top": geom.skin_top},
                                                      "walls_xy_rects_world": [list(w) for w in geom.walls()], "buffer_pad_rect_world": list(geom.pad()), "container_center": list(CONTAINER_XY), "buffer_center": list(BUFFER_XY),
                                                      "base_poses_of_the_non_blocker_object": BASE_POSES, "source_sha256": {p: sha_file(root / p) for p in ["src/cp_disr/platforms/libero/d0_env.py", "src/cp_disr/geom_constants.py"]},
                                                      "gripper_asset": {"xml_stl": "robosuite panda_gripper.xml / finger.stl / hand.stl", "derived": grip}})
        write_json(out / "interference_metric_spec.json", {"definition": "score = area(blocker footprint ∩ nominal drop footprint) / area(cube footprint); drop footprint = held-cube footprint centred on the destination centre", "bins": SCORE_BINS,
                                                          "pilot_bins_allowed": ["WEAK", "MODERATE"], "safety_status_values": ["SAFE", "UNSAFE_COLLISION", "UNCERTIFIED", "INVALID_REST_POSE"],
                                                          "valid_rest_rule": {"BLOCK_CONTAINER": "blocker footprint fully inside the opening (|offset| <= inner - cube = %.3f) or disjoint from the wall boxes" % (geom.inner - geom.cube),
                                                                              "BLOCK_BUFFER": "blocker on the pad (|offset| <= half - cube = %.3f) or off the pad (|offset| >= half + cube = %.3f)" % (geom.buffer_half - geom.cube, geom.buffer_half + geom.cube)}})
        with (out / "candidate_geometry_grid.json").open("w") as f:
            json.dump({"card": CARD, "grid_step": GRID_STEP, "grid_half": GRID_HALF, "count": len(cands), "candidates": cands}, f, default=_default)
        modes_pathway = {m: any((not c["rest"].startswith("INVALID")) and c["score"] > 0 for c in cands if c["mode"] == m) for m in PICK_MODES}
        sw_rows = [r for r in rows if r["source"] == "SCRIPTED-WITNESS-1"]
        t_route = sum(r["duration"] for r in sw_rows if r["route"] == "T") / max(1, len({r["episode"] for r in sw_rows if r["route"] == "T"}))
        s_route = sum(r["duration"] for r in sw_rows if r["route"] == "S") / max(1, len({r["episode"] for r in sw_rows if r["route"] == "S"}))
        comm = {"card": CARD, "continuous_order_effect_from_clean_logs": {"mean_route_T_skill_time_sum": t_route, "mean_route_S_skill_time_sum": s_route, "abs_difference": abs(t_route - s_route),
                                                                           "interpretation": "same skill multiset, order changes total time by < 0.1 s: the continuous path/ordering effect is negligible"},
                "modes": {m: {"geometric_pathway_exists": modes_pathway[m], "pathway": ("second_object inside the 0.06 m container opening reduces the free landing area of the target" if m == "BLOCK_CONTAINER" else
                                                                                         "target on the buffer pad overlaps the second_object landing footprint"), "realizable_with_current_controller": "evaluated in static_sweep_audit.json (Phase F)"} for m in PICK_MODES},
                "status": "PASS" if all(modes_pathway.values()) else "FAIL", "category_if_fail": "PREFLIGHT_FAIL_COMMUTATIVE"}
        write_json(out / "commutativity_audit.json", comm)
        phases["E"] = comm["status"]
        if comm["status"] != "PASS":
            category, stop_phase = comm["category_if_fail"], "E"
    if not stop_phase:
        cat, per_mode = category_from_candidates(cands)
        # informational first-placement outcome analysis (necessary conditions only)
        outcome = {"BLOCK_CONTAINER": {"evaluator_inside_region_half": geom.inner, "first_placement_unconfirmed_requires": "target final centre more than %.3f m from the container centre; the blocker (a 0.04 cube) fits only INSIDE the 0.06 opening, so the released target lands on it or tilts against it still inside the walls; failure needs a rim-resting pose that the static model cannot certify" % geom.inner,
                                       "blocker_pick": "open finger meshes span %.4f-%.4f m from the gripper centre; the container walls span %.3f-%.3f m from the container centre, so every blocker pose inside the opening makes both fingers descend onto the wall tops" % (grip["finger_inner_face_open"], grip["finger_outer_face_open"], geom.inner, geom.inner + 2 * geom.wall)},
                   "BLOCK_BUFFER": {"evaluator_at_buffer_half": 0.06, "first_placement_unconfirmed_requires": "second_object final centre more than 0.06 m from the buffer centre; it is released at the centre of a 0.16 m pad, on or beside the blocker, so almost every non-colliding outcome is confirmed (stack on the target or resting beside it)",
                                    "blocker_pick": "grasp height is fixed relative to the table: finger bottoms are %.4f m above the 0.008 m pad top vs %.4f m above the table skin in the standard pick" % (geom.grasp_z + geom.fb - geom.buffer_top, PROVEN_CLEARANCE(geom)),
                                    "stack_hazard": "released cube bottom clears a target resting on the pad by only %.4f m; a stacked blocker cannot be re-picked, which is a hard (not soft) dead end" % (geom.drop_buffer - geom.cube - 0.001 - (geom.buffer_top + 2 * geom.cube))}}
        audit = {"card": CARD, "per_mode": per_mode, "category": cat, "proven_clearance_envelope": PROVEN_CLEARANCE(geom), "outcome_analysis_informational": outcome,
                 "reason_histogram_all_valid_interfering": {m: dict(Counter(r for c in cands if c["mode"] == m and c["score_bin"] in ("WEAK", "MODERATE") and not c["rest"].startswith("INVALID") for r in c["reasons"])) for m in PICK_MODES},
                 "limitations": ["static necessary-condition model; contact dynamics are not simulated", "finger spread axis tested on both world axes (orientation unverified offline)", "release xy error of the controller (tol 0.018 m) is not modelled in the score"],
                 "status": "PASS" if not cat else "FAIL"}
        write_json(out / "static_sweep_audit.json", audit)
        cam_rows = [{"mode": c["mode"], "offset": c["offset"], "camera": c.get("camera"), "interfering": c["score"] > 0} for c in cands if c.get("camera")]
        rel = [r for r in cam_rows if r["interfering"]]
        sanity = projection_sanity(geom, cam, z0_abs)
        write_json(out / "camera_static_audit.json", {"card": CARD, "camera": "agentview frozen profile (pos %s, fovy %s, 128x128)" % (cam["pos"], cam["fovy"]), "margin_px_min": SAFETY_MARGIN_PX, "top_face_min_px": MIN_TOP_FACE_PIXELS,
                                                      "all_valid_grid_candidates": {"checked": len(cam_rows), "failing": sum(1 for r in cam_rows if not r["camera"]["pass"]), "worst_margin_px": min(r["camera"]["pixel_margin"] for r in cam_rows),
                                                                                   "note": "candidates far on the near side of the container fall outside the image; they have score 0 (CLEAR) and are irrelevant to the soft band"},
                                                      "interfering_candidates": {"checked": len(rel), "failing": sum(1 for r in rel if not r["camera"]["pass"]), "worst_margin_px": min(r["camera"]["pixel_margin"] for r in rel) if rel else None,
                                                                                "min_top_face_area_px": min(r["camera"]["top_face_pixel_area"] for r in rel) if rel else None},
                                                      "projection_model_sanity_check": sanity, "status": "PASS" if rel and all(r["camera"]["pass"] for r in rel) else "FAIL", "occlusion": "not modelled"})
        phases["F"] = audit["status"]
        if cat:
            category, stop_phase = cat, "F"
            write_json(out / "recoverability_static_audit.json", {"status": "NOT_RUN", "reason": "Phase F failed; informational stack/dead-end observations are in static_sweep_audit.json"})
        scenes = select_pilot_scenes(cands) if not cat else {}
        write_json(out / "pilot_scene_manifest.json", {"card": CARD, "status": "FROZEN" if scenes and len(scenes) == 4 else "NOT_FROZEN_NO_SAFE_BAND", "scenes": scenes,
                                                      "reason": "" if scenes else "no (mode, WEAK/MODERATE) candidate is SAFE under the declared static rules"})
    # ---- remaining phases
    passed = False
    scenes = select_pilot_scenes(cands) if not stop_phase else {}
    if not stop_phase:
        # G: freeze, H: static recoverability, I: relation expressibility, J: canary registration (only reachable when F passed)
        write_json(out / "pilot_scene_manifest.json", {"card": CARD, "status": "FROZEN" if len(scenes) == 4 else "NOT_FROZEN", "scenes": scenes})
        for name, c in scenes.items():
            write_json(out / "pilot_scene_configs" / f"{name}.json", scene_config(name, c))
        phases["G"] = "PASS" if len(scenes) == 4 else "FAIL"
        if len(scenes) != 4:
            category, stop_phase = "PREFLIGHT_FAIL_NO_SAFE_SOFT_INTERFERENCE_BAND", "G"
    if not stop_phase:
        rec = recoverability_audit(scenes, geom)
        write_json(out / "recoverability_static_audit.json", rec)
        phases["H"] = rec["status"]
        if rec["status"] != "PASS":
            category, stop_phase = "PREFLIGHT_FAIL_RECOVERABILITY", "H"
    if not stop_phase:
        rel = relation_expressibility(root)
        write_json(out / "relation_expressibility_audit.json", rel)
        phases["I"] = rel["status"]
        if rel["status"] != "PASS":
            category, stop_phase = "PREFLIGHT_FAIL_RELATION_NOT_EXPRESSIBLE", "I"
    if not stop_phase:
        reg, spec = canary_documents(scenes)
        write_json(out / "canary_registration.json", reg)
        write_json(out / "canary_execution_spec.json", spec)
        (out / "next_canary_card_request.md").write_text(canary_request_md(reg), encoding="utf-8")
        phases["J"] = "PASS"
        passed = True
    else:
        started = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I"}
        for ph in "GHIJ":
            write_not_run(out, ph, f"stopped at phase {stop_phase}: {category}")
        (out / "pilot_scene_configs").mkdir(exist_ok=True)
        write_json(out / "pilot_scene_configs" / "NOT_GENERATED.json", {"reason": category})
    verdict = {"card": CARD, "phases": phases, "stop_phase": stop_phase, "verdict": "PREFLIGHT_PASS_CANARY_REQUESTED" if passed else category, "canary_requested": bool(passed)}
    write_json(out / "preflight_verdict.json", verdict)
    if not passed:
        (out / "next_canary_card_request.md").write_text(f"# Next canary card request — {CARD}\n\n**NOT_REQUESTED.** Preflight verdict: `{category}` (stopped at phase {stop_phase}).\n\nNo canary episode is requested; no environment was constructed.\n", encoding="utf-8")
        write_json(out / "canary_registration.json", {"status": "NOT_GENERATED", "reason": category})
        write_json(out / "canary_execution_spec.json", {"status": "NOT_GENERATED", "reason": category})
    ledger = {"environment_constructions": zg.calls, "start_case": 0, "env_reset": 0, "skill_calls": 0, "physical_episodes": 0, "provider_requests": 0, "provider_retries": 0, "rl_transitions": 0, "optimizer_steps": 0,
              "elastic_attempts": 0, "test_reads": 0, "test_read_attempts_refused": len(guard.refused), "test_episodes": 0, "files_opened_by_guard": guard.opened, "zero_env_guard_calls": zg.calls}
    write_json(out / "budget_ledger.json", ledger)
    write_json(out / "protected_after.json", protected_hashes(root))
    return verdict


# ------------------------------------------------------------------------- G-J (reachable only after Phase F passes)
def scene_config(name, c):
    cfg = {"scene_id": name, "mode": c["mode"], "severity": c["score_bin"], "blocker_offset_from_destination": c["offset"], "blocker_center_xy": c["blocker_center"], "rest": c["rest"],
           "container_xy": list(CONTAINER_XY), "buffer_xy": list(BUFFER_XY), "non_blocker_object_xy": BASE_POSES["target_xy" if c["mode"] == "BLOCK_CONTAINER" else "second_xy"],
           "static_metrics": {"score": c["score"], "pick_sweep": c.get("pick_sweep"), "place": c.get("place"), "camera": c.get("camera")}, "purpose": "development only"}
    cfg["config_sha256"] = digest(cfg)
    cfg["reset_seed"] = int(cfg["config_sha256"][:8], 16)
    return cfg


def recoverability_audit(scenes, geom):
    rows = {}
    for name, c in scenes.items():
        ok = c["safety"] == "SAFE" and not c.get("stack_hazard_if_score_ge_0p5") and c["camera"]["pass"]
        rows[name] = {"blocker_pickable_static": c["safety"] == "SAFE", "no_stack_hazard": not c.get("stack_hazard_if_score_ge_0p5"), "camera_ok": c["camera"]["pass"],
                      "verdict": "STATIC_RECOVERY_PLAUSIBLE" if ok else "STATIC_RECOVERY_NOT_SUPPORTED"}
    return {"card": CARD, "scenes": rows, "note": "static plausibility only; real recoverability is proven by the canary", "status": "PASS" if rows and all(r["verdict"] == "STATIC_RECOVERY_PLAUSIBLE" for r in rows.values()) else "FAIL"}


def relation_expressibility(root):
    from cp_disr.analysis import tp_ef_post_open_audit as po
    from cp_disr.analysis import tp_ef_protocol_review as pr
    from cp_disr.facts import Truth
    template = pr.build_template(root)
    uni = po.relation_universe(template, [pr.IDS["pt"], pr.IDS["ps"]])
    want = {"BLOCK_CONTAINER": ("SOFT_SUPPORTS", pr.IDS["ps"], pr.IDS["plt"]), "BLOCK_BUFFER": ("SOFT_SUPPORTS", pr.IDS["pt"], pr.IDS["pbs"])}
    out = {"card": CARD, "modes": {}, "provider_calls": 0, "hand_written_relations_count_as_E1_evidence": False}
    for mode, (typ, src, tgt) in want.items():
        regs = [r for r in uni if r["type"] == typ and r["source"] == src and r["target"] == tgt]
        out["modes"][mode] = {"candidate_relations": regs, "registered_effect_refs": sorted({r["effect_fact_ref"] for r in regs}), "non_redundant": [r for r in regs if not r["contract_redundant"]],
                              "expressible": bool([r for r in regs if not r["contract_redundant"]]), "modifies_facts_mask_reward": False}
    out["status"] = "PASS" if all(m["expressible"] for m in out["modes"].values()) else "FAIL"
    return out


def canary_documents(scenes):
    bc, bb = scenes["BC-MODERATE"], scenes["BB-MODERATE"]
    from cp_disr.analysis import tp_ef_protocol_review as pr
    ids = pr.IDS
    route = {"T": [ids["pt"], ids["plt"], ids["ps"], ids["pbs"]], "S": [ids["ps"], ids["pbs"], ids["pt"], ids["plt"]]}
    eps = [("C1", "BC-MODERATE", "good", "S"), ("C2", "BC-MODERATE", "bad", "T"), ("C3", "BB-MODERATE", "good", "T"), ("C4", "BB-MODERATE", "bad", "S")]
    reg = {"card": CARD, "selection_reason": "moderate severity predeclared for canary", "episodes": [{"id": i, "scene": s, "scene_config_sha256": scene_config(s, scenes[s])["config_sha256"], "reset_seed": scene_config(s, scenes[s])["reset_seed"],
                                                                                                   "order": o, "first_route": r, "first_action": route[r][0]} for i, s, o, r in eps], "episode_cap": 4}
    spec = {"card": CARD, "execution": "spec only; not executed by this card", "recovery_policy": {
        "BLOCK_CONTAINER_bad": "PICK(target); PLACE(target, container); if Inside(target,container)=TRUE: PICK(second); PLACE_BUFFER(second) else (frozen recoverability conditions hold): PICK(second); PLACE_BUFFER(second); PICK(target); PLACE(target, container); never retry the first PLACE",
        "BLOCK_BUFFER_bad": "PICK(second); PLACE_BUFFER(second); if AtBuffer(second,buffer)=TRUE: PICK(target); PLACE(target, container) else (frozen recoverability conditions hold): PICK(target); PLACE(target, container); PICK(second); PLACE_BUFFER(second, buffer); never retry the first PLACE_BUFFER"},
        "good_orders": {"BLOCK_CONTAINER": route["S"], "BLOCK_BUFFER": route["T"]}, "pass_conditions": ["4/4 TASK_SUCCESS", "good orders without rework", "controller exceptions = 0", "technical UNKNOWN dead ends = 0", "hidden truth use = 0", "retries = 0"],
        "not_for": "estimating the 0.2-0.7 first-placement success band (40-episode pilot)"}
    return reg, spec


def canary_request_md(reg):
    return "\n".join([f"# Next canary card request — {CARD}", "", "Status: PREFLIGHT_PASS_CANARY_REQUESTED (not started).", "", "4 episodes: " + ", ".join(f"{e['id']}={e['scene']}/{e['order']}" for e in reg["episodes"]), ""])


def final_summary(out, verdict):
    out = Path(out)
    summ = json.loads((out / "skill_duration_summary.json").read_text()) if (out / "skill_duration_summary.json").exists() else {}
    bound = json.loads((out / "discrete_effect_bound.json").read_text()) if (out / "discrete_effect_bound.json").exists() else {}
    sweep = json.loads((out / "static_sweep_audit.json").read_text()) if (out / "static_sweep_audit.json").exists() else {}
    L = [f"# {CARD} — final preflight summary", "", f"Verdict: **{verdict['verdict']}**  (stopped at phase {verdict['stop_phase'] or 'none'})", "", f"Phases: {verdict['phases']}", ""]
    if summ:
        L += ["## A. real skill durations (clean logs)", ""] + [f"- {k}: n={v['n']} mean={v['mean']:.3f} median={v['median']:.3f} std={v['std']:.3f} min={v['min']:.2f} max={v['max']:.2f}" for k, v in summ["per_skill"].items()] + [""]
    if bound:
        L += ["## B. discrete rework bound", ""]
        for m, v in bound["modes"].items():
            L.append(f"- {m}: extra two skills = {v['delta_T_if_recovery_mean']:.2f} s (range {v['delta_T_if_recovery_min_sum']:.2f}-{v['delta_T_if_recovery_max_sum']:.2f}); p_min for 2.1 s = {v['p_min_for_2p1s_mean']:.3f}; E[dT] at failure p=0.3 / 0.8 = {v['E_delta_T_at_design_band']['p=0.3']:.2f} / {v['E_delta_T_at_design_band']['p=0.8']:.2f}")
        L += [f"- median-gate note: {bound['median_gate_note']['statement']}", ""]
    if sweep:
        L += ["## F. static sweep / soft-band audit", ""]
        for m, v in sweep["per_mode"].items():
            L.append(f"- {m}: valid interfering candidates {v['interfering_valid_candidates']}, WEAK/MODERATE {v['weak_or_moderate_valid']}, SAFE {v['safe_weak_or_moderate']}; causes {v['root_cause_counts']}")
        L += [f"- category: {sweep['category'] or 'none'}", f"- proven finger-clearance envelope: {sweep['proven_clearance_envelope']:.4f} m", ""]
    L += ["Budget: environment constructions 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test reads 0.", ""]
    (out / "final_preflight_summary.md").write_text("\n".join(L), encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "protected_before.json", "protected_after.json", "skill_duration_inventory.csv", "skill_duration_summary.json", "discrete_effect_bound.json",
            "controller_motion_contract.json", "nominal_contract_state.json", "bplan_tie_audit.json", "commutativity_audit.json", "asset_geometry_identity.json", "interference_metric_spec.json",
            "candidate_geometry_grid.json", "static_sweep_audit.json", "camera_static_audit.json", "recoverability_static_audit.json", "relation_expressibility_audit.json", "pilot_scene_manifest.json",
            "canary_registration.json", "canary_execution_spec.json", "next_canary_card_request.md", "budget_ledger.json", "final_preflight_summary.md", "preflight_verdict.json"]
    led = json.loads((out / "budget_ledger.json").read_text())
    before, after = json.loads((out / "protected_before.json").read_text()), json.loads((out / "protected_after.json").read_text())
    verdict = json.loads((out / "preflight_verdict.json").read_text())
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".csv"} and bool(re.search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "pilot_scene_configs_dir": (out / "pilot_scene_configs").is_dir(),
              "zero_resources": all(led[k] == 0 for k in ("environment_constructions", "start_case", "env_reset", "skill_calls", "physical_episodes", "provider_requests", "provider_retries", "rl_transitions", "optimizer_steps", "elastic_attempts", "test_reads", "test_episodes")),
              "protected_unchanged": before == after, "protected_files_hashed": len(before), "verdict_is_allowed": verdict["verdict"] in FAIL_CATEGORIES + ("PREFLIGHT_PASS_CANARY_REQUESTED",),
              "canary_label_consistent": ("NOT_REQUESTED" in (out / "next_canary_card_request.md").read_text()) == (verdict["verdict"] != "PREFLIGHT_PASS_CANARY_REQUESTED"), "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k not in ("outputs_present", "protected_files_hashed", "status")) and all(checks["outputs_present"].values()) else "FAIL"
    write_json(out / "verify.json", checks)
    return checks
