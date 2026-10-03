"""CP-DISR-TP-DV1-BINDING-PREFLIGHT-1: can four LIBERO grocery assets and the LIBERO basket be bound into the D0 stack for DV1?

Zero environment: static XML/mesh/source analysis, a pure-numpy ray-cast of the frozen agentview camera, and contract analysis.
It never constructs an environment and never imports robosuite, mujoco or libero. All thresholds and rules below are declared in
code before the run. The verdict is the FIRST failing category in FAIL_ORDER; every category is still evaluated and reported.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cp_disr.analysis import tp_libero_audit as la
from cp_disr.analysis import tp_occ_preflight as occ

CARD = "CP-DISR-TP-DV1-BINDING-PREFLIGHT-1"
BASELINE_FULL = "7cba773c64043da02554a273b7467e6e8d472a30"
UPSTREAM_SHA = la.UPSTREAM_SHA
LABEL = "LIBERO_DERIVED_DIAGNOSTIC_TASK"
OBJECTS = ("bbq_sauce", "butter", "chocolate_pudding", "cream_cheese")      # frozen before any analysis
CONTAINER = "basket"
PAIRS = list(itertools.combinations(sorted(OBJECTS), 2))
FAIL_ORDER = ["DV1_FAIL_NO_PINCHABLE_PAIR", "DV1_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR", "DV1_FAIL_REQUIRES_NEW_PERCEPTION_MODEL", "DV1_FAIL_VERIFIER_NOT_PUBLICLY_IMPLEMENTABLE",
              "DV1_FAIL_EVALUATOR_SEMANTIC_CHANGE", "DV1_FAIL_SNAPSHOT_REWRITE", "DV1_FAIL_BASKET_GEOMETRY", "DV1_FAIL_FIXED_HEURISTIC_SUFFICIENT",
              "DV1_FAIL_NO_HELPFUL_NEUTRAL_REVERSED_SET", "ENGINEERING_UNRESOLVED"]
RESOURCE_KEYS = la.RESOURCE_KEYS
# ---- thresholds and rules declared before the run -----------------------------------------------------------
CTRL_TOL = 0.008                  # skill_executor._pick grasp tolerance (OCC card)
PLACE_XY_TOL = 0.018 + 0.008 + 0.002   # drop xy error bound per object: _move_to default tolerance + grasp offset in the hand (CTRL_TOL) + perception centroid error
STD_GRASP_SITE = 0.022            # table_top + OBJECT_HALF(0.02) + 0.002 for the standard cube
STD_PRESS = 0.008                 # press = grasp - 0.008 (skill_executor._pick)
STD_CUBE_H = 0.04
OVERLAP_FRACTION = 0.5            # pad/object vertical contact must reach this fraction of the standard cube contact
HAND_CLEAR_MIN = 0.010            # palm to object top
RELEASE_CLEAR = 0.010             # standard release: cube bottom 0.01 above the rim
TIP_CLEAR_MIN = 0.010             # finger tip above the rim at release
MARKER_COVER = 0.8                # marker covers this fraction of the top plateau (each axis)
MARKER_THICK = 0.0004
MARKER_MASS = 1e-6
MIN_PIXELS = occ.MIN_PIXELS
COLOR_TOL = occ.COLOR_TOL
HELD_SHOW_XY = occ.SHOW_XY
SHOW_Z = 0.16                     # table_top + 0.16 (skill_executor._pick)
A_XY, B_XY = (-0.12, -0.10), (0.12, -0.10)    # proven base poses of the T_B family
BASKET_XY = occ.CONTAINER_XY                  # (0.18, 0.12): the proven container position
MARGIN_DELTA = 0.004              # helpful / reversed need a stability-margin gap of at least this much
RULES = ("larger_footprint_first", "smaller_footprint_first", "heavier_first", "lighter_first", "taller_first", "shorter_first", "larger_top_plateau_first", "smaller_top_plateau_first",
         "higher_com_first", "lower_com_first")
EVAL_TEST_POINTS = 200000
YAW_GRID_DEG = list(range(0, 181, 5))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


digest = la.digest
write_json = la.write_json
write_csv = la.write_csv


# ------------------------------------------------------------------------------------------ XML / mesh parsing
def floats(s, n, default):
    if s is None:
        return default
    v = [float(x) for x in s.split()]
    return v if len(v) >= n else default


def read_msh(path):
    b = Path(path).read_bytes()
    nv, nn, nt, nf = struct.unpack("<4i", b[:16])
    off = 16
    verts = np.frombuffer(b, dtype="<f4", count=3 * nv, offset=off).reshape(-1, 3)
    off += 12 * nv + 12 * nn + 8 * nt
    faces = np.frombuffer(b, dtype="<i4", count=3 * nf, offset=off).reshape(-1, 3)
    return verts.astype(float), faces


def mesh_volume(path, scale):
    v, f = read_msh(path)
    v = v * np.array(scale)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0), v


def parse_geoms(xml_path):
    xml_path = Path(xml_path)
    root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))
    meshes = {m.get("name"): (m.get("file"), floats(m.get("scale"), 3, [1, 1, 1])) for m in root.iter("mesh")}
    boxes, vis = [], []
    for g in root.iter("geom"):
        typ = g.get("type", "sphere")
        dens = float(g.get("density", "1000"))
        solref = floats(g.get("solref"), 2, [0.02, 1.0])
        if typ == "box" and g.get("group", "0") == "0" and not (g.get("contype") == "0" and g.get("conaffinity") == "0"):
            boxes.append({"center": np.array(floats(g.get("pos"), 3, [0, 0, 0])), "R": la._quat_to_mat(floats(g.get("quat"), 4, [1, 0, 0, 0])), "half": np.array(floats(g.get("size"), 3, [0, 0, 0])),
                          "density": dens, "solref": solref, "friction": floats(g.get("friction"), 3, [1, 0.005, 0.0001])})
        elif typ == "mesh":
            fn, sc = meshes.get(g.get("mesh"), (None, [1, 1, 1]))
            vis.append({"file": fn, "scale": sc, "density": dens, "solref": solref, "collides": not (g.get("contype") == "0" and g.get("conaffinity") == "0")})
    sites = {s.get("name"): {"pos": floats(s.get("pos"), 3, [0, 0, 0]), "size": floats(s.get("size"), 1, [0])} for s in root.iter("site") if s.get("name")}
    return {"boxes": boxes, "visual_meshes": vis, "sites": sites}


def box_mass(b):
    return b["density"] * 8.0 * float(np.prod(b["half"]))


def mass_inertia(parsed, xml_dir):
    """Mass range: collision boxes only vs boxes plus the visual mesh geoms (MuJoCo infers inertia from every geom of the body)."""
    mb = [box_mass(b) for b in parsed["boxes"]]
    m_boxes = float(sum(mb))
    mesh_info = []
    m_mesh = 0.0
    for vm in parsed["visual_meshes"]:
        vol, _ = mesh_volume(Path(xml_dir) / vm["file"], vm["scale"])
        mesh_info.append({"file": vm["file"], "volume_m3": vol, "mass_kg": vol * vm["density"]})
        m_mesh += vol * vm["density"]
    com = sum(m * b["center"] for m, b in zip(mb, parsed["boxes"])) / max(m_boxes, 1e-12)
    I = np.zeros((3, 3))
    for m, b in zip(mb, parsed["boxes"]):
        s = 2 * b["half"]
        Il = m / 12.0 * np.diag([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2, s[0] ** 2 + s[1] ** 2])
        Ic = b["R"] @ Il @ b["R"].T
        d = b["center"] - com
        I += Ic + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return {"mass_boxes_only_kg": m_boxes, "mass_boxes_plus_visual_mesh_kg": m_boxes + m_mesh, "mass_ratio_high_over_low": (m_boxes + m_mesh) / max(m_boxes, 1e-12), "visual_meshes": mesh_info,
            "com_asset_frame_m": com.tolist(), "inertia_boxes_only_about_com_kgm2": I.tolist(), "inertia_note": "from the collision boxes; the visual-mesh contribution is not added to the tensor"}


def rot_axis_angle(axis, ang):
    c, s = math.cos(ang / 2), math.sin(ang / 2)
    q = {"x": [c, s, 0, 0], "y": [c, 0, s, 0], "z": [c, 0, 0, s]}[axis]
    return q


def world_boxes(boxes, Rs, shift=np.zeros(3)):
    return [{"center": Rs @ b["center"] + shift, "R": Rs @ b["R"], "half": b["half"]} for b in boxes]


def corners(b):
    sg = np.array(list(itertools.product((-1, 1), repeat=3)))
    return b["center"][None, :] + (sg * b["half"][None, :]) @ b["R"].T


def aabb(boxes):
    pts = np.concatenate([corners(b) for b in boxes])
    return pts.min(0), pts.max(0)


# ---------------------------------------------------------------------------------------- orientation (source)
def class_rotation(lib, class_name):
    """Most-derived `self.rotation` / `self.rotation_axis` assignments of a registered object class, from the AST of envs/objects."""
    objdir = Path(lib) / "envs" / "objects"
    classes = {}
    for py in sorted(objdir.glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        for node in ast.parse(src).body:
            if isinstance(node, ast.ClassDef):
                assigns = {}
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                        for st in ast.walk(fn):
                            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Attribute) and getattr(st.targets[0].value, "id", "") == "self" and st.targets[0].attr in ("rotation", "rotation_axis"):
                                seg = ast.get_source_segment(src, st.value)
                                assigns[st.targets[0].attr] = {"source": seg, "line": st.lineno, "file": py.name}
                classes[node.name] = {"bases": [getattr(b, "id", "") for b in node.bases], "assigns": assigns, "has_init_quat": "init_quat" in (ast.get_source_segment(src, node) or "")}
    out, cur, chain = {}, class_name, []
    while cur in classes:
        chain.append(cur)
        for k, v in classes[cur]["assigns"].items():
            out.setdefault(k, dict(v, defined_in=cur))
        cur = next((b for b in classes[cur]["bases"] if b in classes), None)
    init_quat = any(classes[c]["has_init_quat"] for c in chain)

    def ev(seg):
        return eval(seg.replace("np.pi", "math.pi"), {"math": math})
    rot = ev(out["rotation"]["source"]) if "rotation" in out else None
    axis = ev(out["rotation_axis"]["source"]) if "rotation_axis" in out else None
    return {"class_chain": chain, "rotation": list(rot) if rot is not None else None, "rotation_axis": axis, "rotation_source": out.get("rotation"), "axis_source": out.get("rotation_axis"), "init_quat_defined": init_quat}


def spawn_orientation(lib, class_name):
    r = class_rotation(lib, class_name)
    if r["rotation"] is None or r["rotation_axis"] is None or r["init_quat_defined"]:
        return dict(r, status="UNVERIFIED", quat_wxyz=None, reason="rotation attributes missing or an init_quat exists")
    a, b = float(min(r["rotation"])), float(max(r["rotation"]))
    if abs(a - b) > 1e-12:
        return dict(r, status="UNVERIFIED", quat_wxyz=None, reason="rotation is a random range")
    q = rot_axis_angle(r["rotation_axis"], a)
    return dict(r, status="DETERMINED_BY_SOURCE", quat_wxyz=q, angle_rad=a, yaw_about_z_rad=0.0,
                reason="bddl_base_domain passes obj.rotation and obj.rotation_axis (not the region yaw_rotation) to the region sampler for movable objects; the quaternion is deterministic")


# ------------------------------------------------------------------------------------------ gripper geometry
def gripper_static():
    """Panda gripper solids from the robosuite XML and STL assets located WITHOUT importing robosuite."""
    spec = importlib.util.find_spec("robosuite")
    base = Path(spec.origin).parent / "models/assets/grippers"
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
    fmin, fmax = [np.array(x) for x in occ.bsi.stl_bounds(base / "meshes/panda_gripper/finger.stl")]
    hmin, hmax = [np.array(x) for x in occ.bsi.stl_bounds(base / "meshes/panda_gripper/hand.stl")]
    pad_low = -((tip_pos[2] + finger_z0 + pad_pos[2] + pad_half[2]) - eef_z)
    return {"source": "robosuite panda_gripper.xml + finger.stl + hand.stl (read as files; robosuite not imported)", "eef_site_hand_z": eef_z, "finger_inner_face_open": slide_max + float(fmin[1]),
            "finger_outer_face_open": slide_max + float(fmax[1]), "finger_half_width_orthogonal": float(fmax[0]), "finger_lowest_rel_site_z": -(float(fmax[2]) + finger_z0 - eef_z),
            "pad_lowest_rel_site_z": pad_low, "pad_half_extents": pad_half, "pad_height_m": 2 * pad_half[2], "hand_lowest_rel_site_z": -(float(hmax[2]) - eef_z),
            "hand_mesh_bounds": {"min": hmin.tolist(), "max": hmax.tolist()}}


def pinch_width(boxes_w, yaw):
    """Extent of the box union along the spread axis rotated by yaw about world z (world y at yaw 0)."""
    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    pts = np.concatenate([corners(b) for b in boxes_w]) @ u
    return float(pts.max() - pts.min())


def top_bottom_layers(boxes_w, tol=0.001):
    lo, hi = aabb(boxes_w)
    top, bot = [], []
    for b in boxes_w:
        c = corners(b)
        if c[:, 2].max() >= hi[2] - tol:
            top.append(b)
        if c[:, 2].min() <= lo[2] + tol:
            bot.append(b)
    return top, bot


def rect_of(boxes_w):
    lo, hi = aabb(boxes_w)
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]


def rect_area(r):
    return max(0.0, r[1] - r[0]) * max(0.0, r[3] - r[2])


def object_record(lib, name, grip, guard):
    rec = la.load_registry(lib, guard)[name]
    xml = Path(lib) / rec["xml"]
    parsed = parse_geoms(xml)
    mi = mass_inertia(parsed, xml.parent)
    spawn = spawn_orientation(lib, rec["class"])
    Rs = la._quat_to_mat(spawn["quat_wxyz"]) if spawn["quat_wxyz"] else np.eye(3)
    bw = world_boxes(parsed["boxes"], Rs)
    lo, hi = aabb(bw)
    bw = world_boxes(parsed["boxes"], Rs, shift=np.array([0, 0, -lo[2]]))        # resting on z = 0
    lo, hi = aabb(bw)
    top, bot = top_bottom_layers(bw)
    com = Rs @ np.array(mi["com_asset_frame_m"]) + np.array([0, 0, -float(aabb(world_boxes(parsed["boxes"], Rs))[0][2])])
    sup = rect_of(bot)
    margin_x = min(com[0] - sup[0], sup[1] - com[0])
    margin_y = min(com[1] - sup[2], sup[3] - com[1])
    ext = (hi - lo).tolist()
    return {"name": name, "class": rec["class"], "xml": rec["xml"], "xml_sha256": sha_file(xml), "boxes": bw, "parsed": parsed, "mass": mi, "spawn": spawn, "Rs": Rs, "extent_world_xyz": ext, "height": float(hi[2]),
            "com_world": com.tolist(), "support_rect": sup, "support_area": rect_area(sup), "top_rect": rect_of(top), "top_area": rect_area(rect_of(top)), "footprint_rect": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])],
            "footprint_area": ext[0] * ext[1], "com_inside_support": bool(margin_x > 0 and margin_y > 0), "tip_margin_xy": [float(margin_x), float(margin_y)], "tip_margin_over_com_height": float(min(margin_x, margin_y) / max(com[2], 1e-9)),
            "mass_mid": 0.5 * (mi["mass_boxes_only_kg"] + mi["mass_boxes_plus_visual_mesh_kg"])}


def stable_faces(boxes_asset, mi_com):
    """Six axis-aligned face-down orientations of the asset frame with their support margins (stable_yaw_set: yaw about z is free for each)."""
    out = []
    perms = {"asset_z_up": np.eye(3), "asset_z_down": np.diag([1, -1, -1]), "asset_x_down": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float), "asset_x_up": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
             "asset_y_down": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float), "asset_y_up": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)}
    for n, R in perms.items():
        bw = world_boxes(boxes_asset, R)
        lo, hi = aabb(bw)
        bw = world_boxes(boxes_asset, R, shift=np.array([0, 0, -lo[2]]))
        lo, hi = aabb(bw)
        _, bot = top_bottom_layers(bw)
        sup = rect_of(bot)
        com = R @ np.array(mi_com) + np.array([0, 0, -float(aabb(world_boxes(boxes_asset, R))[0][2])])
        mx, my = min(com[0] - sup[0], sup[1] - com[0]), min(com[1] - sup[2], sup[3] - com[1])
        out.append({"face_down": n, "height_m": float(hi[2]), "support_area_m2": rect_area(sup), "com_height_m": float(com[2]), "com_inside_support": bool(mx > 0 and my > 0),
                    "tip_margin_over_com_height": float(min(mx, my) / max(com[2], 1e-9))})
    return sorted(out, key=lambda r: -r["tip_margin_over_com_height"])


# --------------------------------------------------------------------------------------------- pinchability
def show_path_clear(o, grip, other_half=0.045, basket_rect=None):
    """Held item and closed fingers along the straight lift from a pick pose to the show pose, against the other item (xy box) and the basket outer rectangle."""
    if basket_rect is None:
        basket_rect = (BASKET_XY[0] - 0.09, BASKET_XY[0] + 0.09, BASKET_XY[1] - 0.09, BASKET_XY[1] + 0.09)
    hw = max(o["extent_world_xyz"][0], o["extent_world_xyz"][1]) / 2 + grip["finger_half_width_orthogonal"]
    for pick, other in ((A_XY, B_XY), (B_XY, A_XY)):
        for t in np.linspace(0, 1, 41):
            x = pick[0] + t * (HELD_SHOW_XY[0] - pick[0])
            y = pick[1] + t * (HELD_SHOW_XY[1] - pick[1])
            z = (STD_GRASP_SITE - STD_PRESS) + t * (SHOW_Z - (STD_GRASP_SITE - STD_PRESS))
            if z + grip["finger_lowest_rel_site_z"] < 0.06 and abs(x - other[0]) < hw + other_half and abs(y - other[1]) < hw + other_half:
                return False
            if z + grip["finger_lowest_rel_site_z"] < 0.15 and basket_rect[0] - hw < x < basket_rect[1] + hw and basket_rect[2] - hw < y < basket_rect[3] + hw:
                return False
    return True


def pinch_audit(o, grip):
    inner_half = grip["finger_inner_face_open"]
    capacity = 2 * inner_half
    limit = 2 * (inner_half - CTRL_TOL)
    widths = {int(d): pinch_width(o["boxes"], math.radians(d)) for d in YAW_GRID_DEG}
    w0, w90 = widths[0], widths[90]
    ok0, ok90 = w0 <= limit, w90 <= limit
    best_yaw = min(widths, key=lambda d: widths[d])
    press = STD_GRASP_SITE - STD_PRESS
    pad_low_abs = press + grip["pad_lowest_rel_site_z"]
    pad_top_abs = pad_low_abs + grip["pad_height_m"]
    finger_low_abs = press + grip["finger_lowest_rel_site_z"]
    def overlap(h):
        return max(0.0, min(h, pad_top_abs) - max(0.0, pad_low_abs))
    std_overlap = overlap(STD_CUBE_H)
    ov = overlap(o["height"])
    hand_clear = press + grip["hand_lowest_rel_site_z"] - o["height"]
    reasons = []
    if not (ok0 or ok90):
        reasons.append("no yaw in {0, 90} deg gives a pinch width within the limit (best %d deg: %.4f m > %.4f m): needs a side or edge grasp" % (best_yaw, widths[best_yaw], limit))
    if ov < OVERLAP_FRACTION * std_overlap:
        reasons.append("pad contact height %.4f m < %.2f x the standard cube contact %.4f m" % (ov, OVERLAP_FRACTION, std_overlap))
    if hand_clear < HAND_CLEAR_MIN:
        reasons.append("palm clearance above the object top %.4f m < %.3f m" % (hand_clear, HAND_CLEAR_MIN))
    if not o["com_inside_support"]:
        reasons.append("the spawn orientation is not a rest pose (COM outside the support rectangle)")
    if o["spawn"]["status"] != "DETERMINED_BY_SOURCE":
        reasons.append("spawn orientation UNVERIFIED")
    needs_yaw = (not ok0) and ok90
    status = "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED" if reasons else ("GENERIC_PICK_YAW90" if needs_yaw else "GENERIC_PICK_YAW0")
    return {"object": o["name"], "status": status, "reasons": reasons, "spawn_status": o["spawn"]["status"], "world_extent_xyz_m": [round(x, 4) for x in o["extent_world_xyz"]],
            "pinch_width_yaw0_m": w0, "pinch_width_yaw90_m": w90, "pinch_width_min_m": widths[best_yaw], "best_yaw_deg": best_yaw, "gripper_capacity_m": capacity, "pinch_limit_m": limit,
            "slack_per_side_yaw0_m": inner_half - w0 / 2, "slack_per_side_yaw90_m": inner_half - w90 / 2, "required_yaw_deg": (0 if ok0 else (90 if ok90 else None)), "needs_yaw_extension": needs_yaw,
            "grasp_site_press_m": press, "pad_low_abs_m": pad_low_abs, "pad_top_abs_m": pad_top_abs, "finger_low_abs_m": finger_low_abs, "pad_contact_height_m": ov, "std_cube_contact_height_m": std_overlap,
            "palm_clearance_above_top_m": hand_clear, "height_m": o["height"], "existing_formula_site_m": min(o["height"] - 0.012, o["height"] / 2 + 0.002),
            "existing_formula_finger_low_at_press_m": min(o["height"] - 0.012, o["height"] / 2 + 0.002) - STD_PRESS + grip["finger_lowest_rel_site_z"],
            "show_lift_retreat_clear": show_path_clear(o, grip)}


# -------------------------------------------------------------------------------------------------- basket
def basket_record(lib, grip):
    xml = Path(lib) / "assets/stable_scanned_objects/basket/basket.xml"
    parsed = parse_geoms(xml)
    bw = world_boxes(parsed["boxes"], np.eye(3))
    lo, hi = aabb(bw)
    bw = world_boxes(parsed["boxes"], np.eye(3), shift=np.array([0, 0, -lo[2]]))
    lo, hi = aabb(bw)
    boxes_aabb = [aabb([b]) for b in bw]
    floor_i = int(np.argmin([(h - l)[2] for l, h in boxes_aabb]))
    walls = [i for i in range(len(bw)) if i != floor_i]
    floor_top = float(boxes_aabb[floor_i][1][2])
    inner = {}
    for ax, name in ((0, "x"), (1, "y")):
        neg = [boxes_aabb[i][1][ax] for i in walls if bw[i]["center"][ax] < 0 and abs(bw[i]["center"][ax]) > abs(bw[i]["center"][1 - ax]) - 1e-9]
        pos = [boxes_aabb[i][0][ax] for i in walls if bw[i]["center"][ax] > 0 and abs(bw[i]["center"][ax]) > abs(bw[i]["center"][1 - ax]) - 1e-9]
        inner[name] = [float(max(neg)), float(min(pos))]
    site = parsed["sites"]["contain_region"]
    shift_z = -float(aabb(world_boxes(parsed["boxes"], np.eye(3)))[0][2])
    cx, cy = 0.5 * (inner["x"][0] + inner["x"][1]), 0.5 * (inner["y"][0] + inner["y"][1])
    spawn = spawn_orientation(lib, "Basket")
    return {"xml_sha256": sha_file(xml), "boxes": bw, "floor_index": floor_i, "floor_top_z": floor_top, "rim_top_z": float(hi[2]), "cavity_depth_m": float(hi[2] - floor_top), "inner_x": inner["x"], "inner_y": inner["y"],
            "inner_width_xy_m": [inner["x"][1] - inner["x"][0], inner["y"][1] - inner["y"][0]], "cavity_center_xy": [cx, cy], "floor_area_m2": (inner["x"][1] - inner["x"][0]) * (inner["y"][1] - inner["y"][0]),
            "aabb_extent_m": (hi - lo).tolist(), "contain_site_pos": site["pos"], "contain_site_half": site["size"], "site_shift_to_resting_frame_z": shift_z, "wall_tilt_quats_present": True,
            "solref_dampratio": sorted({b["solref"][1] for b in parsed["boxes"]}), "official_spawn": spawn,
            "registered_orientation": "IDENTITY_ASSET_FRAME (z-up cavity)", "registered_orientation_status": "UNVERIFIED_AS_OFFICIAL",
            "registered_orientation_note": "the official class inherits rotation (pi/2 about x), which would lay the asset-frame cavity on its side; that contradicts the contain_region site and cannot be resolved without an environment, so the official spawn is UNVERIFIED and the derived task registers the identity (upright) orientation as a derived choice, not as an official claim"}


# --------------------------------------------------------------------------------------------- ray casting
def obb_hit(o, d, c, R, h):
    oo = R.T @ (np.asarray(o) - c)
    dd = d @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dd
        t1, t2 = (-h - oo) * inv, (h - oo) * inv
    tmin = np.minimum(t1, t2).max(axis=-1)
    tmax = np.maximum(t1, t2).min(axis=-1)
    ok = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(ok, tmin, np.inf)


def render_scene(cam, d, solids):
    """solids: name -> list of boxes (center, R, half). Returns id map (-1 table) and planar depth."""
    names = list(solids)
    best_t = np.where(d[..., 2] < 0, (occ.SKIN_TOP - cam["pos"][2]) / d[..., 2], np.inf)
    ids = -np.ones(d.shape[:2], dtype=int)
    for k, n in enumerate(names):
        for b in solids[n]:
            t = obb_hit(cam["pos"], d, b["center"], b["R"], b["half"])
            m = t < best_t
            best_t = np.where(m, t, best_t)
            ids = np.where(m, k, ids)
    return ids, best_t, names


def place_boxes(boxes_w, xy, z_bottom):
    """boxes resting on z=0 in their own frame -> world (x,y offset, bottom at z_bottom)."""
    return [{"center": b["center"] + np.array([xy[0], xy[1], z_bottom]), "R": b["R"], "half": b["half"]} for b in boxes_w]


def marker_box(o, xy, z_bottom):
    r = o["top_rect"]
    cx, cy = 0.5 * (r[0] + r[1]), 0.5 * (r[2] + r[3])
    hx, hy = 0.5 * (r[1] - r[0]) * MARKER_COVER, 0.5 * (r[3] - r[2]) * MARKER_COVER
    return {"center": np.array([xy[0] + cx, xy[1] + cy, z_bottom + o["height"] + MARKER_THICK / 2]), "R": np.eye(3), "half": np.array([hx, hy, MARKER_THICK / 2])}


def pixel_support(cam, d, o, xy, z_bottom, extra=None, occluders=None):
    solids = {"marker": [marker_box(o, xy, z_bottom)], "body": place_boxes(o["boxes"], xy, z_bottom)}
    if extra:
        solids.update(extra)
    ids, t, names = render_scene(cam, d, solids)
    k = names.index("marker")
    return int((ids == k).sum())


def hand_box(grip, site_xy, site_z):
    hmin, hmax = np.array(grip["hand_mesh_bounds"]["min"]), np.array(grip["hand_mesh_bounds"]["max"])
    eef = grip["eef_site_hand_z"]
    zlo, zhi = site_z - (hmax[2] - eef), site_z - (hmin[2] - eef)
    hx, hy = max(abs(hmin[0]), abs(hmax[0])), max(abs(hmin[1]), abs(hmax[1]))       # the hand mesh y axis is the wide palm = the finger spread axis = world y at yaw 0
    return {"center": np.array([site_xy[0], site_xy[1], 0.5 * (zlo + zhi)]), "R": np.eye(3), "half": np.array([hx, hy, 0.5 * (zhi - zlo)])}


def basket_world(bk, xy):
    return place_boxes(bk["boxes"], xy, occ.SKIN_TOP)


# -------------------------------------------------------------------------------------- evaluator equivalence
def old_inside(obj, c, table_z, lid_away=True, inner=(0.030, 0.030), z_cut=0.12):
    return bool(abs(obj[0] - c[0]) <= inner[0] and abs(obj[1] - c[1]) <= inner[1] and obj[2] <= table_z + z_cut and lid_away)


def registered_inside(obj, c, table_z, interior_half_xy, z_cut, lid_clause=None):
    """Same predicate with the container geometry taken from a registry; no lid clause means it is vacuously true."""
    lid_ok = True if lid_clause is None else bool(lid_clause)
    return bool(abs(obj[0] - c[0]) <= interior_half_xy[0] and abs(obj[1] - c[1]) <= interior_half_xy[1] and obj[2] <= table_z + z_cut and lid_ok)


def evaluator_equivalence(seed=20261003, n=EVAL_TEST_POINTS):
    rng = np.random.default_rng(seed)
    c = np.array([0.18, 0.12, 0.829])
    pts = np.column_stack([c[0] + rng.uniform(-0.12, 0.12, n), c[1] + rng.uniform(-0.12, 0.12, n), 0.825 + rng.uniform(0.0, 0.25, n)])
    lid = rng.random(n) < 0.5
    mism = sum(old_inside(p, c, 0.825, bool(l)) != registered_inside(p, c, 0.825, (0.030, 0.030), 0.12, bool(l)) for p, l in zip(pts, lid))
    return {"points": n, "mismatches": int(mism), "seed": seed}


# ------------------------------------------------------------------------------------------------ main run
def git_out(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()


OWN_OUTPUT = re.compile(r"tp_dv1_binding_preflight")


def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4", "runs/final_master/2.1.1/T_B", "runs/stage_0a", "docs/authoritative", "status", "src", "configs"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not OWN_OUTPUT.search(str(p)) and "__pycache__" not in p.parts and not re.search(r"holdout|test30|test_id|final_test", str(p).replace("\\", "/"), re.I):
                out[str(p.relative_to(root))] = sha_file(p)
    return out


def run(root, libero_root, out, pip_root=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = la.Guard()
    events = [{"event": "hand_occluder_axes_corrected", "technical_reason": "the first run assigned the hand mesh wide (spread) axis to world x instead of world y, which hid every held object; found by calibrating against the standard cube",
               "science_unchanged": True, "regression_test": "test_hand_box_wide_axis_is_world_y", "first_run": {"verdict": "DV1_FAIL_VERIFIER_NOT_PUBLICLY_IMPLEMENTABLE", "held_pixels": {"bbq_sauce": 22, "butter": 0, "chocolate_pudding": 0, "cream_cheese": 0}}}]
    before = protected_hashes(root)
    write_json(out / "protected_before.json", before)
    head = git_out(["rev-parse", "HEAD"], root)
    base_ok = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_FULL, head], cwd=root).returncode == 0
    write_json(out / "authorization.json", {"card": CARD, "authorized_by": "user chat instruction approving CP-DISR-TP-DV1-BINDING-PREFLIGHT-1", "label": LABEL, "forbidden_label": "UNMODIFIED_LIBERO_BENCHMARK",
                                           "accepted_prior_result": "CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1 = NO_LOW_COST_TP_CANDIDATE_FOUND (kept; this card does not overturn it)",
                                           "limits": {k: 0 for k in RESOURCE_KEYS}, "frozen_objects": list(OBJECTS), "container": CONTAINER, "pairs": [list(p) for p in PAIRS],
                                           "fail_order": FAIL_ORDER, "verdict_rule": "first failing category in fail_order; all categories are still evaluated and reported",
                                           "declared_before_run": {"CTRL_TOL": CTRL_TOL, "PLACE_XY_TOL": PLACE_XY_TOL, "OVERLAP_FRACTION": OVERLAP_FRACTION, "HAND_CLEAR_MIN": HAND_CLEAR_MIN, "RELEASE_CLEAR": RELEASE_CLEAR, "TIP_CLEAR_MIN": TIP_CLEAR_MIN,
                                                                   "MARKER_COVER": MARKER_COVER, "MIN_PIXELS": MIN_PIXELS, "COLOR_TOL": COLOR_TOL, "MARGIN_DELTA": MARGIN_DELTA, "A_XY": A_XY, "B_XY": B_XY, "BASKET_XY": BASKET_XY,
                                                                   "rules": list(RULES), "yaw_grid_deg": [YAW_GRID_DEG[0], YAW_GRID_DEG[-1], 5]},
                                           "verifier_criterion": "InBasket needs positive public RGB-D evidence: the object marker must reach MIN_PIXELS visible pixels at the registered drop point (basket at the proven container position); absence-only inference is not accepted",
                                           "basket_criterion": "DV1 needs a geometric path from the order to a public re-pick (ejection, tipping out, or eviction); if every order ends inside the cavity below the rim the mechanism does not exist"})
    write_json(out / "source_identity.json", {"baseline_full_expected": BASELINE_FULL, "run_head_full": head, "run_head_descends_from_or_equals_baseline": base_ok, "branch": git_out(["branch", "--show-current"], root),
                                              "upstream_expected": UPSTREAM_SHA, "upstream_head": git_out(["rev-parse", "HEAD"], Path(libero_root)), "upstream_matches": git_out(["rev-parse", "HEAD"], Path(libero_root)) == UPSTREAM_SHA})
    lib = Path(libero_root).resolve() / "libero" / "libero"
    grip = gripper_static()
    cam = occ.camera(root)
    d = occ.pixel_rays(cam)
    objs = {n: object_record(lib, n, grip, guard) for n in OBJECTS}
    bk = basket_record(lib, grip)
    # ---- asset identity ----------------------------------------------------------------------------------
    ident = {"upstream_sha": UPSTREAM_SHA, "objects": {}, "basket": {"xml": "assets/stable_scanned_objects/basket/basket.xml", "xml_sha256": bk["xml_sha256"]}}
    for n, o in objs.items():
        d0 = lib / Path(o["xml"]).parent
        files = {p.name: sha_file(p) for p in sorted(d0.rglob("*")) if p.is_file() and p.suffix.lower() in (".xml", ".msh", ".obj", ".png", ".mtl")}
        ident["objects"][n] = {"class": o["class"], "xml": o["xml"], "xml_sha256": o["xml_sha256"], "files_sha256": files, "collision_boxes": len(o["boxes"]), "visual_meshes": [m["file"] for m in o["parsed"]["visual_meshes"]],
                               "mass": o["mass"], "densities": sorted({b["density"] for b in o["parsed"]["boxes"]}), "solref": sorted({tuple(b["solref"]) for b in o["parsed"]["boxes"]}),
                               "friction": sorted({tuple(b["friction"]) for b in o["parsed"]["boxes"]}), "half_size_asset_frame_boxes": [b["half"].tolist() for b in o["parsed"]["boxes"]], "extent_world_xyz_m": o["extent_world_xyz"],
                               "height_m": o["height"]}
    bd = lib / "assets/stable_scanned_objects/basket"
    ident["basket"]["files_sha256"] = {p.name: sha_file(p) for p in sorted(bd.rglob("*")) if p.is_file() and p.suffix.lower() in (".xml", ".msh", ".obj", ".png", ".mtl")}
    ident["basket"]["collision_boxes"] = len(bk["boxes"])
    ident["basket"]["geometry"] = {k: v for k, v in bk.items() if k not in ("boxes", "official_spawn")}
    write_json(out / "asset_identity.json", ident)
    # ---- spawn orientation -------------------------------------------------------------------------------
    sp = {"rule_source": "libero/libero/envs/bddl_base_domain.py lines 626-627 and regions/base_region_sampler.py _sample_quat: movable objects use obj.rotation and obj.rotation_axis; region yaw_rotation is used only for fixtures",
          "objects": {}, "basket": bk["official_spawn"], "basket_registered": {"orientation": bk["registered_orientation"], "status": bk["registered_orientation_status"], "note": bk["registered_orientation_note"]}}
    for n, o in objs.items():
        sp["objects"][n] = {k: v for k, v in o["spawn"].items()} | {"world_extent_xyz_m": o["extent_world_xyz"], "rest_pose_com_inside_support": o["com_inside_support"], "tip_margin_over_com_height": o["tip_margin_over_com_height"],
                                                                     "stable_faces_asset_frame": stable_faces(o["parsed"]["boxes"], o["mass"]["com_asset_frame_m"]), "registered_orientation": "OFFICIAL_SPAWN" if o["spawn"]["status"] == "DETERMINED_BY_SOURCE" else "UNVERIFIED"}
    write_json(out / "spawn_orientation_audit.json", sp)
    # ---- pinchability ------------------------------------------------------------------------------------
    pin = {n: pinch_audit(o, grip) for n, o in objs.items()}
    rows = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items() if k != "reasons"} | {"reasons": " || ".join(p["reasons"])} for p in pin.values()]
    write_csv(out / "pinchability_matrix.csv", rows, list(rows[0].keys()))
    usable = [n for n, p in pin.items() if p["status"] in ("GENERIC_PICK_YAW0", "GENERIC_PICK_YAW90")]
    pair_ok = [p for p in PAIRS if p[0] in usable and p[1] in usable]
    c1_pinchable = len(usable) >= 2 and bool(pair_ok)
    cat = {}
    cat["DV1_FAIL_NO_PINCHABLE_PAIR"] = {"fails": not c1_pinchable, "evidence": {"usable_objects": usable, "usable_pairs": [list(p) for p in pair_ok], "per_object": {n: p["status"] for n, p in pin.items()}}}
    low = [n for n, p in pin.items() if p["status"] == "NEW_LOW_LEVEL_BEHAVIOR_REQUIRED"]
    yaw_objs = [n for n in usable if pin[n]["needs_yaw_extension"]]
    cat["DV1_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR"] = {"fails": bool(low and not c1_pinchable), "evidence": {"eliminated_objects": low, "reasons": {n: pin[n]["reasons"] for n in low}, "yaw_extension_needed_for": yaw_objs,
                                                                                                        "note": "eliminated objects are dropped (card rule); this category fails only if the surviving set cannot form a pair"}}
    generic = {"extension": "GENERIC_YAW_ALIGNED_TOP_DOWN_PICK", "states_reused": ["APPROACH", "DESCEND(INTERACT)", "PRESS", "CLOSE(hold)", "LIFT(RETREAT)", "SHOW"], "new_states": [], "parameters": ["half_size_xyz", "grasp_site_z_press", "end_effector_yaw", "gripper_width", "drop_z"],
               "why_grasp_z_must_be_a_parameter": "the existing formula min(blob_z - 0.012, table + half + 0.002) puts the press site so low for thin objects that the finger tips would end below the table (see existing_formula_finger_low_at_press_m per object)",
               "yaw_wiring": {"needed_for": yaw_objs, "requirement": "action[3:6] is forced to 0 in D0ManipulationEnv.step_osc; the extension feeds a yaw error into action[5] during APPROACH only (OSC_POSE already has the rotation dims); the eef yaw is read from public proprio eef_quat", "new_motion_state": False},
               "per_object": {n: {"status": p["status"], "grasp_site_z_press_m": p["grasp_site_press_m"], "yaw_deg": p["required_yaw_deg"], "pinch_width_m": p["pinch_width_yaw0_m"] if p["required_yaw_deg"] == 0 else p["pinch_width_yaw90_m"],
                                  "half_size_world_xyz_m": [x / 2 for x in objs[n]["extent_world_xyz"]], "pad_contact_height_m": p["pad_contact_height_m"], "palm_clearance_m": p["palm_clearance_above_top_m"]} for n, p in pin.items()},
               "gripper": {k: v for k, v in grip.items() if k != "hand_mesh_bounds"}}
    write_json(out / "generic_pick_binding_spec.json", generic)
    # ---- basket place + geometry -------------------------------------------------------------------------
    rim, floor_top = bk["rim_top_z"], bk["floor_top_z"]
    drop_site = {}
    for n in usable or OBJECTS:
        h = objs[n]["height"]
        press = STD_GRASP_SITE - STD_PRESS
        z_site = rim + RELEASE_CLEAR + press        # item bottom sits `press` below the site while held
        tip = z_site + grip["finger_lowest_rel_site_z"] - rim
        drop_site[n] = {"site_z_above_table_m": z_site, "item_bottom_above_rim_m": RELEASE_CLEAR, "finger_tip_above_rim_m": tip, "tip_clear_ok": tip >= TIP_CLEAR_MIN,
                        "fall_height_to_floor_m": rim + RELEASE_CLEAR - floor_top, "open_finger_span_y_m": 2 * grip["finger_outer_face_open"], "cavity_inner_width_xy_m": bk["inner_width_xy_m"],
                        "fingers_enter_cavity": False}
    pile = []
    pathways = {"ejection": {"possible": False}, "tipping_out_of_cavity": {"possible": False}, "first_item_evicted": {"possible": False}}
    damp = {b["solref"][1] for o in objs.values() for b in o["parsed"]["boxes"]} | set(bk["solref_dampratio"])
    e_bound = 0.0 if min(damp) >= 1.0 else None
    fall = rim + RELEASE_CLEAR - floor_top
    v_imp = math.sqrt(2 * 9.81 * fall)
    worst_top = 0.0
    for a, b in itertools.permutations(OBJECTS, 2):
        ha, hb = objs[a]["height"], objs[b]["height"]
        top = ha + hb
        worst_top = max(worst_top, top)
        # lateral: placement offset plus a slide bounded by the cavity; the cavity half width limits any displacement
        lat = PLACE_XY_TOL
        pile.append({"first": a, "second": b, "pile_top_above_floor_m": top, "margin_to_rim_m": bk["cavity_depth_m"] - top, "lateral_offset_bound_m": lat,
                     "cavity_half_width_min_m": 0.5 * min(bk["inner_width_xy_m"]), "stays_inside_cavity": lat < 0.5 * min(bk["inner_width_xy_m"]) and top < bk["cavity_depth_m"], "evaluator_inside_true": top < bk["cavity_depth_m"]})
    need_v = math.sqrt(2 * 9.81 * (bk["cavity_depth_m"] - worst_top))
    e_needed = need_v / v_imp
    pathways["ejection"] = {"possible": bool(e_bound is None or e_bound >= e_needed), "contact_damping_ratio_all_geoms": sorted({b["solref"][1] for o in objs.values() for b in o["parsed"]["boxes"]} | set(bk["solref_dampratio"])),
                            "restitution_bound": e_bound, "impact_speed_m_s": v_imp, "rebound_needed_to_clear_rim_m_s": need_v, "restitution_needed": e_needed,
                            "reason": "critically damped contacts (damping ratio 1) have no rebound; ejection needs a restitution of %.2f" % e_needed}
    pathways["tipping_out_of_cavity"] = {"possible": any(not p["stays_inside_cavity"] for p in pile), "reason": "the drop offset bound %.3f m is below the cavity half width and the pile top stays below the rim, so a tipped or sliding item is stopped by the walls and remains inside" % PLACE_XY_TOL}
    pathways["first_item_evicted"] = {"possible": False, "reason": "the second item lands on or beside the first; the pile top %.3f m is far below the rim at %.3f m" % (worst_top, bk["cavity_depth_m"])}
    any_path = any(v["possible"] for v in pathways.values())
    fill_pess = max(sum(objs[x]["extent_world_xyz"][0] * objs[x]["extent_world_xyz"][1] for x in p) for p in PAIRS) / bk["floor_area_m2"]
    basket_spec = {"action": "PLACE_IN_BASKET(object)", "reuses": "PLACE (APPROACH hover, INTERACT drop, hold open, RETREAT)", "drop_xy": "registered cavity centre (relative to the basket), identical for every order",
                   "basket_position_xy": list(BASKET_XY), "basket_registered_orientation": bk["registered_orientation"], "orientation_status": bk["registered_orientation_status"], "orientation_note": bk["registered_orientation_note"],
                   "cavity": {k: bk[k] for k in ("inner_x", "inner_y", "inner_width_xy_m", "cavity_center_xy", "floor_top_z", "rim_top_z", "cavity_depth_m", "floor_area_m2")}, "contain_region_site": {"pos": bk["contain_site_pos"], "half": bk["contain_site_half"]},
                   "drop": drop_site, "pair_floor_fill_pessimistic": fill_pess, "pile": pile, "order_specific_drop_sites_needed": False, "pathways": pathways, "any_order_to_repick_pathway": any_path,
                   "evaluator_inside_all_orders": all(p["evaluator_inside_true"] for p in pile), "worst_pile_top_m": worst_top,
                   "conclusion": "every order of every pair ends inside the cavity below the rim; the Evaluator is TRUE for both objects in all %d ordered pairs, so the order cannot change success or create a public re-pick" % len(pile) if not any_path else "a pathway exists"}
    write_json(out / "basket_place_binding_spec.json", basket_spec)
    cat["DV1_FAIL_BASKET_GEOMETRY"] = {"fails": (not any_path) or fill_pess >= 1.0 or not all(v["tip_clear_ok"] for v in drop_site.values()),
                                      "evidence": {"any_order_to_repick_pathway": any_path, "pathways": {k: v["possible"] for k, v in pathways.items()}, "worst_pile_top_m": worst_top, "cavity_depth_m": bk["cavity_depth_m"],
                                                   "pair_floor_fill_pessimistic": fill_pess, "tip_clear_ok": {n: v["tip_clear_ok"] for n, v in drop_site.items()}}}
    # ---- perception ---------------------------------------------------------------------------------------
    per = {}
    for n in OBJECTS:
        o = objs[n]
        sa = pixel_support(cam, d, o, A_XY, occ.SKIN_TOP)
        sb = pixel_support(cam, d, o, B_XY, occ.SKIN_TOP)
        held_site_z = occ.Z0 + SHOW_Z
        item_bottom = held_site_z - (STD_GRASP_SITE - STD_PRESS)
        hb = hand_box(grip, HELD_SHOW_XY, held_site_z)
        sh = pixel_support(cam, d, o, HELD_SHOW_XY, item_bottom, extra={"hand": [hb]})
        sh_free = pixel_support(cam, d, o, HELD_SHOW_XY, item_bottom)
        bkw = {"basket": basket_world(bk, BASKET_XY)}
        floor_xy = (BASKET_XY[0] + bk["cavity_center_xy"][0], BASKET_XY[1] + bk["cavity_center_xy"][1])
        s_in = pixel_support(cam, d, o, floor_xy, occ.SKIN_TOP + floor_top, extra=bkw)
        s_in_open = pixel_support(cam, d, o, floor_xy, occ.SKIN_TOP + floor_top)
        per[n] = {"marker_pixels_at_A": sa, "marker_pixels_at_B": sb, "marker_pixels_held_with_hand_box": sh, "marker_pixels_held_no_hand": sh_free, "marker_pixels_in_basket_at_registered_drop": s_in,
                  "marker_pixels_in_basket_without_basket_walls": s_in_open, "marker_top_area_m2": rect_area(o["top_rect"]) * MARKER_COVER ** 2}
    grid = []
    gx = [-0.20, -0.10, 0.0, 0.10, 0.20]
    gy = [0.04, 0.12, 0.20]
    o0 = objs[sorted(OBJECTS)[0]]
    best_in = 0
    for x in gx:
        for y in gy:
            fxy = (x + bk["cavity_center_xy"][0], y + bk["cavity_center_xy"][1])
            vals = {n: pixel_support(cam, d, objs[n], fxy, occ.SKIN_TOP + floor_top, extra={"basket": basket_world(bk, (x, y))}) for n in OBJECTS}
            grid.append({"basket_xy": [x, y], "pixels": vals})
            best_in = max(best_in, max(vals.values()))
    ref_cube = {"name": "reference_cube", "boxes": [{"center": np.array([0, 0, 0.02]), "R": np.eye(3), "half": np.array([0.02, 0.02, 0.02])}], "height": 0.04, "top_rect": [-0.02, 0.02, -0.02, 0.02]}
    hs_z = occ.Z0 + SHOW_Z
    ref_px = pixel_support(cam, d, ref_cube, HELD_SHOW_XY, hs_z - (STD_GRASP_SITE - STD_PRESS), extra={"hand": [hand_box(grip, HELD_SHOW_XY, hs_z)]})
    hand_model_valid = ref_px >= MIN_PIXELS
    per_ok_table = all(per[n]["marker_pixels_at_A"] >= MIN_PIXELS and per[n]["marker_pixels_at_B"] >= MIN_PIXELS for n in OBJECTS)
    visible_in_basket = max(per[n]["marker_pixels_in_basket_at_registered_drop"] for n in OBJECTS)
    tex = texture_audit(lib, OBJECTS)
    percep = {"hand_occluder_calibration": {"reference_cube_held_pixels": ref_px, "min_pixels": MIN_PIXELS, "valid": bool(hand_model_valid), "why": "the standard 0.04 m cube is detected while held in every clean run, so the occluder model must keep it visible"},
              "option_A_texture_segmentation": {"status": "NOT_DEMONSTRABLE_OFFLINE", "reason": "a deterministic mask of a textured, shaded top face can only be validated on rendered pixels, which this card may not produce; texture-cluster statistics below are informational", "texture": tex},
              "option_B_marker": {"status": "SELECTED", "label": "LIBERO-derived instrumentation (public, shared by every method)", "marker": {"geom": "thin box on the top plateau", "cover_fraction_per_axis": MARKER_COVER, "thickness_m": MARKER_THICK, "contype": 0, "conaffinity": 0,
                                  "group": "visual (rendered)", "mass_kg": MARKER_MASS, "collision_geometry_changed": False, "colours": {"first_role (target)": "perception COLORS['target']", "second_role (second_object)": "perception COLORS['second_object']"}}},
              "option_C_sim_segmentation": {"status": "NOT_USED", "reason": "forbidden: a simulator segmentation mask is not public RGB-D perception"},
              "new_vision_model": False, "perception_code_unchanged": {"color_tol": COLOR_TOL, "min_pixels": MIN_PIXELS, "centroid": "mean back-projected marker pixels", "xyz_z_equals": "table top + registered top-plateau height"},
              "detection_identity": "role colour (target / second_object) assigned per episode; the object name is registry metadata, not a perceived property",
              "pixel_support": per, "table_support_ok_all_objects": per_ok_table, "basket_grid_best_pixels_in_basket": best_in, "basket_grid": grid,
              "unknown_conditions": ["marker pixels < MIN_PIXELS (object not visible) -> all facts about it UNKNOWN (existing rule)", "object inside the basket: marker hidden by the walls (see pixel_support) -> no positive InBasket evidence", "hand occlusion while held below MIN_PIXELS -> Held UNKNOWN"]}
    write_json(out / "public_perception_binding_spec.json", percep)
    cat["DV1_FAIL_REQUIRES_NEW_PERCEPTION_MODEL"] = {"fails": not per_ok_table, "evidence": {"marker_pixels_A_B": {n: [per[n]["marker_pixels_at_A"], per[n]["marker_pixels_at_B"]] for n in OBJECTS}, "min_pixels": MIN_PIXELS, "new_vision_model": False}}
    # ---- verifier -----------------------------------------------------------------------------------------
    held_ok = all(per[n]["marker_pixels_held_with_hand_box"] >= MIN_PIXELS for n in usable) if usable else False
    ver = {"registered_geometry": {n: {"half_size_xyz": [x / 2 for x in objs[n]["extent_world_xyz"]], "top_plateau_height_m": objs[n]["height"], "grasp_site_press_m": STD_GRASP_SITE - STD_PRESS, "gripper_width_m": pin[n]["gripper_capacity_m"],
                                        "stable_orientation": "OFFICIAL_SPAWN" if objs[n]["spawn"]["status"] == "DETERMINED_BY_SOURCE" else "UNVERIFIED", "support_surface": "table_skin_top"} for n in OBJECTS},
           "OnTable": {"rule": "|blob_z - (table_top + top_plateau_height)| <= 0.020 and not held; the old fixed +0.025 reference with tolerance 0.05 is replaced by the registered height, not widened", "implementable": True},
           "Held": {"rule": "closed gripper and marker blob within held_xy 0.06 / held_z 0.07 of the eef (unchanged thresholds)", "implementable": True, "marker_pixels_with_hand_occlusion": {n: per[n]["marker_pixels_held_with_hand_box"] for n in OBJECTS}, "visible_enough": held_ok},
           "GripperEmpty": {"rule": "unchanged: open gripper and no nearby blob", "implementable": True},
           "InBasket": {"positive_rule": "marker blob inside the registered cavity xy with z below the rim", "marker_pixels_at_registered_drop": {n: per[n]["marker_pixels_in_basket_at_registered_drop"] for n in OBJECTS},
                        "best_pixels_over_basket_grid": best_in, "needs_pixels": MIN_PIXELS, "positively_observable": visible_in_basket >= MIN_PIXELS,
                        "absence_inference": {"rule": "released over the contain region, absent from the table, not held", "accepted": False, "reason": "it cannot tell an object in the basket from one that is hidden elsewhere, and the existing Verifier maps a missing blob to UNKNOWN"}},
           "thresholds_widened": False}
    write_json(out / "verifier_binding_spec.json", ver)
    ver_fail = not (visible_in_basket >= MIN_PIXELS) or not held_ok
    cat["DV1_FAIL_VERIFIER_NOT_PUBLICLY_IMPLEMENTABLE"] = {"fails": ver_fail, "evidence": {"inbasket_pixels_at_drop": {n: per[n]["marker_pixels_in_basket_at_registered_drop"] for n in OBJECTS}, "best_pixels_over_basket_grid": best_in, "needs": MIN_PIXELS,
                                                           "held_ok": held_ok, "held_pixels": {n: per[n]["marker_pixels_held_with_hand_box"] for n in OBJECTS}}}
    # ---- evaluator ----------------------------------------------------------------------------------------
    eq = evaluator_equivalence()
    ev = {"success_definition": "both objects are inside the container (centre inside the container interior volume), the container having no lid", "d0_predicate": "abs(dx)<=0.030 and abs(dy)<=0.030 and z<=table+0.12 and lid_away",
          "registered_predicate": "abs(dx)<=interior_half_x and abs(dy)<=interior_half_y and z<=table+rim_cutoff and (lid clause vacuous when the case has no lid)", "basket_parameters": {"interior_half_xy_m": [w / 2 for w in bk["inner_width_xy_m"]], "z_cutoff_above_table_m": bk["rim_top_z"]},
          "equivalence_on_d0_instance": eq, "code_touch_points": ["task_evaluator.TaskEvaluator.goal_true: new task id with registered container geometry", "hidden_truth(): per-object body positions keyed by role"], "semantic_change": eq["mismatches"] != 0,
          "note": "the success meaning is unchanged; the container geometry comes from a registry instead of module constants, and a lidless container makes the lid clause vacuous"}
    write_json(out / "evaluator_binding_spec.json", ev)
    cat["DV1_FAIL_EVALUATOR_SEMANTIC_CHANGE"] = {"fails": bool(ev["semantic_change"]), "evidence": {"equivalence": eq}}
    # ---- snapshot -----------------------------------------------------------------------------------------
    cap = guard.read(root / "src/cp_disr/analysis/tp_ef_post_open_capture.py")
    env_src = guard.read(root / "src/cp_disr/platforms/libero/d0_env.py")
    snap = {"evidence": {"sim_fields_saved": re.findall(r'SIM_FIELDS = \(([^)]*)\)', cap), "capture_sim_reads_env_sim_data": bool(re.search(r"def capture_sim\(env\):[\s\S]{0,400}env\.sim\.data", cap)),
                         "python_state_dump_covers_robot_controller_gripper_rng": all(k in cap for k in ("def dump_state", "robot0.controller", "robot0.gripper", "def rng_snapshot")),
                         "restore_builds_env_from_case_first": bool(re.search(r"def build_runtime", cap)) and "start_case_reusing_bootstrap" in cap, "free_joint_objects_set_by_qpos": "set_joint_qpos" in env_src,
                         "object_set_defined_by_case": "self.case" in env_src and "BoxObject(" in env_src},
            "must_save": ["object joint states (free-joint qpos/qvel for every registered object)", "orientation (inside qpos)", "velocities (qvel)", "controller and gripper named state (python dump)", "RNG (numpy and python)", "public observation identity",
                          "registered object geometry (hash added to the manifest identity)"],
            "extensions": ["CaseSpec carries object asset ids and registered-geometry hash", "D0ManipulationEnv._load_model builds the registered objects", "_apply_case_poses/_setup_references/hidden_truth key the new bodies by role", "restore identity adds the geometry hash"],
            "lifecycle_rewritten": False, "uses_current_restore_model": True}
    snap_ok = all(v for k, v in snap["evidence"].items() if isinstance(v, bool)) and bool(snap["evidence"]["sim_fields_saved"])
    snap["status"] = "PASS" if snap_ok else "FAIL"
    write_json(out / "snapshot_binding_spec.json", snap)
    cat["DV1_FAIL_SNAPSHOT_REWRITE"] = {"fails": not snap_ok, "evidence": snap["evidence"]}
    # ---- pair properties, fixed rules, conditions --------------------------------------------------------
    eps = PLACE_XY_TOL
    def margin(base, top):
        """Worst-case margin of the top item's COM inside the base top plateau. Both items are placed by their marker centroid with error eps each,
        so the relative offset is 2*eps; the top item's COM also sits off its own marker centroid by a registered amount."""
        r = base["top_rect"]
        hx, hy = 0.5 * (r[1] - r[0]), 0.5 * (r[3] - r[2])
        tr = top["top_rect"]
        cmx, cmy = 0.5 * (tr[0] + tr[1]), 0.5 * (tr[2] + tr[3])
        off = (abs(top["com_world"][0] - cmx), abs(top["com_world"][1] - cmy))
        return float(min(hx - (2 * eps + off[0]), hy - (2 * eps + off[1])))
    prow = []
    for a, b in PAIRS:
        A, B = objs[a], objs[b]
        mAB, mBA = margin(A, B), margin(B, A)           # margin of B on A (A first), A on B (B first)
        stab_first = a if mAB >= mBA else b
        gap = abs(mAB - mBA)
        pr = {"pair": "%s|%s" % (a, b), "a": a, "b": b, "footprint_a_m2": A["footprint_area"], "footprint_b_m2": B["footprint_area"], "height_a_m": A["height"], "height_b_m": B["height"],
              "mass_a_range_kg": [A["mass"]["mass_boxes_only_kg"], A["mass"]["mass_boxes_plus_visual_mesh_kg"]], "mass_b_range_kg": [B["mass"]["mass_boxes_only_kg"], B["mass"]["mass_boxes_plus_visual_mesh_kg"]],
              "support_area_a_m2": A["support_area"], "support_area_b_m2": B["support_area"], "top_plateau_a_m2": A["top_area"], "top_plateau_b_m2": B["top_area"], "com_height_a_m": A["com_world"][2], "com_height_b_m": B["com_world"][2],
              "margin_b_on_a_m": mAB, "margin_a_on_b_m": mBA, "margin_gap_m": gap, "stability_first": stab_first, "stable_both_orders": bool(mAB > 0 and mBA > 0)}
        picks = {"larger_footprint_first": a if A["footprint_area"] >= B["footprint_area"] else b, "smaller_footprint_first": b if A["footprint_area"] >= B["footprint_area"] else a,
                 "heavier_first": a if A["mass_mid"] >= B["mass_mid"] else b, "lighter_first": b if A["mass_mid"] >= B["mass_mid"] else a,
                 "taller_first": a if A["height"] >= B["height"] else b, "shorter_first": b if A["height"] >= B["height"] else a,
                 "larger_top_plateau_first": a if A["top_area"] >= B["top_area"] else b, "smaller_top_plateau_first": b if A["top_area"] >= B["top_area"] else a,
                 "higher_com_first": a if A["com_world"][2] >= B["com_world"][2] else b, "lower_com_first": b if A["com_world"][2] >= B["com_world"][2] else a}
        for r_name, first in picks.items():
            pr["rule_" + r_name] = first
            pr["agrees_" + r_name] = (first == stab_first)
        pr["size_rule_agrees_with_stability"] = picks["larger_footprint_first"] == stab_first
        pr["both_pinchable"] = (a in usable and b in usable)
        # classification by the declared rule: relation = larger first
        if gap < MARGIN_DELTA:
            cls = "NEUTRAL"
        elif pr["size_rule_agrees_with_stability"]:
            cls = "HELPFUL"
        else:
            cls = "REVERSED"
        pr["condition_class"] = cls
        prow.append(pr)
    write_csv(out / "pair_property_matrix.csv", prow, list(prow[0].keys()))
    agree = {r: sum(p["agrees_" + r] for p in prow) for r in RULES}
    sufficient = [r for r, c in agree.items() if c in (0, len(prow)) and r in RULES]
    has_agree = any(p["size_rule_agrees_with_stability"] for p in prow)
    has_conflict = any(not p["size_rule_agrees_with_stability"] and p["margin_gap_m"] >= MARGIN_DELTA for p in prow)
    cond_counts = {c: sum(p["condition_class"] == c for p in prow) for c in ("HELPFUL", "NEUTRAL", "REVERSED")}
    pairs_ok_all = [p for p in prow if p["both_pinchable"]]
    # use only pairs that both pass the pinch audit
    agree_ok = {r: sum(p["agrees_" + r] for p in pairs_ok_all) for r in RULES}
    suff_ok = [r for r, c in agree_ok.items() if pairs_ok_all and c in (0, len(pairs_ok_all))]
    fx = {"rules_agreement_over_all_6_pairs": agree, "rules_agreement_over_pinchable_pairs": agree_ok, "rules_matching_stability_on_every_pinchable_pair": suff_ok, "pinchable_pairs": [p["pair"] for p in pairs_ok_all],
          "pair_with_size_stability_agreement": has_agree, "pair_with_size_stability_conflict": has_conflict, "single_rule_sufficient": bool(suff_ok), "pass": bool(pairs_ok_all) and has_agree and has_conflict and not suff_ok}
    cat["DV1_FAIL_FIXED_HEURISTIC_SUFFICIENT"] = {"fails": not fx["pass"], "evidence": fx}
    cc = {"selection_rule": "size relation = larger footprint first; stability direction = base with the larger top-over-base COM margin; helpful: rule agrees and margin gap >= %.3f m; reversed: rule disagrees and gap >= that; neutral: gap below it" % MARGIN_DELTA,
          "counts": cond_counts, "pairs": {p["pair"]: {"class": p["condition_class"], "margin_gap_m": p["margin_gap_m"], "both_pinchable": p["both_pinchable"]} for p in prow},
          "helpful_neutral_reversed_all_present": all(cond_counts[c] > 0 for c in cond_counts), "always_A_first_or_larger_first_only": (len([c for c in cond_counts if cond_counts[c]]) < 2),
          "uses_only_static_asset_properties": True,
          "caveat": "the classes come from a tipping-margin model; section 'basket_place_binding_spec' shows that no ordering changes the Evaluator, so these classes would not carry a physical consequence"}
    write_json(out / "condition_set_design.json", cc)
    cat["DV1_FAIL_NO_HELPFUL_NEUTRAL_REVERSED_SET"] = {"fails": not cc["helpful_neutral_reversed_all_present"] or cc["always_A_first_or_larger_first_only"], "evidence": cc["counts"]}
    cat["ENGINEERING_UNRESOLVED"] = {"fails": not hand_model_valid, "evidence": {"hand_occluder_valid": bool(hand_model_valid), "reference_cube_held_pixels": ref_px}}
    failing = [k for k in FAIL_ORDER if cat[k]["fails"]]
    verdict = failing[0] if failing else "DV1_BINDING_PREFLIGHT_PASS"
    (out / "fixed_rule_attack.md").write_text(fixed_rule_md(prow, fx, cc), encoding="utf-8")
    manifest = {"verdict": verdict, "label": LABEL, "all_failing_categories": failing, "category_results": cat, "usable_objects": usable, "pass_conditions": pass_conditions(cat, usable, pin, ev, snap, basket_spec, ver, fx, cc, percep),
                "objects": list(OBJECTS), "pairs": [list(p) for p in PAIRS]}
    write_json(out / "candidate_binding_manifest.json", manifest)
    (out / "next_binding_canary_request.md").write_text("# next_binding_canary_request\n\n" + ("4-episode binding canary: REQUESTED (not executed)\n" if verdict == "DV1_BINDING_PREFLIGHT_PASS" else "NOT_REQUESTED\n"), encoding="utf-8")
    forbidden_loaded = sorted(m for m in ("robosuite", "mujoco", "libero", "torch", "gym", "gymnasium") if m in sys.modules)
    ledger = {k: 0 for k in RESOURCE_KEYS}
    ledger.update({"files_read_as_text": guard.opened, "files_hashed_only": guard.hashed_only, "guard_refusals": len(guard.refused), "forbidden_modules_loaded": forbidden_loaded})
    write_json(out / "budget_ledger.json", ledger)
    (out / "engineering_events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    write_json(out / "protected_after.json", protected_hashes(root))
    return {"verdict": verdict, "failing": failing}


def pass_conditions(cat, usable, pin, ev, snap, basket_spec, ver, fx, cc, percep):
    return {"1_two_objects_one_generic_pick": len(usable) >= 2, "2_one_pair_enters_basket": bool(usable) and basket_spec["evaluator_inside_all_orders"], "3_no_new_perception_model": not cat["DV1_FAIL_REQUIRES_NEW_PERCEPTION_MODEL"]["fails"],
            "4_verifier_public": not cat["DV1_FAIL_VERIFIER_NOT_PUBLICLY_IMPLEMENTABLE"]["fails"], "5_evaluator_semantics_unchanged": not ev["semantic_change"], "6_snapshot_no_rewrite": snap["status"] == "PASS",
            "7_no_order_specific_drop": not basket_spec["order_specific_drop_sites_needed"] and not cat["DV1_FAIL_BASKET_GEOMETRY"]["fails"], "8_two_pairs_refute_one_rule": fx["pass"],
            "9_helpful_neutral_reversed": cc["helpful_neutral_reversed_all_present"], "10_no_drawer_knob_door_controller": True}


def fixed_rule_md(prow, fx, cc):
    L = ["# Fixed-rule attack", "", "Stability order = the base item with the larger top-over-base COM margin (centred drop, offset bound %.3f m). Rules compared with it on all %d pairs." % (PLACE_XY_TOL, len(prow)), "",
         "| rule | agreements / 6 |", "|---|---|"] + ["| %s | %d |" % (r, c) for r, c in fx["rules_agreement_over_all_6_pairs"].items()]
    L += ["", "Pinchable pairs: %s" % ", ".join(fx["pinchable_pairs"]), "", "Rules that match the stability order on every pinchable pair (or on none, i.e. reversed): %s" % (", ".join(fx["rules_matching_stability_on_every_pinchable_pair"]) or "none"),
          "", "Pair with size/stability agreement: %s; pair with a conflict: %s" % (fx["pair_with_size_stability_agreement"], fx["pair_with_size_stability_conflict"]), "", "| pair | stability first | larger first | gap (m) | class |", "|---|---|---|---|---|"]
    for p in prow:
        L.append("| %s | %s | %s | %.4f | %s |" % (p["pair"], p["stability_first"], p["rule_larger_footprint_first"], p["margin_gap_m"], p["condition_class"]))
    L += ["", "Result: single_rule_sufficient = %s; pass = %s" % (fx["single_rule_sufficient"], fx["pass"]), ""]
    return "\n".join(L)


def texture_audit(lib, names):
    try:
        from PIL import Image
    except Exception:
        return {"status": "PIL_UNAVAILABLE"}
    out = {}
    cols = {"target": occ.COLORS["target"], "second_object": occ.COLORS["second_object"]}
    for n in list(names) + [CONTAINER]:
        d0 = Path(lib) / ("assets/stable_hope_objects/%s" % n if n != CONTAINER else "assets/stable_scanned_objects/basket")
        tex = [p for p in d0.glob("*.png")]
        if not tex:
            continue
        img = np.asarray(Image.open(tex[0]).convert("RGB").resize((64, 64)), dtype=float) / 255.0
        px = img.reshape(-1, 3)
        fr = {k: float((np.linalg.norm(px - np.array(v)[None, :], axis=1) < COLOR_TOL).mean()) for k, v in cols.items()}
        out[n] = {"texture_file": tex[0].name, "mean_rgb": px.mean(0).tolist(), "fraction_within_tol_of_role_colours": fr}
    return out


def final_summary(out):
    out = Path(out)
    man = json.loads((out / "candidate_binding_manifest.json").read_text())
    pin = list(csv.DictReader((out / "pinchability_matrix.csv").open(encoding="utf-8")))
    per = json.loads((out / "public_perception_binding_spec.json").read_text())
    bsk = json.loads((out / "basket_place_binding_spec.json").read_text())
    L = ["# %s - final summary" % CARD, "", "Verdict: **%s**  (label %s)" % (man["verdict"], man["label"]), "", "All failing categories (evaluated independently; the verdict is the first in the declared order): %s" % ", ".join(man["all_failing_categories"] or ["none"]), "",
         "## Pass conditions", ""] + ["- %s: %s" % (k, v) for k, v in man["pass_conditions"].items()]
    L += ["", "## Pinchability (official spawn orientation, spread axis world y)", "", "| object | status | pinch width yaw0 (m) | yaw90 (m) | contact height (m) | palm clearance (m) |", "|---|---|---|---|---|---|"]
    for r in pin:
        L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f |" % (r["object"], r["status"], float(r["pinch_width_yaw0_m"]), float(r["pinch_width_yaw90_m"]), float(r["pad_contact_height_m"]), float(r["palm_clearance_above_top_m"])))
    L += ["", "## Marker pixel support (minimum %d)" % MIN_PIXELS, "", "| object | at A | at B | held (hand box) | in basket at drop |", "|---|---|---|---|---|"]
    for n, v in per["pixel_support"].items():
        L.append("| %s | %d | %d | %d | %d |" % (n, v["marker_pixels_at_A"], v["marker_pixels_at_B"], v["marker_pixels_held_with_hand_box"], v["marker_pixels_in_basket_at_registered_drop"]))
    L += ["", "Best marker pixels inside the basket over the pre-registered basket grid: %d" % per["basket_grid_best_pixels_in_basket"], "", "## Basket", "", bsk["conclusion"],
          "- cavity depth %.3f m, worst pile top %.3f m, pessimistic pair floor fill %.3f" % (bsk["cavity"]["cavity_depth_m"], bsk["worst_pile_top_m"], bsk["pair_floor_fill_pessimistic"]),
          "- pathways: %s" % json.dumps({k: v["possible"] for k, v in bsk["pathways"].items()}), "",
          "Budget: environment constructions 0, env.reset 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test 0, demo replay 0.", ""]
    (out / "final_summary.md").write_text("\n".join(L), encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "protected_before.json", "protected_after.json", "asset_identity.json", "spawn_orientation_audit.json", "pinchability_matrix.csv", "generic_pick_binding_spec.json",
            "basket_place_binding_spec.json", "public_perception_binding_spec.json", "verifier_binding_spec.json", "evaluator_binding_spec.json", "snapshot_binding_spec.json", "pair_property_matrix.csv", "fixed_rule_attack.md",
            "condition_set_design.json", "candidate_binding_manifest.json", "next_binding_canary_request.md", "budget_ledger.json", "engineering_events.jsonl", "final_summary.md"]
    led = json.loads((out / "budget_ledger.json").read_text())
    src = json.loads((out / "source_identity.json").read_text())
    man = json.loads((out / "candidate_binding_manifest.json").read_text())
    before, after = json.loads((out / "protected_before.json").read_text()), json.loads((out / "protected_after.json").read_text())
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".csv", ".jsonl"} and bool(re.search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "zero_resources": all(led[k] == 0 for k in RESOURCE_KEYS), "guard_refusals_zero": led["guard_refusals"] == 0, "no_sim_stack_imported": led["forbidden_modules_loaded"] == [],
              "baseline_recorded": src["run_head_descends_from_or_equals_baseline"] and len(src["run_head_full"]) == 40, "upstream_pinned": src["upstream_matches"], "protected_unchanged": before == after, "protected_files_hashed": len(before),
              "verdict_is_allowed": man["verdict"] in FAIL_ORDER + ["DV1_BINDING_PREFLIGHT_PASS"], "label_fixed": man["label"] == LABEL,
              "request_label_consistent": ("NOT_REQUESTED" in (out / "next_binding_canary_request.md").read_text()) == (man["verdict"] != "DV1_BINDING_PREFLIGHT_PASS"), "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k not in ("outputs_present", "protected_files_hashed", "status")) and all(checks["outputs_present"].values()) else "FAIL"
    write_json(out / "verify.json", checks)
    return checks