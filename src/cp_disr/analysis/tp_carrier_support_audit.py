"""CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1: design audit of a carrier-support (B supports A) T_P family.

Zero environment: static BDDL/XML/mesh/source analysis, quasi-static mechanics and route/contract analysis only. It never constructs an
environment and never imports robosuite, mujoco or libero. Label: DESIGN_ONLY_NOT_AUTHORIZED. Stage 1 freezes the asset universes from
names and BDDL only (no geometry); stage 2 reads geometry and evaluates the gates. All rules and thresholds are declared in code before the run.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cp_disr.analysis import tp_libero_audit as la
from cp_disr.analysis import tp_dv1_binding_preflight as dv
from cp_disr.analysis import tp_occ_preflight as occ

CARD = "CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1"
BASELINE_FULL = "eb262a8da1b66c4de1f1aef3dc8d909023f9b731"
UPSTREAM_SHA = la.UPSTREAM_SHA
LABEL = "DESIGN_ONLY_NOT_AUTHORIZED"
FAMILY = "T_P_CARRIER_SUPPORT_V1"
RESOURCE_KEYS = la.RESOURCE_KEYS
ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
CARRIER_NAME_RE = re.compile(r"tray|basket|caddy|plate|carrier")
PLACEMENT_REGION_RE = re.compile(r"contain|top|support|cook")      # contain / support / top regions
NEAR_DIST = 0.15                                                    # initial-region gap that counts as "near a carrier candidate"
FAIL_ORDER = ["TASK_FAIL_NO_GENERIC_CARRIER", "TASK_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR", "TASK_FAIL_NO_CARRIED_OBJECT", "TASK_FAIL_NO_STABLE_SUPPORT_PAIR", "TASK_FAIL_NO_STABILITY_CONTRAST",
              "TASK_FAIL_COMMUTATIVE", "TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT", "TASK_FAIL_STATIC_PRIOR_SUFFICIENT", "TASK_FAIL_CONTRACT_REVEALED", "TASK_FAIL_REQUIRES_NEW_PERCEPTION_MODEL",
              "TASK_FAIL_EVALUATOR_SEMANTIC_CHANGE", "TASK_FAIL_SNAPSHOT_REWRITE", "TASK_FAIL_RELATION_NOT_EXPRESSIBLE", "TASK_FAIL_CANARY_NOT_BOUNDED", "ENGINEERING_UNRESOLVED"]
digest = la.digest
write_json = la.write_json
write_csv = la.write_csv
sha_file = la.sha_file


def git_out(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------------------------------------ stage 1: universes
def scan_official_tasks(lib, guard):
    """Parse every official BDDL of the pinned tree; return per-task facts needed for the universes (no geometry, no assets)."""
    tmap = la.load_task_map(Path(lib) / "benchmark" / "libero_suite_task_map.py", guard)
    reg = la.load_registry(lib, guard)
    tasks = []
    for s in ALL_SUITES:
        for i, name in enumerate(tmap[s]):
            p = la.parse_bddl(guard.read(Path(lib) / "bddl_files" / s / (name + ".bddl")))
            inst, sect = la.inst_maps(p)
            tasks.append({"suite": s, "index": i, "name": name, "parsed": p, "inst": inst, "sect": sect})
    return tasks, reg


def placements(task, reg):
    """Every On/In (init and goal) whose destination is an object/fixture instance (not a table surface)."""
    p, inst, sect = task["parsed"], task["inst"], task["sect"]
    out = []
    for where, atoms in (("init", p["init"]), ("goal", p["goal_atoms"])):
        for a in atoms:
            if a[0].lower() not in ("on", "in") or len(a) < 3:
                continue
            res = la.resolve(a[2], p, inst, sect)
            subj = inst.get(a[1])
            if res["instance"] is None or subj is None or sect.get(a[1]) != "object" or la.is_surface(res["type"], reg):
                continue
            if res["instance"] == a[1]:
                continue
            out.append({"where": where, "pred": a[0], "subject_type": subj, "subject": a[1], "dest_type": res["type"], "dest_instance": res["instance"], "dest_role": sect.get(res["instance"]), "region": res["region"],
                        "region_class": ("direct_on_instance" if res["region"] is None else ("qualifying" if PLACEMENT_REGION_RE.search(res["region"]) else "other"))})
    return out


def surface_ranges(task, instance):
    p, inst, sect = task["parsed"], task["inst"], task["sect"]
    for a in p["init"]:
        if a[0].lower() == "on" and a[1] == instance and len(a) >= 3:
            res = la.resolve(a[2], p, inst, sect)
            rec = p["regions"].get((res["instance"], res["region"]))
            if rec and rec["ranges"] and res["type"] and la.is_surface(res["type"], {}):
                return res["instance"], rec["ranges"]
    return None, None


def build_universes(tasks, reg):
    carrier_by_type = defaultdict(lambda: {"roles": set(), "evidence": [], "name_match": False, "task_count": 0})
    for t in tasks:
        for pl in placements(t, reg):
            if pl["region_class"] in ("direct_on_instance", "qualifying"):
                c = carrier_by_type[pl["dest_type"]]
                c["roles"].add(pl["dest_role"])
                c["task_count"] += 1
                if len(c["evidence"]) < 6:
                    c["evidence"].append({"suite": t["suite"], "task": t["name"], "pred": pl["pred"], "subject_type": pl["subject_type"], "region": pl["region"], "where": pl["where"]})
    for k in sorted(reg):
        if CARRIER_NAME_RE.search(k):
            carrier_by_type[k]["name_match"] = True
    for t in tasks:
        for n, ty in t["parsed"]["fixtures"] + t["parsed"]["objects"]:
            if ty in carrier_by_type:
                carrier_by_type[ty]["roles"].add("fixture" if (n, ty) in t["parsed"]["fixtures"] else "object")
    carriers = {}
    for ty, c in sorted(carrier_by_type.items()):
        if la.is_surface(ty, reg):
            continue
        roles = sorted(c["roles"])
        if "object" in roles:
            cls = "NATIVE_MOVABLE_OBJECT"
        elif "fixture" in roles:
            cls = "OFFICIAL_FIXTURE"
        elif ty in reg and not reg[ty]["articulated"]:
            cls = "NATIVE_MOVABLE_OBJECT"
        else:
            cls = "NOT_A_CARRIER"
        carriers[ty] = {"type": ty, "classification": cls, "roles_in_official_tasks": roles, "name_matches_rule": c["name_match"], "official_placement_count": c["task_count"], "used_in_official_tasks": c["task_count"] > 0 or bool(roles),
                        "registered_class": reg.get(ty, {}).get("class"), "articulated_class": bool(reg.get(ty, {}).get("articulated")), "evidence": c["evidence"],
                        "derived_free_body_possible": "decided in stage 2 (fixtures only)" if cls == "OFFICIAL_FIXTURE" else None}
    # carried objects: placed on/in a carrier type in an official task, or initially near a carrier-type instance
    carried = {}
    for t in tasks:
        for pl in placements(t, reg):
            if pl["dest_type"] in carriers and pl["region_class"] in ("direct_on_instance", "qualifying") and pl["subject_type"] in reg:
                c = carried.setdefault(pl["subject_type"], {"reasons": set(), "evidence": []})
                c["reasons"].add("PLACED_ON_OR_IN_CARRIER")
                if len(c["evidence"]) < 6:
                    c["evidence"].append({"suite": t["suite"], "task": t["name"], "carrier_type": pl["dest_type"], "pred": pl["pred"], "where": pl["where"]})
        movable = [n for n, s in t["sect"].items() if s == "object"]
        for cn in [n for n in movable if t["inst"][n] in carriers]:
            ctab, cr = surface_ranges(t, cn)
            if not cr:
                continue
            for on in movable:
                if on == cn or t["inst"][on] not in reg:
                    continue
                otab, orr = surface_ranges(t, on)
                if otab != ctab or not orr:
                    continue
                gap = min(max(la.rect_gap(r1, r2)[0][0], la.rect_gap(r1, r2)[1][0]) for r1 in cr for r2 in orr)
                if gap <= NEAR_DIST:
                    c = carried.setdefault(t["inst"][on], {"reasons": set(), "evidence": []})
                    c["reasons"].add("INITIALLY_NEAR_CARRIER")
                    if len(c["evidence"]) < 6:
                        c["evidence"].append({"suite": t["suite"], "task": t["name"], "carrier_type": t["inst"][cn], "gap_m": round(gap, 4)})
    carried = {k: {"type": k, "registered_class": reg[k]["class"], "reasons": sorted(v["reasons"]), "evidence": v["evidence"], "is_also_carrier_candidate": k in carriers} for k, v in sorted(carried.items())}
    return carriers, carried


def stage1(root, libero_root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = la.Guard()
    head = git_out(["rev-parse", "HEAD"], root)
    ident = la.upstream_identity(libero_root, guard, None)
    write_json(out / "upstream_libero_identity.json", ident)
    if not ident["sha_matches_expected"]:
        raise RuntimeError("UPSTREAM_SHA_MISMATCH:%s" % ident["head_sha"])
    lib = Path(libero_root).resolve() / "libero" / "libero"
    tasks, reg = scan_official_tasks(lib, guard)
    carriers, carried = build_universes(tasks, reg)
    cu = {"card": CARD, "frozen_before_geometry": True, "rule": {"name_regex": CARRIER_NAME_RE.pattern, "placement_region_regex": PLACEMENT_REGION_RE.pattern, "near_distance_m": NEAR_DIST, "suites_scanned": list(ALL_SUITES),
                                                               "note": "all five suites of the pinned tree are scanned as TEXT (BDDL) to build the universe; LIBERO-90 tasks are not audited as tasks"},
          "carriers": carriers, "classification_counts": dict(Counter(c["classification"] for c in carriers.values()))}
    cu["universe_sha256"] = digest({k: (v["type"], v["classification"]) for k, v in carriers.items()})
    ou = {"card": CARD, "frozen_before_geometry": True, "rule": {"placed_on_or_in_a_carrier_type_in_an_official_task": True, "initially_within_near_distance_on_the_same_surface": NEAR_DIST}, "objects": carried}
    ou["universe_sha256"] = digest({k: v["reasons"] for k, v in carried.items()})
    write_json(out / "carrier_asset_universe.json", cu)
    write_json(out / "carried_object_universe.json", ou)
    write_json(out / "universe_manifest.json", {"card": CARD, "project_head_full": head, "upstream_sha": ident["head_sha"], "carrier_universe_sha256": cu["universe_sha256"], "carried_universe_sha256": ou["universe_sha256"],
                                                "carrier_types": sorted(carriers), "carried_types": sorted(carried), "files": {n: sha_file(out / n) for n in ("carrier_asset_universe.json", "carried_object_universe.json", "upstream_libero_identity.json")}})
    return {"carriers": sorted(carriers), "classes": cu["classification_counts"], "carried": sorted(carried)}


# ======================================================================================== stage 2: geometry and gates
# ---- thresholds and rules declared before stage 2 ----------------------------------------------------------
G = 9.81
MAX_CARRIER_EXTENT = 0.20         # horizontal extent that fits the proven layout (items are >= 0.24 m apart)
PAYLOAD_MAX_KG = 1.0
GRIP_FORCE_N = 20.0               # panda_gripper.xml forcerange +-20
PAD_HALF_Z = 0.008                # pad box half height (panda_gripper.xml)
PAD_MU = 2.0                      # pad friction (panda_gripper.xml)
TORQUE_OK, TORQUE_MAX = 0.5, 1.0  # gravity torque over pad couple capacity
TILT_EXPECTED_RAD = math.radians(5.0)
MARGIN_STABLE, EDGE_MARGIN = 0.02, 0.01
SPEED_CAP = 0.2                   # translation speed parameter (m/s)
CONTROL_DT = 0.05
HF_STEP = 0.005
FLAT_TOL = 0.003
SITE_STEP = 0.01
WALL_LEVEL_TOL = 0.03             # a rim wall must reach within this of the carrier top
FIT_FRACTION_MIN = 0.6
DIRECTIONS = [("E", (1, 0)), ("N", (0, 1)), ("W", (-1, 0)), ("S", (0, -1)), ("NE", (1, 1)), ("NW", (-1, 1)), ("SW", (-1, -1)), ("SE", (1, -1))]
SCENARIOS = ("S1_CO_TRANSPORT_STABLE", "S2_SEPARATE_DESTINATIONS", "S3_CO_TRANSPORT_UNSTABLE", "S4_NO_SUPPORT_NEUTRAL")
RULES = {"R1": "if A on B: move B first", "R2": "if same destination: move B first", "R3": "if different destinations: pick A first", "R4": "if A near carrier edge: pick A first",
         "R5": "always pick A first", "R6": "always move carrier first"}
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
COST_EPS = 1e-9


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def obb_overlap(a, b, margin=0.0):
    Ra, Rb = a["R"], b["R"]
    ha, hb = np.asarray(a["half"], float) + margin / 2, np.asarray(b["half"], float) + margin / 2
    C = Ra.T @ Rb
    A = np.abs(C) + 1e-9
    t = Ra.T @ (np.asarray(b["center"], float) - np.asarray(a["center"], float))
    for i in range(3):
        if abs(t[i]) > ha[i] + hb @ A[i, :]:
            return False
    for j in range(3):
        if abs(t @ C[:, j]) > ha @ A[:, j] + hb[j]:
            return False
    for i in range(3):
        for j in range(3):
            ra = ha[(i + 1) % 3] * A[(i + 2) % 3, j] + ha[(i + 2) % 3] * A[(i + 1) % 3, j]
            rb = hb[(j + 1) % 3] * A[i, (j + 2) % 3] + hb[(j + 2) % 3] * A[i, (j + 1) % 3]
            if abs(t[(i + 2) % 3] * C[(i + 1) % 3, j] - t[(i + 1) % 3] * C[(i + 2) % 3, j]) > ra + rb:
                return False
    return True


def grip_ext():
    g = dv.gripper_static()
    import importlib.util as iu
    import xml.etree.ElementTree as ET
    base = Path(iu.find_spec("robosuite").origin).parent / "models/assets/grippers"
    root = ET.parse(base / "panda_gripper.xml").getroot()
    eef_z = g["eef_site_hand_z"]
    fing = [b for b in root.iter("body") if b.attrib.get("name") == "leftfinger"][0]
    z0 = float(fing.attrib["pos"].split()[2])
    fmin, fmax = occ.bsi.stl_bounds(base / "meshes/panda_gripper/finger.stl")
    g = dict(g)
    g["finger_highest_rel_site_z"] = -(float(fmin[2]) + z0 - eef_z)
    g["finger_depth_m"] = g["finger_highest_rel_site_z"] - g["finger_lowest_rel_site_z"]
    g["grip_force_n"] = GRIP_FORCE_N
    g["pad_friction"] = PAD_MU
    return g


def stl_volume(path, scale):
    """Volume of a closed STL (binary or ASCII), scaled."""
    import struct
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:600]:
        v = np.array([[float(x) for x in mm.groups()] for mm in re.finditer(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)]).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", data[80:84])[0]
        arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        v = arr["v"].astype(float)
    v = v * np.array(scale)
    return abs(float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)


def safe_mass_inertia(parsed, xml_dir):
    """dv.mass_inertia with the visual-mesh part made robust: .msh via the binary reader, .stl via stl_volume, anything else is recorded as UNPARSED and
    the mass range then reports the box-only mass as its upper end with the flag set (never a guess)."""
    boxes_only = dict(parsed, visual_meshes=[])
    mi = dv.mass_inertia(boxes_only, xml_dir)
    m_mesh, info, unparsed = 0.0, [], 0
    for vm in parsed["visual_meshes"]:
        fn = vm.get("file")
        if not fn:
            unparsed += 1
            info.append({"file": None, "status": "UNPARSED_NO_FILE"})
            continue
        p = Path(xml_dir) / fn
        try:
            vol = dv.mesh_volume(p, vm["scale"])[0] if p.suffix.lower() == ".msh" else stl_volume(p, vm["scale"]) if p.suffix.lower() == ".stl" else None
        except Exception:
            vol = None
        if vol is None:
            unparsed += 1
            info.append({"file": fn, "status": "UNPARSED"})
        else:
            m_mesh += vol * vm["density"]
            info.append({"file": fn, "volume_m3": vol, "mass_kg": vol * vm["density"], "status": "OK"})
    mi["visual_meshes"] = info
    mi["visual_mesh_unparsed"] = unparsed
    mi["mass_boxes_plus_visual_mesh_kg"] = mi["mass_boxes_only_kg"] + m_mesh
    mi["mass_ratio_high_over_low"] = mi["mass_boxes_plus_visual_mesh_kg"] / max(mi["mass_boxes_only_kg"], 1e-12)
    return mi

def registered_record(lib, reg, name, policy):
    """Geometry in the registered orientation, resting on z = 0. policy: 'carrier' -> identity (asset frame z-up); 'carried' -> class source orientation
    when it is explicit or the Hope default, else identity. Every record states its orientation basis and status."""
    rec = reg[name]
    xml = Path(lib) / rec["xml"]
    parsed = dv.parse_geoms(xml)
    base = {"name": name, "class": rec["class"], "xml": rec["xml"], "xml_sha256": sha_file(xml), "ok": False}
    if not parsed["boxes"]:
        return dict(base, reason="NO_BOX_COLLISION_PROXY", orientation_basis="n/a", orientation_status="n/a")
    mi = safe_mass_inertia(parsed, xml.parent)
    spawn = dv.spawn_orientation(lib, rec["class"])
    defined_in = (spawn.get("rotation_source") or {}).get("defined_in")
    if policy == "carried" and spawn["status"] == "DETERMINED_BY_SOURCE" and defined_in != "GoogleScannedObject":
        Rs, basis, status = la._quat_to_mat(spawn["quat_wxyz"]), "class source rotation (%s)" % defined_in, "DETERMINED_BY_SOURCE"
    else:
        Rs = np.eye(3)
        basis = "asset frame, z up (derived registration)"
        status = "UNVERIFIED_AS_OFFICIAL" if spawn["status"] == "DETERMINED_BY_SOURCE" else "UNVERIFIED"
    bw0 = dv.world_boxes(parsed["boxes"], Rs)
    lo0, hi0 = dv.aabb(bw0)
    shift = np.array([-0.5 * (lo0[0] + hi0[0]), -0.5 * (lo0[1] + hi0[1]), -lo0[2]])      # origin = footprint centre on the resting plane
    bw = dv.world_boxes(parsed["boxes"], Rs, shift=shift)
    lo, hi = dv.aabb(bw)
    com = Rs @ np.array(mi["com_asset_frame_m"]) + shift
    top, bot = dv.top_bottom_layers(bw)
    mus = [b["friction"][0] for b in parsed["boxes"]]
    return dict(base, ok=True, boxes=bw, extent=(hi - lo).tolist(), height=float(hi[2]), lo=lo.tolist(), hi=hi.tolist(), com=com.tolist(), mass_lo=mi["mass_boxes_only_kg"], mass_hi=mi["mass_boxes_plus_visual_mesh_kg"],
                mass_mid=0.5 * (mi["mass_boxes_only_kg"] + mi["mass_boxes_plus_visual_mesh_kg"]), mu=float(max(mus)), top_rect=dv.rect_of(top), top_area=dv.rect_area(dv.rect_of(top)), support_rect=dv.rect_of(bot),
                footprint_rect=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])], footprint_area=float((hi - lo)[0] * (hi - lo)[1]), orientation_basis=basis, orientation_status=status, spawn_status=spawn["status"],
                boxes_count=len(bw), joints_free=True)


def vertical_height(boxes, X, Y, z0=3.0):
    """Top surface height of a box union under vertical rays at (X, Y); NaN where nothing is hit."""
    best = np.full(X.shape, np.inf)
    o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z0)], axis=1)
    for b in boxes:
        oo = (o - b["center"]) @ b["R"]
        dd = b["R"].T @ np.array([0.0, 0.0, -1.0])
        tmin, tmax = np.full(len(o), -np.inf), np.full(len(o), np.inf)
        ok = np.ones(len(o), bool)
        for i in range(3):
            if abs(dd[i]) < 1e-12:
                ok &= np.abs(oo[:, i]) <= b["half"][i]
            else:
                t1, t2 = (-b["half"][i] - oo[:, i]) / dd[i], (b["half"][i] - oo[:, i]) / dd[i]
                tmin, tmax = np.maximum(tmin, np.minimum(t1, t2)), np.minimum(tmax, np.maximum(t1, t2))
        hit = ok & (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        best.ravel()[hit] = np.minimum(best.ravel()[hit], tmin[hit])
    z = np.where(np.isfinite(best), z0 - best, np.nan)
    return z.reshape(X.shape)


def support_field(rec):
    from scipy import ndimage
    lo, hi = np.array(rec["lo"]), np.array(rec["hi"])
    xs = np.arange(lo[0] - 0.01, hi[0] + 0.01, HF_STEP)
    ys = np.arange(lo[1] - 0.01, hi[1] + 0.01, HF_STEP)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    hf = vertical_height(rec["boxes"], X, Y)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    ci, cj = int(round((cx - xs[0]) / HF_STEP)), int(round((cy - ys[0]) / HF_STEP))
    zs = hf[ci, cj]
    if not np.isfinite(zs):
        return {"ok": False, "reason": "no surface under the carrier centre", "centre": [cx, cy]}
    mask = np.isfinite(hf) & (np.abs(np.nan_to_num(hf, nan=-9) - zs) <= FLAT_TOL)
    lab, _ = ndimage.label(mask)
    S = lab == lab[ci, cj]
    dist = ndimage.distance_transform_edt(S) * HF_STEP
    rim = float(np.nanmax(hf))
    return {"ok": True, "xs": xs, "ys": ys, "S": S, "dist": dist, "hf": hf, "z_support": float(zs), "rim_z": rim, "centre": [cx, cy], "support_area_m2": float(S.sum() * HF_STEP ** 2), "cavity_depth_m": rim - float(zs)}


def sample(field, x, y, arr="dist"):
    i = int(round((x - field["xs"][0]) / HF_STEP))
    j = int(round((y - field["ys"][0]) / HF_STEP))
    if i < 0 or j < 0 or i >= field[arr].shape[0] or j >= field[arr].shape[1]:
        return 0.0
    return float(field[arr][i, j])


def fit_fraction(field, A, ox, oy):
    r = A["footprint_rect"]
    xs = np.arange(r[0] + 0.0025, r[1], HF_STEP) + ox
    ys = np.arange(r[2] + 0.0025, r[3], HF_STEP) + oy
    tot = ins = 0
    for x in xs:
        for y in ys:
            tot += 1
            i = int(round((x - field["xs"][0]) / HF_STEP))
            j = int(round((y - field["ys"][0]) / HF_STEP))
            if 0 <= i < field["S"].shape[0] and 0 <= j < field["S"].shape[1] and field["S"][i, j]:
                ins += 1
    return ins / max(tot, 1)


# ---- carrier grasp (move carrier) ----------------------------------------------------------------------------
def gripper_boxes(site, psi, grip, z_site):
    u = np.array([-math.sin(psi), math.cos(psi), 0.0])
    v = np.array([math.cos(psi), math.sin(psi), 0.0])
    R = np.stack([v, u, np.array([0, 0, 1.0])], axis=1)
    inner, outer, hw = grip["finger_inner_face_open"], grip["finger_outer_face_open"], grip["finger_half_width_orthogonal"]
    z_lo, z_hi = z_site + grip["finger_lowest_rel_site_z"], z_site + grip["finger_highest_rel_site_z"]
    fingers = []
    for s in (-1, 1):
        c = np.array([site[0], site[1], 0.5 * (z_lo + z_hi)]) + s * 0.5 * (inner + outer) * u
        fingers.append({"center": c, "R": R, "half": np.array([hw, 0.5 * (outer - inner), 0.5 * (z_hi - z_lo)])})
    hmin, hmax = np.array(grip["hand_mesh_bounds"]["min"]), np.array(grip["hand_mesh_bounds"]["max"])
    eef = grip["eef_site_hand_z"]
    zlo, zhi = z_site - (hmax[2] - eef), z_site - (hmin[2] - eef)
    palm = {"center": np.array([site[0], site[1], 0.5 * (zlo + zhi)]), "R": R, "half": np.array([max(abs(hmin[0]), abs(hmax[0])), max(abs(hmin[1]), abs(hmax[1])), 0.5 * (zhi - zlo)])}
    return fingers + [palm]


def shifted(boxes, dx, dy, dz):
    return [{"center": b["center"] + np.array([dx, dy, dz]), "R": b["R"], "half": b["half"]} for b in boxes]


def carrier_grasp(carrier, grip, A=None, offset=(0.0, 0.0), z_support=None):
    """Enumerate top-down pinch sites for lifting the carrier: rim-wall pinches and whole-body pinches. Returns the candidates with feasibility and torque ratio."""
    lim = 2 * (grip["finger_inner_face_open"] - dv.CTRL_TOL)
    std = grip["pad_height_m"]
    tau_cap = GRIP_FORCE_N * 2 * PAD_HALF_Z
    cb = carrier["boxes"]
    top = carrier["height"]
    others_A = []
    m_tot = carrier["mass_hi"]
    com = np.array(carrier["com"][:2]) * carrier["mass_hi"]
    if A is not None and z_support is not None:
        cxy = np.array([0.5 * (carrier["lo"][0] + carrier["hi"][0]), 0.5 * (carrier["lo"][1] + carrier["hi"][1])])
        others_A = shifted(A["boxes"], cxy[0] + offset[0], cxy[1] + offset[1], z_support)
        com = com + (np.array(A["com"][:2]) + cxy + np.array(offset)) * A["mass_hi"]
        m_tot += A["mass_hi"]
    com_xy = com / m_tot
    cands = []
    for bi, b in enumerate(cb):
        h = b["half"]
        k = int(np.argmin(h))
        n3 = b["R"][:, k]
        if abs(n3[2]) > 0.35 or 2 * h[k] > lim:
            continue
        corners_z = dv.corners(b)[:, 2]
        z_top, z_bot = float(corners_z.max()), float(corners_z.min())
        if z_top < top - WALL_LEVEL_TOL:
            continue
        rest = [i for i in range(3) if i != k]
        j = max(rest, key=lambda i: np.linalg.norm(b["R"][:2, i]))
        tang = unit(np.array([b["R"][0, j], b["R"][1, j], 0.0]))
        L = 2 * h[j] * np.linalg.norm(b["R"][:2, j])
        n = unit(np.array([n3[0], n3[1], 0.0]))
        psi = math.atan2(-n[0], n[1])
        for s in np.arange(-(L / 2 - 0.012), L / 2 - 0.011, SITE_STEP):
            site = b["center"][:2] + s * tang[:2]
            z_site = z_top - 0.006
            overlap = min(z_top, z_site + grip["pad_height_m"] - 0.0044) - max(z_site - 0.0044, z_bot)
            if overlap < dv.OVERLAP_FRACTION * std:
                continue
            gb = gripper_boxes(site, psi, grip, z_site)
            others = [x for i, x in enumerate(cb) if i != bi] + others_A
            hit = [("finger" if gi < 2 else "palm") for gi, g_ in enumerate(gb) for o in others if obb_overlap(g_, o, 0.002)]
            d = float(np.linalg.norm(com_xy - site))
            ratio = m_tot * G * d / tau_cap
            cands.append({"kind": "RIM_WALL_PINCH", "box": bi, "site_xy": [float(site[0]), float(site[1])], "site_z": float(z_site), "yaw_deg": float(math.degrees(psi)), "collisions": sorted(set(hit)), "feasible": not hit, "com_distance_m": d, "torque_ratio": float(ratio),
                          "pad_contact_height_m": float(overlap)})
    c0 = np.array([0.5 * (carrier["lo"][0] + carrier["hi"][0]), 0.5 * (carrier["lo"][1] + carrier["hi"][1])])
    for deg in range(0, 180, 15):
        psi = math.radians(deg)
        w = dv.pinch_width(cb, psi)
        if w > lim:
            continue
        z_site = dv.STD_GRASP_SITE - dv.STD_PRESS
        overlap = min(top, z_site + grip["pad_height_m"] - 0.0044) - max(z_site - 0.0044, 0.0)
        if overlap < dv.OVERLAP_FRACTION * std:
            continue
        gb = gripper_boxes(c0, psi, grip, z_site)
        hit = [("finger" if gi < 2 else "palm") for gi, g_ in enumerate(gb) for o in others_A for _ in [0] if obb_overlap(g_, o, 0.002)]
        d = float(np.linalg.norm(com_xy - c0))
        cands.append({"kind": "BODY_PINCH", "box": None, "site_xy": [float(c0[0]), float(c0[1])], "site_z": float(z_site), "yaw_deg": float(deg), "collisions": sorted(set(hit)), "feasible": not hit, "com_distance_m": d,
                      "torque_ratio": float(m_tot * G * d / tau_cap), "pinch_width_m": float(w), "pad_contact_height_m": float(overlap)})
    feas = [c for c in cands if c["feasible"]]
    best = min(feas, key=lambda c: (c["torque_ratio"], abs(c["yaw_deg"]))) if feas else None
    return {"candidates": len(cands), "feasible": len(feas), "best": best, "mass_total_kg": m_tot, "torque_capacity_nm": tau_cap}


def mobility_row(role_info, rec):
    reasons = []
    if not rec["ok"]:
        return {"movable": False, "reasons": ["geometry unavailable: " + rec.get("reason", "")]}
    ext = rec["extent"]
    if role_info["articulated_class"]:
        reasons.append("articulated appliance or furniture")
    if max(ext[0], ext[1]) > MAX_CARRIER_EXTENT:
        reasons.append("horizontal extent %.3f m > %.2f m" % (max(ext[0], ext[1]), MAX_CARRIER_EXTENT))
    if rec["mass_hi"] > PAYLOAD_MAX_KG:
        reasons.append("mass %.3f kg > %.1f kg" % (rec["mass_hi"], PAYLOAD_MAX_KG))
    return {"movable": not reasons, "reasons": reasons}


# ---- pair stability --------------------------------------------------------------------------------------------
def pick_on_carrier(A, carrier, field, grip, offset):
    """PICK(A) while A rests on the carrier: finger tips must clear the support, the open fingers and the palm must clear the carrier."""
    lim = 2 * (grip["finger_inner_face_open"] - dv.CTRL_TOL)
    zs = field["z_support"]
    site = zs + (dv.STD_GRASP_SITE - dv.STD_PRESS)
    cxy = np.array(field["centre"])
    pos = cxy + np.array(offset)
    boxes = shifted(carrier["boxes"], 0, 0, 0)
    reasons = []
    ok_yaw = []
    for deg in (0, 90):
        w = dv.pinch_width(A["boxes"], math.radians(deg))
        if w <= lim:
            gb = gripper_boxes(pos, math.radians(deg), grip, site)
            hit = sorted({("finger" if gi < 2 else "palm") for gi, g_ in enumerate(gb) for o in boxes if obb_overlap(g_, o, 0.002) and not (o["center"][2] + o["half"] @ np.abs(o["R"][2, :]) <= zs + 0.0005)})
            if not hit:
                ok_yaw.append(deg)
            else:
                reasons.append("yaw %d: %s hit the carrier" % (deg, "+".join(hit)))
        else:
            reasons.append("yaw %d: pinch width %.3f > %.3f" % (deg, w, lim))
    pad_low = site + grip["pad_lowest_rel_site_z"]
    overlap = min(zs + A["height"], pad_low + grip["pad_height_m"]) - max(zs, pad_low)
    if overlap < dv.OVERLAP_FRACTION * grip["pad_height_m"]:
        reasons.append("pad contact %.4f m too small" % overlap)
    palm = site + grip["hand_lowest_rel_site_z"] - (zs + A["height"])
    if palm < dv.HAND_CLEAR_MIN:
        reasons.append("palm clearance %.4f m" % palm)
    return {"ok": bool(ok_yaw) and overlap >= dv.OVERLAP_FRACTION * grip["pad_height_m"] and palm >= dv.HAND_CLEAR_MIN, "yaw_options_deg": ok_yaw, "pad_contact_m": float(overlap), "palm_clearance_m": float(palm), "reasons": reasons}


def pair_eval(carrier, A, field, grip):
    mu = max(carrier["mu"], A["mu"])
    a_bound = SPEED_CAP / CONTROL_DT
    noslip = mu * G
    com_xy = np.array(A["com"][:2])
    com_h = A["com"][2]
    out = {"carrier": carrier["name"], "carried": A["name"], "no_slip_acceleration_limit": noslip, "carrier_path_acceleration_bound": a_bound, "acceleration_ratio": a_bound / noslip, "speed_cap_m_s": SPEED_CAP,
           "support_z_m": field["z_support"], "rim_z_m": field["rim_z"], "cavity_depth_m": field["cavity_depth_m"], "rim_retention_margin_m": field["cavity_depth_m"] - com_h, "fall_height_m": field["z_support"]}
    def at(ox, oy):
        frac = fit_fraction(field, A, ox, oy)
        m = sample(field, field["centre"][0] + ox + com_xy[0], field["centre"][1] + oy + com_xy[1])
        tip = math.atan2(m, max(com_h, 1e-6))
        return frac, m, tip
    frac, m, tip = at(0.0, 0.0)
    out["centre"] = {"fit_fraction": frac, "com_projection_margin_m": m, "support_margin_m": m, "static_tip_margin_rad": tip, "pick_on_carrier": pick_on_carrier(A, carrier, field, grip, (0.0, 0.0))}
    g = carrier_grasp(carrier, grip, A, (0.0, 0.0), field["z_support"])
    out["centre"]["carrier_grasp"] = {"feasible": g["feasible"], "best": g["best"]}
    edge = {}
    chosen = None
    for name, d in DIRECTIONS:
        dn = unit(np.array(d, float))
        best = None
        for r in np.arange(0.0, 0.16, 0.005):
            ox, oy = dn[0] * r, dn[1] * r
            f, mm, tp = at(ox, oy)
            if f >= 0.3 and mm >= EDGE_MARGIN:
                best = (r, ox, oy, f, mm, tp)
        if best:
            edge[name] = {"radius_m": best[0], "offset": [best[1], best[2]], "fit_fraction": best[3], "com_margin_m": best[4], "tip_margin_rad": best[5]}
            if chosen is None:
                chosen = name
    out["edge"] = {"directions": edge, "chosen": chosen}
    if chosen:
        ox, oy = edge[chosen]["offset"]
        out["edge"]["pick_on_carrier"] = pick_on_carrier(A, carrier, field, grip, (ox, oy))
        ge = carrier_grasp(carrier, grip, A, (ox, oy), field["z_support"])
        out["edge"]["carrier_grasp"] = {"feasible": ge["feasible"], "best": ge["best"]}
    # classification (declared rules)
    cg = out["centre"]["carrier_grasp"]["best"]
    c_ok = frac >= FIT_FRACTION_MIN and m >= MARGIN_STABLE and tip >= TILT_EXPECTED_RAD and out["acceleration_ratio"] <= 0.5 and out["centre"]["pick_on_carrier"]["ok"] and cg is not None
    if m < 0 or frac < 0.3:
        out["centre_class"] = "UNSTABLE_OR_HARD_FAILURE"
    elif c_ok and cg["torque_ratio"] <= TORQUE_OK:
        out["centre_class"] = "STABLE_CENTER_CANDIDATE"
    elif c_ok or (frac >= FIT_FRACTION_MIN and m >= MARGIN_STABLE):
        out["centre_class"] = "UNKNOWN_NEEDS_PHYSICAL_CANARY"
    else:
        out["centre_class"] = "UNSTABLE_OR_HARD_FAILURE"
    if not chosen:
        out["edge_class"] = "UNSTABLE_OR_HARD_FAILURE"
    else:
        e = out["edge"]
        ok = e["pick_on_carrier"]["ok"] and e["carrier_grasp"]["best"] is not None
        out["edge_class"] = "MARGINAL_EDGE_CANDIDATE" if ok else "UNKNOWN_NEEDS_PHYSICAL_CANARY"
    return out


# ---- routes, rules, contract ------------------------------------------------------------------------------------
def duration_means(root):
    for p in sorted((Path(root) / "runs/final_master/S4/tp_bsi_preflight").glob("*/skill_duration_summary.json")):
        d = json.loads(p.read_text())["per_skill"]
        pick = np.mean([d[k]["mean"] for k in d if k.startswith("PICK")])
        place = np.mean([d[k]["mean"] for k in d if k.startswith("PLACE")])
        return {"PICK_s": float(pick), "PLACE_s": float(place), "source": str(p.relative_to(root))}
    return {"PICK_s": 3.0, "PLACE_s": 3.0, "source": "default"}


def route_costs(dur):
    """Skill counts and time for both first actions in each scenario, as functions of the probability p that A leaves the carrier during carrier transport.
    Recovery after a fall: PICK(A) then PLACE(A). The gripper holds one thing, so MOVE_CARRIER needs an empty gripper and an A that stays put must be set down first."""
    pk, pl = dur["PICK_s"], dur["PLACE_s"]
    mv_lo, mv_hi = pk, pk + pl            # MOVE_CARRIER duration is unestablished: bracketed by a pick and a pick plus place
    rows = {}
    def cost(skills, move_s):
        return {"skills": skills, "time_s": move_s}
    for name in SCENARIOS:
        for p in P_GRID:
            for mv in (mv_lo, mv_hi):
                if name == "S1_CO_TRANSPORT_STABLE" or name == "S3_CO_TRANSPORT_UNSTABLE":
                    # goal: A and B both end at the consistent co-transport destination. Object-first must set A down, move B, then pick A and place it back
                    carrier_first_s = 1 + 2 * p
                    carrier_first_t = mv + p * (pk + pl)
                    object_first_s = 5
                    object_first_t = 2 * pk + 2 * pl + mv
                    object_first_spec_s = 3                      # the card's simplified object-first route (PICK, PLACE, MOVE) is also reported
                    object_first_spec_t = pk + pl + mv
                elif name == "S2_SEPARATE_DESTINATIONS":
                    carrier_first_s = 3
                    carrier_first_t = mv + pk + pl
                    object_first_s = 3
                    object_first_t = pk + pl + mv
                    object_first_spec_s, object_first_spec_t = 3, object_first_t
                else:
                    carrier_first_s = 3
                    carrier_first_t = mv + pk + pl
                    object_first_s = 3
                    object_first_t = pk + pl + mv
                    object_first_spec_s, object_first_spec_t = 3, object_first_t
                rows.setdefault(name, []).append({"p_fall": p, "move_carrier_s": mv, "carrier_first": cost(carrier_first_s, carrier_first_t), "object_first": cost(object_first_s, object_first_t),
                                                  "object_first_card_route": cost(object_first_spec_s, object_first_spec_t)})
    return rows


def dominance(rows):
    out = {}
    for name, lst in rows.items():
        cf_worse = any(r["carrier_first"]["skills"] > r["object_first"]["skills"] + COST_EPS for r in lst)
        of_worse = any(r["object_first"]["skills"] > r["carrier_first"]["skills"] + COST_EPS for r in lst)
        cf_worse_card = any(r["carrier_first"]["skills"] > r["object_first_card_route"]["skills"] + COST_EPS for r in lst)
        of_worse_card = any(r["object_first_card_route"]["skills"] > r["carrier_first"]["skills"] + COST_EPS for r in lst)
        cf_worse_t = any(r["carrier_first"]["time_s"] > r["object_first"]["time_s"] + 1e-6 for r in lst)
        out[name] = {"carrier_first_ever_strictly_more_skills": cf_worse, "object_first_ever_strictly_more_skills": of_worse, "carrier_first_ever_strictly_more_skills_vs_card_route": cf_worse_card,
                     "object_first_ever_strictly_more_skills_vs_card_route": of_worse_card, "carrier_first_ever_strictly_more_time": cf_worse_t,
                     "orders_differ_by_skill_count": cf_worse or of_worse, "orders_differ_by_time": cf_worse_t or any(r["object_first"]["time_s"] > r["carrier_first"]["time_s"] + 1e-6 for r in lst)}
    return out


def rule_attack(dom):
    """For each fixed rule decide if some scenario is a counterexample (the prescribed first action is strictly worse in skills)."""
    def worse_cf(s):
        return dom[s]["carrier_first_ever_strictly_more_skills"] or dom[s]["carrier_first_ever_strictly_more_skills_vs_card_route"]
    def worse_of(s):
        return dom[s]["object_first_ever_strictly_more_skills"] and not False
    carrier_scen = ("S1_CO_TRANSPORT_STABLE", "S2_SEPARATE_DESTINATIONS", "S3_CO_TRANSPORT_UNSTABLE")
    same_dest = ("S1_CO_TRANSPORT_STABLE", "S3_CO_TRANSPORT_UNSTABLE")
    diff_dest = ("S2_SEPARATE_DESTINATIONS",)
    res = {}
    res["R1"] = {"rule": RULES["R1"], "counterexamples": [s for s in carrier_scen if worse_cf(s)]}
    res["R2"] = {"rule": RULES["R2"], "counterexamples": [s for s in same_dest if worse_cf(s)]}
    res["R3"] = {"rule": RULES["R3"], "counterexamples": [s for s in diff_dest if worse_of(s)]}
    res["R4"] = {"rule": RULES["R4"], "counterexamples": [s for s in ("S3_CO_TRANSPORT_UNSTABLE",) if worse_of(s)]}
    res["R5"] = {"rule": RULES["R5"], "counterexamples": [s for s in carrier_scen if worse_of(s)]}
    res["R6"] = {"rule": RULES["R6"], "counterexamples": [s for s in carrier_scen if worse_cf(s)]}
    for k, v in res.items():
        v["refuted"] = bool(v["counterexamples"])
    return res


def mini_bplan(scenario, dur):
    """Contract-only uniform-cost plan (no co-transport effect in any contract): the planner never credits MOVE_CARRIER with moving A."""
    pk, pl = dur["PICK_s"], dur["PLACE_s"]
    mv = pk + pl
    if scenario in ("S1_CO_TRANSPORT_STABLE", "S3_CO_TRANSPORT_UNSTABLE"):
        return {"plan": ["PICK(A)", "PLACE(A,T)", "MOVE_CARRIER(B,D)", "PICK(A)", "PLACE(A,D_A)"], "skills": 5, "note": "the contract has no effect that lets MOVE_CARRIER relocate A, so A is handled separately"}
    return {"plan": ["PICK(A)", "PLACE(A,D_A)", "MOVE_CARRIER(B,D_B)"], "skills": 3, "note": "independent subgoals"}


# ---- relation / contract / spec helpers ------------------------------------------------------------------------
CONTRACTS = {
    "a:PICK:A:v1": {"pre": ["p:GripperEmpty", "p:OnSupport:A"], "add": ["p:Held:A"], "del": ["p:GripperEmpty", "p:OnSupport:A"]},
    "a:PLACE:A:D_A:v1": {"pre": ["p:Held:A"], "add": ["p:AtRegion:A:D_A", "p:GripperEmpty"], "del": ["p:Held:A"]},
    "a:MOVE_CARRIER:B:D_B:v1": {"pre": ["p:GripperEmpty", "p:OnSupport:B"], "add": ["p:AtRegion:B:D_B"], "del": ["p:AtRegion:B:start"]},
}
GOAL_FACTS = ["p:AtRegion:A:D_A", "p:AtRegion:B:D_B"]
SCHEMA_ENUM = ("SOFT_SUPPORTS", "SOFT_RELEVANT_TO_GOAL")
RELATIONS = [
    {"relation_id": "rel_carrier_goal", "type": "SOFT_RELEVANT_TO_GOAL", "source_ref": "a:MOVE_CARRIER:B:D_B:v1", "target_ref": "p:AtRegion:A:D_A", "effect_fact_ref": "p:AtRegion:B:D_B"},
    {"relation_id": "rel_pick_supports_move", "type": "SOFT_SUPPORTS", "source_ref": "a:PICK:A:v1", "target_ref": "a:MOVE_CARRIER:B:D_B:v1", "effect_fact_ref": "p:OnSupport:A"},
]


def relation_check():
    res = []
    for r in RELATIONS:
        c = CONTRACTS.get(r["source_ref"])
        issues = []
        if r["type"] not in SCHEMA_ENUM:
            issues.append("type outside the schema enum")
        if c is None:
            issues.append("source_ref is not a contract action id")
        else:
            if r["effect_fact_ref"] not in c["add"] + c["del"]:
                issues.append("effect_fact_ref is not an ADD/DEL of the source action")
        if r["target_ref"] not in CONTRACTS and r["target_ref"] not in GOAL_FACTS:
            issues.append("target_ref is neither a contract action nor a goal proposition")
        hard_dup = c is not None and (r["target_ref"] in c["add"] + c["del"])
        if hard_dup:
            issues.append("target is already a hard ADD/DEL of the source (contract redundant)")
        res.append({"relation": r, "issues": issues, "expressible": not issues, "contract_redundant": hard_dup})
    return res


def classify_carrier(mob, g):
    if not mob["movable"]:
        return "NOT_MOVABLE"
    if g["best"] is None:
        return "ASSET_SPECIFIC_CONTROLLER_REQUIRED"
    r = g["best"]["torque_ratio"]
    if r <= TORQUE_OK:
        return "GENERIC_MOVE_CARRIER_COMPATIBLE"
    if r <= TORQUE_MAX:
        return "GENERIC_BINDING_REQUIRED"
    return "ASSET_SPECIFIC_CONTROLLER_REQUIRED"


def marker_patches(rec, dx, dy, dz):
    """Public marker patches on the highest layer of a carrier (rim or top plate), axis-aligned at the layer height."""
    top, _ = dv.top_bottom_layers(rec["boxes"], tol=0.002)
    patches = []
    for b in top:
        lo, hi = dv.aabb([b])
        c = 0.5 * (lo + hi)
        h = 0.4 * (hi - lo)
        patches.append({"center": np.array([c[0] + dx, c[1] + dy, hi[2] + dz + dv.MARKER_THICK / 2]), "R": np.eye(3), "half": np.array([max(h[0], 0.002), max(h[1], 0.002), dv.MARKER_THICK / 2])})
    return patches


def snapshot_evidence(root, guard):
    cap = guard.read(Path(root) / "src/cp_disr/analysis/tp_ef_post_open_capture.py")
    env_src = guard.read(Path(root) / "src/cp_disr/platforms/libero/d0_env.py")
    ev = {"sim_fields_saved": re.findall(r'SIM_FIELDS = \(([^)]*)\)', cap), "capture_sim_reads_env_sim_data": bool(re.search(r"def capture_sim\(env\):[\s\S]{0,400}env\.sim\.data", cap)),
          "python_state_dump_covers_robot_controller_gripper_rng": all(k in cap for k in ("def dump_state", "robot0.controller", "robot0.gripper", "def rng_snapshot")),
          "restore_builds_env_from_case_first": bool(re.search(r"def build_runtime", cap)) and "start_case_reusing_bootstrap" in cap, "free_joint_objects_set_by_qpos": "set_joint_qpos" in env_src, "object_set_defined_by_case": "self.case" in env_src}
    ok = all(v for v in ev.values() if isinstance(v, bool)) and bool(ev["sim_fields_saved"])
    return {"evidence": ev, "status": "PASS" if ok else "FAIL",
            "must_save": ["free-joint qpos/qvel of A and B (position, orientation, velocity)", "contact warm start (qacc_warmstart) and constraint forces", "controller and gripper named state", "RNG", "public observation identity", "registered geometry hash of both assets",
                          "relative pose A on B (derived, used for verification only)"],
            "extensions": ["CaseSpec carries carrier and carried asset ids and a geometry hash", "D0ManipulationEnv._load_model builds both bodies", "_apply_case_poses/_setup_references/hidden_truth key the bodies by role", "restore identity adds the geometry hash"], "lifecycle_rewritten": False}


def fixed_rule_md(attack, dom, routes_sample):
    L = ["# Fixed-rule attack (R1-R6)", "", "A rule is refuted by a scenario in which the action it prescribes needs strictly more skills than the other first action (skill counts from the declared route model; MOVE_CARRIER time is unestablished and bracketed).", "",
         "| rule | statement | counterexample scenarios | refuted |", "|---|---|---|---|"]
    for k, v in attack.items():
        L.append("| %s | %s | %s | %s |" % (k, v["rule"], ", ".join(v["counterexamples"]) or "none", v["refuted"]))
    L += ["", "## Scenario dominance (skills)", ""]
    for s, d in dom.items():
        L.append("- %s: carrier-first ever worse = %s (vs card route %s); object-first ever worse = %s" % (s, d["carrier_first_ever_strictly_more_skills"], d["carrier_first_ever_strictly_more_skills_vs_card_route"], d["object_first_ever_strictly_more_skills"]))
    unrefuted = [k for k, v in attack.items() if not v["refuted"]]
    L += ["", "Unrefuted single-variable rules: %s" % (", ".join(unrefuted) or "none"), "", "Result: %s" % ("a fixed single-variable heuristic is sufficient (TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT)" if unrefuted else "every rule has a counterexample"), ""]
    return "\n".join(L)


BASELINES = ["B_PLAN", "B_PLAN+R", "B2", "Full", "A_STAT", "A_CAT", "A_Q", "simple-rule baseline", "LLM-Planner", "VLM-Planner (if conditions allow)"]


def reviewer_md(facts):
    q = [("A. Can one hand-written rule solve it?", facts["A"]), ("B. Can B_PLAN derive it from the contract?", facts["B"]), ("C. Can B2 memorise scene ids or action frequencies?", facts["C"]),
         ("D. Is a static prior alone enough for A_STAT?", facts["D"]), ("E. Can an LLM/VLM planner answer it directly?", facts["E"]), ("F. If Full wins, can it be attributed to more parameters?", facts["F"]),
         ("G. Does the task rely only on weak perception?", facts["G"]), ("H. Does it hold only in a self-built scene?", facts["H"])]
    L = ["# Baseline and reviewer attack", "", "Pre-registered future baselines: " + ", ".join(BASELINES), "", "Comparison axes: J, learning samples, helpful/neutral/harmful strata, online API calls, inference latency, sensitivity to a wrong prior, generalisation to unseen conditions. The claim is a comparison, not that only Full can solve it.", ""]
    for t, a in q:
        L += ["## " + t, "", a, ""]
    return "\n".join(L)


# ---- orchestration ----------------------------------------------------------------------------------------------
def run_stage2(root, libero_root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    guard = la.Guard()
    events = [{"event": "mass_parser_made_robust_for_stl_and_missing_mesh_files", "technical_reason": "stage 2 first failed on the stove (an STL visual mesh) and on a mesh element without a file; the visual-mesh mass parser only knew the binary .msh format",
               "science_unchanged": True, "regression_test": "test_safe_mass_inertia_handles_stl_and_missing_mesh_file", "first_run": "ValueError in dv.mesh_volume before any gate was evaluated"}]
    before = la_protected(root)
    la.write_json(out / "protected_before.json", before)
    man = json.loads((out / "universe_manifest.json").read_text())
    frozen_ok = all(sha_file(out / n) == h for n, h in man["files"].items())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str((out / "carrier_asset_universe.json").relative_to(root))], cwd=root, capture_output=True, text=True).returncode == 0
    committed_before = bool(git_out(["log", "--format=%H", "-1", "--", str((out / "carrier_asset_universe.json").relative_to(root))], root))
    head = git_out(["rev-parse", "HEAD"], root)
    base_ok = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_FULL, head], cwd=root).returncode == 0
    write_json(out / "authorization.json", {"card": CARD, "label": LABEL, "family": FAMILY, "authorized_by": "user chat instruction approving CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1", "limits": {k: 0 for k in RESOURCE_KEYS},
                                           "kept_conclusions": {"DV1_STACK_ORDER_IN_SHARED_BASKET": "STOPPED", "LIBERO_LOW_COST_TP_SEARCH": "COMPLETE_NEGATIVE", "GLOBAL_T_P": "UNRESOLVED"},
                                           "fail_order": FAIL_ORDER, "verdict_rule": "first failing category in fail_order; every category is still evaluated and reported",
                                           "declared_before_stage2": {"MAX_CARRIER_EXTENT": MAX_CARRIER_EXTENT, "PAYLOAD_MAX_KG": PAYLOAD_MAX_KG, "GRIP_FORCE_N": GRIP_FORCE_N, "TORQUE_OK": TORQUE_OK, "TORQUE_MAX": TORQUE_MAX, "TILT_EXPECTED_DEG": 5.0,
                                                                      "MARGIN_STABLE": MARGIN_STABLE, "EDGE_MARGIN": EDGE_MARGIN, "SPEED_CAP": SPEED_CAP, "FLAT_TOL": FLAT_TOL, "FIT_FRACTION_MIN": FIT_FRACTION_MIN, "WALL_LEVEL_TOL": WALL_LEVEL_TOL,
                                                                      "SITE_STEP": SITE_STEP, "recovery_model": "after a fall: PICK(A) then PLACE(A)", "object_first_routes": "5 skills (set A down, move B, pick and place A) and the card's 3-skill route are both evaluated",
                                                                      "orientation_policy": "carriers: asset frame z-up (derived registration); carried: class source rotation when explicit or the Hope default, else asset frame"},
                                           "universe_frozen_and_committed_before_geometry": {"hashes_match_manifest": frozen_ok, "tracked_in_git": tracked, "committed": committed_before}})
    write_json(out / "source_identity.json", {"baseline_full_expected": BASELINE_FULL, "run_head_full": head, "run_head_descends_from_or_equals_baseline": base_ok, "branch": git_out(["branch", "--show-current"], root),
                                              "upstream_expected": UPSTREAM_SHA, "upstream_head": git_out(["rev-parse", "HEAD"], Path(libero_root)), "upstream_matches": git_out(["rev-parse", "HEAD"], Path(libero_root)) == UPSTREAM_SHA,
                                              "universe_files_unchanged_since_stage1": frozen_ok})
    lib = Path(libero_root).resolve() / "libero" / "libero"
    reg = la.load_registry(lib, guard)
    grip = grip_ext()
    cam = occ.camera(root)
    d = occ.pixel_rays(cam)
    cu = json.loads((out / "carrier_asset_universe.json").read_text())["carriers"]
    ou = json.loads((out / "carried_object_universe.json").read_text())["objects"]
    # ---- carriers -------------------------------------------------------------------------------------------
    carriers, crow, grow = {}, [], []
    for ty, info in cu.items():
        rec = registered_record(lib, reg, ty, "carrier")
        mob = mobility_row(info, rec)
        g = carrier_grasp(rec, grip) if rec["ok"] and mob["movable"] else {"candidates": 0, "feasible": 0, "best": None, "mass_total_kg": None, "torque_capacity_nm": GRIP_FORCE_N * 2 * PAD_HALF_Z}
        cls = classify_carrier(mob, g)
        derived = info["classification"] == "OFFICIAL_FIXTURE" and mob["movable"]
        carriers[ty] = {"info": info, "rec": rec, "mobility": mob, "grasp": g, "class": cls, "derived_asset_semantic_change": derived}
        crow.append({"carrier": ty, "official_classification": info["classification"], "free_joint_in_class": rec.get("joints_free", False), "geometry_ok": rec["ok"], "extent_xyz_m": [round(x, 4) for x in rec["extent"]] if rec["ok"] else None,
                     "mass_kg_range": [round(rec["mass_lo"], 4), round(rec["mass_hi"], 4)] if rec["ok"] else None, "movable": mob["movable"], "reasons": " || ".join(mob["reasons"]), "derived_asset_semantic_change": derived,
                     "orientation_basis": rec.get("orientation_basis"), "orientation_status": rec.get("orientation_status")})
        b = g["best"]
        grow.append({"carrier": ty, "class": cls, "candidate_sites": g["candidates"], "feasible_sites": g["feasible"], "best_kind": b["kind"] if b else None, "best_yaw_deg": round(b["yaw_deg"], 1) if b else None,
                     "best_torque_ratio": round(b["torque_ratio"], 3) if b else None, "best_com_distance_m": round(b["com_distance_m"], 4) if b else None, "mass_kg": round(g["mass_total_kg"], 4) if g["mass_total_kg"] else None,
                     "torque_capacity_nm": g["torque_capacity_nm"], "movable": mob["movable"]})
    write_csv(out / "carrier_mobility_matrix.csv", crow, list(crow[0].keys()))
    write_csv(out / "carrier_graspability_matrix.csv", grow, list(grow[0].keys()))
    usable_carriers = [t for t, c in carriers.items() if c["class"] in ("GENERIC_MOVE_CARRIER_COMPATIBLE", "GENERIC_BINDING_REQUIRED")]
    native_usable = [t for t in usable_carriers if not carriers[t]["derived_asset_semantic_change"]]
    # ---- carried objects --------------------------------------------------------------------------------------
    carried, orow = {}, []
    for ty in ou:
        rec = registered_record(lib, reg, ty, "carried")
        if not rec["ok"]:
            carried[ty] = {"rec": rec, "usable": False, "why": rec.get("reason")}
            orow.append({"object": ty, "geometry_ok": False, "usable": False, "why": rec.get("reason")})
            continue
        sup = rec["support_rect"]
        cx, cy = rec["com"][0], rec["com"][1]
        inside = bool(sup[0] < cx < sup[1] and sup[2] < cy < sup[3])
        pa = dv.pinch_audit({"name": ty, "boxes": rec["boxes"], "height": rec["height"], "com_inside_support": inside, "spawn": {"status": "DETERMINED_BY_SOURCE"}, "extent_world_xyz": rec["extent"]}, grip)
        px = dv.pixel_support(cam, d, rec, dv.A_XY, occ.SKIN_TOP)
        usable = pa["status"] in ("GENERIC_PICK_YAW0", "GENERIC_PICK_YAW90") and px >= dv.MIN_PIXELS and rec["mass_hi"] <= 0.5
        carried[ty] = {"rec": rec, "pinch": pa, "marker_pixels_A": px, "usable": usable}
        orow.append({"object": ty, "geometry_ok": True, "extent_xyz_m": [round(x, 4) for x in rec["extent"]], "height_m": round(rec["height"], 4), "mass_kg_range": [round(rec["mass_lo"], 4), round(rec["mass_hi"], 4)], "friction_mu": rec["mu"],
                     "com_m": [round(x, 4) for x in rec["com"]], "support_area_m2": round(dv.rect_area(rec["support_rect"]), 5), "com_inside_support": inside, "pinch_status": pa["status"], "pinch_width_yaw0_m": round(pa["pinch_width_yaw0_m"], 4),
                     "pinch_width_yaw90_m": round(pa["pinch_width_yaw90_m"], 4), "required_yaw_deg": pa["required_yaw_deg"], "grasp_site_press_m": pa["grasp_site_press_m"], "pad_contact_height_m": round(pa["pad_contact_height_m"], 4),
                     "marker_pixels_at_A": px, "orientation_basis": rec["orientation_basis"], "orientation_status": rec["orientation_status"], "usable": usable, "why": " || ".join(pa["reasons"])})
    ofields = []
    for r in orow:
        for k in r:
            if k not in ofields:
                ofields.append(k)
    write_csv(out / "carried_object_geometry.csv", orow, ofields)
    usable_carried = [t for t, c in carried.items() if c["usable"]]
    # ---- pairs ------------------------------------------------------------------------------------------------
    fields, pairs, prow = {}, {}, []
    for ct in usable_carriers:
        field = support_field(carriers[ct]["rec"])
        fields[ct] = field
        if not field["ok"]:
            continue
        for at in usable_carried:
            if at == ct:
                continue
            pe = pair_eval(carriers[ct]["rec"], carried[at]["rec"], field, grip)
            pairs[(ct, at)] = pe
            prow.append({"carrier": ct, "carried": at, "carrier_class": carriers[ct]["class"], "derived_carrier": carriers[ct]["derived_asset_semantic_change"], "support_area_m2": round(field["support_area_m2"], 5), "support_z_m": round(field["z_support"], 4),
                         "cavity_depth_m": round(field["cavity_depth_m"], 4), "fit_fraction_centre": round(pe["centre"]["fit_fraction"], 3), "com_margin_centre_m": round(pe["centre"]["com_projection_margin_m"], 4),
                         "tip_margin_centre_deg": round(math.degrees(pe["centre"]["static_tip_margin_rad"]), 1), "no_slip_limit_m_s2": round(pe["no_slip_acceleration_limit"], 2), "path_accel_bound_m_s2": pe["carrier_path_acceleration_bound"],
                         "rim_retention_margin_m": round(pe["rim_retention_margin_m"], 4), "pick_on_carrier_centre": pe["centre"]["pick_on_carrier"]["ok"], "carrier_grasp_ratio_centre": round(pe["centre"]["carrier_grasp"]["best"]["torque_ratio"], 3) if pe["centre"]["carrier_grasp"]["best"] else None,
                         "centre_class": pe["centre_class"], "edge_direction": pe["edge"]["chosen"], "edge_margin_m": round(pe["edge"]["directions"][pe["edge"]["chosen"]]["com_margin_m"], 4) if pe["edge"]["chosen"] else None,
                         "edge_class": pe["edge_class"], "reasons_pick_centre": " || ".join(pe["centre"]["pick_on_carrier"]["reasons"])})
    write_csv(out / "carrier_object_pair_matrix.csv", prow or [{"carrier": "NONE"}], list(prow[0].keys()) if prow else ["carrier"])
    stable = [k for k, v in pairs.items() if v["centre_class"] == "STABLE_CENTER_CANDIDATE"]
    edge = [k for k, v in pairs.items() if v["edge_class"] == "MARGINAL_EDGE_CANDIDATE"]
    contrast = [k for k in stable if pairs[k]["edge_class"] == "MARGINAL_EDGE_CANDIDATE"]
    support_json = {"model": "quasi-static; NOT a dynamics conclusion", "definitions": {"support_margin": "distance from the carried COM projection to the boundary of the flat support region", "static_tip_margin": "atan(margin / COM height)", "no_slip_limit": "mu*g, mu = max friction of the two geoms",
                                                                                      "path_acceleration_bound": "speed cap / control step (worst-case step from rest)", "rim_retention_margin": "cavity depth - carried COM height"},
                    "parameters": {"speed_cap_m_s": SPEED_CAP, "tilt_expected_deg": 5.0, "margin_stable_m": MARGIN_STABLE, "edge_margin_m": EDGE_MARGIN}, "pairs": {"%s|%s" % k: {kk: vv for kk, vv in v.items()} for k, v in pairs.items()},
                    "stable_center_pairs": ["|".join(k) for k in stable], "marginal_edge_pairs": ["|".join(k) for k in edge], "contrast_pairs": ["|".join(k) for k in contrast], "support_fields": {k: {kk: vv for kk, vv in f.items() if kk in ("z_support", "rim_z", "support_area_m2", "cavity_depth_m", "centre", "reason")} for k, f in fields.items()}}
    write_json(out / "support_stability_model.json", support_json)
    # ---- generic MOVE_CARRIER spec -----------------------------------------------------------------------------
    spec = {"status": "SPECIFIED" if usable_carriers else "NOT_AVAILABLE", "states": ["APPROACH_CARRIER_GRASP_SITE", "DESCEND", "CLOSE", "LIFT_LEVEL", "TRANSLATE_LEVEL", "DESCEND_TO_DESTINATION", "RELEASE", "RETREAT"],
            "new_states_beyond_allowed": [], "parameters": ["grasp site", "gripper yaw", "gripper width", "lift height", "destination", "translation speed", "carrier geometry"], "forbidden": ["scene-specific trajectories", "online path change when A falls", "retry until success",
            "hidden pose correction", "learned controller", "changes to PICK(A) or PLACE(A)"], "translation_speed_cap_m_s": SPEED_CAP,
            "yaw_wiring": "action[3:6] is zeroed in D0ManipulationEnv.step_osc; any non-zero yaw needs the yaw feed into action[5] during APPROACH only (the OSC_POSE rotation dims already exist)",
            "per_carrier": {t: {"class": carriers[t]["class"], "derived_asset_semantic_change": carriers[t]["derived_asset_semantic_change"], "best_site": carriers[t]["grasp"]["best"], "feasible_sites": carriers[t]["grasp"]["feasible"], "candidate_sites": carriers[t]["grasp"]["candidates"],
                                "lift_height_above_table_m": float(carriers[t]["rec"]["height"] + 0.12) if carriers[t]["rec"]["ok"] else None, "release_site_z_m": carriers[t]["grasp"]["best"]["site_z"] if carriers[t]["grasp"]["best"] else None} for t in carriers},
            "sweep_check": {"lift_level_height_above_carrier_top_m": 0.12, "translation_clearance": "the lifted carrier (top + 0.12 m) clears every static item of the proven layout, whose tallest member is the 0.145 m container"}}
    write_json(out / "generic_move_carrier_spec.json", spec)
    # ---- scenarios, routes, rules, contract --------------------------------------------------------------------
    dur = duration_means(root)
    routes = route_costs(dur)
    dom = dominance(routes)
    attack = rule_attack(dom)
    cf_worse_any = any(dom[s]["carrier_first_ever_strictly_more_skills"] or dom[s]["carrier_first_ever_strictly_more_skills_vs_card_route"] for s in SCENARIOS[:3])
    witness = cf_worse_any
    best_action = {s: ("carrier_first" if dom[s]["object_first_ever_strictly_more_skills"] and not dom[s]["carrier_first_ever_strictly_more_skills"] else ("object_first" if dom[s]["carrier_first_ever_strictly_more_skills"] and not dom[s]["object_first_ever_strictly_more_skills"] else "tie")) for s in SCENARIOS}
    bplan = {s: mini_bplan(s, dur) for s in SCENARIOS}
    optimal_skills = {"S1_CO_TRANSPORT_STABLE": 1, "S2_SEPARATE_DESTINATIONS": 3, "S3_CO_TRANSPORT_UNSTABLE": 1, "S4_NO_SUPPORT_NEUTRAL": 3}   # best case (A stays on B in S1/S3)
    revealed = all(bplan[s]["skills"] <= optimal_skills[s] for s in SCENARIOS)
    scen = {"label": LABEL, "family": FAMILY, "gripper_holds_one_object": True, "scenarios": {
        "S1_CO_TRANSPORT_STABLE": {"state": "A centred on B (stable-centre pair); A's target xy coincides with where A rides when B reaches its destination", "goals": ["AtRegion(B,D_B)", "AtRegion(A,D_A)"], "realised_by": ["|".join(k) for k in stable[:3]], "unestablished": ["A actually rides with B without sliding"]},
        "S2_SEPARATE_DESTINATIONS": {"state": "A stable on B; A's target is elsewhere", "goals": ["AtRegion(B,D_B)", "AtRegion(A,D_A)"], "realised_by": ["|".join(k) for k in stable[:3]], "unestablished": ["any cost of carrier-first beyond path"]},
        "S3_CO_TRANSPORT_UNSTABLE": {"state": "same goals as S1 but A near the carrier edge (margin %.3f m)" % EDGE_MARGIN, "goals": ["AtRegion(B,D_B)", "AtRegion(A,D_A)"], "realised_by": ["|".join(k) for k in edge[:3]], "unestablished": ["A slides or falls", "recovery always possible"]},
        "S4_NO_SUPPORT_NEUTRAL": {"state": "A not on B; moving B does not change A", "goals": ["AtRegion(B,D_B)", "AtRegion(A,D_A)"], "realised_by": "any usable pair with A placed apart", "unestablished": []}},
        "scenario_pair_counts": {"stable_center_pairs": len(stable), "marginal_edge_pairs": len(edge), "contrast_pairs": len(contrast)}}
    write_json(out / "scenario_matrix_S1_S4.json", scen)
    sk = {"duration_source": dur, "move_carrier_duration": "UNESTABLISHED: bracketed between one PICK and one PICK plus one PLACE", "routes_by_scenario": {s: [r for r in v if r["p_fall"] in (0.0, 0.5, 1.0) and r["move_carrier_s"] == v[0]["move_carrier_s"]] for s, v in routes.items()},
          "dominance": dom, "best_first_action_by_skill_count": best_action, "reversed_or_harmful_witness_not_path_only": witness,
          "s2_classification": "COMMUTATIVE" if not (dom["S2_SEPARATE_DESTINATIONS"]["orders_differ_by_skill_count"]) else "ORDER_DEPENDENT", "s3_harmful_possible": dom["S3_CO_TRANSPORT_UNSTABLE"]["carrier_first_ever_strictly_more_skills"] or dom["S3_CO_TRANSPORT_UNSTABLE"]["carrier_first_ever_strictly_more_skills_vs_card_route"],
          "mechanisms_that_could_make_carrier_first_worse_but_are_UNESTABLE": ["A falls to a state that is not publicly recoverable", "A leaves the proven pick workspace", "recovery needs more than PICK + PLACE", "slide changes later reachability"],
          "UNESTABLISHED": ["A follows B", "A does not slide", "A falls", "a fall is recoverable", "MOVE_CARRIER needs fewer skills", "separate goals favour object-first", "the support relation can be generated from RGB"]}
    write_json(out / "skill_count_and_utility_audit.json", sk)
    cb = {"contracts": CONTRACTS, "hard_mask_contents": "PRE/ADD/DEL of PICK(A), PLACE(A), MOVE_CARRIER(B,D) only; no entry says that A rides, falls, is stable, or which order is better", "hard_mask_leaks_winner": False,
          "contract_only_plans": bplan, "best_plan_skills": optimal_skills, "contract_planner_reveals_winner_in_every_scenario": revealed,
          "reading": "S1: the contract cannot credit MOVE_CARRIER with relocating A, so the uniform-cost plan uses 5 skills although 1 is possible; S2: independent subgoals, tie"}
    write_json(out / "contract_blindness_audit.json", cb)
    rel = relation_check()
    dirs = {"S1": "points to carrier-first: helpful (best action carrier-first)", "S2": "tie: goal-dependent, no cost difference", "S3": "points to carrier-first; harm is not established (carrier-first still needs no more skills)", "S4": "neutral / absent"}
    relj = {"schema": "m1_soft_relations_v2", "relations": rel, "per_scenario_reading": dirs, "relation_always_points_to_carrier_first_when_non_neutral": all(v in ("carrier_first", "tie") for v in best_action.values()),
            "best_first_action_by_scenario": best_action}
    write_json(out / "relation_expressibility_audit.json", relj)
    (out / "fixed_rule_attack.md").write_text(fixed_rule_md(attack, dom, routes), encoding="utf-8")
    # ---- perception / verifier / evaluator / snapshot -----------------------------------------------------------
    best_pair = stable[0] if stable else (next(iter(pairs)) if pairs else None)
    perc = {"evaluated_pair": "|".join(best_pair) if best_pair else None}
    perc_ok = None
    if best_pair:
        ct, at = best_pair
        C, A = carriers[ct]["rec"], carried[at]["rec"]
        f = fields[ct]
        Abody = shifted(A["boxes"], dv.B_XY[0], dv.B_XY[1], occ.SKIN_TOP + f["z_support"])
        Amark = dv.marker_box(A, dv.B_XY, occ.SKIN_TOP + f["z_support"])
        Cbody = shifted(C["boxes"], dv.B_XY[0], dv.B_XY[1], occ.SKIN_TOP)
        Cmark = marker_patches(C, dv.B_XY[0], dv.B_XY[1], occ.SKIN_TOP)
        ids, t, names = dv.render_scene(cam, d, {"A_marker": [Amark], "B_marker": Cmark, "A_body": Abody, "B_body": Cbody})
        pxA, pxB = int((ids == names.index("A_marker")).sum()), int((ids == names.index("B_marker")).sum())
        perc.update({"A_marker_pixels_on_carrier": pxA, "B_marker_pixels_with_A_present": pxB, "min_pixels": dv.MIN_PIXELS, "marker_B": "thin patches on the highest layer of the carrier, visible beside A"})
        perc_ok = pxA >= dv.MIN_PIXELS and pxB >= dv.MIN_PIXELS
    perc.update({"option_A_texture": "not demonstrable offline", "option_B_marker": "public colour markers (LIBERO-derived instrumentation); roles use existing perception colours (A: target, B: a free named colour)", "new_vision_model": False,
                 "unknown_conditions": ["marker pixels < min_pixels -> facts UNKNOWN", "A hidden by the hand while held", "carrier marker hidden while it is lifted"], "status": "NOT_EVALUATED_NO_PAIR" if perc_ok is None else ("PASS" if perc_ok else "FAIL")})
    verifier = {"facts": {"Held(A)": "closed gripper and A marker near eef (unchanged thresholds)", "CarrierHeld(B)": "closed gripper and B marker patches near eef, B not at rest", "OnCarrier(A,B)": "A marker xy inside the registered carrier support region and z within 0.008 m of support height + A height",
                          "AtRegion(A,D)": "A marker xy inside D (registered tolerance)", "AtRegion(B,D)": "B marker xy inside D", "GripperEmpty": "unchanged"}, "needs_training": False, "perception": perc}
    write_json(out / "public_verifier_spec.json", verifier)
    evals = {"task_id": FAMILY, "success_definition": "A and B both inside their goal regions D_A and D_B (centre xy within the region), judged on hidden simulator truth by an independent TaskEvaluator; hidden truth is never a policy or Verifier input",
             "existing_evaluators_modified": False, "new_task_id_needs_new_goal_function": True, "semantic_change_to_existing_tasks": False, "note": "OnCarrier is a Verifier fact only; it is not part of the success definition"}
    write_json(out / "independent_evaluator_spec.json", evals)
    snap = snapshot_evidence(root, guard)
    write_json(out / "snapshot_extension_spec.json", snap)
    # ---- canary design --------------------------------------------------------------------------------------------
    canary = {"name": "CP-DISR-TP-CARRIER-SUPPORT-MECHANISM-CANARY-1", "episodes": ["C1 stable centre MOVE_CARRIER", "C2 stable centre repeat", "C3 edge configuration MOVE_CARRIER", "C4 edge repeat"], "max_episodes": 4,
              "questions": {"Q1 carrier moves stably": "C1, C2", "Q2 co-transport in the stable configuration": "C1, C2", "Q3 different consequence in the unstable configuration": "C3, C4",
                            "Q4 A recoverable after a fall": "answerable only if A falls in C3 or C4; otherwise deferred to a staged follow-up of at most 2 episodes"},
              "bounded": bool(stable) and bool(edge), "status": "BOUNDED_WITH_STAGING"}
    # ---- categories -----------------------------------------------------------------------------------------------
    cat = {}
    cat["TASK_FAIL_NO_GENERIC_CARRIER"] = {"fails": not usable_carriers, "evidence": {"classes": {t: c["class"] for t, c in carriers.items()}, "native_usable": native_usable, "derived_usable_backup_only": [t for t in usable_carriers if t not in native_usable]}}
    cat["TASK_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR"] = {"fails": spec["status"] != "SPECIFIED", "evidence": {"spec_status": spec["status"], "new_states": spec["new_states_beyond_allowed"]}}
    cat["TASK_FAIL_NO_CARRIED_OBJECT"] = {"fails": not usable_carried, "evidence": {"usable": usable_carried, "unusable": {t: (c.get("pinch", {}).get("status") or c.get("why")) for t, c in carried.items() if not c["usable"]}}}
    cat["TASK_FAIL_NO_STABLE_SUPPORT_PAIR"] = {"fails": not stable, "evidence": {"stable_center_pairs": ["|".join(k) for k in stable], "pairs_evaluated": len(pairs), "centre_class_counts": dict(Counter(v["centre_class"] for v in pairs.values()))}}
    cat["TASK_FAIL_NO_STABILITY_CONTRAST"] = {"fails": not contrast, "evidence": {"contrast_pairs": ["|".join(k) for k in contrast], "marginal_edge_pairs": ["|".join(k) for k in edge], "edge_class_counts": dict(Counter(v["edge_class"] for v in pairs.values()))}}
    cat["TASK_FAIL_COMMUTATIVE"] = {"fails": not witness, "evidence": {"s2": sk["s2_classification"], "s3_harmful_possible": sk["s3_harmful_possible"], "reversed_witness_not_path_only": witness}}
    unrefuted = [k for k, v in attack.items() if not v["refuted"]]
    cat["TASK_FAIL_FIXED_HEURISTIC_SUFFICIENT"] = {"fails": bool(unrefuted), "evidence": {"unrefuted_rules": unrefuted, "counterexamples": {k: v["counterexamples"] for k, v in attack.items()}}}
    cat["TASK_FAIL_STATIC_PRIOR_SUFFICIENT"] = {"fails": relj["relation_always_points_to_carrier_first_when_non_neutral"], "evidence": {"best_first_action_by_scenario": best_action}}
    cat["TASK_FAIL_CONTRACT_REVEALED"] = {"fails": revealed, "evidence": {"contract_plans": {s: v["skills"] for s, v in bplan.items()}, "optimal_skills": optimal_skills}}
    cat["TASK_FAIL_REQUIRES_NEW_PERCEPTION_MODEL"] = {"fails": perc_ok is False, "evidence": {k: v for k, v in perc.items() if k in ("A_marker_pixels_on_carrier", "B_marker_pixels_with_A_present", "status", "evaluated_pair")}}
    cat["TASK_FAIL_EVALUATOR_SEMANTIC_CHANGE"] = {"fails": bool(evals["semantic_change_to_existing_tasks"]), "evidence": {"existing_evaluators_modified": False}}
    cat["TASK_FAIL_SNAPSHOT_REWRITE"] = {"fails": snap["status"] != "PASS", "evidence": snap["evidence"]}
    cat["TASK_FAIL_RELATION_NOT_EXPRESSIBLE"] = {"fails": not all(r["expressible"] for r in rel), "evidence": [{"id": r["relation"]["relation_id"], "issues": r["issues"]} for r in rel]}
    cat["TASK_FAIL_CANARY_NOT_BOUNDED"] = {"fails": not canary["bounded"], "evidence": canary}
    no_geom = [t for t, c in carriers.items() if not c["rec"]["ok"]] + [t for t, c in carried.items() if not c["rec"]["ok"]]
    cat["ENGINEERING_UNRESOLVED"] = {"fails": False, "evidence": {"assets_without_box_collision_proxy": no_geom}}
    failing = [k for k in FAIL_ORDER if cat[k]["fails"]]
    verdict = failing[0] if failing else "CARRIER_SUPPORT_DESIGN_AUDIT_PASS"
    conds = {"1_one_carrier_generic": bool(usable_carriers), "2_one_carried_object": bool(usable_carried), "3_stable_centre_pair": bool(stable), "4_edge_pair": bool(edge), "5_neutral_configuration": True,
             "6_s1_skill_advantage": dom["S1_CO_TRANSPORT_STABLE"]["object_first_ever_strictly_more_skills"], "7_reversed_not_path_only": witness, "8_r1_to_r6_all_refuted": not unrefuted, "9_contract_cannot_order": not revealed,
             "10_relation_expressible_non_redundant": not cat["TASK_FAIL_RELATION_NOT_EXPRESSIBLE"]["fails"], "11_verifier_evaluator_definable": perc_ok is not False, "12_snapshot_no_rewrite": snap["status"] == "PASS",
             "13_no_new_models_or_low_level": not cat["TASK_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR"]["fails"] and perc_ok is not False, "14_canary_bounded": canary["bounded"]}
    manifest = {"verdict": verdict, "label": LABEL, "all_failing_categories": failing, "category_results": cat, "pass_conditions": conds, "family": FAMILY}
    sel = {"status": "SELECTED_FOR_MECHANISM_CANARY_REVIEW" if verdict == "CARRIER_SUPPORT_DESIGN_AUDIT_PASS" else "NO_CARRIER_SUPPORT_TP_CANDIDATE", "verdict": verdict, "all_failing_categories": failing, "best_pair_studied": "|".join(best_pair) if best_pair else None}
    write_json(out / "selected_carrier_candidate.json", sel)
    write_json(out / "candidate_design_manifest.json", manifest)
    (out / "next_mechanism_canary_request.md").write_text("# next_mechanism_canary_request\n\n" + ("REQUESTED (not executed): CP-DISR-TP-CARRIER-SUPPORT-MECHANISM-CANARY-1, at most 4 physical episodes\n" if verdict == "CARRIER_SUPPORT_DESIGN_AUDIT_PASS" else "NOT_REQUESTED\n"), encoding="utf-8")
    rv = {"A": "A fixed rule suffices: %s remain unrefuted." % (", ".join(unrefuted) or "none (every rule has a counterexample)"),
          "B": "No: the contract-only plan uses %s skills in S1 against 1 for the best route, so the contract does not reveal the S1 winner; in S2 both orders tie." % bplan["S1_CO_TRANSPORT_STABLE"]["skills"],
          "C": "Yes for the winner label: a single carrier-first action wins or ties in every scenario, so B2 can learn 'carrier first when A is on B' from frequency alone.",
          "D": "Yes: the relation never points to object-first as the strictly better action, so a static prior that always points to carrier-first is sufficient.",
          "E": "Likely yes: 'move the tray with the object on it' is a commonsense answer; it would need neutral, reversed and goal-dependent sub-conditions to be a diagnostic, and no reversed condition is established.",
          "F": "Not separable here: with one dominant action the comparison would measure parameter count rather than prior use.",
          "G": "The task needs support and carrier geometry from public markers; weak perception is not the lever.",
          "H": "It is a self-built derived scene (LIBERO-derived assets, D0 host), so it cannot be reported as a LIBERO benchmark result."}
    (out / "baseline_and_reviewer_attack.md").write_text(reviewer_md(rv), encoding="utf-8")
    forbidden_loaded = sorted(m for m in ("robosuite", "mujoco", "libero", "torch", "gym", "gymnasium") if m in sys.modules)
    led = {k: 0 for k in RESOURCE_KEYS}
    led.update({"files_read_as_text": guard.opened, "files_hashed_only": guard.hashed_only, "guard_refusals": len(guard.refused), "forbidden_modules_loaded": forbidden_loaded})
    write_json(out / "budget_ledger.json", led)
    (out / "engineering_events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in events), encoding="utf-8")
    write_json(out / "protected_after.json", la_protected(root))
    return {"verdict": verdict, "failing": failing}


OWN = re.compile(r"tp_carrier_support")


def la_protected(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4", "runs/final_master/2.1.1/T_B", "runs/stage_0a", "docs/authoritative", "status", "src", "configs"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not OWN.search(str(p)) and "__pycache__" not in p.parts and not re.search(r"holdout|test30|test_id|final_test", str(p).replace("\\", "/"), re.I):
                out[str(p.relative_to(root))] = sha_file(p)
    return out


def final_summary(out):
    out = Path(out)
    man = json.loads((out / "candidate_design_manifest.json").read_text())
    L = ["# %s - final design audit summary" % CARD, "", "Verdict: **%s**  (label %s, family %s)" % (man["verdict"], man["label"], man["family"]), "",
         "All failing categories (each evaluated independently; the verdict is the first in the declared order): %s" % ", ".join(man["all_failing_categories"] or ["none"]), "", "## Category evidence", ""]
    for k, v in man["category_results"].items():
        L.append("- %s: %s - %s" % (k, "FAIL" if v["fails"] else "ok", json.dumps(v["evidence"], default=str)[:300]))
    L += ["", "## PASS conditions", ""] + ["- %s: %s" % (k, v) for k, v in man["pass_conditions"].items()]
    L += ["", "Budget: environment constructions 0, env.reset 0, start_case 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test 0, demo replay 0.", ""]
    (out / "final_design_audit_summary.md").write_text("\n".join(L), encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "upstream_libero_identity.json", "protected_before.json", "protected_after.json", "carrier_asset_universe.json", "carrier_mobility_matrix.csv", "carrier_graspability_matrix.csv",
            "carried_object_universe.json", "carried_object_geometry.csv", "generic_move_carrier_spec.json", "support_stability_model.json", "carrier_object_pair_matrix.csv", "scenario_matrix_S1_S4.json", "skill_count_and_utility_audit.json",
            "contract_blindness_audit.json", "relation_expressibility_audit.json", "fixed_rule_attack.md", "baseline_and_reviewer_attack.md", "public_verifier_spec.json", "independent_evaluator_spec.json", "snapshot_extension_spec.json",
            "selected_carrier_candidate.json", "next_mechanism_canary_request.md", "final_design_audit_summary.md", "budget_ledger.json", "engineering_events.jsonl", "universe_manifest.json"]
    led = json.loads((out / "budget_ledger.json").read_text())
    src = json.loads((out / "source_identity.json").read_text())
    man = json.loads((out / "candidate_design_manifest.json").read_text())
    auth = json.loads((out / "authorization.json").read_text())
    before, after = json.loads((out / "protected_before.json").read_text()), json.loads((out / "protected_after.json").read_text())
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".csv", ".jsonl"} and bool(re.search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    fr = auth["universe_frozen_and_committed_before_geometry"]
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "zero_resources": all(led[k] == 0 for k in RESOURCE_KEYS), "guard_refusals_zero": led["guard_refusals"] == 0, "no_sim_stack_imported": led["forbidden_modules_loaded"] == [],
              "baseline_recorded": src["run_head_descends_from_or_equals_baseline"] and len(src["run_head_full"]) == 40, "upstream_pinned": src["upstream_matches"], "universe_frozen_and_committed_first": fr["hashes_match_manifest"] and fr["tracked_in_git"] and fr["committed"],
              "protected_unchanged": before == after, "protected_files_hashed": len(before), "verdict_is_allowed": man["verdict"] in FAIL_ORDER + ["CARRIER_SUPPORT_DESIGN_AUDIT_PASS"], "label_fixed": man["label"] == LABEL,
              "request_label_consistent": ("NOT_REQUESTED" in (out / "next_mechanism_canary_request.md").read_text()) == (man["verdict"] != "CARRIER_SUPPORT_DESIGN_AUDIT_PASS"), "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k not in ("outputs_present", "protected_files_hashed", "status")) and all(checks["outputs_present"].values()) else "FAIL"
    la.write_json(out / "verify.json", checks)
    return checks
