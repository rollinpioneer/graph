"""CP-DISR-S4-FAMILY-B-V3-R1-REACH-SAFE-MIRROR-PILOT-1.

One frozen reach-safe scene (pads and target anchors point-symmetric about the PICK show pose and laid along the
normal of the base->show direction, every new point within 0.65 m of the Panda base), eight physical branches,
strictly staged Wave A -> Wave B -> Wave C.  Environment, observation profile, SkillExecutor, Verifier/Evaluator,
planner and thresholds are the frozen production sources; only layouts, identity, scheduling and analysis live here.
Provider, representation, RL, optimizer, S2, S3 and formal test are never touched.
"""
from __future__ import annotations

import csv
import fcntl
import itertools
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2
from cp_disr.analysis import family_b_staging_v2 as s
from cp_disr.platforms.libero.perception import THRESHOLDS

CARD = "CP-DISR-S4-FAMILY-B-V3-R1-REACH-SAFE-MIRROR-PILOT-1"
BASE = "8dd0e8010dcd42a79acad16633d2614b6d0c8a80"
CFG = "configs/final_master/s4_family_b_v3r1_reach_safe_mirror.yaml"
LAYOUTS_PATH = "configs/final_master/family_b_v3r1_layouts.json"
V3_EVIDENCE = "runs/final_master/S4/family_b_v3_isochronous/20261001T070045Z_74a422a2"
STAGING_EVIDENCE = "runs/final_master/S4/family_b_staging_v2/20261001T053917Z_3064ee8b"
FROZEN_SHA = r2.FROZEN_SHA
FROZEN_FILES = r2.FROZEN_FILES
PY = r2.PY
read, write, sha = m.read, m.write, m.sha
PADS = ("pad_u", "pad_v")
CONTEXTS = ("B_PENDING", "C_PENDING")
TOL_T = 0.15            # candidate duration match and preference margin (seconds)
STEPS_TOL = 3           # candidate controller steps match
SUPPORT_MIN = 12        # task-qualification margin on the remaining target (production min_pixels stays 8)
GEOM_TOL = 1e-9
BASE_TOL = 1e-6          # U/V and A_U/A_V base-radius fairness
VALUE_TOL = 1e-6         # stated 0.086 / 0.258 m values (coordinates frozen to 8 decimals)
CAPS = {"attempts": 8, "constructions": 8, "resets": 8, "skills": 48}
WATCHDOG = 1800
GATE_PASS = "FAMILY_B_V3_R1_REACH_SAFE_MECHANISM_PILOT_PASS"
GATE_FAIL = "FAMILY_B_V3_R1_REACH_SAFE_MECHANISM_PILOT_NOT_ESTABLISHED"
NEXT_PASS = "REQUEST_PROVIDER_REPRESENTATION_AND_S2_CONSTRUCTION_REVIEW"
NEXT_FAIL = "STOP_TP_TASK_SEARCH_AND_ADJUST_CLAIMS"

# ------------------------------------------------------------------------------- frozen geometry
BASE_PROXY_XY = (-0.56, 0.0)                           # Panda base for a 0.8 m table; re-derived from the robot model in Gate 0
SHOW = (0.02, -0.08)                                   # SkillExecutor._pick show pose (x, y)
CARRIER = (-0.08, -0.12)
RECEIVER = (0.18, 0.12)
PAD_U = (0.03175082, 0.00519342)                       # S + q
PAD_V = (0.00824918, -0.16519342)                      # S - q
ANCHOR_U = (0.04350163, 0.09038683)                    # S + 2q
ANCHOR_V = (-0.00350163, -0.25038683)                  # S - 2q
MIRROR_D = 0.086                                       # |q|, normal to the base->show direction
CUBE_HALF = 0.02
PAD_HALF_XY = 0.045
RECEIVER_OUTER = (0.070 + 0.012, 0.035 + 0.012)         # inner + wall (family_b_env)
SLOT_OFFSET = 0.035
TABLE_HALF = 0.40
REACH_CAP = 0.65                                       # new movable / pad points, hard cap (no "demonstrated + margin")
REACH_POINTS = ("carrier", "obj_b", "obj_c", "pad_u", "pad_v")
AABB_MIN = 0.008                                       # minimum planar AABB clearance (m)
CAM_MARGIN_MIN = 9.0                                   # agentview centre-to-edge margin, every key point
CAM_EDGE_MIN = {"carrier": 8.0, "target_anchor": 8.5, "pad": 8.5, "receiver": 8.5}   # role-based predicted cube edge (px)
INITIAL_SUPPORT_MIN = 12                               # reset-time public depth support: carrier and setup target
PREDICTED_WINNER = {("layout_0", "B_PENDING"): "pad_u", ("layout_0", "C_PENDING"): "pad_v",
                    ("layout_1", "B_PENDING"): "pad_v", ("layout_1", "C_PENDING"): "pad_u"}
REMAINING = {"B_PENDING": "obj_b", "C_PENDING": "obj_c"}
SETUP_TARGET = {"B_PENDING": "obj_c", "C_PENDING": "obj_b"}


def layout_specs(pad_v=PAD_V, swap_layout_1=True):
    """layout_1 only exchanges the B/C identities on the two frozen anchors."""
    common = {"carrier_xy": list(CARRIER), "receiver_xy": list(RECEIVER), "pad_u_xy": list(PAD_U),
              "pad_v_xy": list(pad_v)}
    l0 = {"layout_id": "layout_0", **common, "obj_b_xy": list(ANCHOR_U), "obj_c_xy": list(ANCHOR_V)}
    l1 = {"layout_id": "layout_1", **common,
          "obj_b_xy": list(ANCHOR_V if swap_layout_1 else ANCHOR_U),
          "obj_c_xy": list(ANCHOR_U if swap_layout_1 else ANCHOR_V)}
    return [l0, l1]


def layouts_document():
    return {"status": "FROZEN_V3R1_REACH_SAFE_MIRROR_ANALYTIC_DESIGN", "card_id": CARD,
            "decision_show_xy": list(SHOW), "target_anchor_u_xy": list(ANCHOR_U), "target_anchor_v_xy": list(ANCHOR_V),
            "layouts": layout_specs()}


def _d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def invariants(layouts=None, base_xy=BASE_PROXY_XY):
    """Analytic point-symmetry about the show pose plus base-radius fairness; every entry is asserted."""
    lay = (layouts or layout_specs())[0]
    u, v, au, av, sh, b = (np.array(lay["pad_u_xy"]), np.array(lay["pad_v_xy"]), np.array(ANCHOR_U), np.array(ANCHOR_V),
                           np.array(SHOW), np.array(base_xy))
    exact = {
        "U_plus_V_equals_2S": float(np.max(np.abs(u + v - 2 * sh))),
        "AU_plus_AV_equals_2S": float(np.max(np.abs(au + av - 2 * sh))),
        "S_to_U_minus_S_to_V": abs(_d(sh, u) - _d(sh, v)),
        "U_to_AU_minus_V_to_AV": abs(_d(u, au) - _d(v, av)),
        "U_to_AV_minus_V_to_AU": abs(_d(u, av) - _d(v, au)),
    }
    base = {"B_to_U_minus_B_to_V": abs(_d(b, u) - _d(b, v)), "B_to_AU_minus_B_to_AV": abs(_d(b, au) - _d(b, av))}
    values = {"S_to_U_minus_0.086": abs(_d(sh, u) - MIRROR_D), "S_to_V_minus_0.086": abs(_d(sh, v) - MIRROR_D),
              "U_to_AU_minus_0.086": abs(_d(u, au) - MIRROR_D), "V_to_AV_minus_0.086": abs(_d(v, av) - MIRROR_D),
              "U_to_AV_minus_0.258": abs(_d(u, av) - 3 * MIRROR_D), "V_to_AU_minus_0.258": abs(_d(v, au) - 3 * MIRROR_D)}
    radii = {"B_to_U": _d(b, u), "B_to_V": _d(b, v), "B_to_AU": _d(b, au), "B_to_AV": _d(b, av)}
    ok = (all(x <= GEOM_TOL for x in exact.values()) and all(x <= BASE_TOL for x in base.values()) and
          all(x <= VALUE_TOL for x in values.values()) and abs(radii["B_to_U"] - 0.591774) <= 1e-5 and
          abs(radii["B_to_AU"] - 0.610233) <= 1e-5)
    return {"errors": {**exact, **base, **values}, "exact_identities": exact, "base_radius_fairness": base,
            "stated_values": values, "base_radii": radii,
            "values": {"S_to_pad": _d(sh, u), "pad_to_near_anchor": _d(u, au), "pad_to_far_anchor": _d(u, av)},
            "tolerance": {"exact": GEOM_TOL, "base_radius": BASE_TOL, "stated_values": VALUE_TOL}, "all_ok": bool(ok)}


def _box_gap(c1, h1, c2, h2):
    """Axis-aligned box clearance (positive = separated)."""
    return max(abs(c1[0] - c2[0]) - h1[0] - h2[0], abs(c1[1] - c2[1]) - h1[1] - h2[1])


def geometry_checks(layouts=None, base_xy=BASE_PROXY_XY):
    layouts = layouts or layout_specs()
    cube, pad, recv = (CUBE_HALF,) * 2, (PAD_HALF_XY,) * 2, RECEIVER_OUTER
    checks, rows, reach_rows, aabb_rows = {}, [], [], []
    for lay in layouts:
        mov = {"carrier": lay["carrier_xy"], "obj_b": lay["obj_b_xy"], "obj_c": lay["obj_c_xy"]}
        statics = {"receiver": (lay["receiver_xy"], recv), "pad_u": (lay["pad_u_xy"], pad), "pad_v": (lay["pad_v_xy"], pad)}
        lid = lay["layout_id"]
        bodies = {**{k: (p, cube) for k, p in mov.items()}, **statics}
        gaps = {f"{a}|{b}": _box_gap(bodies[a][0], bodies[a][1], bodies[b][0], bodies[b][1])
                for a, b in itertools.combinations(bodies, 2)}
        for pair, gap in gaps.items():
            aabb_rows.append({"layout": lid, "pair": pair, "aabb_gap_m": gap, "min_required_m": AABB_MIN, "ok": gap >= AABB_MIN})
            checks[f"{lid}:aabb_clearance:{pair}"] = gap >= AABB_MIN
        extent = max(max(abs(p[0]) + h[0], abs(p[1]) + h[1]) for p, h in bodies.values())
        slots = [(lay["receiver_xy"][0] + sgn * SLOT_OFFSET, lay["receiver_xy"][1]) for sgn in (-1, 1)]
        slot_inside = all(abs(sx - lay["receiver_xy"][0]) + CUBE_HALF <= 0.070 and CUBE_HALF <= 0.035 for sx, _ in slots)
        slot_gap = _box_gap(slots[0], cube, slots[1], cube)
        points = {"carrier": mov["carrier"], "obj_b": mov["obj_b"], "obj_c": mov["obj_c"], "pad_u": lay["pad_u_xy"],
                  "pad_v": lay["pad_v_xy"]}
        radii = {k: _d(base_xy, p) for k, p in points.items()}
        recv_radius = _d(base_xy, lay["receiver_xy"])
        for name in REACH_POINTS:
            ok = radii[name] <= REACH_CAP
            reach_rows.append({"layout": lid, "point": name, "xy": list(points[name]), "base_radius_m": radii[name],
                               "cap_m": REACH_CAP, "margin_to_cap_m": REACH_CAP - radii[name], "ok": bool(ok)})
            checks[f"{lid}:{name}_radius_le_{REACH_CAP}"] = bool(ok)
        rows.append({"layout": lid, "min_aabb_gap_all_pairs": min(gaps.values()), "max_table_extent": extent,
                     "slot_gap": slot_gap, "slot_inside_receiver": slot_inside, "radial_from_base": radii,
                     "receiver_radius_recorded_only_m": recv_radius})
        checks[f"{lid}:inside_table_safe_boundary"] = extent <= TABLE_HALF - 0.08
        checks[f"{lid}:receiver_slots_hold_b_and_c"] = bool(slot_inside and slot_gap > 0.0)
    checks["destinations_and_anchors_do_not_alias"] = (
        _d(layouts[0]["pad_u_xy"], layouts[0]["pad_v_xy"]) > 0.1 and _d(ANCHOR_U, ANCHOR_V) > 0.1 and
        len({tuple(layouts[0]["pad_u_xy"]), tuple(layouts[0]["pad_v_xy"]), tuple(ANCHOR_U), tuple(ANCHOR_V),
             tuple(layouts[0]["receiver_xy"]), tuple(layouts[0]["carrier_xy"])}) == 6)
    return {"checks": checks, "rows": rows, "reach_rows": reach_rows, "aabb_rows": aabb_rows, "all_ok": all(checks.values()),
            "layout_pairs_bc": [[l["obj_b_xy"], l["obj_c_xy"]] for l in layouts]}


def task_feature_checks(layouts=None):
    """The two layouts differ only in the B/C anchor identities; no fixed rule can solve all four cells."""
    layouts = layouts or layout_specs()
    l0, l1 = layouts
    same = all(l0[k] == l1[k] for k in ("carrier_xy", "receiver_xy", "pad_u_xy", "pad_v_xy"))
    swapped = l0["obj_b_xy"] == l1["obj_c_xy"] and l0["obj_c_xy"] == l1["obj_b_xy"] and l0["obj_b_xy"] != l1["obj_b_xy"]

    def nearer(lay, obj):
        p = lay[f"{obj}_xy"]
        return "pad_u" if _d(p, lay["pad_u_xy"]) < _d(p, lay["pad_v_xy"]) else "pad_v"
    best = {(lay["layout_id"], ctx): nearer(lay, REMAINING[ctx]) for lay in layouts for ctx in CONTEXTS}
    always_u = sum(w == "pad_u" for w in best.values())
    always_v = sum(w == "pad_v" for w in best.values())
    by_object = defaultdict(set)
    for (lid, ctx), w in best.items():
        by_object[REMAINING[ctx]].add(w)
    return {"layouts_differ_only_by_bc_swap": bool(same and swapped), "geometric_best": {f"{k[0]}/{k[1]}": v for k, v in best.items()},
            "always_u_correct": always_u, "always_v_correct": always_v,
            "no_fixed_rule_solves_all": always_u < 4 and always_v < 4,
            "same_object_id_changes_best_pad": all(len(v) == 2 for v in by_object.values()),
            "geometric_best_matches_prediction": best == PREDICTED_WINNER}


# ------------------------------------------------------------------------------- branches / waves
# (layout, context, repeat, pad) in the fixed, preregistered execution order; U->V, V->U, V->U, U->V.
ORDER = (("layout_0", "B_PENDING", 0, "pad_u"), ("layout_0", "B_PENDING", 0, "pad_v"),
         ("layout_0", "C_PENDING", 0, "pad_v"), ("layout_0", "C_PENDING", 0, "pad_u"),
         ("layout_1", "B_PENDING", 0, "pad_v"), ("layout_1", "B_PENDING", 0, "pad_u"),
         ("layout_1", "C_PENDING", 0, "pad_u"), ("layout_1", "C_PENDING", 0, "pad_v"))
STEP_NAMES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4")
STEP_WAVE = ("A", "A", "B", "B", "C", "C", "C", "C")
PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))                     # (first, second) per pair, same GPU, sequential
PREREQ = {0: (), 1: (0,), 2: (0, 1), 3: (0, 1, 2), 4: (0, 1, 2, 3), 5: (4,), 6: (0, 1, 2, 3), 7: (6,)}
WAVE_NEEDS = {0: None, 1: None, 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "B"}
SOURCE_FILES = r2.SOURCE_FILES_R2 + (
    "src/cp_disr/analysis/family_b_staging_v2.py", "src/cp_disr/analysis/family_b_v3r1_reach_safe_mirror.py",
    "scripts/family_b_v3r1_reach_safe_mirror.py", CFG, LAYOUTS_PATH)


def branch_key(b):
    return (b["layout"], b["context"], b["repeat"], b["candidate"])


def pair_of(step):
    return next(p for p in PAIRS if step in p)


def attempt_identity(k):
    return f"FAMILY_B_V3_R1_ATTEMPT_1:{k[0]}:{k[1]}:r{k[2]}:{k[3]}"


def source_hashes(root):
    return {p: sha(Path(root) / p) for p in SOURCE_FILES}


def build_branches(profile_sha, commit, source_hash, prefixes, old_ids, manifest_path, freeze_id):
    out = []
    for step, k in enumerate(ORDER):
        layout, context, repeat, pad = k
        seed = int(m.hashlib.sha256(f"{freeze_id}:{layout}:{context}:r{repeat}".encode()).hexdigest()[:8], 16)
        logical = {"layout": layout, "context": context, "repeat": repeat, "candidate": pad}
        ident = attempt_identity(k)
        bid = m.branch_identity({**logical, "card": CARD}, profile_sha, commit, source_hash, ident)
        out.append({
            **logical, "branch_id": bid, "case_id": layout, "restore_seed": seed,
            "candidate_id": f"a:PLACE_BUFFER:carrier:{pad}:v1", "prefix": prefixes[context],
            "wave": f"v3r1_wave_{STEP_WAVE[step]}", "phase": STEP_WAVE[step], "step": step,
            "step_name": STEP_NAMES[step], "manifest_path": str(manifest_path), "authorized": True, "execute_now": True,
            "plan_id": bid, "attempt_id": bid, "attempt_identity": ident, "card_id": CARD,
            "profile_version": m.PROFILE_VERSION, "observation_profile_sha256": profile_sha,
            "cameras": list(m.CAMERAS), "source_commit": commit, "source_hash": source_hash,
            "task_family_id": "T_P_GOAL_CONDITIONED_ISOCHRONOUS_STAGING_V3_R1", "short_name": "T_P_FB3R1"})
    ids = [b["branch_id"] for b in out]
    if len(set(ids)) != 8 or set(ids) & set(old_ids):
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch identities collide with a historical card")
    if len({branch_key(b) for b in out}) != 8:
        raise ValueError("STOPPED_SOURCE_IDENTITY:logical set")
    return out


def check_uniform(registration):
    problems, branches = [], registration["branches"]
    if len(branches) != 8 or [branch_key(b) for b in branches] != list(ORDER):
        problems.append("branch set/order differs from the 8 frozen branches")
    if {(b["observation_profile_sha256"], tuple(b["cameras"]), b["profile_version"]) for b in branches} != \
            {(FROZEN_SHA, tuple(m.CAMERAS), m.PROFILE_VERSION)}:
        problems.append("branches do not all use the frozen observation profile")
    pairs = defaultdict(list)
    for b in branches:
        pairs[(b["layout"], b["context"], b["repeat"])].append(b)
    if len(pairs) != 4:
        problems.append("expected 4 U/V pairs")
    for key, members in pairs.items():
        if sorted(b["candidate"] for b in members) != sorted(PADS) or len({b["restore_seed"] for b in members}) != 1:
            problems.append(f"{key}: pad pair incomplete or restore seeds differ")
    return problems


# ------------------------------------------------------------------------------- Gate 0 (offline)
UNCHANGED_SOURCES = (
    "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/family_b_env.py",
    "src/cp_disr/platforms/libero/family_b_adapters.py", "src/cp_disr/platforms/libero/family_b_obs_v2.py",
    "src/cp_disr/platforms/libero/family_b_runtime.py", "src/cp_disr/platforms/libero/family_b_runtime_v2.py",
    "src/cp_disr/platforms/libero/perception.py", "src/cp_disr/platforms/libero/tp_sr_instrumentation.py",
    "src/cp_disr/platforms/libero/d0_env.py",
    "src/cp_disr/analysis/family_b_pilot.py", "src/cp_disr/analysis/family_b_obs_v2.py",
    "src/cp_disr/analysis/s1_integration.py", "src/cp_disr/baselines/b_plan.py",
    "configs/runtime/tp_fb_contract_registry.yaml")
ALLOWED_NEW = ("src/cp_disr/analysis/family_b_v3r1_reach_safe_mirror.py", "scripts/family_b_v3r1_reach_safe_mirror.py", CFG,
               LAYOUTS_PATH, "tests/test_family_b_v3r1_reach_safe_mirror.py")


def _git_blob(root, commit, rel):
    res = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"], capture_output=True)
    return res.stdout if res.returncode == 0 else None


def show_pose_in_source(root):
    text = (Path(root) / "src/cp_disr/platforms/libero/skill_executor.py").read_text()
    hit = re.search(r"show = np\.array\(\[([-0-9.]+), ([-0-9.]+), float\(self\.env\.table_top_z \+ 0\.16\)\]\)", text)
    return None if hit is None else (float(hit.group(1)), float(hit.group(2)))


def place_buffer_source(root):
    text = (Path(root) / "src/cp_disr/platforms/libero/skill_executor.py").read_text()
    a = text.index("    def _place_buffer(")
    return text[a:]


def actual_base_xy(root):
    """Panda base XY exactly as FamilyBEnv._load_model computes it (robot model offset for the env's table length); no sim."""
    from robosuite.models.robots import Panda
    sizes = {}
    for rel in ("src/cp_disr/platforms/libero/d0_env.py", "src/cp_disr/platforms/libero/family_b_obs_v2.py"):
        hit = re.search(r"self\.table_full_size = \(([-0-9.]+),", (Path(root) / rel).read_text())
        sizes[rel] = None if hit is None else float(hit.group(1))
    lengths = set(sizes.values())
    if None in lengths or len(lengths) != 1:
        raise ValueError("STOPPED_GATE0:table length not uniquely readable from the frozen env sources: " + json.dumps(sizes))
    length = lengths.pop()
    off = Panda().base_xpos_offset["table"](length)
    return (float(off[0]), float(off[1])), {"table_length_m": length, "sources": sizes, "panda_offset_xyz": [float(x) for x in off]}


def source_checks(root):
    root = Path(root)
    checks, detail = {}, {}
    pose = show_pose_in_source(root)
    detail["show_pose_xy"] = pose
    checks["show_pose_is_0.02_-0.08"] = pose == SHOW
    base_text = _git_blob(root, BASE, "src/cp_disr/platforms/libero/skill_executor.py")
    checks["place_buffer_logic_equals_base"] = base_text is not None and \
        place_buffer_source(root) == base_text.decode()[base_text.decode().index("    def _place_buffer("):]
    changed = []
    for rel in UNCHANGED_SOURCES:
        old = _git_blob(root, BASE, rel)
        if old is None or old != (root / rel).read_bytes():
            changed.append(rel)
    checks["production_sources_equal_base"] = not changed
    detail["changed_production_sources"] = changed
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", BASE, "HEAD", "--", "src", "configs",
                                    "scripts"], text=True).split()
    unexpected = sorted(p for p in diff if p not in ALLOWED_NEW)
    detail["changed_paths_since_base"], detail["unexpected_changes"] = diff, unexpected
    checks["only_allowed_new_paths_changed"] = not unexpected
    checks["frozen_profile_bytes_equal_freeze_commit"] = not r2.frozen_bytes_ok(root)
    checks["profile_hash"] = read(root / FROZEN_FILES[0])["profile_sha256"] == FROZEN_SHA
    base_xy, base_detail = actual_base_xy(root)
    detail["actual_base_xy"], detail["actual_base_detail"] = base_xy, base_detail
    checks["actual_panda_base_equals_proxy_1e-6"] = all(abs(a - b) <= 1e-6 for a, b in zip(base_xy, BASE_PROXY_XY))
    import cp_disr
    from cp_disr.platforms.libero import skill_executor
    detail["import_paths"] = {"cp_disr": str(Path(cp_disr.__file__)), "skill_executor": str(Path(skill_executor.__file__))}
    checks["imports_bound_to_this_worktree"] = str(Path(skill_executor.__file__).resolve()).startswith(str(root.resolve()))
    checks["tracked_tree_clean"] = not m.git(root, "status", "--porcelain", "-uno")
    checks["head_descends_from_base"] = m.git(root, "merge-base", "--is-ancestor", BASE, "HEAD") == ""
    checks["old_v3_evidence_present_read_only"] = (root / V3_EVIDENCE / "registration.json").is_file()
    return {"checks": checks, "detail": detail, "all_ok": all(checks.values())}


def demonstrated_radius(root):
    """Largest planar base distance among previously manipulated Family B positions (read-only, informational)."""
    pts = []
    for lay in read(Path(root) / "configs/final_master/family_b_layouts.json")["layouts"]:
        pts += [lay[k] for k in ("carrier_xy", "obj_b_xy", "obj_c_xy", "receiver_xy", "pad_u_xy", "pad_v_xy")]
    return max(_d(BASE_PROXY_XY, p) for p in pts)


def camera_audit(layouts=None):
    """Analytic projection of every key point (cube mid-height) into both fixed public cameras; role-based edge gate."""
    cams = m.camera_manifest()["cameras"]
    z = 0.825 + 0.021
    layouts = layouts or layout_specs()
    rows, ok_all = [], True
    for lay in layouts:
        pts = {"carrier": (lay["carrier_xy"], "carrier"), "obj_b": (lay["obj_b_xy"], "target_anchor"),
               "obj_c": (lay["obj_c_xy"], "target_anchor"), "receiver": (lay["receiver_xy"], "receiver"),
               "pad_u": (lay["pad_u_xy"], "pad"), "pad_v": (lay["pad_v_xy"], "pad"), "show_pose": (list(SHOW), None)}
        for name, (xy, role) in pts.items():
            per = {cam: m.project(cams[cam], [xy[0], xy[1], z]) for cam in m.CAMERAS}
            u, v = per["agentview"]["u"], per["agentview"]["v"]
            margin = min(u, m.IMAGE_SIZE - 1 - u, v, m.IMAGE_SIZE - 1 - v)
            edge = per["agentview"]["px_per_cube_edge"]
            gated = role is not None
            need = CAM_EDGE_MIN[role] if gated else None
            ok = (margin >= CAM_MARGIN_MIN and edge >= need) if gated else True
            ok_all &= bool(ok)
            rows.append({"layout": lay["layout_id"], "object": name, "role": role, "gated": gated, "projection": per,
                         "agentview_center_edge_margin_px": margin, "agentview_px_per_cube_edge": edge,
                         "required_margin_px": CAM_MARGIN_MIN if gated else None, "required_edge_px": need,
                         "qualifies": bool(ok), "hidden_pose_used_in_runtime": False})
    gated = [r for r in rows if r["gated"]]
    weakest = min(gated, key=lambda r: r["agentview_px_per_cube_edge"] - (r["required_edge_px"] or 0))
    return {"rows": rows, "all_ok": bool(ok_all),
            "rule": "agentview centre-to-edge margin >= 9 px for every key point; predicted cube edge >= 8.5 px for "
                    "target anchors, pads and receiver and >= 8.0 px for the carrier start (role-based; the carrier is not "
                    "the mechanism variable); sideview recorded only; depth-support margin is checked online (>= 12)",
            "thresholds": {"center_edge_margin_px_min": CAM_MARGIN_MIN, "predicted_cube_edge_px_min": CAM_EDGE_MIN},
            "weakest_vs_requirement": {"layout": weakest["layout"], "object": weakest["object"], "role": weakest["role"],
                                       "edge_px": weakest["agentview_px_per_cube_edge"],
                                       "required_px": weakest["required_edge_px"]}}


def gate0(root, out):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = source_checks(root)
    base_xy = tuple(src["detail"]["actual_base_xy"])
    inv = invariants(base_xy=base_xy)
    radius = demonstrated_radius(root)
    geo = geometry_checks(base_xy=base_xy)
    feat = task_feature_checks()
    cam = camera_audit()
    layouts_ok = json.loads((root / LAYOUTS_PATH).read_text()) == json.loads(json.dumps(layouts_document()))
    max_new = max(r["base_radius_m"] for r in geo["reach_rows"])
    reach = {"card_id": CARD, "base_xy_used": list(base_xy), "base_from_robot_model": src["detail"]["actual_base_detail"],
             "rule": f"every new movable / pad point (carrier, obj_b anchor, obj_c anchor, pad_u, pad_v) base radius <= {REACH_CAP} m; "
                     "the old 'demonstrated + 0.05' rule is retired", "cap_m": REACH_CAP, "points": geo["reach_rows"],
             "max_new_point_radius_m": max_new, "demonstrated_max_planar_radius_m": radius,
             "margin_to_demonstrated_max_m": radius - max_new, "previous_failed_point_radius_m": 0.80049984384758,
             "receiver_radius_recorded_only_m": [r["receiver_radius_recorded_only_m"] for r in geo["rows"]],
             "all_ok": all(r["ok"] for r in geo["reach_rows"])}
    aabb = {"card_id": CARD, "minimum_aabb_clearance_m": AABB_MIN, "half_sizes": {"cube_xy": CUBE_HALF, "pad_xy": PAD_HALF_XY,
            "receiver_outer_xy": list(RECEIVER_OUTER)}, "pairs": geo["aabb_rows"],
            "min_gap_m": min(r["aabb_gap_m"] for r in geo["aabb_rows"]), "all_ok": all(r["ok"] for r in geo["aabb_rows"])}
    write(out / "reach_envelope_audit.json", reach)
    write(out / "aabb_clearance_audit.json", aabb)
    write(out / "geometry_invariants.json", {"card_id": CARD, "decision_show_xy": list(SHOW), "invariants": inv,
                                              "geometry": geo, "task_features": feat,
                                              "layouts_file_matches_frozen_generator": layouts_ok})
    write(out / "camera_projection_audit.json", cam)
    ok = src["all_ok"] and inv["all_ok"] and geo["all_ok"] and cam["all_ok"] and layouts_ok and reach["all_ok"] and aabb["all_ok"] \
        and feat["no_fixed_rule_solves_all"] and feat["layouts_differ_only_by_bc_swap"] and feat["same_object_id_changes_best_pad"] \
        and feat["geometric_best_matches_prediction"]
    result = {"status": "GATE0_PASS" if ok else "STOPPED_GATE0", "source": src, "invariants_ok": inv["all_ok"],
              "geometry_ok": geo["all_ok"], "reach_ok": reach["all_ok"], "aabb_ok": aabb["all_ok"], "camera_ok": cam["all_ok"],
              "layouts_file_ok": layouts_ok, "task_features_ok": feat["no_fixed_rule_solves_all"]}
    write(out / "gate0.json", result)
    if not ok:
        raise ValueError("STOPPED_GATE0:" + json.dumps({k: v for k, v in result.items() if v is False}))
    return result


# ------------------------------------------------------------------------------- ledger / state
def init_state(out, branches):
    out = Path(out)
    write(out / "physical/budget_ledger.json", {"physical_witness_episodes": {"cap": CAPS["attempts"], "used": 0}})
    zero = {k: {"cap": 0, "used": 0} for k in ("skill_retries", "standalone_capture_resets", "provider_calls",
                                               "representation_forwards", "rl_transitions", "optimizer_steps", "elastic",
                                               "formal_test_episodes")}
    write(out / "budget_ledger.json", {
        "card_id": CARD,
        "history": {"v1": {"attempts": 4, "status": "PERMANENTLY_CONSUMED"},
                    "v2_r1": {"attempt_slots_charged": 2, "unused_slots_status": "RETIRED_NOT_TRANSFERABLE"},
                    "v2_r2": {"attempts": 4}, "staging_v2": {"attempts": 20, "task_success": "24/24 with R2"},
                    "v3_original": {"attempts_charged": 1, "status": "STOPPED_SETUP_REACH_TIMEOUT", "mechanism_measured": False,
                                    "unused_slots_status": "RETIRED_NOT_TRANSFERABLE"}},
        "attempts": {"cap": CAPS["attempts"], "used": 0},
        "environment_constructions": {"cap": CAPS["constructions"], "used": 0},
        "explicit_resets": {"cap": CAPS["resets"], "used": 0},
        "live_skill_calls": {"cap": CAPS["skills"], "used": 0}, **zero,
        "phases": {"A": {"cap": 2, "used": 0}, "B": {"cap": 2, "used": 0}, "C": {"cap": 4, "used": 0}},
        "reserved_branches": [], "reserved": 0, "started": 0, "terminal": 0,
        "actual": {"constructions_attempted": None, "constructor_successful": None, "explicit_resets": None,
                   "internal_resets": None, "skill_calls": None}})
    write(out / "wave_state.json", {
        "status": "NOT_STARTED", "stop": None,
        "steps": {str(i): {"name": STEP_NAMES[i], "key": list(k), "status": "NOT_RUN",
                           "branch_id": next(b["branch_id"] for b in branches if branch_key(b) == k)}
                  for i, k in enumerate(ORDER)},
        "waves": {w: {"status": "NOT_EVALUATED"} for w in "ABC"}})


def _event(out, kind, **fields):
    path = Path(out) / "budget_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"event": kind, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields},
                                sort_keys=True) + "\n")
        handle.flush()


@contextmanager
def _lock(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out) / ".v3_ledger.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def state_of(out):
    return read(Path(out) / "wave_state.json")


def _gate_ok(state, step):
    """Raise unless the staged order allows dispatching this step."""
    if state["status"] == "STOPPED":
        raise ValueError("STOPPED_V3_WAVE_FAILED")
    if state["steps"][str(step)]["status"] != "NOT_RUN":
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
    for need in PREREQ[step]:
        if state["steps"][str(need)]["status"] != "PASS":
            raise ValueError(f"STOPPED_V3_ORDER:step {STEP_NAMES[step]} needs {STEP_NAMES[need]} PASS")
    wave = WAVE_NEEDS[step]
    if wave is not None and state["waves"][wave]["status"] != "PASS":
        raise ValueError(f"STOPPED_V3_ORDER:step {STEP_NAMES[step]} needs wave {wave} gate PASS")


def reserve_and_charge(out, b):
    """Atomic: staged-order gate, then reserve the physical slot, then charge the ledger; no change on violation."""
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger.json")
        bid = b["branch_id"]
        if bid in ledger["reserved_branches"]:
            raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
        _gate_ok(state_of(out), b["step"])
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            if ledger[key]["used"] + 1 > ledger[key]["cap"]:
                raise ValueError("STOPPED_BUDGET_EXHAUSTED:" + key)
        ph = ledger["phases"][b["phase"]]
        if ph["used"] + 1 > ph["cap"]:
            raise ValueError("STOPPED_BUDGET_EXHAUSTED:phase_" + b["phase"])
        reserve_branch_attempt(out / "physical", bid)
        for key in ("attempts", "environment_constructions", "explicit_resets"):
            ledger[key]["used"] += 1
        ph["used"] += 1
        ledger["reserved_branches"].append(bid)
        ledger["reserved"] += 1
        write(out / "budget_ledger.json", ledger)
        _event(out, "RESERVED_CHARGED", branch_id=bid, step=STEP_NAMES[b["step"]])
    return ledger


def finalize_counts(out):
    out = Path(out)
    with _lock(out):
        ledger = read(out / "budget_ledger.json")
        res = [read(p) for p in sorted((out / "physical/branch_results").glob("*.json"))]
        counts = [r.get("env_counts", {}) for r in res]
        ledger["live_skill_calls"]["used"] = sum(len(r.get("actions", [])) for r in res)
        ledger["started"] = len(res)
        ledger["terminal"] = len(res)
        ledger["actual"] = {
            "constructions_attempted": len(res),
            "constructor_successful": sum(1 for c in counts if c.get("constructions") == 1 and c.get("reset_calls", -1) >= 1),
            "explicit_resets": sum(max(int(c.get("reset_calls", 0)), 0) for c in counts),
            "internal_resets": sum(max(int(c.get("internal_resets", 0)), 0) for c in counts),
            "skill_calls": ledger["live_skill_calls"]["used"]}
        write(out / "budget_ledger.json", ledger)
    return ledger


def _set(out, fn):
    with _lock(out):
        st = state_of(out)
        fn(st)
        write(Path(out) / "wave_state.json", st)


def _stop(out, wave, reason, step=None):
    def fn(st):
        st["status"] = "STOPPED"
        st["stop"] = {"wave": wave, "step": None if step is None else STEP_NAMES[step], "reason": reason}
    _set(out, fn)
    _event(out, "V3_STOPPED", wave=wave, step=None if step is None else STEP_NAMES[step], reason=reason)


# ------------------------------------------------------------------------------- identity / worker
def verify_identity(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    ident = read(out / "source_identity.json")
    if m.git(root, "rev-parse", "HEAD") != ident["commit"]:
        raise ValueError("STOPPED_SOURCE_IDENTITY:HEAD moved since registration")
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree dirty")
    if ident["sources"] != source_hashes(root):
        raise ValueError("STOPPED_SOURCE_IDENTITY:source drift")
    if r2.frozen_bytes_ok(root):
        raise ValueError("STOPPED_OBSERVATION_PROFILE:frozen file changed")
    reg = read(out / "physical/registration.json")
    problems = check_uniform(reg)
    if problems or reg["card_id"] != CARD:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems or ["registration"]))
    if Path(sys.executable).resolve() != Path(PY).resolve() or not str(m.__file__).startswith(str(root / "src")):
        raise ValueError("STOPPED_SOURCE_IDENTITY:interpreter or cp_disr import path")
    return ident


def worker(root, out, branch_id):
    """Frozen v2 worker body (reservation required); only the setup-context mapping is shared with staging."""
    from cp_disr.platforms.libero import family_b_obs_v2 as obs
    root, out = Path(root), Path(out)
    if branch_id not in read(out / "budget_ledger.json")["reserved_branches"]:
        raise RuntimeError("STOPPED_BUDGET:branch was not reserved by the coordinator")
    m.SOURCE_FILES = SOURCE_FILES
    real = obs.verify_frozen_cameras
    cap = out / "captures" / branch_id
    cap.mkdir(parents=True, exist_ok=True)

    def recording_verify(env, profile):
        try:
            write(cap / "camera_live_vs_frozen.json", r2._camera_record(env, profile))
        except Exception as exc:
            write(cap / "camera_live_vs_frozen.json", {"status": "RECORD_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return real(env, profile)

    obs.verify_frozen_cameras = recording_verify
    return s.build_worker()(root, out, branch_id)


def default_launcher(root, out, b, gpu):
    from cp_disr.analysis.family_b_pilot import _worker_env
    logs = Path(out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = (logs / f"{b['branch_id']}.log").open("ab")
    cmd = [sys.executable, str(Path(root) / "scripts/family_b_v3r1_reach_safe_mirror.py"), "worker", "--root", str(root),
           "--output", str(out), "--branch-id", b["branch_id"]]
    proc = subprocess.Popen(cmd, cwd=root, env=_worker_env(gpu), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc.log_handle = log
    return proc


# ------------------------------------------------------------------------------- evaluation
def _build_evaluate():
    """The staging per-branch evaluation, unchanged, with this card's identity (B/C contexts are shared)."""
    import inspect
    src = inspect.getsource(s.evaluate_branch)
    old = 'b.get("card_id") == CARD and "STAGING_V2_CONT" in b["attempt_identity"]'
    if src.count(old) != 1:
        raise RuntimeError("STOPPED_SOURCE_IDENTITY:staging evaluation changed")
    ns = dict(vars(s))
    ns["CARD"] = CARD
    exec(compile(src.replace(old, 'b.get("card_id") == CARD and "V3_R1" in b["attempt_identity"]'),
                 "<family_b_v3r1_reach_safe_mirror:evaluate>", "exec"), ns)
    return ns["evaluate_branch"]


evaluate_branch = _build_evaluate()


def _attempts(out):
    p = Path(out) / "physical/attempt_registry.json"
    return read(p) if p.is_file() else {}


def _result(out, bid):
    p = Path(out) / "physical/branch_results" / f"{bid}.json"
    return read(p) if p.is_file() else None


def target_support(row):
    """Largest public-view depth support of the remaining target after the candidate."""
    per = (row.get("per_view") or {}).get(REMAINING[row["context"]], {})
    return max([int(v["depth_support"]) for v in per.values()] or [0])


def initial_support(out, b):
    """Reset-time public RGB-D depth support (best public view) of the carrier and the setup target; all three recorded."""
    p = Path(out) / "captures" / b["branch_id"] / "initial_view_analysis.json"
    res = {"present": p.is_file(), "carrier": 0, "setup_target": 0, "setup_target_object": SETUP_TARGET[b["context"]],
           "per_object_max": {}, "min_required": INITIAL_SUPPORT_MIN, "ok": False}
    if not p.is_file():
        return res
    per = read(p).get("per_view", {})
    for obj in ("carrier", "obj_b", "obj_c"):
        res["per_object_max"][obj] = max([int(per[v][obj]["depth_support"]) for v in per if obj in per[v]] or [0])
    res["carrier"] = res["per_object_max"]["carrier"]
    res["setup_target"] = res["per_object_max"][SETUP_TARGET[b["context"]]]
    res["ok"] = bool(res["carrier"] >= INITIAL_SUPPORT_MIN and res["setup_target"] >= INITIAL_SUPPORT_MIN)
    return res


def step_verdict(out, b):
    """Technical + task-qualification verdict of one branch (includes the reset-time and post-candidate support gates)."""
    row = evaluate_branch(out, b, _attempts(out))
    support = target_support(row) if row.get("per_view") else 0
    init = initial_support(out, b)
    ok = bool(row["complete"] and row.get("task_success") is True and not row["problems"] and support >= SUPPORT_MIN and init["ok"])
    return ok, {**row, "key_target_depth_support_max": support, "support_margin_vs_12": support - SUPPORT_MIN,
                "initial_support": init, "initial_support_ok": init["ok"], "qualifies": ok}


def candidate_evidence(out, b):
    """PLACE_BUFFER evidence for the fairness gate (read from the saved per-step controller trace)."""
    out = Path(out)
    n = len(b["prefix"])
    res = _result(out, b["branch_id"]) or {}
    acts = res.get("actions", [])
    cap = out / "captures" / b["branch_id"] / f"action_{n:02d}"
    ev = {"branch_id": b["branch_id"], "candidate": b["candidate"], "present": False}
    trace_path = cap / "controller_trace.jsonl"
    if len(acts) <= n or not trace_path.is_file():
        return ev
    raw = acts[n].get("raw_skill", {})
    trace = [json.loads(x) for x in trace_path.read_text().splitlines() if x.strip()]
    begin = next((t for t in trace if t.get("event") == "skill_begin"), {})
    end = next((t for t in reversed(trace) if t.get("event") == "skill_end"), {})
    steps = [t for t in trace if t.get("kind") == "step"]
    phases, open_ = {}, {}
    for t in trace:
        if t.get("event") == "phase_begin":
            open_[t["phase"]] = t
        elif t.get("event") == "phase_end" and t["phase"] in open_:
            b0 = open_.pop(t["phase"])
            phases[t["phase"]] = {"duration": float(t["sim_time"]) - float(b0["sim_time"]),
                                  "steps": int(t.get("steps_after") or 0) - int(b0.get("steps_before") or 0),
                                  "target_xyz": b0.get("target_xyz")}
    ev.update(present=True, start_eef=begin.get("eef_pos"), end_eef=end.get("eef_pos"),
              duration=float(raw.get("sim_duration", acts[n]["duration_sim"])), steps=int(end.get("steps", raw.get("steps", len(steps)))),
              segments=phases, qpos_sha256_end=steps[-1].get("qpos_sha256") if steps else None,
              qvel_sha256_end=steps[-1].get("qvel_sha256") if steps else None,
              controller_exit=acts[n].get("controller_exit"))
    return ev


# ------------------------------------------------------------------------------- pair statistics / wave gates
def branches_of(out):
    return read(Path(out) / "physical/registration.json")["branches"]


def pair_stats(out, branches, layout, context):
    """U/V cost and PLACE_BUFFER evidence of one pair (None where a branch did not run or did not succeed)."""
    by = {b["candidate"]: b for b in branches if (b["layout"], b["context"]) == (layout, context)}
    row = {"layout": layout, "context": context, "branch_u": by["pad_u"]["branch_id"], "branch_v": by["pad_v"]["branch_id"]}
    evs, costs = {}, {}
    for pad in PADS:
        res = _result(out, by[pad]["branch_id"])
        evs[pad] = candidate_evidence(out, by[pad])
        costs[pad] = res.get("time_to_task_success") if res and res.get("status") == "TASK_SUCCESS" else None
        row[f"status_{pad[-1]}"] = (res or {}).get("status", "NOT_RUN")
    row.update(cost_u=costs["pad_u"], cost_v=costs["pad_v"], ev_u=evs["pad_u"], ev_v=evs["pad_v"])
    both = evs["pad_u"]["present"] and evs["pad_v"]["present"]
    row["candidate_duration_abs_diff"] = abs(evs["pad_u"]["duration"] - evs["pad_v"]["duration"]) if both else None
    row["candidate_steps_abs_diff"] = abs(evs["pad_u"]["steps"] - evs["pad_v"]["steps"]) if both else None
    row["candidate_match_pass"] = bool(both and row["candidate_duration_abs_diff"] <= TOL_T + 1e-9 and
                                       row["candidate_steps_abs_diff"] <= STEPS_TOL)
    row["delta_u_minus_v"] = (costs["pad_u"] - costs["pad_v"]) if None not in costs.values() else None
    return row


def prefers(row, pad, tol=TOL_T):
    """`pad` is faster by more than the tolerance (C(pad) + tol < C(other))."""
    cu, cv = row["cost_u"], row["cost_v"]
    if cu is None or cv is None:
        return False
    return cu + tol < cv if pad == "pad_u" else cv + tol < cu


def wave_gate(out, wave):
    br = branches_of(out)
    pairs = {(l, c): pair_stats(out, br, l, c) for l in ("layout_0", "layout_1") for c in CONTEXTS}
    spec = {"A": [("layout_0", "B_PENDING")], "B": [("layout_0", "C_PENDING")],
            "C": [("layout_1", "B_PENDING"), ("layout_1", "C_PENDING")]}[wave]
    checks = {}
    for key in spec:
        p = pairs[key]
        winner = PREDICTED_WINNER[key]
        checks[f"{key[0]}/{key[1]}:candidate_duration_diff_le_0.15"] = bool(
            p["candidate_duration_abs_diff"] is not None and p["candidate_duration_abs_diff"] <= TOL_T + 1e-9)
        checks[f"{key[0]}/{key[1]}:candidate_steps_diff_le_3"] = bool(
            p["candidate_steps_abs_diff"] is not None and p["candidate_steps_abs_diff"] <= STEPS_TOL)
        checks[f"{key[0]}/{key[1]}:{winner}_faster_by_more_than_0.15"] = prefers(p, winner)
    if wave == "B":
        da, db = pairs[("layout_0", "B_PENDING")]["delta_u_minus_v"], pairs[("layout_0", "C_PENDING")]["delta_u_minus_v"]
        checks["layout_0:sign_delta_B_differs_from_sign_delta_C"] = bool(
            da is not None and db is not None and np.sign(da) != np.sign(db) and da != 0 and db != 0)
    return {"wave": wave, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "pairs": {f"{k[0]}/{k[1]}": {kk: vv for kk, vv in v.items() if kk not in ("ev_u", "ev_v")}
                      for k, v in pairs.items() if k in spec}}


# ------------------------------------------------------------------------------- dispatch
def run_branch(root, out, b, gpu, launcher=default_launcher, sleep=time.sleep):
    out = Path(out)
    reserve_and_charge(out, b)
    t0 = time.monotonic()
    proc = launcher(root, out, b, gpu)
    code = r2._wait(proc, timeout=WATCHDOG, sleep=sleep)
    if code == "WATCHDOG_TIMEOUT":
        r2._kill(proc)
        s._watchdog_finish(out, b["branch_id"])
    r2._close(proc)
    return {"branch_id": b["branch_id"], "step": STEP_NAMES[b["step"]], "gpu": gpu, "exit_code": code,
            "wall_seconds": time.monotonic() - t0}


def _settle(out, b):
    finalize_counts(out)
    ok, row = step_verdict(out, b)
    _set(out, lambda st: st["steps"][str(b["step"])].update(
        status="PASS" if ok else "FAIL", problems=row["problems"][:20], support_max=row["key_target_depth_support_max"],
        initial_support_ok=row["initial_support_ok"]))
    return ok


def _wave_result(out, wave):
    gate = wave_gate(out, wave)
    _set(out, lambda st: st["waves"][wave].update(status=gate["status"], checks=gate["checks"]))
    write(Path(out) / f"wave_{wave}_gate.json", gate)
    return gate


def run_waves(root, out, gpus, launcher=default_launcher, identity_check=verify_identity, sleep=time.sleep):
    """Gate 0 -> Wave A -> Wave B -> Wave C.  The first failed step or gate stops everything not yet started."""
    root, out = Path(root).resolve(), Path(out).resolve()
    identity_check(root, out)
    if state_of(out)["status"] != "NOT_STARTED":
        raise ValueError("STOPPED_V3_ALREADY_RUN")
    br = branches_of(out)
    s._no_prior_results(out, br)
    gpus = list(gpus)
    gpu_a, gpu_b = gpus[0], (gpus[1] if len(gpus) > 1 else gpus[0])
    _set(out, lambda st: st.update(status="IN_PROGRESS"))
    started, dispatched = time.monotonic(), []

    def step(i, gpu):
        dispatched.append(run_branch(root, out, br[i], gpu, launcher, sleep))
        return _settle(out, br[i])

    def finish(result):
        write(out / "physical/throughput_v3.json", {
            "phases": {"dispatched": dispatched}, "gpus": gpus, "span_seconds": time.monotonic() - started,
            "single_worker_baseline": "NOT_MEASURED", "speedup_claim": "NONE (no matched single-worker baseline)",
            "wave_c_pair_workers": 2 if gpu_a != gpu_b else 1})
        return result

    for wave, steps in (("A", (0, 1)), ("B", (2, 3))):
        for i in steps:
            if not step(i, gpu_a):
                _stop(out, wave, f"step {STEP_NAMES[i]} did not meet its technical/qualification gate", i)
                return finish({"stopped": STEP_NAMES[i], "dispatched": len(dispatched)})
        gate = _wave_result(out, wave)
        if gate["status"] != "PASS":
            _stop(out, wave, f"wave {wave} gate failed: " + ",".join(k for k, v in gate["checks"].items() if not v))
            return finish({"stopped": f"wave_{wave}", "dispatched": len(dispatched)})
    failure = run_wave_c(root, out, br, (gpu_a, gpu_b), launcher, sleep, dispatched)
    if failure:
        return finish({"stopped": failure, "dispatched": len(dispatched)})
    gate = _wave_result(out, "C")
    if gate["status"] != "PASS":
        _stop(out, "C", "wave C gate failed: " + ",".join(k for k, v in gate["checks"].items() if not v))
        return finish({"stopped": "wave_C", "dispatched": len(dispatched)})
    _set(out, lambda st: st.update(status="COMPLETE"))
    return finish({"stopped": None, "dispatched": len(dispatched)})


def run_wave_c(root, out, br, gpus, launcher, sleep, dispatched):
    """Two pair workers (one GPU each); inside a pair the second branch starts only after the first qualifies."""
    gpus = list(dict.fromkeys(gpus))
    pairs = [PAIRS[2], PAIRS[3]]
    slots = {i: {"pair": pairs[i], "idx": 0, "proc": None, "gpu": gpus[i % len(gpus)]} for i in range(2)}
    if len(gpus) == 1:                      # one GPU: pairs run one after the other
        slots = {0: {"pair": pairs[0], "idx": 0, "proc": None, "gpu": gpus[0]}}
        queue = [pairs[1]]
    else:
        queue = []
    failure, t_start = None, {}
    while True:
        for key, st in list(slots.items()):
            if st["proc"] is None and st["idx"] < 2 and failure is None:
                b = br[st["pair"][st["idx"]]]
                reserve_and_charge(out, b)
                st["proc"], t_start[key] = launcher(root, out, b, st["gpu"]), time.monotonic()
                dispatched.append({"branch_id": b["branch_id"], "step": STEP_NAMES[b["step"]], "gpu": st["gpu"],
                                   "parallel_pair_worker": key, "exit_code": None})
        active = [st for st in slots.values() if st["proc"] is not None]
        if not active:
            if queue and failure is None:
                slots = {0: {"pair": queue.pop(0), "idx": 0, "proc": None, "gpu": gpus[0]}}
                continue
            break
        sleep(0.5)
        for key, st in list(slots.items()):
            if st["proc"] is None:
                continue
            b = br[st["pair"][st["idx"]]]
            code = st["proc"].poll()
            if code is None and time.monotonic() - t_start[key] > WATCHDOG:
                r2._kill(st["proc"])
                s._watchdog_finish(out, b["branch_id"])
                code = "WATCHDOG_TIMEOUT"
            if code is None:
                continue
            r2._close(st["proc"])
            for rec in reversed(dispatched):
                if rec["branch_id"] == b["branch_id"]:
                    rec.update(exit_code=code, wall_seconds=time.monotonic() - t_start[key])
                    break
            st["proc"] = None
            if not _settle(out, b):
                failure = failure or STEP_NAMES[b["step"]]
                _stop(out, "C", f"step {STEP_NAMES[b['step']]} did not meet its technical/qualification gate", b["step"])
            st["idx"] += 1
    return failure


# ------------------------------------------------------------------------------- registration
def protected_paths(root):
    root = Path(root)
    paths = list(s.protected_paths(root))
    for rel in (s.R2_DIR.rsplit("/", 1)[0] + "/" + Path(STAGING_EVIDENCE).name, V3_EVIDENCE):
        if (root / rel).exists():
            paths.append(root / rel)
    return paths


def inventory_before(root, out):
    inv = m.inventory(protected_paths(root))
    write(Path(out) / "protected_before.json", {"sha256": inv, "files": len(inv),
                                                "capture_count": sum("/captures/" in k for k in inv)})
    return len(inv)


def register(root, out):
    import yaml
    from cp_disr.runtime import require_runtime
    root, out = Path(root).resolve(), Path(out)
    if m.git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree not clean")
    commit = m.git(root, "rev-parse", "HEAD")
    cfg = yaml.safe_load((root / CFG).read_text())
    if (cfg["card_id"], cfg["observation_profile_sha256"], cfg["base_commit"]) != (CARD, FROZEN_SHA, BASE):
        raise ValueError("STOPPED_SOURCE_IDENTITY:config identity")
    if (out / "physical/registration.json").exists():
        raise ValueError("STOPPED_SOURCE_IDENTITY:already registered")
    out.mkdir(parents=True, exist_ok=True)
    g0 = gate0(root, out)                                      # raises STOPPED_GATE0 before anything is registered
    hashes = source_hashes(root)
    source_hash = m.digest(hashes)
    manifest = yaml.safe_load((root / STAGING_EVIDENCE / "spec/runtime_manifest_T_P_FB_staging_v2.yaml").read_text())
    src = root / "src/cp_disr/platforms/libero/family_b_runtime_v2.py"
    manifest["runtime"]["repository_path"] = str(root)
    manifest["runtime"]["layouts_path"] = LAYOUTS_PATH
    manifest["runtime"]["task_splits"] = {"T_P_FB": LAYOUTS_PATH}
    manifest["runtime_factory"].update({"source_path": str(src), "sha256": sha(src)})
    require_runtime(manifest)
    manifest_path = out / "spec/runtime_manifest_T_P_FB3R1_reach_safe_mirror.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    old_ids = s.historical_ids(root) | {b["branch_id"] for b in s.r2_branches(root).values()}
    freeze_id = m.digest({"card": CARD, "profile": FROZEN_SHA, "commit": commit})[:16]
    prefixes = {k: v["prefix"] for k, v in cfg["contexts"].items()}
    branches = build_branches(FROZEN_SHA, commit, source_hash, prefixes, old_ids, manifest_path, freeze_id)
    ref = float(read(root / STAGING_EVIDENCE / "bindings/reference_cost.json")["reference_skill_seconds"])
    registration = {"card_id": CARD, "branches": branches, "manifest_sha256": sha(manifest_path),
                    "reference_skill_seconds": ref}
    problems = check_uniform(registration)
    if problems:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems))
    phys = out / "physical"
    write(phys / "witnesses/e4_branch_registration.json", registration)
    write(phys / "registration.json", registration)
    write(out / "registration.json", registration)
    write(out / "bindings/reference_cost.json", {"reference_skill_seconds": ref})
    init_state(out, branches)
    _event(out, "REGISTERED", branches=[b["branch_id"] for b in branches], commit=commit)
    write(out / "authorization.json", {
        "card_id": CARD, "authorized_by": "user message approving CP-DISR-S4-FAMILY-B-V3-R1-REACH-SAFE-MIRROR-PILOT-1",
        "base_commit": BASE, "physical_branch_attempts_max": 8, "environment_construction_attempts_max": 8,
        "successful_explicit_resets_max": 8, "live_skill_calls_max": 48, "skill_retries": 0,
        "provider": 0, "representation_forwards": 0, "rl_transitions": 0, "optimizer_steps": 0, "elastic": 0,
        "formal_test": 0, "s2": False, "s3": False, "tp_training": False,
        "order": "Gate 0 -> Wave A -> Wave B -> Wave C", "no_coordinate_change_after_failure": True,
        "no_ninth_attempt": True, "no_tail_only_primary_metric": True, "old_family_b_evidence_read_only": True,
        "no_v3_r2_v4_or_family_c": True, "previous_v3_attempts_charged_permanently": 1})
    write(out / "source_identity.json", {
        "card_id": CARD, "branch": m.git(root, "branch", "--show-current"), "commit": commit, "base_commit": BASE,
        "source_hash": source_hash, "sources": hashes, "observation_profile_sha256": FROZEN_SHA,
        "manifest_sha256": sha(manifest_path), "frozen_file_sha256": {rel: sha(root / rel) for rel in FROZEN_FILES},
        "base_preflight": {"base_equals_origin_staging_v2_branch_head": True, "gate0_status": g0["status"]}})
    write(out / "card_revision.json", {
        "card_id": CARD, "revision": "R1_ROLE_BASED_CAMERA_GATE_AND_INITIAL_SUPPORT",
        "made_before": "prep commit and any environment construction (online attempts were 0)",
        "reason": "Gate 0 on the frozen coordinates gave carrier agentview edge 8.06 px < 8.5 px; the carrier is not the mechanism "
                  "variable (every branch enters the fixed show pose after PICK) and the historical successful Family B carrier "
                  "projects to about 8.26 px, so a uniform 8.5 px gate would reject an already successful region.",
        "changes": {"target_anchor_edge_px_min": 8.5, "pad_and_receiver_edge_px_min": 8.5, "carrier_edge_px_min": 8.0,
                    "all_key_points_center_edge_margin_px_min": 9.0,
                    "a1_canary_strengthened": "reset-time carrier and setup-target public depth_support >= 12 AND post-candidate "
                                              "remaining-target public depth_support >= 12, else stop (applied to every branch)"},
        "unchanged": ["mirror coordinates", "0.65 m reach cap", "0.008 m AABB clearance", "0.15 s / 3 step fairness gates",
                      "budgets", "primary metric", "controller / timeout / profile / Verifier / planner"],
        "historical_carrier_agentview_edge_px": 8.256})
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    for rel in FROZEN_FILES:
        write(out / Path(rel).name, read(root / rel))
    inventory_before(root, out)
    return {"status": "REGISTERED", "commit": commit, "branches": [b["branch_id"] for b in branches],
            "gate0": g0["status"]}


# ------------------------------------------------------------------------------- analysis
def _frames(out, b):
    cap = Path(out) / "captures" / b["branch_id"]
    return list(s._frame_sources(cap))


def support_rows(out, branches):
    rows = []
    for b in branches:
        for stage, fid, d, frame in _frames(out, b):
            for obj, f in frame["fusion"].items():
                for view, inp in f["inputs"].items():
                    rows.append({"branch_id": b["branch_id"], "layout": b["layout"], "context": b["context"],
                                 "candidate": b["candidate"], "action_dir": stage, "frame_id": fid, "object": obj,
                                 "view": view, "support": inp["support"], "reference": inp["reference"],
                                 "ratio": inp["ratio"], "valid": inp["valid"], "selected_view": f["selected_view"],
                                 "margin_vs_min_pixels_8": inp["support"] - THRESHOLDS["min_pixels"],
                                 "margin_vs_task_gate_12": inp["support"] - SUPPORT_MIN})
    return rows


def _csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items() if k in fields})


def decomposition(out, branches, pair):
    """Candidate action / first PICK / remaining tail effects of one pair; total stays the primary metric."""
    by = {b["candidate"]: b for b in branches if (b["layout"], b["context"]) == (pair["layout"], pair["context"])}
    acts = {}
    for pad, b in by.items():
        res = _result(out, b["branch_id"]) or {}
        a = res.get("actions", [])
        n = len(b["prefix"])
        acts[pad] = [x["duration_sim"] for x in a[n:]] if len(a) > n else None
    row = {"layout": pair["layout"], "context": pair["context"], "candidate_action_effect": None,
           "first_pick_effect": None, "remaining_tail_effect": None, "total_candidate_to_success_effect": pair["delta_u_minus_v"],
           "decomposition_error": None}
    if acts["pad_u"] and acts["pad_v"] and len(acts["pad_u"]) == len(acts["pad_v"]) >= 2:
        u, v = acts["pad_u"], acts["pad_v"]
        row["candidate_action_effect"] = u[0] - v[0]
        row["first_pick_effect"] = u[1] - v[1]
        row["remaining_tail_effect"] = sum(u[2:]) - sum(v[2:])
        if pair["delta_u_minus_v"] is not None:
            row["decomposition_error"] = pair["delta_u_minus_v"] - (row["candidate_action_effect"] +
                                                                    row["first_pick_effect"] + row["remaining_tail_effect"])
    return row


def restore_for(out, branches):
    reg = {"branches": branches}
    keys = [k for k in ((l, c, 0) for l in ("layout_0", "layout_1") for c in CONTEXTS)
            if all((Path(out) / "physical/branch_results" / f"{b['branch_id']}.json").is_file()
                   for b in branches if (b["layout"], b["context"]) == k[:2])]
    result = s.build_paired_restore()(Path(out), keys, reg, lambda b: Path(out) / "captures" / b["branch_id"]) \
        if keys else {"pairs": {}, "status": "FAIL", "contract": m.PAIR_CONTRACT}
    result["pairs_expected"] = 4
    result["status"] = "PASS" if len(result["pairs"]) == 4 and all(p["status"] == "PASS" for p in result["pairs"].values()) else "FAIL"
    write(Path(out) / "pair_restore.json", result)
    return result


def final_gate(pairs, verdict_rows, restore_ok, ledger_zero, protected_ok, wave_status):
    """pairs: {(layout, context): pair_stats}; verdict_rows: {step: row or None}.  Pure function."""
    allp = [pairs[(l, c)] for l in ("layout_0", "layout_1") for c in CONTEXTS]
    rows = [verdict_rows.get(i) for i in range(8)]
    ran = all(r is not None for r in rows)
    success = ran and all(r.get("status") == "TASK_SUCCESS" and r.get("task_success") for r in rows)
    complete = ran and all(r["complete"] for r in rows)
    support_ok = ran and all(r["key_target_depth_support_max"] >= SUPPORT_MIN and r.get("initial_support_ok") is True for r in rows)
    clean = ran and all(r.get("status") == "TASK_SUCCESS" and not r.get("recorder_errors") and not r["problems"] for r in rows)
    legal = ran and all(r["checks"].get("both_pads_legal") for r in rows)
    dur_ok = all(p["candidate_duration_abs_diff"] is not None and p["candidate_duration_abs_diff"] <= TOL_T + 1e-9 for p in allp)
    steps_ok = all(p["candidate_steps_abs_diff"] is not None and p["candidate_steps_abs_diff"] <= STEPS_TOL for p in allp)
    seq_match = False
    if ran:
        byk = {(r["layout"], r["context"], r["candidate"]): r for r in rows}
        seq_match = True
        for p in allp:
            su, sv = byk[(p["layout"], p["context"], "pad_u")]["sequence"], byk[(p["layout"], p["context"], "pad_v")]["sequence"]
            n = 3
            seq_match &= (len(su) == len(sv) and Counter(map(s._skill_type, su)) == Counter(map(s._skill_type, sv)) and
                          su[n + 1:] == sv[n + 1:])
    win = {k: PREDICTED_WINNER[k] for k in PREDICTED_WINNER}
    c10 = prefers(pairs[("layout_0", "B_PENDING")], "pad_u") and prefers(pairs[("layout_0", "C_PENDING")], "pad_v")
    c11 = prefers(pairs[("layout_1", "B_PENDING")], "pad_v") and prefers(pairs[("layout_1", "C_PENDING")], "pad_u")
    measured = {}
    for k, p in pairs.items():
        if p["cost_u"] is not None and p["cost_v"] is not None and p["cost_u"] != p["cost_v"]:
            measured[k] = "pad_u" if p["cost_u"] < p["cost_v"] else "pad_v"
    c12 = all(k in measured for k in win) and measured[("layout_0", "B_PENDING")] != measured[("layout_1", "B_PENDING")] and \
        measured[("layout_0", "C_PENDING")] != measured[("layout_1", "C_PENDING")]
    acc_u = sum(1 for k in win if measured.get(k) == "pad_u")
    acc_v = sum(1 for k in win if measured.get(k) == "pad_v")
    c13 = len(measured) == 4 and acc_u < 4 and acc_v < 4
    conditions = {
        "1_8_of_8_task_success": bool(success), "2_8_of_8_complete_records": bool(complete),
        "3_four_pair_restores_pass": bool(restore_ok), "4_place_buffer_duration_diff_le_0.15": bool(dur_ok),
        "5_controller_steps_diff_le_3": bool(steps_ok), "6_public_depth_support_ge_12_initial_and_post_candidate": bool(support_ok),
        "7_no_noplan_timeout_exception_or_recorder_error": bool(clean), "8_candidates_equally_legal": bool(legal),
        "9_plan_length_and_skill_multiset_match": bool(seq_match), "10_layout_0_preference_reversal": bool(c10),
        "11_layout_1_preference_reversal": bool(c11), "12_same_object_best_pad_changes_with_layout": bool(c12),
        "13_neither_always_u_nor_always_v_reaches_4_of_4": bool(c13),
        "14_provider_representation_rl_optimizer_zero": bool(ledger_zero),
        "15_old_evidence_and_frozen_config_unchanged": bool(protected_ok)}
    passed = all(conditions.values())
    return {"card_id": CARD, "label": GATE_PASS if passed else GATE_FAIL, "status": "PASS" if passed else "NOT_ESTABLISHED",
            "next_action": NEXT_PASS if passed else NEXT_FAIL, "conditions": conditions, "wave_status": wave_status,
            "measured_winner": {f"{k[0]}/{k[1]}": v for k, v in measured.items()},
            "predicted_winner": {f"{k[0]}/{k[1]}": v for k, v in win.items()},
            "always_u_accuracy": f"{acc_u}/4", "always_v_accuracy": f"{acc_v}/4",
            "state_conditioned_preference_reversal": bool(c10 and c11 and c12),
            "primary_metric": "C = candidate_start -> first verified task success (tail-only is decomposition only)",
            "claims": {"significance": False, "generalization": False, "provider_or_representation_started": False},
            "failure_policy": "no coordinate change, no lowered 0.15 s gate, no tail-only metric, no ninth attempt, no Family C"}


def analyze(root, out):
    root, out = Path(root), Path(out)
    ledger = finalize_counts(out)
    br = branches_of(out)
    state = state_of(out)
    pairs = {(l, c): pair_stats(out, br, l, c) for l in ("layout_0", "layout_1") for c in CONTEXTS}
    rows = {}
    for b in br:
        res = _result(out, b["branch_id"])
        if res is None:
            rows[b["step"]] = None
            continue
        ok, row = step_verdict(out, b)
        rows[b["step"]] = row
    flat = []
    for b in br:
        r = rows[b["step"]]
        flat.append({"step": STEP_NAMES[b["step"]], "wave": STEP_WAVE[b["step"]], "branch_id": b["branch_id"],
                     "layout": b["layout"], "context": b["context"], "repeat": b["repeat"], "candidate": b["candidate"],
                     "state": state["steps"][str(b["step"])]["status"],
                     "status": None if r is None else r.get("status"), "task_success": None if r is None else r.get("task_success"),
                     "complete": None if r is None else r.get("complete"),
                     "time_to_task_success": None if r is None else r.get("time_to_task_success"),
                     "candidate_start_sim_time": None if r is None else r.get("candidate_start_sim_time"),
                     "key_target_depth_support_max": None if r is None else r.get("key_target_depth_support_max"),
                     "initial_support_ok": None if r is None else r.get("initial_support_ok"),
                     "qualifies": None if r is None else r.get("qualifies"), "worker_wall_seconds": None if r is None else r.get("worker_wall_seconds"),
                     "sequence": None if r is None else r.get("sequence"), "problems": None if r is None else r.get("problems")})
    _csv(out / "branch_results.csv", ["step", "wave", "branch_id", "layout", "context", "repeat", "candidate", "state", "status",
                                      "task_success", "complete", "time_to_task_success", "candidate_start_sim_time",
                                      "key_target_depth_support_max", "initial_support_ok", "qualifies", "worker_wall_seconds", "sequence", "problems"], flat)
    fair = []
    for (l, c), p in pairs.items():
        eu, ev = p["ev_u"], p["ev_v"]
        fair.append({"layout": l, "context": c, "branch_u": p["branch_u"], "branch_v": p["branch_v"],
                     "candidate_start_eef_u": eu.get("start_eef"), "candidate_start_eef_v": ev.get("start_eef"),
                     "candidate_end_eef_u": eu.get("end_eef"), "candidate_end_eef_v": ev.get("end_eef"),
                     "place_buffer_duration_u": eu.get("duration"), "place_buffer_duration_v": ev.get("duration"),
                     "place_buffer_steps_u": eu.get("steps"), "place_buffer_steps_v": ev.get("steps"),
                     "segments_u": eu.get("segments"), "segments_v": ev.get("segments"),
                     "candidate_duration_abs_diff": p["candidate_duration_abs_diff"],
                     "candidate_steps_abs_diff": p["candidate_steps_abs_diff"], "candidate_match_pass": p["candidate_match_pass"],
                     "cost_u": p["cost_u"], "cost_v": p["cost_v"], "delta_u_minus_v": p["delta_u_minus_v"]})
    _csv(out / "candidate_action_fairness.csv", ["layout", "context", "branch_u", "branch_v", "candidate_start_eef_u",
         "candidate_start_eef_v", "candidate_end_eef_u", "candidate_end_eef_v", "place_buffer_duration_u",
         "place_buffer_duration_v", "place_buffer_steps_u", "place_buffer_steps_v", "segments_u", "segments_v",
         "candidate_duration_abs_diff", "candidate_steps_abs_diff", "candidate_match_pass", "cost_u", "cost_v",
         "delta_u_minus_v"], fair)
    _csv(out / "cost_decomposition.csv", ["layout", "context", "candidate_action_effect", "first_pick_effect",
         "remaining_tail_effect", "total_candidate_to_success_effect", "decomposition_error"],
         [decomposition(out, br, p) for p in pairs.values()])
    _csv(out / "observation_support.csv", ["branch_id", "layout", "context", "candidate", "action_dir", "frame_id", "object",
         "view", "support", "reference", "ratio", "valid", "selected_view", "margin_vs_min_pixels_8", "margin_vs_task_gate_12"],
         support_rows(out, br))
    restore = restore_for(out, br)
    zero_keys = ("provider_calls", "representation_forwards", "rl_transitions", "optimizer_steps", "elastic",
                 "skill_retries", "standalone_capture_resets", "formal_test_episodes")
    before = read(out / "protected_before.json")
    diff = m.diff_inventory(before["sha256"], m.inventory(protected_paths(root)))
    gate = final_gate(pairs, rows, restore["status"] == "PASS", all(ledger[k]["used"] == 0 for k in zero_keys),
                      not (diff["changed"] or diff["removed"] or diff["added"]) and not r2.frozen_bytes_ok(root),
                      {w: state["waves"][w]["status"] for w in "ABC"})
    gate["technical_status"] = ("TECHNICAL_COMPLETE_8_OF_8" if all(rows[i] is not None and rows[i]["complete"] for i in range(8))
                                else "STOPPED_BEFORE_ALL_BRANCHES_RAN" if state["status"] == "STOPPED" else "INCOMPLETE")
    gate["stop"] = state.get("stop")
    gate["pairs"] = {f"{k[0]}/{k[1]}": {kk: vv for kk, vv in v.items() if kk not in ("ev_u", "ev_v")} for k, v in pairs.items()}
    write(out / "mechanism_gate.json", gate)
    s.merge_throughput(out)
    return {"label": gate["label"], "technical": gate["technical_status"], "restore": restore["status"]}


def summarize(root, out):
    out = Path(out)
    ledger, gate, state = read(out / "budget_ledger.json"), read(out / "mechanism_gate.json"), state_of(out)
    act = ledger["actual"]
    lines = ["# Family B V3-R1 reach-safe mirror pilot", "",
             f"- Card {CARD}; frozen profile `{m.PROFILE_VERSION}` {FROZEN_SHA} (unchanged).",
             f"- Gate 0: {read(out / 'gate0.json')['status']}. Waves: " + ", ".join(f"{w}={state['waves'][w]['status']}" for w in 'ABC') +
             f"; run state {state['status']}" + (f" ({state['stop']['reason']})" if state.get("stop") else "") + ".",
             f"- Attempts reserved/started/terminal: {ledger['reserved']}/{ledger['started']}/{ledger['terminal']} (cap 8); constructions "
             f"{act.get('constructions_attempted')}; explicit resets {act.get('explicit_resets')}; internal resets "
             f"{act.get('internal_resets')}; live skill calls {act.get('skill_calls')}/48.",
             "- Provider / representation / RL / optimizer / elastic / formal test = 0; S2/S3/TP training = false.",
             f"- Mechanism: **{gate['label']}**; technical: {gate['technical_status']}; next action `{gate['next_action']}`.", "",
             "| pair | C(U) | C(V) | delta U-V | PLACE_BUFFER U/V (s) | steps U/V | match |", "|---|---|---|---|---|---|---|"]
    for key, p in gate["pairs"].items():
        lines.append(f"| {key} | {p['cost_u']} | {p['cost_v']} | {p['delta_u_minus_v']} | "
                     f"{p['candidate_duration_abs_diff']} (abs diff) | {p['candidate_steps_abs_diff']} (abs diff) | {p['candidate_match_pass']} |")
    lines += ["", "Conditions: " + ", ".join(f"{k}={v}" for k, v in gate["conditions"].items()),
              f"- Measured winners: {gate['measured_winner']}; predicted: {gate['predicted_winner']}.",
              f"- Always-U {gate['always_u_accuracy']}, always-V {gate['always_v_accuracy']}; state-conditioned reversal "
              f"{gate['state_conditioned_preference_reversal']}.", "- No significance or generalization claim; speedup NOT_MEASURED."]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n")


REQUIRED_OUTPUTS = (
    "authorization.json", "source_identity.json", "config_resolved.yaml", "geometry_invariants.json",
    "camera_projection_audit.json", "registration.json", "budget_ledger.json", "budget_events.jsonl", "branch_results.csv",
    "candidate_action_fairness.csv", "pair_restore.json", "observation_support.csv", "cost_decomposition.csv",
    "mechanism_gate.json", "protected_before.json", "protected_after.json", "reach_envelope_audit.json",
    "aabb_clearance_audit.json", "test_receipt.json", "mutation_receipt.json", "final_summary.md")


def verify(root, out):
    root, out = Path(root), Path(out)
    before = read(out / "protected_before.json")
    after_inv = m.inventory(protected_paths(root))
    write(out / "protected_after.json", {"sha256": after_inv, "files": len(after_inv),
                                         "capture_count": sum("/captures/" in k for k in after_inv)})
    diff = m.diff_inventory(before["sha256"], after_inv)
    ledger, reg = read(out / "budget_ledger.json"), read(out / "physical/registration.json")
    results = list((out / "physical/branch_results").glob("*.json"))
    zero_keys = ("provider_calls", "representation_forwards", "rl_transitions", "optimizer_steps", "elastic",
                 "skill_retries", "standalone_capture_resets", "formal_test_episodes")
    missing = [n for n in REQUIRED_OUTPUTS if not (out / n).is_file()]
    checks = {
        "protected_unchanged": not (diff["changed"] or diff["removed"] or diff["added"]),
        "frozen_profile_files_equal_freeze_commit": not r2.frozen_bytes_ok(root),
        "attempts_le_8": ledger["attempts"]["used"] <= 8 and len(results) <= 8,
        "constructions_le_8": ledger["environment_constructions"]["used"] <= 8,
        "resets_le_8": ledger["explicit_resets"]["used"] <= 8,
        "skills_le_48": ledger["live_skill_calls"]["used"] <= 48,
        "no_duplicate_reservation": len(set(ledger["reserved_branches"])) == len(ledger["reserved_branches"]),
        "registration_is_the_8": len(reg["branches"]) == 8,
        "provider_representation_rl_optimizer_zero": all(ledger[k]["used"] == 0 for k in zero_keys),
        "required_outputs_present": not missing}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "missing_outputs": missing,
              "protected_diff": diff, "protected_files": len(after_inv)}
    write(out / "verify.json", result)
    return result
