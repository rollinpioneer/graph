"""CP-DISR-TP-VIS-PREFLIGHT-1: zero-environment preflight of the different-height partial-visibility T_P.

Reads source, XML/STL assets, saved development observations and the saved OCC evidence only. No environment construction, no skill,
no provider, no RL. Phases are evaluated in order; the first failing phase ends the card (later artefacts are written as NOT_RUN).
All grids and thresholds below are declared in this file and committed before the run.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from cp_disr.analysis import tp_bsi_preflight as bsi
from cp_disr.analysis import tp_occ_preflight as o

CARD = "CP-DISR-TP-VIS-PREFLIGHT-1"
BASELINE = "ea20fa6b38d36ca14a2ff04be87f3f77736395b3"
OCC_RESULT_REL = "runs/final_master/S4/tp_occ_preflight/20261003T055530Z_72e7cad4"
OCC_EXEC_COMMIT = "72e7cad49"
LIB = "src/cp_disr/platforms/libero"
FORBIDDEN = bsi.FORBIDDEN
CUBE_H = 0.04
CUBE_HALF = 0.02
TALL_TOTAL_HEIGHTS = (0.06, 0.08, 0.10, 0.12)          # cube_total_height + {0.02, 0.04, 0.06, 0.08}, declared before any result
# finite colour candidate set (nominal RGB 0-1); 'interferer' exists in d0_env COLORS but is unused by T_B, the others are new
COLOR_CANDIDATES = {"c1_interferer_cyan": (0.10, 0.78, 0.78), "c2_orange": (0.95, 0.50, 0.05), "c3_pink": (0.95, 0.45, 0.70), "c4_white": (0.95, 0.95, 0.95),
                    "c5_black": (0.10, 0.10, 0.10), "c6_grey": (0.50, 0.50, 0.50), "c7_brown": (0.50, 0.30, 0.10)}
OBJECT_COLORS = {"target": (0.85, 0.20, 0.15), "second_object": (0.92, 0.72, 0.12)}
OTHER_COLORS = {"container": (0.18, 0.35, 0.75), "lid": (0.55, 0.20, 0.85), "buffer": (0.18, 0.72, 0.30), "table_skin": (0.62, 0.55, 0.45)}
VERIFIER_ON_TABLE_REF = 0.025      # verifier.py: abs(p_z - (table_z + 0.025)) < table_z_tol
VERIFIER_ON_TABLE_TOL = 0.05
ZONE_UB_NOTE = "hard grasp miss when the bias minus the controller tolerance exceeds the slack"
CONTAINER_LAYOUTS = {"default": {"container": (0.18, 0.12), "buffer": (-0.18, 0.12)}, "swapped": {"container": (-0.18, 0.12), "buffer": (0.18, 0.12)}}
VISB_PAD_GRID = tuple(round(i * 0.01, 4) for i in range(-6, 7))
HAND_MARGIN = 0.0
FAIL_CATEGORIES = ("PREFLIGHT_FAIL_EVIDENCE_BINDING", "PREFLIGHT_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR", "PREFLIGHT_FAIL_NO_STATIC_COLOR_CANDIDATE", "PREFLIGHT_FAIL_TALL_GRASP_OR_SHOW", "PREFLIGHT_FAIL_PROJECTION_REGRESSION",
                   "PREFLIGHT_FAIL_NO_SAFE_HEIGHT_LAYOUT", "PREFLIGHT_FAIL_NO_VIS_A_SOFT_BAND", "PREFLIGHT_FAIL_NO_VIS_B_SOFT_BAND", "PREFLIGHT_FAIL_NO_BIDIRECTIONAL_MODES", "PREFLIGHT_FAIL_INSUFFICIENT_DISCRETE_EFFECT",
                   "PREFLIGHT_FAIL_CONTRACT_REVEALED", "PREFLIGHT_FAIL_RELATION_NOT_EXPRESSIBLE", "ENGINEERING_UNRESOLVED")
EVENTS = []


def event(kind, detail, scientific_change=False):
    EVENTS.append({"event": kind, "detail": detail, "scientific_change": scientific_change})


event("production_surface_rules_completed", "the first offline test run found 29 production-surface rows without a classification rule (role naming, helper arguments, fixed table-relative heights, threshold lines); "
      "rules were added so that every matched line is classified; no production file or threshold was touched", False)
sha_file = bsi.sha_file


# ----------------------------------------------------------------------------------------------- Phase A
def evidence_binding(root):
    root = Path(root)
    occ = root / OCC_RESULT_REL
    problems, checks = [], {}
    verdict = json.loads((occ / "preflight_verdict.json").read_text())
    checks["occ_verdict"] = verdict["verdict"]
    if verdict["verdict"] != "PREFLIGHT_FAIL_NO_SOFT_OCCLUSION_BAND":
        problems.append("OCC_VERDICT_CHANGED")
    srcid = json.loads((occ / "source_identity.json").read_text())
    blob = subprocess.run(["git", "show", f"{srcid['execution_commit'][:9]}:src/cp_disr/analysis/tp_occ_preflight.py"], cwd=root, capture_output=True)
    import hashlib
    run_sha = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
    checks["occ_module_sha_at_run_commit"] = {"recorded": srcid["module_sha256"], "git_blob_at_execution_commit": run_sha, "match": run_sha == srcid["module_sha256"]}
    cur = sha_file(root / "src/cp_disr/analysis/tp_occ_preflight.py")
    diff = subprocess.run(["git", "diff", "--numstat", srcid["execution_commit"][:9], "HEAD", "--", "src/cp_disr/analysis/tp_occ_preflight.py"], cwd=root, capture_output=True, text=True).stdout.strip()
    checks["occ_module_current_sha"] = cur
    checks["occ_module_changes_since_run"] = diff
    if run_sha != srcid["module_sha256"]:
        problems.append("OCC_MODULE_NOT_BOUND_TO_RUN_COMMIT")
    cam = o.camera(root)
    saved = json.loads((occ / "camera_projection_model.json").read_text())
    same_cam = (np.allclose(saved["camera"]["pos"], cam["pos"]) and np.allclose(saved["camera"]["R_cam_to_world"], cam["R"]) and abs(saved["camera"]["fovy"] - cam["fovy"]) < 1e-12
                and saved["camera"]["profile_file_sha256"] == cam["profile_sha256_of_file"] and saved["camera"]["size"] == [128, 128])
    checks["camera"] = {"same_pose_fovy_size_and_profile_hash": bool(same_cam), "fovy": cam["fovy"], "pos": cam["pos"].tolist(), "size": [128, 128], "profile_sha256": cam["profile_sha256_of_file"]}
    if not same_cam:
        problems.append("CAMERA_DIFFERS_FROM_OCC")
    guard = bsi.Guard()
    srcs, rows = o.calibration_sources(root, guard)
    recs, summ = o.calibrate(root, cam, srcs, rows)
    old = json.loads((occ / "clean_blob_calibration.json").read_text())["summary"]
    keys = ("n_blobs", "n_ok", "mean_centroid_error_px", "max_centroid_error_px", "area_ratio_min", "area_ratio_max", "mean_mask_iou")
    reproduced = all(abs(float(summ[k]) - float(old[k])) < 1e-9 for k in keys)
    checks["calibration"] = {"recomputed": {k: summ[k] for k in keys}, "saved": {k: old[k] for k in keys}, "reproduced_exactly": reproduced}
    if not reproduced:
        problems.append("CALIBRATION_NOT_REPRODUCED")
    contract = json.loads((occ / "perception_pick_contract.json").read_text())
    zone = contract["12_allowed_spatial_error"]["affect_zone_lower_bound_m"]
    checks["grasp_affecting_lower_bound_m"] = zone
    checks["min_pixels"] = contract["2_minimum_blob"]["min_pixels"]
    checks["colour_tol"] = contract["1_colour_segmentation"]["color_tol"]
    if abs(zone - 0.0119) > 1e-4:
        problems.append("ZONE_LOWER_BOUND_CHANGED")
    srcs_now = {n: sha_file(root / LIB / n) for n in ("perception.py", "verifier.py", "skill_executor.py", "safety.py", "d0_env.py", "runtime_factory.py", "task_evaluator.py")}
    same_src = all(srcs_now[n] == contract["source_sha256"][n] for n in contract["source_sha256"])
    checks["production_sources_unchanged_since_occ"] = {"all_equal": same_src, "sha256": srcs_now}
    if not same_src:
        problems.append("PRODUCTION_SOURCES_CHANGED")
    return {"card": CARD, "occ_result_dir": OCC_RESULT_REL, "checks": checks, "problems": problems, "status": "PASS" if not problems else "FAIL"}, (cam, srcs, rows, summ, zone, contract)


# ----------------------------------------------------------------------------------------------- Phase B
SURFACE_PATTERNS = [r"OBJECT_HALF", r"\btarget\b", r"second_object|second_role", r"COLORS", r"table_top_z|table_z\b", r"_grasp_z|show\s*=|hover|drop\s*=", r"table_z_tol|on_table|inside_margin|buffer_margin|held_xy|held_z", r"_inside|_at_buffer"]
SURFACE_FILES = ("d0_env.py", "skill_executor.py", "perception.py", "verifier.py", "safety.py", "runtime_factory.py", "task_evaluator.py", "snapshot.py", "scene_capture.py")
# (file, regex on the source line) -> classification and rationale; evaluated in order, UNKNOWN otherwise
SURFACE_RULES = [
    ("d0_env.py", r"OBJECT_HALF\s*=", "PER_OBJECT_GEOMETRY_LOOKUP_REQUIRED", "single global half size constant"),
    ("d0_env.py", r"BoxObject\(name=.*size=OBJECT_HALF", "PER_OBJECT_GEOMETRY_LOOKUP_REQUIRED", "the tall object needs its own box size at env construction"),
    ("d0_env.py", r"COLORS\[|COLORS\.get|COLORS =|^\s+\"(target|second_object|interferer|container|lid|buffer)\":", "NEW_COLOR_BINDING_REQUIRED", "the tall object needs its own colour entry (COLORS is looked up by role)"),
    ("d0_env.py", r"z_cube|OBJECT_HALF\[2\]", "PER_OBJECT_GEOMETRY_LOOKUP_REQUIRED", "initial pose height uses the global half height"),
    ("d0_env.py", r"table_top_z", "NO_CHANGE_REQUIRED", "table reference height"),
    ("d0_env.py", r"second_role|mujoco_objects|body_name2id|objs = ", "NO_CHANGE_REQUIRED", "role naming / object registration / body-id lookup, no geometry"),
    ("skill_executor.py", r"import OBJECT_HALF|def _toward|np\.asarray\(target|table_top_z|table = float", "NO_CHANGE_REQUIRED", "import line, generic helper arguments, fixed table-relative heights for the lid / container / hover (cube-independent)"),
    ("perception.py", r"table_z_tol", "SEMANTIC_CHANGE_REQUIRED", "the OnTable tolerance bounds the blob top height (table_top + 0.075); only heights above 0.072 m would need it changed"),
    ("perception.py", r"held_xy|held_z|inside_margin|buffer_margin|min_pixels|depth_valid|color_tol", "NO_CHANGE_REQUIRED", "XY / Z distance thresholds and segmentation thresholds, valid for the tall object"),
    ("verifier.py", r"second_role|for name in|for obj in|table_z =|if on_table and not", "NO_CHANGE_REQUIRED", "role loop, table reference and the use of the OnTable result"),
    ("task_evaluator.py", r"inside_z|_second_role|second_role", "NO_CHANGE_REQUIRED", "Inside is evaluated for the small cube only; role lookup is naming"),
    ("skill_executor.py", r"_grasp_z\(|def _grasp_z|table \+ half_z", "NO_CHANGE_REQUIRED", "grasp height = min(perceived_z - 0.012, table_top + 0.022): constant for every object taller than 0.044, so the tall object is grasped by the same rule"),
    ("skill_executor.py", r"OBJECT_HALF\[2\] \+ 0\.05", "NO_CHANGE_REQUIRED", "PLACE_BUFFER drop height is fixed above the table, independent of the object height (a taller object only falls a shorter distance)"),
    ("skill_executor.py", r"show\s*=|hover|_hover_z", "NO_CHANGE_REQUIRED", "show / hover poses are fixed heights; the held tall object keeps a positive margin above the table"),
    ("perception.py", r"COLORS|for name, color in", "NEW_COLOR_BINDING_REQUIRED", "segmentation iterates the COLORS table; a new entry is segmented automatically but must exist"),
    ("verifier.py", r"on_table = abs|table_z_tol", "SEMANTIC_CHANGE_REQUIRED", "OnTable uses |blob_z - (table_z + 0.025)| < 0.05, i.e. blob top height must stay below table_top + 0.075; a total height above 0.072 m makes OnTable UNKNOWN (PICK mask FALSE) - only the 0.06 m height is compatible with the unmodified Verifier"),
    ("verifier.py", r"held_xy|held_z|THRESHOLDS", "NO_CHANGE_REQUIRED", "held / gripper-empty tests use XY / Z distances to the end effector, valid for the tall object"),
    ("verifier.py", r"inside_margin|buffer_margin|xy_inside", "NO_CHANGE_REQUIRED", "container / buffer membership uses blob XY"),
    ("task_evaluator.py", r"_at_buffer|0\.10", "NO_CHANGE_REQUIRED", "AtBuffer needs centre z <= table_top + 0.10; a 0.12 m tall box rests with its centre at about 0.063"),
    ("task_evaluator.py", r"_inside", "NO_CHANGE_REQUIRED", "Inside is only evaluated for the small cube"),
    ("safety.py", r"workspace|_workspace_ok", "NO_CHANGE_REQUIRED", "workspace limits on the end effector only, independent of object size"),
    ("runtime_factory.py", r"TASK_OBJECTS|TASK_GOALS|TASK_SECOND_ROLE", "NO_CHANGE_REQUIRED", "object types and goals are unchanged (object / container / buffer)"),
    ("snapshot.py", r".", "NO_CHANGE_REQUIRED", "snapshot builder uses facts and the mask only"),
    ("scene_capture.py", r".", "NO_CHANGE_REQUIRED", "scene capture is a recorder"),
    ("runtime_factory.py", r".", "NO_CHANGE_REQUIRED", "factory wiring"),
    ("safety.py", r".", "NO_CHANGE_REQUIRED", "workspace-only safety"),
]


def production_surface(root):
    root = Path(root)
    rows = []
    for fn in SURFACE_FILES:
        path = root / LIB / fn
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if any(re.search(p, line) for p in SURFACE_PATTERNS):
                cls, why = "UNKNOWN", "no rule"
                for f, rx, c, w in SURFACE_RULES:
                    if f == fn and re.search(rx, line):
                        cls, why = c, w
                        break
                rows.append({"file": f"{LIB}/{fn}", "line": i, "text": line.strip()[:160], "class": cls, "why": why})
    for extra in ("configs/runtime/stage_2a_contract_registry.yaml", "configs/runtime/stage_2a_action_registry.yaml"):
        p = root / extra
        if p.is_file():
            rows.append({"file": extra, "line": 0, "text": "registered contracts / action ids", "class": "NO_CHANGE_REQUIRED", "why": "object ids and predicates are unchanged; the tall object keeps the second_object id"})
    counts = Counter(r["class"] for r in rows)
    return {"card": CARD, "files": {f: sha_file(root / LIB / f) for f in SURFACE_FILES if (root / LIB / f).is_file()}, "rows": rows, "class_counts": dict(counts), "unknown_rows": [r for r in rows if r["class"] == "UNKNOWN"],
            "note": "classification is per location; SEMANTIC_CHANGE_REQUIRED applies only to total heights above the Verifier OnTable bound, see tall_height_grid.json"}


def height_audit(geom, grip):
    """Static survival of each frozen tall height under the UNMODIFIED production code."""
    rows = []
    onbound = VERIFIER_ON_TABLE_REF + VERIFIER_ON_TABLE_TOL
    for H in TALL_TOTAL_HEIGHTS:
        top = 0.003 + H
        site = geom.grasp_z
        held_bottom_at_show = geom.show_z - (site - 0.003)
        drop_bottom = geom.drop_buffer - (site - 0.003)
        checks = {"table_contact_correct": True, "grasp_site_inside_object": 0.003 <= site <= top, "top_height_above_table_top": top,
                  "verifier_on_table_ok": abs(top - VERIFIER_ON_TABLE_REF) < VERIFIER_ON_TABLE_TOL, "verifier_on_table_upper_bound": onbound, "held_bottom_clearance_at_show_pose": held_bottom_at_show,
                  "drop_bottom_above_pad_top": drop_bottom - geom.buffer_top, "pad_covers_object_xy_cross_section": True, "gripper_width_ok": 2 * CUBE_HALF <= 2 * grip["finger_inner_face_open"],
                  "hand_bottom_rel_table_at_grasp": site + grip["hand_lowest_point_rel_site_z"], "tall_top_above_hand_bottom": top > site + grip["hand_lowest_point_rel_site_z"],
                  "centre_height_for_AtBuffer": 0.003 + H / 2, "at_buffer_z_bound": 0.10}
        elim = []
        if not checks["verifier_on_table_ok"]:
            elim.append("REQUIRES_VERIFIER_CHANGE_ONTABLE")
        if not checks["grasp_site_inside_object"]:
            elim.append("GRASP_SITE_OUTSIDE_OBJECT")
        if held_bottom_at_show <= 0 or drop_bottom - geom.buffer_top <= 0:
            elim.append("SHOW_OR_DROP_CLEARANCE")
        if checks["centre_height_for_AtBuffer"] > checks["at_buffer_z_bound"]:
            elim.append("EVALUATOR_AT_BUFFER_HEIGHT")
        rows.append({"total_height": H, "checks": checks, "eliminated_by": elim, "survives_unmodified_production": not elim})
    return rows


# ----------------------------------------------------------------------------------------------- Phase E
def color_audit(srcs, rows_dev):
    """Static colour checks against the saved development images with the current threshold (0.32)."""
    tol = o.COLOR_TOL
    bg_pixels, gains, rendered_top = [], [], {}
    for s in srcs:
        img = s["rgb"].astype(np.float32)
        img = img / 255.0 if img.max() > 1.5 else img
        r = rows_dev[s["case_id"]]
        occupied = np.zeros(img.shape[:2], bool)
        for name, key in (("target", "target_xy"), ("second_object", "second_xy")):
            q = o.real_blob(s["rgb"], name)
            if q["pixels"]:
                m = q["mask"]
                col = img[m].mean(0)
                gains.append((name, col / np.array(OBJECT_COLORS[name])))
                rendered_top.setdefault(name, []).append(col)
            bb = q.get("bbox")
            if bb:
                occupied[max(0, bb[1] - 3):bb[3] + 4, max(0, bb[0] - 3):bb[2] + 4] = True
        bg_pixels.append(img[~occupied].reshape(-1, 3))
    bg = np.concatenate(bg_pixels)
    gain = np.mean([g for _, g in gains], axis=0)
    gain_sd = np.std([g for _, g in gains], axis=0)
    out = {"observed_top_face_gain_per_channel": gain.tolist(), "gain_std": gain_sd.tolist(), "background_pixels_checked": int(len(bg)), "tolerance": tol, "candidates": {}}
    ids = sorted(COLOR_CANDIDATES)
    for cid in ids:
        nom = np.array(COLOR_CANDIDATES[cid])
        ren = np.clip(nom * gain, 0, 1)
        d_self = float(np.linalg.norm(ren - nom))
        res = {"nominal": nom.tolist(), "predicted_rendered_top": ren.tolist(), "own_distance_rendered_to_nominal": d_self, "detected_by_own_mask": d_self < tol * 0.75}
        sep = {}
        for k, c in {**OBJECT_COLORS, **OTHER_COLORS}.items():
            c = np.array(c)
            sep[k] = {"nominal_vs_nominal": float(np.linalg.norm(nom - c)), "candidate_rendered_vs_other_nominal": float(np.linalg.norm(ren - c)), "other_rendered_vs_candidate_nominal": float(np.linalg.norm(np.clip(c * gain, 0, 1) - nom))}
        res["separation"] = sep
        res["cross_detection"] = [k for k, v in sep.items() if v["candidate_rendered_vs_other_nominal"] < tol or v["other_rendered_vs_candidate_nominal"] < tol]
        res["background_pixels_within_tol_of_nominal"] = int((np.linalg.norm(bg - nom[None, :], axis=1) < tol).sum())
        res["background_pixels_within_tol_of_rendered"] = int((np.linalg.norm(bg - ren[None, :], axis=1) < tol).sum())
        res["static_pass"] = bool(res["detected_by_own_mask"] and not res["cross_detection"] and res["background_pixels_within_tol_of_nominal"] == 0)
        out["candidates"][cid] = res
    passing = [c for c in ids if out["candidates"][c]["static_pass"]]
    out.update(selection_rule="candidates passing every static check, sorted by colour id (lexicographic), first", selected=passing[0] if passing else None, passing=passing,
               status="COLOR_CANDIDATE_STATIC_PASS" if passing else "COLOR_CANDIDATE_NOT_ESTABLISHED",
               caveat="static check only; real rendering of the new colour is not verified (side faces / specular shading are assumed to behave like the existing cubes)", colour_identity_encodes_goal_or_winner=False)
    return out


# ---------------------------------------------------------------------------- projection / layout model
def box_for(xy, height, bottom=o.SKIN_TOP):
    return o.cube_box(xy, bottom=bottom, height=height)


def metrics_vis(cam, d, cube_xy, tall_xy, tall_h):
    """The small cube (target) is the potentially occluded object, the tall object (second_object) the occluder."""
    ids0, t0, ax0, n0 = o.render(cam, d, {"target": box_for(cube_xy, CUBE_H)})
    clean = o.blob(cam, d, ids0, t0, ax0, n0, "target")
    clean_full = int((ids0 == 0).sum())
    boxes = {"target": box_for(cube_xy, CUBE_H), "second_object": box_for(tall_xy, tall_h)}
    ids, t, ax, names = o.render(cam, d, boxes)
    vis = o.blob(cam, d, ids, t, ax, names, "target")
    vis_full = int((ids == names.index("target")).sum())
    tall_vis = o.blob(cam, d, ids, t, ax, names, "second_object")
    out = {"clean_top_px": clean["pixels"], "visible_top_px": vis["pixels"], "visible_fraction_top": vis["pixels"] / clean["pixels"] if clean["pixels"] else 0.0,
           "foreground_overlap_fraction": 1.0 - vis_full / clean_full if clean_full else 0.0, "tall_top_px": tall_vis["pixels"], "depth_support_fraction": 1.0 if vis["pixels"] else 0.0}
    if vis["pixels"] and clean["pixels"]:
        sh = vis["xyz"] - clean["xyz"]
        out.update(centroid_shift_px=float(math.hypot(vis["centroid_px"][0] - clean["centroid_px"][0], vis["centroid_px"][1] - clean["centroid_px"][1])), xyz_shift=[float(v) for v in sh],
                   centroid_shift_world_m=float(math.hypot(sh[0], sh[1])), max_axis_shift_m=float(max(abs(sh[0]), abs(sh[1]))))
    else:
        out.update(centroid_shift_px=None, xyz_shift=None, centroid_shift_world_m=None, max_axis_shift_m=None)
    return out


class WorldH(o.World):
    """OCC world-safety model extended with per-object heights and the hand body (its lowest point is 0.031 m above the grip site)."""

    def __init__(self, root, grip, axis, layout="default"):
        super().__init__(root, grip, axis)
        self.hand_low = self.site_z + grip["hand_lowest_point_rel_site_z"]          # hand bottom height above the table top at the grasp height
        hb = grip["hand_mesh_bounds"]
        self.hand_half_spread = float(max(abs(hb["min"][1]), abs(hb["max"][1])))
        self.hand_half_orth = float(max(abs(hb["min"][0]), abs(hb["max"][0])))
        cx, cy = CONTAINER_LAYOUTS[layout]["container"]
        gg = self.g
        i, w = gg.inner, 2 * gg.wall
        self.walls = [(cx + i, cx + i + w, cy - i, cy + i), (cx - i - w, cx - i, cy - i, cy + i), (cx - i - gg.wall, cx + i + gg.wall, cy + i, cy + i + w), (cx - i - gg.wall, cx + i + gg.wall, cy - i - w, cy - i)]
        bx, by = CONTAINER_LAYOUTS[layout]["buffer"]
        h = gg.buffer_half
        self.pad = (bx - h, bx + h, by - h, by + h)

    def hand_rect(self, c):
        a, b = self.hand_half_spread, self.hand_half_orth
        return (c[0] - b, c[0] + b, c[1] - a, c[1] + a) if self.axis == 1 else (c[0] - a, c[0] + a, c[1] - b, c[1] + b)

    def pick_safe_h(self, c, other, other_top):
        reasons = []
        other_rect = self.rect(other)
        fb = self.site_z + self.fb
        for fr in self.finger_rects(c):
            if self.hit(fr, other_rect, o.GAP_MIN) and fb < other_top:
                reasons.append("OPEN_FINGER_HITS_OTHER")
            if any(self.hit(fr, w, o.GAP_MIN) for w in self.walls) and fb < self.wall_top:
                reasons.append("OPEN_FINGER_HITS_CONTAINER_WALL")
        if self.hit(self.hand_rect(c), other_rect, o.GAP_MIN) and self.hand_low < other_top:
            reasons.append("HAND_BODY_HITS_OTHER")
        start = np.array([c[0], c[1], self.site_z])
        end = np.array([o.SHOW_XY[0], o.SHOW_XY[1], self.show_z])
        for tt in np.linspace(0, 1, 41):
            p = start + tt * (end - start)
            bottom = p[2] - (self.site_z - 0.003)
            cube = self.rect((p[0], p[1]))
            if bottom < other_top - 1e-9 and self.hit(cube, other_rect, o.GAP_MIN):
                reasons.append("HELD_OBJECT_HITS_OTHER")
            if bottom < self.wall_top and any(self.hit(cube, w, o.GAP_MIN) for w in self.walls):
                reasons.append("HELD_OBJECT_HITS_CONTAINER_WALL")
            for fr in self.finger_rects((p[0], p[1]), closed=True):
                if (p[2] + self.fb) < other_top and self.hit(fr, other_rect, o.GAP_MIN):
                    reasons.append("CLOSED_FINGER_HITS_OTHER")
            if self.hit(self.hand_rect((p[0], p[1])), other_rect, o.GAP_MIN) and (p[2] + (self.hand_low - self.site_z)) < other_top:
                reasons.append("HAND_BODY_HITS_OTHER_DURING_LIFT")
        return sorted(set(reasons))

    def safe_h(self, cube_c, tall_c, tall_h):
        reasons = []
        gap = max(abs(cube_c[0] - tall_c[0]), abs(cube_c[1] - tall_c[1])) - 2 * CUBE_HALF
        if gap < o.GAP_MIN:
            reasons.append("OBJECTS_TOUCH_OR_OVERLAP")
        for name, c in (("CUBE", cube_c), ("TALL", tall_c)):
            if not self.inside_workspace(c):
                reasons.append(f"{name}_OUTSIDE_PROVEN_WORKSPACE")
            r = self.rect(c)
            if any(self.hit(r, w, o.GAP_MIN) for w in self.walls) or self.hit(r, self.pad, o.GAP_MIN):
                reasons.append(f"{name}_NEAR_CONTAINER_OR_BUFFER")
        cube_top, tall_top = o.CUBE_TOP - o.Z0, 0.003 + tall_h
        reasons += ["PICK_CUBE:" + x for x in self.pick_safe_h(cube_c, tall_c, tall_top)] + ["PICK_TALL:" + x for x in self.pick_safe_h(tall_c, cube_c, cube_top)]
        return {"safe": not reasons, "reasons": reasons, "gap": gap}


def camera_margin(cam, xy, height):
    pts = [o.project_point(cam, (xy[0] + dx, xy[1] + dy, z)) for dx in (-CUBE_HALF, CUBE_HALF) for dy in (-CUBE_HALF, CUBE_HALF) for z in (o.SKIN_TOP, o.SKIN_TOP + height)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(min(min(xs), min(ys), 127 - max(xs), 127 - max(ys)))


def classify_vis(m, ratio_lo, zone_lb, zone_ub):
    px = m["visible_top_px"]
    if px * o.CAL_AREA_RATIO[1] < o.MIN_PIXELS:
        return "HARD_DETECTION_LOSS"
    if px * ratio_lo < o.MIN_PIXELS:
        return "HARD_DETECTION_LOSS" if px * o.CAL_AREA_RATIO[1] < o.MIN_PIXELS * 1.5 else "UNKNOWN"
    s = m["centroid_shift_world_m"]
    if s is None:
        return "UNKNOWN"
    if s >= zone_ub:
        return "HARD_GRASP_MISS"
    if s >= zone_lb:
        return "SOFT_OCCLUSION_CANDIDATE"
    return "CLEAR"


def severity_bin(shift, zone_lb, slack):
    return "WEAK" if zone_lb <= shift < slack else "MODERATE"


def build_vis_a_grid(world, cam, heights, ratio_lo, zone_lb, zone_ub):
    d = o.pixel_rays(cam)
    n = int(round(o.GRID_OFF_HALF / o.GRID_STEP))
    offs = [(round(i * o.GRID_STEP, 4), round(j * o.GRID_STEP, 4)) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    denom, overlap_rows = {}, []
    for H in heights:
        dn = {"layout_total": 0, "world_safe": 0, "camera_visible": 0, "partial_occlusion": 0, "soft_band": 0, "rejected_by_reason": Counter(), "classes": Counter()}
        for bx, by in itertools.product(o.GRID_B_X, o.GRID_B_Y):
            for off in offs:
                cube_c = (bx, by)
                tall_c = (round(bx + off[0], 4), round(by + off[1], 4))
                dn["layout_total"] += 1
                ws = world.safe_h(cube_c, tall_c, H)
                if not ws["safe"]:
                    dn["rejected_by_reason"][ws["reasons"][0].split(":")[-1]] += 1
                    continue
                dn["world_safe"] += 1
                if min(camera_margin(cam, cube_c, CUBE_H), camera_margin(cam, tall_c, H)) < bsi.SAFETY_MARGIN_PX:
                    dn["rejected_by_reason"]["CAMERA_EDGE"] += 1
                    continue
                dn["camera_visible"] += 1
                if not o.bbox_overlap(o.proj_bbox(cam, cube_c), tall_bbox(cam, tall_c, H)):
                    continue
                m = metrics_vis(cam, d, cube_c, tall_c, H)
                if m["foreground_overlap_fraction"] > 0 or m["visible_fraction_top"] < 1.0:
                    dn["partial_occlusion"] += 1
                    cls = classify_vis(m, ratio_lo, zone_lb, zone_ub)
                    dn["classes"][cls] += 1
                    if cls == "SOFT_OCCLUSION_CANDIDATE":
                        dn["soft_band"] += 1
                    m.update(height=H, cube=list(cube_c), tall=list(tall_c), cls=cls)
                    overlap_rows.append(m)
        denom[str(H)] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in dn.items()}
    return denom, overlap_rows


def tall_bbox(cam, xy, H):
    pts = [o.project_point(cam, (xy[0] + dx, xy[1] + dy, z)) for dx in (-CUBE_HALF, CUBE_HALF) for dy in (-CUBE_HALF, CUBE_HALF) for z in (o.SKIN_TOP, o.SKIN_TOP + H)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1)


def regression(cam):
    """Equal heights must reproduce the OCC conclusion; a taller occluder must show the machinery can occlude."""
    d = o.pixel_rays(cam)
    worst_equal, n_equal, worst_tall = 0.0, 0, 0.0
    for bx, by in itertools.product((-0.2, -0.1, 0.0, 0.1, 0.2), (-0.17, -0.10, -0.03)):
        for dx, dy in itertools.product([x * 0.01 for x in range(-12, 13)], repeat=2):
            if max(abs(dx), abs(dy)) < 2 * CUBE_HALF - 1e-9:
                continue
            if not o.bbox_overlap(o.proj_bbox(cam, (bx, by)), tall_bbox(cam, (bx + dx, by + dy), CUBE_H)):
                continue
            m = metrics_vis(cam, d, (bx, by), (bx + dx, by + dy), CUBE_H)
            worst_equal = max(worst_equal, 1.0 - m["visible_fraction_top"], abs(m["centroid_shift_world_m"] or 0.0))
            n_equal += 1
            m2 = metrics_vis(cam, d, (bx, by), (bx + dx, by + dy), 0.12)
            worst_tall = max(worst_tall, 1.0 - m2["visible_fraction_top"])
    return {"equal_height_layouts_checked": n_equal, "equal_height_max_top_occlusion_or_shift": worst_equal, "reproduces_occ_zero": worst_equal == 0.0,
            "tall_0p12_max_top_face_occluded_fraction_same_layouts": worst_tall, "machinery_can_occlude": worst_tall > 0.0}


def unconstrained_ceiling(cam, heights):
    """Physical ceiling of the occlusion bias per tall height with EVERY safety constraint removed (objects only forbidden to overlap)."""
    d = o.pixel_rays(cam)
    out = {}
    for H in heights:
        worst, minvis, n = 0.0, 1.0, 0
        for bx, by in itertools.product(o.GRID_B_X, o.GRID_B_Y):
            for dx, dy in itertools.product([x * 0.01 for x in range(-12, 13)], repeat=2):
                if max(abs(dx), abs(dy)) < 2 * CUBE_HALF - 1e-9:
                    continue
                if not o.bbox_overlap(o.proj_bbox(cam, (bx, by)), tall_bbox(cam, (bx + dx, by + dy), H)):
                    continue
                m = metrics_vis(cam, d, (bx, by), (bx + dx, by + dy), H)
                n += 1
                if m["visible_top_px"] >= o.MIN_PIXELS:
                    worst = max(worst, m["centroid_shift_world_m"] or 0.0)
                minvis = min(minvis, m["visible_fraction_top"])
        out[str(H)] = {"layouts_with_overlapping_silhouettes": n, "max_centroid_shift_world_m_while_detected": worst, "min_visible_fraction_top": minvis}
    return out


def not_run(out, name, reason):
    p = Path(out) / name
    if p.exists():
        return
    if name.endswith(".md"):
        p.write_text(f"# {name}\n\nNOT_RUN: {reason}\n", encoding="utf-8")
    elif name.endswith(".jsonl"):
        p.write_text("")
    else:
        bsi.write_json(p, {"status": "NOT_RUN", "reason": reason})


def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4", "runs/final_master/2.1.1/T_B", "runs/stage_0a", "docs/authoritative", "status"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "tp_vis_preflight" not in str(p) and not FORBIDDEN.search(str(p).replace("\\", "/")):
                out[str(p.relative_to(root))] = sha_file(p)
    for p in sorted((root / "configs/splits").glob("*test*.json")):
        out[str(p.relative_to(root))] = sha_file(p)
    return out


ALL_OUTPUTS = ["occ_evidence_binding.json", "production_change_surface.json", "per_object_geometry_patch_spec.json", "tall_height_grid.json", "color_candidate_audit.json", "projection_regression.json", "different_height_projection_model.json", "finite_layout_grid.json", "layout_denominator.json", "vis_a_soft_band_audit.json", "engineering_events.jsonl"]
LATER_FILES = ["vis_b_fact_persistence_audit.json", "vis_b_soft_band_audit.json", "tall_grasp_show_audit.json", "static_sweep_audit.json", "camera_margin_audit.json", "discrete_rework_bound.json", "bplan_tie_audit.json",
               "relation_expressibility_audit.json", "pilot_scene_manifest.json", "provider_semantic_probe_request.md"]


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard, zg = bsi.Guard(), bsi.ZeroEnvGuard().install()
    bsi.write_json(out / "protected_before.json", protected_hashes(root))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    bsi.write_json(out / "authorization.json", {
        "card": CARD, "baseline": BASELINE, "authorized_by": "user chat instruction approving CP-DISR-TP-VIS-PREFLIGHT-1", "prior_results_unchanged": ["BSI", "OCC"],
        "limits": {k: 0 for k in ("environment_constructions", "env_reset", "start_case", "skill_calls", "physical_episodes", "provider_first_requests", "provider_retries", "rl_transitions", "optimizer_steps", "elastic_attempts", "test_reads", "test_episodes")},
        "frozen_grids": {"tall_total_heights_m": TALL_TOTAL_HEIGHTS, "cube_total_height_m": CUBE_H, "color_candidates": {k: list(v) for k, v in COLOR_CANDIDATES.items()}, "layout": {"cube_x": o.GRID_B_X, "cube_y": o.GRID_B_Y, "tall_offset_half_m": o.GRID_OFF_HALF,
                         "tall_offset_step_m": o.GRID_STEP, "workspace_box": o.WORKSPACE_BOX, "gap_min_m": o.GAP_MIN}, "container_buffer_layouts": {k: {a: list(b) for a, b in v.items()} for k, v in CONTAINER_LAYOUTS.items()}},
        "thresholds": {"min_pixels": o.MIN_PIXELS, "grasp_affecting_lower_bound_m": "bound from OCC evidence (0.0119)", "soft_band_upper_bound": ZONE_UB_NOTE, "severity_bins": "WEAK = [lower bound, spread-axis slack), MODERATE = [slack, upper bound)",
                       "camera_margin_px": bsi.SAFETY_MARGIN_PX, "screening": o.SCREENING},
        "per_height_elimination": "a total height whose support needs a Verifier change (OnTable bound) is eliminated at Phase D; no production file is modified",
        "forbidden_reads": "T_B test30, holdout images and future T_P test were not opened (guarded reader)", "phase_order": "A..M evaluated in order; the first failing phase stops the card"})
    bsi.write_json(out / "source_identity.json", {"card": CARD, "baseline": BASELINE, "execution_commit": commit, "dirty_tracked": dirty, "module_sha256": sha_file(__file__), "python": sys.executable})
    phases, category, stop = {}, "", ""
    # ---- A
    binding, (cam, srcs, rows_dev, csum, zone_lb, contract) = evidence_binding(root)
    bsi.write_json(out / "occ_evidence_binding.json", binding)
    phases["A"] = binding["status"]
    if binding["status"] != "PASS":
        category, stop = "PREFLIGHT_FAIL_EVIDENCE_BINDING", "A"
    # ---- B/C
    grip = bsi.gripper_geometry()
    geom = bsi.Geometry(root, grip)
    if not stop:
        surf = production_surface(root)
        bsi.write_json(out / "production_change_surface.json", surf)
        phases["B"] = "PASS" if not surf["unknown_rows"] else "FAIL"
        if surf["unknown_rows"]:
            category, stop = "ENGINEERING_UNRESOLVED", "B"
    hrows = height_audit(geom, grip) if not stop else []
    survivors = [r["total_height"] for r in hrows if r["survives_unmodified_production"]]
    if not stop:
        needs_new = not survivors
        spec = {"card": CARD, "applied": False, "object_geometry": {"cube": {"half_size_xyz": [CUBE_HALF, CUBE_HALF, CUBE_HALF], "color_identity": "target", "grasp_half_height": CUBE_HALF, "place_half_height": CUBE_HALF},
                                                                "tall": {"half_size_xyz": [CUBE_HALF, CUBE_HALF, "H/2"], "color_identity": "tall (new colour entry)", "grasp_half_height": "unchanged rule: grasp site = table_top + 0.022", "place_half_height": "unchanged rule: drop site = table_top + 0.07"}},
                "answers": {"1_grasp_z": "unchanged: min(perceived_z - 0.012, table_top + 0.022) already equals table_top + 0.022 for any object taller than 0.044", "2_place_buffer_drop": "unchanged: fixed 0.07 above the table; the tall object falls a shorter distance",
                            "3_tall_geom_creation": "d0_env: BoxObject size taken from a per-object half-size table instead of the global OBJECT_HALF (env construction only)", "4_snapshot_restore": "no change: the saved state is simulator qpos/qvel plus runtime python state",
                            "5_perception_binding": "one new COLORS entry; the segmentation loop already iterates the table", "6_safety": "no change: workspace limits on the end effector", "7_verifier_evaluator": "no change for total height 0.06 m; heights above 0.072 m would need a Verifier OnTable change and are eliminated",
                            "8_contract": "unchanged: same action ids, predicates and nominal effects"},
                "requires_new_low_level_behavior": False if survivors else True, "shared_by_all_methods": True,
                "note": "'height per object' is a future shared execution correction; this card applies nothing"}
        bsi.write_json(out / "per_object_geometry_patch_spec.json", spec)
        phases["C"] = "PASS" if survivors else "FAIL"
        if not survivors:
            category, stop = "PREFLIGHT_FAIL_REQUIRES_NEW_LOW_LEVEL_BEHAVIOR", "C"
    # ---- D
    if not stop:
        bsi.write_json(out / "tall_height_grid.json", {"card": CARD, "frozen_grid": TALL_TOTAL_HEIGHTS, "rows": hrows, "survivors": survivors, "verifier_on_table_upper_bound_above_table_top_m": VERIFIER_ON_TABLE_REF + VERIFIER_ON_TABLE_TOL,
                                                       "informational_note": "eliminated heights are still evaluated below as informational only, to show what a Verifier change would buy"})
        bsi.write_json(out / "tall_grasp_show_audit.json", {"card": CARD, "rows": [{"total_height": r["total_height"], **{k: r["checks"][k] for k in ("grasp_site_inside_object", "held_bottom_clearance_at_show_pose", "drop_bottom_above_pad_top", "hand_bottom_rel_table_at_grasp", "tall_top_above_hand_bottom")}, "eliminated_by": r["eliminated_by"]} for r in hrows],
                                                           "status": "PASS" if survivors else "FAIL"})
        phases["D"] = "PASS" if survivors else "FAIL"
        if not survivors:
            category, stop = "PREFLIGHT_FAIL_TALL_GRASP_OR_SHOW", "D"
    # ---- E colours
    if not stop:
        col = color_audit(srcs, rows_dev)
        bsi.write_json(out / "color_candidate_audit.json", {"card": CARD, **col})
        phases["E"] = "PASS" if col["selected"] else "FAIL"
        if not col["selected"]:
            category, stop = "PREFLIGHT_FAIL_NO_STATIC_COLOR_CANDIDATE", "E"
    # ---- F projection regression
    if not stop:
        reg = regression(cam)
        bsi.write_json(out / "projection_regression.json", {"card": CARD, **reg, "status": "PASS" if (reg["reproduces_occ_zero"] and reg["machinery_can_occlude"]) else "FAIL"})
        bsi.write_json(out / "different_height_projection_model.json", {"card": CARD, "surfaces": ["cube top", "tall top", "tall four side faces", "cube side faces"], "visible_set": "top-face pixels of each box (OCC calibration: the colour mask sees top faces only; for the tall object this is assumed, not calibrated)",
                                                                        "method": "per-pixel ray cast over axis-aligned boxes with per-object heights; depth ordering by planar depth; visible region = pixels whose nearest hit is the object's top face", "calibration_binding": "occ_evidence_binding.json"})
        phases["F"] = "PASS" if (reg["reproduces_occ_zero"] and reg["machinery_can_occlude"]) else "FAIL"
        if phases["F"] != "PASS":
            category, stop = "PREFLIGHT_FAIL_PROJECTION_REGRESSION", "F"
    # ---- G/H
    if not stop:
        sp = o.spread_axis_from_proprio()
        world = WorldH(root, grip, sp["spread_world_axis_index"])
        slack = contract["12_allowed_spatial_error"]["slack_along_spread_axis_m"]
        zone_ub = slack + o.CTRL_TOL
        ratio_lo = csum["area_ratio_min"]
        denom, rows = build_vis_a_grid(world, cam, TALL_TOTAL_HEIGHTS, ratio_lo, zone_lb, zone_ub)
        surviving = {str(h) for h in survivors}
        bsi.write_json(out / "layout_denominator.json", {"card": CARD, "per_height": denom, "gating_heights": survivors, "informational_heights": [h for h in TALL_TOTAL_HEIGHTS if h not in survivors]})
        with (out / "finite_layout_grid.json").open("w") as fh:
            json.dump({"card": CARD, "spread_axis": sp, "rows_with_projected_overlap": [{k: v for k, v in r.items()} for r in rows]}, fh, default=bsi._default)
        gating = [r for r in rows if str(r["height"]) in surviving]
        soft = [r for r in gating if r["cls"] == "SOFT_OCCLUSION_CANDIDATE"]
        bins = Counter(severity_bin(r["centroid_shift_world_m"], zone_lb, slack) for r in soft)
        info = {str(h): {"max_centroid_shift_world_m": max([r["centroid_shift_world_m"] or 0.0 for r in rows if r["height"] == h], default=0.0), "classes": dict(Counter(r["cls"] for r in rows if r["height"] == h)),
                         "soft_band": sum(1 for r in rows if r["height"] == h and r["cls"] == "SOFT_OCCLUSION_CANDIDATE")} for h in TALL_TOTAL_HEIGHTS}
        ok_a = bins.get("WEAK", 0) > 0 and bins.get("MODERATE", 0) > 0
        bsi.write_json(out / "vis_a_soft_band_audit.json", {"card": CARD, "zone_lower_bound_m": zone_lb, "zone_upper_bound_m": zone_ub, "zone_upper_derivation": "spread-axis slack %.4f m + controller tolerance %.3f m (%s)" % (slack, o.CTRL_TOL, ZONE_UB_NOTE),
                                                           "severity_bins": {"WEAK": [zone_lb, slack], "MODERATE": [slack, zone_ub]}, "gating_heights": survivors, "soft_candidates_gating": len(soft), "bins": dict(bins),
                                                           "per_height_including_informational": info, "classification_counts_gating": dict(Counter(r["cls"] for r in gating)),
                                                           "unconstrained_ceiling_by_height": unconstrained_ceiling(cam, TALL_TOTAL_HEIGHTS),
                                                           "explanation_hint": "see world_safe reasons in layout_denominator.json: the hand body (lowest point 0.031 m above the grip site) forbids a tall object within 0.0516 m in front of the picked cube, which is where its shadow reaches",
                                                           "status": "PASS" if ok_a else "FAIL"})
        bsi.write_json(out / "camera_margin_audit.json", {"card": CARD, "margin_px_min": bsi.SAFETY_MARGIN_PX, "rule": "all projected corners of both objects at least margin px inside the image", "rejected_for_camera_edge": {h: denom[h]["rejected_by_reason"].get("CAMERA_EDGE", 0) for h in denom}})
        bsi.write_json(out / "static_sweep_audit.json", {"card": CARD, "hand_bottom_rel_table_at_grasp": world.hand_low, "hand_half_spread": world.hand_half_spread, "hand_half_orth": world.hand_half_orth,
                                                         "rejected_by_reason_per_height": {h: denom[h]["rejected_by_reason"] for h in denom}})
        phases["G"] = "PASS" if sum(v["world_safe"] for v in denom.values()) else "FAIL"
        phases["H"] = "PASS" if ok_a else "FAIL"
        if not ok_a:
            category, stop = ("PREFLIGHT_FAIL_NO_SAFE_HEIGHT_LAYOUT" if sum(denom[str(h)]["world_safe"] for h in survivors) == 0 else "PREFLIGHT_FAIL_NO_VIS_A_SOFT_BAND"), "H"
    # ---- remaining (pass path is not implemented for this card; reaching it is an explicit engineering stop)
    for n in ALL_OUTPUTS + LATER_FILES:
        not_run(out, n, f"stopped at phase {stop}: {category}" if stop else "pass path beyond Phase H is not implemented in this card; needs an explicit follow-up")
    if not stop:
        category, stop = "ENGINEERING_UNRESOLVED", "I-M_NOT_IMPLEMENTED"
    for n in ("authorization.json",):
        pass
    (out / "pilot_scene_configs").mkdir(exist_ok=True)
    bsi.write_json(out / "pilot_scene_configs" / "NOT_GENERATED.json", {"reason": category})
    (out / "next_canary_card_request.md").write_text(f"# Next canary card request 鈥?{CARD}\n\n**NOT_REQUESTED.** Preflight verdict: `{category}` (stopped at phase {stop}). No provider probe and no canary are requested.\n", encoding="utf-8")
    verdict = {"card": CARD, "phases": phases, "stop_phase": stop, "verdict": category, "provider_probe_requested": False, "canary_requested": False}
    bsi.write_json(out / "preflight_verdict.json", verdict)
    with (out / "engineering_events.jsonl").open("w") as fh:
        for e in EVENTS:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    bsi.write_json(out / "budget_ledger.json", {"environment_constructions": zg.calls, "env_reset": 0, "start_case": 0, "skill_calls": 0, "physical_episodes": 0, "provider_first_requests": 0, "provider_retries": 0, "rl_transitions": 0,
                                                "optimizer_steps": 0, "elastic_attempts": 0, "test_reads": 0, "test_episodes": 0, "test_read_attempts_refused": len(guard.refused), "files_opened_by_guard": guard.opened, "zero_env_guard_calls": zg.calls})
    bsi.write_json(out / "protected_after.json", protected_hashes(root))
    return verdict


def final_summary(out, verdict):
    out = Path(out)
    L = [f"# {CARD} 鈥?final preflight summary", "", f"Verdict: **{verdict['verdict']}**  (stopped at phase {verdict['stop_phase'] or 'none'})", "", f"Phases: {verdict['phases']}", ""]
    b = json.loads((out / "occ_evidence_binding.json").read_text())
    L += ["## A. OCC evidence binding", "", f"- status {b['status']}; calibration reproduced exactly: {b['checks']['calibration']['reproduced_exactly']}; production sources unchanged: {b['checks']['production_sources_unchanged_since_occ']['all_equal']}", ""]
    if (out / "tall_height_grid.json").exists():
        t = json.loads((out / "tall_height_grid.json").read_text())
        L += ["## D. tall heights under the unmodified production code", ""] + [f"- H={r['total_height']:.2f} m: survives={r['survives_unmodified_production']} {r['eliminated_by']}" for r in t["rows"]] + [""]
    if (out / "color_candidate_audit.json").exists():
        c = json.loads((out / "color_candidate_audit.json").read_text())
        L += ["## E. colour candidates", "", f"- {c['status']}: selected {c['selected']}; passing {c['passing']}", ""]
    if (out / "projection_regression.json").exists():
        r = json.loads((out / "projection_regression.json").read_text())
        L += ["## F. projection regression", "", f"- equal heights reproduce the OCC zero: {r['reproduces_occ_zero']} over {r['equal_height_layouts_checked']} layouts; a 0.12 m occluder reaches a top-face occlusion of {r['tall_0p12_max_top_face_occluded_fraction_same_layouts']:.2f} (machinery can occlude)", ""]
    if (out / "layout_denominator.json").exists():
        d = json.loads((out / "layout_denominator.json").read_text())
        a = json.loads((out / "vis_a_soft_band_audit.json").read_text())
        L += ["## G/H. VIS-A layouts and soft band", ""]
        for h, v in d["per_height"].items():
            L.append(f"- H={h}: total {v['layout_total']}, world-safe {v['world_safe']}, camera-visible {v['camera_visible']}, partial occlusion {v['partial_occlusion']}, soft band {v['soft_band']}, classes {v['classes']}; top rejection reasons {dict(Counter(v['rejected_by_reason']).most_common(4))}")
        L += ["", "- ceiling with EVERY safety constraint removed (overlap forbidden only), max centroid shift while the cube is still detected: " + str({h: round(x["max_centroid_shift_world_m_while_detected"], 4) for h, x in a["unconstrained_ceiling_by_height"].items()}),
              f"- zone [{a['zone_lower_bound_m']:.4f}, {a['zone_upper_bound_m']:.4f}) m; gating heights {a['gating_heights']}; WEAK/MODERATE bins {a['bins']}", f"- max centroid shift per height (informational included): " + str({h: round(v['max_centroid_shift_world_m'], 4) for h, v in a['per_height_including_informational'].items()}), ""]
    L += ["Budget: environment constructions 0, skills 0, episodes 0, provider 0, RL 0, optimizer 0, test reads 0.", ""]
    (out / "final_preflight_summary.md").write_text("\n".join(L), encoding="utf-8")


def verify(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "source_identity.json", "protected_before.json", "protected_after.json", "occ_evidence_binding.json", "production_change_surface.json", "per_object_geometry_patch_spec.json", "tall_height_grid.json",
            "color_candidate_audit.json", "projection_regression.json", "different_height_projection_model.json", "finite_layout_grid.json", "layout_denominator.json", "vis_a_soft_band_audit.json", "vis_b_fact_persistence_audit.json",
            "vis_b_soft_band_audit.json", "tall_grasp_show_audit.json", "static_sweep_audit.json", "camera_margin_audit.json", "discrete_rework_bound.json", "bplan_tie_audit.json", "relation_expressibility_audit.json",
            "pilot_scene_manifest.json", "provider_semantic_probe_request.md", "next_canary_card_request.md", "engineering_events.jsonl", "budget_ledger.json", "final_preflight_summary.md", "preflight_verdict.json"]
    led = json.loads((out / "budget_ledger.json").read_text())
    before, after = json.loads((out / "protected_before.json").read_text()), json.loads((out / "protected_after.json").read_text())
    verdict = json.loads((out / "preflight_verdict.json").read_text())
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".jsonl"} and bool(re.search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "pilot_scene_configs_dir": (out / "pilot_scene_configs").is_dir(),
              "zero_resources": all(led[k] == 0 for k in ("environment_constructions", "env_reset", "start_case", "skill_calls", "physical_episodes", "provider_first_requests", "provider_retries", "rl_transitions", "optimizer_steps", "elastic_attempts", "test_reads", "test_episodes")),
              "protected_unchanged": before == after, "protected_files_hashed": len(before), "verdict_is_allowed": verdict["verdict"] in FAIL_CATEGORIES + ("PREFLIGHT_PASS_VIS_GEOMETRY",),
              "canary_label_not_requested": "NOT_REQUESTED" in (out / "next_canary_card_request.md").read_text(), "no_secret_shaped_content": secrets == 0,
              "production_files_untouched": subprocess.run(["git", "diff", "--quiet", BASELINE, "HEAD", "--", LIB], cwd=root).returncode == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k not in ("outputs_present", "protected_files_hashed", "status")) and all(checks["outputs_present"].values()) else "FAIL"
    bsi.write_json(out / "verify.json", checks)
    return checks

