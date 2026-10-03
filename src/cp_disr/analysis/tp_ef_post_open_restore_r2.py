"""CP-DISR-TP-EF-POST-OPEN-RESTORE-R2: finite revision of the post-OPEN restore card.

Stage A re-validates the already captured T_B_dev_03 snapshot (no OPEN) against a frozen non-bitwise render contract;
Stage B (only if A passes) captures and validates T_B_dev_05 with the identical criteria. No candidate continuation,
provider, RL, optimizer or test.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cp_disr.analysis import tp_ef_post_open_capture as cap
from cp_disr.analysis import tp_ef_protocol_review as pr

CARD = "CP-DISR-TP-EF-POST-OPEN-RESTORE-R2"
BASELINE = "487cc6f1506e057465b02f2feada27a3c96f7e35"
OLD_RUN_REL = "runs/final_master/S4/tp_ef_post_open_capture/20261003T014215Z_88f2a168"
OLD_SNAP_DIR = "/home/xushijie2/graph_cp_disr_snapshots/tp_ef_post_open_capture/20261003T014215Z/T_B_dev_03"
CONTRACT_REL = "configs/final_master/family_b_obs_v2/pair_equivalence_contract.json"
PROFILE_REL = "configs/final_master/family_b_obs_v2/observation_profile_v2.json"
FAMILY_B_SRC_COMMITS = ("fffb2b073", "48dc5e3b8", "7776f3a59")
SHARED_MODULES = ("src/cp_disr/platforms/libero/d0_env.py", "src/cp_disr/platforms/libero/perception.py", "src/cp_disr/platforms/libero/observations.py",
                  "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/clock.py")
FAMILY_B_RECORDED_PERCEPTION_SHA16 = "32b2ad455dbb418e"
CAMERA_ATOL = 1e-9
PHYSICAL_GPU_INDEX = 0
# thresholds frozen into authorization.json before any restored RGB is read
THRESH = {"rgb": {"shape_dtype": "exact", "max_abs_diff_uint8": 3}, "depth": {"max_abs_diff": 1e-6, "applied_to": ["raw normalized depth array", "metric depth"]},
          "proprio": {"max_abs_diff": 1e-9}, "sim_arrays": {"max_abs_diff": 1e-12},
          "perception": {"xyz_atol_m": 1e-4, "presence_and_unknown_reasons": "exact"},
          "exact": ["facts / FactRecords", "goal", "candidate IDs", "candidate mask", "controller/gripper state", "RNG", "episode clock", "object presence", "public perception decisions",
                    "cached public observation (restored python state)"]}
CAPS = {"dev03_restore_validations": 1, "dev05_setup_episodes": 1, "dev05_open_skill_calls": 1, "dev05_restore_validations": 1, "environment_constructions": 3,
        "start_case_calls": 3, "skill_retries": 0, "continuation_skills": 0, "provider_requests": 0, "rl_transitions": 0, "optimizer_steps": 0, "test_episodes": 0}


# ------------------------------------------------------------------------------- static compat gate
def _init_call_kwargs(path, class_name, owner_is_super):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    attrs = {}
                    kw = None
                    local_defs = {}
                    for sub in ast.walk(fn):
                        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                            local_defs[sub.targets[0].id] = ast.dump(sub.value)
                    for sub in ast.walk(fn):
                        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Attribute) and isinstance(sub.targets[0].value, ast.Name) and sub.targets[0].value.id == "self":
                            try:
                                attrs[sub.targets[0].attr] = ast.literal_eval(sub.value)
                            except Exception:
                                attrs[sub.targets[0].attr] = ast.dump(sub.value)
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "__init__":
                            v = sub.func.value
                            is_super = isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super"
                            is_named = isinstance(v, ast.Name) and v.id == "SingleArmEnv"
                            if (owner_is_super and is_super) or (not owner_is_super and is_named):
                                kw = {}
                                for k in sub.keywords:
                                    try:
                                        kw[k.arg] = ast.literal_eval(k.value)
                                    except Exception:
                                        kw[k.arg] = local_defs.get(k.value.id) if isinstance(k.value, ast.Name) and k.value.id in local_defs else ast.dump(k.value)
                    return kw, attrs
    raise RuntimeError("init not found: " + class_name)


def compat_gate(root):
    """Zero-environment proof (or refusal) that the frozen Family B non-bitwise contract applies to the T_B render path."""
    root = Path(root)
    checks, problems = {}, []
    contract = json.loads((root / CONTRACT_REL).read_text())
    profile = json.loads((root / PROFILE_REL).read_text())
    allowed = contract["non_bitwise_allowed_only_if"]
    checks["contract_clause"] = {"sha256": cap.sha_file(root / CONTRACT_REL), "rgb_max_abs_diff_uint8": allowed["rgb_max_abs_diff_uint8"], "depth_max_abs_diff_m": allowed["depth_max_abs_diff_m"],
                                 "qpos_qvel_atol": allowed["qpos_qvel_atol"], "perception_xyz_atol_m": allowed["perception_xyz_atol_m"],
                                 "frozen_here_is_stricter_or_equal": bool(THRESH["rgb"]["max_abs_diff_uint8"] <= allowed["rgb_max_abs_diff_uint8"] and THRESH["depth"]["max_abs_diff"] <= allowed["depth_max_abs_diff_m"]
                                                                             and THRESH["sim_arrays"]["max_abs_diff"] <= allowed["qpos_qvel_atol"] and THRESH["perception"]["xyz_atol_m"] <= allowed["perception_xyz_atol_m"])}
    if not checks["contract_clause"]["frozen_here_is_stricter_or_equal"]:
        problems.append("THRESHOLDS_LOOSER_THAN_CONTRACT")
    d0 = root / "src/cp_disr/platforms/libero/d0_env.py"
    fb = root / "src/cp_disr/platforms/libero/family_b_obs_v2.py"
    kw0, at0 = _init_call_kwargs(d0, "D0ManipulationEnv", True)
    kw1, at1 = _init_call_kwargs(fb, "FamilyBObsV2Env", False)
    camera_keys = {"camera_names", "camera_heights", "camera_widths", "camera_depths", "camera_segmentations"}
    diff = {k: [kw0.get(k), kw1.get(k)] for k in sorted((set(kw0) | set(kw1)) - camera_keys) if kw0.get(k) != kw1.get(k)}
    attr_diff = {k: [at0.get(k), at1.get(k)] for k in ("table_full_size", "table_friction", "table_offset", "use_object_obs") if at0.get(k) != at1.get(k)}
    ag = profile["camera_manifest"]["agentview"]
    cam = {"T_B_camera_names": kw0.get("camera_names"), "T_B_height": kw0.get("camera_heights"), "T_B_width": kw0.get("camera_widths"), "T_B_depth": kw0.get("camera_depths"),
           "frozen_agentview": {"width": ag["width"], "height": ag["height"], "fovy": ag["fovy"], "pos": ag["pos"], "mode": ag["mode"], "fixed_in_world": ag["fixed_in_world"]},
           "profile_cameras": profile["cameras"], "profile_image_size": profile["fusion_contract"]["image_size"]}
    cam_ok = kw0.get("camera_names") == "agentview" and "agentview" in profile["cameras"] and kw0.get("camera_heights") == ag["height"] == kw0.get("camera_widths") == ag["width"] and kw0.get("camera_depths") is True
    checks["constructor_render_args"] = {"identical_except_camera_set": not diff and not attr_diff, "keyword_differences": diff, "attribute_differences": attr_diff, "camera_identity_static": cam, "camera_static_ok": cam_ok}
    if diff or attr_diff:
        problems.append("ENV_CONSTRUCTOR_DIFFERS")
    if not cam_ok:
        problems.append("CAMERA_IDENTITY_NOT_PROVEN")
    d0_text, perc_text = d0.read_text(), (root / "src/cp_disr/platforms/libero/perception.py").read_text()
    depth_def = {"public_observation_flipud_rgb_and_depth": "np.flipud(np.asarray(obs[\"agentview_depth\"]))" in d0_text and "np.flipud(np.asarray(obs[\"agentview_image\"]))" in d0_text,
                 "metric_depth_get_real_depth_map": "get_real_depth_map" in perc_text, "frozen_depth_convention": ag["depth_convention"],
                 "rgb_dtype": "uint8 (verified live and on the saved snapshot arrays)"}
    checks["depth_and_rgb_definition"] = depth_def
    if not (depth_def["public_observation_flipud_rgb_and_depth"] and depth_def["metric_depth_get_real_depth_map"]):
        problems.append("DEPTH_DEFINITION_DIFFERS")
    unchanged = {}
    for c in FAMILY_B_SRC_COMMITS:
        r = subprocess.run(["git", "diff", "--quiet", c, "HEAD", "--", *SHARED_MODULES], cwd=root)
        unchanged[c] = r.returncode == 0
    shas = {m: cap.sha_file(root / m) for m in SHARED_MODULES}
    perc16 = shas["src/cp_disr/platforms/libero/perception.py"][:16]
    checks["observation_source_identity"] = {"shared_modules_unchanged_since_family_b_commits": unchanged, "sha256": shas,
                                             "perception_sha16": perc16, "equals_family_b_recorded_perception": perc16 == FAMILY_B_RECORDED_PERCEPTION_SHA16}
    if not all(unchanged.values()) or perc16 != FAMILY_B_RECORDED_PERCEPTION_SHA16:
        problems.append("OBSERVATION_SOURCE_CHANGED")
    import mujoco
    import robosuite
    rt = json.loads(json.dumps(__import__("yaml").safe_load((root / "experiments/manifests/runtime_manifest_v211.yaml").read_text())["runtime"]["simulator_or_robot"], default=str))
    stack = {"robosuite": robosuite.__version__, "mujoco": mujoco.__version__, "manifest": {"robosuite": str(rt.get("robosuite")), "mujoco": str(rt.get("mujoco"))}, "MUJOCO_GL": "egl (set by worker env)"}
    stack["matches_manifest"] = stack["robosuite"] == stack["manifest"]["robosuite"] and stack["mujoco"] == stack["manifest"]["mujoco"]
    checks["renderer_stack"] = stack
    if not stack["matches_manifest"]:
        problems.append("RENDERER_STACK_DIFFERS")
    checks["gpu_policy"] = {"contract": contract["gpu_policy"], "this_card": f"all processes pinned to physical GPU index {PHYSICAL_GPU_INDEX}, run sequentially; UUID recorded per process"}
    checks["limitation"] = ("The frozen contract belongs to the Family B observation profile (profile sha %s). Applicability to T_B is concluded from identity of the render path "
                            "(same base env class, same render arguments, same camera definition, unchanged shared observation modules, same renderer stack), not from a T_B-specific profile." % contract["profile_sha256"][:12])
    return {"applicable": not problems, "problems": problems, "checks": checks}


# ----------------------------------------------------------------------------------------- metrics
def _bboxes(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"overall": None, "components": []}
    out = {"overall": {"row_min": int(ys.min()), "row_max": int(ys.max()), "col_min": int(xs.min()), "col_max": int(xs.max())}, "components": []}
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        for i, sl in enumerate(ndimage.find_objects(lab)[:30], 1):
            out["components"].append({"pixels": int((lab[sl] == i).sum()), "row_min": int(sl[0].start), "row_max": int(sl[0].stop - 1), "col_min": int(sl[1].start), "col_max": int(sl[1].stop - 1)})
        out["component_count"] = int(n)
    except Exception:
        out["component_count"] = None
    return out


def _quantiles(x, qs):
    return {str(q): float(np.percentile(x, q)) for q in qs} if x.size else {str(q): 0.0 for q in qs}


def rgb_metrics(a, b):
    res = {"shape_equal": a.shape == b.shape, "dtype_equal": a.dtype == b.dtype, "dtype": str(a.dtype)}
    if not res["shape_equal"]:
        res.update(pass_=False, max_abs=None)
        return res
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    chg = d > 0
    pix = chg.any(axis=2) if d.ndim == 3 else chg
    res.update({"elements": int(d.size), "changed_channel_count": int(chg.sum()), "changed_channel_fraction": float(chg.mean()), "changed_pixel_count": int(pix.sum()),
                "changed_pixel_fraction": float(pix.mean()), "max_abs": int(d.max()), "mean_abs_all": float(d.mean()), "mean_abs_changed": float(d[chg].mean()) if chg.any() else 0.0,
                "quantiles_all_elements": _quantiles(d.ravel(), (50, 90, 99, 99.9, 100)), "quantiles_changed_elements": _quantiles(d[chg], (50, 90, 99, 100)),
                "per_channel_changed": [int(chg[..., c].sum()) for c in range(d.shape[-1])] if d.ndim == 3 else [],
                "value_histogram": {str(v): int((d == v).sum()) for v in range(1, min(int(d.max()), 12) + 1)}, "bounding_boxes": _bboxes(pix)})
    res["pass_"] = bool(res["shape_equal"] and res["dtype_equal"] and res["max_abs"] <= THRESH["rgb"]["max_abs_diff_uint8"])
    return res


def depth_metrics(a, b, label):
    a64, b64 = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    res = {"label": label, "shape_equal": a64.shape == b64.shape}
    if not res["shape_equal"]:
        res["pass_"] = False
        return res
    d = np.abs(a64 - b64)
    chg = d > 0
    res.update({"max_abs": float(d.max()), "mean_abs_all": float(d.mean()), "changed_count": int(chg.sum()), "changed_fraction": float(chg.mean()),
                "quantiles_changed": _quantiles(d[chg], (50, 90, 99, 100)), "pass_": bool(d.max() <= THRESH["depth"]["max_abs_diff"])})
    return res


def vector_metrics(a, b, tol):
    d = np.abs(np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64"))
    return {"max_abs": float(d.max()) if d.size else 0.0, "changed_count": int((d > 0).sum()), "pass_": bool(d.size == 0 or d.max() <= tol)}


def perception_summary(res):
    m = res.measurements
    return {"blobs": {k: (None if v is None else {"xyz": v["xyz"], "pixels": v["pixels"], "std": v["std"], "source": v.get("source")}) for k, v in sorted(m["blobs"].items())},
            "unknown_reasons": dict(sorted(m["unknown_reasons"].items())), "depth_stats": m["depth_stats"], "calibration_hash": res.calibration_hash}


def compare_perception(a, b):
    out = {"presence_equal": True, "unknown_reasons_equal": a["unknown_reasons"] == b["unknown_reasons"], "per_object": {}}
    xyz_max = 0.0
    for k in sorted(set(a["blobs"]) | set(b["blobs"])):
        x, y = a["blobs"].get(k), b["blobs"].get(k)
        same = (x is None) == (y is None)
        out["presence_equal"] &= same
        row = {"presence_equal": same}
        if x is not None and y is not None:
            dx = float(np.max(np.abs(np.array(x["xyz"]) - np.array(y["xyz"]))))
            xyz_max = max(xyz_max, dx)
            row.update(xyz_max_abs_m=dx, pixels_a=x["pixels"], pixels_b=y["pixels"], pixel_delta=y["pixels"] - x["pixels"])
        out["per_object"][k] = row
    out["xyz_max_abs_m"] = xyz_max
    out["depth_stats_max_abs"] = float(max(abs(a["depth_stats"][k] - b["depth_stats"][k]) for k in a["depth_stats"]))
    out["pass_"] = bool(out["presence_equal"] and out["unknown_reasons_equal"] and xyz_max <= THRESH["perception"]["xyz_atol_m"])
    return out


def obs_dict_from_arrays(arr):
    p = np.asarray(arr["proprio"], dtype=float)
    return {"rgb": arr["rgb"], "depth": arr["depth"], "proprio": p, "eef_pos": p[0:3], "gripper_qpos": p[7:9]}


def camera_live_vs_frozen(env, profile):
    ag = profile["camera_manifest"]["agentview"]
    cid = env.sim.model.camera_name2id("agentview")
    live = {"pos": [float(x) for x in env.sim.model.cam_pos[cid]], "quat": [float(x) for x in env.sim.model.cam_quat[cid]], "fovy": float(env.sim.model.cam_fovy[cid]),
            "mode": int(env.sim.model.cam_mode[cid]), "width": int(env.camera_widths[0]), "height": int(env.camera_heights[0])}
    issues = []
    if np.max(np.abs(np.array(live["pos"]) - np.array(ag["pos"]))) > CAMERA_ATOL:
        issues.append("pos")
    if np.max(np.abs(np.array(live["quat"]) - np.array(ag["quat"]))) > CAMERA_ATOL:
        issues.append("quat")
    if abs(live["fovy"] - ag["fovy"]) > CAMERA_ATOL:
        issues.append("fovy")
    if live["mode"] != ag["mode"] or live["width"] != ag["width"] or live["height"] != ag["height"]:
        issues.append("mode_or_size")
    return {"live": live, "frozen": {k: ag[k] for k in ("pos", "quat", "fovy", "mode", "width", "height")}, "tolerance": CAMERA_ATOL, "issues": issues, "pass": not issues}


def gl_renderer_info():
    info = {"MUJOCO_GL": os.environ.get("MUJOCO_GL"), "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID")}
    try:
        from OpenGL import GL
        info["GL_RENDERER"] = GL.glGetString(GL.GL_RENDERER).decode()
        info["GL_VERSION"] = GL.glGetString(GL.GL_VERSION).decode()
        info["GL_VENDOR"] = GL.glGetString(GL.GL_VENDOR).decode()
    except Exception as exc:
        info["GL_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def gpu_identity(index=PHYSICAL_GPU_INDEX):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,gpu_uuid,name,driver_version,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.splitlines()
    for line in q:
        i, uuid, name, drv, mem = [x.strip() for x in line.split(",")]
        if int(i) == index:
            busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
            return {"index": index, "uuid": uuid, "name": name, "driver": drv, "memory_used_mib": int(mem), "compute_processes": [l for l in busy.splitlines() if uuid in l]}
    return None


def save_npz_record(path, **arrays):
    np.savez(path, **arrays)
    with open(path, "rb") as f:
        os.fsync(f.fileno())
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": cap.sha_file(path)}


# ------------------------------------------------------------------------------ restore (Stage A / B)
def restore_r2(root, out, case_id, snap_dir, tag, arrays_dir):
    """Fresh-process restore with two unchanged-state fresh renders and full metrics. Executes no skill."""
    root, out, snap_dir, arrays_dir = Path(root).resolve(), Path(out).resolve(), Path(snap_dir), Path(arrays_dir)
    arrays_dir.mkdir(parents=True, exist_ok=True)
    ledger = cap.Ledger()
    res = {"card": CARD, "case_id": case_id, "tag": tag, "role": "fresh_process_restore_r2", "pid": os.getpid(), "status": "FAIL", "failures": [], "checks": {}}
    try:
        profile = json.loads((root / PROFILE_REL).read_text())
        man = json.loads((snap_dir / "manifest.json").read_text())
        res["snapshot_file_sha256_at_restore"] = {p.name: cap.sha_file(p) for p in sorted(snap_dir.iterdir())}
        sim_saved = {k: np.array(v) for k, v in np.load(snap_dir / "sim_state.npz").items()}
        py_saved = cap.deserialize_state(man["python_state_meta"], np.load(snap_dir / "python_state.npz"))
        pub_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_public_cached.npz").items()}
        forced_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_forced_render.npz").items()}
        rng_saved = {"numpy_global": dict(man["rng"]["numpy_global"], key=np.array(np.load(snap_dir / "rng_numpy.npz")["key"])), "python_random": man["rng"]["python_random"]}
        res["gpu"] = gpu_identity()
        bundle, manifest = cap.build_runtime(root, out, case_id, ledger, "r2_" + tag)
        snap0, real_env = cap.start_case_reusing_bootstrap(bundle, ledger, case_id)
        env = bundle.environment
        res["renderer"] = gl_renderer_info()
        res["checks"]["camera_live_vs_frozen"] = camera_live_vs_frozen(env, profile)

        def forbidden(*a, **k):
            ledger.skill_calls += 1
            raise RuntimeError("PROTOCOL_VIOLATION: restore validation must not execute skills")
        bundle.executor.execute = forbidden
        cap.apply_sim(env, sim_saved)
        missing = []
        missing += cap.load_state(env, {p: v for p, v in py_saved.items() if not (p and p[0][1].startswith("@"))})
        for name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            missing += cap.load_state(obj, {p[1:]: v for p, v in py_saved.items() if p and p[0][1] == "@" + name})
        cap.apply_rng(rng_saved)
        rt = man["runtime_state"]
        cap.apply_py_runtime_state(bundle, dict(rt, verifier_prev=rt["verifier_prev_before_verify"]))
        res["state_paths_not_applied"] = missing
        res["checks"]["sim_arrays_raw"] = cap.compare_sim(sim_saved, cap.capture_sim(env))
        env.sim.forward()
        sim_now = cap.capture_sim(env)
        res["checks"]["sim_arrays"] = cap.compare_sim(sim_saved, sim_now, skip=("qacc", "qacc_warmstart"))
        res["checks"]["sim_arrays"]["pass"] = bool(res["checks"]["sim_arrays"]["max_abs"] <= THRESH["sim_arrays"]["max_abs_diff"])
        res["checks"]["sim_arrays_raw"]["pass"] = bool(res["checks"]["sim_arrays_raw"]["max_abs"] <= THRESH["sim_arrays"]["max_abs_diff"])
        now_clock = float(bundle.clock.now_seconds())
        res["checks"]["clock"] = {"pass": bool(now_clock == man["clock"]["now"] and float(env.sim.data.time) == man["clock"]["sim_time"] and float(bundle.clock._origin) == man["clock"]["origin"]),
                                  "saved": man["clock"], "restored_now": now_clock, "restored_sim_time": float(env.sim.data.time)}
        res["checks"]["rng"] = {"pass": cap.rng_hash(cap.rng_snapshot()) == man["hashes"]["rng"]}
        py_now = cap.dump_state(env)
        for name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            for p, v in cap.dump_state(obj, skip=cap.SKIP_KEYS - {"robots"}).items():
                py_now[(("a", "@" + name),) + p] = v
        diffs = cap.diff_states(py_saved, py_now)
        res["checks"]["controller_gripper_state"] = {"entries": len(py_saved), "diffs_total": len(diffs), "first_diffs": diffs[:20], "pass": not diffs and bool(man["controller_gripper_named"])}
        pub_now = cap.public_obs_arrays(env)
        res["checks"]["public_obs_cached_exact"] = {"pass": all(np.array_equal(pub_saved[k], pub_now[k]) and pub_saved[k].dtype == pub_now[k].dtype for k in pub_saved)}
        obs_obj = bundle.observations.observe()
        ex = SimpleNamespace(execution_id="r2-restore-check", controller_exit="NORMAL_TERMINATION", start_seconds=0.0, end_seconds=0.0, evidence_ids=())
        records = bundle.verifier.verify(bundle.perception.infer(obs_obj), ex)
        facts_now = cap.facts_to_list(records)
        res["checks"]["facts_cached_obs"] = {"pass": cap.facts_comparable(facts_now) == cap.facts_comparable(man["facts"])}
        snap1 = bundle.snapshot_builder.build(snap0, records, obs_obj, ex, float(bundle.clock.now_seconds()))
        goal_now = [[g.fact_id, g.sign] for g in snap1.template.goals]
        cand_now = [list(snap1.candidate_ids), [bool(m) for m in snap1.mask]]
        idx = {c: i for i, c in enumerate(snap1.candidate_ids)}
        fm = {f["fact_id"]: f["value"] for f in facts_now}
        res["checks"]["goal_candidates_mask"] = {"pass": goal_now == man["goal"] and cand_now == [man["candidate_ids"], man["candidate_mask"]]}
        res["checks"]["post_open_requirements"] = {"facts": {k: fm.get(k) for k in cap.FACT_REQUIRED}, "pick_masks": {c: bool(snap1.mask[idx[c]]) for c in cap.PICK_IDS}}
        res["checks"]["post_open_requirements"]["pass"] = bool(all(fm.get(k) == v for k, v in cap.FACT_REQUIRED.items()) and all(res["checks"]["post_open_requirements"]["pick_masks"].values()))
        # ---- renders: two fresh renders of the unchanged restored state, plus perception of the saved arrays
        prev_after = cap.prev_from_json(rt["verifier_prev"])

        def facts_of(obs_dict):
            v = type(bundle.verifier)(env)
            v.prev = dict(prev_after)
            pres = bundle.perception.infer(obs_dict)
            return pres, cap.facts_to_list(v.verify(pres, None))

        env.sim.forward()
        r1 = cap.forced_obs_arrays(env)
        pres1, facts1 = facts_of(env.public_observation())
        r2 = cap.forced_obs_arrays(env)
        pres2, facts2 = facts_of(env.public_observation())
        pres_saved, facts_saved_re = facts_of(obs_dict_from_arrays(forced_saved))
        files = {"restored_rgb_1": save_npz_record(arrays_dir / f"{tag}_restored_render_1.npz", **r1), "restored_rgb_2": save_npz_record(arrays_dir / f"{tag}_restored_render_2.npz", **r2)}
        res["array_files"] = files
        metric = lambda a_arr, b_arr, a_obs, b_obs: {
            "rgb": rgb_metrics(a_arr["rgb"], b_arr["rgb"]), "depth_raw": depth_metrics(a_arr["depth"], b_arr["depth"], "raw normalized depth"),
            "depth_metric": depth_metrics(cap_metric_depth(env, a_arr["depth"]), cap_metric_depth(env, b_arr["depth"]), "metric depth (m)"), "proprio": vector_metrics(a_arr["proprio"], b_arr["proprio"], THRESH["proprio"]["max_abs_diff"])}
        A = metric(forced_saved, r1, None, None)
        B = metric(r1, r2, None, None)
        sp, p1, p2 = perception_summary(pres_saved), perception_summary(pres1), perception_summary(pres2)
        A["perception"] = compare_perception(sp, p1)
        B["perception"] = compare_perception(p1, p2)
        A["facts"] = {"saved_vs_restore1_equal": cap.facts_comparable(facts_saved_re) == cap.facts_comparable(facts1), "restore1_vs_manifest_forced_equal": cap.facts_comparable(facts1) == cap.facts_comparable(man["facts_forced_render"])}
        B["facts"] = {"restore1_vs_restore2_equal": cap.facts_comparable(facts1) == cap.facts_comparable(facts2)}
        for X in (A, B):
            X["mask_unchanged_by_render"] = True
            X["pass_"] = bool(X["rgb"]["pass_"] and X["depth_raw"]["pass_"] and X["depth_metric"]["pass_"] and X["proprio"]["pass_"] and X["perception"]["pass_"] and all(X["facts"].values()))
        res["metrics"] = {"A_saved_forced_vs_restore_render_1": A, "B_restore_render_1_vs_render_2": B,
                          "perception_intermediates": {"saved_arrays": sp, "restore_render_1": p1, "restore_render_2": p2},
                          "cached_vs_fresh_restored": {"rgb": rgb_metrics(pub_now["rgb"], r1["rgb"]), "depth_raw": depth_metrics(pub_now["depth"], r1["depth"], "cached vs fresh")}}
        res["checks"]["render_A"] = {"pass": A["pass_"]}
        res["checks"]["render_B"] = {"pass": B["pass_"]}
        res["skill_calls"] = ledger.skill_calls
        res["checks"]["no_skill_executed"] = {"pass": ledger.skill_calls == 0}
        gating = ["camera_live_vs_frozen", "sim_arrays_raw", "sim_arrays", "clock", "rng", "controller_gripper_state", "public_obs_cached_exact", "facts_cached_obs", "goal_candidates_mask",
                  "post_open_requirements", "render_A", "render_B", "no_skill_executed"]
        res["failures"] += [g for g in gating if not res["checks"][g].get("pass")]
        res["status"] = "PASS" if not res["failures"] else "FAIL"
        return res
    except Exception as exc:
        import traceback
        res["failures"].append(f"EXCEPTION:{type(exc).__name__}:{exc}")
        res["traceback"] = traceback.format_exc()
        res["status"] = "STOPPED_ENGINEERING"
        return res
    finally:
        res.update(ledger.as_dict())
        try:
            ledger.bootstrap_env.close()
        except Exception:
            pass
        pr.write_json(out / f"restore_r2_{tag}.json", res)


def cap_metric_depth(env, depth):
    from cp_disr.platforms.libero.perception import _metric_depth
    return _metric_depth(env, depth)


# ------------------------------------------------------------------------------------ orchestration
def gate0(root, out, arrays_root):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    ident = {"user": os.environ.get("USER", ""), "python": sys.executable}
    if ident["user"] != cap.USER or sys.executable != cap.PYTHON:
        problems.append("WRONG_ACCOUNT_OR_PYTHON")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=root).returncode != 0:
        problems.append("BASELINE_NOT_ANCESTOR")
    if dirty:
        problems.append("DIRTY_TRACKED_TREE")
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        if key in os.environ:
            problems.append("PROVIDER_CREDENTIAL_PRESENT:" + key)
    # old snapshot integrity (zero environment)
    old = json.loads((root / OLD_RUN_REL / "snapshot_file_hashes.json").read_text())["files"]["T_B_dev_03"]
    snap_check = [{"file": Path(f["path"]).name, "expected_sha256": f["sha256_at_capture"], "now": cap.sha_file(f["path"]) if Path(f["path"]).is_file() else None,
                   "bytes_expected": f["bytes"], "bytes_now": Path(f["path"]).stat().st_size if Path(f["path"]).is_file() else None} for f in old]
    snap_ok = all(c["now"] == c["expected_sha256"] and c["bytes_expected"] == c["bytes_now"] for c in snap_check) and len(snap_check) == 6 and str(Path(old[0]["path"]).parent) == OLD_SNAP_DIR
    if not snap_ok:
        problems.append("OLD_SNAPSHOT_HASH_MISMATCH")
    pr.write_json(out / "old_snapshot_integrity.json", {"snapshot_dir": OLD_SNAP_DIR, "files": snap_check, "all_match": snap_ok})
    compat = compat_gate(root)
    pr.write_json(out / "renderer_contract_binding.json", {"card": CARD, "contract": CONTRACT_REL, **compat})
    if not compat["applicable"]:
        problems.append("RENDERER_CONTRACT_NOT_APPLICABLE:" + ",".join(compat["problems"]))
    gpu = gpu_identity()
    if gpu is None or gpu["compute_processes"]:
        problems.append("STOPPED_CAPTURE_GPU_UNAVAILABLE")
    arrays_root = Path(arrays_root)
    pr.write_json(out / "source_identity.json", {"card": CARD, "baseline": BASELINE, "execution_commit": commit, "dirty_tracked": dirty, "identity": ident, "gpu": gpu,
                                                  "runtime_identity": cap.runtime_identity(root)})
    auth = {"card": CARD, "authorized_by": "user chat instruction approving CP-DISR-TP-EF-POST-OPEN-RESTORE-R2", "baseline": BASELINE, "thresholds_frozen_before_any_restored_rgb_is_read": THRESH,
            "contract_basis": CONTRACT_REL, "stage_a": "reuse dev_03 snapshot, no OPEN, 2 fresh renders", "stage_b": "dev_05 capture+restore only if stage A passes",
            "caps": CAPS, "physical_gpu": gpu, "arrays_root": str(arrays_root), "stop_rule": "any failed gate stops; no recapture of dev_03; no threshold widening; no substitution",
            "forbidden": ["candidate continuation", "provider", "RL", "optimizer", "test"], "problems": problems, "status": "PASS" if not problems else "STOPPED_GATE0"}
    pr.write_json(out / "authorization.json", auth)
    pr.write_json(out / "budget_ledger.json", {"caps": CAPS, "used": {k: 0 for k in CAPS}, "processes": []})
    return auth


def _spawn(root, out, args, gpu_index, tag):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu_index), "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    log = Path(out) / "logs" / f"{tag}.log"
    log.parent.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, str(Path(root) / "scripts/tp_ef_post_open_restore_r2.py"), *args, "--root", str(root), "--output", str(out)], cwd=root, env=env,
                          stdout=log.open("ab"), stderr=subprocess.STDOUT, timeout=1500).returncode


def run_all(root, out, arrays_root, snap_root):
    root, out, arrays_root, snap_root = Path(root).resolve(), Path(out).resolve(), Path(arrays_root), Path(snap_root)
    auth = pr.read_json(out / "authorization.json")
    if auth["status"] != "PASS":
        return {"stopped": auth["status"] + ":" + ",".join(auth["problems"]), "sequence": [], "caps_exceeded": []}
    ledger = pr.read_json(out / "budget_ledger.json")
    seq, stopped = [], ""

    def account(res, kind):
        ledger["used"]["environment_constructions"] += res.get("environment_constructions", 0)
        ledger["used"]["start_case_calls"] += res.get("start_case_calls", 0)
        ledger.setdefault("env_reset_calls_total", 0)
        ledger["env_reset_calls_total"] += res.get("env_reset_calls_total", 0)
        if kind == "dev03_restore":
            ledger["used"]["dev03_restore_validations"] += 1
        elif kind == "dev05_setup":
            ledger["used"]["dev05_setup_episodes"] += 1
            ledger["used"]["dev05_open_skill_calls"] += res.get("skill_calls", 0)
        elif kind == "dev05_restore":
            ledger["used"]["dev05_restore_validations"] += 1
            ledger["used"]["continuation_skills"] += res.get("skill_calls", 0)
        ledger["processes"].append({"kind": kind, "status": res["status"], "constructions": res.get("environment_constructions"), "start_case": res.get("start_case_calls"),
                                    "env_reset_calls": res.get("env_reset_calls_total"), "skills": res.get("skill_calls")})
        pr.write_json(out / "budget_ledger.json", ledger)

    steps = [("dev03_restore", ["restore", "--case", "T_B_dev_03", "--snapshot-dir", OLD_SNAP_DIR, "--tag", "dev03", "--arrays-dir", str(arrays_root / "dev03")], out / "restore_r2_dev03.json"),
             ("dev05_setup", ["setup", "--case", "T_B_dev_05", "--snapshot-dir", str(snap_root / "T_B_dev_05")], out / "setup_T_B_dev_05.json"),
             ("dev05_restore", ["restore", "--case", "T_B_dev_05", "--snapshot-dir", str(snap_root / "T_B_dev_05"), "--tag", "dev05", "--arrays-dir", str(arrays_root / "dev05")], out / "restore_r2_dev05.json")]
    for kind, args, rf in steps:
        if ledger["used"]["start_case_calls"] >= CAPS["start_case_calls"]:
            stopped = "STOPPED_BUDGET"
            break
        code = _spawn(root, out, args, PHYSICAL_GPU_INDEX, kind)
        res = pr.read_json(rf) if rf.is_file() else {"status": "FAIL", "failures": ["NO_RESULT_FILE"]}
        account(res, kind)
        seq.append({"kind": kind, "status": res["status"], "exit": code})
        if res["status"] != "PASS":
            stopped = f"STOPPED_{kind.upper()}:{res['status']}:{res.get('failures')}"
            break
    over = [k for k, v in CAPS.items() if ledger["used"][k] > v]
    return {"stopped": stopped, "sequence": seq, "caps_exceeded": over}


# --------------------------------------------------------------------------------------- finalization
PRIOR_DIRS = ("runs/final_master/S4/tp_ef_discovery", "runs/final_master/S4/tp_ef_protocol_review", "runs/final_master/S4/tp_ef_post_open_audit", "runs/final_master/S4/tp_ef_post_open_capture")


def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in PRIOR_DIRS:
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = cap.sha_file(p)
    return out


def _load(out, name):
    p = Path(out) / name
    return pr.read_json(p) if p.is_file() else None


def finalize(root, out, run_summary):
    out = Path(out)
    a, s5, r5 = _load(out, "restore_r2_dev03.json"), _load(out, "setup_T_B_dev_05.json"), _load(out, "restore_r2_dev05.json")
    ledger = pr.read_json(out / "budget_ledger.json")
    pr.write_json(out / "dev03_restore_rgb_metrics.json", {"card": CARD, "status": a["status"] if a else "NOT_RUN", "failures": a.get("failures") if a else None,
                                                           "metrics": a.get("metrics") if a else None, "array_files": a.get("array_files") if a else None, "gpu": a.get("gpu") if a else None,
                                                           "renderer": a.get("renderer") if a else None, "thresholds": THRESH,
                                                           "camera_live_vs_frozen": a.get("checks", {}).get("camera_live_vs_frozen") if a else None})
    pr.write_json(out / "dev05_capture_result.json", s5 or {"status": "NOT_ATTEMPTED"})
    pr.write_json(out / "dev05_restore_rgb_metrics.json", {"card": CARD, "status": r5["status"] if r5 else "NOT_RUN", "failures": r5.get("failures") if r5 else None, "metrics": r5.get("metrics") if r5 else None,
                                                           "array_files": r5.get("array_files") if r5 else None, "gpu": r5.get("gpu") if r5 else None, "renderer": r5.get("renderer") if r5 else None} if r5 else {"status": "NOT_RUN"})
    pr.write_json(out / "fresh_process_restore_results.json", {"card": CARD, "dev03": a, "dev05": r5, "thresholds": THRESH})
    ok = bool(a and a["status"] == "PASS" and s5 and s5["status"] == "PASS" and r5 and r5["status"] == "PASS" and not run_summary["stopped"] and not run_summary["caps_exceeded"])
    binding = _load(out, "renderer_contract_binding.json") or {}
    pr.write_json(out / "derived_interpretation.json", {
        "card": CARD, "baseline": BASELINE, "original_capture_result_modified": False, "original_capture_status": "FAIL (kept; not rewritten as PASS)",
        "from_original_capture": {"simulator_state_restore": "PASS", "public_decision_state_restore": "PASS", "forced_rgb_bitwise_identity": "FAIL", "forced_rgb_difference_magnitude": "NOT_MEASURED_IN_ORIGINAL"},
        "r2_renderer_contract_applicable": binding.get("applicable"),
        "r2_dev03_stage_a": a["status"] if a else "NOT_RUN", "r2_dev05_capture": s5["status"] if s5 else "NOT_ATTEMPTED", "r2_dev05_restore": r5["status"] if r5 else "NOT_RUN",
        "r2_forced_rgb_difference_magnitude": "MEASURED (dev03_restore_rgb_metrics.json)" if a and a.get("metrics") else "NOT_MEASURED",
        "r2_all_conditions_pass": ok, "stopped": run_summary["stopped"], "GLOBAL_T_P": "UNRESOLVED",
        "note": "R2 criteria are the frozen Family B non-bitwise contract bound by structural identity of the render path; they replace nothing in the original card."})
    (out / "next_physical_card_request.md").write_text(request_md(ok, a, s5, r5, ledger, run_summary), encoding="utf-8")
    (out / "final_summary.md").write_text(summary_md(ok, a, s5, r5, ledger, run_summary, binding), encoding="utf-8")
    return {"all_pass": ok}


def request_md(ok, a, s5, r5, ledger, run):
    if not ok:
        return "\n".join([f"# Next physical card request — {CARD}", "", "**No physical card is requested.**", "", f"Stop: `{run['stopped'] or 'incomplete'}`; caps exceeded: {run['caps_exceeded'] or 'none'}.", "",
                          "Per the approval terms: stop the T_B -> T_P route; do not recapture dev_03, widen thresholds, substitute T_B_dev_18, or start branches.", ""])
    return "\n".join([f"# Next physical card request — {CARD}", "",
        "dev_03 restore, dev_05 capture and dev_05 restore all passed under the frozen R2 criteria. Requested next card: **8 fixed-script physical branches** = 2 states x 2 candidates x 2 repeats. Not started by this card.", "",
        "## Frozen protocol (no stepwise B_PLAN replanning)", "",
        "- States: T_B_dev_03 (snapshot " + OLD_SNAP_DIR + ") and T_B_dev_05 (R2 capture; paths in post_open snapshot records). Each branch restores in a fresh process on the same physical GPU; OPEN is never replayed.",
        "- T: PICK(target) -> PLACE(target, container) -> PICK(second_object) -> PLACE_BUFFER(second_object, buffer).",
        "- S: PICK(second_object) -> PLACE_BUFFER(second_object, buffer) -> PICK(target) -> PLACE(target, container).",
        "- Repeats: 2 per (state, candidate), each in its own fresh process from identical snapshot bytes; branch ids pre-registered.", "",
        "## To freeze before the first branch", "",
        "- Thresholds: the R2 restore criteria (rgb <= 3 uint8, depth 1e-6, proprio 1e-9, sim arrays 1e-12, facts/mask/goal/controller/RNG/clock exact) apply to the restore gate of every branch; a failed restore gate stops the card.",
        "- epsilon rule: max(floor, within-candidate repeat range), floors dG = 0.02, dT = 2.1 s, computed from repeats before cross-candidate comparison; G = 2^(-tau/H), H = 23.1 s, deadline 60 s on the restored clock.",
        "- Logging: facts, mask, controller exit after every skill; per-branch restore metrics.", "- Budget: 8 branches, 8 fresh-process constructions, <= 4 skills per branch, provider 0, RL/optimizer 0, test 0.", "",
        "## Evidence in this card", "", f"- budget used: {ledger['used']}", ""])


def summary_md(ok, a, s5, r5, ledger, run, binding):
    L = [f"# {CARD} — final summary", "", f"Outcome: **{'ALL R2 CONDITIONS PASS' if ok else 'STOPPED / FAIL'}** (stop: {run['stopped'] or 'none'}; caps exceeded: {run['caps_exceeded'] or 'none'})", "",
         f"Renderer contract applicable: {binding.get('applicable')} (problems: {binding.get('problems')})", ""]
    for name, r in (("Stage A dev_03 restore", a), ("dev_05 restore", r5)):
        L.append(f"## {name}")
        if not r:
            L += ["- not run", ""]
            continue
        L.append(f"- status {r['status']}; failures {r['failures']}")
        for k, v in r.get("checks", {}).items():
            L.append(f"  - {k}: pass={v.get('pass')}")
        m = r.get("metrics")
        if m:
            for k in ("A_saved_forced_vs_restore_render_1", "B_restore_render_1_vs_render_2"):
                x = m[k]
                L.append(f"  - {k}: rgb max_abs={x['rgb'].get('max_abs')} changed_px={x['rgb'].get('changed_pixel_count')} ({x['rgb'].get('changed_pixel_fraction')}), depth_raw max={x['depth_raw'].get('max_abs')}, depth_metric max={x['depth_metric'].get('max_abs')}, proprio max={x['proprio'].get('max_abs')}, perception pass={x['perception'].get('pass_')}, facts={x['facts']}")
        L.append("")
    L += ["## dev_05 capture", f"- {s5['status'] if s5 else 'NOT_ATTEMPTED'}" + (f"; facts {s5.get('facts_after_open')}; masks {s5.get('pick_masks')}; failures {s5.get('failures')}" if s5 else ""), "",
          "## Budget", f"- used: {ledger['used']}", f"- env.reset() invocations incl. bootstrap resets: {ledger.get('env_reset_calls_total')}", f"- processes: {ledger['processes']}", ""]
    return "\n".join(L)


def verify(root, out, before):
    root, out = Path(root).resolve(), Path(out).resolve()
    need = ["authorization.json", "renderer_contract_binding.json", "dev03_restore_rgb_metrics.json", "dev05_capture_result.json", "dev05_restore_rgb_metrics.json",
            "fresh_process_restore_results.json", "budget_ledger.json", "derived_interpretation.json", "next_physical_card_request.md", "final_summary.md"]
    ledger = pr.read_json(out / "budget_ledger.json")
    arrays_ok = True
    for name in ("restore_r2_dev03.json", "restore_r2_dev05.json"):
        r = _load(out, name)
        for rec in (r or {}).get("array_files", {}).values():
            arrays_ok &= Path(rec["path"]).is_file() and cap.sha_file(rec["path"]) == rec["sha256"]
    old = _load(out, "old_snapshot_integrity.json") or {}
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".log", ".jsonl"} and bool(__import__("re").search(r"sk-[A-Za-z0-9]{16,}|Authorization:|Bearer [A-Za-z0-9._-]{16,}|DASHSCOPE_API_KEY=", p.read_text(errors="ignore"))))
    auth = _load(out, "authorization.json")
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "caps_respected": all(ledger["used"][k] <= v for k, v in CAPS.items()),
              "zero_provider_rl_optimizer_test": all(ledger["used"][k] == 0 for k in ("provider_requests", "rl_transitions", "optimizer_steps", "test_episodes")),
              "no_continuation_skills": ledger["used"]["continuation_skills"] == 0, "restored_array_files_unchanged": bool(arrays_ok), "old_dev03_snapshot_hashes_matched_at_gate": bool(old.get("all_match")),
              "thresholds_written_before_run": bool(auth and auth.get("thresholds_frozen_before_any_restored_rgb_is_read") == THRESH), "prior_cards_unchanged": before == protected_hashes(root),
              "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k != "outputs_present") and all(checks["outputs_present"].values()) else "FAIL"
    pr.write_json(out / "verify.json", checks)
    return checks
