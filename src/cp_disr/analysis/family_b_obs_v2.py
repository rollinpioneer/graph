"""CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1: freeze, execute and gate observation v2.

Technical validation only: 4 branches, no provider/RL/optimizer, never releases
the original 20 Family B branches.  Old v1 evidence is read-only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from cp_disr.platforms.libero.family_b_obs_v2 import (CAMERAS, FIXED_ATOL, FUSION,
                                                      IMAGE_SIZE, PROFILE_VERSION,
                                                      array_sha, digest, quat_to_mat)
from cp_disr.platforms.libero.perception import THRESHOLDS

CARD = "CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1"
BASE = "fffb2b073ab28fa4dc55c82fb2ae58c5aa38dc73"
CFG_DIR = "configs/final_master/family_b_obs_v2"
PROFILE_PATH = f"{CFG_DIR}/observation_profile_v2.json"
OLD_EVIDENCE = "runs/final_master/S4/family_b_staging/20260930T074800Z_preflight"
OLD_REVIEW = "runs/final_master/S4/family_b_obs_review_cb2f0a290"
OLD_SERVER_ROOT = "/home/xushijie2/graph_cp_disr_s4_family_b"
PAD_PAIRS = ("pad_u", "pad_v")
TECHNICAL = (("layout_0", "B_PENDING", 0), ("layout_1", "C_PENDING", 0))
GATE_PASS = "OBS_V2_TECHNICAL_GATE_PASS_READY_FOR_REMAINDER_REVIEW"
CAPS = {"attempts": 4, "resets": 4, "skills": 24}
# Pre-registered before any v2 reset: how a U/V pair must match before the candidate.
PAIR_CONTRACT = {
    "exact": "per camera rgb sha256 and metric-depth sha256 equal -> EXACT",
    "non_bitwise_allowed_only_if": {
        "shape_dtype_equal": True, "qpos_qvel_atol": 1e-8,
        "public_facts_equal": True, "candidate_ids_and_mask_equal": True,
        "perception_selected_view_and_presence_equal": True,
        "perception_xyz_atol_m": 1e-4, "rgb_max_abs_diff_uint8": 3, "depth_max_abs_diff_m": 1e-3,
    },
    "otherwise": "FAIL; thresholds are not widened after seeing results",
    "render_self_repeat": "two direct offscreen renders of the unchanged state compared per camera, plus cached observation vs fresh render (evidence for the non-bitwise cause)",
    "gpu_policy": "both members of a U/V pair run sequentially on the same GPU",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------- freeze
def arena_cameras():
    """Fixed arena cameras exactly as the env builds them (no sim, no env)."""
    from robosuite.models.arenas import TableArena
    arena = TableArena(table_full_size=(0.8, 0.8, 0.05), table_friction=(1.0, 0.005, 0.0001),
                       table_offset=np.array((0.0, 0.0, 0.8)))
    arena.set_origin([0, 0, 0])
    return {c.attrib["name"]: dict(c.attrib) for c in arena.worldbody.findall("camera")}


def project(cam, point):
    f = 0.5 * IMAGE_SIZE / np.tan(np.deg2rad(cam["fovy"]) / 2.0)
    pc = np.array(cam["mat"]).T @ (np.asarray(point, dtype=float) - np.array(cam["pos"]))
    z = -pc[2]
    return {"u": float((IMAGE_SIZE - 1) / 2.0 + pc[0] * f / z),
            "v": float((IMAGE_SIZE - 1) / 2.0 + (-pc[1]) * f / z),
            "depth_m": float(z), "px_per_cube_edge": float(0.04 * f / z)}


def camera_manifest():
    arena = arena_cameras()
    cams, catalogue = {}, {}
    for name, attrib in arena.items():
        pos = [float(x) for x in attrib["pos"].split()]
        quat = np.array([float(x) for x in attrib["quat"].split()])
        quat = quat / np.linalg.norm(quat)
        fovy = float(attrib.get("fovy", 45.0))
        row = {"name": name, "mode": 0, "pos": pos, "quat": quat.tolist(), "fovy": fovy,
               "mat": quat_to_mat(quat).tolist(), "width": IMAGE_SIZE, "height": IMAGE_SIZE,
               "depth_convention": "robosuite normalized depth -> get_real_depth_map metric; image flipud",
               "fixed_in_world": True}
        catalogue[name] = row
        if name in CAMERAS:
            cams[name] = row
    return {"cameras": cams, "arena_catalogue": catalogue,
            "excluded": {"robot0_eye_in_hand": "moves with the wrist", "robot0_robotview": "not in arena; low base-mounted view",
                         "frontview": "same line of sight as agentview, farther", "birdview": "3 m range: ~2 px per cube edge at 128px"}}


def coverage(root, cams):
    layouts = read(Path(root) / "configs/final_master/family_b_layouts.json")["layouts"]
    rows = []
    for lay in layouts:
        z = 0.825 + 0.021
        pts = {"carrier": lay["carrier_xy"], "obj_b": lay["obj_b_xy"], "obj_c": lay["obj_c_xy"],
               "receiver": lay["receiver_xy"], "pad_u": lay["pad_u_xy"], "pad_v": lay["pad_v_xy"]}
        for cam_name, cam in cams.items():
            for obj, xy in pts.items():
                p = project(cam, [xy[0], xy[1], z])
                ok = 8 <= p["u"] <= IMAGE_SIZE - 9 and 8 <= p["v"] <= IMAGE_SIZE - 9
                rows.append({"layout": lay["layout_id"], "camera": cam_name, "object": obj, **p, "inside_margin_8px": bool(ok)})
    return rows


def line_clearance(a, b, p):
    a, b, p = (np.asarray(x, dtype=float) for x in (a, b, p))
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / np.dot(d, d), 0, 1))
    return float(np.linalg.norm(a + t * d - p))


def selection_evidence(root, cams):
    """Offline, v1 saved states only: how close the hand/arm is to each camera's line of sight."""
    base_arm = np.array([-0.56, 0.0, 1.0])
    rows = []
    reg = Path(OLD_SERVER_ROOT) / OLD_EVIDENCE / "captures"
    results = Path(root) / OLD_EVIDENCE / "physical/branch_results"
    for path in sorted(results.glob("*.json")):
        r = read(path)
        qa = reg / r["branch_id"] / "action_03/qa_state.jsonl"
        if not qa.is_file():
            continue
        after = [json.loads(line) for line in qa.read_text().splitlines()][-1]
        eef = after["eef_pos"]
        for obj in ("carrier", "obj_b", "obj_c"):
            target = after["hidden_truth"][obj]
            for cam_name, cam in cams.items():
                rows.append({"v1_branch": r["branch_id"], "candidate": r["candidate"], "layout": r["layout"],
                             "object": obj, "camera": cam_name,
                             "hand_to_los_m": line_clearance(cam["pos"], target, eef),
                             "arm_to_los_m": line_clearance(cam["pos"], target,
                                                            np.array(eef) * 0.5 + base_arm * 0.5),
                             "use": "QA/offline camera preselection only; never enters perception"})
    return rows


def build_profile(root):
    manifest = camera_manifest()
    contract = {"fusion": FUSION, "thresholds": THRESHOLDS, "min_pixels": THRESHOLDS["min_pixels"],
                "cameras": list(CAMERAS), "image_size": IMAGE_SIZE,
                "no_carry_over": ["previous frame", "nominal effect", "hidden QA", "initial coordinates"],
                "dynamic_facts": "p:Held/OnTable/Inside/AtBuffer from the fused public blob only",
                "camera_motion": False, "extra_robot_observation_action": False,
                "candidate_dependent_views": False}
    body = {"profile_version": PROFILE_VERSION, "cameras": list(CAMERAS), "camera_manifest": manifest["cameras"],
            "fusion_contract": contract, "pair_contract": PAIR_CONTRACT,
            "camera_freeze_atol": FIXED_ATOL}
    return {**body, "profile_sha256": digest(body)}, manifest


def freeze_profile(root):
    root = Path(root)
    profile, manifest = build_profile(root)
    cfg = root / CFG_DIR
    write(cfg / "observation_profile_v2.json", profile)
    write(cfg / "camera_manifest.json", {**manifest, "profile_sha256": profile["profile_sha256"]})
    write(cfg / "fusion_contract.json", {**profile["fusion_contract"], "profile_sha256": profile["profile_sha256"]})
    write(cfg / "pair_equivalence_contract.json", {**PAIR_CONTRACT, "profile_sha256": profile["profile_sha256"]})
    write(cfg / "camera_coverage.json", {"rows": coverage(root, manifest["cameras"]),
                                          "note": "128px, centres at cube mid-height; margin 8px"})
    write(cfg / "camera_selection_evidence.json", {"rows": selection_evidence(root, manifest["cameras"]),
          "note": "heuristic: distance from the camera->object ray to the eef and to the mid-arm point at the v1 post-candidate state"})
    return profile


# ------------------------------------------------------------ protection / identity
def inventory(paths):
    """Relative path -> size/sha256 for every file under each path (never mtime)."""
    out = {}
    for base in paths:
        base = Path(base)
        for p in sorted(base.rglob("*")):
            if p.is_file() and ".tmp" not in p.suffix:
                out[str(p)] = {"size": p.stat().st_size, "sha256": sha(p)}
    return out


def protected_paths():
    server = Path(OLD_SERVER_ROOT)
    return [server / OLD_EVIDENCE, server / OLD_REVIEW]


def diff_inventory(before, after):
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    removed = sorted(k for k in before if k not in after)
    added = sorted(k for k in after if k not in before)
    return {"changed": changed, "removed": removed, "added": added}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


SOURCE_FILES = (
    "src/cp_disr/platforms/libero/family_b_env.py",
    "src/cp_disr/platforms/libero/family_b_adapters.py",
    "src/cp_disr/platforms/libero/family_b_runtime.py",
    "src/cp_disr/platforms/libero/family_b_obs_v2.py",
    "src/cp_disr/platforms/libero/family_b_runtime_v2.py",
    "src/cp_disr/platforms/libero/skill_executor.py",
    "src/cp_disr/platforms/libero/runtime_factory.py",
    "src/cp_disr/platforms/libero/tp_sr_instrumentation.py",
    "src/cp_disr/platforms/libero/perception.py",
    "src/cp_disr/analysis/family_b_pilot.py",
    "src/cp_disr/analysis/family_b_obs_v2.py",
    "src/cp_disr/analysis/s1_integration.py",
    "scripts/family_b_obs_v2.py",
    "configs/runtime/tp_fb_contract_registry.yaml",
    "configs/final_master/family_b_layouts.json",
    f"{CFG_DIR}/config.yaml", f"{CFG_DIR}/observation_profile_v2.json",
    f"{CFG_DIR}/camera_manifest.json", f"{CFG_DIR}/fusion_contract.json",
    f"{CFG_DIR}/pair_equivalence_contract.json",
)


def source_hashes(root):
    return {p: sha(Path(root) / p) for p in SOURCE_FILES}


def check_profile(root):
    profile = read(Path(root) / PROFILE_PATH)
    body = {k: v for k, v in profile.items() if k != "profile_sha256"}
    if digest(body) != profile["profile_sha256"]:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:hash")
    live, _ = build_profile(root)
    if live["profile_sha256"] != profile["profile_sha256"]:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:regenerated profile differs from frozen file")
    return profile


# --------------------------------------------------------------------- registration
def check_registration_uniform(registration):
    """One observation configuration for every branch; return list of problems."""
    problems = []
    branches = registration["branches"]
    keys = {(b["observation_profile_sha256"], tuple(b["cameras"]), b["profile_version"]) for b in branches}
    if len(keys) != 1:
        problems.append("branches use different observation configs")
    if {b["cameras"][0] for b in branches} != {CAMERAS[0]} or any(tuple(b["cameras"]) != CAMERAS for b in branches):
        problems.append("camera set differs from the frozen set")
    for layout, context, repeat in TECHNICAL:
        pair = [b for b in branches if (b["layout"], b["context"], b["repeat"]) == (layout, context, repeat)]
        if sorted(b["candidate"] for b in pair) != sorted(PAD_PAIRS):
            problems.append(f"{layout}/{context}: pad pair incomplete")
        if len({(b["observation_profile_sha256"], tuple(b["cameras"]), b["restore_seed"]) for b in pair}) != 1:
            problems.append(f"{layout}/{context}: U/V observation config or seed differs")
    return problems


def branch_identity(logical, profile_sha256, commit, source_hash, attempt_identity):
    """v2 id binds logical identity, profile version/hash, source commit/hash and the attempt identity."""
    return digest({"logical": logical, "profile_version": PROFILE_VERSION, "profile_sha256": profile_sha256,
                   "source_commit": commit, "source_hash": source_hash, "attempt_identity": attempt_identity})[:16]


def register(root, out):
    """After the freeze commit: v2 branch identities, ledger, manifest, authorization."""
    import yaml
    from cp_disr.analysis import family_b_pilot as v1
    from cp_disr.runtime import require_runtime
    root, out = Path(root).resolve(), Path(out)
    if git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree not clean")
    commit = git(root, "rev-parse", "HEAD")
    profile = check_profile(root)
    cfg = yaml.safe_load((root / CFG_DIR / "config.yaml").read_text())
    if cfg["base_commit"] != BASE or not git(root, "merge-base", "--is-ancestor", BASE, "HEAD") == "":
        raise ValueError("STOPPED_SOURCE_IDENTITY:base")
    if (out / "physical/witnesses/e4_branch_registration.json").exists():
        raise ValueError("STOPPED_SOURCE_IDENTITY:already registered")
    hashes = source_hashes(root)
    source_hash = digest(hashes)
    prior = yaml.safe_load(v1.SOURCE_MANIFEST.read_text())
    ref = float(prior["runtime"]["reference_skill_seconds_by_task"]["T_P_SO_MVP"])
    runtime = dict(prior["runtime"])
    runtime.update({
        "repository_path": str(root), "active_task_id": "T_P_FB",
        "contract_path": cfg["runtime"]["contract_path"], "layouts_path": cfg["runtime"]["layouts_path"],
        "observation_profile_path": cfg["runtime"]["observation_profile_path"],
        "reference_skill_seconds_by_task": {"T_P_FB": ref},
        "task_deadlines": {"T_P_FB": 60.0},
        "skill_timeouts": {"T_P_FB": {"PICK": 9., "PLACE": 8., "PLACE_BUFFER": 8.}},
        "task_assets": {"T_P_FB": "src/cp_disr/platforms/libero/family_b_env.py",
                        "license": "assets/cp_disr/LICENSE", "owned_by": "cp_disr_project"},
        "task_evaluator_version": {"T_P_FB": "family-b-evaluator-v1"},
        "task_splits": {"T_P_FB": cfg["runtime"]["layouts_path"]},
        "decision_cap_by_task": {"T_P_FB": 6},
    })
    src = root / "src/cp_disr/platforms/libero/family_b_runtime_v2.py"
    manifest = dict(prior)
    manifest["runtime"] = runtime
    manifest["runtime_factory"] = {
        "factory": "create_family_b_obs_v2_runtime",
        "module": "cp_disr.platforms.libero.family_b_runtime_v2",
        "source_path": str(src), "sha256": sha(src),
    }
    require_runtime(manifest)
    manifest_path = out / "spec/runtime_manifest_T_P_FB_obs_v2.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    old = {(b["layout"], b["context"], b["repeat"], b["candidate"]): b["branch_id"]
           for b in read(Path(root) / OLD_EVIDENCE / "physical/registration.json")["branches"]}
    branches = []
    for layout, context, repeat in TECHNICAL:
        pair_key = digest({"layout": layout, "context": context, "repeat": repeat,
                           "profile": profile["profile_sha256"], "commit": commit})[:16]
        seed = int(hashlib.sha256(f"obsv2:{pair_key}".encode()).hexdigest()[:8], 16)
        for pad in PAD_PAIRS:
            logical = {"layout": layout, "context": context, "repeat": repeat, "candidate": pad}
            attempt_identity = f"FAMILY_B_OBS_V2_ATTEMPT_1:{pad}"
            branch_id = branch_identity(logical, profile["profile_sha256"], commit, source_hash, attempt_identity)
            branches.append({
                **logical, "branch_id": branch_id, "case_id": layout, "restore_seed": seed,
                "candidate_id": f"a:PLACE_BUFFER:carrier:{pad}:v1",
                "prefix": cfg["contexts"][context]["prefix"], "wave": "obs_v2_technical",
                "manifest_path": str(manifest_path), "authorized": True, "execute_now": True,
                "plan_id": branch_id, "attempt_id": branch_id, "attempt_identity": attempt_identity,
                "v1_logical_branch_id_reference_only": old[(layout, context, repeat, pad)],
                "profile_version": PROFILE_VERSION, "observation_profile_sha256": profile["profile_sha256"],
                "cameras": list(CAMERAS), "source_commit": commit, "source_hash": source_hash,
            })
    if len({b["branch_id"] for b in branches}) != 4 or {b["branch_id"] for b in branches} & set(old.values()):
        raise ValueError("STOPPED_SOURCE_IDENTITY:branch identities")
    registration = {"card_id": CARD, "branches": branches, "manifest_sha256": sha(manifest_path),
                    "reference_skill_seconds": ref}
    problems = check_registration_uniform(registration)
    if problems:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems))
    phys = out / "physical"
    write(phys / "witnesses/e4_branch_registration.json", registration)
    write(phys / "registration.json", registration)
    write(phys / "budget_ledger.json", {"physical_witness_episodes": {"cap": CAPS["attempts"], "used": 0}})
    write(out / "registration_v2.json", registration)
    write(out / "budget_ledger.json", {
        "old_v1": {"attempts": 4, "resets": 4, "skill_calls": 20, "status": "PERMANENTLY_CONSUMED",
                   "original_remaining_20": "NOT_RELEASED"},
        "v2_validation_attempts": {"cap": CAPS["attempts"], "used": 0},
        "environment_constructions": {"cap": 4, "used": 0},
        "explicit_resets": {"cap": CAPS["resets"], "used": 0},
        "live_skill_calls": {"cap": CAPS["skills"], "used": 0},
        "provider_calls": {"cap": 0, "used": 0}, "rl_transitions": {"cap": 0, "used": 0},
        "optimizer_steps": {"cap": 0, "used": 0}, "elastic": {"cap": 0, "used": 0},
        "standalone_capture_resets": {"cap": 0, "used": 0},
    })
    write(out / "bindings/reference_cost.json", {"reference_skill_seconds": ref})
    write(out / "authorization.json", {
        "card_id": CARD, "authorized_by": "user message approving CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1",
        "v2_validation_attempts_max": 4, "environment_constructions_max": 4, "explicit_resets_max": 4,
        "skill_retry": 0, "provider": 0, "rl_transitions": 0, "optimizer_steps": 0, "elastic": 0,
        "s2": False, "s3": False, "formal_test": False, "original_remaining_20_released": False,
        "no_fifth_attempt_after_failure": True})
    write(out / "source_identity.json", {
        "card_id": CARD, "branch": git(root, "branch", "--show-current"), "commit": commit, "base_commit": BASE,
        "source_hash": source_hash, "sources": hashes, "observation_profile_sha256": profile["profile_sha256"],
        "manifest_sha256": sha(manifest_path), "old_branch_commit": git(root, "rev-parse",
                                                                         "codex/cp-disr-s4-family-b-staging-v1"),
    })
    for rel in (PROFILE_PATH, f"{CFG_DIR}/camera_manifest.json", f"{CFG_DIR}/fusion_contract.json"):
        write(out / Path(rel).name, read(root / rel))
    return {"status": "REGISTERED", "commit": commit, "branches": [b["branch_id"] for b in branches]}


# ------------------------------------------------------------------------ worker
def _boundary_views(bundle, out, branch_id):
    """Per-camera pre-candidate evidence (pure; no sim step, no state change)."""
    from cp_disr.platforms.libero.family_b_obs_v2 import fuse
    from cp_disr.platforms.libero.perception import _metric_depth
    env = bundle.environment
    d = Path(out) / "captures" / branch_id / "boundary"
    d.mkdir(parents=True, exist_ok=True)
    first = env.public_observation()

    def fresh(cam):
        # direct offscreen render of the unchanged state; observables/caches are not touched
        rgb, depth = env.sim.render(camera_name=cam, width=IMAGE_SIZE, height=IMAGE_SIZE, depth=True)
        return np.flipud(rgb).copy(), np.flipud(depth).copy()

    rec = {"profile_version": PROFILE_VERSION, "profile_sha256": bundle.perception.profile_hash,
           "cameras": list(CAMERAS), "views": {}, "self_repeat": {}, "cached_vs_fresh": {},
           "hidden_truth_used": False, "qpos_qvel_used": False, "sim_time": float(env.sim.data.time)}
    for cam in CAMERAS:
        rgb = np.asarray(first["views"][cam]["rgb"])
        raw_depth = np.asarray(first["views"][cam]["depth"])
        depth = np.asarray(_metric_depth(env, raw_depth), dtype=np.float32)
        bundle.recorder._png(d / f"{cam}_rgb.png", rgb)
        np.save(d / f"{cam}_depth_metric.npy", depth)
        rec["views"][cam] = {"rgb_sha256": array_sha(rgb), "rgb_shape": list(rgb.shape), "rgb_dtype": str(rgb.dtype),
                             "depth_sha256": array_sha(depth), "depth_shape": list(depth.shape),
                             "depth_dtype": str(depth.dtype)}
        try:
            r1, d1 = fresh(cam)
            r2, d2 = fresh(cam)
            rec["self_repeat"][cam] = {"rgb_equal": bool(np.array_equal(r1, r2)), "depth_equal": bool(np.array_equal(d1, d2))}
            rec["cached_vs_fresh"][cam] = {"rgb_equal": bool(np.array_equal(rgb, r1)),
                                           "depth_equal": bool(np.array_equal(raw_depth.reshape(d1.shape), d1))}
        except Exception as exc:  # evidence only; never aborts the branch
            rec["self_repeat"][cam] = {"rgb_equal": None, "depth_equal": None, "error": f"{type(exc).__name__}: {exc}"}
            rec["cached_vs_fresh"][cam] = {"rgb_equal": None, "depth_equal": None}
    per_view, calibs, _ = bundle.perception.analyze(first)
    fused = fuse(per_view, bundle.perception.reference)
    rec["perception"] = {
        "per_view": per_view, "reference": bundle.perception.reference,
        "fused": {n: {"selected_view": f["selected_view"], "inputs": f["inputs"],
                      "xyz": None if f["blob"] is None else f["blob"]["xyz"]} for n, f in fused.items()},
        "calibrations": calibs}
    write(d / "views_boundary.json", rec)
    return rec


def worker(root, out, branch_id):
    import time
    import traceback
    import yaml
    from cp_disr.analysis.family_b_pilot import (BranchStop, _action, _boundary, _truth,
                                                 branch)
    from cp_disr.analysis.s1_integration import claim_branch_attempt, finish_branch_attempt
    from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
    from cp_disr.platforms.libero.family_b_runtime import TASK_ID
    from cp_disr.platforms.libero.tp_sr_instrumentation import RecordingPlanner
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    phys = out / "physical"
    b = branch(out, branch_id)
    manifest_path = Path(b["manifest_path"])
    if sha(manifest_path) != read(phys / "registration.json")["manifest_sha256"]:
        raise RuntimeError("STOPPED_SOURCE_IDENTITY:manifest drift")
    if read(out / "source_identity.json")["sources"] != source_hashes(root):
        raise RuntimeError("STOPPED_SOURCE_IDENTITY:source drift")
    receipt = claim_branch_attempt(phys, branch_id)
    bundle, rows = None, []
    final = {"branch_id": branch_id, "layout": b["case_id"], "context": b["context"], "repeat": b["repeat"],
             "candidate": b["candidate"], "wave": b["wave"], "restore_seed": b["restore_seed"],
             "receipt_id": receipt["reservation_id"], "status": "EXCEPTION", "task_success": False,
             "actions": rows, "time_to_task_success": None, "observation_profile_sha256": b["observation_profile_sha256"],
             "attempt_identity": b["attempt_identity"]}
    wall = time.monotonic()
    try:
        manifest = yaml.safe_load(manifest_path.read_text())
        bundle = load_runtime(manifest)
        bundle.configure(out, b)
        snapshot = bundle.start_case(b["case_id"], restore_seed=int(b["restore_seed"]))
        if not bundle.restore_verified:
            raise BranchStop("RESTORE_RECEIPT_INVALID")
        deadline = float(manifest["runtime"]["task_deadlines"][TASK_ID])
        final["restore_receipt"] = bundle.restore_receipt
        final["initial_sim_time"] = float(bundle.episode_start_seconds)
        for aid in b["prefix"]:
            snapshot, task = _action(bundle, snapshot, aid, "setup", rows, deadline)
            if task.terminated or task.truncated:
                raise BranchStop("SCRIPT_DIVERGENCE_SETUP_TERMINAL:" + task.reason)
        if not _truth(snapshot, "p:Held:carrier"):
            raise BranchStop("SCRIPT_DIVERGENCE_CARRIER_NOT_HELD")
        done = {"B_PENDING": ("obj_c",), "C_PENDING": ("obj_b",)}[b["context"]]
        if any(not _truth(snapshot, f"p:Inside:{o}:receiver") for o in done):
            raise BranchStop("SCRIPT_DIVERGENCE_SETUP_GOAL")
        both = [f"a:PLACE_BUFFER:carrier:{p}:v1" for p in PAD_PAIRS]
        if any(c not in snapshot.candidate_ids or not snapshot.mask[snapshot.candidate_ids.index(c)] for c in both):
            raise BranchStop("SCRIPT_DIVERGENCE_UV_MASK")
        boundary = _boundary(bundle, snapshot, rows, out)
        views = _boundary_views(bundle, out, branch_id)
        final["boundary_sha256"] = sha(out / "captures" / branch_id / "boundary/public.json")
        final["boundary_sim_time"] = boundary["sim_time"]
        final["boundary_views_sha256"] = sha(out / "captures" / branch_id / "boundary/views_boundary.json")
        final["boundary_self_repeat"] = views["self_repeat"]
        start = float(bundle.clock.now_seconds())
        final["candidate_start_sim_time"] = start
        ref = float(read(out / "bindings/reference_cost.json")["reference_skill_seconds"])
        planner = RecordingPlanner(BPlanPlanner(SearchConfig(
            depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=ref)), bundle.recorder)
        aid = b["candidate_id"]
        for decision in range(6 - len(b["prefix"])):
            stage = "candidate" if decision == 0 else "continuation"
            snapshot, task = _action(bundle, snapshot, aid, stage, rows, deadline)
            if task.terminated or task.truncated:
                end = float(bundle.clock.now_seconds())
                final.update({"status": task.reason, "task_success": bool(task.success), "terminal_sim_time": end,
                              "time_to_task_success": end - start if task.success else None,
                              "terminal_planner": "NOT_APPLICABLE_TERMINAL"})
                break
            remaining = deadline - (float(bundle.clock.now_seconds()) - bundle.episode_start_seconds)
            plan = planner.plan(snapshot.facts, snapshot.template, remaining)
            rows[-1]["planner_status"] = plan.status
            rows[-1]["planner_plan"] = list(plan.plan)
            if plan.status != "PLAN_FOUND" or not plan.plan:
                final["status"], final["planner_reason"] = plan.status, plan.reason
                break
            aid = plan.plan[0]
        else:
            final["status"] = "DECISION_CAP"
    except BranchStop as exc:
        final["status"], final["detail"] = str(exc).split(":", 1)[0], str(exc)
    except Exception as exc:
        final["status"], final["detail"] = "EXCEPTION", f"{type(exc).__name__}:{exc}"
        final["traceback"] = traceback.format_exc()[-4000:]
    finally:
        final["worker_wall_seconds"] = time.monotonic() - wall
        if bundle is not None:
            final["env_counts"] = bundle.env_counts()
            final["recorder_errors"] = list(bundle.recorder.errors) if bundle.recorder else []
            if bundle.recorder:
                bundle.recorder.close(bundle.env_counts())
            bundle.environment.close()
        write(phys / "branch_results" / f"{branch_id}.json", final)
        finish_branch_attempt(phys, branch_id, "COMPLETED" if final["status"] != "EXCEPTION" else "FAILED",
                              execution_status=final["status"])
    return final


# ---------------------------------------------------------------------- dispatch
def _charge(out):
    path = Path(out) / "budget_ledger.json"
    ledger = read(path)
    for key in ("v2_validation_attempts", "environment_constructions", "explicit_resets"):
        if ledger[key]["used"] + 1 > ledger[key]["cap"]:
            raise ValueError("STOPPED_BUDGET_EXHAUSTED:" + key)
        ledger[key]["used"] += 1
    write(path, ledger)


def queues(branches, slots):
    """Both members of a U/V pair share one slot (same GPU), run sequentially."""
    pairs = {}
    for b in branches:
        pairs.setdefault((b["layout"], b["context"], b["repeat"]), []).append(b)
    ordered = [sorted(v, key=lambda b: PAD_PAIRS.index(b["candidate"])) for _, v in sorted(pairs.items())]
    if len(slots) >= 2:
        return {slots[i % len(slots)]: [x for j, q in enumerate(ordered) if j % len(slots) == i for x in q]
                for i in range(len(slots))}
    return {slots[0]: [x for q in ordered for x in q]}


def run_wave(root, out, gpus, max_workers=2):
    import subprocess as sp
    import sys
    import time
    from cp_disr.analysis.family_b_pilot import _worker_env
    from cp_disr.analysis.s1_integration import reserve_branch_attempt
    from cp_disr.runtime import require_runtime
    root, out = Path(root).resolve(), Path(out).resolve()
    if git(root, "rev-parse", "HEAD") != read(out / "source_identity.json")["commit"]:
        raise ValueError("STOPPED_SOURCE_IDENTITY:HEAD moved since registration")
    if git(root, "status", "--porcelain", "-uno"):
        raise ValueError("STOPPED_SOURCE_IDENTITY:tracked worktree dirty")
    if read(out / "source_identity.json")["sources"] != source_hashes(root):
        raise ValueError("STOPPED_SOURCE_IDENTITY:source drift")
    import yaml
    reg = read(out / "physical/registration.json")
    require_runtime(yaml.safe_load(Path(reg["branches"][0]["manifest_path"]).read_text()))
    problems = check_registration_uniform(reg)
    if problems or len(reg["branches"]) != 4:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:" + ";".join(problems))
    if any((out / "physical/branch_results" / f"{b['branch_id']}.json").exists() for b in reg["branches"]):
        raise ValueError("STOPPED_DUPLICATE_ATTEMPT")
    slots = list(gpus)[:max(1, min(max_workers, len(gpus)))]
    if not slots or len(set(slots)) != len(slots):
        raise ValueError("STOPPED_GPU_BINDING")
    todo = queues(reg["branches"], slots)
    active, faults, started = {}, [], time.monotonic()
    logs = out / "physical/worker_logs"
    logs.mkdir(parents=True, exist_ok=True)
    dispatched = []
    while any(todo.values()) or active:
        for gpu in slots:
            if gpu in active or not todo[gpu] or len(faults) >= 2:
                continue
            b = todo[gpu].pop(0)
            reserve_branch_attempt(out / "physical", b["branch_id"])
            _charge(out)
            log = (logs / f"{b['branch_id']}.log").open("ab")
            cmd = [sys.executable, str(root / "scripts/family_b_obs_v2.py"), "worker",
                   "--root", str(root), "--output", str(out), "--branch-id", b["branch_id"]]
            proc = sp.Popen(cmd, cwd=root, env=_worker_env(gpu), stdout=log, stderr=sp.STDOUT, start_new_session=True)
            active[gpu] = (proc, log, b["branch_id"], time.monotonic())
            dispatched.append({"branch_id": b["branch_id"], "gpu": gpu, "t_start": time.monotonic() - started})
        if not active:
            break
        time.sleep(0.5)
        for gpu, (proc, log, bid, t0) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            path = out / "physical/branch_results" / f"{bid}.json"
            if not path.is_file():
                faults.append({"branch_id": bid, "exit_code": code, "reason": "RESULT_MISSING"})
            elif read(path)["status"] == "EXCEPTION":
                faults.append({"branch_id": bid, "exit_code": code, "reason": read(path).get("detail")})
            else:
                faults.clear()
    span = time.monotonic() - started
    results = [read(out / "physical/branch_results" / f"{b['branch_id']}.json") for b in reg["branches"]
               if (out / "physical/branch_results" / f"{b['branch_id']}.json").is_file()]
    skills = sum(len(r.get("actions", [])) for r in results)
    ledger = read(out / "budget_ledger.json")
    ledger["live_skill_calls"]["used"] = skills
    ledger["observed_explicit_resets"] = sum(r.get("env_counts", {}).get("reset_calls", 0) for r in results)
    ledger["observed_environment_constructions"] = sum(r.get("env_counts", {}).get("constructions", 0) for r in results)
    write(out / "budget_ledger.json", ledger)
    report = {"workers": len(slots), "gpus": slots, "dispatched": dispatched, "faults": faults,
              "span_seconds": span, "branches_per_second": len(dispatched) / span if span else None,
              "median_branch_wall_seconds": float(np.median([r["worker_wall_seconds"] for r in results])) if results else None,
              "max_concurrency": len(slots), "single_worker_baseline": "NOT_MEASURED",
              "speedup_claim": "NONE (no matched single-worker baseline)"}
    write(out / "physical/throughput_report.json", report)
    return report


# --------------------------------------------------------------- pairing / gate
def _png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _pair_members(reg, key):
    pair = [b for b in reg["branches"] if (b["layout"], b["context"], b["repeat"]) == key]
    return sorted(pair, key=lambda b: PAD_PAIRS.index(b["candidate"]))


def paired_restore(out):
    out = Path(out)
    reg = read(out / "physical/registration.json")
    contract = PAIR_CONTRACT["non_bitwise_allowed_only_if"]
    rows = {}
    for key in TECHNICAL:
        u, v = _pair_members(reg, key)
        name = "/".join(map(str, key))
        du, dv = (out / "captures" / b["branch_id"] / "boundary" for b in (u, v))
        row = {"pair": name, "branches": [u["branch_id"], v["branch_id"]], "candidates": ["pad_u", "pad_v"]}
        need = ("public.json", "qa_state.json", "views_boundary.json")
        if not all((d / f).is_file() for d in (du, dv) for f in need):
            row.update(status="FAIL", reason="boundary evidence missing")
            rows[name] = row
            continue
        pu, pv = read(du / "public.json"), read(dv / "public.json")
        qu, qv = read(du / "qa_state.json"), read(dv / "qa_state.json")
        wu, wv = read(du / "views_boundary.json"), read(dv / "views_boundary.json")
        row["public_facts_equal"] = pu["fact_values"] == pv["fact_values"]
        row["candidate_ids_equal"] = pu["candidate_ids"] == pv["candidate_ids"]
        row["candidate_mask_equal"] = pu["candidate_mask"] == pv["candidate_mask"]
        for field in ("qpos", "qvel"):
            a, b2 = np.asarray(qu[field], dtype=float), np.asarray(qv[field], dtype=float)
            row[f"{field}_max_abs_diff"] = float(np.max(np.abs(a - b2)))
            row[f"{field}_within_tol"] = bool(np.allclose(a, b2, atol=contract["qpos_qvel_atol"], rtol=0))
        row["profile_identity_equal"] = (wu["profile_sha256"] == wv["profile_sha256"] ==
                                         u["observation_profile_sha256"] == v["observation_profile_sha256"])
        row["camera_set_equal"] = wu["cameras"] == wv["cameras"] == list(CAMERAS)
        cams, exact_all, within_all = {}, True, True
        for cam in CAMERAS:
            iu, iv = wu["views"][cam], wv["views"][cam]
            rgb_u, rgb_v = _png(du / f"{cam}_rgb.png"), _png(dv / f"{cam}_rgb.png")
            dep_u, dep_v = np.load(du / f"{cam}_depth_metric.npy"), np.load(dv / f"{cam}_depth_metric.npy")
            same_type = (iu["rgb_shape"] == iv["rgb_shape"] and iu["rgb_dtype"] == iv["rgb_dtype"] and
                         iu["depth_shape"] == iv["depth_shape"] and iu["depth_dtype"] == iv["depth_dtype"])
            c = {"shape_dtype_equal": same_type, "rgb_sha_equal": iu["rgb_sha256"] == iv["rgb_sha256"],
                 "depth_sha_equal": iu["depth_sha256"] == iv["depth_sha256"],
                 "self_repeat": {"pad_u": wu["self_repeat"][cam], "pad_v": wv["self_repeat"][cam]}}
            if same_type:
                diff = np.abs(rgb_u.astype(int) - rgb_v.astype(int))
                finite = np.isfinite(dep_u) & np.isfinite(dep_v)
                c.update(rgb_n_diff_pixels=int((diff.max(axis=2) > 0).sum()), rgb_max_abs_diff=int(diff.max()),
                         depth_finite_mask_equal=bool((np.isfinite(dep_u) == np.isfinite(dep_v)).all()),
                         depth_max_abs_diff=float(np.max(np.abs(dep_u[finite] - dep_v[finite]))) if finite.any() else 0.0)
            exact = c["rgb_sha_equal"] and c["depth_sha_equal"]
            within = same_type and c.get("rgb_max_abs_diff", 99) <= contract["rgb_max_abs_diff_uint8"] and \
                c.get("depth_max_abs_diff", 9) <= contract["depth_max_abs_diff_m"] and c.get("depth_finite_mask_equal", False)
            c["classification"] = "EXACT" if exact else ("NON_BITWISE_WITHIN_CONTRACT" if within else "OUT_OF_CONTRACT")
            exact_all, within_all = exact_all and exact, within_all and (exact or within)
            cams[cam] = c
        row["cameras"] = cams
        pe, pf = wu["perception"]["fused"], wv["perception"]["fused"]
        sel_equal = all(pe[o]["selected_view"] == pf[o]["selected_view"] for o in pe)
        xyz_diff = [float(np.max(np.abs(np.asarray(pe[o]["xyz"]) - np.asarray(pf[o]["xyz"]))))
                    for o in pe if pe[o]["xyz"] is not None and pf[o]["xyz"] is not None]
        row["perception_selected_view_and_presence_equal"] = bool(sel_equal and all(
            (pe[o]["xyz"] is None) == (pf[o]["xyz"] is None) for o in pe))
        row["perception_xyz_max_abs_diff_m"] = max(xyz_diff) if xyz_diff else None
        row["perception_xyz_within_tol"] = all(d <= contract["perception_xyz_atol_m"] for d in xyz_diff)
        row["initial_images_informational"] = {
            cam: array_sha(_png(out / "captures" / u["branch_id"] / f"initial_{cam}_rgb.png")) ==
            array_sha(_png(out / "captures" / v["branch_id"] / f"initial_{cam}_rgb.png"))
            for cam in CAMERAS if (out / "captures" / u["branch_id"] / f"initial_{cam}_rgb.png").is_file()
            and (out / "captures" / v["branch_id"] / f"initial_{cam}_rgb.png").is_file()}
        row["pixel_equivalence"] = "EXACT" if exact_all else ("NON_BITWISE_WITHIN_CONTRACT" if within_all else "OUT_OF_CONTRACT")
        if row["pixel_equivalence"] == "NON_BITWISE_WITHIN_CONTRACT":
            flags = [r.get(k) for w in (wu, wv) for r in w["self_repeat"].values() for k in ("rgb_equal", "depth_equal")]
            row["non_bitwise_cause"] = ("SELF_REPEAT_NOT_MEASURED" if any(f is None for f in flags) else
                                        "CROSS_PROCESS_RENDER_DIFFERENCE_UNEXPLAINED" if all(flags)
                                        else "RENDER_NONDETERMINISTIC_WITHIN_PROCESS")
        ok = all([row["public_facts_equal"], row["candidate_ids_equal"], row["candidate_mask_equal"],
                  row["qpos_within_tol"], row["qvel_within_tol"], row["profile_identity_equal"],
                  row["camera_set_equal"], row["perception_selected_view_and_presence_equal"],
                  row["perception_xyz_within_tol"] is True, row["pixel_equivalence"] != "OUT_OF_CONTRACT"])
        row["status"] = "PASS" if ok else "FAIL"
        rows[name] = row
    result = {"contract": PAIR_CONTRACT, "pairs": rows,
              "status": "PASS" if rows and all(r["status"] == "PASS" for r in rows.values()) else "FAIL"}
    write(out / "paired_restore_v2.json", result)
    return result


def remaining_object(context):
    return {"B_PENDING": "obj_b", "C_PENDING": "obj_c"}[context]


def action_dirs(cap):
    return sorted(p for p in Path(cap).glob("action_[0-9][0-9]") if p.is_dir())


def planner_semantics(cap, actions, evaluators):
    """CONTINUE needs a planner record; TASK_SUCCESS must not have a real planner call."""
    problems = []
    last = len(actions) - 1
    for i, _ in enumerate(actions):
        d = Path(cap) / f"action_{i:02d}"
        ev = evaluators.get(i)
        has = (d / "planner.json").is_file()
        if i < 3:
            continue
        if ev is None:
            problems.append(f"action_{i:02d}:evaluator missing")
        elif ev["terminated"] or ev["task_success"]:
            if has:
                problems.append(f"action_{i:02d}:planner called after TASK_SUCCESS")
        elif not has:
            problems.append(f"action_{i:02d}:CONTINUE without planner record")
        elif i < last:
            plan = read(d / "planner.json")
            if plan["status"] != "PLAN_FOUND" or plan["plan"][:1] != [actions[i + 1]["action_id"]]:
                problems.append(f"action_{i:02d}:planner does not match next action")
    return problems


def remaining_support(cap, context, min_pixels):
    """Public chain after the candidate action: pixels -> blob -> fact -> PICK mask."""
    obj = remaining_object(context)
    d = Path(cap) / "action_03"
    out = {"object": obj}
    frames = read(d / "views_perception.json") if (d / "views_perception.json").is_file() else []
    if not frames:
        return {**out, "ok": False, "reason": "no views_perception record"}
    last = frames[-1]
    fusion = last["fusion"][obj]
    out["last_frame"] = last["frame_id"]
    out["selected_view"] = fusion["selected_view"]
    out["inputs"] = fusion["inputs"]
    snapshot = read(d / "snapshot.json")
    facts = snapshot["fact_values"]
    pick = f"a:PICK:{obj}:v1"
    out["on_table"] = facts.get(f"p:OnTable:{obj}")
    out["held"] = facts.get(f"p:Held:{obj}")
    out["pick_mask"] = bool(snapshot["candidate_mask"][snapshot["candidate_ids"].index(pick)])
    out["supported"] = fusion["blob"] is not None and fusion["inputs"][fusion["selected_view"]]["support"] >= min_pixels
    out["ok"] = bool(out["supported"] and out["on_table"] == "TRUE" and out["held"] == "FALSE" and out["pick_mask"])
    return out


def _scan_forbidden():
    root = Path(__file__).resolve().parents[1] / "platforms/libero/family_b_obs_v2.py"
    text = root.read_text()
    forbidden = ("hidden_truth", "body_xpos", ".qpos", ".qvel", "get_joint_qpos", "geom_xpos")
    return [t for t in forbidden if t in text]


def technical_gate(root, out):
    out = Path(out)
    reg = read(out / "physical/registration.json")
    attempts = read(out / "physical/attempt_registry.json") if (out / "physical/attempt_registry.json").is_file() else {}
    pair = paired_restore(out)
    problems, per_branch = [], {}
    min_pixels = THRESHOLDS["min_pixels"]
    for b in reg["branches"]:
        bid = b["branch_id"]
        cap = out / "captures" / bid
        res_path = out / "physical/branch_results" / f"{bid}.json"
        row = {"branch_id": bid, "layout": b["layout"], "context": b["context"], "candidate": b["candidate"]}
        per_branch[bid] = row
        if not res_path.is_file():
            row["terminal_receipt"] = False
            problems.append(f"{bid}:no terminal result")
            continue
        r = read(res_path)
        acts = r["actions"]
        row["status"] = r["status"]
        row["terminal_receipt"] = attempts.get(bid) in ("COMPLETED", "FAILED", "UNKNOWN")
        setup = acts[:3]
        row["setup_complete"] = len(setup) == 3 and [a["action_id"] for a in setup] == b["prefix"] and \
            all(a["controller_exit"] == "NORMAL_TERMINATION" for a in setup) and (cap / "boundary/public.json").is_file()
        if not row["setup_complete"]:
            problems.append(f"{bid}:setup incomplete")
        if (cap / "boundary/public.json").is_file():
            pub = read(cap / "boundary/public.json")
            ok = all(c in pub["candidate_ids"] and pub["candidate_mask"][pub["candidate_ids"].index(c)]
                     for c in (f"a:PLACE_BUFFER:carrier:{p}:v1" for p in PAD_PAIRS))
            row["both_pads_legal_at_decision"] = ok
            if not ok:
                problems.append(f"{bid}:pad candidates not both legal")
        rem = remaining_object(b["context"])
        seq = [a["action_id"] for a in acts]
        expected = b["prefix"] + [b["candidate_id"], f"a:PICK:{rem}:v1", f"a:PLACE:{rem}:receiver:v1"]
        row["task_success"] = bool(r["status"] == "TASK_SUCCESS" and r["task_success"] and seq == expected)
        if not row["task_success"]:
            problems.append(f"{bid}:not TASK_SUCCESS with expected sequence:{r['status']}")
        evaluators = {i: read(cap / f"action_{i:02d}/evaluator.json") for i in range(len(acts))
                      if (cap / f"action_{i:02d}/evaluator.json").is_file()}
        row["evaluator_terminal"] = bool(evaluators and evaluators[max(evaluators)]["task_success"])
        psem = planner_semantics(cap, acts, evaluators)
        row["planner_semantics_problems"] = psem
        problems += [f"{bid}:{p}" for p in psem]
        if len(acts) > 3 and (cap / "action_03/snapshot.json").is_file():
            row["remaining_public_support"] = remaining_support(cap, b["context"], min_pixels)
            if not row["remaining_public_support"]["ok"]:
                problems.append(f"{bid}:remaining target not publicly supported after {b['candidate']}")
        else:
            row["remaining_public_support"] = {"ok": False, "reason": "candidate action not executed"}
            problems.append(f"{bid}:candidate action missing")
        qa_flags = []
        for qa in cap.glob("action_*/qa_state.jsonl"):
            for line in qa.read_text().splitlines():
                rec = json.loads(line)
                qa_flags += [rec.get(k) for k in ("used_by_planner", "used_by_policy", "used_by_provider", "used_by_verifier")]
        vflags = []
        for vp in cap.glob("action_*/views_perception.json"):
            vflags += [(f["hidden_truth_used"], f["qpos_qvel_used"]) for f in read(vp)]
        row["hidden_truth_flags_clean"] = bool(qa_flags) and not any(qa_flags) and not any(any(f) for f in vflags)
        if not row["hidden_truth_flags_clean"]:
            problems.append(f"{bid}:hidden-truth flags not clean")
        row["recorder_errors"] = r.get("recorder_errors", [])
        counts = r.get("env_counts", {})
        row["reset_calls"], row["constructions"] = counts.get("reset_calls"), counts.get("constructions")
        if row["recorder_errors"] or counts.get("reset_calls") != 1 or counts.get("constructions") != 1:
            problems.append(f"{bid}:recorder errors or reset/construction count")
    if pair["status"] != "PASS":
        problems.append("paired restore equivalence not established")
    scan = _scan_forbidden()
    if scan:
        problems.append("perception module references hidden/QA state: " + ",".join(scan))
    ledger = read(out / "budget_ledger.json")
    counts_ok = (ledger["v2_validation_attempts"]["used"] <= 4 and ledger["explicit_resets"]["used"] <= 4 and
                 ledger["environment_constructions"]["used"] <= 4 and
                 all(ledger[k]["used"] == 0 for k in ("provider_calls", "rl_transitions", "optimizer_steps", "elastic")))
    if not counts_ok:
        problems.append("budget counts exceed authorization")
    gate = {
        "card_id": CARD, "status": "PASS" if not problems else "FAIL",
        "label": GATE_PASS if not problems else "OBS_V2_TECHNICAL_GATE_FAIL",
        "problems": problems, "branches": per_branch, "paired_restore_status": pair["status"],
        "hidden_scan_hits": scan, "v2_validation_attempts_used": ledger["v2_validation_attempts"]["used"],
        "remaining_20_released": False, "compares_u_vs_v_cost": False,
    }
    write(out / "technical_gate_v2.json", gate)
    return gate


# ------------------------------------------------------------ timeline / reports
TIMELINE_FIELDS = (
    "branch_id", "layout", "context", "candidate", "action_index", "action_id", "frame_id", "sim_time",
    "object_id", "view", "mask_pixels", "depth_support", "blob_present", "blob_x", "blob_y", "blob_z",
    "reference_support", "support_ratio", "view_valid", "selected_view", "fused_present", "fused_x", "fused_y",
    "fused_z", "unknown_reason", "final_OnTable", "final_Held", "final_Inside_receiver", "final_AtBuffer_pad_u",
    "final_AtBuffer_pad_v", "final_GripperEmpty", "pick_mask_for_object", "planner_status", "evaluator_reason",
    "evaluator_success")


def timeline(out):
    import csv
    out = Path(out)
    reg = read(out / "physical/registration.json")
    rows = []
    for b in reg["branches"]:
        cap = out / "captures" / b["branch_id"]
        res = out / "physical/branch_results" / f"{b['branch_id']}.json"
        if not res.is_file():
            continue
        acts = read(res)["actions"]
        for i, a in enumerate(acts):
            d = cap / f"action_{i:02d}"
            vp = d / "views_perception.json"
            if not vp.is_file():
                continue
            facts = {f["fact_id"]: f["value"] for f in read(d / "facts.json")} if (d / "facts.json").is_file() else {}
            snap = read(d / "snapshot.json") if (d / "snapshot.json").is_file() else None
            ev = read(d / "evaluator.json") if (d / "evaluator.json").is_file() else {}
            pl = read(d / "planner.json")["status"] if (d / "planner.json").is_file() else "NOT_CALLED"
            for frame in read(vp):
                for obj in ("carrier", "obj_b", "obj_c"):
                    fused = frame["fusion"][obj]
                    for view in CAMERAS:
                        e = frame["cameras"][view]["objects"][obj]
                        inp = fused["inputs"][view]
                        mask = None
                        if snap is not None:
                            pid = f"a:PICK:{obj}:v1"
                            mask = bool(snap["candidate_mask"][snap["candidate_ids"].index(pid)])
                        xyz = e["xyz"] or [None] * 3
                        fx = (fused["blob"] or {}).get("xyz") or [None] * 3
                        rows.append({
                            "branch_id": b["branch_id"], "layout": b["layout"], "context": b["context"],
                            "candidate": b["candidate"], "action_index": i, "action_id": a["action_id"],
                            "frame_id": frame["frame_id"], "sim_time": frame["sim_time"], "object_id": obj, "view": view,
                            "mask_pixels": e["mask_pixels"], "depth_support": e["depth_support"],
                            "blob_present": e["blob_present"], "blob_x": xyz[0], "blob_y": xyz[1], "blob_z": xyz[2],
                            "reference_support": inp["reference"], "support_ratio": inp["ratio"], "view_valid": inp["valid"],
                            "selected_view": fused["selected_view"], "fused_present": fused["blob"] is not None,
                            "fused_x": fx[0], "fused_y": fx[1], "fused_z": fx[2], "unknown_reason": frame["unknown_reasons"].get(obj),
                            "final_OnTable": facts.get(f"p:OnTable:{obj}"), "final_Held": facts.get(f"p:Held:{obj}"),
                            "final_Inside_receiver": facts.get(f"p:Inside:{obj}:receiver"),
                            "final_AtBuffer_pad_u": facts.get(f"p:AtBuffer:{obj}:pad_u"),
                            "final_AtBuffer_pad_v": facts.get(f"p:AtBuffer:{obj}:pad_v"),
                            "final_GripperEmpty": facts.get("p:GripperEmpty"), "pick_mask_for_object": mask,
                            "planner_status": pl, "evaluator_reason": ev.get("reason"), "evaluator_success": ev.get("task_success")})
    with (out / "observation_support_timeline.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=TIMELINE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def branch_results_csv(out):
    import csv
    out = Path(out)
    reg = read(out / "physical/registration.json")
    fields = ("branch_id", "layout", "context", "repeat", "candidate", "status", "task_success", "actions",
              "controller_failures", "worker_wall_seconds", "reset_calls", "constructions", "recorder_errors",
              "time_to_task_success_NOT_COMPARED", "v1_branch_reference_only")
    with (out / "branch_results.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for b in reg["branches"]:
            p = out / "physical/branch_results" / f"{b['branch_id']}.json"
            r = read(p) if p.is_file() else {}
            w.writerow({"branch_id": b["branch_id"], "layout": b["layout"], "context": b["context"], "repeat": b["repeat"],
                        "candidate": b["candidate"], "status": r.get("status", "NO_RESULT"), "task_success": r.get("task_success"),
                        "actions": len(r.get("actions", [])),
                        "controller_failures": sum(a.get("controller_exit") not in ("NORMAL_TERMINATION", "SUCCESS")
                                                   for a in r.get("actions", [])),
                        "worker_wall_seconds": r.get("worker_wall_seconds"),
                        "reset_calls": r.get("env_counts", {}).get("reset_calls"),
                        "constructions": r.get("env_counts", {}).get("constructions"),
                        "recorder_errors": len(r.get("recorder_errors", [])),
                        "time_to_task_success_NOT_COMPARED": r.get("time_to_task_success"),
                        "v1_branch_reference_only": b["v1_logical_branch_id_reference_only"]})


def summarize(root, out):
    out = Path(out)
    gate = read(out / "technical_gate_v2.json")
    ledger = read(out / "budget_ledger.json")
    reg = read(out / "physical/registration.json")
    prof = read(out / "observation_profile_v2.json")
    lines = [
        "# Family B observation v2 technical validation", "",
        f"- Card: {CARD}; observation profile `{PROFILE_VERSION}` (sha256 {prof['profile_sha256'][:16]}), frozen before any v2 reset: yes (committed with the freeze).",
        f"- old v1 attempts = 4 (permanently preserved); new v2 validation attempts = {ledger['v2_validation_attempts']['used']} (cap 4).",
        "- old remaining 20 = NOT_RELEASED; provider = 0; RL = 0; optimizer = 0; S2/S3 = false; formal test = false.",
        f"- Technical gate: **{gate['status']}** ({gate['label']}).", "",
        "| branch | layout/context | candidate | status | remaining target supported | selected view |", "|---|---|---|---|---|---|"]
    for b in reg["branches"]:
        r = gate["branches"][b["branch_id"]]
        sup = r.get("remaining_public_support", {})
        lines.append(f"| {b['branch_id']} | {b['layout']}/{b['context']} | {b['candidate']} | {r.get('status')} | "
                     f"{sup.get('ok')} (OnTable={sup.get('on_table')}, PICK mask={sup.get('pick_mask')}) | {sup.get('selected_view')} |")
    lines += ["", f"- Paired restore: {gate['paired_restore_status']}.",
              "- Gate problems: " + ("none" if not gate["problems"] else "; ".join(gate["problems"])),
              "- U vs V cost is not compared by this card."]
    (out / "final_summary.md").write_text("\n".join(lines) + "\n")


def verify(root, out):
    out = Path(out)
    before = read(out / "protected_before.json")
    after = {"sha256": inventory(protected_paths())}
    after["capture_count"] = sum("/captures/" in k for k in after["sha256"])
    after["old_branch_commit"] = git(root, "rev-parse", "codex/cp-disr-s4-family-b-staging-v1")
    write(out / "protected_after.json", after)
    diff = diff_inventory(before["sha256"], after["sha256"])
    ledger = read(out / "budget_ledger.json")
    reg = read(out / "physical/registration.json")
    results = list((out / "physical/branch_results").glob("*.json"))
    checks = {
        "protected_unchanged": not (diff["changed"] or diff["removed"] or diff["added"]),
        "old_capture_count_unchanged": before["capture_count"] == after["capture_count"],
        "old_branch_commit_unchanged": before["old_branch_commit"] == after["old_branch_commit"] == BASE,
        "attempts_le_4": ledger["v2_validation_attempts"]["used"] <= 4 and len(results) <= 4,
        "resets_le_4": ledger["explicit_resets"]["used"] <= 4,
        "no_fifth_attempt": len(reg["branches"]) == 4,
        "provider_rl_optimizer_zero": all(ledger[k]["used"] == 0 for k in ("provider_calls", "rl_transitions", "optimizer_steps", "elastic")),
        "remaining_20_not_released": ledger["old_v1"]["original_remaining_20"] == "NOT_RELEASED",
        "gate_file_present": (out / "technical_gate_v2.json").is_file(),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protected_diff": diff,
              "protected_files": len(after["sha256"]), "capture_count": after["capture_count"]}
    write(out / "verify.json", result)
    return result


def junit_receipt(junit_path, out):
    import xml.etree.ElementTree as ET
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = [{"name": c.attrib["name"], "classname": c.attrib["classname"],
              "failed": c.find("failure") is not None or c.find("error") is not None} for c in suite.iter("testcase")]
    groups = {"A_v1_protection": "test_A", "B_observation_v2_config": "test_B", "C_runtime_binding": "test_C",
              "D_reverse_regressions": "test_D"}
    receipt = {"tests": len(cases), "failures": sum(c["failed"] for c in cases),
               "groups": {k: {"tests": sum(p in c["name"] for c in cases),
                              "failed": sum(p in c["name"] and c["failed"] for c in cases)} for k, p in groups.items()},
               "online_attempts_before_tests_passed": 0, "cases": cases}
    write(Path(out) / "test_receipt.json", receipt)
    return receipt
