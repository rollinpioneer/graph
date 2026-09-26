"""Plan v1.1 / Method 2.1.1 Stage 2A core ladder pilot. Historical stage-2a-run is unchanged."""
from __future__ import annotations

import csv, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import BindingError, DataIntegrityError, DiagnosticAbort, canonical, digest
from .neural import canonical_method
from .rl import gamma, set_suite_half_life, suite_half_life
from .platforms.libero.d0_env import CaseSpec, make_env
from .platforms.libero.snapshot import load_cache_edges
from .stage0c import PREPROCESSING, file_hash, prepare_scene
from .vlm import cache_key
from .vlm_cache_pipeline import request_and_process, write_audit_cache
from .vlm_provider import DashScopeProvider, ProviderConfig, validate_payload

from .stage1a_v11 import s1, save_ckpt, select_checkpoint, sha256_text, utc_now, utc_stamp, write_json, git_commit

STAGE_DIR = Path("runs/stage_2a")
STATUS_PATH = Path("status/stage_2a.json")
REPORT_PATH = Path("reports/stage_2a_summary.md")
RUNTIME_REL = Path("experiments/manifests/runtime_manifest_v211.yaml")
REFERENCE_REL = Path("runs/stage_0a/reference_execution_manifest.json")
STOP_REL = Path("experiments/part_2_exploration/stage_2a/orchestrator/STOP_SUPERSEDED_BY_PLAN_V1_1.json")
CLOCK_STOP_REL = Path("runs/stage_2a/orchestrator/STOP_CLOCK_INTEGRITY.json")
RUNTIME_REVISION = "clock_integrity_r2"
OLD_STAGE_DIR = Path("experiments/part_2_exploration/stage_2a")

CLOCK_RECOVERY_ALLOW_REL = Path("runs/stage_2a/orchestrator/CLOCK_RECOVERY_ALLOW.json")
DENYLISTED_STAMPS = {"20260923T121938Z"}
DENYLISTED_CONFIGSHA = {"79c690f9"}


def _sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_if_clock_stop(root, planned_id=None, stamp=None, configsha8=None, job_dir=None, phase=None):
    root = Path(root)
    hist = root / STOP_REL
    stop = root / CLOCK_STOP_REL
    if not hist.is_file():
        raise BindingError("historical Stage 2A STOP missing; refusing")
    if not stop.is_file():
        raise BindingError("CLOCK_INTEGRITY stop missing; refusing to run without stop evidence")
    allow_path = root / CLOCK_RECOVERY_ALLOW_REL
    if not allow_path.is_file():
        raise BindingError("CLOCK_INTEGRITY stop present; no CLOCK_RECOVERY_ALLOW record")
    allow = json.loads(allow_path.read_text(encoding="utf-8"))
    stop_hash = _sha256_file(stop)
    if allow.get("stop_clock_integrity_sha256") != stop_hash:
        raise BindingError("recovery allow record does not match STOP_CLOCK_INTEGRITY.json hash")
    if allow.get("runtime_revision") != RUNTIME_REVISION:
        raise BindingError("recovery allow runtime_revision mismatch")
    if allow.get("from_scratch") is not True:
        raise BindingError("recovery allow must set from_scratch=true")
    if stamp in DENYLISTED_STAMPS or configsha8 in DENYLISTED_CONFIGSHA:
        raise BindingError("refusing denylisted contaminated stamp/configsha")
    allowed_ids = list(allow.get("allowed_planned_ids") or [])
    if planned_id is not None and planned_id not in allowed_ids:
        raise BindingError("planned_id not in recovery allow record: %s" % planned_id)
    allowed_dirs = [str(Path(x)) for x in (allow.get("allowed_attempt_dirs") or [])]
    if job_dir is not None:
        jd = str(Path(job_dir))
        if jd not in allowed_dirs:
            raise BindingError("job_dir not in recovery allow record: %s" % jd)
    if phase in ("select", "eval", "report") and not allow.get("allow_select_eval_report"):
        raise BindingError("recovery allow does not yet permit select/eval/report")
    return allow
BASE_COMMIT = "b16fc1619ceb2a1b603a9bf151420bb672239291"
N_CAP = 65536
ROLLOUT_N = 1024
MAX_UPDATES = 64
EVAL_EVERY_N = 8192
DEV_EPISODES = 20
TEST_EPISODES = 30
POOL_N = 50
ACTIVE_N = 30
OBS_DIM = 48
CAND_DIM = 8
B_PRIOR = 0.5
MAX_EMPTY = 8
NAMESPACE = "cp_disr_exp_v1_1"
CACHE_ROOT = Path("experiments/vlm_cache/stage_2a")
TASKS = ("T_B", "T_C")
METHODS = ("B0", "B1-K", "B2", "Full")
EMPTY_PRIOR_METHODS = {"B0", "B1", "B1-K", "B2"}
TASK_ROLE = {"T_B": "second_object", "T_C": "interferer"}
SOURCE_SPLITS = {"T_B": Path("configs/splits/T_B_stage_0a.json"), "T_C": Path("configs/splits/T_C_stage_2a.json")}
ENABLED_SPLITS = {"T_B": Path("configs/splits/T_B_stage_2a_v11.json"), "T_C": Path("configs/splits/T_C_stage_2a_v11.json")}
TEST_POOL = {"T_B": Path("configs/splits/T_B_stage_2a_test_ids.json"), "T_C": Path("configs/splits/T_C_stage_2a_test_ids.json")}
TEST_ACTIVE = {"T_B": Path("configs/splits/T_B_stage_2a_test30.json"), "T_C": Path("configs/splits/T_C_stage_2a_test30.json")}
CONTAINER = [0.18, 0.12]
BUFFER = [-0.18, 0.12]
TARGET_BOX = (-0.22, -0.02, -0.18, -0.02)
SECOND_BOX = (0.04, 0.22, -0.18, -0.02)
MIN_DIST = 0.09
PLANNED = {
    ("T_B", "B0"): "v11_2A_T_B_B0_s0",
    ("T_B", "B1-K"): "v11_2A_T_B_B1-K_s0",
    ("T_B", "B2"): "v11_2A_T_B_B2_s0",
    ("T_B", "Full"): "v11_2A_T_B_Full_s0",
    ("T_C", "B0"): "v11_2A_T_C_B0_s0",
    ("T_C", "B1-K"): "v11_2A_T_C_B1-K_s0",
    ("T_C", "B2"): "v11_2A_T_C_B2_s0",
    ("T_C", "Full"): "v11_2A_T_C_Full_s0",
}


def clock_audit_transitions(records):
    from .platforms.libero.clock import CONTROL_DT
    n_logged = len(records)
    n_valid = 0
    n_invalid = 0
    n_unknown = 0
    exits = {}
    issues = []
    normal_failures = 0
    technical = 0
    mask_mismatch = 0
    for rec in records:
        reason = str(rec.get("reason") or rec.get("controller_exit") or "")
        exits[reason] = exits.get(reason, 0) + 1
        if reason in {"EXECUTION_FAILED", "CRITICAL_FACT_LOST", "VERIFICATION_FAILED", "TIMEOUT", "SAFETY_STOP"}:
            normal_failures += 1
        if reason.startswith("INTERRUPT_CONTROLLER_EXCEPTION") or reason.startswith("INTERRUPT_NAN") or reason.startswith("INTERRUPT_CLOCK"):
            technical += 1
            if len(issues) < 32:
                issues.append({"kind": "technical_exit_in_ppo", "reason": reason, "decision_id": rec.get("decision_id")})
        cid = rec.get("selected_candidate_id")
        ids = list(rec.get("candidate_ids") or [])
        mask = list(rec.get("mask") or [])
        if cid not in ids:
            mask_mismatch += 1
            if len(issues) < 32:
                issues.append({"kind": "candidate_not_in_ids", "decision_id": rec.get("decision_id"), "cid": cid})
        else:
            idx = ids.index(cid)
            if idx >= len(mask) or not bool(mask[idx]):
                mask_mismatch += 1
                if len(issues) < 32:
                    issues.append({"kind": "candidate_masked_false", "decision_id": rec.get("decision_id"), "cid": cid})
        sim_d = rec.get("sim_duration")
        start = rec.get("raw_sim_start")
        end = rec.get("raw_sim_end")
        steps = rec.get("control_steps")
        dur = rec.get("duration_seconds")
        if sim_d is None or start is None or end is None or dur is None:
            n_unknown += 1
            if len(issues) < 32:
                issues.append({"kind": "clock_fields_missing", "decision_id": rec.get("decision_id")})
            continue
        try:
            sim_d_f = float(sim_d)
            start_f = float(start)
            end_f = float(end)
            dur_f = float(dur)
        except (TypeError, ValueError):
            n_invalid += 1
            continue
        endpoint = end_f - start_f
        ok = (
            math.isfinite(sim_d_f) and math.isfinite(start_f) and math.isfinite(end_f) and math.isfinite(dur_f)
            and dur_f > 0.0 and sim_d_f > 0.0
            and abs(sim_d_f - endpoint) < 1e-9
            and abs(dur_f - sim_d_f) < 1e-9
        )
        if steps is None:
            ok = False
        else:
            try:
                steps_i = int(steps)
                ok = ok and steps_i > 0 and abs(sim_d_f - (steps_i * CONTROL_DT)) < 1e-6
            except (TypeError, ValueError):
                ok = False
        if ok:
            n_valid += 1
        else:
            n_invalid += 1
            if len(issues) < 32:
                issues.append({
                    "kind": "clock_mismatch",
                    "decision_id": rec.get("decision_id"),
                    "sim_duration": sim_d_f,
                    "duration_seconds": dur_f,
                    "raw_delta": endpoint,
                    "steps": steps,
                })
    return {
        "N_logged": n_logged,
        "N_used_by_PPO": n_logged,
        "N_clock_verified_valid": n_valid,
        "N_clock_invalid": n_invalid,
        "N_unknown": n_unknown,
        "exit_counts": exits,
        "normal_failure_n": normal_failures,
        "technical_n": technical,
        "mask_mismatch_n": mask_mismatch,
        "issues": issues,
    }


def log(msg):
    print("[stage2a_v11] %s %s" % (utc_now(), msg), flush=True)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def method_dir_name(method):
    return "B1-K" if canonical_method(method) == "B1" else method


def policy_method(method):
    return method if method != "B1" else "B1-K"


def unit_interval(payload):
    digest_bytes = hashlib.sha256(canonical(payload).encode("utf-8")).digest()
    return int.from_bytes(digest_bytes[:8], "big") / float(2 ** 64)


def derived_u32(payload):
    digest_bytes = hashlib.sha256(canonical(payload).encode("utf-8")).digest()
    return int.from_bytes(digest_bytes[:4], "big")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pose_key(row):
    return (
        round(float(row["target_xy"][0]), 12),
        round(float(row["target_xy"][1]), 12),
        round(float(row["second_xy"][0]), 12),
        round(float(row["second_xy"][1]), 12),
        round(float(row.get("container_xy", CONTAINER)[0]), 12),
        round(float(row.get("container_xy", CONTAINER)[1]), 12),
        round(float(row.get("buffer_xy", BUFFER)[0]), 12),
        round(float(row.get("buffer_xy", BUFFER)[1]), 12),
        bool(row.get("lid_closed", True)),
        str(row.get("task_id") or ""),
        str(row.get("second_role") or ""),
    )


def sample_xy(base, tag, box):
    x_lo, x_hi, y_lo, y_hi = box
    x = x_lo + (x_hi - x_lo) * unit_interval({**base, "axis": tag + "_x"})
    y = y_lo + (y_hi - y_lo) * unit_interval({**base, "axis": tag + "_y"})
    return [float(x), float(y)]


def dist(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def resolve_runtime(root):
    root = Path(root)
    ref = json.loads((root / REFERENCE_REL).read_text(encoding="utf-8"))
    runtime = load_yaml(root / RUNTIME_REL)
    H = float(ref["H"])
    rt_h = float(runtime["runtime"]["suite_H_seconds"])
    if H != rt_h:
        raise BindingError("reference H %s != runtime suite_H_seconds %s" % (H, rt_h))
    drefs = {}
    for task in TASKS:
        d_ref = float(ref["families"][task]["d_ref_median"])
        rt_dref = float(runtime["runtime"]["reference_skill_seconds_by_task"][task])
        if d_ref != rt_dref:
            raise BindingError("reference %s d_ref %s != runtime %s" % (task, d_ref, rt_dref))
        drefs[task] = d_ref
    return {
        "H": H,
        "d_ref": drefs,
        "Tcap": {t: float(N_CAP) * float(drefs[t]) for t in TASKS},
        "Ncap": N_CAP,
        "rollout": ROLLOUT_N,
        "max_updates": MAX_UPDATES,
        "actor_episode_discount_weight": False,
        "task_deadlines": {t: float(runtime["runtime"]["task_deadlines"][t]) for t in TASKS},
        "runtime_path": str(RUNTIME_REL).replace("\\", "/"),
        "runtime": runtime,
        "ref": ref,
    }


def bind_H(root):
    prof = resolve_runtime(root)
    set_suite_half_life(prof["H"])
    if suite_half_life() != prof["H"]:
        raise BindingError("suite H not bound")
    if abs(gamma(prof["H"]) - 0.5) > 1e-12:
        raise BindingError("gamma(H) is %s" % gamma(prof["H"]))
    return prof


def make_bundle(root, task_id, split_rel=None):
    from .platforms.libero.runtime_factory import create_task_runtime
    manifest = load_yaml(root / RUNTIME_REL)
    runtime = dict(manifest["runtime"])
    runtime["active_task_id"] = task_id
    splits = dict(runtime.get("task_splits") or {})
    splits[task_id] = str(split_rel or ENABLED_SPLITS[task_id]).replace("\\", "/")
    runtime["task_splits"] = splits
    manifest = dict(manifest)
    manifest["runtime"] = runtime
    return create_task_runtime(manifest, task_id)


def make_policy(template, method, device):
    from .neural import Policy
    from .platforms.libero.runtime_factory import PREDICATES, TASK_OBJECTS
    actions = sorted({c.name for c in template.contracts})
    objects = TASK_OBJECTS[template.task_id] if hasattr(template, "task_id") else None
    # types from template object map via runtime_factory
    from .platforms.libero.runtime_factory import TASK_OBJECTS as TO
    # Policy types are object type names
    types = sorted(set(TO.get("T_B" if "second_object" in str(TO) else "T_B").values()))
    # Use union of task object types
    type_set = set()
    for m in TO.values():
        type_set.update(m.values())
    model = Policy(actions=actions, predicates=sorted(PREDICATES), types=sorted(type_set), observation_dim=OBS_DIM, candidate_dim=CAND_DIM, method=policy_method(method), B=B_PRIOR)
    return model.to(device)


def apply_episode_prior(method, bundle, snapshot, sampler, eval_original=False):
    from .prior import EpisodePrior
    source_n = len(bundle.original_prior_edges)
    original = tuple(tuple(e) for e in bundle.original_prior_edges)
    original_hash = digest(original)
    m = policy_method(method)
    if m in EMPTY_PRIOR_METHODS or canonical_method(m) in ("B0", "B1", "B2"):
        snap = s1.empty_prior(snapshot)
        prior = EpisodePrior(snap.env_id, snap.episode_id, (), original_hash, digest(()), "absent")
        return snap, prior, source_n
    if eval_original:
        snap = s1.set_prior(snapshot, original)
        prior = EpisodePrior(snap.env_id, snap.episode_id, original, original_hash, digest(original), "original")
        return snap, prior, source_n
    prior = sampler.start(snapshot.env_id, snapshot.episode_id, original)
    snap = s1.set_prior(snapshot, prior.edges)
    return snap, prior, source_n


def require_case_cache(root, rec):
    case_id = rec["case_id"]
    cache_dir = rec.get("cache_dir")
    if not cache_dir:
        raise BindingError("missing cache for enabled case %s; not a legal empty prior" % case_id)
    folder = root / cache_dir
    if not (folder / "COMPLETE").is_file():
        raise BindingError("missing COMPLETE cache for enabled case %s; not a legal empty prior" % case_id)
    if not ((folder / "final_edges.json").is_file() or (folder / "accepted_relations.json").is_file()):
        raise BindingError("missing cache edges for enabled case %s; not a legal empty prior" % case_id)
    return folder


def attach_split_cases(bundle, root, rows, task_id, deadline):
    role = TASK_ROLE[task_id]
    for row in rows:
        spec = CaseSpec(
            case_id=row["case_id"], split=row["split"], seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row.get("container_xy") or CONTAINER), buffer_xy=tuple(row.get("buffer_xy") or BUFFER),
            lid_closed=bool(row.get("lid_closed", True)), task_id=task_id, second_role=role, deadline=float(deadline),
        )
        bundle.cases[row["case_id"]] = spec
        if row.get("cache_dir"):
            bundle.caches[row["case_id"]] = root / row["cache_dir"]


def frozen_configsha8(prof, git_hash):
    payload = {
        "H": prof["H"],
        "d_ref": prof["d_ref"],
        "Tcap": prof["Tcap"],
        "Ncap": N_CAP,
        "actor_episode_discount_weight": False,
        "runtime_path": prof["runtime_path"],
        "enabled_splits": {k: str(v).replace("\\", "/") for k, v in ENABLED_SPLITS.items()},
        "task_deadlines": prof["task_deadlines"],
        "git_hash": git_hash,
        "base_commit": BASE_COMMIT,
        "profile": "method-2.1.1-stage-2a",
        "methods": list(METHODS),
        "tasks": list(TASKS),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=s1._json_default))[:8]

def collect_used_inits(root, task_id):
    used = []
    root = Path(root)
    paths = [
        root / SOURCE_SPLITS[task_id],
        root / "configs/splits/T_A_stage_2a.json",
        root / "configs/splits/D0_stage_1a.json",
        root / "configs/splits/D0_stage_1a_test_ids.json",
        root / "configs/splits/T_B_stage_0a.json",
        root / "configs/splits/T_C_stage_2a.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        if isinstance(doc, dict):
            for key in ("train", "dev", "test", "active"):
                if isinstance(doc.get(key), list):
                    rows.extend(doc.get(key) or [])
        for row in rows:
            if isinstance(row, dict) and "target_xy" in row and "second_xy" in row:
                used.append({"source": str(path.relative_to(root)) + ":" + str(row.get("case_id")), "row": row})
    for base in [
        root / "experiments/stage_1a_inputs",
        root / "experiments/stage_2a_inputs",
        root / "experiments/stage_0c_inputs",
        root / "experiments/stage_0c_v11_inputs",
        root / "experiments/part_0_validation",
        root / "runs/stage_0a",
        root / "runs/stage_0d",
    ]:
        if not base.exists():
            continue
        for fp in base.rglob("reset_config.json"):
            try:
                rec = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(rec, dict) and "target_xy" in rec and "second_xy" in rec:
                used.append({"source": str(fp.relative_to(root)), "row": rec})
    seeds, poses = set(), set()
    for item in used:
        row = item["row"]
        if "seed" in row:
            try:
                seeds.add(int(row["seed"]))
            except Exception:
                pass
        if "target_xy" in row and "second_xy" in row:
            poses.add(pose_key({**row, "task_id": row.get("task_id") or task_id}))
    return used, seeds, poses


def legal_tb_pose(t, s):
    if not (TARGET_BOX[0] <= t[0] <= TARGET_BOX[1] and TARGET_BOX[2] <= t[1] <= TARGET_BOX[3]):
        return False, "target_out_of_box"
    if not (SECOND_BOX[0] <= s[0] <= SECOND_BOX[1] and SECOND_BOX[2] <= s[1] <= SECOND_BOX[3]):
        return False, "second_out_of_box"
    if dist(t, s) < MIN_DIST:
        return False, "min_distance"
    return True, None


def sample_test_case(task_id, index, used_seeds, used_poses):
    role = TASK_ROLE[task_id]
    reject_index = 0
    while reject_index < 128:
        base = {"namespace": NAMESPACE, "task": task_id, "split": "test", "case_or_episode": int(index), "stream": "case_init", "reject_index": int(reject_index)}
        t = sample_xy(base, "target", TARGET_BOX)
        if task_id == "T_C":
            vec = [CONTAINER[0] - t[0], CONTAINER[1] - t[1]]
            nrm = math.sqrt(vec[0] ** 2 + vec[1] ** 2) or 1.0
            s = [t[0] + 0.038 * vec[0] / nrm, t[1] + 0.038 * vec[1] / nrm]
            ok, reason = True, None
        else:
            s = sample_xy(base, "second", SECOND_BOX)
            ok, reason = legal_tb_pose(t, s)
        seed = derived_u32({"namespace": NAMESPACE, "task": task_id, "split": "test", "case_or_episode": int(index), "stream": "env_reset", "reject_index": int(reject_index)})
        row = {
            "case_id": "%s_test_%02d" % (task_id, index),
            "split": "test",
            "pool_index": int(index),
            "seed": int(seed),
            "target_xy": t,
            "second_xy": s,
            "container_xy": list(CONTAINER),
            "buffer_xy": list(BUFFER),
            "lid_closed": True,
            "task_id": task_id,
            "second_role": role,
            "stream": {"namespace": NAMESPACE, "task": task_id, "split": "test", "case_or_episode": int(index), "streams": ["case_init", "env_reset"], "reject_index": int(reject_index), "note": "test case stream omits method and training_seed"},
        }
        if not ok:
            reject_index += 1
            continue
        if seed in used_seeds or pose_key(row) in used_poses:
            reject_index += 1
            continue
        return row
    raise BindingError("unable to sample legal independent test case %s %s" % (task_id, index))


def sync_stage_1a_status(root):
    root = Path(root)
    path = root / "status/stage_1a.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    hist = root / "status/history" / ("stage_1a.pre_jobs_test_sync.%s.json" % utc_stamp())
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ft = doc.get("final_test") or {}
    fdir = doc.get("final_test_dir")
    for method in ("B2", "Full"):
        job = (doc.get("jobs") or {}).get(method) or {}
        rec = ft.get(method) or {}
        job["test"] = {
            "status": "PASS" if int(rec.get("valid") or 0) == 30 else "INCOMPLETE",
            "valid": rec.get("valid"),
            "success": rec.get("success"),
            "mean_G": rec.get("mean_G"),
            "final_test_dir": fdir,
            "note": "synced from completed Stage 1A final_test; training numbers unchanged",
        }
        doc["jobs"][method] = job
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"history": str(hist.relative_to(root)), "B2": doc["jobs"]["B2"]["test"], "Full": doc["jobs"]["Full"]["test"]}


def freeze_enabled_splits(root):
    root = Path(root)
    out = root / STAGE_DIR / "freeze"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for task in TASKS:
        src = json.loads((root / SOURCE_SPLITS[task]).read_text(encoding="utf-8"))
        train = list(src["train"])
        dev = list(src["dev"])
        if len(dev) != 20:
            raise BindingError("%s dev count is %s, required 20" % (task, len(dev)))
        if len(train) < 1:
            raise BindingError("%s has no train cases" % task)
        enabled = {
            "task_id": task,
            "version": "stage-2a-v11-enabled",
            "policy": "frozen_before_main_training_and_before_test_model_results",
            "source_split": str(SOURCE_SPLITS[task]).replace("\\", "/"),
            "source_split_sha256": sha256_file(root / SOURCE_SPLITS[task]),
            "reserved_pool_count": {"train": 128, "dev": 20, "test": 50},
            "active_case_count": {"train": len(train), "dev": len(dev), "test": ACTIVE_N},
            "train_count": len(train),
            "dev_count": len(dev),
            "test_isolated": True,
            "second_role": TASK_ROLE[task],
            "note": "Old split file not rewritten. Plan train128/test50 are pool capacities, not materialized counts.",
            "train": train,
            "dev": dev,
            "test": [],
        }
        dest = root / ENABLED_SPLITS[task]
        if dest.exists():
            prev = json.loads(dest.read_text(encoding="utf-8"))
            enabled["train"] = prev.get("train") or train
            enabled["dev"] = prev.get("dev") or dev
            enabled["test"] = prev.get("test") or []
            enabled["train_count"] = len(enabled["train"])
            enabled["dev_count"] = len(enabled["dev"])
            enabled["active_case_count"]["train"] = len(enabled["train"])
        write_json(dest, enabled)
        result[task] = {"train": len(enabled["train"]), "dev": len(enabled["dev"]), "path": str(ENABLED_SPLITS[task]).replace("\\", "/")}
        log("froze %s train=%s dev=%s" % (task, result[task]["train"], result[task]["dev"]))
    write_json(out / "enabled_counts.json", result)
    return result


def register_test_ids(root):
    root = Path(root)
    out = root / STAGE_DIR / "freeze"
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        pool_path = root / TEST_POOL[task]
        active_path = root / TEST_ACTIVE[task]
        if pool_path.is_file() and active_path.is_file():
            log("%s test-ID pool already frozen" % task)
            continue
        used, used_seeds, used_poses = collect_used_inits(root, task)
        pool = []
        for i in range(POOL_N):
            row = sample_test_case(task, i, used_seeds, used_poses)
            used_seeds.add(int(row["seed"]))
            used_poses.add(pose_key(row))
            pool.append(row)
        active = pool[:ACTIVE_N]
        registration = {
            "task_id": task,
            "registration_timing": "pre-training freeze; not a pre-training claim for unused historical stages",
            "frozen_before_main_training": True,
            "frozen_before_any_stage2a_test_model_result": True,
            "reserved_pool_count": POOL_N,
            "active_case_count": ACTIVE_N,
            "shared_by_methods": list(METHODS),
            "same_order": True,
            "same_reset_seed": True,
            "stream_omits_method_and_training_seed": True,
            "filtered_by_model_performance": False,
            "filtered_by_scripted_success": False,
            "filtered_by_nonempty_prior": False,
            "test": pool,
            "active": active,
            "reserved": pool[ACTIVE_N:],
        }
        write_json(pool_path, registration)
        write_json(active_path, {"task_id": task, "active": active, "n": len(active), "shared_by_methods": list(METHODS), "same_order": True, "same_reset_seed": True})
        enabled = json.loads((root / ENABLED_SPLITS[task]).read_text(encoding="utf-8"))
        enabled["test"] = active
        enabled["test_count"] = len(active)
        write_json(root / ENABLED_SPLITS[task], enabled)
        write_json(out / ("%s_split_overlap_audit.json" % task), {
            "n_used_sources": len(used),
            "n_used_seeds": len(used_seeds),
            "n_used_poses": len(used_poses),
            "any_overlap": False,
            "independence_rule": "reject exact pose/seed collision; T_B uses min-distance box; T_C uses interferer offset",
        })
        log("registered %s test pool 50 / active 30" % task)
    return {"status": "FROZEN"}

def capture_rows(root, task_id, rows, gpu=0):
    from PIL import Image
    mapping = []
    dest_root = root / "experiments/stage_2a_inputs" / task_id
    dest_root.mkdir(parents=True, exist_ok=True)
    deadline = float(resolve_runtime(root)["task_deadlines"][task_id])
    for row in rows:
        dest = dest_root / row["split"] / row["case_id"]
        dest.mkdir(parents=True, exist_ok=True)
        rgb_path = dest / "rgb.png"
        if (dest / "CAPTURE_COMPLETE").is_file() and rgb_path.is_file():
            mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)).replace("\\", "/"), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root)).replace("\\", "/")})
            continue
        spec = CaseSpec(
            case_id=row["case_id"], split=row["split"], seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row.get("container_xy") or CONTAINER), buffer_xy=tuple(row.get("buffer_xy") or BUFFER),
            lid_closed=bool(row.get("lid_closed", True)), task_id=task_id, second_role=TASK_ROLE[task_id], deadline=deadline,
        )
        env = make_env(spec, gpu=gpu)
        try:
            env.reset()
            env._apply_case_poses()
            env._open_gripper_reset()
            obs = env.public_observation()
            Image.fromarray(np.asarray(obs["rgb"]).astype(np.uint8)).save(rgb_path)
            np.save(dest / "depth.npy", np.asarray(obs["depth"]))
            write_json(dest / "reset_config.json", row)
            (dest / "CAPTURE_COMPLETE").write_text("complete\n", encoding="utf-8")
        finally:
            env.close()
        mapping.append({**row, "image_ref": str(rgb_path.relative_to(root)).replace("\\", "/"), "image_sha256": file_hash(rgb_path), "input_dir": str(dest.relative_to(root)).replace("\\", "/")})
        log("captured %s" % row["case_id"])
    return mapping


def _grounded_template(root, task_id):
    from .contracts import Registry, from_dict
    from .graph import Goal, build_template
    from .platforms.libero.runtime_factory import PREDICATES, TASK_OBJECTS, TASK_GOALS
    doc = load_yaml(root / "configs/runtime/stage_2a_contract_registry.yaml")
    objects = TASK_OBJECTS[task_id]
    reg = Registry(doc["predicate_types"])
    for c in doc["contracts"]:
        reg.register(from_dict(c))
    contracts = reg.ground(objects)
    goals = tuple(Goal(g, 1) for g in TASK_GOALS[task_id])
    template = build_template(contracts, goals, PREDICATES, objects)
    return doc, contracts, template


def initial_facts(task_id):
    role = TASK_ROLE[task_id]
    return {
        "p:GripperEmpty": "TRUE", "p:Held:target": "FALSE", "p:Held:%s" % role: "FALSE",
        "p:OnTable:target": "TRUE", "p:OnTable:%s" % role: "TRUE", "p:Open:container": "FALSE",
        "p:Inside:target:container": "FALSE", "p:Inside:%s:container" % role: "FALSE",
        "p:AtBuffer:target:buffer": "FALSE", "p:AtBuffer:%s:buffer" % role: "FALSE",
    }


def find_reusable_cache(root, key, split):
    candidates = [
        root / CACHE_ROOT / split / key,
        root / "experiments/vlm_cache/plan_v11" / split / key,
        root / "experiments/vlm_cache" / split / key,
    ]
    for path in candidates:
        if path.is_dir() and (path / "COMPLETE").is_file() and (path / "manifest.json").is_file():
            try:
                man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            if man.get("processing_status") == "SUCCESS":
                return path
    return None


def materialize_task_caches(root, task_id, rows, gpu=0):
    root = Path(root)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise BindingError("DASHSCOPE_API_KEY missing from process environment; BLOCKED without model fallback")
    vlm = load_yaml(root / "experiments/manifests/vlm_manifest.yaml")
    config = ProviderConfig(model=vlm["model_snapshot"], region=vlm["region"], endpoint=vlm["base_http_api_url"], sdk_version=vlm["sdk_version"])
    provider = DashScopeProvider(config)
    if not provider.key_present():
        raise BindingError("DASHSCOPE_API_KEY missing from process environment")
    few_path = root / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json"
    few_doc = json.loads(few_path.read_text(encoding="utf-8"))
    few = few_doc.get("examples", [])
    examples = [prepare_scene(root, s, True) for s in few]
    prompt_path = root / "experiments/sources/v2.1_interfaces/system_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    schema_path = root / "schemas/relation_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for ex, record in zip(examples, few):
        messages.extend([{"role": "user", "content": ex["content"]}, {"role": "assistant", "content": [{"text": canonical(record["expected_json"])}]}])
    contract_doc, contracts, template = _grounded_template(root, task_id)
    actions = sorted(c.id for c in contracts)
    props = sorted(n.id for n in template.nodes if n.kind == "PROPOSITION")
    contract_sha = file_hash(root / "configs/runtime/stage_2a_contract_registry.yaml")
    task_path = root / ("configs/tasks/resolved/%s.yaml" % task_id)
    if not task_path.exists():
        raise BindingError("resolved task missing: " + str(task_path))
    task_sha = file_hash(task_path)
    import base64
    from PIL import Image
    role = TASK_ROLE[task_id]
    signed = [{"fact_id": g, "sign": 1} for g in __import__("cp_disr.platforms.libero.runtime_factory", fromlist=["TASK_GOALS"]).TASK_GOALS[task_id]]
    ledger = root / STAGE_DIR / "cache" / ("%s_request_ledger.jsonl" % task_id)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    mapping = capture_rows(root, task_id, rows, gpu=gpu)
    results = []
    vlm_ids = {
        "model_snapshot": config.model,
        "sdk_api_version": config.sdk_version,
        "region": config.region,
        "endpoint": config.endpoint,
        "prompt_hash": file_hash(prompt_path),
        "fewshot_hash": digest(few_doc),
        "schema_hash": file_hash(schema_path),
    }
    for row in mapping:
        existing = row.get("cache_dir")
        if existing:
            folder = root / existing
            if (folder / "COMPLETE").is_file() and (folder / "manifest.json").is_file():
                man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                ok = man.get("processing_status") == "SUCCESS" and man.get("task_id") == task_id and man.get("model_snapshot") == vlm_ids["model_snapshot"] and man.get("prompt_hash") == vlm_ids["prompt_hash"] and man.get("fewshot_hash") == vlm_ids["fewshot_hash"] and man.get("schema_hash") == vlm_ids["schema_hash"] and man.get("region") == vlm_ids["region"] and man.get("endpoint") == vlm_ids["endpoint"]
                img_ok = (not row.get("image_sha256")) or man.get("initial_RGB_content_hash") == row.get("image_sha256")
                if ok and img_ok:
                    results.append({**row, "cache_dir": existing.replace("\\", "/"), "cache_key": folder.name, "cache_status": "REUSED_EXISTING", "status": "REUSED"})
                    continue

        image = root / row["image_ref"]
        with Image.open(image) as im:
            mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[im.format]
        scene = {
            "scene_id": row["case_id"], "task_id": task_id, "split": row["split"],
            "image_ref": row["image_ref"], "image_sha256": row["image_sha256"],
            "object_table": [
                {"id": "target", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": role, "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "container", "type": "container", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
                {"id": "buffer", "type": "buffer", "binding_status": "VERIFIED", "visual_evidence_ref": row["image_ref"]},
            ],
            "signed_goals": signed, "initial_facts": initial_facts(task_id),
            "allowed_action_ids": actions, "allowed_proposition_ids": props,
            "contracts_sha256": contract_sha, "task_definition_sha256": task_sha,
            "predicate_version": "CP-DISR-v2.1",
            "asset_binding_sha256": digest({"layout": [row["target_xy"], row["second_xy"], row["container_xy"], row["buffer_xy"]], "runtime": "stage-2a-v11", "task_id": task_id}),
        }
        context = {k: scene[k] for k in ["scene_id", "task_id", "object_table", "signed_goals", "initial_facts", "allowed_action_ids", "allowed_proposition_ids"]}
        context["object_table"] = sorted(context["object_table"], key=lambda x: x["id"])
        context["skill_contracts"] = contract_doc
        context["registered_effect_edges"] = [e for e in template.edges if e[2] in ("ADD", "DEL")]
        content = [{"image": "data:image/" + mime + ";base64," + base64.b64encode(image.read_bytes()).decode()}, {"text": canonical(context)}]
        payload = {"model": config.model, "messages": messages + [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 2048, "response_format": {"type": "json_object"}, "enable_thinking": False, "enable_search": False, "stream": False, "result_format": "message"}
        validate_payload(payload)
        manifest = {
            "split": row["split"], "task_definition_hash": scene["task_definition_sha256"],
            "initial_RGB_content_hash": scene["image_sha256"], "preprocessing_hash": digest(PREPROCESSING),
            "object_binding_hash": digest(context["object_table"]),
            "allowed_ID_hash": digest({"actions": actions, "propositions": props, "signed_goals": scene["signed_goals"]}),
            "contract_version": contract_sha, "predicate_version": scene["predicate_version"],
            "model_snapshot": config.model, "sdk_api_version": config.sdk_version, "region": config.region, "endpoint": config.endpoint,
            "prompt_hash": file_hash(prompt_path), "fewshot_hash": digest(few_doc), "schema_hash": file_hash(schema_path),
            "decoding_config": {"temperature": 0, "max_tokens": 2048, "max_relations": 8, "thinking": False, "response_format": "json_object"},
            "scene_id": scene["scene_id"], "task_id": task_id, "synthetic_unit_fixture": False,
            "initial_facts_hash": digest(scene["initial_facts"]), "input_context_hash": digest(context),
            "request_payload_hash": digest(payload), "asset_binding_hash": scene["asset_binding_sha256"],
        }
        key = cache_key(manifest)
        reused = find_reusable_cache(root, key, row["split"])
        if reused is not None:
            rec = {**row, "cache_key": key, "cache_dir": str(reused.relative_to(root)).replace("\\", "/"), "cache_status": "REUSED", "status": "REUSED"}
            results.append(rec)
            continue
        path = root / CACHE_ROOT / row["split"] / key
        if path.exists():
            raise BindingError("Refusing to rmtree existing cache path for %s; missing cache is not EMPTY_PRIOR" % row["case_id"])
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({"scene_id": row["case_id"], "state": "REQUEST_STARTED", "time": time.time()}) + "\n")
        execution = request_and_process(provider, payload, template, schema, ())
        written = write_audit_cache(root / CACHE_ROOT, manifest, prompt, scene, execution)
        rec = {**row, "cache_key": key, "cache_dir": str(Path(written).relative_to(root)).replace("\\", "/"), "cache_status": execution["status"], "status": execution["status"], "attempts": len(execution["attempts"])}
        if execution["status"] != "SUCCESS":
            complete_file = Path(written) / "COMPLETE"
            if complete_file.exists():
                complete_file.unlink()
            (Path(written) / "FAILED").write_text(str(execution["status"]) + "\n", encoding="utf-8")
            raise BindingError("cache generation failed for %s: %s; missing cache is not EMPTY_PRIOR" % (row["case_id"], execution["status"]))
        results.append(rec)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(canonical({"scene_id": row["case_id"], "state": rec["status"], "cache_key": key}) + "\n")
        log("cache %s %s %s" % (task_id, row["case_id"], rec["status"]))
    return results


def materialize_all(root, gpu=0):
    root = Path(root)
    audit = {}
    for task in TASKS:
        enabled = json.loads((root / ENABLED_SPLITS[task]).read_text(encoding="utf-8"))
        rows = list(enabled["train"]) + list(enabled["dev"]) + list(enabled.get("test") or [])
        need = [r for r in rows if (not r.get("cache_dir")) or not (root / r["cache_dir"] / "COMPLETE").is_file()]
        # Always identity-check by rematerialize reuse path for rows missing COMPLETE.
        produced = materialize_task_caches(root, task, need, gpu=gpu) if need else []
        by_id = {r["case_id"]: r for r in produced}
        for split in ("train", "dev", "test"):
            updated = []
            for r in enabled.get(split) or []:
                if r["case_id"] in by_id:
                    src = by_id[r["case_id"]]
                    r = {**r, "cache_dir": src.get("cache_dir") or r.get("cache_dir"), "cache_key": src.get("cache_key") or r.get("cache_key"), "cache_status": src.get("cache_status") or r.get("cache_status"), "image_ref": src.get("image_ref") or r.get("image_ref"), "image_sha256": src.get("image_sha256") or r.get("image_sha256")}
                updated.append(r)
            enabled[split] = updated
        write_json(root / ENABLED_SPLITS[task], enabled)
        nonempty = 0
        empty = 0
        missing = 0
        for r in enabled["train"] + enabled["dev"] + enabled["test"]:
            folder = root / r["cache_dir"] if r.get("cache_dir") else None
            if folder is None or not (folder / "COMPLETE").is_file():
                missing += 1
                continue
            edges = load_cache_edges(folder)
            if edges:
                nonempty += 1
            else:
                empty += 1
        audit[task] = {"n": len(enabled["train"]) + len(enabled["dev"]) + len(enabled["test"]), "n_missing": missing, "n_legal_empty": empty, "n_nonempty": nonempty, "new_or_rechecked": len(produced)}
        log("cache audit %s %s" % (task, audit[task]))
        if missing:
            raise BindingError("%s still missing caches; not EMPTY_PRIOR" % task)
    write_json(root / STAGE_DIR / "freeze" / "cache_audit.json", audit)
    return audit

def gate_case_id(task_id, split_doc):
    for rec in split_doc["dev"]:
        if rec.get("cache_dir"):
            return rec["case_id"]
    raise BindingError("no cached dev case for startup gate")


def startup_hard_checks(root, out, device, task_id):
    root = Path(root)
    out = Path(out)
    gates = []
    prof = bind_H(root)
    split = json.loads((root / ENABLED_SPLITS[task_id]).read_text(encoding="utf-8"))
    stop_ok = (root / STOP_REL).is_file()
    gates.append({"gate": "OLD_STAGE2A_STOP_GUARD", "passed": stop_ok, "issues": [] if stop_ok else ["STOP missing"]})
    old_dir_ok = (root / OLD_STAGE_DIR).is_dir()
    gates.append({"gate": "OLD_STAGE2A_DIR_PRESERVED", "passed": old_dir_ok, "issues": [] if old_dir_ok else ["old dir missing"]})
    d_ref = prof["d_ref"][task_id]
    tcap = prof["Tcap"][task_id]
    gates.append({"gate": "CALIBRATED_H_DREF_TCAP", "passed": True, "H": prof["H"], "d_ref": d_ref, "Tcap": tcap, "actor_episode_discount_weight": False})
    gates.append({"gate": "GAMMA_HALF_LIFE", "passed": True, "gamma_H": gamma(prof["H"])})
    enabled = list(split["train"]) + list(split["dev"])
    missing = [r["case_id"] for r in enabled if not r.get("cache_dir") or not (root / r["cache_dir"] / "COMPLETE").is_file()]
    gates.append({"gate": "CACHE_AUDIT", "passed": not missing, "issues": missing, "n": len(enabled), "n_missing": len(missing)})
    erratum = root / "runs/stage_0d/empty_patch_erratum.json"
    gates.append({"gate": "STAGE0D_ERRATUM", "passed": erratum.is_file(), "issues": [] if erratum.is_file() else ["0D erratum missing"]})
    bundle = make_bundle(root, task_id)
    attach_split_cases(bundle, root, enabled + list(split.get("test") or []), task_id, prof["task_deadlines"][task_id])
    gate_case = gate_case_id(task_id, split)
    try:
        for method in METHODS:
            empty = method != "Full"
            dry = local_dry_run(bundle, method, device, gate_case, empty=empty, force_original=(method == "Full"))
            write_json(out / "startup" / task_id / ("%s_dry.json" % method_dir_name(method)), dry)
            issues = []
            if method in ("B0", "B1-K", "B2") and dry["effective_prior_relation_count"] != 0:
                issues.append("%s effective prior not empty" % method)
            if method == "B2" and any((s.get("dp_rms") or 0) != 0 or (s.get("residual_abs") or 0) != 0 for s in dry["forward_structs"]):
                issues.append("B2 DP/Delta not strictly 0")
            if method == "B1-K":
                if any(s.get("successor_used") for s in dry["forward_structs"]):
                    issues.append("B1-K successor used")
            if method == "B0":
                if any(s.get("successor_used") for s in dry["forward_structs"]):
                    issues.append("B0 used graph/nominal successor")
            if not dry["logits_finite"]:
                issues.append("%s logits nonfinite" % method)
            note = None
            if method == "Full":
                src_n = dry["source_cache_relation_count"]
                eff_n = dry["effective_prior_relation_count"]
                qualifying = [s for s in dry["forward_structs"] if (not s["masked"]) and s["nominal_patch_nonempty"] and eff_n > 0]
                dp_hit = any(s["dp_nonzero"] for s in qualifying)
                delta_hit = any(s["residual_nonzero"] for s in qualifying)
                if src_n <= 0:
                    note = "N/A: source cache legally empty"
                elif not qualifying:
                    note = "N/A: no qualifying nonempty-prior changed-patch candidate"
                elif not dp_hit or not delta_hit:
                    issues.append("Full has opportunity but DP/Delta witness missing")
            gates.append({"gate": "%s_%s_DRY" % (task_id, method), "passed": not issues, "issues": issues, "note": note, "alias": "B1-K" if method == "B1-K" else method})
        s1.seed_all(0)
        p1 = make_policy(bundle.template, "B0", device)
        s1.seed_all(0)
        p2 = make_policy(bundle.template, "Full", device)
        shared = {id(p) for p in p1.parameters()} & {id(p) for p in p2.parameters()}
        gates.append({"gate": "INDEPENDENT_INIT", "passed": not shared, "issues": ["shared parameters"] if shared else []})
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
    blocked = any(not g["passed"] for g in gates)
    write_json(out / "startup" / task_id / "summary.json", {"gates": gates, "blocked": blocked, "time": utc_now(), "task": task_id, "H": prof["H"], "d_ref": d_ref, "Tcap": tcap})
    return gates, blocked, prof


def source_hashes(root, task_id):
    root = Path(root)
    files = {
        "runtime_manifest": root / RUNTIME_REL,
        "enabled_split": root / ENABLED_SPLITS[task_id],
        "source_split": root / SOURCE_SPLITS[task_id],
        "stage2a_v11": root / "src/cp_disr/stage2a_v11.py",
        "neural": root / "src/cp_disr/neural.py",
        "torch_rl": root / "src/cp_disr/torch_rl.py",
        "runtime_factory": root / "src/cp_disr/platforms/libero/runtime_factory.py",
        "clock": root / "src/cp_disr/platforms/libero/clock.py",
        "safety": root / "src/cp_disr/platforms/libero/safety.py",
        "skill_executor": root / "src/cp_disr/platforms/libero/skill_executor.py",
        "collector": root / "src/cp_disr/collector.py",
        "reference": root / REFERENCE_REL,
    }
    return {k: sha256_file(v) if v.is_file() else None for k, v in files.items()} | {"git_commit": git_commit(root), "base_commit": BASE_COMMIT}


def aggregate_train_buffer(method, trans_rows):
    agg = s1.aggregate_struct(trans_rows)
    qual = [r for r in trans_rows if r.get("qualifying_for_dp")]
    agg["dp_opportunity_n"] = len(qual)
    agg["dp_nonzero_n"] = sum(1 for r in qual if r["struct"]["dp_nonzero"])
    agg["delta_nonzero_n"] = sum(1 for r in qual if r["struct"]["residual_nonzero"])
    if policy_method(method) in EMPTY_PRIOR_METHODS:
        agg["dp_opportunity_n"] = 0
        agg["dp_rate"] = "N/A"
        agg["delta_rate"] = "N/A"
    else:
        opp = agg["dp_opportunity_n"]
        agg["dp_rate"] = None if opp == 0 else agg["dp_nonzero_n"] / opp
        agg["delta_rate"] = None if opp == 0 else agg["delta_nonzero_n"] / opp
        if opp == 0:
            agg["dp_rate"] = "N/A"
            agg["delta_rate"] = "N/A"
    return agg


def eval_episodes(root, task_id, method, ckpt_path, cases, split_index, device, hashes_doc, out_eval, n_episodes=None, label="dev"):
    from .collector import Collector
    from .torch_rl import load_checkpoint
    bind_H(root)
    rng_before = s1.capture_rng()
    split_rel = ENABLED_SPLITS[task_id]
    bundle = make_bundle(root, task_id, split_rel)
    prof = resolve_runtime(root)
    attach_split_cases(bundle, root, list(split_index.values()), task_id, prof["task_deadlines"][task_id])
    s1.seed_all(0)
    policy = make_policy(bundle.template, method, device)
    load_checkpoint(ckpt_path, policy)
    policy.eval()
    collector = Collector(bundle, policy)
    rows = []
    success_n = 0
    returns = []
    try:
        for case in cases[:(n_episodes or len(cases))]:
            rec = split_index[case]
            require_case_cache(root, rec)
            snap = bundle.start_case(case)
            snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler=None, eval_original=True)
            collector.reset_episode(snap.env_id, snap.episode_id)
            G, steps, success, reason, success_seconds = 0.0, 0, False, None, None
            while True:
                t, result = collector.step(snap, deterministic=True)
                if t is None:
                    reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                    success = bool(result.get("success") if isinstance(result, dict) else getattr(result, "success", False))
                    break
                G += float(t.reward) * float(t.weight)
                steps += 1
                if result.success if not isinstance(result, dict) else result.get("success"):
                    success = True
                    success_seconds = float(t.snapshot.elapsed_seconds + t.duration) if hasattr(t.snapshot, "elapsed_seconds") else t.duration
                snap = t.next_snapshot
                if t.terminated or t.truncated:
                    reason = result.reason if not isinstance(result, dict) else result.get("reason")
                    break
            if success:
                success_n += 1
            returns.append(G)
            rows.append({"case_id": case, "success": success, "G": G, "steps": steps, "reason": reason, "success_seconds": success_seconds if success else None, "source_n": source_n, "prior_mode": prior.audit_mode})
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
        s1.restore_rng(rng_before)
    mean_G = float(sum(returns) / len(returns)) if returns else None
    payload = {
        "task": task_id, "method": method, "alias": method_dir_name(method), "checkpoint": str(ckpt_path),
        "success_n": success_n, "n": len(rows),
        "success_rate": (success_n / max(1, len(rows))) if rows else None,
        "mean_discounted_return": mean_G, "rows": rows, "hashes": hashes_doc,
        "isolated_rng": True, "H": suite_half_life(), "eval_action": "deterministic_argmax", "label": label,
        "optimizer_steps": 0,
    }
    write_json(out_eval, payload)
    return payload

def local_dry_run(bundle, method, device, case_id, empty=False, force_original=False):
    from .collector import Collector
    s1.seed_all(0)
    snap0 = bundle.start_case(case_id)
    source_n = len(bundle.original_prior_edges)
    original_hash = digest(tuple(tuple(e) for e in bundle.original_prior_edges))
    if empty:
        snap = s1.empty_prior(snap0)
        prior_mode = "absent"
    elif force_original:
        snap = s1.set_prior(snap0, bundle.original_prior_edges)
        prior_mode = "original"
    else:
        snap = snap0
        prior_mode = "source_cache"
    bundle.current_snapshot = snap
    policy = make_policy(bundle.template, method, device)
    policy.eval()
    collector = Collector(bundle, policy)
    collector.reset_episode(snap.env_id, snap.episode_id)
    records = []
    structs = []
    with torch.no_grad():
        hidden = policy.initial_hidden()
        out0 = policy(snap, hidden)
    for cid, m in zip(snap.candidate_ids, snap.mask):
        if m:
            st = s1.candidate_struct(out0, snap, cid)
            st["successor_used"] = bool((out0.diagnostics or {}).get("successor_used"))
            structs.append(st)
    for _ in range(2):
        t, result = collector.step(bundle.current_snapshot)
        if t is None:
            records.append({"no_transition": True, "result": str(result)})
            break
        rec = s1.compact_transition(method, 0, case_id, t, result, collector.last_output, collector.last_execution, prior_mode, original_hash, source_n, None, "startup_gate")
        records.append(rec)
        bundle.current_snapshot = t.next_snapshot
        if t.terminated or t.truncated:
            break
    diag = out0.diagnostics or {}
    safe_diag = {
        "successor_used": bool(diag.get("successor_used")),
        "method": diag.get("method"),
        "actor_episode_discount_weight": diag.get("actor_episode_discount_weight"),
    }
    return {
        "method": method, "alias": method_dir_name(method), "case_id": case_id,
        "source_cache_relation_count": source_n, "effective_prior_relation_count": len(snap.prior_edges),
        "candidate_ids": list(snap.candidate_ids), "mask": [bool(x) for x in snap.mask],
        "logits_finite": s1.finite_masked_logits(out0), "forward_structs": structs, "transitions": records,
        "diagnostics": safe_diag,
    }


def maybe_eval_at_n(root, task_id, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns):
    if count in done_ns:
        return
    if count != 0 and (count % EVAL_EVERY_N) != 0:
        return
    extra = dict(extra_base)
    extra.update({"update_count": extra_base.get("complete_updates"), "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds})
    ckpt = job_dir / "checkpoints" / ("n_%06d.pt" % count)
    save_ckpt(ckpt, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector)
    log("%s %s eval N=%s" % (task_id, method, count))
    ev = eval_episodes(root, task_id, method, ckpt, eval_cases, split_index, device, hashes_doc, job_dir / ("eval_n_%06d.json" % count), n_episodes=DEV_EPISODES, label="dev")
    eval_rows.append({"update": extra_base.get("complete_updates"), "skill_transitions": count, "success_rate": ev["success_rate"], "success_n": ev["success_n"], "mean_discounted_return": ev["mean_discounted_return"], "checkpoint": str(ckpt)})
    s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
    done_ns.add(count)


def start_episode(root, task_id, method, bundle, cases, split_index, sampler, collector, job_dir, eval_original=False):
    case = bundle.next_case(cases, 0)
    rec = split_index[case]
    require_case_cache(root, rec)
    snap = bundle.start_case(case)
    snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler, eval_original=eval_original)
    bundle.current_snapshot = snap
    collector.reset_episode(snap.env_id, snap.episode_id)
    cache_key_s = str(root / rec["cache_dir"])
    s1.append_jsonl(job_dir / "episode_priors.jsonl", {
        "env_id": prior.env_id, "episode_id": prior.episode_id, "audit_mode": prior.audit_mode,
        "original_hash": prior.original_hash, "effective_hash": prior.hash,
        "effective_relation_count": len(prior.edges), "source_cache_relation_count": source_n,
        "case_id": case, "method": method, "task": task_id, "eval_original": eval_original,
    })
    return case, prior, source_n, cache_key_s


def param_fingerprint(policy):
    h = hashlib.sha256()
    for k, v in policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_job(root, task_id, method, device, device_name, prof, hashes_doc, stamp, configsha8, max_updates=MAX_UPDATES, resume=False):
    bind_H(root)
    root = Path(root)
    mname = method_dir_name(method)
    job_dir = root / STAGE_DIR / task_id / mname / "seed_0" / ("%s_%s" % (stamp, configsha8))
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "checkpoints").mkdir(exist_ok=True)
    (job_dir / "plots").mkdir(exist_ok=True)
    split = json.loads((root / ENABLED_SPLITS[task_id]).read_text(encoding="utf-8"))
    cases = [r["case_id"] for r in split["train"]]
    eval_cases = [r["case_id"] for r in split["dev"]]
    split_index = {r["case_id"]: r for r in (split["train"] + split["dev"] + list(split.get("test") or []))}
    planned_id = PLANNED[(task_id, mname)]
    run_id = "%s_%s_%s" % (planned_id, stamp, configsha8)
    cfg = {
        "planned_id": planned_id, "run_id": run_id, "method": mname, "method_internal": canonical_method(method),
        "alias": "B1-K" if canonical_method(method) == "B1" else mname, "seed": 0, "task": task_id,
        "H": prof["H"], "d_ref": prof["d_ref"][task_id], "Tcap": prof["Tcap"][task_id], "Ncap": N_CAP,
        "actor_episode_discount_weight": False, "gamma_rule": "2**(-duration_seconds/H)",
        "runtime_path": prof["runtime_path"], "split_path": str(ENABLED_SPLITS[task_id]).replace("\\", "/"),
        "task_deadline_seconds": prof["task_deadlines"][task_id], "train_ids": cases, "dev_ids": eval_cases,
        "test_ids": [r["case_id"] for r in split.get("test") or []], "hashes": hashes_doc,
        "old_output_dir_unused": str(OLD_STAGE_DIR), "init": "from_scratch", "warm_start": False,
        "b1_variant": "B1-K" if canonical_method(method) == "B1" else None,
        "runtime_revision": RUNTIME_REVISION,
    }
    write_json(job_dir / "resolved_config.json", cfg)
    (job_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(job_dir / "environment_snapshot.json", s1.environment_snapshot(device, device_name, 0))
    write_json(job_dir / "software_snapshot.json", s1.software_snapshot(root))
    write_json(job_dir / "manifest.json", {"run_id": run_id, "planned_id": planned_id, "method": mname, "seed": 0, "task": task_id, "hashes": hashes_doc, "H": prof["H"], "d_ref": cfg["d_ref"], "Tcap": cfg["Tcap"]})
    from .collector import Collector
    from .torch_rl import PPO, load_checkpoint
    from .prior import PriorSampler
    from .rl import Rollout
    s1.seed_all(0)
    bundle = make_bundle(root, task_id)
    attach_split_cases(bundle, root, list(split_index.values()), task_id, cfg["task_deadline_seconds"])
    policy = make_policy(bundle.template, method, device)
    trainer = PPO(policy)
    collector = Collector(bundle, policy)
    sampler = None if policy_method(method) in EMPTY_PRIOR_METHODS else PriorSampler(0)
    eval_rows, train_rows, resume_doc = [], [], {}
    if resume and (job_dir / "resume.json").is_file():
        resume_doc = json.loads((job_dir / "resume.json").read_text(encoding="utf-8"))
        ckpt_load = Path(resume_doc["checkpoint"])
        ckpt_s = str(ckpt_load)
        if "20260923T121938Z_79c690f9" in ckpt_s or "clock_integrity_repair" in ckpt_s:
            raise BindingError("refusing denylisted contaminated checkpoint %s" % ckpt_s)
        load_checkpoint(ckpt_load, policy, trainer.optimizer)
        rngp = json.loads(ckpt_load.with_suffix(".rng.json").read_text(encoding="utf-8"))
        s1.restore_rng({"python": rngp["python_rng"], "numpy": rngp["numpy_rng"], "torch": rngp["torch_rng"], "cuda": rngp["cuda_rng"]})
        episode_path = ckpt_load.with_suffix(".episode.pkl")
        if episode_path.is_file():
            import pickle
            with episode_path.open("rb") as f:
                ep = pickle.load(f)
            sim = bundle.environment.sim
            sim.data.qpos[:] = ep["qpos"]
            sim.data.qvel[:] = ep["qvel"]
            sim.data.ctrl[:] = ep["ctrl"]
            sim.data.time = float(ep["sim_time"])
            sim.forward()
            bundle.clock._origin = float(ep["clock_origin"])
            bundle.episode_start_seconds = float(ep["episode_start_seconds"])
            bundle._n = int(ep["case_scheduler_n"])
            bundle.snapshot_builder.episode_id = ep["snapshot_episode_id"]
            bundle.evaluator._rewarded = bool(ep["evaluator_rewarded"])
            bundle.evaluator._success_time = ep["evaluator_success_time"]
            bundle.evaluator.deadline = float(ep["evaluator_deadline"])
            bundle.original_prior_edges = ep["original_prior_edges"]
            bundle.current_snapshot = ep["current_snapshot"]
            collector.prefixes = ep["prefixes"]
            collector.weights = ep["weights"]
            collector.success_seen = ep["success_seen"]
            bundle._phase_a_case_id = ep.get("case_id")
            bundle._phase_a_prior = ep.get("prior")
            bundle._phase_a_source_n = ep.get("source_n")
            bundle._phase_a_cache_key = ep.get("cache_key")
            if sampler is not None and rngp.get("prior_sampler"):
                prior_state = rngp["prior_sampler"]
                import random
                streams = {}
                for k, state in (prior_state.get("rng") or {}).items():
                    stream = random.Random()
                    stream.setstate((state[0], tuple(state[1]), state[2]))
                    streams[k] = stream
                sampler.streams = streams
                from .prior import EpisodePrior
                sampler.active = {
                    k: EpisodePrior(
                        v["env_id"], v["episode_id"], tuple(tuple(e) for e in v["edges"]),
                        v["original_hash"], v["hash"], v["audit_mode"],
                    )
                    for k, v in (prior_state.get("active") or {}).items()
                }
                sampler.draw_count = int(prior_state.get("draw_count") or 0)
        episode_path = ckpt_load.with_suffix(".episode.pkl")
        if episode_path.is_file():
            import pickle
            with episode_path.open("rb") as f:
                ep = pickle.load(f)
            sim = bundle.environment.sim
            sim.data.qpos[:] = ep["qpos"]
            sim.data.qvel[:] = ep["qvel"]
            sim.data.ctrl[:] = ep["ctrl"]
            sim.data.time = float(ep["sim_time"])
            sim.forward()
            bundle.clock._origin = float(ep["clock_origin"])
            bundle.episode_start_seconds = float(ep["episode_start_seconds"])
            bundle._n = int(ep["case_scheduler_n"])
            bundle.snapshot_builder.episode_id = ep["snapshot_episode_id"]
            bundle.evaluator._rewarded = bool(ep["evaluator_rewarded"])
            bundle.evaluator._success_time = ep["evaluator_success_time"]
            bundle.evaluator.deadline = float(ep["evaluator_deadline"])
            bundle.original_prior_edges = ep["original_prior_edges"]
            bundle.current_snapshot = ep["current_snapshot"]
            collector.prefixes = ep["prefixes"]
            collector.weights = ep["weights"]
            collector.success_seen = ep["success_seen"]
            bundle._phase_a_case_id = ep.get("case_id")
            bundle._phase_a_prior = ep.get("prior")
            bundle._phase_a_source_n = ep.get("source_n")
            bundle._phase_a_cache_key = ep.get("cache_key")
            if sampler is not None and rngp.get("prior_sampler"):
                prior_state = rngp["prior_sampler"]
                import random
                streams = {}
                for k, state in (prior_state.get("rng") or {}).items():
                    stream = random.Random()
                    stream.setstate((state[0], tuple(state[1]), state[2]))
                    streams[k] = stream
                sampler.streams = streams
                sampler.active = {k: v for k, v in (prior_state.get("active") or {}).items()}
                sampler.draw_count = int(prior_state.get("draw_count") or 0)
        if (job_dir / "eval_metrics.csv").exists():
            with (job_dir / "eval_metrics.csv").open(encoding="utf-8") as f:
                eval_rows = list(csv.DictReader(f))
                for r in eval_rows:
                    for k in ("update", "skill_transitions", "success_n"):
                        if r.get(k) not in (None, ""):
                            r[k] = int(float(r[k]))
                    for k in ("success_rate", "mean_discounted_return"):
                        if r.get(k) not in (None, ""):
                            r[k] = float(r[k])
        if (job_dir / "train_metrics.csv").exists():
            with (job_dir / "train_metrics.csv").open(encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
    count = int(resume_doc.get("count") or 0)
    complete_updates = int(resume_doc.get("complete_updates") or 0)
    fragment_updates = int(resume_doc.get("fragment_updates") or 0)
    interaction_seconds = float(resume_doc.get("interaction_seconds") or 0.0)
    empty_episodes = 0
    train_success_episodes = int(resume_doc.get("train_success_episodes") or 0)
    zero_reward_episodes = int(resume_doc.get("zero_reward_episodes") or 0)
    original_ep = int(resume_doc.get("original_episode_n") or 0)
    absent_ep = int(resume_doc.get("absent_episode_n") or 0)
    nan_n = int(resume_doc.get("NaN_n") or 0)
    optimizer_steps = int(resume_doc.get("optimizer_steps") or 0)
    ep_reward = 0.0
    case = prior = source_n = cache_key_s = None
    if resume and bundle.current_snapshot is not None:
        case = getattr(bundle, "_phase_a_case_id", None)
        prior = getattr(bundle, "_phase_a_prior", None)
        source_n = getattr(bundle, "_phase_a_source_n", None)
        cache_key_s = getattr(bundle, "_phase_a_cache_key", None)
        if case is None or prior is None or source_n is None:
            raise BindingError("resume generation missing active episode identity")
    trans_buffer = []
    hard_fail = None
    first_update_ok = bool(resume_doc.get("first_update_ok"))
    done_ns = set(int(r.get("skill_transitions")) for r in eval_rows if r.get("skill_transitions") not in (None, ""))
    extra_base = {"method": mname, "task": task_id, "complete_updates": complete_updates, "fragment_updates": fragment_updates, **hashes_doc, "device": str(device), "H": prof["H"], "d_ref": cfg["d_ref"], "Tcap": cfg["Tcap"], "Ncap": N_CAP, "actor_episode_discount_weight": False, "planned_id": planned_id}
    log("%s %s training start dir=%s" % (task_id, mname, job_dir))
    rollout = Rollout()
    fp_init = param_fingerprint(policy)
    try:
        if 0 not in done_ns:
            extra0 = dict(extra_base)
            extra0.update({"update_count": 0, "interaction_count": 0, "interaction_seconds": 0.0, "N": 0, "T": 0.0})
            ckpt0 = job_dir / "checkpoints" / "n_000000.pt"
            save_ckpt(ckpt0, policy, trainer.optimizer, extra0, s1.capture_rng(), sampler, collector)
            ev0 = eval_episodes(root, task_id, method, ckpt0, eval_cases, split_index, device, hashes_doc, job_dir / "eval_n_000000.json", n_episodes=DEV_EPISODES, label="dev")
            eval_rows.append({"update": 0, "skill_transitions": 0, "success_rate": ev0["success_rate"], "success_n": ev0["success_n"], "mean_discounted_return": ev0["mean_discounted_return"], "checkpoint": str(ckpt0)})
            s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
            done_ns.add(0)

        def maybe_update():
            nonlocal complete_updates, fragment_updates, nan_n, trans_buffer, optimizer_steps, first_update_ok
            extra_base["complete_updates"] = complete_updates
            extra_base["fragment_updates"] = fragment_updates
            hit_cap = count >= N_CAP or interaction_seconds >= cfg["Tcap"]
            n_roll = len(rollout.transitions)
            if n_roll >= ROLLOUT_N or (hit_cap and n_roll > 0):
                complete = n_roll >= ROLLOUT_N
                use_n = ROLLOUT_N if complete else n_roll
                extra_ts = list(rollout.transitions[use_n:])
                extra_buf = list(trans_buffer[use_n:])
                rollout.transitions = list(rollout.transitions[:use_n])
                used_buf = list(trans_buffer[:use_n])
                kind = "complete" if complete else "fragment"
                fp_before = param_fingerprint(policy)
                adam_before = trainer.optimizer.state_dict()
                log("%s %s PPO %s update transitions=%s N=%s T=%s" % (task_id, mname, kind, use_n, count, interaction_seconds))
                ppo_wall_start = time.perf_counter()
                logs = trainer.update(rollout)
                collector._phase_a_ppo_seconds = float(getattr(collector, "_phase_a_ppo_seconds", 0.0)) + (time.perf_counter() - ppo_wall_start)
                optimizer_steps += len(logs)
                if complete:
                    complete_updates += 1
                else:
                    fragment_updates += 1
                if complete and complete_updates == 1 and not first_update_ok:
                    fp_after = param_fingerprint(policy)
                    audit = clock_audit_transitions(used_buf)
                    check = {
                        "n_transitions": use_n, "required": ROLLOUT_N, "optimizer_steps_this_update": len(logs),
                        "optimizer_steps_total": optimizer_steps, "param_changed": fp_before != fp_after,
                        "init_param_changed": fp_init != fp_after, "losses": logs[:2],
                        "clock_audit": audit,
                        "runtime_revision": RUNTIME_REVISION,
                        "note": "first complete update is a real PPO update on this run, not an extra qualification job",
                    }
                    if use_n != ROLLOUT_N:
                        raise DataIntegrityError("first complete update did not use 1024 transitions")
                    if not logs:
                        raise DataIntegrityError("first update produced no optimizer steps")
                    if not (fp_before != fp_after):
                        raise DataIntegrityError("first update did not change parameters")
                    if audit["N_clock_invalid"] or audit["N_unknown"] or audit["technical_n"] or audit["mask_mismatch_n"] or audit["N_clock_verified_valid"] != use_n:
                        write_json(job_dir / "first_update_selfcheck.json", check)
                        raise DataIntegrityError("first update clock/mask audit failed")
                    write_json(job_dir / "first_update_selfcheck.json", check)
                    first_update_ok = True
                    log("%s %s first-update self-check PASS" % (task_id, mname))
                agg = aggregate_train_buffer(method, used_buf)
                tot = [row.get("total") for row in logs]
                grad = [row.get("grad_norm") for row in logs]
                if any((x is None) or (not math.isfinite(x)) for x in tot + grad):
                    nan_n += 1
                    raise DataIntegrityError("Nonfinite PPO log")
                row = {
                    "update": complete_updates, "fragment_updates": fragment_updates, "complete": complete,
                    "valid_transitions": count, "rollout_n": use_n, "optimizer_steps": optimizer_steps,
                    "success_episodes": train_success_episodes, "zero_reward_episodes": zero_reward_episodes,
                    "policy_loss": float(sum(r["actor"] for r in logs) / len(logs)),
                    "value_loss": float(sum(r["v"] for r in logs) / len(logs)),
                    "q_loss": float(sum(r["q"] for r in logs) / len(logs)),
                    "total_loss": float(sum(r["total"] for r in logs) / len(logs)),
                    "grad_norm": float(sum(r["grad_norm"] for r in logs) / len(logs)),
                    "NaN_n": nan_n, "original_episode_n": original_ep, "absent_episode_n": absent_ep,
                    "interaction_seconds": interaction_seconds, "H": suite_half_life(),
                    "actor_episode_discount_weight": False, **agg,
                }
                train_rows.append(row)
                s1.csv_write(job_dir / "train_metrics.csv", train_rows)
                extra = dict(extra_base)
                extra.update({"complete_updates": complete_updates, "fragment_updates": fragment_updates, "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds, "optimizer_steps": optimizer_steps})
                ckpt_u = job_dir / "checkpoints" / (("update_complete_%s.pt" % complete_updates) if complete else ("update_fragment_%s.pt" % fragment_updates))
                save_ckpt(ckpt_u, policy, trainer.optimizer, extra, s1.capture_rng(), sampler, collector)
                trans_buffer[:] = extra_buf
                for t2 in extra_ts:
                    rollout.append(t2)
                write_json(job_dir / "resume.json", {"complete_updates": complete_updates, "fragment_updates": fragment_updates, "count": count, "interaction_seconds": interaction_seconds, "train_success_episodes": train_success_episodes, "zero_reward_episodes": zero_reward_episodes, "original_episode_n": original_ep, "absent_episode_n": absent_ep, "NaN_n": nan_n, "optimizer_steps": optimizer_steps, "first_update_ok": first_update_ok, "checkpoint": str(ckpt_u)})
            extra_base["complete_updates"] = complete_updates
            extra_base["fragment_updates"] = fragment_updates
            maybe_eval_at_n(root, task_id, method, policy, trainer, collector, sampler, count, interaction_seconds, hashes_doc, job_dir, eval_cases, split_index, device, eval_rows, extra_base, done_ns)
            return count >= N_CAP or interaction_seconds >= cfg["Tcap"] or complete_updates >= int(max_updates)

        while complete_updates < int(max_updates) and count < N_CAP and interaction_seconds < cfg["Tcap"]:
            if bundle.current_snapshot is None:
                case, prior, source_n, cache_key_s = start_episode(root, task_id, method, bundle, cases, split_index, sampler, collector, job_dir)
                if prior.audit_mode == "original":
                    original_ep += 1
                else:
                    absent_ep += 1
                ep_reward = 0.0
            snap = bundle.current_snapshot
            try:
                tstep, result = collector.step(snap, deterministic=False)
            except DiagnosticAbort as exc:
                payload = getattr(exc, "payload", None) or {"message": str(exc)}
                rec = {
                    "event": "DIAGNOSTIC_ABORT",
                    "task": task_id,
                    "method": mname,
                    "case_id": case,
                    "complete_updates": complete_updates,
                    "fragment_updates": fragment_updates,
                    "count": count,
                    "interaction_seconds": interaction_seconds,
                    "payload": payload,
                    "note": "attempt stopped; not DEADLINE; incomplete rollout is not a complete update",
                }
                s1.append_jsonl(job_dir / "diagnostic_abort.jsonl", rec)
                write_json(job_dir / "diagnostic_abort.json", rec)
                write_json(job_dir / "job_summary.json", {
                    "task": task_id, "method": mname, "planned_id": planned_id, "run_id": run_id,
                    "hard_fail": True, "stop_reason": "DIAGNOSTIC_ABORT",
                    "complete_updates": complete_updates, "fragment_updates": fragment_updates,
                    "valid_transitions": count, "interaction_seconds": interaction_seconds,
                    "runtime_revision": RUNTIME_REVISION, "diagnostic": rec,
                })
                raise
            ended = False
            success = False
            if tstep is None:
                empty_episodes += 1
                reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
                s1.append_jsonl(job_dir / "decision_log.jsonl", {"method": mname, "no_transition": True, "reason": reason, "complete_updates": complete_updates, "case_id": case, "empty_streak": empty_episodes})
                ended = True
                if empty_episodes >= MAX_EMPTY:
                    raise BindingError("Runtime repeatedly exposes no executable skill")
            else:
                empty_episodes = 0
                rec = s1.compact_transition(mname, 0, case, tstep, result, collector.last_output, collector.last_execution, prior.audit_mode, prior.original_hash, source_n, cache_key_s, run_id)
                rec["H"] = suite_half_life()
                rec["actor_episode_discount_weight"] = False
                rec["runtime_revision"] = RUNTIME_REVISION
                raw_exec = getattr(bundle.executor, "last", None)
                if isinstance(raw_exec, dict):
                    rec["sim_duration"] = raw_exec.get("sim_duration")
                    rec["control_steps"] = raw_exec.get("steps")
                    rec["raw_sim_start"] = raw_exec.get("raw_sim_start")
                    rec["raw_sim_end"] = raw_exec.get("raw_sim_end")
                s1.append_jsonl(job_dir / "transition_log.jsonl", rec)
                s1.append_jsonl(job_dir / "decision_log.jsonl", {"decision_id": rec["decision_id"], "selected_candidate_id": rec["selected_candidate_id"], "mask_hash": rec["mask_hash"], "prior_mode": rec["prior_mode"], "H": rec["H"]})
                rollout.append(tstep)
                trans_buffer.append(rec)
                count += 1
                interaction_seconds += tstep.duration
                if not rec["logits_finite"] or not rec["value_finite"] or not math.isfinite(rec["old_logp"]):
                    nan_n += 1
                    raise DataIntegrityError("NaN/Inf in transition")
                bundle.current_snapshot = tstep.next_snapshot
                ended = tstep.terminated or tstep.truncated
                success = bool(result.success) if not isinstance(result, dict) else bool(result.get("success"))
                ep_reward += tstep.reward
            if ended:
                if success:
                    train_success_episodes += 1
                if ep_reward == 0:
                    zero_reward_episodes += 1
                s1.append_jsonl(job_dir / "episode_log.jsonl", {"method": mname, "task": task_id, "case_id": case, "success": success, "reward": ep_reward, "prior_mode": None if prior is None else prior.audit_mode, "transitions_so_far": count, "T": interaction_seconds})
                bundle.current_snapshot = None
            if maybe_update():
                break
        extra_final = dict(extra_base)
        extra_final.update({"complete_updates": complete_updates, "fragment_updates": fragment_updates, "interaction_count": count, "interaction_seconds": interaction_seconds, "N": count, "T": interaction_seconds, "final": True, "optimizer_steps": optimizer_steps})
        ckpt_final = job_dir / "checkpoints" / ("final_n_%06d_u%02d.pt" % (count, complete_updates))
        save_ckpt(ckpt_final, policy, trainer.optimizer, extra_final, s1.capture_rng(), sampler, collector)
        if count not in done_ns:
            evf = eval_episodes(root, task_id, method, ckpt_final, eval_cases, split_index, device, hashes_doc, job_dir / "eval_final.json", n_episodes=DEV_EPISODES, label="dev")
            eval_rows.append({"update": complete_updates, "skill_transitions": count, "success_rate": evf["success_rate"], "success_n": evf["success_n"], "mean_discounted_return": evf["mean_discounted_return"], "checkpoint": str(ckpt_final)})
            s1.csv_write(job_dir / "eval_metrics.csv", eval_rows)
        selected, sel_doc = select_checkpoint(eval_rows)
        write_json(job_dir / "checkpoint_selection.json", sel_doc)
        stop_reason = "Ncap" if count >= N_CAP else ("Tcap" if interaction_seconds >= cfg["Tcap"] else ("updates_cap" if complete_updates >= int(max_updates) else "unknown"))
        summary = {
            "task": task_id, "method": mname, "planned_id": planned_id, "run_id": run_id, "job_dir": str(job_dir),
            "complete_updates": complete_updates, "fragment_updates": fragment_updates, "valid_transitions": count,
            "interaction_seconds": interaction_seconds, "optimizer_steps": optimizer_steps,
            "train_success_episodes": train_success_episodes, "step0_dev_success": None if not eval_rows else eval_rows[0].get("success_rate"),
            "final_dev_success": None if not eval_rows else eval_rows[-1].get("success_rate"),
            "NaN_n": nan_n, "original_episode_n": original_ep, "absent_episode_n": absent_ep,
            "hard_fail": hard_fail, "selected_checkpoint": selected, "stop_reason": stop_reason,
            "H": suite_half_life(), "d_ref": cfg["d_ref"], "Tcap": cfg["Tcap"], "eval": eval_rows, "train": train_rows,
            "first_update_ok": first_update_ok,
            "runtime_revision": RUNTIME_REVISION,
        }
        tlog = job_dir / "transition_log.jsonl"
        n_logged = 0
        if tlog.is_file():
            n_logged = sum(1 for line in tlog.read_text(encoding="utf-8").splitlines() if line.strip())
        summary["N_logged"] = n_logged
        summary["N_used_by_PPO"] = int(count)
        summary["clock_audit_remainder"] = clock_audit_transitions(trans_buffer) if trans_buffer else {"N_clock_verified_valid": 0, "N_clock_invalid": 0, "N_unknown": 0}
        write_json(job_dir / "job_summary.json", summary)
        return summary
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass

def freeze_selection(root, stamp, configsha8):
    root = Path(root)
    selected = {}
    for task in TASKS:
        for method in METHODS:
            job_dir = root / STAGE_DIR / task / method / "seed_0" / ("%s_%s" % (stamp, configsha8))
            summary = json.loads((job_dir / "job_summary.json").read_text(encoding="utf-8"))
            ckpt = summary.get("selected_checkpoint")
            if not ckpt:
                raise BindingError("missing selected checkpoint for %s %s" % (task, method))
            path = Path(ckpt)
            selected["%s/%s" % (task, method)] = {
                "task": task, "method": method, "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path), "job_dir": str(job_dir.relative_to(root)).replace("\\", "/"),
                "N": summary.get("valid_transitions"), "T": summary.get("interaction_seconds"),
                "complete_updates": summary.get("complete_updates"), "fragment_updates": summary.get("fragment_updates"),
                "optimizer_steps": summary.get("optimizer_steps"), "stop_reason": summary.get("stop_reason"),
                "selection": json.loads((job_dir / "checkpoint_selection.json").read_text(encoding="utf-8")).get("selected"),
            }
    dest = root / STAGE_DIR / stamp / "selection_manifest.json"
    write_json(dest, {"frozen_before_any_test_model_result": True, "stamp": stamp, "configsha8": configsha8, "selected": selected, "n": len(selected)})
    if len(selected) != 8:
        raise BindingError("selection must freeze 8 checkpoints, got %s" % len(selected))
    return dest


def evaluate_final_tests(root, stamp, configsha8, gpu=0):
    import torch
    root = Path(root)
    sel = json.loads((root / STAGE_DIR / stamp / "selection_manifest.json").read_text(encoding="utf-8"))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        device = torch.device("cuda", int(gpu))
    else:
        device = torch.device("cpu")
    torch.set_grad_enabled(False)
    results = {}
    for key, info in sel["selected"].items():
        task, method = info["task"], info["method"]
        split = json.loads((root / ENABLED_SPLITS[task]).read_text(encoding="utf-8"))
        active = list(split.get("test") or [])
        if len(active) != ACTIVE_N:
            raise BindingError("%s active test is %s" % (task, len(active)))
        split_index = {r["case_id"]: r for r in split["train"] + split["dev"] + active}
        ckpt = root / info["path"]
        live = sha256_file(ckpt)
        if live != info["sha256"]:
            raise BindingError("checkpoint bytes changed after freeze: %s" % key)
        hashes_doc = source_hashes(root, task)
        out = root / STAGE_DIR / task / method / "seed_0" / ("%s_%s" % (stamp, configsha8)) / "evaluation" / "final_test_id"
        out.mkdir(parents=True, exist_ok=True)
        ev = eval_episodes(root, task, method, ckpt, [r["case_id"] for r in active], split_index, device, hashes_doc, out / "test_metrics.json", n_episodes=ACTIVE_N, label="test")
        s1.csv_write(out / "test_metrics.csv", ev["rows"])
        write_json(out / "test_summary.json", {"valid": ev["n"], "success": ev["success_n"], "mean_G": ev["mean_discounted_return"], "status": "PASS" if ev["n"] == ACTIVE_N else "INCOMPLETE"})
        results[key] = {"valid": ev["n"], "success": ev["success_n"], "mean_G": ev["mean_discounted_return"], "success_rate": ev["success_rate"]}
        log("final test %s valid=%s success=%s G=%s" % (key, ev["n"], ev["success_n"], ev["mean_discounted_return"]))
    write_json(root / STAGE_DIR / stamp / "final_test_summary.json", results)
    return results


def write_status(root, payload):
    write_json(root / STATUS_PATH, payload)


def write_stage_report(root, payload):
    lines = [
        "# Stage 2A — Core Ladder Pilot (Plan v1.1 / Method 2.1.1)",
        "",
        "Status: `%s`." % payload.get("status"),
        "",
        "This report is the new-profile ledger. It does not overwrite old `experiments/part_2_exploration/stage_2a/` and does not lift STOP_SUPERSEDED_BY_PLAN_V1_1.",
        "",
        "execution_scope: STAGE_2A_CORE_LADDER_PILOT",
        "planned_jobs: 8",
        "auto_advance: false",
        "single_seed_exploratory: true",
        "",
        "## Frozen profile",
        "",
        "- H = %s" % payload.get("H"),
        "- T_B d_ref/Tcap = %s / %s" % ((payload.get("d_ref") or {}).get("T_B"), (payload.get("Tcap") or {}).get("T_B")),
        "- T_C d_ref/Tcap = %s / %s" % ((payload.get("d_ref") or {}).get("T_C"), (payload.get("Tcap") or {}).get("T_C")),
        "- actor_episode_discount_weight = false",
        "- runner: `python -m cp_disr.cli stage-2a-v11-run`",
        "",
        "## Jobs",
        "",
    ]
    for job in (payload.get("jobs") or []):
        lines.extend([
            "### %s" % job.get("planned_id"),
            "",
            "- task/method: %s / %s" % (job.get("task"), job.get("method")),
            "- N/T: %s / %s" % (job.get("valid_transitions"), job.get("interaction_seconds")),
            "- complete/fragment updates: %s / %s" % (job.get("complete_updates"), job.get("fragment_updates")),
            "- optimizer_steps: %s" % job.get("optimizer_steps"),
            "- stop_reason: %s" % job.get("stop_reason"),
            "- selected_checkpoint: %s" % job.get("selected_checkpoint"),
            "- test: %s" % job.get("test"),
            "",
        ])
    lines.extend(["## Final tests", "", json.dumps(payload.get("final_test") or {}, indent=2, ensure_ascii=False), "", "## Cost", "", json.dumps(payload.get("cost") or {}, indent=2, ensure_ascii=False), "", "Next suggestion only: 3A-pilot T_C A_CAT seed0 after explicit authorization.", ""])
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text("\n".join(lines), encoding="utf-8")
    (root / STAGE_DIR / "results_draft_v0.md").write_text("\n".join([
        "# Stage 2A results draft v0",
        "",
        "single-seed exploratory. std across training seeds = NA.",
        "30 test episodes are not 30 training seeds.",
        "",
        json.dumps(payload.get("final_test") or {}, indent=2, ensure_ascii=False),
        "",
    ]), encoding="utf-8")


def cmd_stage_2a_v11_run(root, gpu=0, phase="all", task=None, method=None, stamp=None, configsha=None, resume=False, skip_startup_gates=False, max_updates=MAX_UPDATES):
    root = Path(root)
    os.chdir(root)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        device = torch.device("cuda", int(gpu))
        device_name = torch.cuda.get_device_name(int(gpu))
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    prof = bind_H(root)
    git_hash = git_commit(root)
    stamp = stamp or utc_stamp()
    configsha8 = configsha or frozen_configsha8(prof, git_hash)
    stage_out = root / STAGE_DIR / stamp
    stage_out.mkdir(parents=True, exist_ok=True)
    (root / STAGE_DIR / "CURRENT.json").write_text(json.dumps({"stamp": stamp, "configsha8": configsha8}) + "\n", encoding="utf-8")
    if phase in ("sync-1a", "all"):
        sync = sync_stage_1a_status(root)
        write_json(stage_out / "stage_1a_jobs_test_sync.json", sync)
        if phase == "sync-1a":
            return {"status": "SYNCED", "sync": sync}
    if phase in ("freeze", "all"):
        counts = freeze_enabled_splits(root)
        tests = register_test_ids(root)
        write_status(root, {"stage": "2A", "status": "RUNNING", "phase": "freeze", "started_at": utc_now(), "stamp": stamp, "configsha8": configsha8, "execution_scope": "STAGE_2A_CORE_LADDER_PILOT", "planned_jobs": 8, "auto_advance": False, "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"], "base_commit": BASE_COMMIT, "git_hash": git_hash, "old_stop_intact": (root / STOP_REL).is_file()})
        if phase == "freeze":
            return {"status": "FROZEN", "counts": counts, "tests": tests, "stamp": stamp, "configsha8": configsha8}
    if phase in ("materialize", "all"):
        audit = materialize_all(root, gpu=gpu)
        if phase == "materialize":
            return {"status": "MATERIALIZED", "audit": audit}
    tasks = [task] if task in TASKS else list(TASKS)
    methods = [method] if method in METHODS else list(METHODS)
    if phase in ("startup", "all") and not skip_startup_gates:
        all_gates = []
        for t in tasks:
            gates, blocked, prof = startup_hard_checks(root, stage_out, device, t)
            all_gates.extend(gates)
            if blocked:
                write_status(root, {"stage": "2A", "status": "BLOCKED", "phase": "startup", "stamp": stamp, "configsha8": configsha8, "gates": gates, "task": t})
                raise BindingError("startup hard checks failed for %s" % t)
        if phase == "startup":
            return {"status": "STARTUP_GATES_PASS", "stamp": stamp, "configsha8": configsha8}
    if phase in ("train", "all"):
        if task not in TASKS or method not in METHODS:
            raise BindingError("train phase requires --task and --method")
        mname = method_dir_name(method)
        planned_id = PLANNED[(task, mname)]
        job_dir = root / STAGE_DIR / task / mname / "seed_0" / ("%s_%s" % (stamp, configsha8))
        refuse_if_clock_stop(root, planned_id=planned_id, stamp=stamp, configsha8=configsha8, job_dir=job_dir, phase="train")
        hashes_doc = source_hashes(root, task)
        write_status(root, {"stage": "2A", "status": "RUNNING", "phase": "train", "task": task, "method": method, "stamp": stamp, "configsha8": configsha8, "started_sampling_at": utc_now()})
        summary = train_job(root, task, method, device, device_name, prof, hashes_doc, stamp, configsha8, max_updates=max_updates, resume=resume)
        write_json(stage_out / ("job_%s_%s.json" % (task, method_dir_name(method))), summary)
        return {"status": "JOB_COMPLETE" if not summary.get("hard_fail") else "NEEDS_RERUN", "job": summary, "stamp": stamp, "configsha8": configsha8}
    if phase == "select":
        refuse_if_clock_stop(root, stamp=stamp, configsha8=configsha8, phase="select")
        path = freeze_selection(root, stamp, configsha8)
        return {"status": "SELECTION_FROZEN", "path": str(path)}
    if phase == "eval":
        refuse_if_clock_stop(root, stamp=stamp, configsha8=configsha8, phase="eval")
        tests = evaluate_final_tests(root, stamp, configsha8, gpu=gpu)
        return {"status": "TEST_COMPLETE", "final_test": tests}
    if phase == "report":
        refuse_if_clock_stop(root, stamp=stamp, configsha8=configsha8, phase="report")
        jobs = []
        for t in TASKS:
            for m in METHODS:
                p = stage_out / ("job_%s_%s.json" % (t, m))
                job_dir = root / STAGE_DIR / t / m / "seed_0" / ("%s_%s" % (stamp, configsha8))
                src = p if p.is_file() else job_dir / "job_summary.json"
                rec = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {"task": t, "method": m, "missing": True}
                evp = job_dir / "evaluation" / "final_test_id" / "test_summary.json"
                rec["test"] = json.loads(evp.read_text(encoding="utf-8")) if evp.is_file() else "NOT_CREATED"
                jobs.append(rec)
        ft = {}
        ftp = root / STAGE_DIR / stamp / "final_test_summary.json"
        if ftp.is_file():
            ft = json.loads(ftp.read_text(encoding="utf-8"))
        complete_jobs = [j for j in jobs if not j.get("missing") and j.get("test") != "NOT_CREATED" and (j.get("test") or {}).get("status") == "PASS"]
        status = "PASS" if len(complete_jobs) == 8 else "RUNNING"
        payload = {
            "stage": "2A", "plan_version": "1.1", "method_version": "2.1.1", "document_version": "3.1",
            "status": status, "execution_scope": "STAGE_2A_CORE_LADDER_PILOT", "stamp": stamp, "configsha8": configsha8,
            "H": prof["H"], "d_ref": prof["d_ref"], "Tcap": prof["Tcap"], "jobs": jobs, "final_test": ft,
            "planned_jobs": 8, "completed_jobs": len(complete_jobs), "failed_jobs": 0,
            "training_complete": all(not j.get("missing") for j in jobs),
            "checkpoint_selection_complete": (root / STAGE_DIR / stamp / "selection_manifest.json").is_file(),
            "final_test_complete": len(ft) == 8, "auto_advance": False, "base_commit": BASE_COMMIT, "git_hash": git_hash,
            "old_stop_intact": (root / STOP_REL).is_file(), "next_stage": "wait for explicit 3A-pilot authorization",
            "cost": {"note": "see runs/stage_2a/cache ledgers and job summaries"},
        }
        write_status(root, payload)
        write_stage_report(root, payload)
        return payload
    return {"status": "OK", "stamp": stamp, "configsha8": configsha8, "phase": phase}
