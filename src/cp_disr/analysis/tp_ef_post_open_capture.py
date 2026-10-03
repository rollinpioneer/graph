"""CP-DISR-TP-EF-POST-OPEN-CAPTURE-1: capture and fresh-process-verify two recoverable post-OPEN T_B snapshots.

Scope: 2 setup episodes (start_case -> exactly one OPEN(container) -> stop) and 2 fresh-process restore-only
validations. No candidate branch, no continuation skill, no provider, no RL/optimizer, no test.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cp_disr.analysis import tp_ef_protocol_review as pr

CARD = "CP-DISR-TP-EF-POST-OPEN-CAPTURE-1"
BASELINE = "407053b13"
USER, HOST = "xushijie2", "gpu03"
PYTHON = "/home/xushijie2/envs/lerobotpi0-xfs/bin/python"
CASES = ("T_B_dev_03", "T_B_dev_05")           # fixed; no substitution, no retry
SPLIT_REL = pr.SPLIT_REL
RUNTIME_MANIFEST_REL = "experiments/manifests/runtime_manifest_v211.yaml"
SEARCH = dict(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=4.199999999997672)
DEADLINE = 60.0
OPEN_ID = pr.IDS["open"]
FACT_REQUIRED = {"p:Open:container": "TRUE", "p:GripperEmpty": "TRUE", "p:OnTable:target": "TRUE", "p:OnTable:second_object": "TRUE"}
PICK_IDS = (pr.IDS["pt"], pr.IDS["ps"])
CAPS = {"setup_episodes": 2, "open_skill_calls": 2, "fresh_restore_validations": 2, "environment_constructions": 4, "start_case_calls": 4,
        "provider_requests": 0, "rl_transitions": 0, "optimizer_steps": 0, "test_episodes": 0, "continuation_skills": 0}
# tolerances frozen before the first environment construction
TOL = {"sim_arrays_abs": 1e-12, "rgb": "exact_uint8", "depth_abs": 1e-6, "proprio_abs": 1e-9, "clock_abs": 1e-12, "facts": "exact value+reason",
       "mask_goal_candidates": "exact", "rng": "exact sha256", "controller_gripper_named_paths": "exact"}
SKIP_KEYS = {"sim", "model", "viewer", "robots", "case", "task_manifest", "task_contracts", "reset_identity", "last_perception", "refresh_perception",
             "sim_state_initial", "task_id", "env", "_renderer", "controller_config", "mujoco_robots", "mujoco_objects", "mujoco_arena",
             "robot_model", "mount", "target", "second_object", "lid", "arena", "mujoco_model"}
STATE_ROOT_DEPTH = 4
MAX_ARRAY = 2_000_000
NAMED_GATING = ("goal_pos", "goal_ori", "new_update", "current_action", "torques")


# --------------------------------------------------------------------------------------- utilities
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arr_hash(a):
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(v):
    return sha_bytes(canon(v).encode())


# ------------------------------------------------------------------- generic python-state dump/load
def _is_num(v):
    return isinstance(v, (bool, int, float, np.generic)) and not isinstance(v, str)


def _scalar(v):
    if isinstance(v, np.generic):
        v = v.item()
    return v


def dump_state(root, skip=SKIP_KEYS, max_depth=STATE_ROOT_DEPTH):
    """Dump numeric/array/RNG state of an object graph as {path_tuple: (kind, value)}. Paths are tuples of ('a'|'k', name)."""
    out, seen = {}, set()

    def add(path, v):
        if isinstance(v, np.ndarray):
            if v.dtype.kind in "biufc" and v.size <= MAX_ARRAY:
                out[path] = ("nd", np.array(v, copy=True))
        elif isinstance(v, (bool, int, float, np.generic)):
            out[path] = ("num", _scalar(v))
        elif isinstance(v, str):
            out[path] = ("str", v)
        elif v is None:
            out[path] = ("none", None)
        elif isinstance(v, (list, tuple)) and len(v) <= 64 and all(_is_num(x) for x in v):
            out[path] = ("list" if isinstance(v, list) else "tuple", [_scalar(x) for x in v])
        elif isinstance(v, np.random.RandomState):
            s = v.get_state()
            out[path] = ("rs", (s[0], np.array(s[1], copy=True), int(s[2]), int(s[3]), float(s[4])))
        elif isinstance(v, np.random.Generator):
            out[path] = ("gen", json.dumps(v.bit_generator.state, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
        else:
            return False
        return True

    def walk(obj, path, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            items = [(("k", str(k)), v) for k, v in obj.items() if isinstance(k, (str, int))]
        elif hasattr(obj, "__dict__"):
            items = [(("a", k), v) for k, v in vars(obj).items() if k not in skip and not k.startswith("__")]
        elif hasattr(obj, "maxlen") and hasattr(obj, "__iter__"):
            return
        else:
            return
        for seg, v in items:
            p = path + (seg,)
            try:
                if isinstance(v, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.ModuleType, type)):
                    continue
                if add(p, v):
                    continue
                if isinstance(v, dict):
                    walk(v, p, depth + 1)
                elif hasattr(v, "__dict__") and type(v).__module__.split(".")[0] in {"robosuite", "cp_disr"} and not isinstance(v, type):
                    walk(v, p, depth + 1)
            except Exception:  # never let an exotic attribute abort the dump; the entry is simply not recorded
                continue

    walk(root, (), 0)
    return out


def _navigate(root, path):
    cur = root
    for kind, name in path[:-1]:
        cur = getattr(cur, name) if kind == "a" else cur[name if name in cur else (int(name) if name.lstrip("-").isdigit() and int(name) in cur else name)]
    return cur


def load_state(root, entries):
    """Apply a dump onto a freshly constructed object graph; returns the list of paths that could not be applied."""
    missing = []
    for path, (kind, val) in entries.items():
        try:
            parent = _navigate(root, path)
            seg_kind, name = path[-1]
            if kind == "nd":
                cur = getattr(parent, name) if seg_kind == "a" else parent[name]
                if isinstance(cur, np.ndarray) and cur.shape == val.shape and cur.dtype == val.dtype:
                    cur[...] = val
                    continue
                new = np.array(val, copy=True)
            elif kind in ("num", "str", "none"):
                new = val
            elif kind == "list":
                new = list(val)
            elif kind == "tuple":
                new = tuple(val)
            elif kind == "rs":
                cur = getattr(parent, name) if seg_kind == "a" else parent[name]
                cur.set_state((val[0], val[1], val[2], val[3], val[4]))
                continue
            elif kind == "gen":
                cur = getattr(parent, name) if seg_kind == "a" else parent[name]
                cur.bit_generator.state = json.loads(val)
                continue
            else:
                missing.append(path_str(path))
                continue
            if seg_kind == "a":
                setattr(parent, name, new)
            else:
                parent[name] = new
        except Exception:
            missing.append(path_str(path))
    return missing


def path_str(path):
    return "/".join(("." if k == "a" else "[]") + str(n) for k, n in path)


def serialize_state(entries):
    """Entries -> (json-able meta, dict of npz arrays). No pickle."""
    meta, arrays = [], {}
    for i, (path, (kind, val)) in enumerate(sorted(entries.items(), key=lambda kv: path_str(kv[0]))):
        rec = {"path": [list(s) for s in path], "kind": kind}
        if kind == "nd":
            arrays[f"s{i}"] = val
            rec["key"] = f"s{i}"
        elif kind == "rs":
            arrays[f"s{i}"] = val[1]
            rec["key"] = f"s{i}"
            rec["value"] = [val[0], val[2], val[3], val[4]]
        else:
            rec["value"] = val
        meta.append(rec)
    return meta, arrays


def deserialize_state(meta, npz):
    out = {}
    for rec in meta:
        path = tuple((s[0], s[1]) for s in rec["path"])
        kind = rec["kind"]
        if kind == "nd":
            out[path] = ("nd", np.array(npz[rec["key"]]))
        elif kind == "rs":
            v = rec["value"]
            out[path] = ("rs", (v[0], np.array(npz[rec["key"]]), v[1], v[2], v[3]))
        elif kind in ("list", "tuple"):
            out[path] = (kind, rec["value"])
        else:
            out[path] = (kind, rec.get("value"))
    return out


def state_hash(entries):
    parts = []
    for path in sorted(entries, key=path_str):
        kind, val = entries[path]
        if kind == "nd":
            parts.append((path_str(path), arr_hash(val)))
        elif kind == "rs":
            parts.append((path_str(path), [val[0], arr_hash(val[1]), val[2], val[3], val[4]]))
        else:
            parts.append((path_str(path), [kind, val]))
    return digest(parts)


def diff_states(a, b):
    diffs = []
    for p in sorted(set(a) | set(b), key=path_str):
        if p not in a or p not in b:
            diffs.append({"path": path_str(p), "issue": "only_in_" + ("saved" if p in a else "restored")})
            continue
        (ka, va), (kb, vb) = a[p], b[p]
        if ka != kb:
            diffs.append({"path": path_str(p), "issue": "kind"})
        elif ka == "nd":
            if va.shape != vb.shape or va.dtype != vb.dtype or not np.array_equal(va, vb, equal_nan=True):
                diffs.append({"path": path_str(p), "issue": "array"})
        elif ka == "rs":
            if not (va[0] == vb[0] and np.array_equal(va[1], vb[1]) and va[2:] == vb[2:]):
                diffs.append({"path": path_str(p), "issue": "rng"})
        elif va != vb and not (isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb)):
            diffs.append({"path": path_str(p), "issue": "value"})
    return diffs


# ---------------------------------------------------------------------------- simulator state capture
SIM_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "qacc", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat")


def capture_sim(env):
    d = env.sim.data
    arrays = {"time": np.array([float(d.time)], dtype="float64")}
    for f in SIM_FIELDS:
        try:
            arrays[f] = np.array(getattr(d, f), dtype="float64", copy=True)
        except Exception:
            arrays[f] = np.zeros(0)
    return arrays


def apply_sim(env, arrays):
    """Write the saved simulator bytes. The caller runs sim.forward() afterwards (after comparing the raw assignment)."""
    d = env.sim.data
    d.time = float(arrays["time"][0])
    for f in SIM_FIELDS:
        if f in ("qacc",):
            continue
        tgt = getattr(d, f)
        src = arrays[f]
        if getattr(tgt, "shape", None) == src.shape and src.size:
            tgt[...] = src


def rng_snapshot():
    s = np.random.get_state()
    py = random.getstate()
    return {"numpy_global": {"name": s[0], "key": np.array(s[1], copy=True), "pos": int(s[2]), "has_gauss": int(s[3]), "cached": float(s[4])},
            "python_random": {"version": py[0], "state": list(py[1]), "gauss": py[2]}}


def rng_hash(r):
    return digest({"np": [r["numpy_global"]["name"], arr_hash(r["numpy_global"]["key"]), r["numpy_global"]["pos"], r["numpy_global"]["has_gauss"], r["numpy_global"]["cached"]],
                   "py": [r["python_random"]["version"], r["python_random"]["state"], r["python_random"]["gauss"]]})


def apply_rng(r):
    g = r["numpy_global"]
    np.random.set_state((g["name"], np.array(g["key"], dtype="uint32"), g["pos"], g["has_gauss"], g["cached"]))
    p = r["python_random"]
    random.setstate((p["version"], tuple(p["state"]), p["gauss"]))


def facts_to_list(records):
    return [{"fact_id": r.fact_id, "value": r.value.value, "reason": r.reason, "last_confirmed_value": getattr(r.last_confirmed_value, "value", str(r.last_confirmed_value)),
             "evidence_ids": list(r.evidence_ids)} for r in sorted(records, key=lambda x: x.fact_id)]


def facts_comparable(lst):
    return [(f["fact_id"], f["value"], f["reason"], f["last_confirmed_value"], tuple(f["evidence_ids"])) for f in sorted(lst, key=lambda x: x["fact_id"])]


def prev_to_json(prev):
    return {k: getattr(v, "value", str(v)) for k, v in sorted(prev.items())}


def prev_from_json(d):
    from cp_disr.facts import Truth
    return {k: Truth(v) for k, v in d.items()}


def public_obs_arrays(env):
    o = env.public_observation()
    return {"rgb": np.array(o["rgb"], copy=True), "depth": np.array(o["depth"], copy=True), "proprio": np.array(o["proprio"], copy=True)}


def forced_obs_arrays(env):
    """Force a re-render from the current simulator state (validates that state, not cache, reproduces the image)."""
    env._get_observations(force_update=True)
    return public_obs_arrays(env)


def compare_obs(a, b):
    res = {"rgb_exact": bool(a["rgb"].shape == b["rgb"].shape and a["rgb"].dtype == b["rgb"].dtype and np.array_equal(a["rgb"], b["rgb"])),
           "depth_max_abs": float(np.max(np.abs(a["depth"].astype("float64") - b["depth"].astype("float64")))) if a["depth"].shape == b["depth"].shape else float("inf"),
           "proprio_max_abs": float(np.max(np.abs(a["proprio"] - b["proprio"]))) if a["proprio"].shape == b["proprio"].shape else float("inf")}
    res["pass"] = bool(res["rgb_exact"] and res["depth_max_abs"] <= TOL["depth_abs"] and res["proprio_max_abs"] <= TOL["proprio_abs"])
    return res


def compare_sim(a, b, skip=("qacc",)):
    worst, bad = 0.0, []
    for k in a:
        if k in skip:
            continue
        if a[k].shape != b[k].shape:
            bad.append(k)
            continue
        if a[k].size:
            m = float(np.max(np.abs(a[k] - b[k])))
            worst = max(worst, m)
            if m > TOL["sim_arrays_abs"]:
                bad.append(k)
    return {"max_abs": worst, "mismatched_fields": bad, "pass": not bad}


# -------------------------------------------------------------------------------------- runtime build
class Ledger:
    def __init__(self):
        self.make_env_calls = 0
        self.env_reset_calls = 0
        self.start_case_calls = 0
        self.skill_calls = 0
        self.bootstrap_env = None
        self.bootstrap_case = None
        self.reused = False

    def as_dict(self):
        return {"environment_constructions": self.make_env_calls, "env_reset_calls_total": self.env_reset_calls,
                "start_case_calls": self.start_case_calls, "skill_calls": self.skill_calls, "bootstrap_env_reused_for_start_case": self.reused}


def build_runtime(root, out, case_id, ledger, tag):
    """One real environment construction per process: the bootstrap env made by create_task_runtime is reused as the start_case env."""
    from cp_disr.baselines.b_plan import SearchConfig, bind_t_b_manifest, _make_relation_free_runtime_split
    from cp_disr.platforms.libero import runtime_factory as rf
    from cp_disr.runtime import load_runtime
    root, out = Path(root), Path(out)
    free = out / "bindings" / "T_B_relation_free_split.json"
    free.parent.mkdir(parents=True, exist_ok=True)
    if not free.is_file():
        _make_relation_free_runtime_split(root / SPLIT_REL, free)
    doc = json.loads(free.read_text())
    rows = {r["case_id"]: r for r in doc["dev"]}
    order = [case_id] + [c for c in CASES if c != case_id]
    derived = dict(doc)
    derived["train"] = [rows[c] for c in order]
    derived["dev"] = []
    derived["test"] = []
    split_path = out / "bindings" / f"split_first_{case_id}_{tag}.json"
    split_path.write_text(json.dumps(derived, sort_keys=True, indent=1))
    manifest, _ = bind_t_b_manifest(root, root / RUNTIME_MANIFEST_REL, split_path, SearchConfig(**SEARCH))
    manifest["runtime"]["task_deadlines"]["T_B"] = DEADLINE
    real_make = rf.make_env

    def counting_make(spec, gpu=0):
        if ledger.bootstrap_env is not None and not ledger.reused and spec.case_id == ledger.bootstrap_case:
            ledger.reused = True
            return ledger.bootstrap_env
        ledger.make_env_calls += 1
        env = real_make(spec, gpu=gpu)
        orig_reset = env.reset

        def counting_reset(*a, **k):
            ledger.env_reset_calls += 1
            return orig_reset(*a, **k)
        env.reset = counting_reset
        if ledger.bootstrap_env is None:
            ledger.bootstrap_env, ledger.bootstrap_case = env, spec.case_id
        return env

    rf.make_env = counting_make
    bundle = load_runtime(manifest)
    return bundle, manifest


def start_case_reusing_bootstrap(bundle, ledger, case_id):
    """start_case closes bundle.environment first; shield the bootstrap env, which start_case reuses via counting_make."""
    real_env = bundle.environment
    bundle.environment = SimpleNamespace(close=lambda: None)
    ledger.start_case_calls += 1
    snap = bundle.start_case(case_id)
    bundle.environment.task_id = bundle.task_id
    return snap, real_env


def runtime_identity(root):
    files = ["src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/d0_env.py", "src/cp_disr/platforms/libero/skill_executor.py",
             "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/perception.py", "src/cp_disr/platforms/libero/task_evaluator.py",
             "src/cp_disr/platforms/libero/safety.py", "src/cp_disr/platforms/libero/clock.py", "configs/runtime/stage_2a_contract_registry.yaml"]
    return {f: sha_file(Path(root) / f) for f in files}


def py_runtime_state(bundle):
    ex = bundle.executor.inner if hasattr(bundle.executor, "inner") else bundle.executor
    return {"verifier_prev": prev_to_json(bundle.verifier.prev), "executor_last_xyz": {k: [float(x) for x in v] for k, v in ex._last_xyz.items()},
            "observation_counter": int(bundle.observations._n), "clock_origin": float(bundle.clock._origin), "clock_now": float(bundle.clock.now_seconds()),
            "episode_start_seconds": float(bundle.episode_start_seconds), "env_last_perception": {k: [float(x) for x in np.asarray(v).reshape(-1)] for k, v in (bundle.environment.last_perception or {}).items()}}


def apply_py_runtime_state(bundle, st, verifier_prev_key="verifier_prev"):
    ex = bundle.executor.inner if hasattr(bundle.executor, "inner") else bundle.executor
    bundle.verifier.prev = prev_from_json(st[verifier_prev_key])
    ex._last_xyz = {k: np.array(v, dtype=float) for k, v in st["executor_last_xyz"].items()}
    bundle.observations._n = int(st["observation_counter"])
    bundle.clock._origin = float(st["clock_origin"])
    bundle.episode_start_seconds = float(st["episode_start_seconds"])
    bundle.environment.last_perception = {k: np.array(v, dtype=float) for k, v in st["env_last_perception"].items()}


# ------------------------------------------------------------------------------------------ setup
def setup_episode(root, out, case_id, snap_dir, ledger_path):
    """start_case -> exactly one OPEN(container) -> verification -> capture -> stop. Never executes another skill."""
    root, out, snap_dir = Path(root).resolve(), Path(out).resolve(), Path(snap_dir)
    ledger = Ledger()
    res = {"card": CARD, "case_id": case_id, "role": "setup", "status": "FAIL", "failures": [], "skill_calls": 0}
    bundle = real_env = None
    try:
        bundle, manifest = build_runtime(root, out, case_id, ledger, "setup")
        snap0, real_env = start_case_reusing_bootstrap(bundle, ledger, case_id)
        env = bundle.environment
        pre_ident = pr_identity(bundle, snap0)
        res["pre_open_identity"] = {k: pre_ident[k] for k in ("facts_hash", "candidate_hash", "qpos_hash", "restore_receipt_sha256")}
        res["pre_open_facts"] = {k: v.value for k, v in snap0.facts.values.items()}
        raw_count = ledger.skill_calls
        orig_exec = bundle.executor.execute

        def counting_execute(cid, timeout):
            ledger.skill_calls += 1
            if ledger.skill_calls > CAPS["open_skill_calls"] or cid != OPEN_ID:
                raise RuntimeError("PROTOCOL_VIOLATION: only OPEN(container) may be executed in a setup episode")
            return orig_exec(cid, timeout)
        bundle.executor.execute = counting_execute
        contract = next(c for c in snap0.template.contracts if c.id == OPEN_ID)
        ex = bundle.executor.execute(OPEN_ID, float(contract.timeout_seconds))
        raw = dict(bundle.executor.last)
        res["open_controller_exit"] = ex.controller_exit
        res["open_raw"] = {k: raw[k] for k in ("controller_exit", "sim_duration", "steps", "states", "raw_sim_start", "raw_sim_end") if k in raw}
        if ex.controller_exit != "NORMAL_TERMINATION":
            res["failures"].append("OPEN_NOT_NORMAL_TERMINATION")
        prev_before = prev_to_json(bundle.verifier.prev)
        counter_before = int(bundle.observations._n)
        obs_obj = bundle.observations.observe()
        measured = bundle.perception.infer(obs_obj)
        records = bundle.verifier.verify(measured, ex)
        snap1 = bundle.snapshot_builder.build(snap0, records, obs_obj, ex, float(bundle.clock.now_seconds()))
        facts = facts_to_list(records)
        fact_map = {f["fact_id"]: f["value"] for f in facts}
        for fid, want in FACT_REQUIRED.items():
            if fact_map.get(fid) != want:
                res["failures"].append(f"FACT_{fid}={fact_map.get(fid)}")
        idx = {c: i for i, c in enumerate(snap1.candidate_ids)}
        mask = {c: bool(snap1.mask[idx[c]]) for c in PICK_IDS}
        for c, m in mask.items():
            if not m:
                res["failures"].append("MASK_FALSE:" + c)
        res["facts_after_open"] = fact_map
        res["pick_masks"] = mask
        res["skill_calls"] = ledger.skill_calls
        res.update(ledger.as_dict())
        if res["failures"]:
            return res
        # ----- capture (after the verification pipeline, before any forced re-render)
        py_state = dump_state(env)
        for root_name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            for p, v in dump_state(obj, skip=SKIP_KEYS - {"robots"}).items():
                py_state[(("a", "@" + root_name),) + p] = v
        runtime_state = py_runtime_state(bundle)
        runtime_state["verifier_prev_before_verify"] = prev_before
        runtime_state["observation_counter_before_observe"] = counter_before
        sim = capture_sim(env)
        pub = public_obs_arrays(env)
        rng = rng_snapshot()
        # decision-time observation a restored branch will see: freshly forwarded state, forced re-render (state already captured above)
        env.sim.forward()
        forced = forced_obs_arrays(env)
        v2 = type(bundle.verifier)(env)
        v2.prev = prev_from_json(runtime_state["verifier_prev"])
        facts_forced = facts_to_list(v2.verify(bundle.perception.infer(env.public_observation()), None))
        snap_dir.mkdir(parents=True, exist_ok=True)
        meta, arrays = serialize_state(py_state)
        np.savez(snap_dir / "sim_state.npz", **sim)
        np.savez(snap_dir / "python_state.npz", **arrays)
        np.savez(snap_dir / "obs_public_cached.npz", **pub)
        np.savez(snap_dir / "obs_forced_render.npz", **forced)
        np.savez(snap_dir / "rng_numpy.npz", key=rng["numpy_global"]["key"])
        goal = [[g.fact_id, g.sign] for g in snap1.template.goals]
        manifest_doc = {
            "card": CARD, "case_id": case_id, "restore_seed": int(bundle.cases[case_id].seed), "episode": "post-OPEN setup (OPEN executed once, then stopped)",
            "python_state_meta": meta, "runtime_state": runtime_state, "facts": facts, "facts_forced_render": facts_forced, "goal": goal, "candidate_ids": list(snap1.candidate_ids),
            "candidate_mask": [bool(m) for m in snap1.mask], "rng": {"numpy_global": {k: v for k, v in rng["numpy_global"].items() if k != "key"}, "python_random": rng["python_random"]},
            "hashes": {"sim_state": digest({k: arr_hash(v) for k, v in sorted(sim.items())}), "python_state": state_hash(py_state),
                       "obs_public_cached": digest({k: arr_hash(v) for k, v in sorted(pub.items())}), "obs_forced": digest({k: arr_hash(v) for k, v in sorted(forced.items())}),
                       "rng": rng_hash(rng), "facts": digest(facts_comparable(facts)), "goal": digest(goal),
                       "candidates": digest([list(snap1.candidate_ids), [bool(m) for m in snap1.mask]])},
            "runtime_identity": runtime_identity(root),
            "controller_gripper_named": sorted(path_str(p) for p in py_state if p[-1][1] in NAMED_GATING and p[-1][0] == "a"),
            "clock": {"now": runtime_state["clock_now"], "origin": runtime_state["clock_origin"], "sim_time": float(sim["time"][0])}}
        manifest_doc["state_identity_sha256"] = digest({k: manifest_doc[k] for k in ("case_id", "restore_seed", "hashes", "runtime_identity", "goal")} | {"candidates": manifest_doc["candidate_ids"], "mask": manifest_doc["candidate_mask"]})
        (snap_dir / "manifest.json").write_text(json.dumps(manifest_doc, sort_keys=True, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        files = []
        for p in sorted(snap_dir.iterdir()):
            with open(p, "rb") as f:
                os.fsync(f.fileno())
            files.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha_file(p)})
        res["snapshot_files"] = files
        res["state_identity_sha256"] = manifest_doc["state_identity_sha256"]
        res["python_state_entries"] = len(py_state)
        res["status"] = "PASS"
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
            (real_env or ledger.bootstrap_env).close()
        except Exception:
            pass
        pr.write_json(out / f"setup_{case_id}.json", res)


def pr_identity(bundle, snap):
    from cp_disr.analysis import tp_ef_discovery as d
    ident = d.pre_state_identity(bundle, snap)
    return ident


# ------------------------------------------------------------------------------------------- restore
def restore_validation(root, out, case_id, snap_dir):
    """Fresh process: construct env, load snapshot bytes, compare. No skill is executed."""
    root, out, snap_dir = Path(root).resolve(), Path(out).resolve(), Path(snap_dir)
    ledger = Ledger()
    res = {"card": CARD, "case_id": case_id, "role": "fresh_process_restore", "pid": os.getpid(), "status": "FAIL", "failures": [], "checks": {}}
    bundle = None
    try:
        man = json.loads((snap_dir / "manifest.json").read_text())
        file_hash_check = {}
        for p in sorted(snap_dir.iterdir()):
            file_hash_check[p.name] = sha_file(p)
        res["snapshot_file_sha256_at_restore"] = file_hash_check
        sim_saved = {k: np.array(v) for k, v in np.load(snap_dir / "sim_state.npz").items()}
        py_npz = np.load(snap_dir / "python_state.npz")
        py_saved = deserialize_state(man["python_state_meta"], py_npz)
        pub_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_public_cached.npz").items()}
        forced_saved = {k: np.array(v) for k, v in np.load(snap_dir / "obs_forced_render.npz").items()}
        rng_key = np.array(np.load(snap_dir / "rng_numpy.npz")["key"])
        rng_saved = {"numpy_global": dict(man["rng"]["numpy_global"], key=rng_key), "python_random": man["rng"]["python_random"]}
        bundle, manifest = build_runtime(root, out, case_id, ledger, "restore")
        snap0, real_env = start_case_reusing_bootstrap(bundle, ledger, case_id)
        env = bundle.environment
        orig_exec = bundle.executor.execute

        def forbidden_execute(*a, **k):
            ledger.skill_calls += 1
            raise RuntimeError("PROTOCOL_VIOLATION: restore validation must not execute skills")
        bundle.executor.execute = forbidden_execute
        # ---- apply
        apply_sim(env, sim_saved)
        missing = []
        top = {p: v for p, v in py_saved.items() if not (p and p[0][1].startswith("@"))}
        missing += load_state(env, top)
        for root_name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            sub = {p[1:]: v for p, v in py_saved.items() if p and p[0][1] == "@" + root_name}
            missing += load_state(obj, sub)
        apply_rng(rng_saved)
        rt = man["runtime_state"]
        apply_py_runtime_state(bundle, dict(rt, verifier_prev=rt["verifier_prev_before_verify"]))
        res["state_paths_not_applied"] = missing
        # ---- compare simulator arrays (raw assignment incl. warm start, then after forward) and clock
        res["checks"]["sim_arrays_raw"] = compare_sim(sim_saved, capture_sim(env))
        env.sim.forward()
        sim_now = capture_sim(env)
        res["checks"]["sim_arrays"] = compare_sim(sim_saved, sim_now, skip=("qacc", "qacc_warmstart"))
        res["checks"]["sim_arrays"]["warmstart_after_forward_max_abs"] = float(np.max(np.abs(sim_saved["qacc_warmstart"] - sim_now["qacc_warmstart"]))) if sim_saved["qacc_warmstart"].size else 0.0
        clock_now = float(bundle.clock.now_seconds())
        res["checks"]["clock"] = {"saved_now": man["clock"]["now"], "restored_now": clock_now, "origin_saved": man["clock"]["origin"], "origin_restored": float(bundle.clock._origin),
                                  "sim_time_saved": man["clock"]["sim_time"], "sim_time_restored": float(env.sim.data.time)}
        res["checks"]["clock"]["pass"] = bool(abs(clock_now - man["clock"]["now"]) <= TOL["clock_abs"] and abs(float(env.sim.data.time) - man["clock"]["sim_time"]) <= TOL["clock_abs"]
                                              and abs(float(bundle.clock._origin) - man["clock"]["origin"]) <= TOL["clock_abs"])
        rng_now = rng_snapshot()
        res["checks"]["rng"] = {"saved": man["hashes"]["rng"], "restored": rng_hash(rng_now), "pass": rng_hash(rng_now) == man["hashes"]["rng"]}
        # ---- python/controller/gripper state
        py_now = dump_state(env)
        for root_name, obj in (("robot0", env.robots[0]), ("robot0.controller", env.robots[0].controller), ("robot0.gripper", env.robots[0].gripper)):
            for p, v in dump_state(obj, skip=SKIP_KEYS - {"robots"}).items():
                py_now[(("a", "@" + root_name),) + p] = v
        diffs = diff_states(py_saved, py_now)
        named = [d for d in diffs if d["path"].split("/")[-1].lstrip(".") in NAMED_GATING]
        named_present = [p for p in man["controller_gripper_named"]]
        res["checks"]["controller_gripper_state"] = {"entries_saved": len(py_saved), "diffs_total": len(diffs), "named_paths_saved": named_present,
                                                      "named_diffs": named, "first_diffs": diffs[:20], "pass": not named and bool(named_present)}
        # ---- observations (cached as restored) and facts
        pub_now = public_obs_arrays(env)
        res["checks"]["public_obs_cached"] = compare_obs(pub_saved, pub_now)
        obs_obj = bundle.observations.observe()
        measured = bundle.perception.infer(obs_obj)
        records = bundle.verifier.verify(measured, SimpleNamespace(execution_id="restore-check", controller_exit="NORMAL_TERMINATION", start_seconds=0.0, end_seconds=0.0, evidence_ids=()))
        facts_now = facts_to_list(records)
        res["checks"]["facts"] = {"pass": facts_comparable(facts_now) == facts_comparable(man["facts"]), "restored": {f["fact_id"]: f["value"] for f in facts_now}}
        snap1 = bundle.snapshot_builder.build(snap0, records, obs_obj, SimpleNamespace(execution_id="restore-check", controller_exit="NORMAL_TERMINATION", start_seconds=0.0, end_seconds=0.0, evidence_ids=()),
                                              float(bundle.clock.now_seconds()))
        goal_now = [[g.fact_id, g.sign] for g in snap1.template.goals]
        cand_now = [list(snap1.candidate_ids), [bool(m) for m in snap1.mask]]
        res["checks"]["goal_candidates_mask"] = {"goal_pass": goal_now == man["goal"], "candidates_pass": cand_now == [man["candidate_ids"], man["candidate_mask"]],
                                                 "pass": goal_now == man["goal"] and cand_now == [man["candidate_ids"], man["candidate_mask"]]}
        fm = {f["fact_id"]: f["value"] for f in facts_now}
        idx = {c: i for i, c in enumerate(snap1.candidate_ids)}
        res["checks"]["post_open_requirements"] = {"facts": {k: fm.get(k) for k in FACT_REQUIRED}, "pick_masks": {c: bool(snap1.mask[idx[c]]) for c in PICK_IDS}}
        res["checks"]["post_open_requirements"]["pass"] = bool(all(fm.get(k) == v for k, v in FACT_REQUIRED.items()) and all(res["checks"]["post_open_requirements"]["pick_masks"].values()))
        # ---- forced re-render from restored simulator state
        env.sim.forward()
        forced_now = forced_obs_arrays(env)
        res["checks"]["forced_render_obs"] = compare_obs(forced_saved, forced_now)
        v2 = type(bundle.verifier)(env)
        v2.prev = prev_from_json(rt["verifier_prev"])
        facts_forced_now = facts_to_list(v2.verify(bundle.perception.infer(env.public_observation()), None))
        res["checks"]["facts_forced_render"] = {"pass": facts_comparable(facts_forced_now) == facts_comparable(man["facts_forced_render"]),
                                                "restored": {f["fact_id"]: f["value"] for f in facts_forced_now},
                                                "saved": {f["fact_id"]: f["value"] for f in man["facts_forced_render"]}}
        res["checks"]["identity"] = {"saved_state_identity": man["state_identity_sha256"],
                                     "includes_episode_counter": False}
        res["skill_calls"] = ledger.skill_calls
        res["checks"]["no_skill_executed"] = {"pass": ledger.skill_calls == 0}
        gating = ["sim_arrays_raw", "sim_arrays", "clock", "rng", "facts_forced_render", "controller_gripper_state", "public_obs_cached", "facts", "goal_candidates_mask", "post_open_requirements", "forced_render_obs", "no_skill_executed"]
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
        pr.write_json(out / f"restore_{case_id}.json", res)


# ------------------------------------------------------------------------------------ orchestration
def gate0(root, out):
    from cp_disr.analysis import tp_ef_discovery as d
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    ident = {"user": os.environ.get("USER", ""), "hostname": socket.gethostname(), "python": sys.executable, "platform": platform.platform()}
    if ident["user"] != USER or not ident["hostname"].startswith(HOST):
        problems.append("WRONG_ACCOUNT_OR_HOST")
    if sys.executable != PYTHON:
        problems.append("WRONG_PYTHON")
    import cp_disr
    ident["cp_disr_file"] = str(Path(cp_disr.__file__).resolve())
    if not ident["cp_disr_file"].startswith(str(root)):
        problems.append("CP_DISR_IMPORT_OUTSIDE_WORKTREE")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True).stdout.strip()
    base_ok = subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE, "HEAD"], cwd=root).returncode == 0
    if not base_ok:
        problems.append("BASELINE_NOT_ANCESTOR")
    if dirty:
        problems.append("DIRTY_TRACKED_TREE")
    for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE"):
        if key in os.environ:
            problems.append("PROVIDER_CREDENTIAL_PRESENT:" + key)
    fs = subprocess.run(["df", "-T", "-B1", str(out)], capture_output=True, text=True).stdout.splitlines()[-1].split()
    storage = {"filesystem": fs[0], "type": fs[1], "free_bytes": int(fs[4]), "mount": fs[-1]}
    if storage["type"] != "xfs" or storage["free_bytes"] < 5 * 1024 ** 3:
        problems.append("STORAGE_UNHEALTHY")
    free, _ = d._gpu_pick(1)
    gpu = free[0] if free else None
    if gpu is None:
        problems.append("STOPPED_CAPTURE_GPU_UNAVAILABLE")
    pr.write_json(out / "source_identity.json", {"card": CARD, "execution_base_commit": commit, "baseline": BASELINE, "baseline_is_ancestor": base_ok,
                                                  "branch": subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True).stdout.strip(),
                                                  "dirty_tracked": dirty, "identity": ident, "storage": storage, "gpu": gpu, "runtime_identity": runtime_identity(root)})
    auth = {"card": CARD, "authorized_by": "user chat instruction approving CP-DISR-TP-EF-POST-OPEN-CAPTURE-1", "cases": list(CASES),
            "substitution_allowed": False, "retry_allowed": False, "caps": CAPS, "tolerances_frozen_before_first_env": TOL,
            "gpu": gpu, "forbidden": ["candidate branches", "continuation skills", "provider", "RL", "optimizer", "test"],
            "reuse_rule": "each process makes exactly one real environment construction (bootstrap env reused as the start_case env)",
            "stop_rule": "any setup or restore failure stops the card; no replacement case; no next physical card",
            "problems": problems, "status": "PASS" if not problems else "STOPPED_GATE0"}
    pr.write_json(out / "authorization.json", auth)
    pr.write_json(out / "budget_ledger.json", {"caps": CAPS, "used": {k: 0 for k in CAPS}, "processes": []})
    return auth


def _spawn(root, out, cmd, case, gpu, snap_dir):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for k in ("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_FILE", "MUJOCO_EGL_DEVICE_ID"):
        env.pop(k, None)
    log = Path(out) / "logs" / f"{cmd}_{case}.log"
    log.parent.mkdir(exist_ok=True)
    p = subprocess.run([sys.executable, str(Path(root) / "scripts/tp_ef_post_open_capture.py"), cmd, "--root", str(root), "--output", str(out), "--case", case,
                        "--snapshot-dir", str(snap_dir)], cwd=root, env=env, stdout=log.open("ab"), stderr=subprocess.STDOUT, timeout=1200)
    return p.returncode


def run_all(root, out, snap_root):
    root, out, snap_root = Path(root).resolve(), Path(out).resolve(), Path(snap_root)
    auth = pr.read_json(out / "authorization.json")
    if auth["status"] != "PASS":
        raise RuntimeError("STOPPED_GATE0")
    gpu = auth["gpu"]
    ledger = pr.read_json(out / "budget_ledger.json")
    sequence = []
    stopped = ""
    for case in CASES:
        snap_dir = snap_root / case
        for cmd in ("setup", "restore"):
            if ledger["used"]["start_case_calls"] >= CAPS["start_case_calls"]:
                stopped = "STOPPED_BUDGET"
                break
            code = _spawn(root, out, cmd, case, gpu, snap_dir)
            rf = out / f"{'setup' if cmd == 'setup' else 'restore'}_{case}.json"
            res = pr.read_json(rf) if rf.is_file() else {"status": "FAIL", "failures": ["NO_RESULT_FILE"]}
            ledger["used"]["environment_constructions"] += res.get("environment_constructions", 0)
            ledger["used"]["start_case_calls"] += res.get("start_case_calls", 0)
            ledger["used"]["open_skill_calls"] += res.get("skill_calls", 0) if cmd == "setup" else 0
            ledger["used"]["continuation_skills"] += res.get("skill_calls", 0) if cmd == "restore" else 0
            ledger["used"]["setup_episodes"] += 1 if cmd == "setup" else 0
            ledger["used"]["fresh_restore_validations"] += 1 if cmd == "restore" else 0
            ledger.setdefault("env_reset_calls_total", 0)
            ledger["env_reset_calls_total"] += res.get("env_reset_calls_total", 0)
            ledger["processes"].append({"cmd": cmd, "case": case, "exit": code, "status": res["status"], "constructions": res.get("environment_constructions"),
                                        "start_case": res.get("start_case_calls"), "env_reset_calls": res.get("env_reset_calls_total"), "skills": res.get("skill_calls")})
            pr.write_json(out / "budget_ledger.json", ledger)
            sequence.append({"cmd": cmd, "case": case, "status": res["status"]})
            if res["status"] != "PASS":
                stopped = f"STOPPED_{cmd.upper()}_FAIL:{case}"
                break
        if stopped:
            break
    over = [k for k, v in CAPS.items() if ledger["used"][k] > v]
    return {"sequence": sequence, "stopped": stopped, "caps_exceeded": over}


# --------------------------------------------------------------------------------------- finalization
def protected_hashes(root):
    root = Path(root)
    out = {}
    for sub in ("runs/final_master/S4/tp_ef_discovery", "runs/final_master/S4/tp_ef_protocol_review", "runs/final_master/S4/tp_ef_post_open_audit"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(root))] = sha_file(p)
    return out


def finalize(root, out, snap_root, run_summary):
    root, out, snap_root = Path(root).resolve(), Path(out).resolve(), Path(snap_root)
    setups = {c: pr.read_json(out / f"setup_{c}.json") for c in CASES if (out / f"setup_{c}.json").is_file()}
    restores = {c: pr.read_json(out / f"restore_{c}.json") for c in CASES if (out / f"restore_{c}.json").is_file()}
    ledger = pr.read_json(out / "budget_ledger.json")
    pr.write_json(out / "setup_episode_results.json", {"card": CARD, "results": setups, "cases_attempted": sorted(setups), "cases_not_attempted": [c for c in CASES if c not in setups]})
    pr.write_json(out / "fresh_process_restore_results.json", {"card": CARD, "results": restores, "tolerances": TOL})
    manifest = {}
    hashes = {}
    fs = subprocess.run(["df", "-T", "-B1", str(snap_root if snap_root.exists() else snap_root.parent)], capture_output=True, text=True).stdout.splitlines()[-1].split()
    for c, s in setups.items():
        if s.get("status") != "PASS":
            continue
        d = snap_root / c
        man = pr.read_json(d / "manifest.json")
        manifest[c] = {"snapshot_dir": str(d), "state_identity_sha256": man["state_identity_sha256"], "hashes": man["hashes"], "runtime_identity": man["runtime_identity"],
                       "restore_seed": man["restore_seed"], "candidate_ids": man["candidate_ids"], "candidate_mask": man["candidate_mask"], "goal": man["goal"],
                       "clock": man["clock"], "facts": {f["fact_id"]: f["value"] for f in man["facts"]}, "python_state_entries": s.get("python_state_entries"),
                       "controller_gripper_named_paths": man["controller_gripper_named"], "capture_restore_status": restores.get(c, {}).get("status", "NOT_RUN")}
        r = restores.get(c, {})
        now = r.get("snapshot_file_sha256_at_restore", {})
        hashes[c] = [{"path": f["path"], "bytes": f["bytes"], "sha256_at_capture": f["sha256"], "sha256_at_fresh_restore": now.get(Path(f["path"]).name),
                      "unchanged": now.get(Path(f["path"]).name) == f["sha256"], "sha256_now": sha_file(f["path"]) if Path(f["path"]).is_file() else None} for f in s["snapshot_files"]]
    pr.write_json(out / "post_open_snapshot_manifest.json", {"card": CARD, "snapshots": manifest, "not_stored_in_git": "raw snapshot bytes live under the paths listed (outside the repository)"})
    pr.write_json(out / "snapshot_file_hashes.json", {"card": CARD, "files": hashes, "storage": {"filesystem": fs[0], "type": fs[1], "free_bytes": int(fs[4]), "mount": fs[-1]}})
    audit = {}
    for c in CASES:
        s, r = setups.get(c, {}), restores.get(c, {})
        audit[c] = {"setup_status": s.get("status", "NOT_ATTEMPTED"), "open_controller_exit": s.get("open_controller_exit"), "facts_after_open_at_capture": s.get("facts_after_open"),
                    "pick_masks_at_capture": s.get("pick_masks"), "failures_at_capture": s.get("failures"),
                    "restore_status": r.get("status", "NOT_RUN"), "post_open_requirements_after_restore": r.get("checks", {}).get("post_open_requirements"),
                    "facts_match_after_restore": r.get("checks", {}).get("facts", {}).get("pass"), "goal_candidates_mask_match": r.get("checks", {}).get("goal_candidates_mask", {}).get("pass"),
                    "hidden_truth_used_to_fill_unknown": False, "controller_exit_used_in_place_of_facts": False}
    pr.write_json(out / "post_open_fact_and_mask_audit.json", {"card": CARD, "cases": audit})
    ok = (not run_summary["stopped"] and not run_summary["caps_exceeded"] and all(setups.get(c, {}).get("status") == "PASS" and restores.get(c, {}).get("status") == "PASS" for c in CASES))
    (out / "next_physical_card_request.md").write_text(request_md(ok, setups, restores, manifest, ledger, run_summary), encoding="utf-8")
    (out / "final_summary.md").write_text(summary_md(ok, setups, restores, ledger, run_summary), encoding="utf-8")
    return {"all_pass": ok, "run": run_summary}


def request_md(ok, setups, restores, manifest, ledger, run):
    if not ok:
        eng = [f"{k}:{v['case_id']}" for k, v in list(((f"setup", s) for s in setups.values())) + list((("restore", r) for r in restores.values())) if v.get("status") == "STOPPED_ENGINEERING"]
        note = (f"Engineering stop recorded for {eng}: an exception in the capture/restore code, not a physical mismatch. This is not evidence against T_B; "
                "it needs a user decision on a corrected re-run budget." if eng else "Physical or protocol mismatch recorded (see fresh_process_restore_results.json).")
        return "\n".join([f"# Next physical card request — {CARD}", "", "**No physical card is requested.** Capture or fresh-process restore did not pass for both fixed cases.", "", note, "",
                          f"Stop: `{run['stopped'] or 'incomplete'}`; caps exceeded: {run['caps_exceeded'] or 'none'}.", "",
                          "Per the approval terms: stop mining T_B as the T_P carrier, go to the new-task-structure research decision; do not substitute T_B_dev_18, widen the case pool, or adjust contracts.", ""])
    sid = {c: manifest[c]["state_identity_sha256"][:16] for c in CASES}
    return "\n".join([f"# Next physical card request — {CARD}", "",
        "Both snapshots passed capture and fresh-process restore. Requested next card: **8 fixed-script physical branches** = 2 states x 2 candidates x 2 repeats. Not started by this card.", "",
        "## Frozen candidate protocol (no stepwise B_PLAN replanning)", "",
        "- States (fixed): " + ", ".join(f"{c} (identity {sid[c]})" for c in CASES) + ". Each branch restores the saved bytes in a fresh process and never replays OPEN.",
        "- Candidate T: PICK(target). Continuation T: PLACE(target, container) -> PICK(second_object) -> PLACE_BUFFER(second_object, buffer).",
        "- Candidate S: PICK(second_object). Continuation S: PLACE_BUFFER(second_object, buffer) -> PICK(target) -> PLACE(target, container).",
        "- Repeats: 2 per (state, candidate), each in its own fresh process from the same snapshot bytes. Branch order and ids pre-registered before the first branch.", "",
        "## To freeze before the first branch of that card", "",
        "- epsilon rule: max(floor, within-candidate repeat range) with floors dG = 0.02 and dT = 2.1 s, computed from repeats before the cross-candidate comparison (same as DISCOVERY-1).",
        "- Return: G = 2^(-tau_success / H), H = 23.1 s, deadline 60 s measured on the restored episode clock. Reliable-difference reasons: feasibility, return, time, skill count.",
        "- Stop rules: any restore mismatch, any candidate-skill postcondition failure, or any engineering exception stops the card; no substitute state, no retry. All 8 branches stay in the denominator.",
        "- Logging: facts, mask and controller exit after every skill (the DISCOVERY-1 gap), plus the planner-free script trace.",
        "- Budget: 8 branches, 8 fresh-process constructions, <= 7 scripted skills per branch (candidate + 3 continuation = 4 skills nominal), provider 0, RL/optimizer 0, test 0.", "",
        "## Evidence this card provides", "",
        "- Setup used " + str(ledger["used"]["open_skill_calls"]) + " OPEN skill calls, " + str(ledger["used"]["environment_constructions"]) + " environment constructions, " + str(ledger["used"]["start_case_calls"]) + " start_case calls; continuation skills " + str(ledger["used"]["continuation_skills"]) + ".",
        "- Provider is requested only after the 8-branch card yields two reliable utility witnesses.", ""])


def summary_md(ok, setups, restores, ledger, run):
    lines = [f"# {CARD} — final summary", "", f"Outcome: **{'BOTH SNAPSHOTS CAPTURED AND RESTORED' if ok else 'STOPPED / FAIL'}**  (stop reason: {run['stopped'] or 'none'}; caps exceeded: {run['caps_exceeded'] or 'none'})", ""]
    for c in CASES:
        s, r = setups.get(c), restores.get(c)
        lines.append(f"## {c}")
        lines.append(f"- setup: {s['status'] if s else 'NOT_ATTEMPTED'}" + (f"; OPEN exit {s.get('open_controller_exit')}; facts {s.get('facts_after_open')}; masks {s.get('pick_masks')}; failures {s.get('failures')}" if s else ""))
        if r:
            lines.append(f"- fresh-process restore: {r['status']}; failures {r['failures']}")
            for k, v in r.get("checks", {}).items():
                if isinstance(v, dict):
                    lines.append(f"  - {k}: pass={v.get('pass')}")
        lines.append("")
    lines += ["## Budget (as used)", "", f"- used: {ledger['used']}", f"- low-level env.reset() invocations including create_task_runtime bootstrap resets: {ledger.get('env_reset_calls_total')} "
              "(start_case-level resets equal the start_case count; each process makes one real construction and reuses it for start_case)", f"- processes: {ledger['processes']}", ""]
    return "\n".join(lines)


def verify(root, out, snap_root, before):
    root, out, snap_root = Path(root).resolve(), Path(out).resolve(), Path(snap_root)
    need = ["authorization.json", "budget_ledger.json", "setup_episode_results.json", "post_open_snapshot_manifest.json", "snapshot_file_hashes.json", "fresh_process_restore_results.json",
            "post_open_fact_and_mask_audit.json", "source_identity.json", "next_physical_card_request.md", "final_summary.md"]
    ledger = pr.read_json(out / "budget_ledger.json")
    hashes = pr.read_json(out / "snapshot_file_hashes.json")["files"]
    rehash = all(Path(f["path"]).is_file() and sha_file(f["path"]) == f["sha256_at_capture"] for fl in hashes.values() for f in fl)
    secrets = sum(1 for p in out.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".log", ".jsonl"} and any(x in p.read_text(errors="ignore") for x in ("sk-", "Authorization", "DASHSCOPE_API_KEY=")))
    checks = {"outputs_present": {n: (out / n).is_file() for n in need}, "caps_respected": all(ledger["used"][k] <= v for k, v in CAPS.items()),
              "zero_provider_rl_optimizer_test": all(ledger["used"][k] == 0 for k in ("provider_requests", "rl_transitions", "optimizer_steps", "test_episodes")),
              "no_continuation_skills": ledger["used"]["continuation_skills"] == 0, "snapshot_files_unchanged_since_capture": rehash,
              "prior_cards_unchanged": before == protected_hashes(root), "no_secret_shaped_content": secrets == 0}
    checks["status"] = "PASS" if all(v is True for k, v in checks.items() if k != "outputs_present") and all(checks["outputs_present"].values()) else "FAIL"
    pr.write_json(out / "verify.json", checks)
    return checks
