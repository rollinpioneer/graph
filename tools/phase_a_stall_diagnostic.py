#!/usr/bin/env python3
"""Bounded, QA-only Phase-A rollout-stall diagnostics.

This tool never constructs a Policy/PPO trainer and never updates parameters.  It
records incremental phase markers before and after each simulator-facing call so
an external watchdog can classify a native call that does not return.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import faulthandler
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Marker:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, phase: str, **payload):
        row = {"time": now(), "wall": time.monotonic(), "phase": phase, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def case_spec(row: dict):
    from cp_disr.platforms.libero.d0_env import CaseSpec

    fields = {f.name for f in dataclasses.fields(CaseSpec)}
    return CaseSpec(**{k: row[k] for k in fields if k in row})


def contact_snapshot(env) -> dict:
    sim = env.sim
    contacts = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = sim.model.geom_id2name(g1)
        n2 = sim.model.geom_id2name(g2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        contacts.append({
            "geom1": n1, "geom2": n2,
            "body1": sim.model.body_id2name(b1),
            "body2": sim.model.body_id2name(b2),
            "dist": float(c.dist),
        })
    return {
        "ncon": int(sim.data.ncon),
        "contacts": contacts,
        "qpos_finite": bool(np.isfinite(np.asarray(sim.data.qpos)).all()),
        "qvel_finite": bool(np.isfinite(np.asarray(sim.data.qvel)).all()),
        "sim_time": float(sim.data.time),
    }


def run_case(root: Path, row: dict, method: str, out: Path, gpu: int, max_decisions: int, wall_limit: float):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out.mkdir(parents=True, exist_ok=True)
    marker = Marker(out / "exact_phase_timeline.jsonl")
    timing_path = out / "per_call_timing.csv"
    timing_new = not timing_path.exists()
    timing = timing_path.open("a", newline="", encoding="utf-8")
    tw = csv.DictWriter(timing, fieldnames=["phase", "case_id", "index", "wall_start", "wall_end", "wall_seconds", "sim_start", "sim_end", "ncon", "status"])
    if timing_new:
        tw.writeheader(); timing.flush()
    case_id = row["case_id"]
    faulthandler.enable()
    stack_path = out / "python_stacks.log"
    sf = stack_path.open("a", encoding="utf-8")
    faulthandler.dump_traceback_later(20, repeat=True, file=sf)
    result = {"case_id": case_id, "method": method, "gpu": gpu, "status": "UNKNOWN", "decisions": 0, "steps": 0, "optimizer_steps": 0, "vlm_requests": 0, "test_id": False, "started_at": now()}
    env = None
    try:
        from cp_disr.platforms.libero.d0_env import D0ManipulationEnv
        marker.write("process_start", method=method, case_id=case_id, pid=os.getpid())
        t0 = time.monotonic(); marker.write("env_construct_start")
        env = D0ManipulationEnv(case_spec(row), render_gpu_device_id=int(gpu))
        marker.write("env_construct_end", wall_seconds=time.monotonic() - t0)
        t0 = time.monotonic(); marker.write("reset_start")
        env.reset()
        marker.write("reset_end", wall_seconds=time.monotonic() - t0, **contact_snapshot(env))
        t0 = time.monotonic(); marker.write("pose_apply_start")
        env._apply_case_poses()
        marker.write("pose_apply_end", wall_seconds=time.monotonic() - t0, **contact_snapshot(env))
        t0 = time.monotonic(); marker.write("sim_forward_start")
        env.sim.forward()
        marker.write("sim_forward_end", wall_seconds=time.monotonic() - t0, **contact_snapshot(env))
        # QA-only observation/render timing; these values are not fed to policy.
        t0 = time.monotonic(); marker.write("observation_start")
        obs = env.public_observation()
        marker.write("observation_end", wall_seconds=time.monotonic() - t0, rgb_shape=list(obs["rgb"].shape), depth_shape=list(obs["depth"].shape), sim_time=float(env.sim.data.time))
        marker.write("n_000000_checkpoint_observed", **contact_snapshot(env))
        start_wall = time.monotonic()
        for decision in range(max_decisions):
            if time.monotonic() - start_wall > wall_limit:
                result["status"] = "WATCHDOG_TIMEOUT"; marker.write("watchdog_timeout", decision=decision); break
            sim_start = float(env.sim.data.time); wall_start = time.monotonic()
            marker.write("decision_start", decision=decision, sim_start=sim_start)
            action = np.zeros(int(getattr(env, "action_dim", 7)), dtype=np.float32)
            marker.write("controller_start", decision=decision, action=action.tolist())
            step_status = "returned"
            try:
                ret = env.step(action)
            except Exception as exc:
                step_status = "exception"; marker.write("step_exception", decision=decision, error=repr(exc)); raise
            wall_end = time.monotonic(); sim_end = float(env.sim.data.time)
            snap = contact_snapshot(env)
            marker.write("controller_end", decision=decision, wall_seconds=wall_end-wall_start, sim_start=sim_start, sim_end=sim_end, **snap)
            tw.writerow({"phase":"env_step", "case_id":case_id, "index":decision, "wall_start":wall_start, "wall_end":wall_end, "wall_seconds":wall_end-wall_start, "sim_start":sim_start, "sim_end":sim_end, "ncon":snap["ncon"], "status":step_status}); timing.flush(); os.fsync(timing.fileno())
            result["decisions"] += 1; result["steps"] += 1
            marker.write("transition_observed", decision=decision, sim_start=sim_start, sim_end=sim_end, **snap)
        else:
            result["status"] = "COMPLETED_BOUND"
        if result["status"] == "UNKNOWN": result["status"] = "COMPLETED_BOUND"
    except Exception as exc:
        result["status"] = "EXCEPTION"; result["error"] = repr(exc); marker.write("diagnostic_exception", error=repr(exc))
    finally:
        faulthandler.cancel_dump_traceback_later()
        sf.flush(); sf.close(); timing.close()
        if env is not None:
            marker.write("close_start")
            try: env.close(); marker.write("close_end")
            except Exception as exc: marker.write("close_exception", error=repr(exc))
        result["finished_at"] = now(); (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return result


def geometry_audit(root: Path, out: Path):
    from cp_disr.platforms.libero.d0_env import OBJECT_HALF
    split_path = root / "configs/splits/T_C_stage_2a_v11.json"
    data = json.loads(split_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    with (out / "geometry_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["case_id","split","seed","target_x","target_y","interferer_x","interferer_y","dx","dy","abs_dx","abs_dy","aabb_overlap_x","aabb_overlap_y","aabb_overlap","classification","source_sha256"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for split in ("train", "dev"):
            for r in data[split]:
                dx=float(r["second_xy"][0]-r["target_xy"][0]); dy=float(r["second_xy"][1]-r["target_xy"][1])
                ox=max(0.0, 2*float(OBJECT_HALF[0])-abs(dx)); oy=max(0.0, 2*float(OBJECT_HALF[1])-abs(dy))
                cls="INITIAL_INTERPENETRATION" if ox>0 and oy>0 else ("VALID_TOUCHING_CONTACT" if ox==0 or oy==0 else "NO_INITIAL_CONTACT")
                w.writerow({"case_id":r["case_id"],"split":split,"seed":r["seed"],"target_x":r["target_xy"][0],"target_y":r["target_xy"][1],"interferer_x":r["second_xy"][0],"interferer_y":r["second_xy"][1],"dx":dx,"dy":dy,"abs_dx":abs(dx),"abs_dy":abs(dy),"aabb_overlap_x":ox,"aabb_overlap_y":oy,"aabb_overlap":bool(ox>0 and oy>0),"classification":cls,"source_sha256":sha256(split_path)})
    return {"object_half": list(map(float, OBJECT_HALF)), "source_split": str(split_path), "source_sha256": sha256(split_path), "train_count":len(data["train"]), "dev_count":len(data["dev"])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path.cwd()); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--mode", choices=["geometry","run"], required=True); ap.add_argument("--case-json", type=Path); ap.add_argument("--method", default="B2"); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--max-decisions", type=int, default=4); ap.add_argument("--wall-limit", type=float, default=600.0); args=ap.parse_args(); os.chdir(args.root)
    if args.mode=="geometry": print(json.dumps(geometry_audit(args.root,args.out), indent=2)); return
    if not args.case_json: raise SystemExit("--case-json required")
    row=json.loads(args.case_json.read_text())
    print(json.dumps(run_case(args.root,row,args.method,args.out,args.gpu,args.max_decisions,args.wall_limit), indent=2))


if __name__ == "__main__": main()
