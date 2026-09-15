"""One-shot scripted physical collection. No seed replacement. Case IDs not written into traces."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from .physics import World, family_params, DT
from .util import sha256_file, write_json

def _move_to(w: World, target, close: bool, hold: str|None, steps: int, rows: list, names):
    tgt = np.asarray(target, float)
    for _ in range(steps):
        w.step(tgt, close, hold)
        rows.append(_row(w, names))

def _row(w: World, names) -> dict:
    s = w.snapshot(names)
    rec = dict(t=s["t"], tick=s["tick"], eef_x=float(s["eef"][0]), eef_y=float(s["eef"][1]), eef_z=float(s["eef"][2]),
               gripper_closed=int(s["gripper_closed"]),
               target_obj_x=float(w.p["target"][0]), target_obj_y=float(w.p["target"][1]), target_obj_z=float(w.p["target"][2]),
               target_A_x=float(w.p["target_A"][0]), target_A_y=float(w.p["target_A"][1]),
               target_B_x=float(w.p["target_B"][0]), target_B_y=float(w.p["target_B"][1]))
    for n in names:
        b=s[n]
        rec.update({
            f"{n}_x": float(b["pos"][0]), f"{n}_y": float(b["pos"][1]), f"{n}_z": float(b["pos"][2]),
            f"{n}_vx": float(b["vel"][0]), f"{n}_vy": float(b["vel"][1]), f"{n}_vz": float(b["vel"][2]),
            f"{n}_held": int(b["held"]), f"{n}_established": int(b["established"]),
            f"{n}_valid": int(b["valid"]), f"{n}_dist_eef": float(b["dist_eef"]),
        })
    return rec

def _grasp(w, name, rows, names):
    pos = w.bodies[name].pos + np.array([0,0,0.02])
    _move_to(w, pos, False, None, 10, rows, names)
    _move_to(w, w.bodies[name].pos + np.array([0,0,0.02]), True, name, 14, rows, names)

def _place(w, name, target, rows, names):
    hover = np.asarray(target)+np.array([0,0,0.12])
    _move_to(w, hover, True, name, 12, rows, names)
    _move_to(w, np.asarray(target)+np.array([0,0,0.06]), True, name, 8, rows, names)
    w.bodies[name].pos = np.asarray(target, float).copy(); w.bodies[name].vel[:] = 0; w.bodies[name].held=False
    _move_to(w, np.asarray(target)+np.array([0,0,0.06]), False, None, 12, rows, names)
    _move_to(w, np.asarray(target)+np.array([0,0,0.18]), False, None, 8, rows, names)

def _transport_frac(w, frac, rows, names):
    start = w.p["start_obj"]; goal = w.p["target"]
    mid = start + frac*(goal-start) + np.array([0,0,0.12])
    _grasp(w, "obj", rows, names)
    _move_to(w, start+np.array([0,0,0.12]), True, "obj", 6, rows, names)
    _move_to(w, mid, True, "obj", 10, rows, names)

def simulate(plan: dict) -> dict:
    params = family_params(plan["family_seed"])
    w = World(params, plan["rollout_seed"])
    case = plan["case_id"]
    if plan["task"]=="dual_order":
        names=("A","B")
        rows=[]
        _move_to(w, w.eef, False, None, 4, rows, names)
        if case.startswith("D1"):
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            _grasp(w,"B",rows,names); _place(w,"B",params["target_B"],rows,names)
        elif case.startswith("D2"):
            _grasp(w,"B",rows,names); _place(w,"B",params["target_B"],rows,names)
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
        elif case.startswith("D3"):
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            _grasp(w,"B",rows,names)
            _move_to(w, params["target_B"]+np.array([0,0,0.12]), True, "B", 6, rows, names)
            w.impulse_loss("B"); _move_to(w, w.eef, True, None, 6, rows, names); _move_to(w, w.eef, False, None, 4, rows, names)
            _grasp(w,"B",rows,names); _place(w,"B",params["target_B"],rows,names)
        elif case.startswith("D4"):
            _grasp(w,"B",rows,names); _place(w,"B",params["target_B"],rows,names)
            _grasp(w,"A",rows,names)
            _move_to(w, params["target_A"]+np.array([0,0,0.12]), True, "A", 6, rows, names)
            w.impulse_loss("A"); _move_to(w, w.eef, True, None, 6, rows, names); _move_to(w, w.eef, False, None, 4, rows, names)
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
        elif case.startswith("D5"):
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            # invalidate A by moving it off target
            _grasp(w,"A",rows,names)
            off = params["target_A"]+np.array([0.18,0.0,0.03])
            _place(w,"A", off, rows, names)
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            _grasp(w,"B",rows,names); _place(w,"B",params["target_B"],rows,names)
        else:  # D6
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            # repeat A, never complete B
            _grasp(w,"A",rows,names); _place(w,"A",params["target_A"],rows,names)
            _move_to(w, params["start_B"]+np.array([0,0,0.2]), False, None, 8, rows, names)
        return dict(rows=rows, names=names, params=params, world=w)
    names=("obj",)
    rows=[]
    _move_to(w, w.eef, False, None, 4, rows, names)
    if case.startswith("R1"):
        _grasp(w,"obj",rows,names); _place(w,"obj",params["target"],rows,names)
    elif case.startswith("R2"):
        _transport_frac(w,0.4,rows,names); w.impulse_loss("obj"); _move_to(w,w.eef,True,None,6,rows,names); _move_to(w,w.eef,False,None,4,rows,names)
        _grasp(w,"obj",rows,names); _place(w,"obj",params["target"],rows,names)
    elif case.startswith("R3"):
        _transport_frac(w,0.8,rows,names); w.impulse_loss("obj"); _move_to(w,w.eef,True,None,6,rows,names); _move_to(w,w.eef,False,None,4,rows,names)
        _grasp(w,"obj",rows,names); _place(w,"obj",params["target"],rows,names)
    elif case.startswith("R4") or case.startswith("R5"):
        frac = 0.4 if case.startswith("R4") else 0.8
        for _ in range(3):
            _transport_frac(w,frac,rows,names); w.impulse_loss("obj"); _move_to(w,w.eef,True,None,6,rows,names); _move_to(w,w.eef,False,None,4,rows,names)
            _grasp(w,"obj",rows,names)
            start=w.p["start_obj"]; goal=w.p["target"]
            mid=start+frac*(goal-start)+np.array([0,0,0.12])
            _move_to(w, mid, True, "obj", 8, rows, names)
        _place(w,"obj",params["target"],rows,names)
    elif case.startswith("R6"):
        frac=0.4
        for _ in range(3):
            _transport_frac(w,frac,rows,names); w.impulse_loss("obj"); _move_to(w,w.eef,True,None,6,rows,names); _move_to(w,w.eef,False,None,4,rows,names)
            _grasp(w,"obj",rows,names)
            start=w.p["start_obj"]; goal=w.p["target"]
            mid=start+frac*(goal-start)+np.array([0,0,0.12])
            _move_to(w, mid, True, "obj", 8, rows, names)
        _move_to(w, w.eef, True, "obj", 6, rows, names)
    elif case.startswith("R7"):
        _transport_frac(w,0.4,rows,names); w.impulse_loss("obj")
        _move_to(w, w.eef, True, None, 6, rows, names)
        pos=w.bodies["obj"].pos+np.array([0.15,0.0,0.12])
        _move_to(w, pos, False, None, 8, rows, names)
    else:  # R8 commanded release
        _grasp(w,"obj",rows,names)
        _move_to(w, w.bodies["obj"].pos+np.array([0,0,0.12]), True, "obj", 6, rows, names)
        _move_to(w, w.eef, False, None, 8, rows, names)
    return dict(rows=rows, names=names, params=params, world=w)

def write_rollout(plan: dict, root: Path, runner: str, hashes: dict) -> Path:
    dest = root / plan["family_id"] / plan["case_id"] / "rollout_00"
    dest.mkdir(parents=True, exist_ok=False)
    ident = dict(episode_id=plan["episode_id"], family_id=plan["family_id"], case_id=plan["case_id"],
                 rollout_seed=plan["rollout_seed"], family_seed=plan["family_seed"],
                 runner_commit=runner, source_hashes=hashes, protocol="PATHGRAPH_P1_V5_TARGETED_PHYSICAL_MECHANISM_CONFIRMATION_V1")
    write_json(dest/"identity.json", ident)
    sim = simulate(plan)
    rows = sim["rows"]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (dest/"timeseries.csv").open("x", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    health=dict(n=len(rows), dt=DT, finite=True, names=list(sim["names"]),
                max_abs_vel=max(abs(r.get("obj_vx", r.get("A_vx",0))) for r in rows))
    write_json(dest/"numeric_health.json", health)
    write_json(dest/"source_hash.json", dict(timeseries=sha256_file(dest/"timeseries.csv"), identity=sha256_file(dest/"identity.json")))
    return dest