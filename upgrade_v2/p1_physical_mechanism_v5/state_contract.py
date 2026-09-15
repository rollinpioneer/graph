"""DELTA state events from physical timeseries. Case IDs never enter the kernel."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import numpy as np
from .util import sha256_file

TRI=("hold_current","hold_loss_confirmed","release_intent","a_current_valid","b_current_valid",
     "goal_verified","terminal_failure","recovery_completed")

def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _f(row,k,default=0.0):
    return float(row[k]) if k in row and row[k]!="" else default

def _i(row,k):
    return int(float(row[k])) if k in row and row[k]!="" else 0

def events_from_rollout(dest: Path) -> list[dict[str, Any]]:
    ident=json_load(dest/"identity.json")
    rows=_read(dest/"timeseries.csv")
    sha=sha256_file(dest/"timeseries.csv")
    episode=ident["episode_id"]
    names=["A","B"] if "A_x" in rows[0] else ["obj"]
    events=[]
    established={n:False for n in names}
    prev_hold={n:False for n in names}
    loss_i=0; loss_id=None; attempt="0"
    prev={k:None for k in TRI}
    rec_started=False
    for i,row in enumerate(rows):
        ns=int(round(float(row["t"])*1e9))
        close=bool(_i(row,"gripper_closed"))
        facts={k:False for k in TRI}
        facts["release_intent"]=not close
        if names==["obj"]:
            hold=bool(_i(row,"obj_held"))
            if hold: established["obj"]=True
            facts["hold_current"]=hold
            if established["obj"] and prev_hold["obj"] and not hold and close:
                facts["hold_loss_confirmed"]=True
                loss_i+=1; loss_id=f"{episode}:loss:{loss_i}"; attempt=str(loss_i)
            facts["a_current_valid"]=hold or established["obj"]
            facts["b_current_valid"]=True  # single-object task: B unused structural true? NO - leave false
            facts["b_current_valid"]=False
            # goal: near target and not held. target inferred from last positions when placed
            tx=_f(row,"target_obj_x", 0.62); ty=_f(row,"target_obj_y", 0.0)
            dist=(( _f(row,"obj_x")-tx)**2+(_f(row,"obj_y")-ty)**2)**0.5
            facts["goal_verified"]= (not hold) and established["obj"] and dist<=0.08 and _f(row,"obj_z")<=0.06 and abs(_f(row,"obj_vx"))<0.08
            if facts["hold_loss_confirmed"]:
                rec_started=False
            if (not hold) and loss_id and close and _f(row,"obj_dist_eef")<0.12:
                rec_started=True
            if rec_started and hold and loss_id:
                facts["recovery_completed"]=True
                rec_started=False
        else:
            # dual: hold if either held
            holdA=bool(_i(row,"A_held")); holdB=bool(_i(row,"B_held"))
            if holdA: established["A"]=True
            if holdB: established["B"]=True
            facts["hold_current"]=holdA or holdB
            facts["a_current_valid"]=bool(_i(row,"A_valid"))
            facts["b_current_valid"]=bool(_i(row,"B_valid"))
            facts["goal_verified"]=facts["a_current_valid"] and facts["b_current_valid"]
            for n,h,ph in (("A",holdA,prev_hold["A"]),("B",holdB,prev_hold["B"])):
                if established[n] and ph and not h and close:
                    facts["hold_loss_confirmed"]=True
                    loss_i+=1; loss_id=f"{episode}:loss:{loss_i}"; attempt=str(loss_i)
            if facts["hold_loss_confirmed"]:
                rec_started=False
            if loss_id and close and (holdA or holdB):
                facts["recovery_completed"]=True
        if names!=["obj"]:
            facts["terminal_failure"]= (i==len(rows)-1 and facts["a_current_valid"] and not facts["b_current_valid"]) or \
                                       (i==len(rows)-1 and facts["b_current_valid"] and not facts["a_current_valid"] and not facts["goal_verified"] and i>20 and not facts["a_current_valid"])
            # simpler: last frame and not both valid
            facts["terminal_failure"]= (i==len(rows)-1 and not (facts["a_current_valid"] and facts["b_current_valid"]))
        else:
            facts["terminal_failure"]= (i==len(rows)-1 and not facts["goal_verified"] and established["obj"] and not facts["hold_current"] and _f(row,"obj_x")<0.45)
        updates={k:facts[k] for k in TRI if facts[k]!=prev[k]}
        if "hold_current" not in updates:
            updates["hold_current"]=facts["hold_current"]
        ev=dict(episode_id=episode, object_id=("obj" if names==["obj"] else "dual"),
                goal_id=("place" if names==["obj"] else "dual_place"),
                attempt_id=attempt, event_id=f"{episode}:t{i}",
                available_at_ns=ns, physical_time_ns=ns, capture_order=i,
                updates=updates, field_observed_at_ns={k:ns for k in updates},
                field_known_at_ns={k:ns for k in updates},
                release_intent_at_loss_onset=(facts["release_intent"] if facts["hold_loss_confirmed"] else None),
                loss_event_id=loss_id, source_file=str(dest/"timeseries.csv"), source_row=i, source_sha256=sha,
                hold_current=facts["hold_current"], a_current_valid=facts["a_current_valid"],
                b_current_valid=facts["b_current_valid"], goal_verified=facts["goal_verified"],
                recovery_completed=facts["recovery_completed"],
                object_goal_distance=_goal_dist(row, names),
                raw=row)
        events.append(ev)
        prev=facts
        if names==["obj"]:
            prev_hold["obj"]=bool(_i(row,"obj_held"))
        else:
            prev_hold["A"]=bool(_i(row,"A_held")); prev_hold["B"]=bool(_i(row,"B_held"))
    return events

def json_load(p: Path):
    import json
    return json.loads(p.read_text(encoding="utf-8"))

def _goal_dist(row, names):
    if names==["obj"]:
        return float(np.hypot(_f(row,"obj_x")-0.62, _f(row,"obj_y")))
    return float(min(np.hypot(_f(row,"A_x")-0.55,_f(row,"A_y")-0.12), np.hypot(_f(row,"B_x")-0.55,_f(row,"B_y")+0.12)))