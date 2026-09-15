from __future__ import annotations
from typing import Any

REC_NODES=["start","grasped","in_transit","placed","dropped_or_misaligned","recovery","success","terminal_failure"]
REC_IDX={n:i for i,n in enumerate(REC_NODES)}
REC_COST={"start":4/6,"grasped":3/6,"in_transit":2/6,"placed":1/6,"dropped_or_misaligned":5/6,
          "recovery":4/6,"success":0.0,"terminal_failure":1.0}
EDGES={("start","grasped"):("start_to_grasped","forward",0),
       ("grasped","in_transit"):("grasped_to_in_transit","forward",1),
       ("in_transit","placed"):("in_transit_to_placed","forward",2),
       ("placed","success"):("placed_to_success","forward",3),
       ("in_transit","dropped_or_misaligned"):("in_transit_to_dropped_or_misaligned","failure",4),
       ("dropped_or_misaligned","recovery"):("dropped_or_misaligned_to_recovery","recovery",6),
       ("recovery","grasped"):("recovery_to_grasped","recovery",7)}
LINEAR_REC={"start":0.0,"grasped":0.25,"in_transit":0.5,"placed":0.75,"success":1.0,
            "dropped_or_misaligned":0.0,"recovery":0.25,"terminal_failure":0.0}

def rec_node(ev: dict, prev: str) -> str:
    hold=ev["hold_current"]; goal=ev["goal_verified"]; z=float(ev["raw"].get("obj_z",0.03))
    loss=bool(ev["updates"].get("hold_loss_confirmed"))
    if ev.get("terminal_failure") or ev["updates"].get("terminal_failure"):
        if not goal and prev!="success":
            return "terminal_failure"
    if goal:
        return "success"
    if loss or (prev in ("in_transit","grasped") and not hold and not ev["updates"].get("release_intent", False) and prev!="start"):
        if prev in ("in_transit","grasped"):
            return "dropped_or_misaligned"
    if prev=="dropped_or_misaligned":
        return "recovery" if not hold else "grasped"
    if prev=="recovery":
        return "grasped" if hold else "recovery"
    if hold and z>=0.08:
        return "in_transit"
    if hold:
        return "grasped"
    if prev=="in_transit" and not hold and goal:
        return "placed"
    if prev in ("in_transit","placed") and not hold and float(ev["raw"].get("obj_x",0))>0.45:
        return "placed"
    return prev if prev!="success" else "success"

def classify(a,b):
    if a==b: return "BOUND_NODE_DWELL", None, "none", 9
    if (a,b) in EDGES:
        e=EDGES[(a,b)]; return "BOUND", e[0], e[1], e[2]
    if (a,b)==("recovery","in_transit"):
        return "SAMPLED_PATH_COMPRESSED_UNRESOLVED", None, "none", 9
    return "GRAPH_FORBIDDEN", None, "none", 9

def dual_node(ev):
    a=ev["a_current_valid"]; b=ev["b_current_valid"]; g=ev["goal_verified"]
    if g and a and b: return "success|{A,B}"
    if a and b: return "AB|{A,B}"
    if a and not b: return "A_done|{A}"
    if b and not a: return "B_done|{B}"
    if ev.get("hold_loss_confirmed") or ev["updates"].get("hold_loss_confirmed"):
        if a: return "dropped|{A}"
        if b: return "dropped|{B}"
        return "dropped|{A,B}"
    return "start|none"

def bind_recovery(events: list[dict]) -> list[dict]:
    node="start"; out=[]
    for i in range(len(events)-1):
        a=node
        b=rec_node(events[i+1], a)
        cls,eid,et,ix=classify(a,b)
        out.append(dict(episode_id=events[i]["episode_id"], step=i, node_before=a, node_after=b,
                        classification=cls, edge_id=eid, edge_type=et, edge_index=ix,
                        cost_before=REC_COST[a], cost_after=REC_COST[b],
                        linear_before=LINEAR_REC[a], linear_after=LINEAR_REC[b],
                        unordered_before=LINEAR_REC[a], unordered_after=LINEAR_REC[b],
                        t0_ns=events[i]["available_at_ns"], t1_ns=events[i+1]["available_at_ns"],
                        available_at_ns=events[i+1]["available_at_ns"],
                        distance_before=events[i]["object_goal_distance"],
                        distance_after=events[i+1]["object_goal_distance"],
                        loss_episode_id=events[i+1].get("loss_event_id"),
                        recovery_completed=bool(events[i+1].get("recovery_completed")),
                        recovered_loss_episode_id=(events[i+1].get("loss_event_id") if events[i+1].get("recovery_completed") else None),
                        a_valid=events[i+1]["a_current_valid"], b_valid=events[i+1]["b_current_valid"],
                        goal=events[i+1]["goal_verified"], hold=events[i+1]["hold_current"]))
        node=b
    return out