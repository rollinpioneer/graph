from __future__ import annotations
from typing import Any

REC_COST={"start":4/6,"grasped":3/6,"in_transit":2/6,"placed":1/6,"dropped_or_misaligned":5/6,
          "recovery":4/6,"success":0.0,"terminal_failure":1.0}
REC_IDX={n:i for i,n in enumerate(["start","grasped","in_transit","placed","dropped_or_misaligned","recovery","success","terminal_failure"])}
EDGES={("start","grasped"):("start_to_grasped","forward",0),
       ("grasped","in_transit"):("grasped_to_in_transit","forward",1),
       ("in_transit","placed"):("in_transit_to_placed","forward",2),
       ("placed","success"):("placed_to_success","forward",3),
       ("in_transit","dropped_or_misaligned"):("in_transit_to_dropped_or_misaligned","failure",4),
       ("dropped_or_misaligned","recovery"):("dropped_or_misaligned_to_recovery","recovery",6),
       ("recovery","grasped"):("recovery_to_grasped","recovery",7)}

def rec_phase(obj: dict, prev: str, close: bool|None) -> str:
    held=obj["held"]; z=obj["pos"][2] if obj["pos"][2] is not None else 0.03
    if obj.get("success_bit"):
        return "success"
    if prev in ("in_transit","grasped") and held is False and close is True:
        return "dropped_or_misaligned"
    if prev=="dropped_or_misaligned":
        return "grasped" if held is True else "dropped_or_misaligned"  # do not auto-enter recovery
    if prev=="recovery":
        return "grasped" if held is True else "recovery"
    if held is True and z is not None and z>=0.08:
        return "in_transit"
    if held is True:
        return "grasped"
    if prev=="in_transit" and held is False:
        return "placed"
    return prev or "start"

def classify(a,b):
    if a==b: return "BOUND_NODE_DWELL", None, "none", 9
    if (a,b) in EDGES:
        e=EDGES[(a,b)]; return "BOUND", e[0], e[1], e[2]
    if (a,b)==("dropped_or_misaligned","in_transit"):
        return "SAMPLED_PATH_COMPRESSED_UNRESOLVED", None, "none", 9
    return "GRAPH_FORBIDDEN", None, "none", 9

def dual_product(frame: dict) -> dict:
    A=frame["objects"].get("A"); B=frame["objects"].get("B")
    a=A["valid"] if A else None; b=B["valid"] if B else None
    V=tuple(x for x,ok in (("A",a),("B",b)) if ok is True)
    active=frame.get("active_object")
    phase="IDLE"
    if active and active in frame["objects"]:
        o=frame["objects"][active]
        if o["held"] is True:
            phase="HELD"
            z=o["pos"][2]
            if z is not None and z>=0.08: phase="IN_TRANSIT"
        elif o.get("open_loss_id"):
            phase="LOST"
        elif o["ever_held"] and o["held"] is False:
            phase="LOST"
    success=frame.get("success")
    label=f"V={{{','.join(V)}}};active={active};phase={phase};succ={success}"
    return dict(V=V, active=active, phase=phase, success=success, label=label, a=a, b=b)

def dual_cost(st: dict) -> float:
    if st["success"] is True: return 0.0
    n=len(st["V"])
    if st["phase"]=="LOST": return 0.75
    if n==2: return 0.25
    if n==1: return 0.5
    return 0.75