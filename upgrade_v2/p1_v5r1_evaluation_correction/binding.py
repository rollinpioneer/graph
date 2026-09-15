from __future__ import annotations
from . import product_graph as G
from .raw_reader import parse_flag

def transitions_from_facts(episode_id: str, facts: list[dict], task: str) -> list[dict]:
    out=[]
    if task=="recovery":
        node="start"
        for i in range(len(facts)-1):
            a=node
            obj=facts[i+1]["objects"]["obj"]
            obj=dict(obj, success_bit=facts[i+1].get("success"))
            b=G.rec_phase(obj, a, facts[i+1]["gripper_closed"])
            if facts[i+1].get("success") is True:
                b="success"
            cls,eid,et,ix=G.classify(a,b)
            # phi later
            rec_done=bool(facts[i]["objects"]["obj"].get("open_loss_id") and facts[i+1]["objects"]["obj"].get("open_loss_id") is None and facts[i+1]["objects"]["obj"]["held"] is True)
            out.append(_row(episode_id,i,a,b,cls,eid,et,ix,facts[i],facts[i+1],
                            G.REC_COST.get(a,0.75), G.REC_COST.get(b,0.75),
                            facts[i]["objects"]["obj"]["distance_actual"],
                            facts[i+1]["objects"]["obj"]["distance_actual"],
                            facts[i+1]["objects"]["obj"].get("open_loss_id") or facts[i]["objects"]["obj"].get("open_loss_id"),
                            rec_done, None, None, True if facts[i+1].get("success") else False))
            node=b
        return out
    node=None
    for i in range(len(facts)-1):
        sa=G.dual_product(facts[i]); sb=G.dual_product(facts[i+1])
        a=sa["label"]; b=sb["label"]
        et="none" if a==b else ("failure" if sb["phase"]=="LOST" and sa["phase"]!="LOST" else ("recovery" if sa["phase"]=="LOST" and sb["phase"] in ("HELD","IN_TRANSIT","RESTORED") else "forward"))
        cls="BOUND_NODE_DWELL" if a==b else "BOUND"
        rec_done=et=="recovery" and sb["phase"] in ("HELD","IN_TRANSIT","RESTORED")
        out.append(_row(episode_id,i,a,b,cls,None,et,9,facts[i],facts[i+1],
                        G.dual_cost(sa), G.dual_cost(sb),
                        _active_dist(facts[i]), _active_dist(facts[i+1]),
                        _loss_id(facts[i+1]), rec_done, sa, sb, facts[i+1].get("success") is True))
        node=b
    return out

def _active_dist(fr):
    act=fr.get("active_object")
    if act and act in fr["objects"]:
        return fr["objects"][act]["distance_actual"]
    return None

def _loss_id(fr):
    for o,v in fr["objects"].items():
        if v.get("open_loss_id"):
            return v["open_loss_id"]
    return None

def _row(eid,i,a,b,cls,edge_id,et,ix,f0,f1,c0,c1,d0,d1,lid,rec,sa,sb,succ):
    return dict(episode_id=eid, step=i, node_before=a, node_after=b, classification=cls,
                edge_id=edge_id, edge_type=et, edge_index=ix,
                cost_before=float(c0), cost_after=float(c1),
                t0_ns=f0["available_at_ns"], t1_ns=f1["available_at_ns"],
                available_at_ns=f1["available_at_ns"],
                distance_before=d0, distance_after=d1,
                loss_episode_id=lid, recovery_completed=bool(rec),
                recovered_loss_episode_id=(lid if rec else None),
                a_valid=(f1["objects"]["A"]["valid"] if "A" in f1["objects"] else None),
                b_valid=(f1["objects"]["B"]["valid"] if "B" in f1["objects"] else None),
                success_after=succ, success_before=(f0.get("success") is True),
                terminal_before=False, terminal_after=False,
                product_before=sa, product_after=sb,
                tick_before=f0["tick"], tick_after=f1["tick"])