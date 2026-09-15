from __future__ import annotations
from collections import defaultdict
from typing import Any
from .graph_binding import REC_IDX, REC_COST

METHODS=["SPARSE_TERMINAL","LINEAR_A_FIRST","LINEAR_B_FIRST","UNORDERED_COMPLETION_PROGRESS",
         "GRAPH_COST_ONLY","GRAPH_COST_PHI","GRAPH_COST_DEBT","FULL_FROZEN","NO_RECOVERY_CREDIT",
         "PAIRED_EVENT_BALANCED_V1","GLOBAL_POTENTIAL_AUDIT_V1"]

def clip(x, b=1.5):
    return max(-b, min(b, x))

def potential(c,p):
    if not (0<=c<=1 and 0<=p<=1):
        raise ValueError("domain")
    return -c + 0.5*p

def attach_phi(trans):
    entry=None; prev=0.0
    for i,t in enumerate(trans):
        a,b=t["node_before"], t["node_after"]
        d1=float(t.get("distance_after") or 0)
        d0=float(t.get("distance_before") or 0)
        if b=="in_transit" and entry is None and d1>1e-9:
            entry=d1
        def phi(d):
            if entry is None: return 0.0
            return max(0.0, min(1.0, 1.0-d/max(entry,1e-9)))
        pb = phi(d0) if a=="in_transit" else 0.0
        pa = phi(d1) if b=="in_transit" else 0.0
        if i:
            pb=prev if a==trans[i-1]["node_after"] else pb
        t["phi_before"]=pb; t["phi_after"]=pa; prev=pa
    return trans

def dual_progress(t, kind):
    a=bool(t.get("a_valid")); b=bool(t.get("b_valid")); g=bool(t.get("goal"))
    if kind=="UA":
        return ((1 if a else 0)+(1 if b else 0)+(1 if g else 0))/3
    if kind=="LA":
        if g: return 1.0
        if a and b: return 2/3
        if a: return 1/3
        if b: return 1/3
        return 0.0
    if kind=="LB":
        if g: return 1.0
        if a and b: return 2/3
        if b: return 1/3
        if a: return 1/3
        return 0.0
    return 0.0

def score_episode(trans: list[dict], task: str) -> list[dict]:
    trans=attach_phi([dict(t) for t in trans])
    out=[]
    debt=0.0; tokens=set(); ever=set(); sparse_done=False
    for t in trans:
        c0,c1=float(t["cost_before"]), float(t["cost_after"])
        p0,p1=float(t["phi_before"]), float(t["phi_after"])
        et=t["edge_type"]; same=(t["node_before"]==t["node_after"] and et not in ("failure","recovery"))
        pd=(p1-p0) if same else 0.0
        core=clip((c0-c1)+0.5*pd)
        recs={}
        # sparse
        r=0.0
        if t["node_after"] in ("success","success|{A,B}") and t["node_before"] not in ("success","success|{A,B}") and not sparse_done:
            r=1.0; sparse_done=True
        recs["SPARSE_TERMINAL"]=r
        # linear/unordered
        if task=="dual_order":
            recs["LINEAR_A_FIRST"]=dual_progress(t,"LA")-dual_progress({**t,"a_valid":t.get("a_valid"),"b_valid":t.get("b_valid"),"goal":t.get("goal")},"LA")
            # use before from previous node labels - store a/b on transition after; need before flags
            recs["LINEAR_A_FIRST"]=None
        recs["LINEAR_A_FIRST"]=float(t.get("linA_after", t["linear_after"]))-float(t.get("linA_before", t["linear_before"]))
        recs["LINEAR_B_FIRST"]=float(t.get("linB_after", t["linear_after"]))-float(t.get("linB_before", t["linear_before"]))
        recs["UNORDERED_COMPLETION_PROGRESS"]=float(t["unordered_after"])-float(t["unordered_before"])
        recs["GRAPH_COST_ONLY"]=clip(c0-c1)
        recs["GRAPH_COST_PHI"]=clip((c0-c1)+0.5*pd)
        # debt methods
        d=debt; r=core
        if et=="failure" and core<0: d+=-core
        if et=="recovery" and core>0:
            cred=min(core,d); d-=cred; r=cred
        recs["GRAPH_COST_DEBT"]=r if et!="none" or True else r
        # recompute GRAPH_COST_DEBT properly without phi
        core_n=clip(c0-c1); d2=debt; r2=core_n
        if et=="failure" and core_n<0: d2+=-core_n
        if et=="recovery" and core_n>0:
            cred=min(core_n,d2); d2-=cred; r2=cred
        recs["GRAPH_COST_DEBT"]=r2
        recs["FULL_FROZEN"]=r
        recs["NO_RECOVERY_CREDIT"]= 0.0 if et=="recovery" else (clip((c0-c1)+0.5*pd) if et!="recovery" else 0.0)
        # paired
        pr=0.0; st="SCORED"
        lid=t.get("loss_episode_id")
        if et=="failure" and lid:
            if lid not in ever:
                ever.add(lid); tokens.add(lid); pr=-1.0
        if t.get("recovery_completed"):
            rid=t.get("recovered_loss_episode_id") or lid
            if rid in tokens:
                tokens.remove(rid); pr+=1.0
            else:
                st="UNMATCHED_OR_ALREADY_REPAID_RECOVERY_NO_CREDIT"
        recs["PAIRED_EVENT_BALANCED_V1"]=pr
        recs["GLOBAL_POTENTIAL_AUDIT_V1"]=potential(c1,p1)-potential(c0,p0)
        debt=d
        for m,val in recs.items():
            out.append(dict(episode_id=t["episode_id"], step=t["step"], method=m, edge_type=et,
                            reward_mu=float(val), weight_positive=max(0.0,float(val)),
                            failure_debt_after=debt if m=="FULL_FROZEN" else None,
                            node_before=t["node_before"], node_after=t["node_after"],
                            available_at_ns=t["available_at_ns"], score_status=st if m=="PAIRED_EVENT_BALANCED_V1" else "SCORED"))
    return out