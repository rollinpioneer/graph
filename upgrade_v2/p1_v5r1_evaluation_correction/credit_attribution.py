from __future__ import annotations

def classify_positive(row: dict, ledger_by_ep: dict) -> str:
    if float(row.get("reward_mu",0))<=0:
        return "NONPOSITIVE"
    et=row.get("edge_type")
    rec=bool(row.get("recovery_completed"))
    if rec:
        return "COMPLETION_EVENT_CREDIT"
    if et=="recovery" and not rec:
        # auto graph jump without independent progress
        d0=row.get("distance_before"); d1=row.get("distance_after")
        if d0 is not None and d1 is not None and float(d1)+1e-9<float(d0):
            return "MEASURED_PARTIAL_PROGRESS"
        return "GRAPH_LABEL_ONLY_CREDIT"
    if d_progress(row):
        return "MEASURED_PARTIAL_PROGRESS"
    return "CREDIT_CAUSE_UNRESOLVED"

def d_progress(row):
    d0=row.get("distance_before"); d1=row.get("distance_after")
    if d0 is None or d1 is None: return False
    return float(d1)+1e-9<float(d0)