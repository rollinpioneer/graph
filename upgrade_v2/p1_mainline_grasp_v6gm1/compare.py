"""Same-input method comparison and claim table. No per-episode method picking."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path

FOCUS = ["SPARSE_TERMINAL","LINEAR_A_FIRST_R1","LINEAR_B_FIRST_R1","UNORDERED_VALID_COUNT",
         "VALID_COUNT_PLUS_MATCHED_EVENTS_V1","GRAPH_COST_ONLY","FULL_FROZEN","V6_CAP_POTENTIAL"]

def summarize_ledger(ledger_rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in ledger_rows:
        if r.get("reward") in (None, ""):
            continue
        by[r["episode_id"]][r["method"]].append(float(r["reward"]))
    out=[]
    for eid, methods in by.items():
        for m, rs in methods.items():
            s=math.fsum(rs); pos=math.fsum(max(0.0,x) for x in rs)
            out.append({"episode_id":eid,"method":m,"signed_return":s,"positive_weight_sum":pos,
                        "n":len(rs),"nonzero_pos":sum(x>0 for x in rs)})
    return out

def compare(summaries, states_meta):
    dual=set(); rec=set()
    for s in states_meta:
        (dual if s.get("task")=="dual_order" else rec).add(s["episode_id"])
    rows=[]
    by=defaultdict(dict)
    for r in summaries:
        by[r["episode_id"]][r["method"]]=r
    for eid, ms in by.items():
        row={"episode_id":eid,"task":"dual_order" if eid in dual else "recovery"}
        for m in FOCUS:
            if m in ms:
                row[m+"_signed"]=ms[m]["signed_return"]
                row[m+"_pos"]=ms[m]["positive_weight_sum"]
        if "LINEAR_A_FIRST_R1" in ms and "LINEAR_B_FIRST_R1" in ms:
            row["A_minus_B"]=ms["LINEAR_A_FIRST_R1"]["signed_return"]-ms["LINEAR_B_FIRST_R1"]["signed_return"]
        if "UNORDERED_VALID_COUNT" in ms and "V6_CAP_POTENTIAL" in ms:
            row["v6_minus_count"]=ms["V6_CAP_POTENTIAL"]["signed_return"]-ms["UNORDERED_VALID_COUNT"]["signed_return"]
        if "VALID_COUNT_PLUS_MATCHED_EVENTS_V1" in ms and "V6_CAP_POTENTIAL" in ms:
            row["v6_minus_event"]=ms["V6_CAP_POTENTIAL"]["signed_return"]-ms["VALID_COUNT_PLUS_MATCHED_EVENTS_V1"]["signed_return"]
        rows.append(row)
    return rows
