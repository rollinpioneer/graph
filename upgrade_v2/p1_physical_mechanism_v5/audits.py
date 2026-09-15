from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
from .state_contract import events_from_rollout, json_load
from .util import write_json, write_csv

def rollout_dirs(data: Path) -> list[Path]:
    out=[]
    for ident in data.rglob("identity.json"):
        out.append(ident.parent)
    return sorted(out)

def semantics(dest: Path) -> dict:
    ident=json_load(dest/"identity.json")
    ev=events_from_rollout(dest)
    a=[e["a_current_valid"] for e in ev]; b=[e["b_current_valid"] for e in ev]
    g=[e["goal_verified"] for e in ev]; h=[e["hold_current"] for e in ev]
    loss=any(e["updates"].get("hold_loss_confirmed") for e in ev)
    rec=any(e.get("recovery_completed") for e in ev)
    firstA=next((i for i,x in enumerate(a) if x), None)
    firstB=next((i for i,x in enumerate(b) if x), None)
    both=any(x and y for x,y in zip(a,b))
    return dict(case_id=ident["case_id"], task=("dual" if "A_x" in open(dest/"timeseries.csv",encoding="utf-8").readline() else "rec"),
                goal=any(g), loss=loss, rec=rec, firstA=firstA, firstB=firstB, both=both,
                a_end=a[-1], b_end=b[-1], n=len(ev))

def generator_gate(data: Path, art: Path) -> dict:
    dirs=rollout_dirs(data)
    rows=[]
    ok=len(dirs)==112
    for d in dirs:
        try:
            s=semantics(d); s["path"]=str(d); s["readable"]=True
        except Exception as e:
            s=dict(path=str(d), readable=False, error=str(e))
            ok=False
        rows.append(s)
    # case counts
    from collections import Counter
    c=Counter(r.get("case_id") for r in rows if r.get("readable"))
    checks={}
    def n(prefix, pred):
        xs=[r for r in rows if str(r.get("case_id","")).startswith(prefix)]
        return sum(1 for r in xs if pred(r)), len(xs)
    checks["D1"]=n("D1", lambda r: r.get("firstA") is not None and r.get("firstB") is not None and r["firstA"]<=r["firstB"] and r.get("goal"))
    checks["D2"]=n("D2", lambda r: r.get("firstA") is not None and r.get("firstB") is not None and r["firstB"]<=r["firstA"] and r.get("goal"))
    checks["D3"]=n("D3", lambda r: r.get("loss") and r.get("rec") and r.get("goal") and r.get("a_end"))
    checks["D4"]=n("D4", lambda r: r.get("loss") and r.get("rec") and r.get("goal") and r.get("b_end"))
    checks["D5"]=n("D5", lambda r: r.get("goal"))
    checks["D6"]=n("D6", lambda r: (not r.get("both")) and (not r.get("goal")))
    checks["R1"]=n("R1", lambda r: r.get("goal") and not r.get("loss"))
    checks["R2"]=n("R2", lambda r: r.get("loss") and r.get("rec") and r.get("goal"))
    checks["R3"]=n("R3", lambda r: r.get("loss") and r.get("rec") and r.get("goal"))
    checks["R4"]=n("R4", lambda r: r.get("loss") and r.get("rec") and r.get("goal"))
    checks["R5"]=n("R5", lambda r: r.get("loss") and r.get("rec") and r.get("goal"))
    checks["R6"]=n("R6", lambda r: r.get("loss") and r.get("rec") and not r.get("goal"))
    checks["R7"]=n("R7", lambda r: r.get("loss") and not r.get("rec") and not r.get("goal"))
    checks["R8"]=n("R8", lambda r: not r.get("loss"))
    hard=all(v[0]==8 and v[1]==8 for k,v in checks.items()) and len(dirs)==112
    payload=dict(status="PASS" if hard else "P1_V5_GENERATOR_FAIL", n_dirs=len(dirs), checks=checks, seed_replacements=0)
    write_json(art/"generator_gate.json", payload)
    write_csv(art/"rollout_manifest.csv", [dict(path=r.get("path"), case_id=r.get("case_id"), readable=r.get("readable"), goal=r.get("goal"), loss=r.get("loss"), rec=r.get("rec")) for r in rows])
    return payload