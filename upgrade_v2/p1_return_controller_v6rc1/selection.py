from __future__ import annotations
from .util import require_new, write_csv, write_json

CASE_E16 = ["E1_LOSS_RETURN_P40","E2_LOSS_RETURN_P80","E3_THREE_RETURNS_P40","E4_THREE_RETURNS_P80",
            "E5_REGRASP_OFFSET_LEFT_P40","E6_REGRASP_OFFSET_RIGHT_P40"]
COMPLEXITY = {"RC1_OBJECT_COMPENSATED_DIRECT":1,"RC2_OBJECT_COMPENSATED_STAGED":2,"RC3_OBJECT_ERROR_SERVO":3}

def _pass(metrics, case):
    v = (metrics or {}).get(case) or metrics.get(case)
    if isinstance(v, (list, tuple)):
        return int(v[0])
    if isinstance(v, dict):
        return int(v.get("pass_families") or 0)
    return 0

def select(eval_root, contract, out):
    out = require_new(out)
    import json
    from pathlib import Path
    rows=[]
    p=Path(eval_root)/"candidate_metrics.csv"
    # prefer evaluation_summary
    summ=json.loads((Path(eval_root)/"evaluation_summary.json").read_text(encoding="utf-8"))
    cands=summ.get("candidates") or []
    eligible=[]
    trace=[]
    for c in cands:
        cid=c["controller_id"]
        if not c.get("hard_ok"):
            trace.append({"controller_id": cid, "eliminated": "hard_gate"})
            continue
        mets=c.get("metrics") or {}
        min_e16=min(_pass(mets,k) for k in CASE_E16)
        e34=_pass(mets,"E3_THREE_RETURNS_P40")+_pass(mets,"E4_THREE_RETURNS_P80")
        e56=_pass(mets,"E5_REGRASP_OFFSET_LEFT_P40")+_pass(mets,"E6_REGRASP_OFFSET_RIGHT_P40")
        eligible.append({"controller_id": cid, "min_e16": min_e16, "e34": e34, "e56": e56,
                         "p90": c.get("p90_return_s") if c.get("p90_return_s") is not None else 9e9,
                         "complexity": COMPLEXITY[cid], "raw": c})
        rows.append({"controller_id": cid, "min_e16": min_e16, "e34": e34, "e56": e56, "p90": c.get("p90_return_s"), "complexity": COMPLEXITY[cid]})
    eligible.sort(key=lambda x: (-x["min_e16"], -x["e34"], -x["e56"], x["p90"], x["complexity"]))
    selected = eligible[0] if eligible else None
    write_csv(out/"candidate_comparison.csv", rows)
    write_json(out/"selection_trace.json", {"trace": trace, "ranked": eligible, "reward_score_used": False})
    write_json(out/"selected_controller.json", {"selected": selected, "confirmation_passed": False, "holdout_may_change_selection": False})
    return selected
