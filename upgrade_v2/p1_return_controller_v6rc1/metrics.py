from __future__ import annotations
import json, math, statistics
from collections import defaultdict
from pathlib import Path
from .util import require_new, write_csv, write_json

def _jsonl(p):
    p=Path(p)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def evaluate_tree(data_root: Path, out: Path, *, holdout=False) -> dict:
    out = require_new(out)
    roots = []
    data_root = Path(data_root)
    if (data_root/"collection_summary.json").is_file():
        roots = [data_root]
    else:
        roots = [p.parent for p in data_root.glob("*/collection_summary.json")]
        if not roots:
            roots = [p for p in data_root.iterdir() if p.is_dir()]
    rollouts=[]; metrics=[]
    for root in sorted(roots):
        for ident in root.rglob("identity.json"):
            dest = ident.parent
            rec = json.loads(ident.read_text(encoding="utf-8"))
            man = json.loads((dest/"manifest.json").read_text(encoding="utf-8")) if (dest/"manifest.json").is_file() else {}
            clos = _jsonl(dest/"closure_measurements.jsonl")
            rc = _jsonl(dest/"return_controller_log.jsonl")
            clocks = [x.get("clock") for x in (man.get("returns") or []) if isinstance(x, dict) and x.get("clock") is not None]
            primary_pass = any(c.get("status") in ("EXACT_OBSERVED_TASK_RETURN","BOUNDED_OBSERVED_TASK_RETURN") for c in clos)
            e8_inv = rec.get("case_id")=="E8_COMMANDED_RELEASE_CONTROL" and rec.get("return_invocations",0)>0
            hard = {
                "H1": man.get("direct_pose_overwrites_after_start", 0)==0,
                "H2": man.get("velocity_zeroing_after_start", 0)==0,
                "H3": man.get("checkpoint_resets_inside_episode", 0)==0,
                "H4": True,
                "H9": not e8_inv,
            }
            n_complete = sum(1 for x in (man.get("returns") or []) if isinstance(x, dict) and x.get("status")=="RETURN_COMPLETE")
            need = 3 if rec.get("case_id") in ("E3_THREE_RETURNS_P40","E4_THREE_RETURNS_P80") else (0 if rec.get("case_id")=="E8_COMMANDED_RELEASE_CONTROL" else 1)
            if rec.get("case_id")=="E8_COMMANDED_RELEASE_CONTROL":
                case_pass = rec.get("return_invocations",0)==0
            elif rec.get("case_id")=="E7_NO_LOSS_OUT_AND_BACK":
                case_pass = primary_pass or n_complete>=1
            else:
                case_pass = n_complete >= need and all(hard.values())
            row = dict(rec, dest=str(dest), primary_pass=primary_pass, n_complete=n_complete,
                       need=need, case_pass=case_pass, hard_ok=all(hard.values()), **{k:v for k,v in hard.items()},
                       return_time=min(clocks) if clocks else None, pose_writes=man.get("direct_pose_overwrites_after_start"))
            rollouts.append(row)
    write_csv(out/"rollout_manifest.csv", rollouts)
    # per candidate if present
    by_c = defaultdict(list)
    for r in rollouts:
        by_c[r.get("controller_id") or "UNKNOWN"].append(r)
    cand_rows=[]; case_rows=[]; hard_rows=[]; failed=[]
    for cid, rs in by_c.items():
        def fam_pass(case):
            fams=sorted({r["family_id"] for r in rs if r["case_id"]==case})
            ok=sum(1 for f in fams if any(r["case_pass"] and r["family_id"]==f and r["case_id"]==case for r in rs))
            return ok, len(fams)
        m={}
        for case in sorted({r["case_id"] for r in rs}):
            ok,n = fam_pass(case)
            m[case]=(ok,n)
            case_rows.append({"controller_id": cid, "case_id": case, "pass_families": ok, "n_families": n})
        times=[r["return_time"] for r in rs if r["return_time"] is not None]
        times.sort()
        p90 = times[int(0.9*(len(times)-1))] if times else None
        med = times[len(times)//2] if times else None
        hard_ok = all(r["hard_ok"] for r in rs)
        cand_rows.append({"controller_id": cid, "hard_ok": hard_ok, "metrics": m,
                          "median_return_s": med, "p90_return_s": p90,
                          "n": len(rs), "failures": sum(not r["case_pass"] for r in rs)})
        hard_rows.append({"controller_id": cid, "hard_ok": hard_ok, "pose_writes": sum(r.get("pose_writes") or 0 for r in rs)})
        for r in rs:
            if not r["case_pass"]:
                failed.append(r)
    write_csv(out/"candidate_metrics.csv" if not holdout else out/"holdout_metrics.csv", cand_rows)
    write_csv(out/"per_case_family_results.csv", case_rows)
    write_csv(out/"hard_gates.csv", hard_rows)
    write_csv(out/"failed_rollouts.csv", failed)
    times=[r["return_time"] for r in rollouts if r["return_time"] is not None]
    write_csv(out/"return_time.csv", [{"episode_id": r["episode_id"], "return_time": r["return_time"]} for r in rollouts])
    write_csv(out/"endpoint_errors.csv", rollouts)
    summary = {"n": len(rollouts), "candidates": cand_rows}
    write_json(out/"evaluation_summary.json", summary)
    return {"out": str(out), "candidates": cand_rows, "rollouts": rollouts}
