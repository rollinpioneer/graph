from __future__ import annotations

import math, statistics
from pathlib import Path

from upgrade_v2.l2r_contact_loss_guard.temporal_evaluation import score_records
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, write_csv, write_json
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _run_o
from .clock import load_with_canonical_time
from .guard import apply_guard


def _projection(rows): return [(r.get("selected_action"),r.get("reason_code"),r.get("hold_state"),r.get("time"),r.get("capture_order"),r.get("physical_time_ns")) for r in rows]


def replay_dataset(name:str,root:Path,expected:int,output:Path)->dict:
    decisions=[]; prefix={"passed":True,"checked":0,"expected":expected*4,"mismatches":[]}
    for meta in sorted(root.glob("*/metadata.json")):
        rollout=meta.parent; rows=load_with_canonical_time(rollout); base=[{k:v for k,v in row.items() if k!="physical_time_ns"} for row in rows]
        records=apply_guard(rows,_run_o(base,"O_C3")); ref=read_json(rollout/"reference/physical_reference.json")
        scored=score_records(ref,base,_actions(records)); onset=ref.get("loss_onset_time_abs") if ref.get("reference_action")=="recover_object" else None
        latency=None if onset is None or scored["primary_action_time"] is None else scored["primary_action_time"]-float(onset)
        decisions.append({"dataset":name,"rollout_id":rollout.name,"case_id":ref["case_id"],"reference_action":ref["reference_action"],"reference_resolved":bool(ref["resolvable"]),**{k:v for k,v in scored.items() if k!="all_actions"},"latency_from_physical_onset":latency})
        for fraction in (.25,.5,.75,1.0):
            cut=max(1,math.ceil(len(rows)*fraction)); rerun=apply_guard(rows[:cut],_run_o(base[:cut],"O_C3")); prefix["checked"]+=1
            if _projection(rerun)!=_projection(records[:cut]): prefix["passed"]=False; prefix["mismatches"].append({"rollout_id":rollout.name,"fraction":fraction})
    prefix["passed"]=prefix["passed"] and prefix["checked"]==prefix["expected"]
    resolved=[r for r in decisions if r["reference_resolved"]]; lat=[float(r["latency_from_physical_onset"]) for r in decisions if r["latency_from_physical_onset"] is not None and r["temporal_outcome"]=="CORRECT_IN_WINDOW"]
    p90=sorted(lat)[min(len(lat)-1,math.ceil(.9*len(lat))-1)] if lat else None
    result={"schema":"l2rar2_clp3_historical_replay_v1","dataset":name,"rollouts":len(decisions),"expected_rollouts":expected,"correct":sum(bool(r["accurate"]) for r in resolved),"resolved":len(resolved),"early":sum(r["temporal_outcome"]=="EARLY_ACTION" for r in decisions),"false":sum(r["temporal_outcome"] in {"FALSE_RECOVERY","FALSE_RETRY"} for r in decisions),"miss":sum(r["temporal_outcome"]=="MISSED_REQUIRED_ACTION" for r in decisions),"unknown":0,"latency_p50_s":statistics.median(lat) if lat else None,"latency_p90_s":p90,"prefix":prefix}
    result["status"]="PASS" if len(decisions)==expected and result["correct"]==expected and result["early"]==result["false"]==result["miss"]==0 and prefix["passed"] else "FAIL"
    output.mkdir(parents=True,exist_ok=True); write_csv(output/"decisions.csv",decisions); write_json(output/"summary.json",result); return result


def replay_all(r17:Path,r20:Path,r21:Path,output:Path)->dict:
    rows=[replay_dataset("R17",r17,72,output/"r17"),replay_dataset("R20",r20,72,output/"r20"),replay_dataset("R21",r21,32,output/"r21")]
    result={"schema":"l2rar2_clp3_historical_replay_gate_v1","status":"PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL","datasets":rows,"total_rollouts":sum(r["rollouts"] for r in rows),"total_correct":sum(r["correct"] for r in rows),"parameter_search":False,"physical_executions":0}
    write_json(output/"gate.json",result); return result
