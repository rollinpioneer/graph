from __future__ import annotations

import json, math, statistics
from pathlib import Path

from upgrade_v2.l2r_contact_loss_guard.temporal_evaluation import score_records
from upgrade_v2.l2r_logical_clock_confirmation.audits import candidate_source_audit, input_provenance_audit, parameter_audit
from upgrade_v2.l2r_logical_clock_confirmation.candidate_adapter import apply_logical_guard
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_json, read_jsonl, write_csv, write_json
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _run_o, _run_s

METHODS=("O_B2_RAW","O_C3_RAW","O_C3_CLP2_LOGICAL_CLOCK","S_G_H")


def _run(method,rows,physics=None):
    if method=="O_B2_RAW": return _run_o(rows,"O_B2")
    if method=="O_C3_RAW": return _run_o(rows,"O_C3")
    if method=="O_C3_CLP2_LOGICAL_CLOCK": return apply_logical_guard(rows,_run_o(rows,"O_C3"))
    return _run_s(rows,physics or [],"S_G_H")


def _projection(rows): return [(r.get("selected_action"),r.get("reason_code"),r.get("hold_state"),r.get("time"),r.get("capture_order")) for r in rows]
def _p90(values):
    values=sorted(values); return values[min(len(values)-1,math.ceil(.9*len(values))-1)] if values else None


def _fault_audit(root: Path) -> dict:
    manifests=[read_json(path) for path in sorted(root.glob("*/online_raw/fault_injection_manifest.json"))]
    errors=[]
    for value in manifests:
        if value.get("reference_hashes_before")!=value.get("reference_hashes_after"): errors.append("reference_changed")
        if value.get("raw_contact_sha256_before")!=value.get("raw_contact_sha256_after"): errors.append("raw_contact_changed")
        if value.get("candidate_fields_exclude_fault_metadata") is not True: errors.append("fault_metadata_visible")
    return {"schema":"l2rar2_r21_fault_audit_v1","passed":len(manifests)==32 and not errors,"rollouts":len(manifests),"errors":errors}


def evaluate(repo: Path, root: Path, reference_root: Path, gate_path: Path, output: Path) -> dict:
    if not read_json(gate_path).get("performance_evaluation_allowed"): raise RuntimeError("GENERATOR_GATE_REQUIRED")
    output.mkdir(parents=True,exist_ok=False); online={}; saved={}
    for meta in sorted(root.glob("*/metadata.json")):
        rid=meta.parent.name; online[rid]=load_candidate_input(meta.parent/"candidate_input")
        for method in METHODS[:3]: saved[(method,rid)]=read_jsonl(meta.parent/"candidate_output"/f"{method}.jsonl")
    refs=read_json(reference_root/"physical_reference_index.json")["rows"]; decisions=[]
    prefix={"schema":"l2rar2_r21_prefix_audit_v1","passed":True,"checked":0,"expected":512,"mismatches":[]}
    for method in METHODS:
        for ref in refs:
            rid=ref["rollout_id"]; rows=online[rid]; physics=read_jsonl((reference_root/ref["physics_trace_path"]).resolve())
            records=saved[(method,rid)] if method!="S_G_H" else _run(method,rows,physics)
            scored=score_records(ref,rows,_actions(records)); onset=ref.get("loss_onset_time_abs") if ref["reference_action"]=="recover_object" else None
            latency=None if onset is None or scored["primary_action_time"] is None else scored["primary_action_time"]-float(onset)
            decisions.append({"method":method,"rollout_id":rid,"case_id":ref["case_id"],"truth_action":ref["reference_action"],
                              "reference_resolved":bool(ref["resolvable"]),**{k:v for k,v in scored.items() if k!="all_actions"},
                              "actions_json":json.dumps(_actions(records),sort_keys=True),"latency_from_physical_onset":latency,
                              "unknown":bool(not _actions(records) and records and records[-1].get("selected_action")=="needs_observation")})
            for fraction in (.25,.5,.75,1.0):
                cut=max(1,math.ceil(len(rows)*fraction)); rerun=_run(method,rows[:cut],physics); prefix["checked"]+=1
                if _projection(rerun)!=_projection(records[:cut]): prefix["passed"]=False; prefix["mismatches"].append({"method":method,"rollout_id":rid,"fraction":fraction})
    prefix["passed"]=prefix["passed"] and prefix["checked"]==prefix["expected"]
    metrics=[]
    for method in METHODS:
        items=[r for r in decisions if r["method"]==method]; resolved=[r for r in items if r["reference_resolved"]]
        negative=[r for r in items if r["case_id"].startswith(("T1_","T2_","T3_","T4_","T12_"))]
        strong=[r for r in items if r["case_id"].startswith(("T7_","T9_","T11_"))]
        lat=[float(r["latency_from_physical_onset"]) for r in strong if r["temporal_outcome"]=="CORRECT_IN_WINDOW" and r["latency_from_physical_onset"] is not None]
        metrics.append({"method":method,"eligible_for_selection":method=="O_C3_CLP2_LOGICAL_CLOCK","correct":sum(bool(r["accurate"]) for r in resolved),"resolved":len(resolved),
                        "accuracy":sum(bool(r["accurate"]) for r in resolved)/len(resolved),"early_actions":sum(r["temporal_outcome"]=="EARLY_ACTION" for r in items),
                        "false_actions":sum(r["primary_action"] is not None for r in negative),"misses":sum(r["temporal_outcome"]=="MISSED_REQUIRED_ACTION" for r in items),
                        "unknown_rate":sum(bool(r["unknown"]) for r in items)/len(items),"strong_correct":sum(r["temporal_outcome"]=="CORRECT_IN_WINDOW" for r in strong),
                        "t3_no_action":sum(r["primary_action"] is None for r in items if r["case_id"].startswith("T3_")),
                        "t4_no_action":sum(r["primary_action"] is None for r in items if r["case_id"].startswith("T4_")),
                        "t12_no_recovery":sum(r["primary_action"]!="recover_object" for r in items if r["case_id"].startswith("T12_")),
                        "latency_p50_s":statistics.median(lat) if lat else None,"latency_p90_s":_p90(lat)})
    candidate=next(r for r in metrics if r["method"]=="O_C3_CLP2_LOGICAL_CLOCK")
    audits={"input_provenance":input_provenance_audit(),"candidate_source":candidate_source_audit(repo),"parameters":parameter_audit(),"faults":_fault_audit(root),"prefix":prefix}
    passed=(all(v["passed"] for v in audits.values()) and candidate["accuracy"]>=71/72 and candidate["strong_correct"]/12>=29/30
            and candidate["early_actions"]==0 and candidate["false_actions"]==0 and candidate["misses"]==0 and candidate["unknown_rate"]<=.05
            and candidate["latency_p90_s"] is not None and candidate["latency_p90_s"]<=.20 and candidate["t3_no_action"]==4 and candidate["t4_no_action"]==4 and candidate["t12_no_recovery"]==4)
    result={"schema":"l2rar2_r21_performance_evaluation_v1","status":"PASS" if passed else "FAIL","candidate_id":"O_C3_CLP2_LOGICAL_CLOCK","candidate_performance_pass":passed,"methods":metrics}
    write_csv(output/"per_event_decisions.csv",decisions); write_json(output/"method_metrics.json",result)
    for name,value in audits.items(): write_json(output/f"{name}_audit.json",value)
    return result
