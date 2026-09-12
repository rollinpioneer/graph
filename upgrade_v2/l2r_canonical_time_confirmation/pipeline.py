from __future__ import annotations

from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.detector_adapter import detect_rollout
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_csv, read_json, write_json, write_jsonl
from upgrade_v2.l2r_logical_clock_confirmation.logical_fault_injection import build_rollout_candidate_input
from upgrade_v2.l2r_logical_clock_confirmation.reference_builder import build_reference as build_frozen_reference
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o
from .clock import load_with_canonical_time
from .guard import apply_guard


def roots(root): return [p.parent for p in sorted(root.glob("*/metadata.json"))]
def detect(root:Path)->dict:
    rows=[detect_rollout(p) for p in roots(root)]; result={"schema":"l2rar2_r22_detection_v1","status":"PASS" if len(rows)==24 and all(r["status"]=="PASS" for r in rows) else "FAIL","rollouts":len(rows),"rows":rows}; write_json(root/"detection_status.json",result); return result
def build_inputs(root:Path)->dict:
    rows=[build_rollout_candidate_input(p) for p in roots(root)]; result={"schema":"l2rar2_r22_inputs_v1","status":"PASS" if len(rows)==24 else "FAIL","rollouts":len(rows),"rows":rows}; write_json(root/"candidate_input_status.json",result); return result
def run_candidates(root:Path)->dict:
    rows=[]
    for rollout in roots(root):
        base=load_candidate_input(rollout/"candidate_input"); enriched=load_with_canonical_time(rollout); raw=_run_o(base,"O_C3"); clp3=apply_guard(enriched,raw)
        out=rollout/"candidate_output_clp3"; out.mkdir(parents=True,exist_ok=False); write_jsonl(out/"O_C3_RAW.jsonl",raw); write_jsonl(out/"O_C3_CLP3_CANONICAL_TIME.jsonl",clp3); rows.append({"rollout_id":rollout.name,"rows":len(clp3)})
    result={"schema":"l2rar2_r22_candidate_execution_v1","status":"PASS" if len(rows)==24 else "FAIL","rollouts":len(rows)}; write_json(root/"candidate_execution_status.json",result); return result
def reference(root:Path,output:Path)->dict:
    build_frozen_reference(root,output); count=len(read_json(output/"physical_reference_index.json")["rows"]); return {"schema":"l2rar2_r22_reference_v1","status":"PASS" if count==24 else "FAIL","rollouts":count}
def generator(root:Path,refroot:Path,output:Path)->dict:
    refs=read_json(refroot/"physical_reference_index.json")["rows"]; frames=read_csv(refroot/"frame_manifest.csv"); det=read_csv(refroot/"detection_manifest.csv"); faults=read_csv(refroot/"fault_injection_manifest.csv")
    subset=lambda *n:[r for r in refs if r["case_id"].startswith(tuple(f"T{x}_" for x in n))]
    missing=[r for r in frames if str(r.get("frame_missing","")).lower()=="true"]
    clocks=[]
    for rollout in roots(root): clocks.extend(load_with_canonical_time(rollout))
    gates={"rollouts_24":len(refs)==24,"trace_24":sum(bool(r["trace_complete"]) for r in refs)==24,"numeric_24":sum(bool(r["numeric_health_pass"]) for r in refs)==24,"prehold_24":sum(bool(r["pre_hold_verified"]) for r in refs)==24,"detector_24":sum(r["status"]=="PASS" for r in det)==24,"detector_errors_0":sum(int(float(r.get("errors") or 0)) for r in det)==0,"canonical_time_all_rows":all(isinstance(r["physical_time_ns"],int) for r in clocks),"strong_loss_8":sum(bool(r["physical_loss_confirmed"]) for r in subset(7,11))==8,"negative_no_loss_12":sum(not bool(r["physical_loss_confirmed"]) and bool(r["resolvable"]) for r in subset(1,3,4))==12,"release_4":sum(r["state"]=="COMMANDED_RELEASE" for r in subset(12))==4,"t3_fault_4":sum(r.get("fault_type")=="T3_SINGLE_FALSE" for r in faults)==4,"t4_fault_4":sum(r.get("fault_type")=="T4_SAME_TIME_FALSE_TRUE" for r in faults)==4,"planned_missing_12":sum(r.get("frame_missing_reason")=="T11_SUCCESSOR_PERIODIC_RGB_DROPOUT" for r in missing)==12,"unplanned_missing_0":all(r.get("frame_missing_reason")=="T11_SUCCESSOR_PERIODIC_RGB_DROPOUT" for r in missing)}
    result={"schema":"l2rar2_r22_generator_v1",**gates,"evaluation_allowed":all(gates.values())}; write_json(output,result); return result
