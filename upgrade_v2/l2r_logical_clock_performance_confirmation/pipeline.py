from __future__ import annotations

from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.candidate_adapter import run_online_rollout
from upgrade_v2.l2r_logical_clock_confirmation.detector_adapter import detect_rollout
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_csv, read_json, write_json
from upgrade_v2.l2r_logical_clock_confirmation.logical_fault_injection import build_rollout_candidate_input
from upgrade_v2.l2r_logical_clock_confirmation.reference_builder import build_reference as build_r20_reference


def _roots(root: Path) -> list[Path]: return [path.parent for path in sorted(root.glob("*/metadata.json"))]


def detect(root: Path) -> dict:
    rows = [detect_rollout(path) for path in _roots(root)]
    result = {"schema": "l2rar2_r21_detection_status_v1", "status": "PASS" if len(rows)==32 and all(row["status"]=="PASS" for row in rows) else "FAIL", "rollouts": len(rows), "rows": rows}
    write_json(root/"detection_status.json",result); return result


def build_inputs(root: Path) -> dict:
    rows = [build_rollout_candidate_input(path) for path in _roots(root)]
    result = {"schema":"l2rar2_r21_candidate_input_status_v1","status":"PASS" if len(rows)==32 else "FAIL","rollouts":len(rows),"rows":rows}
    write_json(root/"candidate_input_status.json",result); return result


def run_candidates(root: Path) -> dict:
    rows=[run_online_rollout(path) for path in _roots(root)]
    result={"schema":"l2rar2_r21_candidate_execution_status_v1","status":"PASS" if len(rows)==32 else "FAIL","rollouts":len(rows)}
    write_json(root/"candidate_execution_status.json",result); return result


def build_reference(root: Path, output: Path) -> dict:
    build_r20_reference(root,output)
    count=len(read_json(output/"physical_reference_index.json")["rows"])
    return {"schema":"l2rar2_r21_reference_status_v1","status":"PASS" if count==32 else "FAIL","rollouts":count}


def generator(root: Path, reference_root: Path, output: Path) -> dict:
    refs=read_json(reference_root/"physical_reference_index.json")["rows"]
    frames=read_csv(reference_root/"frame_manifest.csv"); detections=read_csv(reference_root/"detection_manifest.csv"); faults=read_csv(reference_root/"fault_injection_manifest.csv")
    subset=lambda *n:[r for r in refs if r["case_id"].startswith(tuple(f"T{x}_" for x in n))]
    strong=sum(bool(r["physical_loss_confirmed"]) for r in subset(7,9,11))
    negative=sum(not bool(r["physical_loss_confirmed"]) and bool(r["resolvable"]) for r in subset(1,2,3,4))
    release=sum(r["state"]=="COMMANDED_RELEASE" for r in subset(12))
    missing=[r for r in frames if str(r.get("frame_missing","")).lower()=="true"]
    planned=[r for r in missing if r.get("frame_missing_reason")=="T11_SUCCESSOR_PERIODIC_RGB_DROPOUT"]
    t3=[r for r in faults if r.get("fault_type")=="T3_SINGLE_FALSE"]; t4=[r for r in faults if r.get("fault_type")=="T4_SAME_TIME_FALSE_TRUE"]
    gates={"rollouts_32":len(refs)==32,"trace_32":sum(bool(r["trace_complete"]) for r in refs)==32,
           "numeric_32":sum(bool(r["numeric_health_pass"]) for r in refs)==32,"prehold_32":sum(bool(r["pre_hold_verified"]) for r in refs)==32,
           "detector_32":sum(r["status"]=="PASS" for r in detections)==32,"detector_errors_0":sum(int(float(r.get("errors") or 0)) for r in detections)==0,
           "strong_loss_12":strong==12,"negative_no_loss_16":negative==16,"release_4":release==4,
           "t3_fault_4":len(t3)==4,"t4_fault_4":len(t4)==4,"planned_missing_12":len(planned)==12,"unplanned_missing_0":len(missing)==len(planned),
           "phase_0_4":sum(r.get("requested_phase_offset_ms")==0 and abs(float(r["actual_phase_offset_ms"]))<1e-6 for r in subset(7))==4,
           "phase_20_4":sum(r.get("requested_phase_offset_ms")==20 and abs(float(r["actual_phase_offset_ms"])-20)<1e-6 for r in subset(9))==4,
           "phase_40_4":sum(r.get("requested_phase_offset_ms")==40 and abs(float(r["actual_phase_offset_ms"])-40)<1e-6 for r in subset(11))==4}
    result={"schema":"l2rar2_r21_generator_gate_v1",**gates,"performance_evaluation_allowed":all(gates.values())}
    write_json(output,result); return result
