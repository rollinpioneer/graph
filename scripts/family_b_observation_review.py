#!/usr/bin/env python3
"""Offline saved-evidence review for Family B."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, subprocess
from pathlib import Path
from typing import Any

BRANCHES = {
    "599795abeaf80f6f": {"layout":"layout_0","context":"B_PENDING","candidate":"pad_u","expected":"TASK_SUCCESS"},
    "9ac693fc9b60d62c": {"layout":"layout_0","context":"B_PENDING","candidate":"pad_v","expected":"NO_PLAN"},
    "40517e4bb6a02609": {"layout":"layout_1","context":"C_PENDING","candidate":"pad_u","expected":"TASK_SUCCESS"},
    "3aaf043a49195ee7": {"layout":"layout_1","context":"C_PENDING","candidate":"pad_v","expected":"NO_PLAN"},
}
OBJECTS = ("carrier","obj_b","obj_c")
FIELDS = ["branch_id","layout","context","candidate","action_index","action_id","object_id","frame_ref","frame_sha256",
          "blob_present","blob_pixels","blob_xyz","depth_support_count","perception_unknown_reason","OnTable","Held",
          "Inside_receiver","AtBuffer_u","AtBuffer_v","GripperEmpty","Open_receiver","legal_candidate_ids",
          "evaluator_success","evaluator_reason","planner_status"]

def jread(p: Path, default=None):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError): return default

def jwrite(p: Path, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")

def sha256_file(p: Path):
    try:
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(1048576), b""): h.update(b)
        return h.hexdigest()
    except OSError: return None

def source_dir(root, value):
    p=Path(value); return p if p.is_absolute() else root/p

def branch_dir(source, bid): return source/"captures"/bid

def receipt(source, bid): return jread(source/"physical"/"branch_results"/(bid+".json"), {}) or {}

def action_dirs(bdir):
    def key(p):
        m=re.search(r"(\d+)$",p.name); return int(m.group(1)) if m else 10**9
    return sorted([p for p in bdir.glob("action_*") if p.is_dir()], key=key)

def last_json(p):
    v=jread(p,{})
    return v[-1] if isinstance(v,list) and v else (v if isinstance(v,dict) else {})

def facts_from(p):
    v=jread(p,[])
    if isinstance(v,dict):
        return dict(v.get("facts",v)) if isinstance(v.get("facts",v),dict) else {}
    return {r["fact_id"]:r.get("value","UNKNOWN") for r in v if isinstance(r,dict) and r.get("fact_id")}

def truth(v):
    if hasattr(v,"name"): return v.name
    if isinstance(v,bool): return "TRUE" if v else "FALSE"
    return str(v).upper() if v is not None else "UNKNOWN"

def production_mask(root: Path, facts: dict[str,Any]):
    try:
        from cp_disr.platforms.libero.family_b_runtime import build_task_template
        from cp_disr.contracts import precondition_value
        from cp_disr.facts import Truth
        t=build_task_template(str(root/"configs/runtime/tp_fb_contract_registry.yaml"))
        vals={k:getattr(Truth,str(v).upper(),Truth.UNKNOWN) for k,v in facts.items()}
        ids=[c.id for c in t.contracts]
        return ids,[precondition_value(c,vals) is Truth.TRUE for c in t.contracts]
    except Exception:
        return [],[]

def action_record(rec, idx):
    a=rec.get("actions",[]) if isinstance(rec,dict) else []
    return a[idx] if isinstance(a,list) and idx<len(a) and isinstance(a[idx],dict) else {}

def action_id(rec, idx):
    return action_record(rec,idx).get("action_id","")

def blob_details(per, obj):
    blobs=per.get("blobs",{}) if isinstance(per.get("blobs"),dict) else {}
    ids=per.get("blob_ids",[]) if isinstance(per.get("blob_ids"),list) else []
    xyzs=per.get("blob_xyz",{}) if isinstance(per.get("blob_xyz"),dict) else {}
    counts=per.get("pixel_counts",{}) if isinstance(per.get("pixel_counts"),dict) else {}
    b=blobs.get(obj) if isinstance(blobs,dict) else None
    xyz=b.get("xyz") if isinstance(b,dict) else xyzs.get(obj)
    return bool((obj in blobs or obj in ids) and xyz is not None), (b.get("pixels") if isinstance(b,dict) else counts.get(obj)), xyz, counts.get(obj)

def image_info(p):
    try:
        from PIL import Image
        with Image.open(p) as im:
            return {"mode":im.mode,"shape":[im.height,im.width,len(im.getbands())],
                    "pixels_sha256":hashlib.sha256(im.tobytes()).hexdigest()}
    except Exception as e: return {"error":str(e)}

def frame_path(adir, ref):
    p=adir/(str(ref)+"_rgb.png") if ref else None
    if p and p.is_file(): return p
    fs=sorted(adir.glob("frame_*_rgb.png")); return fs[-1] if fs else None

def inventory(root, source, output):
    rows=[]
    for p in sorted((source/"captures").rglob("*")):
        if p.is_file(): rows.append({"path":p.relative_to(root).as_posix(),"size":p.stat().st_size,"sha256":sha256_file(p)})
    jwrite(output/"inventory/capture_manifest.json",{"source":str(source),"capture_count":len(rows),
        "expected_receipt_count":402,"count_matches_receipt":len(rows)==402,"files":rows})
    try:
        branch=subprocess.check_output(["git","-C",str(root),"branch","--show-current"],text=True).strip()
        head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    except Exception: branch=head="UNKNOWN"
    jwrite(output/"inventory/source_identity.json",{"branch":branch,"head":head,
        "baseline_sha":"cb2f0a2901ba81ab22f9875ebeca6ec000a5b361","source":str(source),"server_only_captures":True})

def inspect_observation(root, source, output):
    rows=[]; contacts=[]; diag=[]
    for bid,meta in BRANCHES.items():
        bdir=branch_dir(source,bid); rec=receipt(source,bid)
        for adir in action_dirs(bdir):
            m=re.search(r"(\d+)$",adir.name); idx=int(m.group(1)) if m else -1
            per=last_json(adir/"perception.json"); facts=facts_from(adir/"facts.json")
            ev=jread(adir/"evaluator.json",{}) or {}; pl=jread(adir/"planner.json",{}) or {}
            ref=per.get("frame_id"); fp=frame_path(adir,ref); ids,mask=production_mask(root,facts)
            legal=[x for x,ok in zip(ids,mask) if ok]; ar=action_record(rec,idx)
            for obj in OBJECTS:
                present,pixels,xyz,depth=blob_details(per,obj)
                rows.append({"branch_id":bid,"layout":meta["layout"],"context":meta["context"],"candidate":meta["candidate"],
                    "action_index":idx,"action_id":ar.get("action_id",action_id(rec,idx)),"object_id":obj,
                    "frame_ref":ref or "","frame_sha256":per.get("rgb_sha256") or (sha256_file(fp) if fp else ""),
                    "blob_present":present,"blob_pixels":pixels if pixels is not None else "",
                    "blob_xyz":json.dumps(xyz,separators=(",",":")) if xyz is not None else "",
                    "depth_support_count":depth if depth is not None else "",
                    "perception_unknown_reason":(per.get("unknown_reasons") or {}).get(obj,""),
                    "OnTable":truth(facts.get("p:OnTable:"+obj)),"Held":truth(facts.get("p:Held:"+obj)),
                    "Inside_receiver":truth(facts.get("p:Inside:"+obj+":receiver")),
                    "AtBuffer_u":truth(facts.get("p:AtBuffer:"+obj+":pad_u")),"AtBuffer_v":truth(facts.get("p:AtBuffer:"+obj+":pad_v")),
                    "GripperEmpty":truth(facts.get("p:GripperEmpty")),"Open_receiver":truth(facts.get("p:Open:receiver")),
                    "legal_candidate_ids":json.dumps(legal),"evaluator_success":ev.get("task_success",""),
                    "evaluator_reason":ev.get("reason",""),"planner_status":pl.get("status","")})
        if meta["expected"]=="NO_PLAN":
            npdir=next((a for a in action_dirs(bdir) if (jread(a/"planner.json",{}) or {}).get("status")=="NO_PLAN"),None)
            if npdir:
                per=last_json(npdir/"perception.json"); facts=facts_from(npdir/"facts.json")
                missing=[o for o in OBJECTS if not blob_details(per,o)[0]]
                diag.append((bid,meta,npdir.name,missing,facts))
                contacts.append({"branch_id":bid,"layout":meta["layout"],"context":meta["context"],"action":npdir.name,
                    "image":str(frame_path(npdir,per.get("frame_id")) or ""),"candidate":meta["candidate"],"objects_missing":",".join(missing)})
    with (output/"observations/object_evidence_timeline.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    with (output/"observations/diagnostic_contact_table.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["branch_id","layout","context","action","image","candidate","objects_missing"],lineterminator="\n"); w.writeheader(); w.writerows(contacts)
    lines=["# Missing blob diagnosis","","All conclusions use saved public RGB-D/perception/facts only.",""]
    for bid,meta,act,missing,facts in diag:
        lines += ["## %s (%s/%s)"%(bid,meta["layout"],meta["candidate"]),"action="+act,"",
            "carrier at %s: %s -> GripperEmpty: %s -> missing blobs: %s -> OnTable: %s -> PICK precondition is not TRUE; saved candidate mask is a planning barrier."%
            (meta["candidate"],facts.get("p:AtBuffer:carrier:"+meta["candidate"],"UNKNOWN"),facts.get("p:GripperEmpty","UNKNOWN"),
             ",".join(missing) or "none",",".join("%s=%s"%(o,facts.get("p:OnTable:"+o,"UNKNOWN")) for o in missing)),"",
            "Root-cause label: UNRESOLVED_FROM_SAVED_FRAMES.","Missing blob does not prove occlusion or physical motion.",""]
    (output/"observations/missing_blob_diagnosis.md").write_text("\n".join(lines),encoding="utf-8")

def validate_planner_record_semantics(action,evaluator,planner,later_events=None):
    reason=(evaluator or {}).get("reason"); success=bool((evaluator or {}).get("task_success")); terminated=bool((evaluator or {}).get("terminated"))
    status=(planner or {}).get("status"); present=bool(planner)
    if (action or {}).get("stage")=="setup" and not present:
        return {"valid":True,"classification":"SETUP_PREFIX_NO_PLANNER","planner_required":False,"planner_present":False,"reason":"setup prefix did not call search"}
    if reason=="CONTINUE" and not success and not terminated:
        return {"valid":present and status in {"PLAN_FOUND","NO_PLAN","SEARCH_TIMEOUT","GOAL_ALREADY_SATISFIED"},
                "classification":"CONTINUE_PLANNER_REQUIRED","planner_required":True,"planner_present":present,"reason":"planner record retained after evaluator CONTINUE"}
    if success or terminated or reason in {"TASK_SUCCESS","TERMINATED"}:
        actual=present and status not in {None,"","N/A","NOT_CALLED"}
        return {"valid":not actual,"classification":"SUCCESS_NO_PLANNER_EXPECTED","planner_required":False,"planner_present":present,"reason":"success/termination must stop before a real planner call"}
    return {"valid":False,"classification":"UNKNOWN","planner_required":None,"planner_present":present,"reason":"missing or ambiguous evaluator control flow"}

def inspect_protocol(root,source,output):
    rows=[]
    for bid in BRANCHES:
        rec=receipt(source,bid)
        for adir in action_dirs(branch_dir(source,bid)):
            m=re.search(r"(\d+)$",adir.name); idx=int(m.group(1)) if m else -1
            ev=jread(adir/"evaluator.json",{}) or {}; pl=jread(adir/"planner.json",{}) or {}
            rows.append({"branch_id":bid,"action_index":idx,"action_id":action_id(rec,idx),"evaluator_reason":ev.get("reason",""),
                "evaluator_success":ev.get("task_success",""),"planner_status":pl.get("status",""),
                **validate_planner_record_semantics(action_record(rec,idx),ev,pl,[])})
    with (output/"protocol/planner_record_review.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    jwrite(output/"protocol/technical_gate_reviewed.json",{"status":"FAIL","technical_success_gate":"FAIL",
        "remaining_branches_released":False,"planner_stop_semantics":"CONTINUE followed by NO_PLAN retains planner evidence",
        "terminal_planner_unexpectedly_present_removed":True,"not_task_success":True,"skill_sequence":"REVIEW_REQUIRED"})
    jwrite(output/"protocol/checker_amendment.json",{"amendment":"last action number does not decide whether planner evidence is required",
        "old_gate_overwritten":False,"reviewed_gate":"FAIL","remaining_released":False})
    comps=[]
    for a,b in [("599795abeaf80f6f","9ac693fc9b60d62c"),("40517e4bb6a02609","3aaf043a49195ee7")]:
        pa=branch_dir(source,a)/"initial_rgb.png"; pb=branch_dir(source,b)/"initial_rgb.png"
        ba=pa.read_bytes() if pa.exists() else b""; bb=pb.read_bytes() if pb.exists() else b""
        ia,ib=image_info(pa),image_info(pb); same=ia.get("pixels_sha256")==ib.get("pixels_sha256") and ia.get("pixels_sha256") is not None
        comps.append({"layout":BRANCHES[a]["layout"],"branches":[a,b],"byte_sha256":[sha256_file(pa),sha256_file(pb)],
            "decoded":[ia,ib],"byte_equal":ba==bb,"decoded_pixel_equal":same,
            "classification":"ENCODING_ONLY_DIFFERENCE" if same and ba!=bb else ("PIXEL_CONTENT_DIFFERENCE" if not same else "PIXEL_CONTENT_EQUAL"),
            "public_facts_qpos_qvel_equal_does_not_imply_rgb_equal":True})
    jwrite(output/"protocol/initial_image_identity.json",{"comparisons":comps,"threshold_widening":False})
    (output/"protocol/verification_scope_comparison.md").write_text("# Verification scope comparison\n\n"
        "- verify_artifacts.json: file/budget artifact scope; retained as an input.\n"
        "- physical/mechanism_gate.json and final summary: scientific/physical scope; retained separately.\n"
        "- Derived review keeps file/budget integrity separate from scientific status.\n"
        "- Technical success gate: FAIL; physical mechanism: UNRESOLVED; method incremental value: NOT_TESTED.\n",encoding="utf-8")

def replay_public(root,source,output):
    jwrite(output/"replay/replay_result.json",{"status":"INSUFFICIENT_FOR_REPLAY","mode":"OFFLINE_REPLAY_ONLY","coverage":list(BRANCHES),
        "new_environment_constructions":0,"new_resets":0,"new_skill_calls":0,"new_provider_calls":0,
        "reason":"Saved frames establish missing public blobs but no uniform repair rule; no repaired verifier output asserted.",
        "baseline_reproduction":"NOT_ESTABLISHED"})
    (output/"replay/proposed_observation_fix.md").write_text("# Proposed observation fix\n\n"
        "Status: NEEDS_OBSERVATION_CONFIGURATION_REVIEW.\n\nSaved frames do not support a uniform segmentation/depth repair for all four branches. "
        "Do not fill dynamic facts from prior frames, nominal effects, hidden QA, or initialization coordinates.\n",encoding="utf-8")

def assemble(root,source,output):
    jwrite(output/"decision/next_action.json",{"review_scope":"SAVED_EVIDENCE_ONLY","original_technical_gate":"FAIL","reviewed_technical_gate":"FAIL",
        "original_physical_mechanism":"UNRESOLVED","new_physical_attempts":0,"new_resets":0,"new_skill_calls":0,
        "remaining_20_released":False,"provider_authorized_now":False,"s2_authorized":False,"s3_authorized":False,
        "tp_training_authorized":False,"observation_fix_readiness":"NEEDS_OBSERVATION_CONFIGURATION_REVIEW"})
    (output/"final_summary.md").write_text("# Family B observation review\n\n"
        "- Scope: SAVED_EVIDENCE_ONLY; technical gates remain FAIL.\n- Physical mechanism: UNRESOLVED; no remaining branches released.\n"
        "- The two pad_v branches retain their terminal NO_PLAN planner records.\n- Missing blobs explain the public observation-to-fact barrier but do not prove occlusion or physical movement.\n"
        "- Counts preserved: 4 attempts / 4 resets / 20 skill calls; new attempts/resets/skill calls/providers: 0.\n"
        "- Raw captures remain server-local and are represented by manifest/hash, not committed to GitHub.\n",encoding="utf-8")

def verify(root,source,output):
    before=jread(output/"inventory/protected_before.json",{}) or {}; current={}; changed=[]
    for item in before.get("files",[]):
        p=root/item["path"]; current[item["path"]]={"size":p.stat().st_size,"sha256":sha256_file(p)} if p.is_file() else {"missing":True}
        if current[item["path"]].get("sha256")!=item.get("sha256") or current[item["path"]].get("size")!=item.get("size"):
            changed.append({"path":item["path"],"before":item,"after":current[item["path"]]})
    jwrite(output/"inventory/protected_after.json",{"files":current,"changed":changed,"protected_integrity":not changed})
    jwrite(output/"verify.json",{"protected_integrity":not changed,"protected_changed_count":len(changed),
        "raw_capture_count":(jread(output/"inventory/capture_manifest.json",{}) or {}).get("capture_count"),
        "technical_gate":"FAIL","physical_mechanism":"UNRESOLVED","remaining_20_released":False,
        "new_execution_counts":{"attempts":0,"resets":0,"skill_calls":0,"providers":0}})

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=["inventory","inspect-observation","inspect-protocol","replay-public","assemble","verify"])
    ap.add_argument("--root",required=True); ap.add_argument("--source",required=True); ap.add_argument("--output",required=True)
    ns=ap.parse_args(argv); root=Path(ns.root).resolve(); source=source_dir(root,ns.source); output=source_dir(root,ns.output); output.mkdir(parents=True,exist_ok=True)
    return {"inventory":inventory,"inspect-observation":inspect_observation,"inspect-protocol":inspect_protocol,
            "replay-public":replay_public,"assemble":assemble,"verify":verify}[ns.command](root,source,output)

if __name__=="__main__": raise SystemExit(main())
