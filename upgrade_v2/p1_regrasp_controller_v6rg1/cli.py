from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
from .util import require_new, sha256_file, write_json, write_text
from .forensics import run_forensics
from .collector import RegraspCollector, CASES
from .metrics import evaluate_tree
from .selection import select
from .report import write_final

PKG_TEMPLATES = Path("/home/xushijie/PathGraph_P1_V6RG1_Closed_Loop_Regrasp_Engineering_Agent_Package_V1.0/templates")
BASE = "07cb6d6c311e53d75ffc805e4bf1c1d61d65d460"
MAIN = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"

def _copy_templates(out: Path):
    src = PKG_TEMPLATES
    names = ["model_registry.json","family_registry.json","case_registry.json","selection_contract.json",
             "holdout_gate.json","event_contract.json","state_contract.json"]
    for n in names:
        shutil.copy2(src/n, out/n)
    proto = json.loads((src/"protocol_lock.template.json").read_text(encoding="utf-8"))
    proto["confirmation_passed"] = False
    write_json(out/"protocol_lock.json", proto)

def cmd_preflight(a):
    out = require_new(a.output_root)
    repo = Path(a.repo)
    head = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"], text=True).strip()
    files = [
        repo/"upgrade_v2/p1_reward_repair_v6/backend_adapter.py",
        repo/"upgrade_v2/p1_return_controller_v6rc1/collector.py",
        repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6rc1_return_controller_engineering_v1/05_frozen_controller_v1/controller_spec.json",
        repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/final/decision.json",
    ]
    hashes = {str(p.relative_to(repo)): sha256_file(p) for p in files if p.is_file()}
    hist=set()
    for root in (Path(a.v6rc1_raw), Path(a.v6_raw)):
        for p in Path(root).rglob("identity.json"):
            try: hist.add(json.loads(p.read_text(encoding="utf-8")).get("family_id"))
            except Exception: pass
    _copy_templates(out)
    fams=json.loads((out/"family_registry.json").read_text(encoding="utf-8"))
    new={x["family_id"] for x in fams["development"]+fams["engineering_holdout"]}
    collision=sorted(new & {x for x in hist if x is not None})
    os.environ.pop("MUJOCO_GL", None)
    import mujoco
    from upgrade_v2.p1_reward_repair_v6.backend_adapter import REL_HOLD_M, HOLD_STABLE_S
    parent={"head":head,"base":BASE,"parent_ok":head==BASE,"main":MAIN,"mujoco":mujoco.__version__,
            "REL_HOLD_M":REL_HOLD_M,"HOLD_STABLE_S":HOLD_STABLE_S,"family_collision":collision,
            "v6rc1_raw":str(a.v6rc1_raw),"confirmation_passed":False}
    write_json(out/"parent_validation.json", parent)
    write_json(out/"source_lock.json", {"base_commit":BASE,"hashes":hashes})
    write_json(out/"runner_file_hashes.json", hashes)
    write_json(out/"tests.json", {"status":"PENDING"})
    if collision:
        write_json(out/"FAMILY_OR_SEED_COLLISION_STOP.json", {"ids":collision}); return 2
    print(json.dumps({"status":"OK","head":head,"mujoco":mujoco.__version__,"collision":collision}, indent=2))
    return 0 if head==BASE else 2

def cmd_audit(a):
    rec=run_forensics(Path(a.raw_root), Path(a.output_root))
    print(json.dumps(rec, indent=2, default=str)); return 0

def cmd_collect_dev(a):
    fams=json.loads(Path(a.family_registry).read_text(encoding="utf-8"))
    cases=json.loads(Path(a.case_registry).read_text(encoding="utf-8"))
    col=RegraspCollector(Path(a.repo), a.model_id)
    rec=col.collect(fams["development"], cases["cases"], Path(a.output_root))
    print(json.dumps({"root":rec["root"],"n":len(rec["rows"]),"failed":sum(r.get("status")=="FAILED" for r in rec["rows"])}, indent=2))
    return 0

def cmd_collect_holdout(a):
    fams=json.loads(Path(a.family_registry).read_text(encoding="utf-8"))
    cases=json.loads(Path(a.case_registry).read_text(encoding="utf-8"))
    frozen=json.loads(Path(a.frozen_model).read_text(encoding="utf-8"))
    mid=frozen.get("model_id") or frozen.get("selected",{}).get("model_id")
    col=RegraspCollector(Path(a.repo), mid)
    rec=col.collect(fams["engineering_holdout"], cases["cases"], Path(a.output_root))
    print(json.dumps({"root":rec["root"],"n":len(rec["rows"]),"failed":sum(r.get("status")=="FAILED" for r in rec["rows"])}, indent=2))
    return 0

def cmd_eval_dev(a):
    rec=evaluate_tree(Path(a.data_root), Path(a.output_root), False)
    print(json.dumps({"n":len(rec["rollouts"])}, indent=2)); return 0

def cmd_eval_holdout(a):
    rec=evaluate_tree(Path(a.data_root), Path(a.output_root), True)
    c=rec["candidates"][0] if rec["candidates"] else {}
    m=c.get("metrics") or {}
    def pf(case, need):
        v=m.get(case) or (0,6); return (v[0] if isinstance(v,(list,tuple)) else 0)>=need
    ok=c.get("hard_ok") and pf("G1_LOSS_REGRASP_RETURN_P40",5) and pf("G2_LOSS_REGRASP_RETURN_P80",5) \
       and pf("G3_THREE_REGRASP_RETURN_P40",4) and pf("G4_THREE_REGRASP_RETURN_P80",4) \
       and pf("G5_HIGH_RESIDUAL_SPEED_P40",4) and pf("G6_LOW_FRICTION_LONG_SLIDE_P80",4) \
       and pf("G7_NO_LOSS_INITIAL_GRASP_CONTROL",6) and pf("G8_COMMANDED_RELEASE_NO_REGRASP",6) \
       and (c.get("all_attempt_rate_g16") or 0)>=0.90 and (c.get("cond_rc1_return") or 0)>=0.90 \
       and (c.get("p90_regrasp_s") or 9)<=3.5
    decision={"confirmation_passed":False,"status":"CLOSED_LOOP_REGRASP_READY_FOR_INDEPENDENT_CONFIRMATION" if ok else "CLOSED_LOOP_REGRASP_ENGINEERING_NOT_READY","gate_ok":ok,"metrics":c}
    write_json(Path(a.output_root)/"decision.json", decision)
    print(json.dumps(decision, indent=2, default=str)); return 0

def cmd_select(a):
    sel=select(Path(a.evaluation_root), Path(a.selection_contract), Path(a.output_root))
    print(json.dumps(sel, indent=2, default=str)); return 0 if sel else 2

def cmd_freeze(a):
    out=require_new(a.output_root)
    sel=json.loads(Path(a.selection).read_text(encoding="utf-8"))
    chosen=sel.get("selected") or {}
    mid=chosen.get("model_id")
    write_json(out/"model_spec.json", {"model_id":mid,"max_attempts":2 if mid and "ONE_RETRY" in mid else 1,"confirmation_passed":False})
    repo=Path.cwd()
    files=sorted((repo/"upgrade_v2/p1_regrasp_controller_v6rg1").rglob("*.py"))
    write_json(out/"model_source_files.json", [str(p.relative_to(repo)) for p in files if "__pycache__" not in str(p)])
    write_json(out/"model_source_sha256.json", {str(p.relative_to(repo)): sha256_file(p) for p in files if "__pycache__" not in str(p)})
    write_text(out/"model_formula.md", "p_above(t)=p_obj(t)+[0,0,0.16]; p_grasp(t)=p_obj(t)+[0,0,0.13]; 3-tick 8mm close gate.\n")
    write_json(out/"state_machine.json", {"stages":["WAIT_OBJECT_SETTLE","TRACK_ABOVE","TRACK_DESCEND","FINAL_RECENTER","CLOSE_TRACK","VERIFY_HOLD"]})
    write_json(out/"next_confirmation_requirements.json", {"confirmation_data_reuse_allowed":False,"V6_CAP_POTENTIAL_frozen":True,"RC1_frozen":True,"confirmation_passed":False})
    print(json.dumps({"frozen":mid}, indent=2)); return 0

def cmd_summarize(a):
    rec=write_final(Path(a.artifact_root), Path(a.output_root))
    print(json.dumps(rec, indent=2, default=str)); return 0

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command", required=True)
    q=sub.add_parser("preflight"); q.add_argument("--repo", required=True); q.add_argument("--v6rc1-raw", required=True); q.add_argument("--v6-raw", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("audit-v6rc1"); q.add_argument("--raw-root", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("collect-development"); q.add_argument("--repo", required=True); q.add_argument("--model-id", required=True); q.add_argument("--family-registry", required=True); q.add_argument("--case-registry", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("evaluate-development"); q.add_argument("--data-root", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("select"); q.add_argument("--evaluation-root", required=True); q.add_argument("--selection-contract", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("freeze-model"); q.add_argument("--selection", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("collect-holdout"); q.add_argument("--repo", required=True); q.add_argument("--frozen-model", required=True); q.add_argument("--family-registry", required=True); q.add_argument("--case-registry", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("evaluate-holdout"); q.add_argument("--data-root", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("summarize"); q.add_argument("--artifact-root", required=True); q.add_argument("--output-root", required=True)
    a=p.parse_args(argv)
    return {"preflight":cmd_preflight,"audit-v6rc1":cmd_audit,"collect-development":cmd_collect_dev,
            "evaluate-development":cmd_eval_dev,"select":cmd_select,"freeze-model":cmd_freeze,
            "collect-holdout":cmd_collect_holdout,"evaluate-holdout":cmd_eval_holdout,"summarize":cmd_summarize}[a.command](a)

if __name__=="__main__":
    raise SystemExit(main())
