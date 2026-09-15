from __future__ import annotations
import argparse, json, os, shutil, hashlib, subprocess
from pathlib import Path
from .util import require_new, sha256_file, write_json, write_text
from .forensics import run_forensics
from .collector import ReturnCollector, CASES
from .metrics import evaluate_tree
from .selection import select
from .report import write_final

PKG_TEMPLATES = Path("/home/xushijie/PathGraph_P1_V6RC1_Return_Controller_Engineering_Agent_Package_V1.0/templates")
BASE = "b05ab87cbeea5c798a0fea830a6381cb089640d1"
V6_RUNNER = "cf5ba1ed81bb69c60df67a23714a2e5e7575e142"
MAIN = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"
CANDIDATES = [
    {"id": "RC1_OBJECT_COMPENSATED_DIRECT", "complexity_rank": 1, "max_speed_m_s": 0.45, "path": "direct"},
    {"id": "RC2_OBJECT_COMPENSATED_STAGED", "complexity_rank": 2, "coarse_speed_m_s": 0.45, "fine_speed_m_s": 0.12, "safe_clearance_m": 0.08},
    {"id": "RC3_OBJECT_ERROR_SERVO", "complexity_rank": 3, "kp_per_s": 3.0, "coarse_speed_m_s": 0.45, "fine_speed_m_s": 0.12},
]

def _copy_templates(out: Path):
    src = PKG_TEMPLATES
    if not src.is_dir():
        # local fallback during package verify
        here = Path(__file__).resolve().parents[3]
        alt = here / "p1_v6rc1_package" / "PathGraph_P1_V6RC1_Return_Controller_Engineering_Agent_Package_V1.0" / "templates"
        src = alt if alt.is_dir() else src
    for name in ("controller_candidate_registry.json","family_registry.json","case_registry.json",
                 "closure_contract.json","selection_contract.json","holdout_gate.json"):
        shutil.copy2(src/name, out/name)
    proto = json.loads((src/"protocol_lock.template.json").read_text(encoding="utf-8"))
    proto["confirmation_passed"] = False
    write_json(out/"protocol_lock.json", proto)

def cmd_preflight(a):
    out = require_new(a.output_root)
    repo = Path(a.repo)
    head = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"], text=True).strip()
    parent_ok = head == BASE
    v6_files = [
        repo/"upgrade_v2/p1_reward_repair_v6/backend_adapter.py",
        repo/"upgrade_v2/p1_reward_repair_v6/collector.py",
        repo/"upgrade_v2/p1_reward_repair_v6/confirm_analysis.py",
        repo/"upgrade_v2/p1_reward_repair_v6/macro_cycle_windows.py",
        repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/final/decision.json",
        repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/v6_three_issue_v1/04_frozen_candidate_v1/candidate_lock.json",
    ]
    hashes = {str(p.relative_to(repo)): sha256_file(p) for p in v6_files if p.is_file()}
    v6_dec = json.loads(v6_files[4].read_text(encoding="utf-8")) if v6_files[4].is_file() else {}
    fam_v6=set()
    for root in (Path(a.v6_confirm), Path(a.v6_engineering)):
        for p in Path(root).rglob("identity.json"):
            try:
                fam_v6.add(json.loads(p.read_text(encoding="utf-8")).get("family_id"))
            except Exception:
                pass
    _copy_templates(out)
    fams = json.loads((out/"family_registry.json").read_text(encoding="utf-8"))
    new_ids={x["family_id"] for x in fams["development"]+fams["engineering_holdout"]}
    collision = sorted(new_ids & {x for x in fam_v6 if x is not None})
    mujoco_ver=None
    try:
        os.environ.pop("MUJOCO_GL", None)
        import mujoco
        mujoco_ver=mujoco.__version__
    except Exception as exc:
        mujoco_ver=f"UNAVAILABLE:{exc}"
    parent = {
        "head": head, "base": BASE, "parent_ok": parent_ok, "v6_runner": V6_RUNNER, "main": MAIN,
        "v6_confirmation_passed": v6_dec.get("confirmation_passed"),
        "backend": v6_dec.get("backend",{}).get("backend_id"),
        "physics_dt": 0.002, "control_dt": 0.020, "return_timeout_s": 2.0,
        "family_collision": collision, "mujoco": mujoco_ver,
        "v6_confirm_exists": Path(a.v6_confirm).is_dir(),
        "v6_engineering_exists": Path(a.v6_engineering).is_dir(),
    }
    write_json(out/"parent_validation.json", parent)
    write_json(out/"source_lock.json", {"base_commit": BASE, "hashes": hashes, "read_only": [
        "upgrade_v2/p1_reward_repair_v6/", "tools/stage5/lib/"]})
    write_json(out/"runner_file_hashes.json", hashes)
    write_json(out/"tests.json", {"status": "PENDING_LOCAL_PYTEST"})
    if collision:
        write_json(out/"FAMILY_ID_COLLISION_STOP.json", {"ids": collision})
        print(json.dumps({"status":"FAMILY_ID_COLLISION_STOP","ids":collision}))
        return 2
    print(json.dumps({"status":"OK","head":head,"mujoco":mujoco_ver,"parent_ok":parent_ok}, indent=2))
    return 0 if parent_ok else 2

def cmd_audit(a):
    rec = run_forensics(Path(a.confirmation_root), Path(a.engineering_root), Path(a.output_root))
    print(json.dumps(rec, indent=2, default=str))
    return 0

def _spec(cid):
    for c in CANDIDATES:
        if c["id"]==cid:
            return dict(c)
    raise ValueError(cid)

def cmd_collect_dev(a):
    fams=json.loads(Path(a.family_registry).read_text(encoding="utf-8"))
    cases=json.loads(Path(a.case_registry).read_text(encoding="utf-8"))
    col=ReturnCollector(Path(a.repo), a.controller_id, _spec(a.controller_id))
    rec=col.collect(fams["development"], cases["cases"], Path(a.output_root))
    print(json.dumps({"root": rec["root"], "n": len(rec["rows"]),
                      "failed": sum(r.get("status")=="FAILED" for r in rec["rows"])}, indent=2))
    return 0

def cmd_collect_holdout(a):
    fams=json.loads(Path(a.family_registry).read_text(encoding="utf-8"))
    cases=json.loads(Path(a.case_registry).read_text(encoding="utf-8"))
    frozen=json.loads(Path(a.frozen_controller).read_text(encoding="utf-8"))
    cid=frozen.get("controller_id") or frozen.get("selected",{}).get("controller_id")
    spec=frozen.get("spec") or _spec(cid)
    col=ReturnCollector(Path(a.repo), cid, spec)
    rec=col.collect(fams["engineering_holdout"], cases["cases"], Path(a.output_root))
    print(json.dumps({"root": rec["root"], "n": len(rec["rows"]),
                      "failed": sum(r.get("status")=="FAILED" for r in rec["rows"])}, indent=2))
    return 0

def cmd_eval_dev(a):
    rec=evaluate_tree(Path(a.data_root), Path(a.output_root), holdout=False)
    print(json.dumps({"n": rec["n"] if False else len(rec["rollouts"])}, indent=2))
    return 0

def cmd_eval_holdout(a):
    rec=evaluate_tree(Path(a.data_root), Path(a.output_root), holdout=True)
    # gate
    gate=json.loads((Path(a.output_root).parent.parent/"00_static_v1"/"holdout_gate.json").read_text()) if False else {
        "requirements": {"E1": 3, "E2": 3}
    }
    cands=rec["candidates"]
    decision={"confirmation_passed": False, "status": "RETURN_CONTROLLER_ENGINEERING_NOT_READY"}
    if cands:
        c=cands[0]
        mets=c.get("metrics") or {}
        def pf(case, need, total=4):
            v=mets.get(case) or (0,4)
            ok=v[0] if isinstance(v,(list,tuple)) else 0
            return ok>=need
        ok = c.get("hard_ok") and pf("E1_LOSS_RETURN_P40",3) and pf("E2_LOSS_RETURN_P80",3) \
             and pf("E3_THREE_RETURNS_P40",3) and pf("E4_THREE_RETURNS_P80",3) \
             and pf("E5_REGRASP_OFFSET_LEFT_P40",3) and pf("E6_REGRASP_OFFSET_RIGHT_P40",3) \
             and pf("E7_NO_LOSS_OUT_AND_BACK",4) and pf("E8_COMMANDED_RELEASE_CONTROL",4)
        med=c.get("median_return_s"); p90=c.get("p90_return_s")
        ok = ok and (med is not None and med<=1.5) and (p90 is not None and p90<=2.0)
        decision["status"] = "RETURN_CONTROLLER_READY_FOR_INDEPENDENT_CONFIRMATION" if ok else "RETURN_CONTROLLER_ENGINEERING_NOT_READY"
        decision["metrics"]=c
        decision["gate_ok"]=ok
    from .util import write_json as wj
    wj(Path(a.output_root)/"decision.json", decision)
    print(json.dumps(decision, indent=2, default=str))
    return 0

def cmd_select(a):
    sel=select(Path(a.evaluation_root), Path(a.selection_contract), Path(a.output_root))
    print(json.dumps(sel, indent=2, default=str))
    return 0 if sel else 2

def cmd_freeze(a):
    out=require_new(a.output_root)
    sel=json.loads(Path(a.selection).read_text(encoding="utf-8"))
    chosen=sel.get("selected") or {}
    cid=chosen.get("controller_id")
    spec=_spec(cid) if cid else {}
    write_json(out/"controller_spec.json", {"controller_id": cid, "spec": spec, "compensation_mode": "TRANSLATION_COMPENSATED_FIXED_ORIENTATION", "confirmation_passed": False})
    repo=Path.cwd()
    files=sorted((repo/"upgrade_v2/p1_return_controller_v6rc1").rglob("*.py"))
    write_json(out/"controller_source_files.json", [str(p.relative_to(repo)) for p in files if "__pycache__" not in str(p)])
    write_json(out/"controller_source_sha256.json", {str(p.relative_to(repo)): sha256_file(p) for p in files if "__pycache__" not in str(p)})
    write_text(out/"controller_formula.md", "p_WE_des = p_WO_star - R_WE * r_EO_regrasp\nMode: TRANSLATION_COMPENSATED_FIXED_ORIENTATION\n")
    write_json(out/"state_machine.json", {"stages": ["WAIT_REHOLD_STABLE","ESTIMATE_ATTACHMENT","PLAN_RETURN","SAFE_CLEARANCE","XY_ALIGN","Z_ALIGN","FINE_SERVO","SETTLE_AND_VERIFY"]})
    shutil.copy2(Path(a.selection).resolve().parents[1]/"00_static_v1"/"closure_contract.json", out/"closure_contract.json") if False else write_json(out/"closure_contract.json", json.loads((PKG_TEMPLATES/"closure_contract.json").read_text(encoding="utf-8")) if PKG_TEMPLATES.is_dir() else {"return_timeout_s": 2.0})
    write_json(out/"next_confirmation_requirements.json", {
        "confirmation_data_reuse_allowed": False, "V6_CAP_POTENTIAL_frozen": True,
        "future_families_must_be_new": True, "cases": ["P40","P80","triple","left/right offset","no-loss","release"],
        "confirmation_passed": False,
    })
    print(json.dumps({"frozen": cid}, indent=2))
    return 0

def cmd_summarize(a):
    rec=write_final(Path(a.artifact_root), Path(a.output_root))
    print(json.dumps(rec, indent=2, default=str))
    return 0

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command", required=True)
    q=sub.add_parser("preflight"); q.add_argument("--repo", required=True); q.add_argument("--v6-confirm", required=True); q.add_argument("--v6-engineering", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("audit-v6"); q.add_argument("--confirmation-root", required=True); q.add_argument("--engineering-root", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("collect-development"); q.add_argument("--repo", required=True); q.add_argument("--controller-id", required=True); q.add_argument("--family-registry", required=True); q.add_argument("--case-registry", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("evaluate-development"); q.add_argument("--data-root", required=True); q.add_argument("--closure-contract", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("select"); q.add_argument("--evaluation-root", required=True); q.add_argument("--selection-contract", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("freeze-controller"); q.add_argument("--selection", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("collect-holdout"); q.add_argument("--repo", required=True); q.add_argument("--frozen-controller", required=True); q.add_argument("--family-registry", required=True); q.add_argument("--case-registry", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("evaluate-holdout"); q.add_argument("--data-root", required=True); q.add_argument("--closure-contract", required=True); q.add_argument("--output-root", required=True)
    q=sub.add_parser("summarize"); q.add_argument("--artifact-root", required=True); q.add_argument("--output-root", required=True)
    a=p.parse_args(argv)
    return {
        "preflight": cmd_preflight, "audit-v6": cmd_audit, "collect-development": cmd_collect_dev,
        "evaluate-development": cmd_eval_dev, "select": cmd_select, "freeze-controller": cmd_freeze,
        "collect-holdout": cmd_collect_holdout, "evaluate-holdout": cmd_eval_holdout, "summarize": cmd_summarize,
    }[a.command](a)

if __name__=="__main__":
    raise SystemExit(main())
