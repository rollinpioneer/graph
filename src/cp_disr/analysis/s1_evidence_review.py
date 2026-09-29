from __future__ import annotations
import csv, hashlib, inspect, json, os, subprocess
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def rows(p): return list(csv.DictReader(Path(p).open(newline="",encoding="utf-8")))
def cwrite(p,fields,rs):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rs)
def locations(root):
    import cp_disr.analysis.s1_revision as r; out={}
    for n in ("execute_registered_branch","run_witnesses","register_physical_branches","probe_production_representation","offline_rescore_planner","finalize_eligibility"):
        o=getattr(r,n,None)
        try: out[n]={"module":o.__module__,"file":inspect.getsourcefile(o),"line":inspect.getsourcelines(o)[1],"bound_to_resume":o.__module__.endswith("s1_revision_resume")}
        except Exception as e: out[n]={"error":type(e).__name__}
    return out
def inspect_runner(root,source,out):
    d=Path(out)/"audit/runner"; d.mkdir(parents=True,exist_ok=True); reg=json.loads((Path(source)/"witnesses/e4_branch_registration.json").read_text()); rs=rows(Path(source)/"witnesses/e4_physical_witnesses_rev1.csv"); jwrite(d/"source_locations.json",locations(root)); bm={x["branch_id"]:x for x in reg["branches"]}
    rr=[]
    for r in rs:
        b=bm.get(r.get("branch_id"),{}); rr.append({"branch_id":r.get("branch_id",""),"case_id":r.get("case_id",""),"candidate_id":r.get("candidate_id",""),"original_status":r.get("status",""),"controller_exit":r.get("controller_exit",""),"normal_controller_exit":r.get("controller_exit")=="NORMAL_TERMINATION","continuation_plan_found":r.get("continuation_status")=="PLAN_FOUND","continuation_execution_evidence_ref":"","continuation_executed":False,"independent_evaluator_evidence_ref":"","full_episode_completed":False,"initial_snapshot_hash_verified":False,"paired_rng_verified":False,"primary_outcome_available":False,"reviewed_classification":"INCOMPLETE_CONTINUATION","reason":"first action and plan query only; no continuation/evaluator endpoint","registered_restore_seed":b.get("restore_seed","")})
    cwrite(d/"witness_reclassification.csv",rr[0],rr); aa=[{"branch_id":b["branch_id"],"case_id":b["case_id"],"candidate_id":b["candidate_id"],"repeat":b.get("repeat"),"registered_restore_seed":b.get("restore_seed"),"seed_includes_candidate_id":True,"restore_seed_consumed_by_executor":False,"snapshot_hash_present":False,"paired_seed_verified":False,"reviewed_status":"NOT_VERIFIED","reason":"candidate-derived seed and no restore consumption"} for b in reg["branches"]]; cwrite(d/"registration_audit.csv",aa[0],aa)
    inv=[{"path":str(p.relative_to(source)),"sha256":sha(p),"exists":True} for p in sorted((Path(source)/"witnesses").glob("*")) if p.is_file()]; cwrite(d/"raw_evidence_inventory.csv",inv[0],inv)
    counts={"rows":len(rs),"NO_PLAN":sum(x.get("continuation_status")=="NO_PLAN" for x in rs),"PLAN_FOUND":sum(x.get("continuation_status")=="PLAN_FOUND" for x in rs)}; (d/"runner_root_cause.md").write_text("# Runner Root Cause\n\nNORMAL_TERMINATION is controller state, not task success. PLAN_FOUND was queried but not consumed; no complete evaluator endpoint or paired restore evidence exists.\n\n"+json.dumps(counts,sort_keys=True)+"\n",encoding="utf-8"); return counts
def inspect_gates(root,source,out):
    d=Path(out)/"audit/gates"; d.mkdir(parents=True,exist_ok=True); gs=[]
    for gate,cond,need,miss in [("E1","nonempty admitted cache rows","independent relation adjudication","source/effect/scene truth"),("E2","PASS representation rows","actual ID/mask/goal/node alignment","comparison evidence"),("E3","common_shift_only field","separate gradients and capacity evidence","gradient provenance"),("E4","DIAGNOSTIC rows","complete paired continuations and outcomes","all continuation/evaluator/restore evidence"),("E5","bool(natural)","ranking versus reliable outcomes","controlled consequence comparison"),("E6","bool(provider)","quality and utility evidence","truth/utility/opportunity labels")]:
        gs.append({"gate":gate,"reported_status":"PASS" if gate!="E4" else "FAIL","actual_condition_in_code":cond,"required_evidence":need,"existing_evidence_refs":"source evidence","missing_evidence":miss,"reviewed_status":"NOT_ESTABLISHED"})
    cwrite(d/"gate_evidence_matrix.csv",gs[0],gs); return {x["gate"]:x["reviewed_status"] for x in gs}
def inspect_probe(root,source,out):
    d=Path(out)/"audit/probe"; d.mkdir(parents=True,exist_ok=True); text="\n".join(p.read_text(encoding="utf-8") for p in [Path(root)/"src/cp_disr/analysis/s1_revision.py",Path(root)/"src/cp_disr/analysis/s1_revision_resume.py"]); f=[{"check":"alignment","found":"candidate_id_alignment" in text,"status":"REQUIRES_REPAIR"},{"check":"separate_gradients","found":"gradient_to_patch_input" in text,"status":"REQUIRES_REPAIR"},{"check":"offline_runtime","found":"load_runtime(manifest)" in text,"status":"REQUIRES_REPAIR"},{"check":"broad_exception","found":"except Exception" in text,"status":"REQUIRES_REPAIR"}]; (d/"probe_and_rescore_audit.md").write_text("# Probe and Rescore Audit\n\nNo live execution was performed.\n\n"+json.dumps(f,indent=2)+"\n",encoding="utf-8"); return f
def summarize(root,source,out):
    led=json.loads((Path(source)/"budget_ledger.json").read_text()); result={"source_reported_decision":"NOT_ELIGIBLE_AFTER_SINGLE_REVISION","review_status":"ENGINEERING_DEFECTS_IDENTIFIED","scientific_result":"NOT_ESTABLISHED","original_source_revision_index":1,"new_source_revision_performed":False,"tp_training_authorized":False,"method_upgrade_authorized":False,"new_physical_episodes_authorized":0,"next_action":"S4_RESEARCH_DECISION_ON_REPAIR_AND_EVALUATION_BUDGET"}; jwrite(Path(out)/"reviewed_eligibility.json",result); jwrite(Path(out)/"budget_after.json",led); jwrite(Path(out)/"scheduling_report.json",{"workers_requested":3,"workers_used":3,"scientific_samples_consumed":0,"reduced_concurrency":False}); jwrite(Path(out)/"proposed/e4_recovery_budget_request.json",{"purpose":"REPAIR_INCOMPLETE_E4_PROTOCOL","source_revision_index":1,"new_source_revision":False,"original_witness_attempts_used":8,"original_witness_cap":8,"requested_additional_physical_episodes":8,"approved_additional_physical_episodes":0,"projected_total_if_approved":16,"new_provider_calls_requested":0,"rl_attempts_requested":0,"elastic_rl_slots_requested":0,"requires_explicit_user_budget_amendment":True,"execute_now":False,"status":"NOT_READY"}); jwrite(Path(out)/"review_manifest.json",{"status":"COMPLETE","source":str(source),"new_provider_calls":0,"new_physical_episodes":0,"new_revision":False}); (Path(out)/"reviewed_summary.md").write_text("# S1-REV1 Engineering Repair Review\n\nOriginal final_summary is retained as STALE_SOURCE_SUMMARY. E4 is NOT_ESTABLISHED; no new episodes are authorized.\n",encoding="utf-8"); return result
def verify(root,source,out):
    before=json.loads((Path(out)/"protected_source_files.json").read_text()); after={}
    for rel,h in before.items():
        p=Path(root)/rel
        if not p.is_file() or sha(p)!=h: raise RuntimeError("PROTECTED_FILE_CHANGED:"+rel)
        after[rel]=h
    if json.loads((Path(out)/"budget_before.json").read_text())!=json.loads((Path(source)/"budget_ledger.json").read_text()): raise RuntimeError("BUDGET_CHANGED")
    jwrite(Path(out)/"protected_source_files_after.json",after); return {"status":"PASS","protected_files":len(after),"budget_unchanged":True,"training_authorized":False}


def evaluate_gate_evidence(evidence_bundle: dict) -> dict:
    """Pure gate adjudication. Labels alone never satisfy a gate."""
    out = {}
    relation = evidence_bundle.get("relation_adjudications", [])
    align = evidence_bundle.get("alignment_records", [])
    gradients = evidence_bundle.get("gradient_records", [])
    witnesses = evidence_bundle.get("witness_records", [])
    consequences = evidence_bundle.get("consequence_comparisons", [])
    utility = evidence_bundle.get("utility_records", [])
    def gate(name, satisfied, missing, refs):
        out[name] = {
            "status": "PASS" if satisfied else "NOT_ESTABLISHED",
            "satisfied": bool(satisfied),
            "missing": list(missing),
            "evidence_refs": list(refs),
            "source_hashes": evidence_bundle.get("source_hashes", {}),
        }
    independent = [x for x in relation if x.get("independent_adjudication") is True and x.get("truth_status") == "ADJUDICATED"]
    gate("E1", len({x.get("case_id") for x in independent}) >= 2,
         [] if len({x.get("case_id") for x in independent}) >= 2 else ["independent relation truth adjudication for two configurations"],
         [x.get("evidence_ref","") for x in independent])
    aligned = [x for x in align if x.get("actual_comparison") is True and all(x.get(k) is True for k in ("candidate_id_alignment","mask_alignment","goal_alignment","node_alignment"))]
    gate("E2", len({x.get("case_id") for x in aligned}) >= 2,
         [] if len({x.get("case_id") for x in aligned}) >= 2 else ["actual ID/mask/goal/node comparisons"],
         [x.get("evidence_ref","") for x in aligned])
    grads = [x for x in gradients if x.get("relation_gradient_measured") is True and x.get("patch_gradient_measured") is True and x.get("separate_sources") is True]
    gate("E3", len(grads) >= 2 and any(x.get("relative_response") is not None for x in grads),
         [] if len(grads) >= 2 else ["separate gradient provenance and relative response"],
         [x.get("evidence_ref","") for x in grads])
    full = [x for x in witnesses if x.get("protocol_complete") is True and x.get("full_episode_completed") is True and x.get("independent_evaluator") is True and x.get("paired_restore_verified") is True and x.get("eligible_for_e4") is True]
    cases = {x.get("case_id") for x in full}
    effects = [x for x in consequences if x.get("reliable") is True and x.get("outcome_different") is True]
    gate("E4", len(cases) >= 2 and len(effects) >= 1,
         [] if len(cases) >= 2 and effects else ["complete paired continuation, restore, evaluator endpoint, and outcome difference"],
         [x.get("evidence_ref","") for x in full + effects])
    e5 = [x for x in consequences if x.get("reliable") is True and x.get("contract_ranking_comparable") is True and x.get("outcome_different") is True]
    gate("E5", bool(e5), [] if e5 else ["controlled contract ranking versus reliable real outcome"], [x.get("evidence_ref","") for x in e5])
    e6 = [x for x in utility if x.get("truth_adjudicated") is True and x.get("utility_adjudicated") is True and x.get("opportunity_classified") is True and x.get("synthetic_unit_fixture") is not True and x.get("natural") is not False and x.get("provider") is not False]
    gate("E6", bool(e6), [] if e6 else ["independent truth, utility, and opportunity evidence"], [x.get("evidence_ref","") for x in e6])
    return out
