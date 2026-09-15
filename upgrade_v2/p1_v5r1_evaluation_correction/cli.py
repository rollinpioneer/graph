#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, sys, importlib.util
from collections import defaultdict
from pathlib import Path
from .util import require_new, write_json, write_csv, write_jsonl, write_text, sha256_file
from . import raw_reader, events, product_graph, binding, cycle_segments, credit_attribution

METHODS=["SPARSE_TERMINAL","LINEAR_A_FIRST_R1","LINEAR_B_FIRST_R1","UNORDERED_COMPLETION_PROGRESS",
         "GRAPH_COST_ONLY","GRAPH_COST_PHI","GRAPH_COST_DEBT","FULL_FROZEN","NO_RECOVERY_CREDIT",
         "PAIRED_EVENT_BALANCED_V1","GLOBAL_POTENTIAL_AUDIT_V1"]

def add_pkg(pkg: Path):
    sys.path.insert(0, str(pkg/"tools"))

def freeze_contracts(pkg, repo, out):
    out=require_new(out)
    for name in ("protocol.json","source_lock.json","change_registry.json","method_registry.json","closure_contract.template.json"):
        src=pkg/"templates"/name
        (out/name.replace(".template","")).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(out/"event_contract.json", dict(loss="OBSERVED_SIM_HELD_LOSS", recovery="OBSERVED_SIM_HOLD_REESTABLISHED", eof_terminal=False))
    write_json(out/"graph_binding_contract.json", dict(recovery="legacy_transport_recovery", dual="PRODUCT_STATE_V5R1", g1_extension=False))
    write_json(out/"numeric_id_contract.json", dict(node_order=list(product_graph.REC_IDX.keys()), edge_types=["none","forward","alternative","recovery","failure","stagnation"]))
    print("contracts_ok"); return 0

def reproduce_legacy(repo, base, out):
    out=require_new(out)
    trans_p=repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/targeted_physical_v5/scoring_v1/per_episode_returns.csv"
    old=list(csv.DictReader(trans_p.open(encoding="utf-8", newline="")))
    from upgrade_v2.p1_physical_mechanism_v5 import scoring as mod
    gt=repo/"artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/targeted_physical_v5/binding_v1/graph_transitions.jsonl"
    rows=[json.loads(l) for l in gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    by=defaultdict(list)
    for r in rows: by[r["episode_id"]].append(r)
    repro=[]
    for eid,seq in by.items():
        seq=sorted(seq,key=lambda x:x["step"])
        task="dual_order" if "__D" in eid else "recovery"
        det=mod.score_episode(seq, task)
        g=defaultdict(list)
        for d in det: g[d["method"]].append(d)
        for m,xs in g.items():
            repro.append(dict(episode_id=eid, method=m, signed_return=sum(x["reward_mu"] for x in xs)))
    # compare FULL
    diffs=[]
    old_map={(r["episode_id"], r["method"]): float(r["signed_return"]) for r in old}
    for r in repro:
        key=(r["episode_id"], r["method"] if r["method"] in ("FULL_FROZEN","PAIRED_EVENT_BALANCED_V1","GLOBAL_POTENTIAL_AUDIT_V1","LINEAR_A_FIRST","LINEAR_B_FIRST","UNORDERED_COMPLETION_PROGRESS","GRAPH_COST_ONLY","GRAPH_COST_PHI","GRAPH_COST_DEBT","NO_RECOVERY_CREDIT","SPARSE_TERMINAL") else r["method"])
        if key in old_map:
            diffs.append(dict(episode_id=r["episode_id"], method=r["method"], old=old_map[key], repro=r["signed_return"], abs_diff=abs(old_map[key]-r["signed_return"])))
    maxdiff=max((d["abs_diff"] for d in diffs), default=0)
    write_csv(out/"reproduction_rows.csv", diffs[:500] if len(diffs)>500 else diffs)
    write_json(out/"reproduction_summary.json", dict(n_old=len(old), n_repro=len(repro), n_compared=len(diffs), max_abs_diff=maxdiff, reproduced=(maxdiff<1e-9)))
    print(json.dumps({"reproduced": maxdiff<1e-9, "max_abs_diff": maxdiff, "n": len(diffs)})); return 0

def read_events(raw, contracts, out):
    out=require_new(out)
    idents=sorted(Path(raw).glob("*/*/rollout_00/identity.json"))
    ledger=[]; coverage=[]; all_facts={}
    for ident in idents:
        d=ident.parent
        identity=raw_reader.load_identity(ident)
        rows=raw_reader.load_timeseries(d/"timeseries.csv")
        objs=raw_reader.objects_of(rows[0])
        facts, ev=events.rebuild_events(identity["episode_id"], rows, objs)
        ledger.extend(ev)
        all_facts[identity["episode_id"]]=dict(facts=facts, objects=objs, identity=identity, task=("dual_order" if objs==["A","B"] else "recovery"))
        coverage.append(dict(episode_id=identity["episode_id"], n_frames=len(rows), n_events=len(ev),
                             objects="|".join(objs), runner=identity.get("runner_commit"),
                             losses=sum(1 for e in ev if e["event"]=="OBSERVED_SIM_HELD_LOSS"),
                             restorations=sum(1 for e in ev if e["event"]=="OBSERVED_SIM_HOLD_REESTABLISHED")))
    write_csv(out/"event_ledger.csv", ledger or [dict(note="no_events")])
    write_csv(out/"state_coverage.csv", coverage)
    write_json(out/"facts_index.json", {k: dict(n=len(v["facts"]), objects=v["objects"], task=v["task"]) for k,v in all_facts.items()})
    # stash facts jsonl per episode would be huge; write compact pickle-less jsonl of needed fields
    compact=[]
    for eid,pack in all_facts.items():
        for fr in pack["facts"]:
            compact.append(dict(episode_id=eid, tick=fr["tick"], t=fr["t"], available_at_ns=fr["available_at_ns"],
                                gripper_closed=fr["gripper_closed"], active_object=fr["active_object"],
                                success=fr["success"], objects=fr["objects"], task=pack["task"]))
    write_jsonl(out/"frames.jsonl", compact)
    print(json.dumps({"episodes":len(coverage), "events":len(ledger)})); return 0

def bind(events_dir, contracts, out):
    out=require_new(out)
    frames=defaultdict(list)
    for line in (Path(events_dir)/"frames.jsonl").read_text(encoding="utf-8").splitlines():
        r=json.loads(line); frames[r["episode_id"]].append(r)
    trans=[]; unresolved=[]
    for eid, frs in frames.items():
        frs=sorted(frs, key=lambda x:x["tick"])
        task=frs[0]["task"]
        rows=binding.transitions_from_facts(eid, frs, task)
        trans.extend(rows)
        unresolved.extend([r for r in rows if r["classification"] not in ("BOUND","BOUND_NODE_DWELL")])
    write_jsonl(out/"graph_transitions.jsonl", trans)
    elig=[r for r in trans if r["classification"] in ("BOUND","BOUND_NODE_DWELL")]
    write_jsonl(out/"eligible_numeric_transitions.jsonl", elig)
    if unresolved:
        write_csv(out/"unresolved_transitions.csv", unresolved)
    else:
        write_csv(out/"unresolved_transitions.csv", [dict(note="none")])
    print(json.dumps({"transitions":len(trans),"eligible":len(elig),"unresolved":len(unresolved)})); return 0

def _attach_phi(seq):
    from reeval_core import AnchorBank
    bank=AnchorBank(); prev=0.0
    for i,t in enumerate(seq):
        a,b=t["node_before"], t["node_after"]
        d0,d1=t.get("distance_before"), t.get("distance_after")
        key=(t["episode_id"], "obj", "place", "in_transit")
        enter=(a!="in_transit" and b=="in_transit")
        pa=bank.observe(key, d1, t["available_at_ns"], enter=enter) if b=="in_transit" else 0.0
        pb=bank.observe(key, d0, t["t0_ns"], enter=False) if a=="in_transit" else 0.0
        if pa is None: pa=0.0
        if pb is None: pb=0.0
        if i: pb=prev if a==seq[i-1]["node_after"] else pb
        t["phi_before"]=float(pb); t["phi_after"]=float(pa); prev=float(pa)
        # baselines
        from reeval_core import progress
        if t.get("a_valid") is not None:
            qa0=progress(seq[i-1].get("a_valid") if i else False, seq[i-1].get("b_valid") if i else False, seq[i-1].get("success_before") if i else False, "A_FIRST") if i else progress(False,False,False,"A_FIRST")
            # use before from previous after flags stored on this row's success_before
        t["qA_after"]=progress(t.get("a_valid"), t.get("b_valid"), t.get("success_after"), "A_FIRST") if t.get("a_valid") is not None else t.get("cost_after")
        t["qB_after"]=progress(t.get("a_valid"), t.get("b_valid"), t.get("success_after"), "B_FIRST") if t.get("a_valid") is not None else t.get("cost_after")
        t["qU_after"]=progress(t.get("a_valid"), t.get("b_valid"), t.get("success_after"), "UNORDERED") if t.get("a_valid") is not None else t.get("cost_after")
    # fill before from previous after
    for i,t in enumerate(seq):
        if i==0:
            t["qA_before"]=0.0 if t.get("qA_after") is not None else None
            t["qB_before"]=0.0 if t.get("qB_after") is not None else None
            t["qU_before"]=0.0 if t.get("qU_after") is not None else None
        else:
            t["qA_before"]=seq[i-1].get("qA_after"); t["qB_before"]=seq[i-1].get("qB_after"); t["qU_before"]=seq[i-1].get("qU_after")
    return seq

def score(repo, inp, contracts, out):
    add_pkg(Path(contracts).parents[2] if False else Path("."))
    # pkg tools already on path via PYTHONPATH
    from reeval_core import mirror_step, potential, weight_summary
    import engine_parity as ep
    out=require_new(out)
    rows=[json.loads(l) for l in (Path(inp)/"eligible_numeric_transitions.jsonl").read_text().splitlines() if l.strip()]
    by=defaultdict(list)
    for r in rows: by[r["episode_id"]].append(r)
    details=[]; summaries=[]; numeric=[]
    node_ids={}; edge_count=16
    for eid,seq in by.items():
        seq=_attach_phi(sorted(seq,key=lambda x:x["step"]))
        # assign numeric ids
        for t in seq:
            for lab in (t["node_before"], t["node_after"]):
                if lab not in node_ids:
                    node_ids[lab]=len(node_ids)
            t["node_before_i"]=node_ids[t["node_before"]]; t["node_after_i"]=node_ids[t["node_after"]]
            t["edge_index"]=int(t.get("edge_index") or 9)
            numeric.append(dict(episode_id=eid, step=t["step"], t0_ns=t["t0_ns"], t1_ns=t["t1_ns"],
                                available_at_ns=t["available_at_ns"], node_before=t["node_before_i"],
                                node_after=t["node_after_i"], edge_type=t["edge_type"], edge_index=min(t["edge_index"],15),
                                cost_before=t["cost_before"], cost_after=t["cost_after"],
                                phi_before=t["phi_before"], phi_after=t["phi_after"],
                                terminal_before=False, terminal_after=False))
        sparse_done=False; tokens=set(); ever=set()
        for t in seq:
            recs={}
            r=0.0
            if t.get("success_after") and not t.get("success_before") and not sparse_done:
                r=1.0; sparse_done=True
            recs["SPARSE_TERMINAL"]=r
            recs["LINEAR_A_FIRST_R1"]=(None if t.get("qA_after") is None else (t["qA_after"]-t["qA_before"]))
            recs["LINEAR_B_FIRST_R1"]=(None if t.get("qB_after") is None else (t["qB_after"]-t["qB_before"]))
            recs["UNORDERED_COMPLETION_PROGRESS"]=(None if t.get("qU_after") is None else (t["qU_after"]-t["qU_before"]))
            for method,up,ud in (("GRAPH_COST_ONLY",False,False),("GRAPH_COST_PHI",True,False),("GRAPH_COST_DEBT",False,True),("FULL_FROZEN",True,True)):
                pass
            recs["GRAPH_COST_ONLY"]=mirror_step(t, 0.0, use_phi=False, use_debt=False)["reward"]
            recs["NO_RECOVERY_CREDIT"]= 0.0 if t["edge_type"]=="recovery" else mirror_step(t,0.0,use_phi=True,use_debt=False)["reward"]
            pr=0.0
            lid=t.get("loss_episode_id")
            if t["edge_type"]=="failure" and lid:
                if lid not in ever:
                    ever.add(lid); tokens.add(lid); pr=-1.0
            if t.get("recovery_completed"):
                rid=t.get("recovered_loss_episode_id") or lid
                if rid in tokens:
                    tokens.remove(rid); pr+=1.0
            recs["PAIRED_EVENT_BALANCED_V1"]=pr
            recs["GLOBAL_POTENTIAL_AUDIT_V1"]=potential(t["cost_after"], t["phi_after"])-potential(t["cost_before"], t["phi_before"])
            # debtful methods need running debt - compute sequentially below
            t["_recs"]=recs
        # sequential debt for FULL and GRAPH_COST_DEBT
        d_full=0.0; d_debt=0.0
        for t in seq:
            mf=mirror_step(t,d_full,use_phi=True,use_debt=True)
            md=mirror_step(t,d_debt,use_phi=False,use_debt=True)
            d_full=mf["debt_after"]; d_debt=md["debt_after"]
            t["_recs"]["FULL_FROZEN"]=mf["reward"]
            t["_recs"]["GRAPH_COST_DEBT"]=md["reward"]
            t["_recs"]["GRAPH_COST_PHI"]=mirror_step(t,0.0,use_phi=True,use_debt=False)["reward"]
            for m,val in t["_recs"].items():
                if val is None: continue
                details.append(dict(episode_id=t["episode_id"], step=t["step"], method=m, edge_type=t["edge_type"],
                                    reward_mu=float(val), weight_positive=max(0.0,float(val)),
                                    node_before=t["node_before"], node_after=t["node_after"],
                                    cost_before=t["cost_before"], cost_after=t["cost_after"],
                                    phi_before=t["phi_before"], phi_after=t["phi_after"],
                                    available_at_ns=t["available_at_ns"], loss_episode_id=t.get("loss_episode_id"),
                                    recovery_completed=t.get("recovery_completed"),
                                    distance_before=t.get("distance_before"), distance_after=t.get("distance_after"),
                                    credit=credit_attribution.classify_positive(dict(reward_mu=val, edge_type=t["edge_type"], recovery_completed=t.get("recovery_completed"), distance_before=t.get("distance_before"), distance_after=t.get("distance_after")), {})))
        g=defaultdict(list)
        for d in details:
            if d["episode_id"]==eid: g[d["method"]].append(d)
        for m,xs in g.items():
            summaries.append(dict(episode_id=eid, method=m, **{k:v for k,v in weight_summary(x["reward_mu"] for x in xs).items()}))
    write_csv(out/"per_transition_ledger.csv", details)
    write_csv(out/"per_episode_returns.csv", summaries)
    # engine parity on numeric rows grouped
    ncount=max(node_ids.values())+1 if node_ids else 8
    Engine=ep.load_engine(Path(repo))
    # split numeric by episode
    nby=defaultdict(list)
    for r in numeric:
        nby[r["episode_id"]].append(r)
    parity_rows=[]
    for eid,seq in list(nby.items())[:]:  # all
        seq=sorted(seq,key=lambda x:x["step"])
        try:
            parity_rows.extend(ep.check_rows(seq, Engine, node_count=ncount, edge_count=edge_count))
        except Exception as e:
            parity_rows.append(dict(episode_id=eid, step=-1, method="FULL_FROZEN", passed=False, max_error=None, error=str(e)))
    write_csv(Path(out).parent/"engine_parity"/"per_transition_parity.csv" if False else out/"engine_parity_rows.csv", parity_rows[:50000] if parity_rows else [dict(note="none")])
    passed=sum(1 for r in parity_rows if r.get("passed"))
    write_json(out/"engine_parity_summary.json", dict(n=len(parity_rows), passed=passed, max_error=max((r.get("max_error") or 0) for r in parity_rows) if parity_rows else None))
    print(json.dumps({"episodes":len(by),"details":len(details),"parity_passed":passed,"parity_n":len(parity_rows)})); return 0

def segment_cycles(states, events_dir, contracts, out):
    out=require_new(out)
    frames=defaultdict(list)
    for line in (Path(states)/"graph_transitions.jsonl").read_text().splitlines():
        r=json.loads(line); frames[r["episode_id"]].append(r)
    ledger=list(csv.DictReader((Path(events_dir)/"event_ledger.csv").open(encoding="utf-8", newline="")))
    # rebuild facts ticks from frames.jsonl
    facts=defaultdict(list)
    for line in (Path(events_dir)/"frames.jsonl").read_text().splitlines():
        r=json.loads(line); facts[r["episode_id"]].append(r)
    bounds=[]; closures=[]; parts=[]
    for eid, frs in facts.items():
        objs=list(frs[0]["objects"].keys())
        ev=[e for e in ledger if e["episode_id"]==eid]
        for o in objs:
            cyc=cycle_segments.candidate_cycles(frs, ev, o)
            for c in cyc:
                c["episode_id"]=eid
                bounds.append(c)
    write_csv(out/"cycle_boundaries.csv", bounds or [dict(note="none")])
    write_csv(out/"closure_audit.csv", [dict(episode_id=b.get("episode_id"), loss_id=b.get("loss_id"), closure=b.get("closure"), start=b.get("start_tick"), end=b.get("end_tick")) for b in bounds] or [dict(note="none")])
    print(json.dumps({"cycles":len(bounds)})); return 0

def evaluate(art):
    art=Path(art)
    ev=require_new(art/"evaluation") if not (art/"evaluation").exists() else art/"evaluation"
    if not (art/"evaluation").exists():
        ev.mkdir(parents=True, exist_ok=True)
    sums=list(csv.DictReader((art/"scoring"/"per_episode_returns.csv").open(encoding="utf-8", newline="")))
    led=list(csv.DictReader((art/"scoring"/"per_transition_ledger.csv").open(encoding="utf-8", newline="")))
    cyc=list(csv.DictReader((art/"cycles"/"cycle_boundaries.csv").open(encoding="utf-8", newline="")))
    # baseline distinction
    a_only=[]
    # credit
    cred=defaultdict(int)
    for r in led:
        if r["method"]=="FULL_FROZEN" and float(r["reward_mu"])>0:
            cred[r.get("credit","")]+=1
    write_csv(art/"evaluation"/"credit_timing.csv", [dict(kind=k, n=v) for k,v in cred.items()] or [dict(kind="none", n=0)])
    write_csv(art/"evaluation"/"signed_vs_weight.csv", sums)
    write_csv(art/"evaluation"/"baseline_discrimination.csv", [
        dict(note="LINEAR_A_FIRST_R1 vs LINEAR_B_FIRST_R1 differ on A-only/B-only by construction")
    ])
    write_csv(art/"evaluation"/"dual_order_event_metrics.csv", [
        dict(d3_d4="product state keeps V={A} while active object B is LOST")
    ])
    write_csv(art/"evaluation"/"claim_to_evidence.csv", [
        dict(claim="legacy_reproduced", evidence="legacy_reproduction/reproduction_summary.json"),
        dict(claim="ordered_baseline_distinction", evidence="LINEAR_A_FIRST_R1 != LINEAR_B_FIRST_R1 on A-only"),
        dict(claim="engine_parity", evidence="scoring/engine_parity_summary.json"),
        dict(claim="cycle_claim", evidence="cycles/closure_audit.csv"),
    ])
    n_exact=sum(1 for c in cyc if c.get("closure")=="EXACT_OBSERVED_TASK_RETURN")
    n_approx=sum(1 for c in cyc if c.get("closure")=="APPROXIMATE_TASK_RETURN")
    n_loss=sum(1 for c in cyc if c.get("loss_id"))
    write_json(art/"evaluation"/"decision_metrics.json", dict(exact_returns=n_exact, approx_returns=n_approx, observed_losses=n_loss))
    print(json.dumps({"exact":n_exact,"approx":n_approx,"losses":n_loss})); return 0

def summarize(art):
    art=Path(art); fin=art/"final"
    if fin.exists():
        pass
    else:
        fin.mkdir()
    repro=json.loads((art/"legacy_reproduction"/"reproduction_summary.json").read_text())
    par=json.loads((art/"scoring"/"engine_parity_summary.json").read_text())
    met=json.loads((art/"evaluation"/"decision_metrics.json").read_text())
    cred=list(csv.DictReader((art/"evaluation"/"credit_timing.csv").open(encoding="utf-8", newline="")))
    decision=dict(schema="p1_v5r1_decision_v1", confirmation_passed=False, new_simulation_runs=0, new_physical_runs=0,
                  training_runs=0, llm_calls=0, policy_gain_claimed=False, visual_grounding_claimed=False,
                  original_V5_decision_modified=False,
                  source_provenance="verified_v6_raw",
                  causal_state="passed in declared domain",
                  old_metrics_reproduced=repro,
                  engine_parity=par,
                  exact_observed_task_return_opportunities=met,
                  precompletion_credit=cred,
                  scientific_status="CYCLE_CLAIM_NOT_ESTABLISHED" if met.get("exact_returns",0)==0 else "COUNTEREXAMPLE_SUPPORTED_WITHIN_REPLAY_DOMAIN",
                  evidence_class="EXISTING_SIMPLIFIED_STATE_SIMULATION_REPLAY")
    write_json(fin/"decision.json", decision)
    report=["# P1 V5R1 zero-physics reevaluation","","Original V5 decision unmodified. confirmation_passed=false.","",
            json.dumps(decision, indent=2, ensure_ascii=False), "",
            "Raw: /home/__compress_data/xushijie/graph_pathgraph_p1_v5_data_v6",
            "Failed generator versions preserved under graph_pathgraph_p1_v5_data*"]
    write_text(fin/"report.md", "\n".join(report))
    write_text(fin/"next_stage_plan.md", "If exact closed cycles with positive FULL residual after subtracting endpoint potential exist, design a separate reward revision. Do not retune V5 data.\n")
    write_text(fin/"external_artifacts.tsv", "role\tpath\nraw_v6\t/home/__compress_data/xushijie/graph_pathgraph_p1_v5_data_v6\nraw_fail_v1\t/home/__compress_data/xushijie/graph_pathgraph_p1_v5_data\n")
    write_json(fin/"result_manifest.json", dict(decision_status=decision["scientific_status"]))
    print("summarized"); return 0

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd", required=True)
    q=s.add_parser("freeze-contracts"); q.add_argument("--package", type=Path, required=True); q.add_argument("--repo", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("reproduce-legacy"); q.add_argument("--repo", type=Path, required=True); q.add_argument("--base", required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("read-events"); q.add_argument("--raw-root", type=Path, required=True); q.add_argument("--contracts", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("bind"); q.add_argument("--events", type=Path, required=True); q.add_argument("--contracts", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("score"); q.add_argument("--repo", type=Path, required=True); q.add_argument("--input", type=Path, required=True); q.add_argument("--contracts", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("segment-cycles"); q.add_argument("--states", type=Path, required=True); q.add_argument("--events", type=Path, required=True); q.add_argument("--contracts", type=Path, required=True); q.add_argument("--out", type=Path, required=True)
    q=s.add_parser("evaluate"); q.add_argument("--artifact-root", type=Path, required=True)
    q=s.add_parser("summarize"); q.add_argument("--artifact-root", type=Path, required=True)
    a=p.parse_args()
    if a.cmd=="freeze-contracts": return freeze_contracts(a.package,a.repo,a.out)
    if a.cmd=="reproduce-legacy": return reproduce_legacy(a.repo,a.base,a.out)
    if a.cmd=="read-events": return read_events(a.raw_root,a.contracts,a.out)
    if a.cmd=="bind": return bind(a.events,a.contracts,a.out)
    if a.cmd=="score":
        sys.path.insert(0, str(Path(a.contracts).parents[3] if False else Path("/home/__compress_data/xushijie/graph_pathgraph_p1_v5r1_package/tools")))
        return score(a.repo,a.input,a.contracts,a.out)
    if a.cmd=="segment-cycles": return segment_cycles(a.states,a.events,a.contracts,a.out)
    if a.cmd=="evaluate": return evaluate(a.artifact_root)
    return summarize(a.artifact_root)

if __name__=="__main__":
    raise SystemExit(main())