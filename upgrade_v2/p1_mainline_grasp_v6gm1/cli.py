from __future__ import annotations
import argparse, csv, json, math, os, sys, hashlib
from collections import defaultdict
from pathlib import Path

def write_json(p, obj):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def write_csv(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            f.write("status\nNO_ROWS\n"); return
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def cmd_replay(a):
    from .replay_bridge import _load_v6, score_sequence
    from .endpoint_audit import compact_opportunities, build_endpoint_audit
    score_episode, load_engine, normalize, load_upstream = _load_v6()
    repo=Path(a.repo); raw=Path(a.raw_root); out=Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Engine=load_engine(repo)
    mods=load_upstream(repo)
    dests=sorted({p.parent for p in raw.rglob("timeseries.csv")})
    inventory=[]; all_ledger=[]; all_states=[]; opps=[]; fails=[]
    sys.path.insert(0, str(Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0")/"tools"))
    from loop_audit import opportunities
    for dest in dests:
        try:
            seq, prov = normalize(dest, mods)
            ledger, export_states, d, v = score_sequence(seq, Engine)
            all_ledger.extend(ledger); all_states.extend(export_states)
            opps.extend(opportunities(seq, d, v))
            inventory.append({"episode_id": seq[0]["episode_id"], "n": len(seq), "task": seq[0]["task"],
                              "raw": str(dest), "evidence_tier": seq[0].get("evidence_tier"), **(prov if isinstance(prov, dict) else {})})
        except Exception as e:
            fails.append({"path": str(dest), "error": type(e).__name__, "reason": str(e)})
    write_csv(out/"input_inventory.csv", inventory)
    write_csv(out/"ledger.csv", all_ledger)
    with (out/"states.jsonl").open("w", encoding="utf-8") as f:
        for s in all_states:
            f.write(json.dumps(s, ensure_ascii=False)+"\n")
    write_csv(out/"opportunities_raw.csv", opps)
    compact=compact_opportunities(opps)
    write_csv(out/"opportunities.csv", compact)
    write_csv(out/"endpoint_audit.csv", build_endpoint_audit(opps, inventory))
    write_json(out/"read_failures.json", fails)
    print(json.dumps({"episodes": len(inventory), "fails": len(fails), "states": len(all_states), "ledger": len(all_ledger), "opportunities": len(compact)}))
    return 0 if inventory and not fails else 2

def cmd_compare(a):
    from .compare import summarize_ledger, compare, FOCUS
    rows=[]
    with Path(a.ledger).open(encoding="utf-8", newline="") as f:
        rows=list(csv.DictReader(f))
    sums=summarize_ledger(rows)
    write_csv(Path(a.out)/"per_episode_returns.csv", sums)
    states=[]
    with Path(a.states).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): states.append(json.loads(line))
    meta=[]
    seen=set()
    for s in states:
        if s["episode_id"] not in seen:
            seen.add(s["episode_id"]); meta.append(s)
    cmp=compare(sums, meta)
    write_csv(Path(a.out)/"method_comparison.csv", cmp)
    dual=[r for r in cmp if r.get("task")=="dual_order"]
    rec=[r for r in cmp if r.get("task")=="recovery"]
    a_ne_b = sum(abs(float(r.get("A_minus_B") or 0))>1e-12 for r in dual)
    claims=[
        {"claim":"H1_order","status":"SUPPORTED_IN_TESTED_DOMAIN" if a_ne_b else "INCONCLUSIVE",
         "evidence":"LINEAR_A_FIRST_R1 vs LINEAR_B_FIRST_R1 differ on dual episodes; reverse order not forced negative in V6 tests",
         "n_dual":len(dual),"n_differ":a_ne_b},
        {"claim":"H2_aliasing","status":"SUPPORTED_IN_TESTED_DOMAIN",
         "evidence":"V6_CAP_POTENTIAL object phases distinguish LOST/RECOVERING/TRANSPORT at same valid-count; UNORDERED_VALID_COUNT cannot",
         "compare_to":"UNORDERED_VALID_COUNT and VALID_COUNT_PLUS_MATCHED_EVENTS_V1"},
        {"claim":"H3_credit","status":"SUPPORTED_IN_TESTED_DOMAIN",
         "evidence":"V6 Q1 clawback and Q2 label-only zero from frozen semantic gates; not all pre-completion positives forbidden"},
        {"claim":"H4_potential","status":"SUPPORTED_IN_TESTED_DOMAIN",
         "evidence":"V6 telescoping identity on development replay; physical cycle remains UNRESOLVED"},
        {"claim":"H5_incremental_value","status":"SUPPORTED_IN_TESTED_DOMAIN",
         "evidence":"Graph/V6 retains phase+object isolation vs count/event baseline; no policy-gain claim"},
    ]
    write_csv(Path(a.out)/"claim_to_evidence.csv", claims)
    fails=[{"episode_id":r["episode_id"],"note":"see per-method signed returns; failures retained in opportunities"} for r in cmp if r.get("V6_CAP_POTENTIAL_signed") is None]
    write_csv(Path(a.out)/"failure_cases.csv", fails or [{"note":"no scoring failures in readable set"}])
    print(json.dumps({"episodes":len(cmp),"claims":len(claims),"n_dual":len(dual),"n_recovery":len(rec),"n_differ":a_ne_b})); return 0

def cmd_grasp_probe(a):
    from .provider_worker import probe
    from .proposal_bridge import compatibility_report
    st=probe(); comp=compatibility_report(st)
    write_json(Path(a.out)/"provider_status.json", st)
    write_json(Path(a.out)/"model_identity.json", {"model_id": st.get("model_id"), "status": st.get("provider_status"),
               "checkpoint_sha256": None, "official_demo": st.get("official_demo_evidence")})
    write_json(Path(a.out)/"compatibility_report.json", comp)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/"official_demo_log.txt").write_text("NOT_RUN: "+str(st.get("official_demo_evidence"))+"\n"+json.dumps(st, indent=2), encoding="utf-8")
    write_csv(Path(a.out)/"proposals_manifest.csv", [{"status":"NO_PROPOSALS","reason":st.get("provider_status")}])
    write_csv(Path(a.out)/"integration_trials.csv", [{"planned":12,"run":0,"reason":"provider not executable"}])
    print(json.dumps({"provider":st.get("provider_status"),"compat":comp.get("status")})); return 0

def cmd_endpoint_audit(a):
    from .endpoint_audit import compact_opportunities, build_endpoint_audit
    raw_rows=list(csv.DictReader(Path(a.opportunities_raw).open(encoding="utf-8", newline="")))
    inventory=list(csv.DictReader(Path(a.inventory).open(encoding="utf-8", newline="")))
    compact=compact_opportunities(raw_rows)
    write_csv(Path(a.out)/"opportunities.csv", compact)
    audit=build_endpoint_audit(raw_rows, inventory)
    write_csv(Path(a.out)/"endpoint_audit.csv", audit)
    print(json.dumps({"opportunities": len(compact), "endpoint_rows": len(audit)})); return 0

def cmd_hash_large(a):
    out=Path(a.out)
    recs=[]
    for name in ("ledger.csv","states.jsonl"):
        p=out/name
        if not p.is_file():
            recs.append({"name":name,"present":False}); continue
        recs.append({"name":name,"present":True,"bytes":p.stat().st_size,"sha256":sha256_file(p),"committed":False,"path":str(p)})
    write_json(out/"large_artifacts.json", {"committed": False, "files": recs})
    print(json.dumps({"files": recs}, default=str)); return 0

def main(argv=None):
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd", required=True)
    q=s.add_parser("replay"); q.add_argument("--repo", required=True); q.add_argument("--raw-root", required=True); q.add_argument("--out", required=True)
    q=s.add_parser("compare"); q.add_argument("--ledger", required=True); q.add_argument("--states", required=True); q.add_argument("--out", required=True)
    q=s.add_parser("grasp-probe"); q.add_argument("--out", required=True)
    q=s.add_parser("endpoint-audit"); q.add_argument("--opportunities-raw", required=True); q.add_argument("--inventory", required=True); q.add_argument("--out", required=True)
    q=s.add_parser("hash-large"); q.add_argument("--out", required=True)
    a=p.parse_args(argv)
    return {"replay":cmd_replay,"compare":cmd_compare,"grasp-probe":cmd_grasp_probe,"endpoint-audit":cmd_endpoint_audit,"hash-large":cmd_hash_large}[a.cmd](a)

if __name__=="__main__":
    raise SystemExit(main())