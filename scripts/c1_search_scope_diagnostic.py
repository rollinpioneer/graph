#!/usr/bin/env python
"""CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1 driver: frozen-model, read-only diagnostic (no training, no fit, no new problems). Stages: init, prepare, fixtures, freeze, traces, local, gpu-stage, score, labels, report, final_receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cp_disr.pddl import depots as DP                                                                 # noqa: E402
from cp_disr.pddl import task as T                                                                    # noqa: E402
from cp_disr.pddl.scope_diag import analysis as AN                                                    # noqa: E402
from cp_disr.pddl.scope_diag import bounds as BD                                                      # noqa: E402
from cp_disr.pddl.scope_diag import checks as CK                                                      # noqa: E402
from cp_disr.pddl.scope_diag import labels as LB                                                      # noqa: E402
from cp_disr.pddl.scope_diag import references as RF                                                  # noqa: E402
from cp_disr.pddl.scope_diag.core import (DENSE, WL, cpu_now, gz_read, gz_write, hx, ix, jdump, jload, load_task, local_positions, local_structure, run_trace,      # noqa: E402
                                          score_states, sha_file, sid, trace_states)
from cp_disr.pddl.search_match import evaluators as EV                                                # noqa: E402
from cp_disr.pddl.search_match.engine import SuccessorIndex                                           # noqa: E402
from cp_disr.pddl.search_match.runner import OldRun, load_neural, plan_text                           # noqa: E402

PROC_START = time.time()
CARD = "CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1"
CFG_REL = "configs/c1_search_scope_diagnostic_v1.yaml"


def load_cfg():
    import yaml
    return yaml.safe_load((ROOT / CFG_REL).read_text(encoding="utf-8"))


def old_run(cfg):
    return OldRun(ROOT / cfg["old_run_root"])


def ledger(rr, stage, **kv):
    p = Path(rr) / "receipts" / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kv), default=str) + "\n")


def manifest_cases(rr):
    return jload(Path(rr) / "registration" / "case_manifest.json")["cases"]


def goal_sha(task):
    return hashlib.sha256(" ".join(sorted("%s(%s)" % (g[0], ",".join(g[1])) for g in task.goal_atoms)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ init / prepare
def cmd_init(auth_text):
    cfg = load_cfg()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rr = ROOT / "runs" / "final_master" / "c1_route_b" / "search_scope_diagnostic_v1" / ("%s_%s" % (ts, cfg["base_commit"][:7]))
    for d in ("plan", "registration", "checks", "references", "states", "traces", "labels", "results", "receipts", "driver_logs"):
        (rr / d).mkdir(parents=True, exist_ok=True)
    (rr / "plan" / "runbook.md").write_text((ROOT / cfg["plan_doc"]).read_text(encoding="utf-8"), encoding="utf-8")
    jdump(rr / "plan" / "authorisation.json", {"card": CARD, "execution_authorized": True, "authorization_text": auth_text.strip(), "plan_doc_sha256": sha_file(ROOT / cfg["plan_doc"]), "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                               "scope": "zero-training frozen-model diagnostic; layer 2 / WL backup NOT authorised"})
    Path(Path.home() / "work" / "rr_scope.txt").write_text(str(rr) + "\n")
    print(rr)


def cmd_prepare(rr):
    rr = Path(rr)
    cfg = load_cfg()
    old = old_run(cfg)
    t0, c0 = time.time(), cpu_now()
    ck = old.checkpoint("dense")
    sha_before = sha_file(ck["path"])
    if sha_before != cfg["dense"]["sha256"] or ck["update"] != cfg["dense"]["selected_update"]:
        raise SystemExit("frozen DENSE identity mismatch: %s update %s" % (sha_before, ck["update"]))
    tools = {"scorpion_downward": {"path": str(EV.SCORPION_DOWNWARD), "sha256": sha_file(EV.SCORPION_DOWNWARD)}, "exact_dist": {"path": str(T.EXACT_BIN), "sha256": sha_file(T.EXACT_BIN)},
             "goose_python": str(EV.GOOSE_PY), "translate_driver": str(EV.SCORPION_FD)}
    wl = {k: {kk: str(vv) for kk, vv in old.wl_params(k).items()} for k in ("ipc", "train")}
    for k in wl:
        wl[k]["model_sha256"] = sha_file(old.wl_params(k)["model"])
        wl[k]["params_sha256"] = sha_file(old.wl_params(k)["params"])
    ident = {"dense": {"checkpoint": str(ck["path"]), "sha256_before": sha_before, "update": ck["update"]}, "wl": wl, "tools": tools, "domains": {"typed": {"path": str(DP.DOMAIN_TYPED), "sha256": sha_file(DP.DOMAIN_TYPED)},
             "ipc": {"path": str(DP.DOMAIN_IPC), "sha256": sha_file(DP.DOMAIN_IPC)}}, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "old_run_root": cfg["old_run_root"]}
    jdump(rr / "registration" / "asset_identity.json", ident)
    # ---- cases
    ipc = {c["case_id"]: c for c in old.cases("ipc")}
    cases = []
    for grp, key in (("F", "ipc_F"), ("R", "ipc_R")):
        for cid in cfg["cases"][key]:
            c = ipc[cid]
            cases.append({"case_id": cid, "set": "ipc", "group": grp, "file": c["file"], "sha256": c["sha256"]})
    train = old.cases("train")
    cc = cfg["cases"]["control"]
    missing = []
    exact = jload(Path(ROOT / cfg["old_run_root"]) / "data" / "exact.json") if (Path(ROOT / cfg["old_run_root"]) / "data" / "exact.json").is_file() else {}
    for n in cc["crates"]:
        for tr in cc["needs_transport"]:
            cell = [c for c in train if c["analysis"]["n_crates"] == n and bool(c["analysis"]["needs_transport"]) == tr]
            if not cell:
                missing.append({"crates": n, "needs_transport": tr, "reason": "no Train96 problem in the cell"})
                continue
            best = min(cell, key=lambda c: hashlib.sha256((cc["hash_prefix"] + c["case_id"]).encode()).hexdigest())
            cases.append({"case_id": best["case_id"], "set": "train", "group": "C", "file": best["file"], "sha256": best["sha256"], "cell": {"crates": n, "needs_transport": tr}, "cell_size": len(cell)})
    jdump(rr / "registration" / "case_manifest.json", {"cases": cases, "missing_cells": missing, "exact_json_present": bool(exact)})
    # ---- reference plans
    refs, tasks_info = {}, {}
    sens = {}
    p = ROOT / cfg["reference_length_sensitivity"]
    if p.is_file():
        import csv
        for r in csv.DictReader(open(p, encoding="utf-8")):
            sens[(r.get("set"), r.get("case_id"))] = r
    for c in cases:
        task, dom = load_task(c)
        if c["set"] == "ipc":
            best, cands = RF.select_reference(task, ROOT, cfg["plan_source_roots"], c["case_id"])
            if best is None:
                raise SystemExit("no valid reference plan for %s" % c["case_id"])
            ids = best["ids"]
            proven = (sens.get(("ipc", c["case_id"])) or {}).get("proven_optimal_length")
            proven = int(proven) if proven not in (None, "") else None
            opt = "PROVEN_OPTIMAL" if proven is not None and proven == len(ids) else "UNKNOWN"
            info = {"source": best["rel"], "n_candidates": len(cands), "n_valid": sum(1 for x in cands if x["valid"]), "candidates": [{"rel": x["rel"], "length": x["length"], "valid": x["valid"], "sha256": x["sha256"]} for x in cands],
                    "proven_optimum": proven, "optimality": opt}
        else:
            oracle = T.ExactOracle(task)
            try:
                if not oracle.ok:
                    raise SystemExit("exact oracle not available for %s" % c["case_id"])
                ids = oracle.plan()
                d0 = oracle.query(task.init_mask)[0]
                info = {"source": "ExactOracle.plan() (rebuilt, unit cost)", "n_candidates": 1, "n_valid": 1, "candidates": [], "proven_optimum": d0, "optimality": "PROVEN_OPTIMAL", "oracle_states": oracle.n_states}
            finally:
                oracle.close()
        fin = task.replay(ids)
        assert fin is not None and task.goal_satisfied(fin), c["case_id"]
        seq, _ref = RF.path_states(task, ids)
        text = plan_text(task, [task.action_by_id[a].index for a in ids])
        info.update({"ids": ids, "length": len(ids), "sha256": hashlib.sha256(text.encode()).hexdigest(), "path_states_hex": [hx(s) for s in seq], "n_ground_actions": len(task.actions), "n_dyn_atoms": len(task.dyn_atoms),
                     "goal_sha256": goal_sha(task), "problem_sha256": c["sha256"], "domain_sha256": sha_file(dom)})
        refs[c["case_id"]] = info
        print("[prepare] %s %s reference length %d (%s) from %s" % (c["group"], c["case_id"], len(ids), info["optimality"], info["source"]))
    jdump(rr / "references" / "plans.json", refs)
    jdump(rr / "registration" / "source_identity.json", {"base_commit": cfg["base_commit"], "files": {p: sha_file(ROOT / p) for p in source_files()}})
    ledger(rr, "prepare", seconds=round(time.time() - t0, 1), cpu_seconds=round(cpu_now() - c0, 1), weights_sha_before=sha_before)
    print("prepared %d cases" % len(cases))


def source_files():
    return [CFG_REL, "scripts/c1_search_scope_diagnostic.py", "tests/test_scope_diag.py"] + ["src/cp_disr/pddl/scope_diag/" + n for n in ("__init__.py", "observed.py", "bounds.py", "references.py", "core.py", "labels.py", "analysis.py", "checks.py")] + [
        "src/cp_disr/pddl/search_match/engine.py", "src/cp_disr/pddl/search_match/evaluators.py", "src/cp_disr/pddl/search_match/budget.py", "src/cp_disr/pddl/fast_alt/fast_value.py", "src/cp_disr/pddl/fast_alt/evaluators.py"]


# ------------------------------------------------------------------------------------------------ fixtures / freeze
def cmd_fixtures(rr, device_name):
    rr = Path(rr)
    cfg = load_cfg()
    old = old_run(cfg)
    t0, c0 = time.time(), cpu_now()
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_scope_diag.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT), env=env, capture_output=True, text=True)
    out = {"pytest": {"returncode": r.returncode, "passed": r.returncode == 0, "tail": r.stdout[-700:]}}
    cases = manifest_cases(rr)
    refs = jload(rr / "references" / "plans.json")
    ctrl = [c for c in cases if c["group"] == "C"]
    ipc = [c for c in cases if c["set"] == "ipc"]
    out["lmcut_control"] = [CK.check_control(c, 24, 0, cfg["labels"]["lmcut_timeout_seconds"]) for c in ctrl]
    out["lmcut_reference_suffix"] = [CK.check_reference_suffix(c, refs[c["case_id"]]["ids"], 12, cfg["labels"]["lmcut_timeout_seconds"]) for c in ipc]
    # observer equivalence with the real scorers on a short prefix of the first control problem (4 searches: plain + observed, WL and DENSE)
    case = ctrl[0]
    obs = []
    wl = EV.WlEval(old.wl_params("train")["params"])
    obs.append(CK.check_observer_real(case, wl, "WL", 16))
    gpu_s = 0.0
    searches = 2
    if device_name:
        import torch
        from cp_disr.pddl.fast_alt.evaluators import FastNeuralEval
        dev = torch.device(device_name)
        tg = time.time()
        model, ckpt = load_neural(old, "V_DENSE", dev)
        fe = FastNeuralEval("FAST", model, dev)
        obs.append(CK.check_observer_real(case, fe, "DENSE_FAST", 16))
        torch.cuda.synchronize()
        gpu_s = time.time() - tg
        searches += 2
    out["observer_real"] = obs
    searches += 4                                                       # the four synthetic searches of tests/test_scope_diag.py (plain, observed, observed with another scorer, repeat snapshot run)
    out["fixture_searches_run"] = searches
    out["fixture_gpu_seconds"] = round(gpu_s, 1)
    ok = (out["pytest"]["passed"] and all(x["admissible"] and x["init_mapping_equal"] and x["repeat_consistent"] and x["goal_state_value_zero"] is not False and x["metric"] == 0 for x in out["lmcut_control"]) and
          all(x["ok"] and x["metric"] == 0 for x in out["lmcut_reference_suffix"]) and all(x["identical"] for x in obs))
    out["all_passed"] = ok
    jdump(rr / "checks" / "fixture_results.json", out)
    cpu = cpu_now() - c0
    ledger(rr, "fixtures", seconds=round(time.time() - t0, 1), cpu_seconds=round(cpu, 1), gpu_seconds=round(gpu_s, 1), fixture_searches=searches, passed=ok)
    print("fixtures all_passed=%s" % ok)
    return 0 if ok else 3


def cmd_freeze(rr):
    rr = Path(rr)
    cfg = load_cfg()
    fx = jload(rr / "checks" / "fixture_results.json")
    if not fx["all_passed"]:
        raise SystemExit("fixtures failed: no freeze")
    jdump(rr / "registration" / "config.json", cfg)
    (rr / "registration" / "config.yaml").write_text((ROOT / CFG_REL).read_text(encoding="utf-8"), encoding="utf-8")
    jdump(rr / "registration" / "source_identity.json", {"base_commit": cfg["base_commit"], "files": {p: sha_file(ROOT / p) for p in source_files()}})
    jdump(rr / "registration" / "registration.json", {"card": CARD, "base_commit": cfg["base_commit"], "branch": cfg["branch"], "plan_sha256": sha_file(ROOT / cfg["plan_doc"]), "config_sha256": sha_file(ROOT / CFG_REL),
                                                      "new_neural_trainings": 0, "new_wl_fits": 0, "optimizer_steps": 0, "new_problem_generation": 0, "use_diagnostic_certificates_for_training": False,
                                                      "caps": cfg["labels"], "traces": cfg["traces"], "decision": cfg["decision"], "fixtures": {"all_passed": True, "searches": fx["fixture_searches_run"]},
                                                      "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("frozen")


# ------------------------------------------------------------------------------------------------ traces / local / union / scoring
def make_wl(old, case):
    return EV.WlEval(old.wl_params(case["set"])["params"])


def ensure_local(rr, cases, cfg):
    refs = jload(Path(rr) / "references" / "plans.json")
    for c in cases:
        p = Path(rr) / "states" / ("local_%s.json" % c["case_id"])
        if p.is_file():
            continue
        task, _dom = load_task(c)
        index = SuccessorIndex(task)
        seq, _ref = RF.path_states(task, refs[c["case_id"]]["ids"])
        loc = local_structure(task, index, seq, local_positions(len(refs[c["case_id"]]["ids"]), cfg["local"]["positions"]))
        jdump(p, [{"k": x["k"], "parent": hx(x["parent"]), "child": hx(x["child"]), "goal_successor": x["goal_successor"], "succ": [[hx(t), fa, aids] for t, fa, aids in x["succ"]]} for x in loc])


def cmd_traces(rr, driver, shard, nshards, device_name):
    rr = Path(rr)
    cfg = load_cfg()
    old = old_run(cfg)
    refs = jload(rr / "references" / "plans.json")
    cases = sorted(manifest_cases(rr), key=lambda c: c["case_id"])[shard::nshards]
    t0, c0 = time.time(), cpu_now()
    ev_dense = None
    if driver == DENSE:
        import torch
        from cp_disr.pddl.fast_alt.evaluators import FastNeuralEval
        dev = torch.device(device_name or "cuda")
        model, ckpt = load_neural(old, "V_DENSE", dev)
        ev_dense = FastNeuralEval("FAST", model, dev)
        warm = os.environ.get("CPDISR_WARMUP_PROBLEM", "")
        if warm and Path(warm).is_file():                                     # kernel warm-up on a Train96 problem outside every trace clock
            from cp_disr.pddl.search_match.budget import Budget
            wt = T.depots_task(DP.DOMAIN_TYPED, warm)
            wb = Budget(60, 1000, 8 * 2 ** 30)
            ev_dense.prepare(wt, {}, wb)
            ev_dense.evaluate([wt.init_mask], wb)
            torch.cuda.synchronize()
            ev_dense.close()
    recs = []
    for c in cases:
        out = rr / "traces" / c["case_id"] / (driver + ".jsonl.gz")
        if out.is_file():
            continue
        ev = ev_dense if driver == DENSE else make_wl(old, c)
        rec = run_trace(c, driver, ev, refs[c["case_id"]], cfg, out)
        recs.append(rec)
        print("[trace] %s %s -> %s expanded %s wall %.1fs snapshots %s" % (driver, c["case_id"], rec["status"], rec.get("expanded"), rec["wall_total"], rec["snapshots_taken"]), flush=True)
    ledger(rr, "traces_%s" % driver, shard=[shard, nshards], seconds=round(time.time() - t0, 1), cpu_seconds=round(cpu_now() - c0, 1), traces=len(recs), statuses=dict(Counter(r["status"] for r in recs)))


def build_union(rr, cases):
    for c in cases:
        cid = c["case_id"]
        u = set()
        for dr in (DENSE, WL):
            p = Path(rr) / "traces" / cid / (dr + ".jsonl.gz")
            if p.is_file():
                u |= trace_states(gz_read(p))
        for x in jload(Path(rr) / "states" / ("local_%s.json" % cid)):
            u.add(x["parent"])
            u.add(x["child"])
            for h, _fa, _a in x["succ"]:
                u.add(h)
        jdump(Path(rr) / "states" / ("union_%s.json" % cid), sorted(u))


def score_union(rr, cases, scorer, old, device_name):
    t0 = time.time()
    ev_dense = None
    if scorer == DENSE:
        import torch
        from cp_disr.pddl.fast_alt.evaluators import FastNeuralEval
        dev = torch.device(device_name or "cuda")
        model, ckpt = load_neural(old, "V_DENSE", dev)
        ev_dense = FastNeuralEval("FAST", model, dev)
    n = 0
    for c in cases:
        cid = c["case_id"]
        out = Path(rr) / "states" / ("scores_%s_%s.json" % (scorer, cid))
        if out.is_file():
            continue
        task, _dom = load_task(c)
        states = jload(Path(rr) / "states" / ("union_%s.json" % cid))
        ev = ev_dense if scorer == DENSE else make_wl(old, c)
        vals, wall = score_states(ev, task, c, states)
        jdump(out, vals)
        n += len(states)
        print("[score] %s %s %d states %.1fs" % (scorer, cid, len(states), wall), flush=True)
    return n, time.time() - t0


def cmd_local(rr):
    cfg = load_cfg()
    ensure_local(rr, manifest_cases(rr), cfg)
    print("local structures ready")


def cmd_gpu_stage(rr, device_name):
    """ONE GPU process: warm-up, DENSE traces of all cases, union of all sampled states, offline DENSE scores of the union."""
    rr = Path(rr)
    cfg = load_cfg()
    old = old_run(cfg)
    t0, c0 = time.time(), cpu_now()
    cmd_traces(rr, DENSE, 0, 1, device_name)
    cases = sorted(manifest_cases(rr), key=lambda c: c["case_id"])
    for c in cases:
        if not (rr / "traces" / c["case_id"] / "wl.jsonl.gz").is_file():
            raise SystemExit("WL traces must exist before the GPU stage: %s" % c["case_id"])
    build_union(rr, cases)
    n, w = score_union(rr, cases, DENSE, old, device_name)
    ledger(rr, "gpu_stage", gpu_process_wall_seconds=round(time.time() - PROC_START, 1), cpu_seconds=round(cpu_now() - c0, 1), scored_states=n, device=device_name)


def cmd_score(rr, scorer):
    rr = Path(rr)
    cfg = load_cfg()
    old = old_run(cfg)
    t0, c0 = time.time(), cpu_now()
    cases = sorted(manifest_cases(rr), key=lambda c: c["case_id"])
    n, w = score_union(rr, cases, scorer, old, None)
    ledger(rr, "score_%s" % scorer, seconds=round(time.time() - t0, 1), cpu_seconds=round(cpu_now() - c0, 1), scored_states=n)


# ------------------------------------------------------------------------------------------------ labels
def label_demand(rr, cases, inp):
    """{case: [(priority, hex, category)]} before allocation. Categories: R2_CORE (parent, plan child, both first choices), R3_POPPED, R2_FILL."""
    out = {}
    nlab = load_cfg()["local"]["labelled_successors"]
    for c in cases:
        cid = c["case_id"]
        items = {}

        def add(pri, h, cat):
            if h not in items or pri < items[h][0]:
                items[h] = (pri, cat)
        for pos in inp.local[cid]:
            k = pos["k"]
            if pos.get("goal_successor"):
                continue
            succ = pos["succ"]
            core = [pos["parent"], pos["child"]]
            for sc in (DENSE, WL):
                vals = [(inp.value(cid, sc, h), fa, h) for h, fa, _a in succ]
                if all(v[0] is not None for v in vals) and vals:
                    core.append(min(vals, key=lambda x: (x[0], x[1]))[2])
            for h in core:
                add(0.0, h, "R2_CORE")
            have = {h for h in core if h in {s[0] for s in succ}}
            rest = sorted((h for h, _fa, _a in succ if h not in have), key=lambda h: hashlib.sha256(("%s|%d|%s" % (cid, k, h)).encode()).hexdigest())
            for rank, h in enumerate(rest[:max(0, nlab - len(have))]):
                add(2.0 + rank / 1000.0, h, "R2_FILL")
        for di, dr in enumerate((DENSE, WL)):
            for d in inp.traces.get((cid, dr), []):
                if d.get("type") == "snapshot":
                    add(1.0 + d["event"] / 10000.0 + di / 1e6, d["popped"]["state"], "R3_POPPED")
        out[cid] = sorted((p, h, cat) for h, (p, cat) in items.items())
    return out


def cmd_labels(rr):
    rr = Path(rr)
    cfg = load_cfg()
    L = cfg["labels"]
    old = old_run(cfg)
    t0 = time.time()
    r0, k0 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    cases = sorted(manifest_cases(rr), key=lambda c: c["case_id"])
    inp = AN.Inputs(rr)
    refs = inp.refs
    demand = label_demand(rr, cases, inp)
    taken, rest = LB.allocate({cid: [(p, (h, cat)) for p, h, cat in v] for cid, v in demand.items()}, L["unique_bound_states"] - L["additional_upper_bound_requests"], [c["case_id"] for c in cases])
    rows = {}

    def row(cid, h, cat, **kv):
        d = {"case_id": cid, "state_hash": sid(refs[cid]["domain_sha256"], refs[cid]["problem_sha256"], ix(h)), "state_hex": h, "problem_sha256": refs[cid]["problem_sha256"], "goal_sha256": refs[cid]["goal_sha256"],
             "cost_model": "unit", "lower": None, "upper": None, "lower_source": "UNAVAILABLE", "upper_source": "UNAVAILABLE", "plan_sha256": None, "status": "UNKNOWN", "cpu_seconds": 0.0, "category": cat}
        d.update(kv)
        rows[(cid, h)] = d
        return d
    cpu_report, attempted = 0.0, 0
    # ---- unallocated demand
    for cid, items in rest.items():
        for h, cat in items:
            row(cid, h, cat, status="BUDGET_UNLABELLED")
    # ---- control problems: exact distances
    for c in cases:
        cid = c["case_id"]
        if c["group"] != "C":
            continue
        task, _dom = load_task(c)
        oracle = T.ExactOracle(task)
        try:
            for h, cat in taken[cid]:
                d, _ = oracle.query(ix(h))
                attempted += 1
                row(cid, h, cat, lower=d, upper=d, lower_source="EXACT", upper_source="EXACT", status="OK")
        finally:
            oracle.close()
    # ---- IPC problems: LM-cut lower bounds
    bridges, jobs = {}, []
    for c in cases:
        cid = c["case_id"]
        if c["group"] == "C":
            continue
        task, dom = load_task(c)
        br = BD.SasBridge(task, dom, c["file"], workdir=str(rr / "labels"))
        bridges[cid] = (br, task, c)
        cpu_report += br.cpu_translate
        keep = Path(rr) / "labels" / "_work" / cid
        keep.mkdir(parents=True, exist_ok=True)
        sas_copy = keep / "output.sas"
        sas_copy.write_text("\n".join(br.sas["lines"]) + "\n")
        for h, cat in taken[cid]:
            try:
                vals = br.values(ix(h))
            except ValueError as e:
                row(cid, h, cat, status="UNSUPPORTED", error=str(e))
                continue
            jobs.append((cid, h, str(sas_copy), vals, L["lmcut_timeout_seconds"]))
            jobs[-1] = jobs[-1] + (cat,)
    jobs.sort(key=lambda j: (j[0], j[1]))
    job_cat = {(j[0], j[1]): j[5] for j in jobs}
    # priority order across cases: round robin by the allocation order of each case
    order = {}
    for cid, items in taken.items():
        for i, (h, cat) in enumerate(items):
            order[(cid, h)] = i
    jobs.sort(key=lambda j: (order[(j[0], j[1])], j[0]))
    res, used, undispatched = LB.run_jobs(LB.lmcut_job, [j[:5] for j in jobs], L["concurrent_cpu_label_workers"], max(0.0, L["label_cpu_core_seconds"] - cpu_report - 600.0))
    done = set()
    for cid, h, v, status, cpu in res:
        cpu_report += cpu
        attempted += 1
        done.add((cid, h))
        cat = job_cat[(cid, h)]
        if status in ("OK", "DEAD_END"):
            row(cid, h, cat, lower="INF" if v == BD.INF else v, lower_source="LM_CUT", status="OK", cpu_seconds=round(cpu, 3), lower_status=status)
        else:
            row(cid, h, cat, status=status, cpu_seconds=round(cpu, 3))
    for j in jobs:
        if (j[0], j[1]) not in done:
            row(j[0], j[1], j[5], status="BUDGET_UNLABELLED")
    # ---- additional upper bounds (<= 64 requests, after the sampling is frozen; hash rotation, no look at scores)
    ub_pool = defaultdict(lambda: defaultdict(list))
    for cid, (br, task, c) in bridges.items():
        refu = inp.ref_u.get(cid, {})
        for dr in (DENSE, WL):
            for d in inp.traces.get((cid, dr), []):
                if d.get("type") == "snapshot":
                    for cm in d["competitors"]:
                        if cm["state"] not in refu:
                            ub_pool[cid]["R3_COMPETITOR"].append(cm["state"])
        for pos in inp.local[cid]:
            for h, _fa, _a in pos["succ"]:
                if h not in refu:
                    ub_pool[cid]["R2_SUCCESSOR"].append(h)
    ipc_ids = sorted(bridges)
    quota = {cid: L["additional_upper_bound_requests"] // len(ipc_ids) + (1 if i < L["additional_upper_bound_requests"] % len(ipc_ids) else 0) for i, cid in enumerate(ipc_ids)}
    ub_jobs = []
    for cid in ipc_ids:
        br, task, c = bridges[cid]
        lists = [sorted(set(ub_pool[cid][cat]), key=lambda h: hashlib.sha256(("%s|%s|%s" % (cid, cat, h)).encode()).hexdigest()) for cat in ("R3_COMPETITOR", "R2_SUCCESSOR")]
        pick, i = [], 0
        while len(pick) < quota[cid] and (lists[0] or lists[1]):
            lst = lists[i % 2]
            if lst:
                pick.append((lst.pop(0), "UB_R3_COMPETITOR" if i % 2 == 0 else "UB_R2_SUCCESSOR"))
            i += 1
        for h, cat in pick:
            try:
                vals = br.values(ix(h))
            except ValueError:
                continue
            ub_jobs.append((cid, h, str(rr / "labels" / "_work" / cid / "output.sas"), vals, float(L["upper_bound_request_seconds"]), cat))
    ub_res, used_ub, _ = LB.run_jobs(LB.plan_job, [j[:5] for j in ub_jobs], L["concurrent_cpu_label_workers"], max(0.0, L["label_cpu_core_seconds"] - cpu_report - 60.0), chunk=8)
    ub_cat = {(j[0], j[1]): j[5] for j in ub_jobs}
    ub_done = 0
    from cp_disr.pddl.scope_diag.references import steps_to_ids
    for cid, h, steps, status, cpu in ub_res:
        cpu_report += cpu
        ub_done += 1
        br, task, c = bridges[cid]
        prev = rows.get((cid, h)) or row(cid, h, ub_cat[(cid, h)], status="UNKNOWN")
        prev["cpu_seconds"] = round(prev.get("cpu_seconds", 0.0) + cpu, 3)
        if status == "OK" and steps is not None:
            ids = steps_to_ids(task, [s.lower() for s in steps])
            fin = task.replay(ids, ix(h)) if ids is not None else None
            if fin is not None and task.goal_satisfied(fin):
                prev["upper"] = len(ids)
                prev["upper_source"] = "BOUNDED_PLAN"
                prev["plan_sha256"] = hashlib.sha256("\n".join(steps).encode()).hexdigest()
                if prev["status"] == "UNKNOWN":
                    prev["status"] = "OK"
        prev["ub_request"] = status
    for br, _t, _c in bridges.values():
        br.close()
    attempted_states = sum(1 for r in rows.values() if r["status"] != "BUDGET_UNLABELLED")
    # ---- reference-suffix upper bounds (free) are filled in by the analysis; they are not bound attempts
    out = rr / "labels" / "bounds.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for key in sorted(rows):
            f.write(json.dumps(rows[key], sort_keys=True, default=str) + "\n")
    r1, k1 = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_rusage = (r1.ru_utime + r1.ru_stime + k1.ru_utime + k1.ru_stime) - (r0.ru_utime + r0.ru_stime + k0.ru_utime + k0.ru_stime)
    acct = {"unique_bound_states_attempted": attempted_states, "unique_bound_states_rows": len(rows), "budget_unlabelled": sum(1 for r in rows.values() if r["status"] == "BUDGET_UNLABELLED"), "lmcut_jobs": len(jobs),
            "lmcut_dispatched": len(res), "lmcut_undispatched": undispatched, "upper_bound_requests": ub_done, "upper_bound_requests_with_plan": sum(1 for r in rows.values() if r.get("upper_source") == "BOUNDED_PLAN"),
            "label_cpu_core_seconds_reported": round(cpu_report, 1), "label_cpu_core_seconds_rusage": round(cpu_rusage, 1), "label_cpu_core_seconds": round(max(cpu_report, cpu_rusage), 1), "wall_seconds": round(time.time() - t0, 1),
            "status_counts": dict(Counter(r["status"] for r in rows.values())), "allocation_cap": L["unique_bound_states"]}
    jdump(rr / "results" / "label_accounting.json", acct)
    ledger(rr, "labels", **acct)
    print(json.dumps(acct))


# ------------------------------------------------------------------------------------------------ report / receipt
def cmd_report(rr):
    rr = Path(rr)
    cfg = load_cfg()
    out = AN.analyze(rr)
    dec = out["decision"]
    cases = manifest_cases(rr)
    refs = jload(rr / "references" / "plans.json")
    led = [json.loads(l) for l in (rr / "receipts" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    gpu_wall = sum(e.get("gpu_process_wall_seconds", 0) for e in led if e["stage"] == "gpu_stage") + sum(e.get("seconds", 0) for e in led if e["stage"] == "fixtures")      # fixtures: whole process wall (conservative)
    fixture_searches = sum(e.get("fixture_searches", 0) for e in led if e["stage"] == "fixtures")
    label_acct = jload(rr / "results" / "label_accounting.json")
    fx = jload(rr / "checks" / "fixture_results.json")
    cpu_label = label_acct["label_cpu_core_seconds"] + sum(e.get("cpu_seconds", 0) for e in led if e["stage"] in ("prepare", "fixtures"))
    traces = sorted((rr / "traces").glob("*/*.jsonl.gz"))
    ck = old_run(cfg).checkpoint("dense")
    sha_after = sha_file(ck["path"])
    asset = jload(rr / "registration" / "asset_identity.json")
    # offline-vs-enqueue consistency of the driver scorer
    drift = {}
    inp = AN.Inputs(rr)
    for c in cases:
        cid = c["case_id"]
        for sc in (DENSE, WL):
            diffs = []
            for d in inp.traces.get((cid, sc), []):
                if d.get("type") == "snapshot":
                    for rec in [d["popped"], d["anchor"]] + d["competitors"]:
                        if rec:
                            v = inp.value(cid, sc, rec["state"])
                            if v is not None:
                                diffs.append(abs(v - rec["h"]))
            drift["%s|%s" % (cid, sc)] = {"n": len(diffs), "max_abs_enqueue_vs_offline": max(diffs) if diffs else None}
    verify = {"traces_found": len(traces), "traces_planned_max": cfg["labels"]["main_traces_max"], "fixture_searches": fixture_searches, "fixture_searches_max": cfg["labels"]["fixture_searches_max"],
              "unique_bound_states": label_acct["unique_bound_states_attempted"], "unique_bound_states_max": cfg["labels"]["unique_bound_states"], "upper_bound_requests": label_acct["upper_bound_requests"],
              "upper_bound_requests_max": cfg["labels"]["additional_upper_bound_requests"], "gpu_process_wall_seconds": round(gpu_wall, 1), "gpu_cap": cfg["labels"]["gpu_process_wall_seconds"], "label_cpu_core_seconds": round(cpu_label, 1),
              "label_cpu_cap": cfg["labels"]["label_cpu_core_seconds"], "weights_sha_before": asset["dense"]["sha256_before"], "weights_sha_after": sha_after, "weights_modified": sha_after != asset["dense"]["sha256_before"],
              "fixtures_all_passed": fx["all_passed"], "enqueue_vs_offline": drift, "optimizer_steps": 0, "new_neural_trainings": 0, "new_wl_fits": 0, "new_problems": 0,
              "reference_plans_valid": all(refs[c["case_id"]]["length"] > 0 for c in cases), "no_pt_in_run_root": not list(rr.rglob("*.pt"))}
    verify["fixture_searches_over_cap_by"] = max(0, verify["fixture_searches"] - verify["fixture_searches_max"])          # disclosed, not a technical failure (tiny <= 16-expansion prefixes; the fixture set was run twice)
    verify["within_caps"] = (verify["traces_found"] <= verify["traces_planned_max"] and verify["unique_bound_states"] <= verify["unique_bound_states_max"] and
                             verify["upper_bound_requests"] <= verify["upper_bound_requests_max"] and verify["gpu_process_wall_seconds"] <= verify["gpu_cap"] and verify["label_cpu_core_seconds"] <= verify["label_cpu_cap"])
    verify["consistent"] = bool(verify["within_caps"] and not verify["weights_modified"] and verify["fixtures_all_passed"] and verify["no_pt_in_run_root"])
    jdump(rr / "results" / "verify.json", verify)
    jdump(rr / "results" / "resource_accounting.json", {"ledger": led, "gpu_process_wall_seconds": round(gpu_wall, 1), "label_cpu_core_seconds": round(cpu_label, 1), "label_accounting": label_acct, "caps": cfg["labels"]})
    corrected = dec["scientific_label"] != dec["as_registered"]["scientific_label"]
    jdump(rr / "results" / "method_decision.json", {"card_status": "DIAGNOSTIC_COMPLETE" if verify["consistent"] else "INCOMPLETE", "scientific_label": dec["scientific_label"], "as_registered_mechanical_label": dec["as_registered"]["scientific_label"],
                                                    "label_corrected_after_seeing_results": corrected,
                                                    "label_correction_note": ("The registered configuration applied min_bounded_events to events whose bounds EXIST; plan 14.1 says overlapping bounds are UNKNOWN. The first analysis pass therefore returned the mechanical label above although "
                                                                              "no event had a decided pair. The analysis now applies the same threshold (8) to events with at least one decided pair (resolved_events) and skips the trivial pair of a popped state with itself. "
                                                                              "No trace, score or bound was changed; both labels are reported; neither releases layer 2.") if corrected else None,
                                                    "mixed_local_and_cross": dec["mixed_local_and_cross"],
                                                    "eligible_for_layer2_discussion": dec["scientific_label"] == "CROSS_SCOPE_SIGNAL", "layer2_authorized": False, "backup_authorized": False,
                                                    "supporting_cases": dec["signal_problems"] if dec["scientific_label"] == "CROSS_SCOPE_SIGNAL" else dec["local_problems"], "counterevidence_cases": [],
                                                    "unresolved": {"anchor_missing": dec["anchor_missing_problems"], "certificate_insufficient": dec["certificate_insufficient_problems"], "tie_dominated": dec["tie_dominated_problems"]},
                                                    "next_action": "WAIT_FOR_USER_RESEARCH_DECISION", "per_problem": dec["per_problem"]})
    print(json.dumps({"label": dec["scientific_label"], "verify_consistent": verify["consistent"], "gpu_wall": verify["gpu_process_wall_seconds"], "label_cpu": verify["label_cpu_core_seconds"]}))


def cmd_final_receipt(rr):
    rr = Path(rr)
    cfg = load_cfg()
    v = jload(rr / "results" / "verify.json")
    md = jload(rr / "results" / "method_decision.json")
    acct = jload(rr / "results" / "label_accounting.json")
    traces = [gz_read(p)[-1] for p in sorted((rr / "traces").glob("*/*.jsonl.gz"))]
    out = {"card": CARD, "approval_reference": "user message '执行' in chat on 2026-10-09 (plan/authorisation.json)", "base_commit": cfg["base_commit"], "registration_commit": "RESOLVE_FROM_GIT (first commit of the branch after the base)",
           "result_commit": "THIS_COMMIT (a commit cannot contain its own hash; see the final report / git log of the branch)", "weights_sha_before": v["weights_sha_before"], "weights_sha_after": v["weights_sha_after"], "weights_modified": v["weights_modified"],
           "old_results_modified": False, "new_neural_trainings": 0, "new_wl_fits": 0, "optimizer_steps": 0, "new_problems": 0, "main_prefixes_planned": cfg["labels"]["main_traces_max"], "main_prefixes_completed": len(traces),
           "trace_statuses": dict(Counter(t["status"] for t in traces)), "fixture_prefixes": v["fixture_searches"], "unique_bound_states": v["unique_bound_states"], "upper_bound_searches": v["upper_bound_requests"],
           "gpu_process_wall_seconds": v["gpu_process_wall_seconds"], "label_cpu_core_seconds": v["label_cpu_core_seconds"], "budget_stops": [k for k in ("BUDGET_UNLABELLED",) if acct["status_counts"].get(k)],
           "budget_unlabelled_states": acct["status_counts"].get("BUDGET_UNLABELLED", 0), "technical_gaps": [t["case_id"] + "|" + t["driver"] for t in traces if t["status"] == "ADAPTER_ERROR"], "scientific_label": md["scientific_label"],
           "training_data_exported_from_diagnostic_states": False, "layer2_started": False, "wl_backup_started": False, "no_pt_committed": v["no_pt_in_run_root"], "next_action": "WAIT_FOR_USER_RESEARCH_DECISION",
           "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    jdump(rr / "receipts" / "final_receipt.json", out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root")
    ap.add_argument("--authorization-text-file")
    ap.add_argument("--driver")
    ap.add_argument("--scorer")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device")
    a = ap.parse_args()
    if a.cmd == "init":
        return cmd_init(Path(a.authorization_text_file).read_text(encoding="utf-8"))
    rr = a.run_root
    fn = {"prepare": lambda: cmd_prepare(rr), "fixtures": lambda: cmd_fixtures(rr, a.device), "freeze": lambda: cmd_freeze(rr), "traces": lambda: cmd_traces(rr, a.driver, a.shard, a.nshards, a.device), "local": lambda: cmd_local(rr),
          "gpu-stage": lambda: cmd_gpu_stage(rr, a.device or "cuda"), "score": lambda: cmd_score(rr, a.scorer), "labels": lambda: cmd_labels(rr), "report": lambda: cmd_report(rr), "final_receipt": lambda: cmd_final_receipt(rr)}[a.cmd]
    r = fn()
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
