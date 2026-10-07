#!/usr/bin/env python
"""C1-BW-SCORER-CONTROL-V1 entry point (zero training).

    prep    --gpu N --authorization-text-file F   # fixtures, fresh 112-problem set + isolation manifest, registration / identity
    eval    --run-root R --condition B_C3|B_G1C3|MG_C3|MG_G1C3 --gpu N
    planner --run-root R
    report  --run-root R
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NEW_SOURCES = ("src/cp_disr/blocksworld/scorer_control.py", "scripts/c1_bw_scorer_control.py", "tests/test_c1_bw_scorer_control.py",
               "docs/c1_blocksworld/CP_DISR_C1_Attribution_Closeout_and_Execution_Control_Check_v1.md", "configs/splits/c1_bw_scorer_control_fresh112_v1.json")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wj(p, doc):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def wcsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        if rows:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def shaobj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ prep
def cmd_prep(root, auth):
    import torch
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import scorer_control as C
    from cp_disr.blocksworld import splits as SP
    from cp_disr.blocksworld import state as S
    from cp_disr.rl import set_suite_half_life
    if E.git(root, "branch", "--show-current") != C.BRANCH or E.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("clean tracked tree on %s required" % C.BRANCH)
    if E.git(root, "rev-parse", "HEAD") != C.BASE_COMMIT:
        raise SystemExit("HEAD must equal the base commit before registration")
    b2, mg = root / E.MODELS["B2"]["path"], root / C.M1GOAL["path"]
    assert E.sha256_file(b2) == E.MODELS["B2"]["sha256"] and b2.stat().st_size == E.MODELS["B2"]["bytes"]
    assert E.sha256_file(mg) == C.M1GOAL["sha256"] and mg.stat().st_size == C.M1GOAL["bytes"]
    dtp = root / A.GP_ROOT / "datasets" / "D_train.json"
    assert E.sha256_file(dtp) == A.D_TRAIN["file_sha256"]
    d_train = json.loads(dtp.read_text())
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    set_suite_half_life(half_life)
    run_root = root / C.SC_REL / ("%s_%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), E.sha256_file(root / C.PLAN_REL)[:8]))
    for sub in ("prep", "eval", "results", "receipts"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    prep = run_root / "prep"
    # fixtures: train-split cases and artificial logits only
    fx_rules = C.check_rule_fixtures(cases)
    fx_load = C.check_scorer_loading(root, cases, torch.device("cpu"))
    wj(prep / "fixtures.json", {"rules": fx_rules, "scorers": fx_load})
    print(json.dumps({"fx_rules": fx_rules["pass"], "fx_scorers": fx_load["pass"], "decisions": fx_rules["decisions_compared"]}), flush=True)
    if not (fx_rules["pass"] and fx_load["pass"]):
        raise SystemExit("fixture failed: fix the implementation before registering")
    # fresh set
    forbidden, by_source = C.excluded_hashes(root, d_train, cases)
    excluded_at_start = set(forbidden)
    b = SP.Builder()
    out, report, rejects_all = [], {}, {}

    def rec(cid, p, q, extra):
        return SP.case_record(cid, "scorer_control_fresh", p, q, A._extra(p, q, **{"namespace": C.NAMESPACE, **extra}))
    for gname, n, heights in A.STRUCTURE_GROUPS + (("H4", 6, (4, 2)),):
        for stratum in ("MONO", "BREAK"):
            rej = Counter()
            made, att = A.gen_cell(b, forbidden, "%s:%s:%s" % (C.NAMESPACE, gname, stratum), n, heights, S.RED, stratum == "BREAK", A.PER_CELL, 0.88, A.CELL_MAX_ATTEMPTS, rej)
            layer = ("T-" + stratum) if gname != "H4" else "H4"
            for i, (p, q, h) in enumerate(made):
                out.append(rec("BW_scf_%s_%s_%02d" % (gname, stratum, i), p, q, {"cell": "%s|%s" % (gname, stratum), "structure_group": gname, "stratum": stratum, "layer": layer, "goal_heights": sorted(heights, reverse=True)}))
            report["%s|%s" % (gname, stratum)] = {"planned": A.PER_CELL, "built": len(made), "attempts": att, "shortfall": A.PER_CELL - len(made)}
            rejects_all["%s|%s" % (gname, stratum)] = dict(rej)
            print(json.dumps({"%s|%s" % (gname, stratum): report["%s|%s" % (gname, stratum)]}), flush=True)
    for stratum in ("MONO", "BREAK"):
        rej = Counter()
        pairs, att = A.gen_color_pairs(b, forbidden, "%s:COLOR:%s" % (C.NAMESPACE, stratum), stratum == "BREAK", 8, A.COLOR_MAX_ATTEMPTS, rej)
        for i, ((p, q, h), (pb, qb, hb)) in enumerate(pairs):
            skel = shaobj([list(p.names), list(p.init), list(p.goal)])
            for var, (pp, qq, hh) in (("RED", (p, q, h)), ("BLUE", (pb, qb, hb))):
                out.append(rec("BW_scf_COLOR_%s_%02d_%s" % (stratum, i, var), pp, qq, {"cell": "COLOR|%s" % stratum, "structure_group": "COLOR", "stratum": stratum, "layer": "COLOR", "goal_heights": [3, 1, 1],
                                                                                    "pair_id": "CP_%s_%02d" % (stratum, i), "color_variant": var, "skeleton_hash": skel}))
        report["COLOR|%s" % stratum] = {"planned_pairs": 8, "built_pairs": len(pairs), "attempts": att, "shortfall_pairs": 8 - len(pairs)}
        rejects_all["COLOR|%s" % stratum] = dict(rej)
        print(json.dumps({"COLOR|%s" % stratum: report["COLOR|%s" % stratum]}), flush=True)
    hashes = [c["problem_iso_hash"] for c in out]
    overlap = sorted(set(hashes) & excluded_at_start)
    assert not overlap, overlap
    ids = [c["case_id"] for c in out]
    assert len(ids) == len(set(ids))
    doc = {"namespace": C.NAMESPACE, "excluded_problem_classes_at_start": len(excluded_at_start), "cells": report, "counts": len(out), "cases": out,
           "note": "fresh set for the scorer-control card; no model or rule was queried to select, filter or order any problem"}
    wj(root / C.FRESH_REL, doc)
    wj(prep / "generation_shortfalls.json", {"cells": report, "rejection_counts": rejects_all, "frozen_cases": len(out)})
    wj(prep / "isolation_manifest.json", {"excluded_by_source": by_source, "excluded_total_at_start": len(excluded_at_start), "excluded_prior_split_files": list(C.PRIOR_SPLITS) + ["c1_bw_train_dev_v1 (train+dev)", "D_train hand-empty problems"],
                                          "overlap_with_excluded": 0, "fresh_problem_classes": len(set(hashes)), "note": "colour-preserving problem_iso_hash classes (renamed copies excluded); colour twins are checked as pairs"})
    wj(prep / "fresh_manifest.json", {"sha256": E.sha256_file(root / C.FRESH_REL), "counts": len(out), "layers": dict(Counter(c["layer"] + "|" + c["stratum"] for c in out)), "case_ids": ids,
                                      "color_pairs": sorted({c["pair_id"] for c in out if "pair_id" in c})})
    wj(prep / "weight_identity.json", {"B2_A02": {k: E.MODELS["B2"][k] for k in ("run_id", "path", "sha256", "bytes")}, "M1_GOAL": C.M1GOAL, "verified_on_server_bytes": True, "committed_to_git": False})
    wj(prep / "source_identity.json", C.source_identity(root, NEW_SOURCES))
    reg = {"card": C.CARD, "plan": C.PLAN_REL, "plan_sha256": E.sha256_file(root / C.PLAN_REL), "base_commit": C.BASE_COMMIT, "branch": C.BRANCH, "authorization_text": auth, "authorization_text_sha256": hashlib.sha256(auth.encode()).hexdigest(),
           "conditions": {k: {"scorer": C.SCORER[k], "rule": C.RULE[k]} for k in C.CONDITIONS}, "new_training_runs": 0, "optimizer_steps": 0,
           "fresh": {"path": C.FRESH_REL, "sha256": E.sha256_file(root / C.FRESH_REL), "frozen_cases": len(out), "shortfalls": {k: v for k, v in report.items() if v.get("shortfall") or v.get("shortfall_pairs")}},
           "planned_episodes": len(C.CONDITIONS) * len(out), "planner_references": len(out), "dev_reruns": 0, "thresholds": {"N>=32": 5, "16<=N<32": "ceil(5N/32)", "N<16": "no label", "pairs16": 3, "note": "decision aids, not significance"},
           "step_cap": "2*L*+4", "fixtures": {"rules": fx_rules["pass"], "scorers": fx_load["pass"]}, "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_root_naming": "<UTC>_<plan sha8>"}
    wj(prep / "registration.json", reg)
    print(json.dumps({"run_root": str(run_root), "fresh_cases": len(out)}))


# ------------------------------------------------------------------------------------------------ eval / planner
def fresh_cases(root):
    from cp_disr.blocksworld import scorer_control as C
    from cp_disr.blocksworld import train as T
    doc = json.loads((root / C.FRESH_REL).read_text())
    by, meta = defaultdict(list), {}
    for c in doc["cases"]:
        by[c["cell"]].append(T.case_from_json(c))
        meta[c["case_id"]] = c
    return by, meta


def cmd_eval(root, run_root, cond, gpu):
    import torch
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import scorer_control as C
    from cp_disr.rl import set_suite_half_life
    C.check_identity(root, run_root)
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    set_suite_half_life(float(td["half_life"]))
    device = torch.device("cuda", 0)
    policy = C.load_scorer(root, cond, device)
    d0 = C.weights_digest(policy)
    chooser = C.chooser_for(cond)
    by, _ = fresh_cases(root)
    solver = P.Solver()
    counts = R.run_cases(run_root, cond, [(c, by[c]) for c in sorted(by)], lambda c: A.run_episode_with(policy, c, solver, chooser), run_root / "eval" / cond / "episodes.jsonl")
    m = getattr(policy, "model", policy)
    d1 = C.weights_digest(policy)
    wj(run_root / "eval" / cond / "weight_integrity.json", {"digest_before": d0, "digest_after": d1, "unchanged": d0 == d1, "any_grad_tensor": any(p.grad is not None for p in m.parameters()), "optimizer_created": False, "counts": counts})
    print(json.dumps(counts))


def cmd_planner(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import scorer_control as C
    from cp_disr.blocksworld import state as S
    C.check_identity(root, run_root)
    by, _ = fresh_cases(root)

    def ref(c):
        s = P.Solver()
        t0 = time.perf_counter()
        plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
        cpu = time.perf_counter() - t0
        st = c.problem.init
        for a in plan:
            st = S.apply(st, a)
        return {"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st), "expanded_nodes": s.stats.expanded, "wall_seconds_cold_cache": cpu}
    print(json.dumps(R.run_cases(run_root, "planner", [(c, by[c]) for c in sorted(by)], ref, run_root / "eval" / "planner" / "episodes.jsonl")))


# ------------------------------------------------------------------------------------------------ report
PAIRS = (("MG_C3", "B_C3", "scorer swap under C3"), ("MG_G1C3", "B_G1C3", "scorer swap under G1+C3"), ("MG_G1C3", "MG_C3", "G1 added, new scorer"), ("B_G1C3", "B_C3", "G1 added, old scorer"),
         ("MG_C3", "B_G1C3", "deployment: new scorer + C3 vs current best system"))


def cmd_report(root, run_root):
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import gp_attribution as A
    from cp_disr.blocksworld import scorer_control as C
    res = run_root / "results"
    reg = json.loads((run_root / "prep" / "registration.json").read_text())
    conf = json.loads((root / C.FRESH_REL).read_text())
    meta = {c["case_id"]: c for c in conf["cases"]}
    eps = {c: jl(run_root / "eval" / c / "episodes.jsonl") for c in C.CONDITIONS}
    for c in C.CONDITIONS:
        ids = [e["case_id"] for e in eps[c]]
        assert len(ids) == len(set(ids)) == len(meta), (c, len(ids))
    ep = {c: {e["case_id"]: e for e in eps[c]} for c in C.CONDITIONS}
    planner = {e["case_id"]: e for e in jl(run_root / "eval" / "planner" / "episodes.jsonl")}
    layers = {"T-MONO": lambda m: m["layer"] == "T-MONO", "T-BREAK": lambda m: m["layer"] == "T-BREAK", "H4-MONO": lambda m: m["layer"] == "H4" and m["stratum"] == "MONO",
              "H4-BREAK": lambda m: m["layer"] == "H4" and m["stratum"] == "BREAK", "COLOR-MONO": lambda m: m["layer"] == "COLOR" and m["stratum"] == "MONO",
              "COLOR-BREAK": lambda m: m["layer"] == "COLOR" and m["stratum"] == "BREAK", "COLOR-RED": lambda m: m.get("color_variant") == "RED", "COLOR-BLUE": lambda m: m.get("color_variant") == "BLUE",
              "H4": lambda m: m["layer"] == "H4", "COLOR": lambda m: m["layer"] == "COLOR", "ALL": lambda m: True}
    percase = []
    for c in C.CONDITIONS:
        for cid, e in ep[c].items():
            m = meta[cid]
            percase.append({"condition": c, "case_id": cid, "cell": m["cell"], "layer": m["layer"], "stratum": m["stratum"], "n": m["n_blocks"], "k": m["k_nontrivial_towers"], "h": m["max_tower_height"], "L": m["optimal_length"],
                            "initial_goal_atoms_satisfied": m["initial_goal_atoms_satisfied"], "pair_id": m.get("pair_id"), "color_variant": m.get("color_variant"), "success": e["success"], "decision_perfect": e["decision_perfect"],
                            "steps": e["steps"], "step_cap": e["step_cap"], "excess_steps": e["excess_steps"], "reason": e["reason"], "cycle": e["cycle"], "first_divergence": e["first_divergence"], "interventions": e["interventions"],
                            "first_intervention": e["first_intervention"], "necessary_destruction_steps": e["necessary_destruction_steps"], "avoidable_destruction_steps": e["avoidable_destruction_steps"], "wall_seconds": round(e["wall_seconds"], 4)})
    wcsv(res / "by_case.csv", percase)

    def cnt(c, pred):
        es = [ep[c][cid] for cid, m in meta.items() if pred(m)]
        succ = [e for e in es if e["success"]]
        fail = [e for e in es if not e["success"]]
        return {"n": len(es), "success": len(succ), "perfect": sum(e["decision_perfect"] for e in es), "total_actions": sum(e["steps"] for e in es), "mean_actions_all": round(statistics.mean(e["steps"] for e in es), 2) if es else None,
                "mean_excess_success": round(statistics.mean(e["excess_steps"] for e in succ), 2) if succ else None, "sum_excess_success": sum(e["excess_steps"] for e in succ), "failures": len(fail),
                "failure_reasons": json.dumps(dict(Counter(e["reason"] for e in fail)), sort_keys=True), "failure_actions_spent": sum(e["steps"] for e in fail), "cycles": sum(e["cycle"] for e in es),
                "interventions": sum(e["interventions"] for e in es), "avoidable_destruction_steps": sum(e["avoidable_destruction_steps"] for e in es), "wall_seconds": round(sum(e["wall_seconds"] for e in es), 2)}
    rows = [{"condition": c, "layer": ln, **cnt(c, pred)} for c in C.CONDITIONS for ln, pred in layers.items()]
    wcsv(res / "by_layer_and_cost.csv", rows)
    st = []
    for c in C.CONDITIONS:
        g = defaultdict(list)
        for r in percase:
            if r["condition"] == c:
                g["n=%d k=%d h=%d layer=%s" % (r["n"], r["k"], r["h"], r["layer"] + "|" + r["stratum"])].append(r)
                g["L " + ("1-8" if r["L"] <= 8 else "9-16" if r["L"] <= 16 else "17-24" if r["L"] <= 24 else ">24")].append(r)
                g["initial_satisfied=%d" % r["initial_goal_atoms_satisfied"]].append(r)
        for k, rs in sorted(g.items()):
            st.append({"condition": c, "stratum": k, "n": len(rs), "success": sum(r["success"] for r in rs), "decision_perfect": sum(r["decision_perfect"] for r in rs), "mean_L": round(statistics.mean(r["L"] for r in rs), 2)})
    wcsv(res / "by_structure.csv", st)

    def pair4(a, b, pred, field):
        ids = [cid for cid, m in meta.items() if pred(m)]
        both = sum(ep[a][i][field] and ep[b][i][field] for i in ids)
        oa = sum(ep[a][i][field] and not ep[b][i][field] for i in ids)
        ob = sum(ep[b][i][field] and not ep[a][i][field] for i in ids)
        return {"n": len(ids), "both": both, "only_first": oa, "only_second": ob, "neither": len(ids) - both - oa - ob, "net": oa - ob}
    prs = []
    for a, b, label in PAIRS:
        for ln, pred in layers.items():
            common = [cid for cid, m in meta.items() if pred(m) and ep[a][cid]["success"] and ep[b][cid]["success"]]
            diff = [ep[a][i]["steps"] - ep[b][i]["steps"] for i in common]
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                p = pair4(a, b, pred, field)
                prs.append({"first": a, "second": b, "comparison": label, "layer": ln, "metric": nm, **p, "direction": A.direction(p["net"], p["n"]) if ln not in ("COLOR-RED", "COLOR-BLUE") else "NA",
                            "common_success_n": len(common), "common_success_mean_steps_first_minus_second": round(statistics.mean(diff), 3) if diff else None, "common_success_first_shorter": sum(d < 0 for d in diff), "common_success_first_longer": sum(d > 0 for d in diff)})
    wcsv(res / "paired_comparisons.csv", prs)
    tw = []
    for c in C.CONDITIONS:
        for strat in ("MONO", "BREAK", "ALL"):
            pids = sorted({m["pair_id"] for m in meta.values() if m.get("pair_id") and (strat == "ALL" or m["stratum"] == strat)})
            for field, nm in (("success", "S"), ("decision_perfect", "D")):
                both = ro = bo = nei = 0
                for pid in pids:
                    r = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "RED")
                    bl = next(i for i, m in meta.items() if m.get("pair_id") == pid and m["color_variant"] == "BLUE")
                    x, y = ep[c][r][field], ep[c][bl][field]
                    both += x and y
                    ro += x and not y
                    bo += y and not x
                    nei += (not x) and (not y)
                tw.append({"condition": c, "stratum": strat, "metric": nm, "pairs": len(pids), "both": both, "RED_only": ro, "BLUE_only": bo, "neither": nei, "net_RED_minus_BLUE": ro - bo})
    wcsv(res / "colour_twin_pairs.csv", tw)
    wcsv(res / "planner_reference.csv", [{"case_id": k, "optimal_length": v["optimal_length"], "plan_length": v["plan_length"], "success": v["success"], "expanded_nodes": v["expanded_nodes"], "wall_seconds_cold_cache": round(v["wall_seconds_cold_cache"], 4)} for k, v in planner.items()])
    wjint = {c: json.loads((run_root / "eval" / c / "weight_integrity.json").read_text()) for c in C.CONDITIONS}
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    started, completed = Counter(), Counter()
    for e in ledger:
        k = (e["actor"], e["split"], e["case_id"])
        started[k] += e["event"] == "STARTED"
        completed[k] += e["event"] == "COMPLETED"
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    srcid = json.loads((run_root / "prep" / "source_identity.json").read_text())
    verify = {"episodes_completed": sum(v for k, v in completed.items() if k[0] in C.CONDITIONS), "episodes_planned": reg["planned_episodes"], "planner_completed": sum(v for k, v in completed.items() if k[0] == "planner"),
              "planner_all_optimal": all(v["success"] and v["plan_length"] == v["optimal_length"] for v in planner.values()), "technical_incomplete": tech,
              "ledger_once_each": all(v == 1 for v in started.values()) and all(v == 1 for v in completed.values()), "weights_unchanged_in_eval": all(v["unchanged"] and not v["any_grad_tensor"] for v in wjint.values()),
              "weight_digests": {c: wjint[c]["digest_after"] for c in C.CONDITIONS}, "old_roots_unchanged": all(E.git(root, "rev-parse", "HEAD:%s" % rel) == srcid["old_root_tree_ids_at_base"][rel] for rel in C.OLD_ROOTS),
              "checkpoints_unchanged": E.sha256_file(root / E.MODELS["B2"]["path"]) == E.MODELS["B2"]["sha256"] and E.sha256_file(root / C.M1GOAL["path"]) == C.M1GOAL["sha256"], "frozen_sources_unchanged": srcid["all_frozen_unchanged"],
              "new_training_runs": 0, "optimizer_steps": 0, "dev_reruns": 0, "head": E.git(root, "rev-parse", "HEAD")}
    verify["verdict"] = "PASS" if (verify["episodes_completed"] == verify["episodes_planned"] and verify["planner_completed"] == len(meta) and tech == 0 and verify["ledger_once_each"] and verify["weights_unchanged_in_eval"]
                                   and verify["old_roots_unchanged"] and verify["checkpoints_unchanged"] and verify["planner_all_optimal"]) else "FAIL"
    wj(res / "verify.json", verify)
    names = ("T-MONO", "T-BREAK", "H4-MONO", "H4-BREAK", "COLOR-MONO", "COLOR-BREAK", "ALL")
    lines = ["# C1-BW-SCORER-CONTROL-V1 results (auto-generated numbers)", "", "Cells are success / all-steps-optimal (n).", "", "| condition | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in C.CONDITIONS:
        cells = []
        for ln in names:
            x = cnt(c, layers[ln])
            cells.append("%d / %d (%d)" % (x["success"], x["perfect"], x["n"]))
        lines.append("| %s | %s |" % (c, " | ".join(cells)))
    lines += ["", "Verify: %s (%d/%d episodes, %d planner references)" % (verify["verdict"], verify["episodes_completed"], verify["episodes_planned"], verify["planner_completed"]), ""]
    (res / "final_summary_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verify": verify["verdict"], "completed": verify["episodes_completed"]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "eval", "planner", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name == "prep":
            p.add_argument("--authorization-text-file", required=True)
        else:
            p.add_argument("--run-root", required=True)
        if name == "eval":
            p.add_argument("--condition", required=True, choices=("B_C3", "B_G1C3", "MG_C3", "MG_G1C3"))
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if hasattr(a, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    root = Path(a.root).resolve()
    if a.cmd == "prep":
        cmd_prep(root, Path(a.authorization_text_file).read_text(encoding="utf-8"))
        return
    rr = Path(a.run_root).resolve()
    if a.cmd == "eval":
        cmd_eval(root, rr, a.condition, a.gpu)
    elif a.cmd == "planner":
        cmd_planner(root, rr)
    else:
        cmd_report(root, rr)


if __name__ == "__main__":
    main()
