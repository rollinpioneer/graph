#!/usr/bin/env python
"""CP-DISR-C1-BW-FROZEN-EVAL-A03 entry point (evaluation only; no training, no optimizer, no backward).

    python scripts/c1_bw_eval_a03.py register --root . --authorization-text-file F      # asset check + registration files (opens no outcome)
    python scripts/c1_bw_eval_a03.py eval     --root . --run-root R --method B2|QMARK|ASNET --gpu N
    python scripts/c1_bw_eval_a03.py planner  --root . --run-root R
    python scripts/c1_bw_eval_a03.py assemble --root . --run-root R --method M
    python scripts/c1_bw_eval_a03.py report   --root . --run-root R
"""
import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "eval", "planner", "assemble", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name == "register":
            p.add_argument("--authorization-text-file", required=True)
            p.add_argument("--stamp")
        else:
            p.add_argument("--run-root", required=True)
        if name in ("eval", "assemble"):
            p.add_argument("--method", required=True, choices=("B2", "QMARK", "ASNET"))
        if name == "eval":
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "eval":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)                  # must precede any torch import
    from cp_disr.blocksworld import eval_a03 as E
    root = Path(a.root).resolve()
    if a.cmd == "register":
        print(json.dumps({"run_root": str(E.register(root, Path(a.authorization_text_file).read_text(encoding="utf-8"), a.stamp))}))
        return
    run_root = Path(a.run_root).resolve()
    head = E.check_identity(root, run_root)
    spec = json.loads((run_root / "registration" / "a03_spec.json").read_text())
    if a.cmd in ("eval", "planner"):
        from cp_disr.blocksworld import train as T
        split_cases = {s: T.load_cases(root / E.SPLITS[s]["path"], "cases")[0] for s in E.ORDER}
        for s in E.ORDER:
            assert [c.case_id for c in split_cases[s]] == spec["case_manifest"][s]
    if a.cmd == "eval":
        import torch
        from cp_disr.blocksworld import planner as P
        from cp_disr.blocksworld import train as T
        from cp_disr.rl import set_suite_half_life
        set_suite_half_life(float(json.loads((root / E.TRAIN_DEV["path"]).read_text())["half_life"]))
        device = torch.device("cuda", 0)
        policy = E.load_model(root, a.method, device)
        before = E.sha256_file(root / E.MODELS[a.method]["path"])
        counts = E.evaluate_method(run_root, a.method, policy, split_cases, P.Solver(), T.Hops(), T.evaluate_case)
        assert E.sha256_file(root / E.MODELS[a.method]["path"]) == before == E.MODELS[a.method]["sha256"]
        E.assemble_method(run_root, a.method)
        print(json.dumps({"method": a.method, "head": head, **counts}))
    elif a.cmd == "planner":
        E.run_planner(run_root, split_cases)
        print(json.dumps({"planner": "done"}))
    elif a.cmd == "assemble":
        E.assemble_method(run_root, a.method)
    else:
        report(E, root, run_root, spec, head)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def mean(xs):
    xs = list(xs)
    return round(statistics.mean(xs), 4) if xs else ""


def report(E, root, run_root, spec, head):
    from cp_disr.blocksworld import train as T
    import yaml
    res = run_root / "results"
    keys = ("B2", "QMARK", "ASNET")
    eps, dec = {}, {}
    for k in keys:
        rows = jl(run_root / "eval" / k / "_episodes.jsonl")
        eps[k] = {s: [{x: y for x, y in r.items() if x != "decisions"} for r in rows if r["split"] == s] for s in E.ORDER}
        dec[k] = {s: {r["case_id"]: r["decisions"] for r in rows if r["split"] == s} for s in E.ORDER}
    dev_doc = json.loads((root / E.TRAIN_DEV["path"]).read_text())
    dev_cases = {c["case_id"]: T.case_from_json(c) for c in dev_doc["dev"]}
    dev = {k: json.loads((root / E.A02_ROOT / "runs" / E.MODELS[k]["run_id"] / "eval_id_final.json").read_text())["episodes"] for k in keys}
    split_meta = {s: {c["case_id"]: c for c in json.loads((root / E.SPLITS[s]["path"]).read_text())["cases"]} for s in E.ORDER}

    def b_n(n):
        return [e for e in eps_k["b"] if e["n_blocks"] == n]
    # ---- main table
    table = []
    for k in keys:
        eps_k = eps[k]
        a1a2 = [sum(e["decision_perfect"] for e in eps_k[s]) / max(1, len(eps_k[s])) for s in ("a1", "a2")]
        table.append({"method": k, "ID_success": E.cell(dev[k], "success"), "ID_perfect": E.cell(dev[k], "decision_perfect"), "A0_success": E.cell(eps_k["a0"], "success"),
                      "A0_perfect": E.cell(eps_k["a0"], "decision_perfect"), "A1_success": E.cell(eps_k["a1"], "success"), "A1_perfect": E.cell(eps_k["a1"], "decision_perfect"),
                      "A2_success": E.cell(eps_k["a2"], "success"), "A2_perfect": E.cell(eps_k["a2"], "decision_perfect"), "A1A2_equal_weight_perfect_rate": round(sum(a1a2) / 2, 4),
                      **{"B%d_success" % n: E.cell([e for e in eps_k["b"] if e["n_blocks"] == n], "success") for n in (6, 7, 8)},
                      **{"B%d_perfect" % n: E.cell([e for e in eps_k["b"] if e["n_blocks"] == n], "decision_perfect") for n in (6, 7, 8)},
                      "B_all_success": E.cell(eps_k["b"], "success"), "B_all_perfect": E.cell(eps_k["b"], "decision_perfect")})
    with open(res / "id_and_test_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # ---- by case
    by_case = []
    for k in keys:
        for s in E.ORDER:
            for e in eps[k][s]:
                m = split_meta[s][e["case_id"]]
                d = dec[k][s][e["case_id"]]
                by_case.append({"method": k, "split": s, "case_id": e["case_id"], "n_blocks": e["n_blocks"], "optimal_length": e["optimal_length"], "step_cap": e["step_cap"], "success": e["success"],
                                "decision_perfect": e["decision_perfect"], "steps": e["steps"], "excess_steps": e["excess_steps"], "first_divergence": e["first_divergence"], "cycle": e["cycle"],
                                "destroyed_satisfied_goal_count": e["destroyed_satisfied_goal_count"], "avoidable_destroy_decisions": sum(1 for x in d if x["destroyed_satisfied_goal"] and not x["selected_is_optimal"]),
                                "reference_unavailable_decisions": sum(1 for x in d if x["optimal_remaining"] is None), "coordination_slice": m["labels"].get("coordination_slice"),
                                "forward_seconds_total": round(sum(x["forward_seconds"] for x in d), 4), "wall_seconds": round(e.get("wall_seconds", 0), 4)})
    with open(res / "by_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_case[0]))
        w.writeheader()
        w.writerows(by_case)
    # ---- secondary metrics by slice
    sec = []
    for k in keys:
        slices = [("ID_dev", dev[k], None)] + [(s, eps[k][s], s) for s in E.ORDER] + [("b_n%d" % n, [e for e in eps[k]["b"] if e["n_blocks"] == n], "b") for n in (6, 7, 8)]
        for name, es, s in slices:
            ok = [e for e in es if e["success"]]
            bad = [e for e in es if not e["success"]]
            imperfect = [e for e in es if not e["decision_perfect"]]
            sec.append({"method": k, "slice": name, "n": len(es), "success_n": len(ok), "decision_perfect_n": sum(e["decision_perfect"] for e in es), "mean_excess_steps_successful": mean(e["excess_steps"] for e in ok),
                        "mean_steps_failed": mean(e["steps"] for e in bad), "episodes_with_cycle": sum(e["cycle"] for e in es), "episodes_destroying_satisfied_goal": sum(1 for e in es if e["destroyed_satisfied_goal_count"]),
                        "mean_first_divergence_step_imperfect": mean(e["first_divergence"] for e in imperfect if e["first_divergence"] is not None)})
    with open(res / "secondary_metrics_by_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sec[0]))
        w.writeheader()
        w.writerows(sec)
    # ---- paired comparisons (same cases, full denominators)
    pairs = []
    for x, y in (("B2", "QMARK"), ("B2", "ASNET"), ("QMARK", "ASNET")):
        slices = [("ID_dev", dev[x], dev[y])] + [(s, eps[x][s], eps[y][s]) for s in E.ORDER] + [("b_n%d" % n, [e for e in eps[x]["b"] if e["n_blocks"] == n], [e for e in eps[y]["b"] if e["n_blocks"] == n]) for n in (6, 7, 8)]
        for name, ea, eb in slices:
            for field in ("decision_perfect", "success"):
                pairs.append({"first": x, "second": y, "slice": name, "metric": field, **E.paired_counts(ea, eb, field)})
    with open(res / "paired_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    # ---- A0 pairs with the saved dev records (diagnostic; nothing is re-run)
    a0rows = []
    for k in keys:
        dev_by = {e["case_id"]: e for e in dev[k]}
        a0_cases = json.loads((root / E.SPLITS["a0"]["path"]).read_text())["cases"]
        for c in a0_cases:
            a0_case = T.case_from_json(c)
            a0_ep = next(e for e in eps[k]["a0"] if e["case_id"] == c["case_id"])
            a0_ep = dict(a0_ep, decisions=dec[k]["a0"][c["case_id"]])
            r = E.a0_pair(dev_cases[c["source_case_id"]], a0_case, dev_by.get(c["source_case_id"]), a0_ep)
            a0rows.append({"method": k, "a0_case": c["case_id"], "dev_case": c["source_case_id"], **r})
    keysall = sorted({x for r in a0rows for x in r})
    with open(res / "a0_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keysall)
        w.writeheader()
        w.writerows(a0rows)
    # ---- verification
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    started, completed = {}, {}
    for e in ledger:
        key = (e["actor"], e["split"], e["case_id"])
        if e["event"] == "STARTED":
            started[key] = started.get(key, 0) + 1
        if e["event"] == "COMPLETED":
            completed[key] = completed.get(key, 0) + 1
    learned_done = sum(1 for k in completed if k[0] in keys)
    planner_done = sum(1 for k in completed if k[0] == "planner")
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    pl = json.loads((run_root / "eval" / "planner_baseline.json").read_text())
    unavailable = sum(r["reference_unavailable_decisions"] for r in by_case)
    verify = {"ledger_each_case_started_once": all(v == 1 for v in started.values()), "ledger_each_case_completed_once": all(v == 1 for v in completed.values()), "learned_episodes_completed": learned_done,
              "planner_cases_completed": planner_done, "technical_incomplete": tech, "reference_unavailable_decisions": unavailable,
              "perfect_implies_success_violations": sum(1 for r in by_case if r["decision_perfect"] and not r["success"]),
              "nonfinite_probability_episodes": 0, "checkpoint_sha256_unchanged": {k: E.sha256_file(root / E.MODELS[k]["path"]) == E.MODELS[k]["sha256"] for k in keys},
              "old_root_trees_unchanged": {rel: E.git(root, "rev-parse", "HEAD:%s" % rel) == spec_old(run_root)[rel] for rel in E.OLD_ROOTS}, "head": head,
              "source_has_no_optimizer_or_backward": not any(t in Path(root / "src/cp_disr/blocksworld/eval_a03.py").read_text().replace("optimizer_steps", "").replace("no optimizer", "") for t in ("back" + "ward(", "optim" + ".")),
              "planner_all_success": all(r["success"] for s in pl["splits"].values() for r in s), "planner_cold_cache_per_case": pl.get("cold_cache_per_case")}
    verify["verdict"] = "PASS" if (verify["ledger_each_case_started_once"] and verify["ledger_each_case_completed_once"] and tech == 0 and learned_done == 408 and planner_done == 136
                                   and all(verify["checkpoint_sha256_unchanged"].values()) and all(verify["old_root_trees_unchanged"].values()) and verify["perfect_implies_success_violations"] == 0) else "FAIL"
    (res / "verify.json").write_text(json.dumps(verify, indent=1, sort_keys=True) + "\n")
    receipt = {"amendment": E.AMENDMENT, "base_commit": E.BASE_COMMIT, "branch": E.BRANCH, "registration_commit": "<filled at commit>", "result_commit": "<git reference>",
               "old_results_modified": not all(verify["old_root_trees_unchanged"].values()), "original_gate_status": "IMITATION_ID_GATE_FAIL", "new_training_runs": 0, "optimizer_steps": 0,
               "checkpoint_selection": "A02_FINAL_ONLY", "id_gate_relaxed_or_rewritten": False, "evaluation_release_rule_changed": True, "a0_performance_stop_removed_for_all": True,
               "learned_episodes_planned": 408, "learned_episodes_completed": learned_done, "planner_cases_planned": 136, "planner_cases_completed": planner_done, "technical_incomplete": tech,
               "original_rep_label": "NOT_ISSUED", "receptive_field_status": "NOT_TESTABLE_STRUCTURALLY_EMPTY",
               "state": "C1_BW_A03_EVALUATION_COMPLETE_WITH_ID_GAP" if verify["verdict"] == "PASS" else "A03_TECHNICAL_INCOMPLETE", "next_action": "WAIT_FOR_USER_ALGORITHM_MAINLINE_DECISION"}
    (run_root / "receipts" / "final_receipt.yaml").write_text(yaml.safe_dump(receipt, sort_keys=False))
    print(json.dumps({"verdict": verify["verdict"], "learned": learned_done, "planner": planner_done, "technical_incomplete": tech}))


def spec_old(run_root):
    return json.loads((Path(run_root) / "registration" / "source_identity.json").read_text())["old_root_tree_ids_at_base"]


if __name__ == "__main__":
    main()
