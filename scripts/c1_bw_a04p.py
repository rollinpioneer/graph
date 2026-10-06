#!/usr/bin/env python
"""CP-DISR C1 goal-progress runbook, phase 1 (C1-BW-A04P): zero-training controllers, goal/time probe, first-error denominators. Evaluation only.

    build-pilot                                   # Pilot80 from the new namespace (planner only, no model)
    register --authorization-text-file F          # registration.json, source/checkpoint identity, split manifest, tau
    denominators --run-root R                     # first-error table from the consumed A03 logs
    pilot --run-root R --controller C0|C1|C3|G1 --gpu N
    planner --run-root R
    probe --run-root R --gpu N                    # G0 paired goal/time forward check
    report --run-root R
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

NEW_SOURCES = ("src/cp_disr/blocksworld/a04p_controls.py", "src/cp_disr/blocksworld/goal_probe.py", "src/cp_disr/blocksworld/a04p_registry.py", "scripts/c1_bw_a04p.py",
               "tests/test_c1_bw_a04p.py", "configs/splits/c1_bw_a04p_pilot80_v1.json", "docs/c1_blocksworld/CP_DISR_C1_Audit_Based_Goal_Progress_Runbook_v1.md")


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


def build_pilot(root):
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import generator as Gen
    from cp_disr.blocksworld import splits as SP
    from cp_disr.blocksworld import state as S
    from cp_disr.blocksworld.a04p_registry import PILOT_NAMESPACE, PILOT_SLICES
    b = SP.Builder()
    forbidden = set()
    td = json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())
    for c in td["train"] + td["dev"]:
        forbidden.add(c["problem_iso_hash"])
    for f in ("a0_iso", "a1_color_reverse", "a2_noniso", "b_scale"):
        for c in json.loads((root / ("configs/splits/c1_bw_%s_v1.json" % f)).read_text())["cases"]:
            forbidden.add(c["problem_iso_hash"])
    n_forbidden = len(forbidden)
    cases, report = [], {}
    for name, n, count, shapes, top in PILOT_SLICES:
        made, attempt = 0, 0
        top_color = S.RED if top == 0 else S.BLUE
        while made < count and attempt < 20000:
            heights = shapes[made % len(shapes)]
            p = Gen.make_problem("%s:%s" % (PILOT_NAMESPACE, name), n, made, heights, top_color, attempt, p_stack=0.88 if n >= 6 else 0.55)
            attempt += 1
            if p is None:
                continue
            h = K.problem_iso_hash(p)
            if h in forbidden:
                continue
            q = b.qualify(p, need_destruction=True)
            if q is None:
                continue
            if n >= 6 and not 8 <= q["optimal_length"] <= 22:
                continue
            tower_h = [len(c) for c in __import__("cp_disr.blocksworld.goal_probe", fromlist=["x"]).goal_components(p.goal)]
            extra = {"slice": name, "goal_heights": sorted(heights, reverse=True), "k_nontrivial_towers": sum(1 for t in tower_h if t >= 2), "max_tower_height": max(tower_h), "n_goal_atoms": n,
                     "n_candidates": len(C.template_for(p).contracts), "namespace": PILOT_NAMESPACE}
            forbidden.add(h)
            cases.append(SP.case_record("BW_a04p_%s_%03d" % (name.replace("P-", "").lower(), made), "a04p_pilot", p, q, extra))
            made += 1
        report[name] = {"planned": count, "built": made, "attempts": attempt}
        print(json.dumps({name: report[name]}), flush=True)
    doc = {"namespace": PILOT_NAMESPACE, "excluded_problem_classes_at_start": n_forbidden, "slices": report, "cases": cases, "counts": len(cases),
           "note": "development evidence; may not serve as a confirmation set for any new model"}
    wj(root / "configs/splits/c1_bw_a04p_pilot80_v1.json", doc)
    print(json.dumps({"pilot_cases": len(cases), "sha256": hashlib.sha256((root / "configs/splits/c1_bw_a04p_pilot80_v1.json").read_bytes()).hexdigest()}))


def register(root, auth_text):
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_probe as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import train as T
    from cp_disr.blocksworld import a04p_registry as R
    if E.git(root, "branch", "--show-current") != R.BRANCH or E.git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("registration needs branch %s with a clean tracked tree" % R.BRANCH)
    ck = root / R.B2_CKPT["path"]
    if E.sha256_file(ck) != R.B2_CKPT["sha256"] or ck.stat().st_size != R.B2_CKPT["bytes"]:
        raise SystemExit("B2 checkpoint identity mismatch")
    cases, half_life = I.load_train_cases(root / "configs/splits/c1_bw_train_dev_v1.json")
    d0 = json.loads((root / E.A02_ROOT / "datasets" / "D0.json").read_text())
    tau, n_tau = GP.tau_from_d0(d0, {c.case_id: c for c in cases})
    pilot = json.loads((root / R.PILOT_REL).read_text())
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_root = root / R.GP_REL / "a04p" / ("%s_%s" % (stamp, E.sha256_file(root / R.RUNBOOK_REL)[:8]))
    for sub in ("eval", "receipts", "results"):
        (run_root / sub).mkdir(parents=True)
    (run_root / "receipts" / "case_ledger.jsonl").write_text("")
    reg = {"card": "C1-BW-A04P", "runbook": R.RUNBOOK_REL, "runbook_sha256": E.sha256_file(root / R.RUNBOOK_REL), "base_commit": R.BASE_COMMIT, "branch": R.BRANCH, "authorization_text": auth_text,
           "authorization_text_sha256": hashlib.sha256(auth_text.encode()).hexdigest(), "phase1_new_training_runs": 0, "controllers": ["C0", "C1", "C3", "G1"],
           "controller_definitions": "runbook section 5 (C1 = A04 EDGE-RETRY count, C3 = visited-successor exclusion, G1 = strict superset of satisfied goal atoms)",
           "pilot": {"path": R.PILOT_REL, "sha256": E.sha256_file(root / R.PILOT_REL), "counts": pilot["counts"], "slices": pilot["slices"]},
           "pilot_episodes_planned": 4 * pilot["counts"], "planner_references_planned": pilot["counts"],
           "probe": {"source": "B2 first non-optimal decisions of all A03 failures in A2 and B", "max_candidates": 70, "min_valid_for_numeric_direction": GP.MIN_VALID, "flip_fraction_min": GP.FLIP_FRACTION,
                     "margin_improved_fraction_min": GP.IMPROVE_FRACTION, "median_margin_change_min": GP.MEDIAN_DELTA, "replay_tolerance": GP.REPLAY_TOL, "tau_reference": tau, "tau_n_steps": n_tau,
                     "tau_source": "median step_index/step_cap over A02 D0 expert steps that stack a held block onto its goal support"},
           "decision_states": ["GOAL_PROGRESS_READY", "TIME_INPUT_FIRST", "PROBE_INCONCLUSIVE", "INSUFFICIENT_PROBE"], "low_headroom_rule": "C3 or G1 >= 90% success on every core pilot slice (P-A2, P-B6) -> LOW_HEADROOM_FOR_SUCCESS_GAIN",
           "test_exposure": {"pilot80": "development evidence", "probe_states": "A03 states already consumed; development diagnostic only, never training data"}, "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    wj(run_root / "registration.json", reg)
    wj(run_root / "source_identity.json", R.source_identity(root, NEW_SOURCES))
    wj(run_root / "checkpoint_identity.json", {"run_id": R.B2_CKPT["run_id"], "path": R.B2_CKPT["path"], "sha256": R.B2_CKPT["sha256"], "bytes": R.B2_CKPT["bytes"], "committed_to_git": False})
    wj(run_root / "split_manifest.json", {"pilot_cases": {s: [c["case_id"] for c in pilot["cases"] if c["slice"] == s] for s in pilot["slices"]}})
    print(json.dumps({"run_root": str(run_root), "tau": tau, "n_tau": n_tau}))


def pilot_cases(root):
    from cp_disr.blocksworld import train as T
    doc = json.loads((root / "configs/splits/c1_bw_a04p_pilot80_v1.json").read_text())
    by = defaultdict(list)
    for c in doc["cases"]:
        by[c["slice"]].append(T.case_from_json(c))
    return doc, by


def cmd_pilot(root, run_root, controller, gpu):
    import torch
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import planner as P
    from cp_disr.rl import set_suite_half_life
    R.check_identity(root, run_root)
    set_suite_half_life(float(json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())["half_life"]))
    policy = E.load_model(root, "B2", torch.device("cuda", 0))
    solver = P.Solver()
    _doc, by = pilot_cases(root)
    groups = [(s, by[s]) for s in sorted(by)]
    before = E.sha256_file(root / R.B2_CKPT["path"])
    counts = R.run_cases(run_root, controller, groups, lambda c: AC.run_episode(policy, c, solver, controller), run_root / "eval" / controller / "episodes.jsonl")
    assert E.sha256_file(root / R.B2_CKPT["path"]) == before == R.B2_CKPT["sha256"]
    print(json.dumps({"controller": controller, **counts}))


def cmd_planner(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    R.check_identity(root, run_root)
    _doc, by = pilot_cases(root)

    def ref(c):
        s = P.Solver()
        t0 = time.perf_counter()
        plan = s.one_optimal_plan(c.problem.init, c.problem.goal)
        cpu = time.perf_counter() - t0
        st = c.problem.init
        for a in plan:
            st = S.apply(st, a)
        return {"case_id": c.case_id, "n_blocks": c.problem.n, "optimal_length": c.optimal_length, "plan_length": len(plan), "success": S.goal_satisfied(c.problem, st), "expanded_nodes": s.stats.expanded,
                "generated_nodes": s.stats.generated, "wall_seconds_cold_cache": cpu}
    print(json.dumps(R.run_cases(run_root, "planner", [(s, by[s]) for s in sorted(by)], ref, run_root / "eval" / "planner" / "episodes.jsonl")))


def cmd_denominators(root, run_root):
    from cp_disr.blocksworld import a04p_controls as AC
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import train as T
    a03 = root / R.A03_ROOT
    rows = []
    for method in ("B2", "QMARK", "ASNET"):
        logs = defaultdict(list)
        for d in jl(a03 / "eval" / method / "decision_log.jsonl"):
            logs[(d["split"], d["case_id"])].append(d)
        for split in ("a0", "a1", "a2", "b"):
            cases = {c["case_id"]: T.case_from_json(c) for c in json.loads((root / ("configs/splits/c1_bw_%s_v1.json" % {"a0": "a0_iso", "a1": "a1_color_reverse", "a2": "a2_noniso", "b": "b_scale"}[split])).read_text())["cases"]}
            for e in json.loads((a03 / "eval" / method / ("%s_final.json" % split)).read_text())["episodes"]:
                decs = sorted(logs[(split, e["case_id"])], key=lambda d: d["decision_index"])
                f = AC.first_error_fields(decs, cases[e["case_id"]].problem)
                grp = "successful_detour" if (e["success"] and f["has_first_error"]) else ("failed" if not e["success"] else "clean_success")
                rows.append({"method": method, "split": split, "case_id": e["case_id"], "success": e["success"], "group": grp, **f})
    wcsv(run_root / "results" / "first_error_denominator_table.csv", rows)
    summ = {}
    for method in ("B2", "QMARK", "ASNET"):
        for split in ("a0", "a1", "a2", "b"):
            for gname, pred in (("all_episodes", lambda r: True), ("failed_episodes", lambda r: r["group"] == "failed"), ("successful_detours", lambda r: r["group"] == "successful_detour")):
                rs = [r for r in rows if r["method"] == method and r["split"] == split and pred(r)]
                err = [r for r in rs if r["has_first_error"]]
                summ["%s|%s|%s" % (method, split, gname)] = {"episodes": len(rs), "with_first_error": len(err), "put_down_instead_of_goal_stack": sum(1 for r in err if r["first_error_is_put_down_instead_of_goal_stack"]),
                                                             "best_optimal_rank_counts": dict(Counter(r["best_optimal_rank"] for r in err)), "first_repeat_present": sum(1 for r in rs if r["first_repeated_action_is_optimal"] is not None),
                                                             "first_repeat_action_optimal": sum(1 for r in rs if r["first_repeated_action_is_optimal"])}
    wj(run_root / "results" / "first_error_denominator_summary.json", {"note": "denominators are explicit per group; 'failed' = not successful; 'successful_detours' = successful with a first error", "groups": summ})
    print(json.dumps({"rows": len(rows)}))


def cmd_probe(root, run_root, gpu):
    import torch
    from cp_disr.blocksworld import a04p_registry as R
    from cp_disr.blocksworld import eval_a03 as E
    from cp_disr.blocksworld import goal_probe as GP
    from cp_disr.blocksworld import imitation as I
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import train as T
    from cp_disr.rl import set_suite_half_life
    R.check_identity(root, run_root)
    reg = json.loads((run_root / "registration.json").read_text())
    tau = reg["probe"]["tau_reference"]
    set_suite_half_life(float(json.loads((root / "configs/splits/c1_bw_train_dev_v1.json").read_text())["half_life"]))
    policy = E.load_model(root, "B2", torch.device("cuda", 0))
    a03 = root / R.A03_ROOT
    logs = defaultdict(list)
    for d in jl(a03 / "eval" / "B2" / "decision_log.jsonl"):
        logs[(d["split"], d["case_id"])].append(d)
    solver = P.Solver()
    records, rows = [], []
    for split in ("a2", "b"):
        fname = {"a2": "a2_noniso", "b": "b_scale"}[split]
        cases = {c["case_id"]: T.case_from_json(c) for c in json.loads((root / ("configs/splits/c1_bw_%s_v1.json" % fname)).read_text())["cases"]}
        for e in json.loads((a03 / "eval" / "B2" / ("%s_final.json" % split)).read_text())["episodes"]:
            if e["success"]:
                continue
            decs = sorted(logs[(split, e["case_id"])], key=lambda d: d["decision_index"])
            rec, rs = GP.probe_sample(policy, cases[e["case_id"]], decs, tau, solver)
            rec["split"] = split
            records.append(rec)
            rows.extend(dict(r, split=split) for r in rs)
    summary = GP.summarize_probe(records, rows)
    summary["tau_reference"] = tau
    summary["note"] = "G0 is a development diagnostic on consumed A03 states; it is not a new episode result and nothing here may enter training."
    wj(run_root / "results" / "probe_manifest.json", {"samples": records})
    wcsv(run_root / "results" / "probe_pairs.csv", rows)
    wj(run_root / "results" / "probe_summary.json", summary)
    print(json.dumps({"state": summary["state"], "n_candidates": summary["n_candidates"], "n_valid": summary["n_valid"], "status_counts": summary["status_counts"]}))


def cmd_report(root, run_root):
    from cp_disr.blocksworld import a04p_registry as R
    res = run_root / "results"
    reg = json.loads((run_root / "registration.json").read_text())
    eps = {c: jl(run_root / "eval" / c / "episodes.jsonl") for c in ("C0", "C1", "C3", "G1")}
    pl = jl(run_root / "eval" / "planner" / "episodes.jsonl")
    pilot = json.loads((root / R.PILOT_REL).read_text())
    meta = {c["case_id"]: c for c in pilot["cases"]}
    percase, comp = [], []
    by0 = {e["case_id"]: e for e in eps["C0"]}
    for c, es in eps.items():
        for e in es:
            m = meta[e["case_id"]]
            percase.append({"controller": c, "slice": e["group"], "case_id": e["case_id"], "n_blocks": e["n_blocks"], "k_towers": m["k_nontrivial_towers"], "max_height": m["max_tower_height"], "optimal_length": e["optimal_length"],
                            "success": e["success"], "reason": e["reason"], "decision_perfect": e["decision_perfect"], "steps": e["steps"], "excess_steps": e["excess_steps"], "cycle": e["cycle"],
                            "first_divergence": e["first_divergence"], "interventions": e["interventions"], "first_intervention": e["first_intervention"],
                            "destroyed_satisfied_goal_count": e["destroyed_satisfied_goal_count"], "requires_goal_destruction": m["labels"].get("requires_goal_destruction"),
                            "rescued_success_vs_C0": bool(e["success"] and not by0[e["case_id"]]["success"]), "harmed_success_vs_C0": bool(by0[e["case_id"]]["success"] and not e["success"]),
                            "rescued_perfect_vs_C0": bool(e["decision_perfect"] and not by0[e["case_id"]]["decision_perfect"]), "harmed_perfect_vs_C0": bool(by0[e["case_id"]]["decision_perfect"] and not e["decision_perfect"])})
    wcsv(res / "per_case.csv", percase)
    slices = sorted({r["slice"] for r in percase}) + ["ALL"]
    for c in eps:
        for s in slices:
            rs = [r for r in percase if r["controller"] == c and (s == "ALL" or r["slice"] == s)]
            comp.append({"controller": c, "slice": s, "n": len(rs), "success": sum(r["success"] for r in rs), "decision_perfect": sum(r["decision_perfect"] for r in rs), "cycle_episodes": sum(r["cycle"] for r in rs),
                         "mean_steps": round(statistics.mean(r["steps"] for r in rs), 3), "rescued_success": sum(r["rescued_success_vs_C0"] for r in rs), "harmed_success": sum(r["harmed_success_vs_C0"] for r in rs),
                         "net_success_vs_C0": sum(r["rescued_success_vs_C0"] for r in rs) - sum(r["harmed_success_vs_C0"] for r in rs), "rescued_perfect": sum(r["rescued_perfect_vs_C0"] for r in rs),
                         "harmed_perfect": sum(r["harmed_perfect_vs_C0"] for r in rs), "episodes_with_intervention": sum(1 for r in rs if r["interventions"]),
                         "no_unvisited_successor_failures": sum(1 for r in rs if r["reason"] == "NO_UNVISITED_SUCCESSOR")})
    wcsv(res / "pilot_rule_comparisons.csv", comp)
    probe = json.loads((res / "probe_summary.json").read_text())
    core = [("P-A2-4", "P-A2-5a", "P-A2-5b"), ("P-B6",)]
    low = {}
    for c in ("C3", "G1"):
        ok = all(sum(r["success"] for r in percase if r["controller"] == c and r["slice"] in grp) >= 0.9 * sum(1 for r in percase if r["controller"] == c and r["slice"] in grp) for grp in core)
        low[c] = ok
    rec = {"state_from_probe": probe["state"], "low_headroom_for_success_gain": any(low.values()), "low_headroom_by_controller": low,
           "phase2_authorized_by_runbook": probe["state"] == "GOAL_PROGRESS_READY", "next": "PROCEED_TO_PHASE2" if probe["state"] == "GOAL_PROGRESS_READY" else "WAIT_FOR_USER_DECISION"}
    wj(res / "phase2_recommendation.json", rec)
    ledger = jl(run_root / "receipts" / "case_ledger.jsonl")
    started, completed = Counter(), Counter()
    for e in ledger:
        k = (e["actor"], e["split"], e["case_id"])
        if e["event"] == "STARTED":
            started[k] += 1
        if e["event"] == "COMPLETED":
            completed[k] += 1
    tech = len({(e["actor"], e["split"], e["case_id"]) for e in ledger if e["event"] == "TECHNICAL_INCOMPLETE"})
    verify = {"pilot_episodes_completed": sum(v for k, v in completed.items() if k[0] in eps), "planner_completed": sum(v for k, v in completed.items() if k[0] == "planner"), "technical_incomplete": tech,
              "ledger_once_each": all(v == 1 for v in started.values()) and all(v == 1 for v in completed.values()), "checkpoint_unchanged": E_sha(root, R) == R.B2_CKPT["sha256"],
              "planner_all_success": all(r["success"] for r in pl), "head": R.E.git(root, "rev-parse", "HEAD"), "old_roots_unchanged": all(R.E.git(root, "rev-parse", "HEAD:%s" % rel) == json.loads((run_root / "source_identity.json").read_text())["old_root_tree_ids_at_base"][rel] for rel in R.OLD_ROOTS)}
    verify["verdict"] = "PASS" if verify["pilot_episodes_completed"] == reg["pilot_episodes_planned"] and verify["planner_completed"] == reg["planner_references_planned"] and tech == 0 and verify["ledger_once_each"] and verify["checkpoint_unchanged"] and verify["old_roots_unchanged"] else "FAIL"
    wj(res / "verify.json", verify)
    print(json.dumps({"verify": verify["verdict"], "probe_state": probe["state"], "low_headroom": low}))


def E_sha(root, R):
    return R.E.sha256_file(root / R.B2_CKPT["path"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-pilot", "register", "denominators", "pilot", "planner", "probe", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".")
        if name not in ("build-pilot", "register"):
            p.add_argument("--run-root", required=True)
        if name == "register":
            p.add_argument("--authorization-text-file", required=True)
        if name == "pilot":
            p.add_argument("--controller", required=True, choices=("C0", "C1", "C3", "G1"))
        if name in ("pilot", "probe"):
            p.add_argument("--gpu", type=int, required=True)
    a = ap.parse_args()
    if a.cmd in ("pilot", "probe"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    root = Path(a.root).resolve()
    if a.cmd == "build-pilot":
        build_pilot(root)
    elif a.cmd == "register":
        register(root, Path(a.authorization_text_file).read_text(encoding="utf-8"))
    else:
        rr = Path(a.run_root).resolve()
        if a.cmd == "denominators":
            cmd_denominators(root, rr)
        elif a.cmd == "pilot":
            cmd_pilot(root, rr, a.controller, a.gpu)
        elif a.cmd == "planner":
            cmd_planner(root, rr)
        elif a.cmd == "probe":
            cmd_probe(root, rr, a.gpu)
        else:
            cmd_report(root, rr)


if __name__ == "__main__":
    main()
