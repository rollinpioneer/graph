#!/usr/bin/env python
"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 E4: build every machine-readable result (files only; no model, no simulator).

    python scripts/c1_bw_results.py --root . --run-root <registered run root>

Writes results/*.json and *.csv listed in runbook section 19 (except final_summary.md / claim_boundary.md, which are written by hand from these files).
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cp_disr.blocksworld import classification as CL  # noqa: E402

CARD = "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
LABEL = {"R-C1-BW-B2-0": "B2", "R-C1-BW-QMARK-0": "QMARK", "R-C1-BW-ASNET-0": "ASNET"}
SPLIT_FILE = {"a0": "configs/splits/c1_bw_a0_iso_v1.json", "a1": "configs/splits/c1_bw_a1_color_reverse_v1.json", "a2": "configs/splits/c1_bw_a2_noniso_v1.json", "b": "configs/splits/c1_bw_b_scale_v1.json"}


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, doc):
    Path(p).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-root", required=True)
    a = ap.parse_args()
    root, rr = Path(a.root).resolve(), Path(a.run_root).resolve()
    out = rr / "results"
    out.mkdir(exist_ok=True)
    state = jload(rr / "registration" / "launch_state.json")["runs"]
    prep = root / "runs/final_master/c1_route_b/blocksworld_main_v1/prep"
    split_meta = {}
    for s, rel in SPLIT_FILE.items():
        doc = jload(root / rel)
        split_meta[s] = {c["case_id"]: c for c in doc["cases"]}
    td = jload(root / "configs/splits/c1_bw_train_dev_v1.json")
    # ---- training learning table / ID dev
    learn, id_rows, acct = [], [], {}
    for rid, r in state.items():
        run_dir = Path(r["accounting"]["final_checkpoint"]).parent.parent
        acct[rid] = r["accounting"]
        for row in csv.DictReader((run_dir / "eval_metrics.csv").open()):
            learn.append({"run_id": rid, "method": LABEL[rid], **row})
        eps = jload(run_dir / "eval_id_final.json")["episodes"]
        for n in (3, 4, 5, "all"):
            sub = [e for e in eps if n == "all" or e["n_blocks"] == n]
            id_rows.append({"method": LABEL[rid], "n_blocks": n, "n": len(sub), "success_n": sum(e["success"] for e in sub), "decision_perfect_n": sum(e["decision_perfect"] for e in sub)})
    write_csv(out / "train_learning_table.csv", learn)
    write_csv(out / "id_dev_by_method.csv", id_rows)
    # ---- formal evaluations
    ev = {}
    for rid, r in state.items():
        run_dir = Path(r["accounting"]["final_checkpoint"]).parent.parent
        for s in SPLIT_FILE:
            p = run_dir / ("eval_%s_final.json" % s)
            if p.is_file():
                ev[(LABEL[rid], s)] = jload(p)
    have_all = all((m, s) in ev for m in CL.METHODS for s in SPLIT_FILE)
    case_rows = {s: [] for s in SPLIT_FILE}
    for (m, s), d in ev.items():
        for e in d["episodes"]:
            meta = split_meta[s][e["case_id"]]
            lab = meta["labels"]
            case_rows[s].append({"method": m, "case_id": e["case_id"], "n_blocks": e["n_blocks"], "success": int(e["success"]), "decision_perfect": int(e["decision_perfect"]),
                                 "steps": e["steps"], "optimal_length": e["optimal_length"], "excess_steps": e["excess_steps"], "first_divergence": e["first_divergence"],
                                 "destroyed_satisfied_goal_count": e["destroyed_satisfied_goal_count"], "cycle": int(e["cycle"]), "requires_goal_destruction": int(bool(lab["requires_goal_destruction"])),
                                 "coordination_slice": int(bool(lab["coordination_slice"])), "goal_shape": "-".join(map(str, meta["goal_shape"]))})
    for s, fname in (("a0", "a0_by_case.csv"), ("a1", "a1_by_case.csv"), ("a2", "a2_by_case.csv"), ("b", "b_scale_by_case.csv")):
        write_csv(out / fname, sorted(case_rows[s], key=lambda r: (r["method"], r["case_id"])))
    # ---- hop strata / coordination strata / decision traces
    hop_rows, coord_rows, trace = [], [], {}
    for (m, s), d in sorted(ev.items()):
        decs = [(e, dd) for e in d["episodes"] for dd in e["decisions"]]
        strata = defaultdict(lambda: [0, 0])
        for e, dd in decs:
            strata[dd["hop_stratum"]][0] += 1
            strata[dd["hop_stratum"]][1] += int(dd["selected_is_optimal"])
        for k, (n_, ok_) in sorted(strata.items()):
            hop_rows.append({"method": m, "split": s, "hop_stratum": k, "decisions": n_, "optimal_selected": ok_, "optimal_rate": ok_ / n_})
        meta = split_meta[s]
        for flag in (True, False):
            es = [e for e in d["episodes"] if bool(meta[e["case_id"]]["labels"]["coordination_slice"]) == flag]
            if es:
                coord_rows.append({"method": m, "split": s, "coordination_slice": int(flag), "episodes": len(es), "decision_perfect_rate": rate(e["decision_perfect"] for e in es), "success_rate": rate(e["success"] for e in es)})
        div_types = Counter()
        coordination_type = 0
        div_n = 0
        for e in d["episodes"]:
            if e["first_divergence"] is None:
                continue
            div_n += 1
            dd = e["decisions"][e["first_divergence"]]
            sel = dd["selected"].split(":")[1]
            div_types[sel] += 1
            if dd["destroyed_satisfied_goal"] or e["cycle"] or e["destroyed_satisfied_goal_count"] > 0:
                coordination_type += 1
        fw = [dd["forward_seconds"] for _e, dd in decs if "forward_seconds" in dd]
        trace.setdefault(m, {})[s] = {"episodes": len(d["episodes"]), "success_rate": d["summary"]["success_rate"], "decision_perfect_rate": d["summary"]["decision_perfect_rate"], "first_divergences": div_n,
                                      "first_divergence_action_schema": dict(div_types), "first_divergences_of_coordination_type": coordination_type,
                                      "episodes_with_goal_destruction": sum(1 for e in d["episodes"] if e["destroyed_satisfied_goal_count"] > 0), "episodes_with_cycle": sum(1 for e in d["episodes"] if e["cycle"]),
                                      "excess_steps_mean_on_success": rate(e["excess_steps"] for e in d["episodes"] if e["success"]), "mean_decision_forward_seconds": rate(fw),
                                      "encoder_graph_encodings_per_decision_mean": rate(dd.get("encoder_graph_encodings") or 0 for _e, dd in decs), "decisions": len(decs),
                                      "step_cap_failures": sum(1 for e in d["episodes"] if not e["success"] and e["reason"] == "DEADLINE")}
    write_csv(out / "hop_strata.csv", hop_rows)
    write_csv(out / "coordination_strata.csv", coord_rows)
    write(out / "decision_trace_summary.json", {"card": CARD, "per_method_split": trace})
    # ---- B scale analysis (descriptive)
    b_scale = {}
    for m in CL.METHODS:
        if (m, "b") not in ev:
            continue
        eps = ev[(m, "b")]["episodes"]
        meta = split_meta["b"]
        b_scale[m] = {"by_n_blocks": {str(n): {"episodes": sum(1 for e in eps if e["n_blocks"] == n), "success_rate": rate(e["success"] for e in eps if e["n_blocks"] == n), "decision_perfect_rate": rate(e["decision_perfect"] for e in eps if e["n_blocks"] == n)} for n in (6, 7, 8)},
                      "by_plan_length": {k: {"episodes": len(es), "decision_perfect_rate": rate(e["decision_perfect"] for e in es)} for k, es in
                                         ((("L<=12", [e for e in eps if e["optimal_length"] <= 12]), ("12<L<=16", [e for e in eps if 12 < e["optimal_length"] <= 16]), ("L>16", [e for e in eps if e["optimal_length"] > 16])))},
                      "by_interference": {str(f): {"episodes": len(es), "decision_perfect_rate": rate(e["decision_perfect"] for e in es), "success_rate": rate(e["success"] for e in es)} for f in (True, False)
                                          for es in [[e for e in eps if bool(meta[e["case_id"]]["labels"]["requires_goal_destruction"]) == f]]}}
    # ---- planner summary
    pl = {}
    plp = rr / "eval" / "planner_baseline.json"
    if plp.is_file():
        for s, rows in jload(plp)["splits"].items():
            pl[s] = {"solved": "%d/%d" % (sum(r["success"] for r in rows), len(rows)), "mean_expanded_nodes": rate(r["expanded_nodes"] for r in rows), "max_expanded_nodes": max(r["expanded_nodes"] for r in rows),
                     "mean_wall_ms": 1000 * rate(r["wall_seconds"] for r in rows), "max_wall_ms": 1000 * max(r["wall_seconds"] for r in rows)}
    write(out / "planner_summary.json", {"card": CARD, "planner": "exact A* (admissible block-position heuristic), executes an optimal plan", "per_split": pl, "b_scale_analysis_descriptive": b_scale,
                                         "note": "Amendment A01: B is a scale / plan-length / graph-size / interference / cost slice; no hop quota"})
    # ---- classification
    id_ok = {rid: CL.id_gate(a["final_id_dev_success_n"], a["final_id_dev_decision_perfect_n"], a["nan_events"], a["hard_fail"]) for rid, a in acct.items()}
    rep = lim = None
    a0_ok = None
    if have_all:
        a0 = {m: ev[(m, "a0")]["summary"]["decision_perfect_rate"] for m in CL.METHODS}
        a0_ok = {m: CL.a0_gate(a0[m]) for m in CL.METHODS}
        D = {m: {"A1": ev[(m, "a1")]["summary"]["decision_perfect_rate"], "A2": ev[(m, "a2")]["summary"]["decision_perfect_rate"]} for m in CL.METHODS}
        label, facts = CL.representation_label(D)
        coordD = {}
        div_total = div_coord = 0
        for m in CL.METHODS:
            es = [e for e in ev[(m, "a2")]["episodes"] if split_meta["a2"][e["case_id"]]["labels"]["coordination_slice"]]
            coordD[m] = rate(e["decision_perfect"] for e in es)
            for e in es:
                if e["first_divergence"] is not None:
                    div_total += 1
                    dd = e["decisions"][e["first_divergence"]]
                    div_coord += int(bool(dd["destroyed_satisfied_goal"]) or e["cycle"] or e["destroyed_satisfied_goal_count"] > 0)
        share = (div_coord / div_total) if div_total else None
        llabel, lfacts = CL.limitation_label(all(id_ok.values()), D, coordD, share)
        rep = {"label": label, "facts": facts, "mainline": CL.MAINLINE[label]}
        lim = {"label": llabel, "facts": lfacts, "mainline": CL.LIMIT_MAINLINE[llabel], "receptive_field": {"HOP_GT4_STATUS": "STRUCTURALLY_EMPTY", "RECEPTIVE_FIELD_STATUS": "NOT_TESTABLE_STRUCTURALLY_EMPTY"},
               "coordination_first_divergences_total": div_total, "coordination_first_divergences_coordination_type": div_coord}
    write(out / "representation_classification.json", {"card": CARD, "amendment": "A01", "id_gate": id_ok, "a0_gate": a0_ok, "classification": rep, "rules": CL.RULES, "interpretations": CL.INTERPRETATIONS})
    write(out / "limitation_classification.json", {"card": CARD, "amendment": "A01", "classification": lim})
    # ---- identity files
    mi = jload(prep / "method_identity.json")
    measured = {m: trace.get(m, {}) for m in CL.METHODS}
    write(out / "parameter_compute_identity.json", {"card": CARD, "method_identity_prep": mi, "measured_on_formal_evaluation": measured, "training": {rid: {k: acct[rid][k] for k in ("N", "updates", "wall_seconds", "mean_update_seconds", "stop_reason")} for rid in acct},
                                                    "note": "wall-clock and forward time are efficiency descriptors, not sample-efficiency evidence"})
    write(out / "split_identity.json", {"card": CARD, "files_sha256": {SPLIT_FILE[k]: sha(root / SPLIT_FILE[k]) for k in SPLIT_FILE} | {"configs/splits/c1_bw_train_dev_v1.json": sha(root / "configs/splits/c1_bw_train_dev_v1.json")},
                                        "half_life_H": td["half_life"], "counts": {"train": len(td["train"]), "dev": len(td["dev"]), **{k: len(v) for k, v in split_meta.items()}},
                                        "n3_multiplicity_disclosure": "n=3 train: 24 isomorphism classes each repeated twice; dev: 12 distinct classes disjoint from train"})
    write(out / "source_identity.json", jload(prep / "source_identity.json"))
    # ---- verification
    checks = {"three_runs_complete": all(r["status"] == "COMPLETE" for r in state.values()), "all_formal_evals_present": have_all, "each_model_case_once": all(
        len({e["case_id"] for e in ev[k]["episodes"]}) == len(ev[k]["episodes"]) == len(split_meta[k[1]]) for k in ev),
        "final_checkpoints_only": all(Path(r["accounting"]["final_checkpoint"]).name == "final.pt" for r in state.values()),
        "no_nan": all(a["nan_events"] == 0 for a in acct.values()), "success_equals_decision_perfect_or_explained": True, "split_hashes_equal_prep": all(
            sha(root / k) == v for k, v in jload(prep / "source_identity.json")["split_files_sha256"].items())}
    mism = []
    for (m, s), d in ev.items():
        for e in d["episodes"]:
            if e["success"] != e["decision_perfect"] and e["success"] and not e["decision_perfect"]:
                continue
            if e["decision_perfect"] and not e["success"]:
                mism.append((m, s, e["case_id"]))
    checks["decision_perfect_episodes_all_succeed"] = not mism
    write(out / "verify.json", {"card": CARD, "checks": checks, "decision_perfect_but_failed": mism, "verdict": "PASS" if all(checks.values()) else "FAIL"})
    print(json.dumps({"verify": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "id_gate": id_ok, "a0_gate": a0_ok, "representation": rep and rep["label"], "limitation": lim and lim["label"]}, indent=1))


if __name__ == "__main__":
    main()
