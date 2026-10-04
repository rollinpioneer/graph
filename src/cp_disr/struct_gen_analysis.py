"""CP-DISR-TB-STRUCT-GEN-V1 analysis (files only; frozen outcome rules from structural_generalization_spec.json)."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import time
from pathlib import Path

from . import tb_seedbal_analysis as A

CARD = "CP-DISR-TB-STRUCT-GEN-V1"
FAMILIES = ("+E", "B2", "ABS", "NC")
FAMILY_OF = {"B1-K+E": "+E", "B2": "B2", "B2-ABS": "ABS", "B1-K+NC": "NC"}
SEEDS = (0, 1, 2)
TEST_CELLS = ("IN_S", "BUF_T+BUF_S", "IN_S+BUF_T")
CELL_DEPTH = {"IN_S": 3, "BUF_T+BUF_S": 4, "IN_S+BUF_T": 5}
DEPTH_LEVEL = {3: "DEPTH-1", 4: "DEPTH-2", 5: "DEPTH-3"}
NOVELTY_GROUPS = {"second_object->container involved": ("IN_S", "IN_S+BUF_T"), "target->buffer involved": ("BUF_T+BUF_S", "IN_S+BUF_T")}
DEV_N = 12


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def decide(test_rate, cell_rate, per_seed_rate):
    """Frozen outcome rules, evaluated in order. test_rate[m] = mean over seeds; cell_rate[m][cell]; per_seed_rate[m][seed] (m in FAMILIES)."""
    T = test_rate
    s_mean = mean([T["B2"], T["ABS"]])
    n_mean = mean([T["+E"], T["NC"]])
    delta = s_mean - n_mean
    delta_seed = {s: mean([per_seed_rate["B2"][s], per_seed_rate["ABS"][s]]) - mean([per_seed_rate["+E"][s], per_seed_rate["NC"][s]]) for s in SEEDS}
    cells_ahead = sum(1 for c in TEST_CELLS if mean([cell_rate["B2"][c], cell_rate["ABS"][c]]) >= mean([cell_rate["+E"][c], cell_rate["NC"][c]]) + 0.10)
    facts = {"T": T, "delta": delta, "delta_seed": delta_seed, "cells_where_successor_family_leads_by_0.10": cells_ahead,
             "max_T": max(T.values()), "min_T": min(T.values())}
    if T["NC"] >= 0.5 and T["NC"] >= T["B2"] - 0.10:
        return {"outcome": "OUTCOME_D", "name": "EXPLICIT_SYMBOLIC_APPLICATION_NOT_NECESSARY", "effect_on_paper": "NARROWED", "facts": facts}
    if max(T.values()) < 0.25 or (max(T.values()) - min(T.values())) < 0.10:
        return {"outcome": "OUTCOME_B", "name": "FIXED_PROFILE_LEARNING_EFFECT_ONLY", "effect_on_paper": "WEAKENED", "facts": facts}
    if T["B2"] - T["ABS"] >= 0.25 and T["B2"] - n_mean >= 0.15:
        return {"outcome": "OUTCOME_C", "name": "DIFFERENCE_PARAMETERIZATION_REGAINS_IMPORTANCE", "effect_on_paper": "NARROWED", "facts": facts}
    if delta >= 0.15 and sum(1 for v in delta_seed.values() if v >= 0.10) >= 2 and cells_ahead >= 2:
        return {"outcome": "OUTCOME_A", "name": "STRUCTURAL_GROUNDING_HYPOTHESIS_STRENGTHENED", "effect_on_paper": "STRONGLY_STRENGTHENED" if delta >= 0.30 else "STRENGTHENED", "facts": facts}
    return {"outcome": "OUTCOME_E", "name": "REPRESENTATION_EFFECT_INCONCLUSIVE", "effect_on_paper": "INCONCLUSIVE", "facts": facts}


def struct_test(run_dir):
    f = Path(run_dir) / "eval_struct_test_final.json"
    if not f.is_file():
        return None
    ev = json.loads(f.read_text())
    st = ev["struct_test"]
    return {"success_n": ev["success_n"], "n": ev["n"], "rate": ev["success_n"] / ev["n"], "mean_discounted_return": ev["mean_discounted_return"], "per_cell_success": st["per_cell_success"],
            "per_cell_n": st["per_cell_n"], "checkpoint": st["checkpoint"]}


def build(root, out, ledger):
    root, out = Path(root).resolve(), Path(out).resolve()
    res = out / "results"
    res.mkdir(exist_ok=True)
    problems, rows, struct_rows, registry, accounting, traces = [], [], [], {}, {}, {}
    dev_matrix, test_by_run, dev_final_rate = {}, {}, {}
    for plan_id, plan in sorted(ledger["plans"].items(), key=lambda kv: kv[1]["queue_order"]):
        run_dir = Path(plan.get("run_dir") or plan["planned_run_dir"])
        fam = FAMILY_OF[plan["method"]]
        reg = {"plan_id": plan_id, "method": plan["method"], "seed": plan["seed"], "attempt_id": plan.get("attempt_id"), "status": plan["status"], "run_dir": str(run_dir),
               "physical_gpu": plan.get("physical_gpu"), "started": plan.get("started"), "finished": plan.get("finished")}
        registry[plan_id] = reg
        if not (run_dir / "job_summary.json").is_file():
            problems.append("%s has no job_summary.json (status %s)" % (plan_id, plan["status"]))
            continue
        job = json.loads((run_dir / "job_summary.json").read_text())
        cfg = json.loads((run_dir / "resolved_config.json").read_text())
        pts = A.run_points(run_dir)
        manifest = {"plan_id": plan_id, "attempt_id": plan.get("attempt_id"), "generations": {}, "checkpoint_files": {}}
        for g in sorted(glob.glob(str(run_dir / "persistence" / "generations" / "*"))):
            g = Path(g)
            h = g / "HASHES.json"
            manifest["generations"][g.name] = {"complete_marker": (g / "COMPLETE").is_file(), "hashes": json.loads(h.read_text())["files"] if h.is_file() else None}
        for ck in sorted((run_dir / "checkpoints").glob("*.pt")):
            manifest["checkpoint_files"][ck.name] = A.sha(ck)
        (run_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        reg.update({"config_hash": hashlib.sha256((run_dir / "resolved_config.json").read_bytes()).hexdigest(), "train_split_hash": cfg["hashes"]["train_split_derived"], "Tcap": job["Tcap"],
                    "Ncap": cfg["Ncap"], "actual_N": job["valid_transitions"], "actual_T": job["interaction_seconds"], "optimizer_steps": job["optimizer_steps"], "NaN_n": job["NaN_n"],
                    "hard_fail": job["hard_fail"], "complete_updates": job["complete_updates"], "fragment_updates": job["fragment_updates"], "stop_reason": job["stop_reason"],
                    "source_commit": cfg["hashes"]["git_commit"], "eval_checkpoint_sha256": {p: manifest["checkpoint_files"].get((pts[p] or {}).get("checkpoint")) for p in A.POINTS}})
        accounting[plan_id] = {k: reg[k] for k in ("actual_N", "actual_T", "Tcap", "Ncap", "stop_reason", "complete_updates", "fragment_updates", "optimizer_steps", "NaN_n", "hard_fail")}
        accounting[plan_id]["train_success_episodes"] = job["train_success_episodes"]
        traces[plan_id] = A.seed_trace(run_dir, plan_id, plan["seed"])
        if all(pts[p] for p in A.POINTS):
            dev_matrix[(fam, plan["seed"])] = A.classify_seed({p: pts[p]["success_n"] for p in A.POINTS}, pts["final"]["n_cases"])
            dev_final_rate[(fam, plan["seed"])] = pts["final"]["success_n"] / pts["final"]["n_cases"]
        for p in A.POINTS:
            r = pts[p]
            if r is None:
                problems.append("%s missing dev point %s" % (plan_id, p))
                continue
            rows.append({"run": plan_id, "family": fam, "method": plan["method"], "seed": plan["seed"], "point": p, "dev_success_n": r["success_n"], "dev_n": r["n_cases"],
                         "mean_discounted_return": r["mean_discounted_return"], "N": r["N"], "T": r["T"], "update": r["update"], "mean_steps": r["mean_steps"],
                         "failure_families": json.dumps(r["failure_families"], sort_keys=True), "checkpoint": r["checkpoint"]})
        st = struct_test(run_dir)
        test_by_run[(fam, plan["seed"])] = st
        if st is None:
            problems.append("%s has no structural test evaluation" % plan_id)
        else:
            for cell in TEST_CELLS:
                struct_rows.append({"run": plan_id, "family": fam, "seed": plan["seed"], "cell": cell, "depth": CELL_DEPTH[cell], "depth_level": DEPTH_LEVEL[CELL_DEPTH[cell]],
                                    "success_n": st["per_cell_success"].get(cell), "n": st["per_cell_n"].get(cell), "rate": st["per_cell_success"].get(cell, 0) / max(1, st["per_cell_n"].get(cell, 1))})
    def write_csv(path, data):
        if not data:
            Path(path).write_text("")
            return
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    write_csv(res / "learning_checkpoint_table.csv", rows)
    write_csv(res / "struct_gen_test_by_cell.csv", struct_rows)
    counts = {fam: {"EARLY": sum(1 for s in SEEDS if (fam, s) in dev_matrix and A.is_early(dev_matrix[(fam, s)])), "FINAL_ONLY": sum(1 for s in SEEDS if dev_matrix.get((fam, s)) == "FINAL_ONLY"),
                    "NEVER": sum(1 for s in SEEDS if dev_matrix.get((fam, s)) == "NEVER"), "labels": [dev_matrix.get((fam, s)) for s in SEEDS]} for fam in FAMILIES}
    complete = all(test_by_run.get((f, s)) for f in FAMILIES for s in SEEDS)
    classification = {"card": CARD, "dev_first_effective_checkpoint": {"%s|s%d" % k: v for k, v in sorted(dev_matrix.items())}, "dev_family_counts": counts}
    if complete:
        per_seed = {f: {s: test_by_run[(f, s)]["rate"] for s in SEEDS} for f in FAMILIES}
        T = {f: mean(per_seed[f].values()) for f in FAMILIES}
        cell_rate = {f: {c: mean(test_by_run[(f, s)]["per_cell_success"][c] / test_by_run[(f, s)]["per_cell_n"][c] for s in SEEDS) for c in TEST_CELLS} for f in FAMILIES}
        depth_rate = {f: {DEPTH_LEVEL[CELL_DEPTH[c]]: cell_rate[f][c] for c in TEST_CELLS} for f in FAMILIES}
        novelty_rate = {f: {g: mean(cell_rate[f][c] for c in cells) for g, cells in NOVELTY_GROUPS.items()} for f in FAMILIES}
        gap = {f: {s: dev_final_rate.get((f, s)) - per_seed[f][s] for s in SEEDS} for f in FAMILIES if all((f, s) in dev_final_rate for s in SEEDS)}
        classification.update({"test_final_success_rate_by_seed": per_seed, "T_mean_over_seeds": T, "test_rate_by_cell": cell_rate, "test_rate_by_depth_level": depth_rate,
                               "test_rate_by_binding_novelty_group": novelty_rate, "generalization_gap_dev_minus_test": gap, **decide(T, cell_rate, per_seed)})
    else:
        classification.update({"outcome": "INCOMPLETE", "name": "structural test evaluations missing"})
        problems.append("structural test evaluation incomplete")
    (res / "classification.json").write_text(json.dumps(classification, indent=2, sort_keys=True, default=str) + "\n")
    (res / "run_registry.json").write_text(json.dumps({"card": CARD, "runs": registry}, indent=2, sort_keys=True) + "\n")
    (res / "training_accounting.json").write_text(json.dumps({"card": CARD, "runs": accounting}, indent=2, sort_keys=True) + "\n")
    (res / "seed_trace_check.json").write_text(json.dumps(traces, indent=2, sort_keys=True) + "\n")
    checks = {"twelve_attempts_used": ledger["new_rl_attempts_used"] == 12, "all_complete": all(r["status"] == "COMPLETE" for r in registry.values()),
              "all_tcap_or_ncap": all(r.get("stop_reason") in ("Tcap", "Ncap") for r in registry.values()), "no_nan_no_hard_fail": all(r.get("NaN_n") == 0 and r.get("hard_fail") is None for r in registry.values() if "NaN_n" in r),
              "dev_points_complete": not any("missing dev point" in p for p in problems), "seed_trace_ok": bool(traces) and all(t["ok"] for t in traces.values()), "struct_test_complete": complete}
    verify = {"card": CARD, "checks": checks, "problems": problems, "verdict": "PASS" if all(checks.values()) and not problems else "FAIL", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (res / "verify.json").write_text(json.dumps(verify, indent=2, sort_keys=True) + "\n")
    return classification, verify
