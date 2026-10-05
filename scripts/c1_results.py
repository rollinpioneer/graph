#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 Stage 8: build every machine-readable result from the decision-level jsonl files (files only; no environment).

    python scripts/c1_results.py --root . --fresh-dir runs/final_master/c1_route_b/mech_confirm_v1/fresh_confirm --reg <QMARK training registration dir> \
        --out runs/final_master/c1_route_b/mech_confirm_v1/results

Writes classification.json, fresh_confirm_by_case.csv, fresh_confirm_by_cell.csv, mechanism_trace_summary.json, planner_summary.json, parameter_compute_identity.json,
claim_boundary.md and verify.json. final_summary.md is written by hand from these files.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cp_disr import c1_analysis as A  # noqa: E402
from cp_disr import c1_classification as K  # noqa: E402

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")
CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
BASE_COMMIT = "5a2b3d18d21cbab19b9d09c3c430b3203e873730"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def qmark_training(reg, seed):
    """ID facts of one QMARK training run from the registration ledger and the run directory."""
    state = jload(Path(reg) / "launch_state.json")
    plan = state["plans"]["R-TB-C1-QMARK-%d" % seed]
    run_dir = Path(plan["run_dir"])
    job = jload(run_dir / "job_summary.json")
    rows = list(csv.DictReader((run_dir / "eval_metrics.csv").open(encoding="utf-8")))
    final = [r for r in rows if "final_" in r["checkpoint"]]
    final = final[-1] if final else rows[-1]
    man = jload(run_dir / "checkpoint_manifest.json")
    fin_name = [k for k in man["checkpoint_files"] if k.startswith("final_n_") and k.endswith(".pt")][0]
    return {"seed": seed, "status": plan["status"], "run_dir": str(run_dir), "final_dev12_success": int(final["success_n"]), "final_dev12_success_rate": float(final["success_rate"]),
            "dev12_points": {r["skill_transitions"]: [int(r["success_n"]), 12] for r in rows},
            "NaN_n": job.get("NaN_n"), "hard_fail": job.get("hard_fail"), "stop_reason": job.get("stop_reason"), "actual_N": job.get("valid_transitions"), "actual_T": job.get("interaction_seconds"),
            "complete_updates": job.get("complete_updates"), "final_checkpoint": str(run_dir / "checkpoints" / fin_name), "final_checkpoint_sha256_recorded": man["checkpoint_files"][fin_name]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fresh-dir", required=True)
    ap.add_argument("--reg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out = Path(a.out)
    out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=True)
    fresh = Path(a.fresh_dir)
    fresh = fresh if fresh.is_absolute() else root / fresh
    prep = root / PREP
    test_rows = jload(root / "configs/splits/c1_fresh_confirm_v1_test.json")["test"]
    case_ids = A.expected_case_ids(test_rows)
    runs = A.load_dir(fresh)
    problems = A.completeness(runs, case_ids)
    # ----------------------------------------------------------------------------- ID qualification and seeds
    qm_seeds = sorted(s for (l, s) in runs if l == "QMARK")
    qtrain = {s: qmark_training(a.reg, s) for s in qm_seeds}
    qualified = {s: K.id_qualified(qtrain[s]["final_dev12_success"], qtrain[s]["NaN_n"], qtrain[s]["hard_fail"]) for s in qm_seeds}
    # ----------------------------------------------------------------------------- per-seed rates and the pooled classification sequence
    per_seed = {}
    for s in qm_seeds:
        if all((l, s) in runs for l in ("B2", "ABS", "QMARK")):
            T, C = {}, {}
            for l in ("B2", "ABS", "QMARK"):
                T[l] = A.total_rate(runs[(l, s)])
                C[l] = A.cell_rates(runs[(l, s)])[0]
            per_seed[s] = {"T": T, "C": C}
    sequence = []
    for k in range(1, len(per_seed) + 1):
        seeds = sorted(per_seed)[:k]
        T, C = K.pool({s: per_seed[s] for s in seeds})
        sequence.append({"seeds": seeds, "T": T, "C": C, "main": K.classify_main(T, C), "b2_vs_abs": K.classify_b2_vs_abs(T, C)})
    final = sequence[-1] if sequence else None
    first_qualified = bool(qualified.get(0))
    if not first_qualified:
        main_class, b2_abs = "CONTROL_NOT_ID_QUALIFIED", None
    else:
        main_class = final["main"]["class"]
        b2_abs = final["b2_vs_abs"]["label"]
        if len(sequence) == 3 and main_class == "M4":
            main_class = "MECHANISM_INCONCLUSIVE"
    seed0_totals = {l: A.total_rate(runs[(l, 0)]) for l in A.NEURAL if (l, 0) in runs}
    planner_eps = runs.get(("B_PLAN", None))
    planner = A.planner_summary(planner_eps) if planner_eps else None
    planner_interp = A.planner_interpretation(planner, seed0_totals) if planner else None
    b2_total = final["T"]["B2"] if final else None
    abs_total = final["T"]["ABS"] if final else None
    rec = K.route_recommendation(final["main"]["class"] if final else "M4", b2_total if b2_total is not None else 0.0, abs_total if abs_total is not None else 0.0, first_qualified,
                                 main_class == "MECHANISM_INCONCLUSIVE")
    route = {"decision": rec["decision"], "recommendation": rec["recommendation"], "note": rec.get("note")}
    # ----------------------------------------------------------------------------- tables and traces
    A.write_csv(out / "fresh_confirm_by_case.csv", A.case_rows(runs))
    A.write_csv(out / "fresh_confirm_by_cell.csv", A.cell_rows(runs))
    mech = {"card": CARD, "per_run": {("%s_s%s" % (l, s) if s is not None else l): A.mechanism(eps) for (l, s), eps in sorted(runs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]))},
            "single_vs_dual_goal": {("%s_s%s" % (l, s) if s is not None else l): A.single_vs_dual(eps) for (l, s), eps in sorted(runs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]))},
            "abs_double_atbuffer_failures": [{"seed": s, "cell_BUF_T+BUF_S": [A.failure_category(e) for e in runs[("ABS", s)] if e["cell"] == "BUF_T+BUF_S"]} for (l, s) in sorted(runs, key=str) if l == "ABS"],
            "definitions": {"first_divergence": "first decision whose selected action is not in the shortest-legal-plan first-action set computed from the current PUBLIC facts after the policy has chosen (analysis only)",
                            "reference_unavailable": "REFERENCE_UNAVAILABLE_PUBLIC_UNKNOWN: no plan within depth 8 exists from the public facts; the hidden state is never used to complete it",
                            "completed_goals": "evaluator-level atoms (frozen atomic checks on the simulator state), logged after the fact; never a policy input",
                            "training_default_binding_error": "a placement of a train binding (target->container or second_object->buffer) selected at a decision where it is not in the optimal set",
                            "repeated_action": "the same candidate id selected again within an episode"}}
    write(out / "mechanism_trace_summary.json", mech)
    write(out / "planner_summary.json", {"card": CARD, "planner": planner, "interpretation": planner_interp, "neural_seed0_totals": seed0_totals})
    ident = jload(prep / "qmark_identity.json")
    eff = {("%s_s%s" % (l, s)): A.efficiency_neural(eps) for (l, s), eps in sorted(runs.items(), key=lambda kv: str(kv[0])) if l != "B_PLAN"}
    write(out / "parameter_compute_identity.json", {"card": CARD, "qmark_vs_b2_identity": ident, "old_methods_effective_parameters": ident.get("old_methods_effective_parameters"),
                                                    "measured_neural_decision_cost_on_fresh_confirm": eff, "planner_cost": None if planner is None else {k: planner[k] for k in ("cpu_seconds_mean", "cpu_seconds_p95", "cpu_seconds_max", "expanded_nodes_mean", "generated_nodes_mean", "planning_calls", "replanning_calls_after_first", "search_status_counts")},
                                                    "note": "wall-clock and forward time are efficiency descriptors, not sample-efficiency evidence; QMARK is not strictly equal-compute to B2 (disclosed in qmark_identity)"})
    # ----------------------------------------------------------------------------- verification
    frozen = jload(prep / "split_manifest.json")["file_sha256"]
    final_rel = "configs/splits/c1_fresh_confirm_v1_test.json"
    inv = jload(prep / "existing_evidence_inventory.json")["runs"]
    release = jload(prep / "training_release.json")
    checks = {"all_runs_have_the_32_frozen_cases_exactly_once": not problems, "final_file_hash_equals_frozen_hash": sha(root / final_rel) == frozen[final_rel],
              "classification_rules_hash_equals_release": sha(prep / "classification_rules.json") == release["classification_rules_sha256"], "qmark_seed0_present": ("QMARK", 0) in runs,
              "b_plan_present": planner_eps is not None, "planner_limits_frozen": planner is None or planner["limits"] == {"depth_limit": 8, "max_nodes": 4096, "cpu_time_limit_seconds": 2.0},
              "old_checkpoint_hashes_match_recorded": True, "qmark_checkpoint_hashes_match_recorded": True, "no_non_finite_probabilities": True, "formal_evals_flagged": True}
    metas = {}
    for (l, s) in runs:
        meta_path = fresh / ("%s_s%s.meta.json" % (l, s) if s is not None else "%s.meta.json" % l)
        if not meta_path.is_file():
            checks["formal_evals_flagged"] = False
            continue
        meta = jload(meta_path)
        metas[(l, s)] = meta
        checks["formal_evals_flagged"] &= bool(meta["formal"]) and meta["split_sha256"] == frozen[final_rel] and meta["optimizer_steps"] == 0
        if l in ("B2", "ABS", "E", "NC"):
            checks["old_checkpoint_hashes_match_recorded"] &= meta["checkpoint_sha256"] == inv["%s-%d" % (l, s)]["sha256_recorded"]
        if l == "QMARK":
            checks["qmark_checkpoint_hashes_match_recorded"] &= meta["checkpoint_sha256"] == qtrain[s]["final_checkpoint_sha256_recorded"]
    for key, eps in runs.items():
        for e in eps:
            for d in e["decisions"]:
                probs = d.get("probs")
                if probs is not None and not all(p == p and abs(p) != float("inf") for p in probs):
                    checks["no_non_finite_probabilities"] = False
    n_eval = sum(len(v) for v in runs.values())
    verify = {"card": CARD, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, "problems": problems, "episodes_evaluated": n_eval, "runs": sorted(["%s_s%s" % k for k in runs]),
              "verdict": "PASS" if all(checks.values()) else "FAIL"}
    write(out / "verify.json", verify)
    cls = {"card": CARD, "base_commit": BASE_COMMIT, "id_qualification": {str(s): {"qualified": qualified[s], **{k: v for k, v in qtrain[s].items() if k not in ("run_dir",)}} for s in qm_seeds},
           "per_seed": {str(s): v for s, v in per_seed.items()}, "pooled_sequence": sequence, "main": main_class, "b2_vs_abs": b2_abs, "route_b_recommendation": route["recommendation"], "decision": route["decision"],
           "decision_note": route["note"], "planner_interpretation": planner_interp, "additional_seeds_used": [s for s in qm_seeds if s != 0], "seed0_totals": seed0_totals,
           "frozen_rule_interpretations": K.INTERPRETATIONS, "next_action": "WAIT_FOR_USER_MAINLINE_DECISION", "state": "C1_MECHANISM_CONFIRMATION_COMPLETE",
           "conditional_release": {"seed1_needed": bool(per_seed) and sequence[0]["main"]["class"] == "M4" and first_qualified and len(sequence) == 1,
                                   "seed2_needed": len(sequence) == 2 and sequence[-1]["main"]["class"] == "M4"}}
    write(out / "classification.json", cls)
    print(json.dumps({"verify": verify["verdict"], "problems": problems, "main": main_class, "b2_vs_abs": b2_abs, "route": route, "seed0_totals": seed0_totals, "conditional": cls["conditional_release"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
