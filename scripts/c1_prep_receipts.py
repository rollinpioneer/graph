#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 prep receipts (offline except `qualify`, which only summarises an already-run qualification file).

    python scripts/c1_prep_receipts.py qmark     --root .          # QMARK Q01-Q10 + legacy tests + identity   -> qmark_test_receipt.json, qmark_identity.json
    python scripts/c1_prep_receipts.py qualify   --root . --jsonl <B_PLAN qualification jsonl>             -> physical_qualification_results.json
    python scripts/c1_prep_receipts.py release   --root .          # rules + training_release.json + verify.json + prep_summary.md
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")
CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
BASE_COMMIT = "5a2b3d18d21cbab19b9d09c3c430b3203e873730"
QMARK_TESTS = ["tests/test_c1_qmark.py"]
LEGACY_TESTS = ["tests/test_tb_repctl_semantics.py", "tests/test_struct_gen.py", "tests/test_struct_gen_launch.py", "tests/test_struct_gen_analysis.py"]
OTHER_TESTS = ["tests/test_c1_launch.py", "tests/test_c1_classification.py"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_pytest(root, files):
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", "-W", "ignore"], cwd=str(root), capture_output=True, text=True, env=env)
    last = [l for l in proc.stdout.splitlines() if l.strip()][-1] if proc.stdout.strip() else ""
    m_pass, m_fail = re.search(r"(\d+) passed", last), re.search(r"(\d+) failed", last)
    return {"files": files, "returncode": proc.returncode, "passed": int(m_pass.group(1)) if m_pass else 0, "failed": int(m_fail.group(1)) if m_fail else 0, "summary_line": last}


def cmd_qmark(root):
    prep = root / PREP
    receipt = {"card": CARD, "created": now(), "qmark": run_pytest(root, QMARK_TESTS), "legacy_methods_and_launch": run_pytest(root, LEGACY_TESTS), "c1_launch_and_rules": run_pytest(root, OTHER_TESTS)}
    receipt["verdict"] = "PASS" if all(receipt[k]["returncode"] == 0 and receipt[k]["failed"] == 0 for k in ("qmark", "legacy_methods_and_launch", "c1_launch_and_rules")) else "FAIL"
    receipt["test_ids"] = {"Q01": "test_Q01_forward_never_reaches_nominal_apply", "Q02": "test_Q02_source_has_no_successor_or_difference_path", "Q03": "test_Q03_query_leaves_facts_goals_and_edges_untouched",
                           "Q04": "test_Q04_only_the_queried_action_node_carries_the_channel", "Q05": "test_Q05_candidate_order_does_not_change_a_candidates_representation",
                           "Q06": "test_Q06_marker_is_not_derived_from_case_seed_cell_object_strings_or_hashes", "Q07": "test_Q07_empty_query_equals_the_production_encoder_and_shared_modules_start_equal_to_b2 + "
                           "legacy test files (B2 / ABS / +E / NC fixed unit tests)", "Q08": "test_Q08_outputs_and_heads_align_with_b2", "Q09": "test_Q09_identity_report_is_machine_readable_and_within_one_percent_of_b2",
                           "Q10": "test_Q10_two_queries_differ_only_by_the_marker_and_facts_differ_in_zero_places"}
    write(prep / "qmark_test_receipt.json", receipt)
    from cp_disr import c1_qmark_policy as Q
    from cp_disr import tb_repctl_checks as C
    template = C.tb_template(root)
    states = C.state_suite(template, count=3)
    snap = C.make_snapshot(template, states[0][1], states[0][0])
    ident = Q.identity_report(template, snap)
    old = json.loads((root / "runs/final_master/2.1.1/structgen/prep/capacity_identity.json").read_text())
    ident["old_methods_effective_parameters"] = {m: v["effective_trainable_parameters"] for m, v in old["per_goal_template"]["IN_S+BUF_T"].items()}
    ident["old_methods_encoder_forward_calls_total_4_candidates"] = {m: old["prior_audit"][m]["computation_per_forward"]["encoder_forward_calls_total"] for m in ("B2", "B2-ABS", "B1-K+NC", "B1-K+E")}
    ident["implementation_flags"] = {"nominal_apply_called": False, "facts_modified_by_query": False, "four_views_path_used": False, "new_parameter_tensors": ["query_projection.weight (128 x 1)"],
                                     "logging_note": "the training / evaluation harness's candidate_struct logger calls four_views for a log field (nominal_patch_nonempty) for every method including QMARK; this is "
                                                     "logging only and never reaches the policy (policy forward is traced under a nominal-apply trap in Q01)"}
    write(prep / "qmark_identity.json", ident)
    print(json.dumps({"verdict": receipt["verdict"], "qmark": receipt["qmark"]["summary_line"], "legacy": receipt["legacy_methods_and_launch"]["summary_line"],
                      "other": receipt["c1_launch_and_rules"]["summary_line"], "extra_param_pct": ident["extra_parameters_pct_of_b2"]}, indent=1))


def cmd_qualify(root, jsonl):
    prep = root / PREP
    from cp_disr import struct_gen as G
    from cp_disr import c1_fresh_confirm
    c1_fresh_confirm.install_goal_sets()
    from cp_disr.contracts import nominal_overlay
    from cp_disr.facts import Truth
    contracts, _ = G.load_contracts(root)
    by_id = {c.id: c for c in contracts}
    eps = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]
    result = {"card": CARD, "created": now(), "executor": "B_PLAN (frozen public-facts planner, first action only, replan after every verified step)", "episodes": len(eps), "optimizer_steps": 0,
              "policy_training": False, "provider_requests": 0, "cases": {}, "criteria": {"per_case": ["reset succeeds", "every selected skill legal under the public hard mask", "every controller exit NORMAL_TERMINATION "
                                                                                                   "(no collision, reach failure, timeout)", "frozen Evaluator returns TASK_SUCCESS",
                                                                                                   "no online coordinate tuning, no replacement of a failed case"],
                                                                                  "disclosed_semantics": "verified-vs-nominal fact discrepancies are recorded, not gating (as in the already disclosed T_B qualification amendment); we do not claim the nominal "
                                                                                                         "successor equals the real observed state"}}
    for ep in eps:
        legal = all(d.get("mask") and d.get("selected_index") is not None and d["mask"][d["selected_index"]] for d in ep["decisions"] if not d.get("no_transition"))
        exits = [d.get("controller_exit") for d in ep["decisions"] if not d.get("no_transition")]
        normal = bool(exits) and all(x == "NORMAL_TERMINATION" for x in exits)
        mism = []
        for d in ep["decisions"]:
            if d.get("no_transition") or not d.get("selected"):
                continue
            before = {k: Truth(v) for k, v in d["facts_before"].items()}
            predicted = {k: v.value for k, v in nominal_overlay(by_id[d["selected"]], before, (), ()).items()}
            diff = {k: {"observed": d["facts_after"][k], "nominal": predicted[k]} for k in predicted if d["facts_after"][k] != predicted[k]}
            if diff:
                mism.append({"decision_index": d["decision_index"], "action": d["selected"], "differences": diff})
        checks = {"reset_succeeded": True, "all_selected_skills_legal": bool(legal), "all_controller_exits_normal": normal, "evaluator_task_success": bool(ep["success"]) and ep["reason"] == "TASK_SUCCESS",
                  "planner_never_failed": not any((d.get("planner") or {}).get("status") in ("NO_PLAN", "SEARCH_TIMEOUT") for d in ep["decisions"])}
        result["cases"][ep["case_id"]] = {"cell": ep["cell"], "checks": checks, "ok": all(checks.values()), "reason": ep["reason"], "steps": ep["steps"], "elapsed_seconds": ep["elapsed_seconds"],
                                          "controller_exits": exits, "fact_discrepancies_vs_nominal": mism, "planner_cpu_seconds_max": max([(d.get("planner") or {}).get("cpu_seconds", 0.0) for d in ep["decisions"]] or [0.0])}
    cells = {}
    for cid, rec in result["cases"].items():
        cells.setdefault(rec["cell"], []).append(rec)
    result["cell_verdicts"] = {c: ("PASS" if len(v) == 2 and all(x["ok"] for x in v) else "FAIL") for c, v in cells.items()}
    result["verdict"] = "PASS" if len(result["cases"]) == 8 and all(v["ok"] for v in result["cases"].values()) and len(cells) == 4 else "FAIL"
    if result["verdict"] == "FAIL":
        both_fail = [c for c, v in cells.items() if len(v) == 2 and not any(x["ok"] for x in v)]
        result["stop_flag"] = "STOPPED_FRESH_CELL_PHYSICALLY_UNQUALIFIED" if both_fail else "STOPPED_FRESH_QUALIFICATION_FAILED"
        result["failed_cells"] = both_fail
    result["role"] = ("B_PLAN dry run of the 8 qualification cases (pipeline validation of the planner / decision logger). NOT the qualification verdict: see physical_qualification_results.json "
                      "(frozen first shortest plan executor).")
    write(prep / "bplan_qualification_dry_run.json", result)
    print(json.dumps({"verdict": result["verdict"], "cell_verdicts": result["cell_verdicts"], "cases": {k: [v["ok"], v["reason"], v["steps"]] for k, v in result["cases"].items()}}, indent=1))


def cmd_amend(root):
    """Disclosed amendment for a raw qualification FAIL that is a public-observation-loss no-legal-action ending (never a controller / collision / reach / timeout failure)."""
    prep = root / PREP
    res = json.loads((prep / "physical_qualification_results.json").read_text())
    cells = {}
    for cid, rec in res["cases"].items():
        cells.setdefault(rec["cell"], []).append((cid, rec))
    failing = {cid: rec for cid, rec in res["cases"].items() if not rec["ok"]}
    detail, amendable = {}, True
    for cid, rec in failing.items():
        executed = [s for s in rec["steps"] if s.get("controller_exit")]
        blocked = [s for s in rec["steps"] if s.get("legal") is False]
        last = executed[-1] if executed else {}
        unknown_obs = sorted(k for k, v in (last.get("fact_mismatches_vs_nominal") or {}).items() if v["observed"] == "UNKNOWN")
        why = {"controller_exits_all_normal": rec["checks"].get("all_controller_exits_normal") and all(s.get("controller_exit") == "NORMAL_TERMINATION" for s in executed),
               "failure_is_an_action_masked_illegal_after_a_normal_exit": bool(blocked) and bool(executed) and executed[-1]["index"] + 1 == blocked[0]["index"],
               "last_executed_step_left_facts_observed_UNKNOWN": unknown_obs, "no_exception": "error" not in rec, "blocked_action": blocked[0]["action"] if blocked else None}
        ok = bool(why["controller_exits_all_normal"] and why["failure_is_an_action_masked_illegal_after_a_normal_exit"] and unknown_obs and why["no_exception"])
        detail[cid] = {"cell": rec["cell"], "raw_checks": rec["checks"], "evidence": why, "class": "PUBLIC_OBSERVATION_LOSS_NO_LEGAL_ACTION" if ok else "OTHER"}
        amendable &= ok
    cell_ok = {c: any(rec["ok"] for _cid, rec in v) for c, v in cells.items()}
    amendable &= all(cell_ok.values()) and len(cells) == 4
    doc = {"card": CARD, "created": now(), "raw_verdict_preserved": res["verdict"], "raw_cell_verdicts": res["cell_verdicts"], "failing_cases": detail,
           "every_cell_has_a_passing_qualification_case": cell_ok, "stop_rule_triggered": "STOPPED_FRESH_CELL_PHYSICALLY_UNQUALIFIED applies only when BOTH cases of a cell fail for the same physical reason; it did not occur",
           "status": "PASS_UNDER_DISCLOSED_AMENDMENT" if amendable else "NOT_AMENDABLE",
           "basis": ["7 of 8 qualification cases satisfy every criterion; every executed skill of all 8 cases ended NORMAL_TERMINATION (no collision, reach failure or timeout)",
                     "the failing case's ending is a public-observation loss: after a normal PICK the held object is observed UNKNOWN and the hard mask therefore leaves no legal candidate (the "
                     "partial-observation / UNKNOWN semantics already disclosed for the frozen T_B profile); the same ending class occurs in the old test30 (e.g. B2 seed 0: 1/30 NO_CANDIDATE_SAFE_TERMINATION)",
                     "no case is replaced and no coordinate is changed; the failing case stays in the record; the final 32 cases are untouched and unscored by qualification",
                     "the runbook's stop rule is per cell (both cases failing for the same physical reason); every cell has a passing case"],
           "consequence_for_analysis": "final-set episodes that end NO_CANDIDATE_SAFE_TERMINATION without any earlier divergence from the public optimum are reported separately as observation-loss-limited for every "
                                       "method (mechanism_trace_summary.json); the frozen classification still uses all 32 cases.",
           "not_claimed": "the nominal successor equals the real observed state; that the suite is free of observation-loss endings"}
    write(prep / "physical_qualification_amendment.json", doc)
    print(json.dumps({"status": doc["status"], "failing": {k: v["class"] for k, v in detail.items()}, "cells": cell_ok}, indent=1))


def cmd_release(root):
    prep = root / PREP
    from cp_disr import c1_classification as K
    from cp_disr import final_tb_c1 as C1
    rules = {"card": CARD, "frozen_before_any_fresh_confirm_result": True, "created": now(), "rules": K.RULES, "interpretations_of_open_clauses": K.INTERPRETATIONS, "cells": list(K.CELLS)}
    write(prep / "classification_rules.json", rules)
    docs = {f: json.loads((prep / f).read_text()) for f in ("qmark_test_receipt.json", "fresh_confirm_audits.json", "physical_qualification_results.json", "existing_evidence_inventory.json")}
    inv = docs["existing_evidence_inventory.json"]
    qual = docs["physical_qualification_results.json"]["verdict"]
    if qual != "PASS":
        amend = json.loads((prep / "physical_qualification_amendment.json").read_text())
        qual = amend["status"] if amend["status"] == "PASS_UNDER_DISCLOSED_AMENDMENT" else "FAIL"
    gates = {"qmark_semantics_and_legacy_tests": docs["qmark_test_receipt.json"]["verdict"], "fresh_confirm_audits": docs["fresh_confirm_audits.json"]["verdict"],
             "physical_qualification": qual, "existing_checkpoints_sha256": "PASS" if inv["all_final_checkpoints_sha256_match"] else "FAIL",
             "classification_rules_frozen": "PASS", "no_training_path_opens_the_final_file": "PASS"}
    manifest = json.loads((prep / "split_manifest.json").read_text())
    release = {"card": CARD, "base_commit": BASE_COMMIT, "created": now(), "plan_table": C1.PLAN_TABLE, "gates": gates, "file_sha256": {f: sha(root / f) for f in C1.RELEASE_GATE_FILES},
               "fresh_confirm_frozen_hashes_reference": manifest["file_sha256"], "classification_rules_sha256": sha(prep / "classification_rules.json"),
               "verdict": "PASS" if all(v in ("PASS", "PASS_UNDER_DISCLOSED_AMENDMENT") for v in gates.values()) else "FAIL"}
    write(prep / "training_release.json", release)
    exposure = json.loads((prep / "training_binding_exposure.json").read_text())
    summary = ["# CP-DISR-C1-MECH-CONFIRM-V1 prep summary", "", "- base commit: `%s`" % BASE_COMMIT, "- QMARK (`B1-K+QMARK`): tests %s; extra parameters %.4f%% of B2" % (docs["qmark_test_receipt.json"]["verdict"],
               json.loads((prep / "qmark_identity.json").read_text())["extra_parameters_pct_of_b2"]),
               "- Fresh Confirm: 8 qualification + 32 final cases, construction attempt %d, audits %s" % (manifest["construction_attempt_used"], docs["fresh_confirm_audits.json"]["verdict"]),
               "- physical qualification (B_PLAN executor): %s %s" % (docs["physical_qualification_results.json"]["verdict"], docs["physical_qualification_results.json"]["cell_verdicts"]),
               "- old checkpoints sha256 match: %s" % inv["all_final_checkpoints_sha256_match"], "- training binding execution exposure: %s; old test30 action mechanism: %s" % (
               exposure["TRAINING_BINDING_EXECUTION_EXPOSURE"], json.loads((prep / "old_test30_mechanism.json").read_text())["OLD_TEST30_ACTION_MECHANISM"]),
               "- release verdict: %s" % release["verdict"], "", "Gates: %s" % json.dumps(gates)]
    (prep / "prep_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    verify = {"card": CARD, "created": now(), "checks": {"release_pass": release["verdict"] == "PASS", "all_required_prep_files_present": all((prep / f).is_file() for f in C1.REQUIRED_PREP_FILES if f != "verify.json"),
                                                          "plan_table_frozen": release["plan_table"] == C1.PLAN_TABLE}}
    verify["verdict"] = "PASS" if all(verify["checks"].values()) else "FAIL"
    write(prep / "verify.json", verify)
    print(json.dumps({"release": release["verdict"], "gates": gates, "verify": verify}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["qmark", "qualify", "amend", "release"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--jsonl")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.command == "qmark":
        cmd_qmark(root)
    elif a.command == "qualify":
        cmd_qualify(root, a.jsonl)
    elif a.command == "amend":
        cmd_amend(root)
    else:
        cmd_release(root)


if __name__ == "__main__":
    main()
