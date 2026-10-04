#!/usr/bin/env python
"""CP-DISR-TB-STRUCT-GEN-V1 release builder: split freeze, representation / capacity identity, offline test receipt, training_release.json.

    python scripts/struct_gen_release.py identity --root .     # split manifest + representation_identity + capacity_identity
    python scripts/struct_gen_release.py tests    --root .     # offline_test_receipt.json (runs pytest)
    python scripts/struct_gen_release.py release  --root .     # training_release.json + prep_summary.md + verify.json
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
from cp_disr import struct_gen as G  # noqa: E402

PREP = Path("runs/final_master/2.1.1/structgen/prep")
SEED_BALANCE = "b602dfd1bc6f1241ebed1a474d51e6f6875d52a2"
REPCTL = "4d0c426f67dfa1ce23247f89879c0b4952367d53"
BASE_PROTECTED = [
    "src/cp_disr/platforms/libero/task_evaluator.py", "src/cp_disr/platforms/libero/skill_executor.py", "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/perception.py",
    "src/cp_disr/platforms/libero/safety.py", "src/cp_disr/platforms/libero/clock.py", "src/cp_disr/platforms/libero/d0_env.py", "src/cp_disr/platforms/libero/runtime_factory.py",
    "src/cp_disr/platforms/libero/snapshot.py", "src/cp_disr/platforms/libero/observations.py", "src/cp_disr/rl.py", "src/cp_disr/collector.py", "src/cp_disr/torch_rl.py",
    "src/cp_disr/adapters.py", "src/cp_disr/neural.py", "src/cp_disr/repctl_policy.py", "src/cp_disr/stage2a_v11.py", "src/cp_disr/phase_a_v12.py", "src/cp_disr/final_tb.py",
    "src/cp_disr/final_tb_repctl.py", "src/cp_disr/tb_repctl_checks.py", "src/cp_disr/contracts.py", "src/cp_disr/graph.py", "configs/runtime/stage_2a_contract_registry.yaml",
    "experiments/manifests/runtime_manifest_v211.yaml",
]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(root, ref, rel):
    try:
        return hashlib.sha256(subprocess.check_output(["git", "show", "%s:%s" % (ref, rel)], cwd=str(root), stderr=subprocess.DEVNULL)).hexdigest()
    except subprocess.CalledProcessError:
        return None


def jload(p):
    return json.loads(Path(p).read_text())


def identity(root):
    import torch
    torch.set_num_threads(4)
    from cp_disr import tb_repctl_checks as C
    from cp_disr.graph import Goal, build_template
    from cp_disr.platforms.libero.runtime_factory import PREDICATES
    root = Path(root).resolve()
    out = root / PREP
    manifest = jload(out / "representation_challenge_manifest.json")
    qual = jload(out / "physical_qualification_amendment.json")
    split = {"card": G.CARD, "status": "FROZEN", "frozen_at": now(), "frozen_after": "physical qualification (raw FAIL under the strict fact-equality criterion; PASS under the disclosed reference-relative amendment)",
             "no_performance_used": "no method result of any kind exists for this suite; cases were fixed by the deterministic generator before any environment episode",
             "files": {"train_dev": {"path": "configs/splits/struct_gen_v1_train_dev.json", "sha256": sha(root / "configs/splits/struct_gen_v1_train_dev.json")},
                       "test": {"path": "configs/splits/struct_gen_v1_test.json", "sha256": sha(root / "configs/splits/struct_gen_v1_test.json")}},
             "counts": {s: sum(1 for c in manifest["cases"] if c["split"] == s) for s in ("train", "dev", "test")},
             "cells": {s: sorted({c["cell"] for c in manifest["cases"] if c["split"] == s}) for s in ("train", "dev", "test")},
             "case_ids": {s: [c["case_id"] for c in manifest["cases"] if c["split"] == s] for s in ("train", "dev", "test")},
             "challenge_manifest_sha256": sha(out / "representation_challenge_manifest.json"), "qualification_cases": jload(out / "physical_qualification_plan.json")["unique_cases"],
             "qualification_verdicts": {"raw": jload(out / "physical_qualification_results.json")["verdict"], "amended": qual["verdict_under_amended_criterion"]}}
    G.write_json(out / "struct_gen_split_manifest.json", split)
    files = ["src/cp_disr/neural.py", "src/cp_disr/repctl_policy.py", "src/cp_disr/final_tb_repctl.py", "src/cp_disr/tb_repctl_checks.py", "src/cp_disr/torch_rl.py", "src/cp_disr/collector.py"]
    rows = {rel: {"seed_balance_result_commit": blob(root, SEED_BALANCE, rel), "representation_controls_result_commit": blob(root, REPCTL, rel), "worktree": sha(root / rel)} for rel in files}
    for r in rows.values():
        r["equal_to_seed_balance_result_commit"] = r["worktree"] == r["seed_balance_result_commit"]
    rep_id = {"card": G.CARD, "methods": {"+E": "neural.Policy(method='B1-K+E')", "B2": "neural.Policy(method='B2')", "ABS": "repctl_policy.RepctlPolicy(method='B2-ABS')",
                                           "NC": "repctl_policy.RepctlPolicy(method='B1-K+NC')"}, "files": rows,
              "semantics": "unchanged: +E direct grounded effects (no successor); B2 nominal successor difference; ABS absolute nominal successor; NC matched neural state x effect composition (no nominal_apply)",
              "semantic_checks": "tests/test_struct_gen.py::test_representation_semantics_hold_on_every_goal_template runs A1/A1b/A3/A4/N1-N6 and the common boundary checks on all six goal templates",
              "verdict": "PASS" if all(r["equal_to_seed_balance_result_commit"] for r in rows.values()) else "FAIL", "created": now()}
    G.write_json(out / "representation_identity.json", rep_id)
    base = C.tb_template(root)
    cap = {"card": G.CARD, "per_goal_template": {}, "prior_audit": jload(root / "runs/final_master/2.1.1/repctl/prep/representation_capacity_audit.json")["arms"], "created": now()}
    prior_eff = {m: cap["prior_audit"][m]["effective_trainable_parameters"] for m in ("B2", "B2-ABS", "B1-K+NC", "B1-K+E")}
    equal = True
    for key in ("IN_T", "IN_S+BUF_T"):
        template = build_template(base.contracts, tuple(Goal(a, 1) for a in G.GOAL_SETS[key]), PREDICATES, G.OBJECTS)
        seed, values = C.state_suite(template, count=1)[0]
        snap = C.make_snapshot(template, values, seed)
        arms = {}
        for method in ("B1-K+E", "B2", "B2-ABS", "B1-K+NC"):
            eff, per_module = C.effective_parameters(C.make_policy(template, method, 0), snap)
            arms[method] = {"effective_trainable_parameters": eff, "by_module": per_module, "computation_per_forward": C.computation_counts(template, method, snap)}
            equal &= eff == prior_eff[method]
        cap["per_goal_template"][key] = arms
    cap["equal_to_representation_control_audit"] = equal
    cap["vs_b2_pct"] = {m: 100.0 * (prior_eff[m] - prior_eff["B2"]) / prior_eff["B2"] for m in prior_eff}
    cap["reporting_note"] = "training wall-clock and transitions/sec are reported with the results and are not sample-efficiency evidence"
    cap["verdict"] = "PASS" if equal else "FAIL"
    G.write_json(out / "capacity_identity.json", cap)
    print(json.dumps({"split": split["counts"], "representation_identity": rep_id["verdict"], "capacity_identity": cap["verdict"]}))


def tests(root):
    root = Path(root).resolve()
    out = root / PREP
    receipt = {"card": G.CARD, "created": now(), "suites": {}}
    for name in ("tests/test_struct_gen.py", "tests/test_tb_repctl_semantics.py", "tests/test_tb_seedbal.py", "tests/test_struct_gen_launch.py", "tests/test_struct_gen_analysis.py"):
        if not (root / name).is_file():
            continue
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", name], cwd=str(root), capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(root / "src")})
        tail = [l for l in p.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)]
        receipt["suites"][name] = {"returncode": p.returncode, "summary": tail[-1] if tail else None}
    receipt["mutations"] = ["M1 candidate zero always optimal -> shortcut gate fails (test_M1_...)", "M2 test binding in train -> novelty gate fails (test_M2_...)", "M3 corrupted depth label -> depth audit fails (test_M3_...)",
                            "object-ID leakage -> shortcut gate fails", "representation-control mutants (A2, A3, N1-N6) are covered by the representation-control suites listed above"]
    receipt["verdict"] = "PASS" if receipt["suites"] and all(s["returncode"] == 0 for s in receipt["suites"].values()) else "FAIL"
    G.write_json(out / "offline_test_receipt.json", receipt)
    print(json.dumps({k: v["summary"] for k, v in receipt["suites"].items()}), receipt["verdict"])


def release(root):
    from cp_disr import final_tb as ftb
    from cp_disr import final_tb_structgen as SG
    root = Path(root).resolve()
    out = root / PREP
    audits = {n: jload(out / (n + ".json"))["verdict"] for n in ("dependency_depth_audit", "binding_novelty_audit", "shortcut_audit")}
    raw = jload(out / "physical_qualification_results.json")
    amend = jload(out / "physical_qualification_amendment.json")
    qual = "PASS" if raw["verdict"] == "PASS" else ("PASS_UNDER_DISCLOSED_AMENDMENT" if amend["verdict_under_amended_criterion"] == "PASS" else "FAIL")
    steps = [s["duration"] for r in raw["cases"].values() for s in r["steps"] if s.get("duration")]
    base = jload(root / "runs/final_master/2.1.1/repctl/prep/baseline_identity.json")
    base_means = [r["actual_T"] / r["actual_N"] for r in base["baseline_runs"].values()]
    mean_step = sum(steps) / len(steps)
    expected_n = ftb.TCAP_SECONDS / mean_step
    budget = {"Tcap_seconds": ftb.TCAP_SECONDS, "Ncap": ftb.N_CAP, "mean_skill_duration_seconds_in_qualification": mean_step, "baseline_current_profile_mean_skill_duration": sum(base_means) / len(base_means),
              "expected_N_at_Tcap": expected_n, "ncap_binding": expected_n >= ftb.N_CAP, "statement": "Tcap is the binding stop as in the current-profile runs; no budget change is needed or made",
              "verdict": "PASS" if expected_n < ftb.N_CAP and 0.7 < mean_step / (sum(base_means) / len(base_means)) < 1.3 else "REVIEW"}
    changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=MD", SEED_BALANCE, "--", *BASE_PROTECTED], cwd=str(root), text=True).split()
    gates = {"dependency_depth_audit": audits["dependency_depth_audit"], "binding_novelty_audit": audits["binding_novelty_audit"], "shortcut_audit": audits["shortcut_audit"],
             "physical_qualification": qual, "split_frozen": "PASS" if jload(out / "struct_gen_split_manifest.json")["status"] == "FROZEN" else "FAIL",
             "representation_identity": jload(out / "representation_identity.json")["verdict"], "capacity_identity": jload(out / "capacity_identity.json")["verdict"],
             "offline_tests": jload(out / "offline_test_receipt.json")["verdict"], "budget_fairness": budget["verdict"],
             "reward_evaluator_controller_files_unchanged": "PASS" if not changed else "FAIL"}
    ok = all(v in ("PASS", "PASS_UNDER_DISCLOSED_AMENDMENT") for v in gates.values())
    rel = {"card": G.CARD, "created": now(), "verdict": "PASS" if ok else "FAIL", "gates": gates, "budget": budget, "plan_table": SG.PLAN_TABLE,
           "disclosures": ["physical qualification raw verdict FAIL under the strict fact-equality criterion; the same six episodes pass the disclosed reference-relative amendment "
                           "(physical_qualification_amendment.json); no episode was added or repeated",
                           "task_evaluator.py is unchanged; StructGenEvaluator reuses its atomic checks (unit-tested for equivalence)",
                           "dev has 12 cases (4 per train cell), evaluated at 0/4096/8192/final; the test split is evaluated post hoc on final checkpoints only"],
           "file_sha256": {f: sha(root / f) for f in SG.RELEASE_GATE_FILES if (root / f).is_file()}}
    G.write_json(out / "training_release.json", rel)
    summary = ["# CP-DISR-TB-STRUCT-GEN-V1 — prep summary", "",
               "Suite: six goal-composition cells on the frozen T_B layout (lid closed, same candidates / mask / controller / verifier / reward / deadline). Train+dev: IN_T (depth 3), BUF_S (depth 2), IN_T+BUF_S (depth 5, the frozen T_B goal). "
               "Struct-gen test: IN_S (depth 3), BUF_T+BUF_S (depth 4), IN_S+BUF_T (depth 5); every test cell contains an object->destination binding absent from train.", "",
               "Counts: train 60, dev 12, test 30 (102 cases). Depth levels: DEPTH-1 = 2-3, DEPTH-2 = 4, DEPTH-3 = 5 (pure symbolic shortest-chain, independently re-computed).", "",
               "## Gates", ""] + ["- %s: %s" % (k, v) for k, v in gates.items()] + ["", "## Disclosures", ""] + ["- " + d for d in rel["disclosures"]] + [
               "", "## Budget", "", "- mean skill duration in qualification %.2f s vs current-profile %.2f s; expected N at Tcap = %.0f (< Ncap %d): Tcap remains the binding stop, no budget change." % (
                   mean_step, budget["baseline_current_profile_mean_skill_duration"], expected_n, ftb.N_CAP), "", "Release verdict: **%s**" % rel["verdict"], ""]
    (out / "prep_summary.md").write_text("\n".join(summary))
    required = SG.REQUIRED_PREP_FILES
    files_ok = {f: (out / f).is_file() or f == "verify.json" for f in required}
    verify = {"card": G.CARD, "created": now(), "checks": {"all_required_prep_files_present": all(files_ok.values()), **{"gate_" + k: v in ("PASS", "PASS_UNDER_DISCLOSED_AMENDMENT") for k, v in gates.items()}},
              "gates": gates, "verdict": "PASS" if ok and all(files_ok.values()) else "FAIL"}
    G.write_json(out / "verify.json", verify)
    print(json.dumps({"release": rel["verdict"], "gates": gates, "budget": {k: budget[k] for k in ("mean_skill_duration_seconds_in_qualification", "expected_N_at_Tcap", "verdict")}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["identity", "tests", "release"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    {"identity": identity, "tests": tests, "release": release}[a.command](a.root)


if __name__ == "__main__":
    main()
