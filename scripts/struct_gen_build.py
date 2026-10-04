#!/usr/bin/env python
"""CP-DISR-TB-STRUCT-GEN-V1 prep builder (pure symbolic; no simulator, no GPU, no training).

    python scripts/struct_gen_build.py design --root .          # suite, audits, split files, spec, qualification plan
    python scripts/struct_gen_build.py freeze --root .          # after physical qualification: freeze split manifest + verify
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cp_disr import struct_gen as G  # noqa: E402

OUT_REL = Path("runs/final_master/2.1.1/structgen/prep")
TRAIN_DEV_REL = Path("configs/splits/struct_gen_v1_train_dev.json")
TEST_REL = Path("configs/splits/struct_gen_v1_test.json")
RESULT_COMMIT = "b602dfd1bc6f1241ebed1a474d51e6f6875d52a2"      # codex/cp-disr-tb-seed-balance result commit (resolved from the remote)
REPCTL_RESULT_COMMIT = "4d0c426f67dfa1ce23247f89879c0b4952367d53"
QUALIFICATION_CASES = {  # fixed before any environment episode: two cases per depth level, chosen by case id order only
    "DEPTH-1": ["SG_test_00", "SG_train_00"],      # IN_S (unseen, depth 3) and IN_T (train, depth 3)
    "DEPTH-2": ["SG_test_01", "SG_test_04"],       # BUF_T+BUF_S (unseen, depth 4), two different poses
    "DEPTH-3": ["SG_test_02", "SG_train_02"],      # IN_S+BUF_T (unseen, depth 5) and IN_T+BUF_S (train, the frozen T_B goal)
}
QUALIFICATION_CRITERIA = {
    "max_episodes": 6, "online_coordinate_tuning": "forbidden", "policy_training": "none",
    "per_case_pass_requires": [
        "reset succeeds and the public initial facts equal the symbolic initial facts (all ten atoms TRUE/FALSE, none UNKNOWN)",
        "every skill of the first optimal plan is legal under the hard mask when it is executed",
        "every skill ends with controller exit NORMAL_TERMINATION (no reach failure, collision, timeout, observation loss)",
        "after every skill the verified public facts equal the nominal successor facts (all ten atoms)",
        "the frozen evaluator reports CONTINUE before the last skill and TASK_SUCCESS after it",
        "total elapsed simulated time < 0.8 x the 60 s task deadline",
    ],
    "level_failure_rule": "any case failing any criterion at a depth level = systematic failure of that level -> STOP (no retry, no re-sampling, no coordinate change)",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=str(root), text=True).strip()


def blob_sha(root, ref, rel):
    try:
        return hashlib.sha256(subprocess.check_output(["git", "show", "%s:%s" % (ref, rel)], cwd=str(root), stderr=subprocess.DEVNULL)).hexdigest()
    except subprocess.CalledProcessError:
        return None


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def spec_document(root, manifest, baseline):
    frozen = baseline["frozen"]
    return {
        "card": G.CARD, "created": now(), "base_result_commits": {"representation_controls": REPCTL_RESULT_COMMIT, "seed_balance": RESULT_COMMIT},
        "question": "Does explicit computation of known state-conditioned action consequences provide a useful inductive bias under structural generalization?",
        "frozen_evidence_not_rerun": {"EARLY_seeds_of_3_on_the_frozen_T_B_profile": {"B2": 3, "ABS": 2, "+E": 1, "NC": 0},
                                      "matrix_first_effective_checkpoint": {"+E": ["FINAL_ONLY", "FINAL_ONLY", "4096"], "B2": ["4096", "4096", "4096"], "ABS": ["FINAL_ONLY", "4096", "8192"], "NC": ["NEVER", "FINAL_ONLY", "NEVER"]}},
        "what_changes": "only the goal composition (which known action consequences must be combined) and which object is bound to which destination; contracts, predicates, candidate IDs, hard mask, "
                        "controller, perception, verifier, reward, deadline, layout and the evaluator's atomic checks are the frozen T_B ones",
        "axes": {
            "A_dependency_depth": {"definition": "length of the shortest legal chain of Skill-Contract actions (preconditions enforced, nominal_apply) from the frozen public initial facts to the goal",
                                   "levels": {"DEPTH-1": "depth 2-3", "DEPTH-2": "depth 4", "DEPTH-3": "depth 5"}, "not_used_to_create_difficulty": ["distance", "grasp difficulty", "deadline", "camera", "controller timeout"]},
            "B_unseen_binding_composition": {"definition": "train/dev cells bind {target->container, second_object->buffer}; every test cell contains a binding absent from train (second_object->container or target->buffer) "
                                                            "composed from skill schemas and effect schemas that all appear in training; no new object type, predicate or skill",
                                             "forbidden_axes": ["UNKNOWN ratio", "conditional-effect complexity", "candidate clutter", "graph-topology perturbation", "perception change", "new skill", "new predicate type"]},
        },
        "goal_sets": {k: list(v) for k, v in G.GOAL_SETS.items()}, "expected_depth": G.EXPECTED_DEPTH,
        "cells": {"train_and_dev": list(G.TRAIN_CELLS), "struct_gen_test": list(G.TEST_CELLS)}, "cases_per_cell": G.PER_CELL,
        "counts": {"train": 3 * G.PER_CELL["train"], "dev": 3 * G.PER_CELL["dev"], "test": 3 * G.PER_CELL["test"]},
        "episode_order": "train cases are cycled in file order (task_cases[n % 60]), interleaved by cell; identical for every method and seed (production semantics)",
        "dev_size_note": "dev has 12 cases (4 per train cell) instead of 10; v11.DEV_EPISODES and v12.DEV10_N are set to 12 in-process; label strings keep the historical name 'dev10'",
        "test_policy": "the test split is a separate file never opened by the training split loader; it is evaluated post hoc, on the final checkpoint of each run only (pre-registered)",
        "methods": {"+E": "B1-K+E", "B2": "B2", "ABS": "B2-ABS", "NC": "B1-K+NC"}, "seeds": [0, 1, 2], "runs": 12,
        "representation_implementations": "frozen: neural.py, repctl_policy.py and the in-process registrations are unchanged (byte hashes in representation_identity.json)",
        "budget": {"Tcap_seconds": frozen["Tcap"], "Ncap": frozen["Ncap"], "H": frozen["H"], "d_ref": frozen["d_ref"], "task_deadline_seconds": frozen["task_deadline_seconds"], "rollout_n": frozen["rollout_n"],
                   "max_updates": frozen["max_updates"], "eval_points": frozen["eval_points"], "ppo": baseline["ppo"], "source": "read from baseline_identity.json (current-profile records), not from memory"},
        "metrics": {"learning_dynamics": "dev success, mean discounted return, actual N/T at 0/4096/8192/final",
                    "structural_generalization": "final-checkpoint success on the 30 test episodes, by dependency depth level and by binding novelty (second_object->container vs target->buffer involvement)",
                    "generalization_gap": "dev final success rate minus struct-gen test final success rate, per method and seed",
                    "earliest_effective_checkpoint": "4096 / 8192 / FINAL_ONLY / NEVER on dev (EARLY = dev 12/12 at 4096 or 8192; FINAL_ONLY = not early and final 12/12; NEVER otherwise)"},
        "outcome_rules_frozen_before_any_run": {
            "notation": "T(m) = mean over seeds of final test success rate of method m; c(m,cell) = mean over seeds of final success rate in a test cell; S = {B2, ABS}, N = {+E, NC}; "
                        "Delta = mean(T(B2),T(ABS)) - mean(T(+E),T(NC)); Delta_seed = the same per seed",
            "evaluated_in_order": [
                "OUTCOME_D EXPLICIT_SYMBOLIC_APPLICATION_NOT_NECESSARY: T(NC) >= 0.5 and T(NC) >= T(B2) - 0.10",
                "OUTCOME_B FIXED_PROFILE_LEARNING_EFFECT_ONLY: max_m T(m) < 0.25, or max_m T(m) - min_m T(m) < 0.10",
                "OUTCOME_C DIFFERENCE_PARAMETERIZATION_REGAINS_IMPORTANCE: T(B2) - T(ABS) >= 0.25 and T(B2) - mean(T(+E),T(NC)) >= 0.15",
                "OUTCOME_A STRUCTURAL_GROUNDING_HYPOTHESIS_STRENGTHENED: Delta >= 0.15, Delta_seed >= 0.10 in at least 2 of 3 seeds, and in at least 2 of 3 test cells mean(c(B2),c(ABS)) >= mean(c(+E),c(NC)) + 0.10",
                "OUTCOME_E REPRESENTATION_EFFECT_INCONCLUSIVE: none of the above (including seed-dominated ordering)"],
            "paper_effect_mapping": {"OUTCOME_A with Delta >= 0.30": "STRONGLY_STRENGTHENED", "OUTCOME_A with 0.15 <= Delta < 0.30": "STRENGTHENED", "OUTCOME_C": "NARROWED", "OUTCOME_D": "NARROWED",
                                     "OUTCOME_B": "WEAKENED", "OUTCOME_E": "INCONCLUSIVE"},
            "no_post_hoc": "no depth level, case or binding may be dropped, re-weighted or selected after any result is seen; raw evidence for every case is kept"},
        "qualification": {"cases": QUALIFICATION_CASES, "criteria": QUALIFICATION_CRITERIA},
        "stop_conditions": ["shortcut cannot be removed", "controller / reach problem", "perception problem", "new task needs a Skill Contract schema change or a new predicate", "needs a reward change",
                            "needs a materially different Tcap", "representation implementation would have to change"],
        "evaluator_disclosure": "task_evaluator.py is byte-identical to the base commit; StructGenEvaluator subclasses it and evaluates the case's goal atoms with its unchanged atomic checks "
                                "(_inside, _at_buffer, thresholds, deadline and first-success reward logic). Equivalence with TaskEvaluator for the T_B, T_A and single-goal modes is unit-tested.",
        "not_authorized_here": ["VLM / prior", "T_P", "Family B", "RoboCasa", "new controller / perception / reward / predicate family", "planner", "learned world model", "new seeds", "new representations"],
    }


def design(root):
    root = Path(root).resolve()
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    rows = G.generate_rows()
    manifest, contracts, cache = G.build_manifest(root, rows)
    baseline = json.loads((root / "runs/final_master/2.1.1/repctl/prep/baseline_identity.json").read_text())
    depth = G.dependency_depth_audit(manifest, contracts)
    novelty = G.binding_novelty_audit(manifest, contracts)
    shortcut = G.shortcut_audit(manifest, contracts, cache)
    hashes = {}
    hashes["representation_challenge_manifest.json"] = G.write_json(out / "representation_challenge_manifest.json", manifest)
    hashes["dependency_depth_audit.json"] = G.write_json(out / "dependency_depth_audit.json", depth)
    hashes["binding_novelty_audit.json"] = G.write_json(out / "binding_novelty_audit.json", novelty)
    hashes["shortcut_audit.json"] = G.write_json(out / "shortcut_audit.json", shortcut)
    train_dev = {"task_id": "T_B", "version": "struct-gen-v1", "second_role": "second_object", "train": [r for r in rows if r["split"] == "train"], "dev": [r for r in rows if r["split"] == "dev"],
                 "test": [], "test_count": 0, "train_count": 60, "dev_count": 12, "no_prior": True,
                 "note": "train + dev only; the structural-generalization test rows are in a separate file that no training code path opens"}
    test_doc = {"task_id": "T_B", "version": "struct-gen-v1", "second_role": "second_object", "train": [], "dev": [], "test": [r for r in rows if r["split"] == "test"], "test_count": 30,
                "note": "post-hoc evaluation only (final checkpoints)"}
    G.write_json(root / TRAIN_DEV_REL, train_dev)
    G.write_json(root / TEST_REL, test_doc)
    plan = {"card": G.CARD, "created": now(), **{"cases": QUALIFICATION_CASES}, "criteria": QUALIFICATION_CRITERIA,
            "case_detail": {cid: next(c for c in manifest["cases"] if c["case_id"] == cid)["cell"] for cases in QUALIFICATION_CASES.values() for cid in cases},
            "unique_cases": sorted({cid for cases in QUALIFICATION_CASES.values() for cid in cases}), "gpu": "one shared GPU, one environment, scripted optimal plan, no policy training"}
    hashes["physical_qualification_plan.json"] = G.write_json(out / "physical_qualification_plan.json", plan)
    spec = spec_document(root, manifest, baseline)
    hashes["structural_generalization_spec.json"] = G.write_json(out / "structural_generalization_spec.json", spec)
    summary = {"depth": depth["verdict"], "novelty": novelty["verdict"], "shortcut": shortcut["verdict"], "n_cases": len(manifest["cases"]),
               "split_files": {"train_dev": str(TRAIN_DEV_REL), "test": str(TEST_REL), "train_dev_sha256": sha(root / TRAIN_DEV_REL), "test_sha256": sha(root / TEST_REL)}}
    print(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["design"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    design(a.root)


if __name__ == "__main__":
    main()
