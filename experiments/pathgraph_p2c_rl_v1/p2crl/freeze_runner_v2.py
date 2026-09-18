"""Freeze runner V2. Does not activate attempt-02 training_release."""
from __future__ import annotations
import csv
import json
import subprocess
from pathlib import Path
from .constants_b import (
    ATTEMPT_01_RESULT_COMMIT, KNOWN_DEVIATIONS, PREREGISTRATION_COMMIT, RUNNER_V1_COMMIT,
)
from .io_utils import load_json, sha256_file, write_new, write_text_new
from .training_shape import FORMAL_SHAPE, SMOKE_SHAPE

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).rstrip("\n")

def freeze_runner_v2(*, repo, out_dir, test_report=None, changed_files=None):
    repo = Path(repo)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = git(repo, "rev-parse", "HEAD")
    rows = []
    if changed_files is None:
        text = git(repo, "diff", "--name-only", ATTEMPT_01_RESULT_COMMIT, "HEAD")
        changed_files = [ln.replace("\\", "/") for ln in text.splitlines() if ln.strip()]
    inv = out / "runner_v2_diff_inventory.csv"
    with inv.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256"], lineterminator="\n")
        w.writeheader()
        for path in changed_files:
            target = repo / path
            rec = {"path": path, "sha256": sha256_file(target) if target.is_file() else None}
            rows.append(rec)
            w.writerow(rec)
    invariance = {
        "schema": "P2CRL_RUNNER_V2_SCIENTIFIC_INVARIANCE_V1",
        "protocol_unchanged": True,
        "job_plan_unchanged": True,
        "dataset_unchanged": True,
        "methods_unchanged": True,
        "reward_unchanged": True,
        "mask_unchanged": True,
        "formal_ppo_unchanged": True,
        "statistics_unchanged": True,
        "smoke_shape": SMOKE_SHAPE.as_dict(),
        "formal_shape": FORMAL_SHAPE.as_dict(),
    }
    write_new(out / "runner_v2_scientific_invariance.json", invariance)
    tests_passed = None if test_report is None else bool(load_json(test_report).get("passed"))
    amendment = {
        "schema": "P2CRL_RUNNER_V2_AMENDMENT_V1",
        "category": "NON_SCIENTIFIC_SMOKE_ROLLOUT_QUANTUM_AND_ACCOUNTING_FIX",
        "base_result_commit": ATTEMPT_01_RESULT_COMMIT,
        "runner_v1_commit": RUNNER_V1_COMMIT,
        "runner_v2_commit": head,
        "scientific_protocol_changed": False,
        "job_plan_changed": False,
        "dataset_changed": False,
        "methods_changed": False,
        "reward_changed": False,
        "mask_changed": False,
        "formal_ppo_changed": False,
        "statistics_changed": False,
        "changes": [
            "SMOKE_N_STEPS_256_TO_64_WITH_N_ENVS_8",
            "CHECKPOINT_REQUESTED_EQUALS_ACTUAL",
            "ACTUAL_OPTIMIZER_STEP_COUNTER",
            "ATTEMPT_RELEASE_TOMBSTONE_ENFORCEMENT",
            "CSV_LF_EXPLICIT",
        ],
        "tests_passed": tests_passed,
        "learn_called": False,
        "optimizer_steps": 0,
        "smoke_jobs": 0,
        "formal_jobs": 0,
        "changed_files": rows,
        "known_deviations": list(KNOWN_DEVIATIONS),
        "preregistration_commit": PREREGISTRATION_COMMIT,
    }
    write_new(out / "runner_v2_amendment.json", amendment)
    if test_report is not None:
        write_new(out / "runner_v2_test_report.json", load_json(test_report))
    freeze = {
        "schema": "P2CRL_RUNNER_V2_FREEZE_V1",
        "status": "RUNNER_V2_FROZEN_WAITING_FOR_ATTEMPT_02_EXECUTION",
        "training_release": False,
        "smoke_jobs": 0,
        "formal_jobs": 0,
        "gradient_updates": 0,
        "optimizer_steps": 0,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "runner_v1_commit": RUNNER_V1_COMMIT,
        "attempt_01_result_commit": ATTEMPT_01_RESULT_COMMIT,
        "head_at_freeze": head,
        "known_deviations": list(KNOWN_DEVIATIONS),
    }
    write_new(out / "runner_v2_freeze.json", freeze)
    write_text_new(out / "README.md", "Runner V2 freeze. Do not execute attempt-02 without the explicit phrase.\n")
    return freeze