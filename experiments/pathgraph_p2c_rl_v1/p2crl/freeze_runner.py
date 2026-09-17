"""Freeze the B0 campaign runner. Does not activate training_release."""
from __future__ import annotations
import csv
import subprocess
from pathlib import Path
from .constants_b import (
    KNOWN_DEVIATIONS, PREREGISTRATION_COMMIT, SOURCE_LOCK_SHA256_A,
)
from .io_utils import load_json, sha256_file, write_new, write_text_new
from .source_lock import CANDIDATE_FILES

ALLOWED_MODIFY = {
    "experiments/pathgraph_p2c_rl_v1/p2crl/cli.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/learner.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/checkpointing.py",
}
ALLOWED_ADD_PREFIXES = (
    "experiments/pathgraph_p2c_rl_v1/p2crl/",
    "experiments/pathgraph_p2c_rl_v1/tests/",
    "artifacts/pathgraph_sarm/upgrade_v2/p2c_rl_policy_utility_v1/campaign_runner_v1/",
)
FORBIDDEN_PREFIXES = (
    "experiments/pathgraph_p2c_rl_v1/protocol.json",
    "experiments/pathgraph_p2c_rl_v1/p2crl/contracts.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/dataset_registry.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/gym_adapter.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/sampler.py",
    "experiments/pathgraph_p2c_rl_v1/p2crl/evaluate.py",
    "experiments/pathgraph_p2c_rm_v1/",
    "experiments/pathgraph_p2c_q_v1/",
)

def git(repo, *args):
    # rstrip newlines only. str.strip() would eat the leading space in
    # unstaged porcelain lines such as ' M path' and corrupt the first path.
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).rstrip("\n")

def _norm(p):
    return str(p).replace("\\", "/").lstrip("./")

def parse_porcelain(text):
    items = {}
    if not text:
        return items
    for line in text.splitlines():
        if not line:
            continue
        status = line[:2]
        path = _norm(line[3:])
        if " -> " in path:
            path = _norm(path.split(" -> ", 1)[1])
        if "__pycache__" in path or path.endswith((".pyc", ".pyo")) or ".pytest_cache" in path:
            continue
        items[path] = status.strip() or "M"
    return items

def changed_paths(repo):
    repo = Path(repo)
    items = parse_porcelain(git(repo, "status", "--porcelain"))
    committed = git(repo, "diff", "--name-only", PREREGISTRATION_COMMIT, "HEAD")
    if committed:
        for path in committed.splitlines():
            items.setdefault(_norm(path), "C")
    return items

def assert_allowlist(paths):
    bad = []
    for path in paths:
        n = _norm(path)
        if any(n == f or n.startswith(f) for f in FORBIDDEN_PREFIXES):
            bad.append(n)
            continue
        if n in ALLOWED_MODIFY:
            continue
        if any(n.startswith(pref) for pref in ALLOWED_ADD_PREFIXES):
            continue
        bad.append(n)
    if bad:
        raise RuntimeError("allowlist violation: " + ", ".join(sorted(bad)))

def freeze_runner(*, repo, out_dir, data_root=None, test_report=None, source_lock_a=None):
    repo = Path(repo)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = changed_paths(repo)
    assert_allowlist(paths)
    rows = []
    for path, status in sorted(paths.items()):
        target = repo / path
        rec = {
            "path": path,
            "status": status,
            "sha256": sha256_file(target) if target.is_file() else None,
        }
        rows.append(rec)
    inv = out / "runner_diff_inventory.csv"
    with inv.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["path", "status", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    head = git(repo, "rev-parse", "HEAD")
    amendment = {
        "schema": "P2CRL_PREREGISTRATION_IMPLEMENTATION_AMENDMENT_V1",
        "category": "NON_SCIENTIFIC_EXECUTION_RUNNER_COMPLETION",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "runner_commit": None,
        "head_at_freeze": head,
        "scientific_protocol_changed": False,
        "dataset_changed": False,
        "methods_changed": False,
        "reward_changed": False,
        "mask_changed": False,
        "statistics_changed": False,
        "only_execution_runner_completed": True,
        "changed_files": rows,
        "diff_sha256": sha256_file(inv),
        "tests_passed": None if test_report is None else bool(load_json(test_report).get("passed")),
        "gradient_updates": 0,
        "smoke_jobs": 0,
        "formal_jobs": 0,
        "known_deviations": list(KNOWN_DEVIATIONS),
    }
    write_new(out / "preregistration_amendment.json", amendment)
    runner_files = []
    for rel in sorted(p.relative_to(repo).as_posix() for p in (repo / "experiments/pathgraph_p2c_rl_v1").rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        if rel.endswith((".py", ".json", ".md")):
            runner_files.append({"path": rel, "sha256": sha256_file(repo / rel)})
    a_lock = None
    if source_lock_a and Path(source_lock_a).exists():
        a_lock = load_json(source_lock_a)
        if sha256_file(source_lock_a) != SOURCE_LOCK_SHA256_A:
            raise RuntimeError("A-stage source lock hash mismatch")
    write_new(out / "execution_source_lock.json", {
        "schema": "P2CRL_EXECUTION_SOURCE_LOCK_V1",
        "a_stage_source_lock_sha256": SOURCE_LOCK_SHA256_A,
        "a_stage_source_lock": a_lock,
        "candidate_files_locked": list(CANDIDATE_FILES),
        "runner_files": runner_files,
        "known_deviations": list(KNOWN_DEVIATIONS),
        "note": "A-stage source_lock.json is not overwritten.",
    })
    if test_report is not None:
        tr = load_json(test_report)
        write_new(out / "runner_test_report.json", tr)
    freeze = {
        "schema": "P2CRL_RUNNER_FREEZE_V1",
        "status": "RUNNER_FROZEN_WAITING_FOR_EXPLICIT_CAMPAIGN_EXECUTION",
        "training_release": False,
        "smoke_jobs": 0,
        "formal_jobs": 0,
        "gradient_updates": 0,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "head_at_freeze": head,
        "known_deviations": list(KNOWN_DEVIATIONS),
        "package_preflight_passed": False,
    }
    write_new(out / "runner_freeze.json", freeze)
    write_text_new(out / "README.md", "B0 runner freeze. Do not execute campaign without the explicit phrase.\n")
    return freeze
