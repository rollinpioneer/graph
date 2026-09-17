"""Source and runtime locks. Distinguishes git blob SHA-1 from byte SHA-256."""
from __future__ import annotations
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from .io_utils import git_blob_sha1, sha256_file, write_new

CANDIDATE_FILES = [
    "experiments/pathgraph_p2c_rm_v1/p2crm/remaining_work_model.py",
    "experiments/pathgraph_p2c_rm_v1/p2crm/potentials_v2.py",
    "experiments/pathgraph_p2c_rm_v1/p2crm/mask_contract.py",
    "experiments/pathgraph_p2c_rm_v1/p2crm/gym_bridge.py",
    "experiments/pathgraph_p2c_rm_v1/p2crm/generator_v2.py",
    "experiments/pathgraph_p2c_rm_v1/p2crm/masked_smoke.py",
    "experiments/pathgraph_p2c_q_v1/p2cq_research/environment.py",
    "experiments/pathgraph_p2c_q_v1/p2cq_research/potentials.py",
    "experiments/pathgraph_p2c_q_v1/p2cq_research/observations.py",
    "experiments/pathgraph_p2c_q_v1/p2cq_research/task_contract.py",
    "experiments/pathgraph_p2c_q_v1/p2cq_research/graph_model.py",
]

# Result-base e31eaa5 unwrapped unused ActionMasker.invalid_selected counters
# from candidate 7a69b84. New P2C-RL runner does not import this file.
# Restoring candidate bytes would write a protected RM path and fail only_new_scoped_paths.
PUBLISHED_RESULT_UNWRAP = {
    "experiments/pathgraph_p2c_rm_v1/p2crm/masked_smoke.py": {
        "candidate_blob_sha1": "7b2b7ec50b6d8950ff67f1fea9527f71d68fc5b5",
        "result_base_blob_sha1": "10dbf5a95095f0a40ee5ee0d9bf8f9eefe95b568",
        "reason": "RM result commit deleted two unused invalid_selected/nonfinite lines from candidate smoke. Keep published result-base bytes; do not rewrite RM.",
    }
}

RUNNER_SUFFIXES = (".py", ".json", ".md")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _is_known_unwrap(rel, rec, seed, repo):
    spec = PUBLISHED_RESULT_UNWRAP.get(rel)
    if spec is None:
        return False
    base = seed.get("base_commit")
    if not base:
        return False
    try:
        base_blob = git(repo, "rev-parse", f"{base}:{rel}")
    except Exception:
        return False
    return (
        rec["git_blob_sha1"] == spec["candidate_blob_sha1"]
        and rec["worktree_git_blob_sha1"] == spec["result_base_blob_sha1"]
        and base_blob == spec["result_base_blob_sha1"]
        and rec["match"] is False
    )


def lock_sources(repo, candidate, seed, out_path, extra_files=None):
    repo = Path(repo)
    files = []
    for rel, expected in seed["git_blob_sha1"].items():
        actual = git(repo, "rev-parse", f"{candidate}:{rel}")
        blob = subprocess.check_output(["git", "-C", str(repo), "show", f"{candidate}:{rel}"])
        target = repo / rel
        raw = target.read_bytes() if target.is_file() else None
        rec = {
            "path": rel,
            "git_blob_sha1": actual,
            "expected_blob": expected,
            "worktree_git_blob_sha1": git_blob_sha1(raw) if raw is not None else None,
            "byte_sha256": sha256_file(target) if target.is_file() else None,
            "match": bool(raw is not None and raw == blob and actual == expected),
        }
        rec["published_result_unwrap"] = _is_known_unwrap(rel, rec, seed, repo)
        files.append(rec)
    extras = []
    for rel in extra_files or []:
        rel = str(rel).replace("\\", "/")
        if "/__pycache__/" in f"/{rel}/":
            continue
        if not rel.endswith(RUNNER_SUFFIXES):
            continue
        p = repo / rel
        if not p.is_file():
            continue
        extras.append({
            "path": rel,
            "byte_sha256": sha256_file(p),
            "git_blob_sha1": git_blob_sha1(p.read_bytes()),
        })
    n_match = sum(1 for x in files if x["match"])
    unwraps = [x["path"] for x in files if x.get("published_result_unwrap")]
    functional = all(x["match"] or x.get("published_result_unwrap") for x in files)
    rec = {
        "schema": "P2CRL_SOURCE_LOCK_V1",
        "candidate_commit": candidate,
        "base_commit": seed.get("base_commit"),
        "candidate_files": files,
        "runner_files": extras,
        "n_candidate_files": len(files),
        "n_candidate_bytes_match": n_match,
        "candidate_bytes_match": all(x["match"] for x in files),
        "candidate_bytes_match_except_published_result_unwrap": functional,
        "published_result_unwrap_files": unwraps,
        "note": "Keep result-base masked_smoke.py. Package preflight candidate_bytes_match remains false. New runner does not import that file.",
    }
    write_new(out_path, rec)
    if not rec["candidate_bytes_match_except_published_result_unwrap"]:
        raise RuntimeError("candidate bytes mismatch beyond published result unwrap")
    return rec


def lock_runtime(out_path):
    import numpy, torch
    rec = {
        "schema": "P2CRL_RUNTIME_LOCK_V1",
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "stable-baselines3": importlib.metadata.version("stable-baselines3"),
        "sb3-contrib": importlib.metadata.version("sb3-contrib"),
        "gymnasium": importlib.metadata.version("gymnasium"),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    write_new(out_path, rec)
    return rec
