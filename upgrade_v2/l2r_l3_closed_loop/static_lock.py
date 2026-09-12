from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import sha256, write_json

from . import BASE_COMMIT
from .protocol import ENVIRONMENT, FROZEN_BLOBS, protocol_lock
from .registry import registry

PATHS = {
    "clp3_guard": "upgrade_v2/l2r_canonical_time_confirmation/guard.py",
    "o_c3_evaluation": "upgrade_v2/l2r_rgb_temporal_confirmation/evaluation.py",
    "online_interface": "upgrade_v2/l2r_task_context/online_interface_repair.py",
    "hold_features": "upgrade_v2/l2r_hold_evidence/hold_features.py",
    "hold_predicates": "upgrade_v2/l2r_hold_evidence/hold_predicates.py",
    "physical_reference": "upgrade_v2/l2r_forced_drop/physical_reference.py",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def validate_frozen(repo: Path) -> dict:
    actual = {name: _git(repo, "rev-parse", f"HEAD:{path}") for name, path in PATHS.items()}
    return {"schema": "l2rar2_r23_frozen_parent_audit_v1", "base_commit": BASE_COMMIT,
            "base_is_ancestor": subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                                  BASE_COMMIT, "HEAD"]).returncode == 0,
            "expected_blobs": FROZEN_BLOBS, "actual_blobs": actual,
            "passed": actual == FROZEN_BLOBS}


def write_static(repo: Path, output: Path, runner_commit: str) -> dict:
    audit = validate_frozen(repo)
    module = repo / "upgrade_v2/l2r_l3_closed_loop"
    hashes = {str(path.relative_to(repo)): sha256(path) for path in sorted(module.rglob("*.py"))}
    run = {"schema": "l2rar2_r23_run_manifest_v1", "runner_commit": runner_commit,
           "python_version": platform.python_version(),
           "environment": {key: os.environ.get(key) for key in ENVIRONMENT},
           "physical_rollouts": 60, "workers_max": 2}
    write_json(output / "protocol_lock.json", protocol_lock())
    write_json(output / "registry.json", registry())
    write_json(output / "frozen_parent_audit.json", audit)
    write_json(output / "runner_file_hashes.json", {"files": hashes})
    write_json(output / "run_manifest.json", run)
    lock = {"schema": "l2rar2_r23_source_lock_v1", "runner_commit": runner_commit,
            "frozen_blobs": FROZEN_BLOBS, "runner_file_hashes": hashes,
            "git_clean": _git(repo, "status", "--porcelain") == ""}
    write_json(output / "source_lock.json", lock)
    return {"status": "PASS" if audit["passed"] and audit["base_is_ancestor"] else "FAIL", **audit}

