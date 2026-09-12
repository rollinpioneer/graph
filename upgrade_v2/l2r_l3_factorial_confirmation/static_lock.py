from __future__ import annotations

import subprocess
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import sha256, write_json

from . import BASE_COMMIT
from .protocol import protocol_lock
from .registry import registry


FROZEN_PATHS = {
    "clp3_guard": "upgrade_v2/l2r_canonical_time_confirmation/guard.py",
    "canonical_time_adapter": "upgrade_v2/l2r_canonical_time_confirmation/collector.py",
    "visual_detector": "upgrade_v2/visual_refine_l2/vision.py",
    "contact_proxy": "upgrade_v2/l2r_rgb_temporal_confirmation/rgb_capture.py",
    "r23_controller": "upgrade_v2/l2r_l3_closed_loop/runner.py",
    "r24_runner": "upgrade_v2/l2r_l3_observation_boundary/runner.py",
    "r24_decision": "artifacts/pathgraph_sarm/upgrade_v2/r24_observation_boundary_closed_loop_v1/results_v1/final_v1/decision.json",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def validate(repo: Path) -> dict:
    expected = {key: _git(repo, "rev-parse", f"{BASE_COMMIT}:{path}") for key, path in FROZEN_PATHS.items()}
    actual = {key: _git(repo, "rev-parse", f"HEAD:{path}") for key, path in FROZEN_PATHS.items()}
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"]).returncode == 0
    return {"schema": "l2rar2_r25_frozen_parent_audit_v1", "base_commit": BASE_COMMIT,
            "base_is_ancestor": ancestor, "expected": expected, "actual": actual,
            "passed": ancestor and expected == actual}


def write_static(repo: Path, output: Path, commit: str) -> dict:
    audit = validate(repo)
    files = {}
    for package in ("l2r_loss_observability_fusion", "l2r_recovery_supervisor", "l2r_l3_factorial_confirmation"):
        for path in sorted((repo / "upgrade_v2" / package).rglob("*.py")):
            files[str(path.relative_to(repo))] = sha256(path)
    write_json(output / "frozen_parent_audit.json", audit)
    write_json(output / "protocol_lock.json", protocol_lock())
    write_json(output / "registry.json", registry())
    write_json(output / "source_lock.json", {"schema": "l2rar2_r25_source_lock_v1",
                                              "runner_commit": commit,
                                              "runner_file_hashes": files,
                                              "git_clean": _git(repo, "status", "--porcelain") == ""})
    return {"status": "PASS" if audit["passed"] else "FAIL", **audit}
