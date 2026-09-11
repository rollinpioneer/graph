from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import canonical_hash, sha256

RUNNER_GLOBS = (
    "upgrade_v2/l2r_reproducible_baseline/*.py",
)
DEPENDENCY_FILES = (
    "upgrade_v2/l2r_task_context/collector.py",
    "upgrade_v2/l2r_hold_evidence/probe_adapter.py",
    "upgrade_v2/visual_refine_l2/dynamic_simulator.py",
    "upgrade_v2/visual_refine_l2/repaired_simulator.py",
    "upgrade_v2/visual_refine_l2/renderer.py",
    "upgrade_v2/visual_refine_l2/vision.py",
)


def current_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def runner_file_hashes(repo: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in RUNNER_GLOBS:
        paths.update(repo.glob(pattern))
    paths.update(repo / relative for relative in DEPENDENCY_FILES)
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.relative_to(repo))] = sha256(path)
    return result


def make_source_lock(repo: Path, protocol_sha256: str, *, analysis_files: dict[str, str] | None = None) -> dict[str, Any]:
    files = runner_file_hashes(repo)
    return {
        "schema": "l2rar2_r14b_source_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_commit": current_commit(repo),
        "protocol_sha256": protocol_sha256,
        "generation_runner_files": files,
        "analysis_files": analysis_files or {},
        "git_status_clean": not bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], text=True, stdout=subprocess.PIPE, check=True).stdout.strip()),
        "program_sha256": None,
        "physical_spec_sha256": None,
        "generated_model_xml_sha256": None,
        "source_lock_sha256": None,
    }


def finalize_source_lock(lock: dict[str, Any]) -> dict[str, Any]:
    result = dict(lock)
    result["source_lock_sha256"] = canonical_hash({key: result[key] for key in result if key != "source_lock_sha256"})
    return result
