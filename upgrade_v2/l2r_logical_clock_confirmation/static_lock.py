from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import BASE_COMMIT, FORMAL_MAIN
from .io_utils import sha256, write_json
from .protocol import FROZEN_BLOBS


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def write_locks(repo: Path, output_root: Path, runner_commit: str) -> dict[str, Any]:
    module = repo / "upgrade_v2/l2r_logical_clock_confirmation"
    hashes = {str(path.relative_to(repo)): sha256(path) for path in sorted(module.rglob("*.py"))}
    blobs = {"guard": git(repo, "hash-object", "upgrade_v2/l2r_logical_observation_clock/guard.py"),
             "detector": git(repo, "hash-object", "upgrade_v2/visual_refine_l2/vision.py"),
             "renderer": git(repo, "hash-object", "upgrade_v2/visual_refine_l2/renderer.py"),
             "temporal_scoring": git(repo, "hash-object", "upgrade_v2/l2r_rgb_temporal_confirmation/temporal_scoring.py"),
             "r17_difficulty_ladder": git(repo, "hash-object", "artifacts/pathgraph_sarm/upgrade_v2/r17_true_rgb_temporal_confirmation_v1/difficulty_calibration_v1/selected_difficulty_ladder.json"),
             "r19_protocol": git(repo, "hash-object", "artifacts/pathgraph_sarm/upgrade_v2/r19_logical_observation_clock_development_v1/static_v1/protocol_lock.json"),
             "r19_registry": git(repo, "hash-object", "artifacts/pathgraph_sarm/upgrade_v2/r19_logical_observation_clock_development_v1/r19_confirmation_design_v1/confirmation_registry.json")}
    source = {"schema": "l2rar2_r20_source_lock_v1", "base_commit": BASE_COMMIT,
              "runner_commit": runner_commit, "formal_main_commit": FORMAL_MAIN,
              "frozen_blobs": blobs, "expected_frozen_blobs": FROZEN_BLOBS,
              "all_frozen_blobs_match": blobs == FROZEN_BLOBS,
              "runner_file_hashes": hashes, "git_clean": git(repo, "status", "--porcelain") == ""}
    run = {"schema": "l2rar2_r20_run_manifest_v1", "runner_commit": runner_commit,
           "python_version": platform.python_version(), "environment": {
               key: os.environ.get(key) for key in ("PYTHONHASHSEED", "PYTHONNOUSERSITE", "MUJOCO_GL",
                                                     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                     "NUMEXPR_NUM_THREADS")},
           "workers_physics_max": 2, "workers_detector_max": 4, "l3_entry_allowed": False}
    write_json(output_root / "runner_file_hashes.json", {"schema": "l2rar2_r20_runner_file_hashes_v1", "files": hashes})
    write_json(output_root / "source_lock.json", source); write_json(output_root / "run_manifest.json", run)
    return source
