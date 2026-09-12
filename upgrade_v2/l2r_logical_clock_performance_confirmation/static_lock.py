from __future__ import annotations

import os, platform, subprocess
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import sha256, write_json
from .protocol import FROZEN_BLOBS

PATHS={
    "guard":"upgrade_v2/l2r_logical_observation_clock/guard.py",
    "temporal_scoring":"upgrade_v2/l2r_rgb_temporal_confirmation/temporal_scoring.py",
    "physical_reference":"upgrade_v2/l2r_forced_drop/physical_reference.py",
    "r20_collector":"upgrade_v2/l2r_logical_clock_confirmation/collector.py",
    "o_tier_runner":"upgrade_v2/l2r_rgb_temporal_confirmation/evaluation.py",
    "online_interface":"upgrade_v2/l2r_task_context/online_interface_repair.py",
    "hold_predicates":"upgrade_v2/l2r_hold_evidence/hold_predicates.py",
    "hold_features":"upgrade_v2/l2r_hold_evidence/hold_features.py",
    "physical_levels":"artifacts/pathgraph_sarm/upgrade_v2/r17_true_rgb_temporal_confirmation_v1/difficulty_calibration_v1/selected_difficulty_ladder.json",
}


def _git(repo:Path,*args:str)->str:
    return subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,check=True).stdout.strip()


def write_locks(repo:Path,output:Path,runner_commit:str)->dict:
    actual={name:_git(repo,"hash-object",path) for name,path in PATHS.items()}
    module=repo/"upgrade_v2/l2r_logical_clock_performance_confirmation"
    files={str(path.relative_to(repo)):sha256(path) for path in sorted(module.rglob("*.py"))}
    source={"schema":"l2rar2_r21_source_lock_v1","runner_commit":runner_commit,"frozen_blobs":actual,
            "expected_frozen_blobs":FROZEN_BLOBS,"all_frozen_blobs_match":actual==FROZEN_BLOBS,
            "runner_file_hashes":files,"git_clean":_git(repo,"status","--porcelain")==""}
    run={"schema":"l2rar2_r21_run_manifest_v1","runner_commit":runner_commit,"python_version":platform.python_version(),
         "environment":{key:os.environ.get(key) for key in ("PYTHONHASHSEED","PYTHONNOUSERSITE","MUJOCO_GL","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
         "physical_workers_max":2,"physical_rollouts":32,"l3_entry_allowed":False}
    write_json(output/"source_lock.json",source); write_json(output/"runner_file_hashes.json",{"schema":"l2rar2_r21_runner_hashes_v1","files":files}); write_json(output/"run_manifest.json",run)
    return source
