from __future__ import annotations
import os,platform,subprocess
from pathlib import Path
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import sha256,write_json

FROZEN_PATHS=("upgrade_v2/l2r_rgb_temporal_confirmation/evaluation.py","upgrade_v2/l2r_rgb_temporal_confirmation/temporal_scoring.py","upgrade_v2/l2r_forced_drop/physical_reference.py","upgrade_v2/l2r_logical_clock_confirmation/collector.py","artifacts/pathgraph_sarm/upgrade_v2/r17_true_rgb_temporal_confirmation_v1/difficulty_calibration_v1/selected_difficulty_ladder.json")


def _git(repo,*args): return subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,check=True).stdout.strip()
def write_locks(repo:Path,output:Path,commit:str)->dict:
    module=repo/"upgrade_v2/l2r_canonical_time_confirmation"; files={str(p.relative_to(repo)):sha256(p) for p in sorted(module.rglob("*.py"))}; blobs={p:_git(repo,"hash-object",p) for p in FROZEN_PATHS}
    result={"schema":"l2rar2_r22_source_lock_v1","runner_commit":commit,"candidate_guard_blob":_git(repo,"hash-object","upgrade_v2/l2r_canonical_time_confirmation/guard.py"),"frozen_source_blobs":blobs,"runner_file_hashes":files,"git_clean":_git(repo,"status","--porcelain")==""}
    run={"schema":"l2rar2_r22_run_manifest_v1","runner_commit":commit,"python_version":platform.python_version(),"environment":{k:os.environ.get(k) for k in ("PYTHONHASHSEED","PYTHONNOUSERSITE","MUJOCO_GL","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},"physical_rollouts":24,"workers_max":2,"l3_entry_allowed":False}
    write_json(output/"source_lock.json",result); write_json(output/"run_manifest.json",run); write_json(output/"runner_file_hashes.json",{"files":files}); return result
