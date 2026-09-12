from __future__ import annotations
import subprocess
from pathlib import Path
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import sha256,write_json
from . import BASE_COMMIT
from .protocol import FROZEN_BLOBS,protocol_lock
from .registry import registry
PATHS={"clp3_guard":"upgrade_v2/l2r_canonical_time_confirmation/guard.py","canonical_capture":"upgrade_v2/l2r_canonical_time_confirmation/collector.py","detector":"upgrade_v2/visual_refine_l2/vision.py","contact_proxy":"upgrade_v2/l2r_rgb_temporal_confirmation/rgb_capture.py","r23_controller":"upgrade_v2/l2r_l3_closed_loop/runner.py"}
def _git(repo,*a):return subprocess.run(["git","-C",str(repo),*a],check=True,text=True,capture_output=True).stdout.strip()
def validate(repo:Path):
 actual={k:_git(repo,"rev-parse",f"HEAD:{v}") for k,v in PATHS.items()};return {"schema":"l2rar2_r24_frozen_audit_v1","base_commit":BASE_COMMIT,"base_is_ancestor":subprocess.run(["git","-C",str(repo),"merge-base","--is-ancestor",BASE_COMMIT,"HEAD"]).returncode==0,"expected":FROZEN_BLOBS,"actual":actual,"passed":actual==FROZEN_BLOBS}
def write_static(repo:Path,out:Path,commit:str):
 audit=validate(repo);files={str(p.relative_to(repo)):sha256(p) for p in sorted((repo/"upgrade_v2/l2r_l3_observation_boundary").rglob("*.py"))};write_json(out/"protocol_lock.json",protocol_lock());write_json(out/"registry.json",registry());write_json(out/"frozen_audit.json",audit);write_json(out/"source_lock.json",{"schema":"l2rar2_r24_source_lock_v1","runner_commit":commit,"frozen_blobs":FROZEN_BLOBS,"runner_file_hashes":files,"git_clean":_git(repo,"status","--porcelain")==""});return {"status":"PASS" if audit["passed"] else "FAIL",**audit}

