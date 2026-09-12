from __future__ import annotations
import json
from pathlib import Path
from .case_registry import development_case_ids
def development_manifest(output_root: Path) -> dict:
    rows=[{"rollout_index":i,"case_id":case,"family_id":f"FAM_{i//8+1:02d}","seed":870000+i} for i,case in enumerate(development_case_ids()*4)]
    return {"schema":"l2rar2_r16_development_manifest_v1","rollouts":rows,"count":len(rows),"physical_executions":0}
def run_development(*, output_root: Path, authorized: bool) -> dict:
    if not authorized: raise PermissionError("R16 development grant required")
    output_root.mkdir(parents=True,exist_ok=True); result=development_manifest(output_root)
    (output_root/"development_manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
