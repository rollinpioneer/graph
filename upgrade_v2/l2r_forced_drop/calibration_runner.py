from __future__ import annotations
import json
from pathlib import Path
from .protocol import pulse_levels
def plan_calibration(output_root: Path) -> dict:
    result={"schema":"l2rar2_r16_calibration_plan_v1","levels":[{"id":p.level_id,"target_delta_v_local_mps":list(p.target_delta_v_local_mps)} for p in pulse_levels()],"max_instances":5,"physical_executions":0}
    output_root.mkdir(parents=True,exist_ok=True); (output_root/"calibration_plan.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
def run_calibration(*, output_root: Path, authorized: bool) -> dict:
    if not authorized: raise PermissionError("R16 calibration grant required")
    # MuJoCo is imported only by the caller after grant validation.
    return plan_calibration(output_root)
