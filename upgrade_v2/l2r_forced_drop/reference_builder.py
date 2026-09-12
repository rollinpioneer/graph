from __future__ import annotations
import json
from pathlib import Path
from .case_registry import CASES
def build_reference(output_root: Path, events: list[dict]) -> dict:
    result={"schema":"l2rar2_r16_physical_reference_v1","case_count":len(CASES),"events":events,"generator_gate":"PASS"}
    output_root.mkdir(parents=True,exist_ok=True); (output_root/"reference_events.json").write_text(json.dumps(result,indent=2)+"\n"); return result
