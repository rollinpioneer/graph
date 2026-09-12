from __future__ import annotations
import json
from pathlib import Path
def write_report(root: Path, *, lock: dict, state: dict, decision: str = "PENDING_APPROVAL") -> Path:
    path=root/"final_report.md"; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("# L2RAR2 Fast Experiment Campaign\n\n"+f"- campaign ID: {lock.get('campaign_id')}\n- state: {state.get('state')}\n- decision: {decision}\n- physical instances used: {state.get('physical_budget_used',0)}\n",encoding="utf-8"); return path
