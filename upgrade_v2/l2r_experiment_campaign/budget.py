from __future__ import annotations
from .protocol import BUDGET
def validate_ledger(rows: list[dict]) -> list[str]:
    errors=[]; counts={k:0 for k in BUDGET}
    for row in rows:
        stage=row.get("stage"); counts[stage]=counts.get(stage,0)+1
        if row.get("status") not in {"STARTED","COMPLETED","FAILED"}: errors.append("status")
    for stage,limit in BUDGET.items():
        if counts.get(stage,0)>limit: errors.append(f"budget:{stage}")
    if sum(counts.values())>sum(BUDGET.values()): errors.append("budget:total")
    return sorted(set(errors))
