from __future__ import annotations
from .protocol import BUDGET
import json
from pathlib import Path
def validate_ledger(rows: list[dict]) -> list[str]:
    errors=[]; counts={k:0 for k in BUDGET}
    for row in rows:
        stage=row.get("stage"); counts[stage]=counts.get(stage,0)+1
        if row.get("status") not in {"STARTED","COMPLETED","FAILED","PREPHYSICS_FAILED"}: errors.append("status")
    for stage,limit in BUDGET.items():
        if counts.get(stage,0)>limit: errors.append(f"budget:{stage}")
    if sum(counts.values())>sum(BUDGET.values()): errors.append("budget:total")
    return sorted(set(errors))
def append_instance(path: Path, row: dict) -> None:
    """Append a physical-instance ledger row atomically and reject duplicate grants."""
    rows=[]
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    key=row.get("authorization_id")
    if any(r.get("authorization_id")==key for r in rows): raise ValueError("duplicate grant consumption")
    errors=validate_ledger(rows+[row])
    if errors: raise ValueError("ledger rejected: "+", ".join(errors))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(row,sort_keys=True)+"\n")

def accounting(rows: list[dict]) -> dict[str, int]:
    return {
        "grant_attempts": len(rows),
        "prephysics_failures": sum(r.get("status") == "PREPHYSICS_FAILED" for r in rows),
        "physical_instances_started": sum(bool(r.get("physical_instance_started")) for r in rows),
        "physical_budget_used": sum(int(r.get("physical_budget_delta", 0)) for r in rows),
        "physical_budget_remaining": sum(BUDGET.values()) - sum(int(r.get("physical_budget_delta", 0)) for r in rows),
    }
