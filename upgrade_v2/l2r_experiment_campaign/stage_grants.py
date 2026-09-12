from __future__ import annotations
import hashlib
from .protocol import CAMPAIGN_ID, budget_for
def derive_nonce(campaign_nonce: str, stage: str, index: int) -> str:
    return hashlib.sha256(f"{campaign_nonce}|{stage}|{index}".encode()).hexdigest()
def grants(approval: dict, *, retry_index: int = 0) -> list[dict]:
    if approval.get("status") != "APPROVED":
        raise PermissionError("campaign approval required")
    out=[]
    for stage in ("A2","B","C","CALIBRATION","DEVELOPMENT"):
        for i in range(budget_for(stage)):
            suffix = f":retry:{retry_index}" if retry_index else ""
            out.append({"authorization_id":f"{CAMPAIGN_ID}:{stage}:{i}{suffix}","stage":stage,"instance_index":i,"retry_index":retry_index,"single_use_nonce":derive_nonce(approval["campaign_nonce"],f"{stage}{suffix}",i),"reviewer_id":approval["reviewer_id"],"approved_at_utc":approval["approved_at_utc"],"expires_at_utc":approval["expires_at_utc"]})
    return out
