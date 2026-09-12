from __future__ import annotations
import hashlib
from .protocol import CAMPAIGN_ID, budget_for
def derive_nonce(campaign_nonce: str, stage: str, index: int) -> str:
    return hashlib.sha256(f"{campaign_nonce}|{stage}|{index}".encode()).hexdigest()
def grants(approval: dict) -> list[dict]:
    out=[]
    for stage in ("A2","B","C","CALIBRATION","DEVELOPMENT"):
        for i in range(budget_for(stage)):
            out.append({"authorization_id":f"{CAMPAIGN_ID}:{stage}:{i}","stage":stage,"instance_index":i,"single_use_nonce":derive_nonce(approval["campaign_nonce"],stage,i),"reviewer_id":approval["reviewer_id"],"approved_at_utc":approval["approved_at_utc"],"expires_at_utc":approval["expires_at_utc"]})
    return out
