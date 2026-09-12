from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from .protocol import CAMPAIGN_ID, BUDGET
def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
def validate_approval(approval: dict, lock: dict, now: datetime | None = None) -> list[str]:
    errors=[]; now=now or datetime.now(timezone.utc)
    if approval.get("schema") != "l2rar2_fast_campaign_approval_v1": errors.append("schema")
    if approval.get("status") != "APPROVED": errors.append("status")
    if approval.get("campaign_id") != CAMPAIGN_ID: errors.append("campaign_id")
    if approval.get("campaign_lock_sha256") != canonical_hash(lock): errors.append("lock_hash")
    if approval.get("max_physical_instances") != sum(BUDGET.values()): errors.append("budget")
    if approval.get("automatic_stage_transition") is not True: errors.append("automatic_transition")
    if approval.get("stop_on_hard_gate_failure") is not True: errors.append("hard_stop")
    if approval.get("automatic_retry_after_physics_start") is not False: errors.append("retry")
    reviewer=str(approval.get("reviewer_id") or "").strip().lower()
    if reviewer in {"", "agent", "auto", "chatgpt", "unknown", "none"}: errors.append("reviewer")
    if not re.fullmatch(r"[0-9a-f]{64}", str(approval.get("campaign_nonce") or "")): errors.append("nonce")
    try:
        approved=datetime.fromisoformat(str(approval["approved_at_utc"]).replace("Z", "+00:00")); expires=datetime.fromisoformat(str(approval["expires_at_utc"]).replace("Z", "+00:00"))
        if approved.tzinfo is None or approved > now: errors.append("approved_at")
        if expires.tzinfo is None or expires <= now or expires <= approved: errors.append("expires")
        if (expires-approved).total_seconds()>7*86400: errors.append("window")
    except Exception: errors.append("time_format")
    return sorted(set(errors))
