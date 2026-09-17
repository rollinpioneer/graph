"""Read-only campaign monitoring and disk gates."""
from __future__ import annotations
from pathlib import Path
from .constants_b import DISK_LIMIT_BYTES
from .errors import ResourceLimit
from .io_utils import write_new

def campaign_bytes(root):
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return int(total)

def assert_disk(root, extra=0):
    used = campaign_bytes(root) + int(extra)
    if used > DISK_LIMIT_BYTES:
        raise ResourceLimit("RESOURCE_LIMIT_STOP")
    return used

def write_status(path, ledger, campaign_root):
    camp = ledger.campaign() or {}
    counts = ledger.counts()
    rec = {
        "schema": "P2CRL_CAMPAIGN_STATUS_V1",
        "campaign": camp,
        "counts": {f"{k[0]}:{k[1]}": v for k, v in counts.items()},
        "campaign_bytes": campaign_bytes(campaign_root),
        "stop_new_claims": bool(camp.get("stop_new_claims")),
    }
    write_new(path, rec)
    return rec
