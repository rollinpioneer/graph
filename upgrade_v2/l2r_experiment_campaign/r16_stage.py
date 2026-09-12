from __future__ import annotations
from pathlib import Path
from upgrade_v2.l2r_forced_drop.calibration_runner import run_calibration
from upgrade_v2.l2r_forced_drop.development_runner import run_development
def calibration(grant: dict, output_root: Path):
    return run_calibration(output_root=output_root, authorized=bool(grant.get("single_use_nonce")))
def development(grants: list[dict], output_root: Path):
    if not grants or any(not g.get("single_use_nonce") for g in grants): raise PermissionError("valid development grants required")
    return run_development(output_root=output_root, authorized=True)
