from __future__ import annotations

from pathlib import Path
from typing import Any


def source_audit(repo: Path) -> dict[str, Any]:
    module = repo / "upgrade_v2/l2r_logical_observation_clock"
    guard = (module / "guard.py").read_text(encoding="utf-8")
    replay = (module / "replay.py").read_text(encoding="utf-8")
    design = (module / "r19_design.py").read_text(encoding="utf-8")
    errors = []
    required = ("required_consecutive_valid_observations = 2", "maximum_physical_gap_s = 0.10",
                "minimum_physical_gap_s = 0.0", "logical_forward", "physical_nondecreasing",
                "0.0 <= physical_dt", '"time": now', '"capture_order": order')
    for token in required:
        if token not in guard: errors.append("guard_missing:" + token)
    for token in ("grid_search", "parameter_search", "for gap in", "for required_samples in"):
        if token in guard: errors.append("guard_search:" + token)
    for token in ("historical_reason_code", "prefix_causality", "input_provenance"):
        if token not in replay: errors.append("replay_missing:" + token)
    if "R19_DEVELOPMENT_GATE_REQUIRED" not in design: errors.append("design_not_fail_closed")
    if (module / "collector.py").exists(): errors.append("physical_collector_present")
    return {"schema": "l2rar2_r19_logical_clock_source_audit_v1",
            "status": "PASS" if not errors else "FAIL", "errors": errors,
            "physical_executions": 0, "mujoco_imported": False}

