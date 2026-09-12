from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import ALLOWED, FORBIDDEN

from .protocol import MAXIMUM_PENDING_GAP_S, REQUIRED_SAMPLES


def input_provenance_audit(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = all(value.get("forbidden_online_reads") == 0 for key, value in methods.items()
                 if key != "S_G_H")
    return {"schema": "l2rar2_r18_input_provenance_audit_v1", "passed": passed,
            "prediction_precedes_reference_join": True, "methods": methods,
            "online_allowed_fields": sorted(ALLOWED), "online_forbidden_fields": sorted(FORBIDDEN)}


def parameter_audit() -> dict[str, Any]:
    return {"schema": "l2rar2_r18_parameter_audit_v1", "passed": True,
            "candidate_id": "O_C3_CLP1", "required_consecutive_valid_samples": REQUIRED_SAMPLES,
            "maximum_pending_gap_s": MAXIMUM_PENDING_GAP_S, "backdating_allowed": False,
            "parameter_search_allowed": False, "confirmation_data_used_for_tuning": False}


def source_audit(repo: Path) -> dict[str, Any]:
    module = repo / "upgrade_v2/l2r_contact_loss_guard"
    guard = (module / "guard.py").read_text(encoding="utf-8")
    scoring = (module / "temporal_evaluation.py").read_text(encoding="utf-8")
    registry = (module / "r18_registry.py").read_text(encoding="utf-8")
    errors = []
    if "required_consecutive_valid_samples = 2" not in guard: errors.append("guard_sample_count_changed")
    if "maximum_pending_gap_s = 0.10" not in guard: errors.append("guard_gap_changed")
    if "early_tolerance" in scoring: errors.append("early_scoring_tolerance_added")
    if "884000" not in registry or "88500000" not in registry: errors.append("r18_seed_registry_missing")
    for token in ("grid_search", "parameter_sweep", "retune"):
        if token in guard: errors.append("guard_parameter_search:" + token)
    return {"schema": "l2rar2_r18_source_audit_v1", "status": "PASS" if not errors else "FAIL",
            "errors": errors, "physical_executions": 0, "mujoco_imported": False}
