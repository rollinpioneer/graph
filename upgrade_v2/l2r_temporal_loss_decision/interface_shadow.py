from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run(output: Path) -> dict[str, Any]:
    from upgrade_v2.l2r_canonical_time_confirmation.guard import CanonicalTimeGuard
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for contact in (False, True, None):
        rows = [{"physical_time_ns": 0, "capture_order": 0, "attempt_id": 1, "context_valid": True, "contact_present": contact, "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "attempt_end": False}, {"physical_time_ns": 100_000_000, "capture_order": 1, "attempt_id": 1, "context_valid": True, "contact_present": contact, "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "attempt_end": False}]
        guard = CanonicalTimeGuard(); outputs = []
        for row in rows:
            proposal = {"selected_action": "recover_object", "reason_code": "NON_RELEASE_CONTACT_LOSS"}
            try:
                outputs.append(guard.step(row, proposal))
            except Exception as exc:
                outputs.append({"error": type(exc).__name__})
        cases.append({"contact": contact, "outputs": outputs})
    payload = {"schema": "l2rar2_r27_interface_shadow_v1", "cases": cases, "muJoCo_imported": False, "physical_executions": 0, "visual_proposal_requires_contact_false_for_clp3": True}
    (output / "contract_test_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["episode_id,proposal_reason,measured_contact_value,clp3_input_proposal,clp3_output_action,guard_state,supervisor_action_request"]
    for item in cases:
        contact = item["contact"]
        for out in item["outputs"]:
            lines.append(f"shadow_{contact},{out.get('reason_code','')},{contact},recover_object,{out.get('selected_action','')},{out.get('guard_state','')},shadow_only")
    (output / "proposal_to_action.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
