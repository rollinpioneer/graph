from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run(output: Path) -> dict[str, Any]:
    from upgrade_v2.l2r_canonical_time_confirmation.guard import CanonicalTimeGuard
    from upgrade_v2.l2r_logical_observation_clock.guard import INTERCEPT_REASON
    output.mkdir(parents=True, exist_ok=True)
    def row(ns: int, order: int, contact: Any, *, attempt: int = 1, command: str = "closed", effect: str = "HOLD_OBJECT", end: bool = False) -> dict[str, Any]:
        return {"physical_time_ns": ns, "capture_order": order, "attempt_id": attempt, "context_valid": True, "contact_present": contact, "gripper_command": command, "requested_effect": effect, "attempt_end": end}

    scenarios = [
        ("pending_then_confirm_100ms", [row(0, 0, False), row(100_000_000, 1, False)]),
        ("same_time_remains_pending", [row(0, 0, False), row(0, 1, False), row(100_000_000, 2, False)]),
        ("contact_true_passthrough", [row(0, 0, True)]),
        ("contact_null_clear_no_fake_contact", [row(0, 0, None)]),
        ("release_clears_pending", [row(0, 0, False), row(50_000_000, 1, False, command="open", effect="RELEASE_OBJECT")]),
        ("attempt_switch_clears_pending", [row(0, 0, False), row(50_000_000, 1, False, attempt=2)]),
        ("wrong_reason_passthrough", [row(0, 0, False)]),
    ]
    cases = []
    for name, rows in scenarios:
        guard = CanonicalTimeGuard(); outputs = []
        for current in rows:
            proposal = {"selected_action": "recover_object", "reason_code": INTERCEPT_REASON}
            if name == "wrong_reason_passthrough":
                proposal["reason_code"] = "NON_RELEASE_CONTACT_LOSS"
            try:
                outputs.append(guard.step(current, proposal))
            except Exception as exc:
                outputs.append({"error": type(exc).__name__})
        cases.append({"name": name, "intercept_reason": INTERCEPT_REASON, "rows": rows, "outputs": outputs})
    payload = {"schema": "l2rar2_r27_interface_shadow_v2", "cases": cases, "muJoCo_imported": False, "physical_executions": 0, "visual_proposal_requires_contact_false_for_clp3": True, "semantics": {"pending": "first exact-intercept proposal with measured contact=false", "confirmed": "strictly later same-attempt contact=false within 100ms", "passthrough": "contact=true, contact=null, invalid reason, release, or attempt switch are never converted to confirmation"}}
    (output / "contract_test_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["case,step,proposal_reason,measured_contact_value,clp3_input_proposal,clp3_output_action,guard_state,contact_loss_pending,interface_disposition,supervisor_action_request"]
    for item in cases:
        for index, out in enumerate(item["outputs"]):
            state = out.get("guard_state", "ERROR")
            semantic = {
                "CONFIRMED": "CONFIRMED_BY_CLP3",
                "PENDING": "PENDING_INTERCEPTED",
                "PASS_THROUGH": "PASSTHROUGH_NOT_CONFIRMED",
                "CLEAR": "PENDING_CLEARED_NOT_CONFIRMED",
            }.get(state, "ERROR_OR_UNKNOWN")
            contact = item["rows"][index].get("contact_present")
            lines.append(f"{item['name']},{index},{out.get('reason_code','')},{contact},{item['intercept_reason']},"
                         f"{out.get('selected_action','')},{state},{out.get('contact_loss_pending','')},{semantic},shadow_only")
    (output / "proposal_to_action.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
