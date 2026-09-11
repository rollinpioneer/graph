"""O tier: the frozen Round-10 repaired online interface (no re-implementation)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_task_context.online_interface_repair import (
    FIXED_CANDIDATES,
    _first_emergency,
    _load_candidate,
    run_repaired_interface,
)
from upgrade_v2.l2r_task_context.reference import build_reference

O_METHOD_TO_CANDIDATE = {"O_B2": "B_count2", "O_C3": "C3_vector_rho035"}
TIME_ATOL = 1e-9

TRACE_FIELDS = (
    "time",
    "capture_order",
    "current_hold_evidence",
    "historical_hold_established",
    "contact_loss_observed",
    "data_status",
    "hold_state",
    "selected_action",
    "reason_code",
    "pending_event_id",
    "request_id",
    "requested_effect",
    "context_valid",
)


def run_o_method(meta: dict[str, Any], method_id: str) -> dict[str, Any]:
    """Run one O method through the frozen Round-10 interface."""
    candidate_id = O_METHOD_TO_CANDIDATE[method_id]
    observations, evidence, requests = _load_candidate(meta, candidate_id)
    rows = run_repaired_interface(observations, evidence, requests, history_complete=True)
    reference = build_reference(meta)
    trace = [
        {
            "method_id": method_id,
            "input_tier": "O",
            "candidate_id": candidate_id,
            "rollout_id": meta.get("rollout_id"),
            "root_family_id": meta.get("root_family_id"),
            **{key: row.get(key) for key in TRACE_FIELDS},
        }
        for row in rows
    ]
    if reference.get("status") != "reference_labeled":
        return {
            "method_id": method_id,
            "candidate_id": candidate_id,
            "trace": trace,
            "decision": {
                "candidate_id": candidate_id,
                "reference_status": "reference_unresolved",
                "reference_reason": reference.get("reason"),
                "expected_action": None,
                "selected_action": None,
                "selected_time": None,
                "correct": None,
                "selected_reason": None,
            },
        }
    event = reference["events"][0]
    start = float(event["decision_window_start"])
    end = float(event["decision_window_end"])
    selected = _first_emergency(rows, start, end)
    expected = event["expected_action"]
    selected_action = selected["selected_action"] if selected else "none"
    return {
        "method_id": method_id,
        "candidate_id": candidate_id,
        "trace": trace,
        "decision": {
            "candidate_id": candidate_id,
            "reference_status": "reference_labeled",
            "reference_reason": None,
            "expected_action": expected,
            "selected_action": selected_action,
            "selected_time": selected.get("time") if selected else None,
            "correct": selected_action == expected,
            "selected_reason": selected.get("reason_code") if selected else "no_emergency_condition",
            "historical_hold_seen": any(
                row["historical_hold_established"]
                for row in rows
                if float(row["time"]) <= start + TIME_ATOL
            ),
            "data_status_at_selection": selected.get("data_status") if selected else rows[-1]["data_status"],
            "hold_state_at_selection": selected.get("hold_state") if selected else rows[-1]["hold_state"],
            "window_start": start,
            "window_end": end,
        },
    }


def load_round10_comparison(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parity_rows(round10_rows: list[dict[str, Any]], v2_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare the frozen Round-10 rows with the R15-B O-layer rows."""
    frozen = {(row["candidate_id"], row["rollout_id"]): row for row in round10_rows}
    output = []
    for decision in v2_decisions:
        key = (decision["candidate_id"], decision["rollout_id"])
        reference = frozen.get(key)
        if reference is None:
            output.append(
                {
                    "method_id": decision["method_id"],
                    "candidate_id": decision["candidate_id"],
                    "rollout_id": decision["rollout_id"],
                    "matched": False,
                    "mismatch_fields": "missing_round10_row",
                }
            )
            continue
        mismatches = []
        if reference.get("selected_action") != decision.get("selected_action"):
            mismatches.append("selected_action")
        if str(reference.get("correct")).lower() != str(decision.get("correct")).lower():
            mismatches.append("correct")
        if (reference.get("selected_reason") or None) != (decision.get("selected_reason") or None):
            mismatches.append("selected_reason")
        frozen_time = reference.get("selected_time")
        current_time = decision.get("selected_time")
        if (frozen_time in (None, "")) and (current_time is None):
            time_match = True
        elif (frozen_time in (None, "")) or (current_time is None):
            time_match = False
        else:
            time_match = abs(float(frozen_time) - float(current_time)) <= TIME_ATOL
        if not time_match:
            mismatches.append("selected_time")
        output.append(
            {
                "method_id": decision["method_id"],
                "candidate_id": decision["candidate_id"],
                "rollout_id": decision["rollout_id"],
                "round10_selected_action": reference.get("selected_action"),
                "v2_selected_action": decision.get("selected_action"),
                "round10_selected_time": frozen_time,
                "v2_selected_time": current_time,
                "round10_selected_reason": reference.get("selected_reason"),
                "v2_selected_reason": decision.get("selected_reason"),
                "round10_correct": reference.get("correct"),
                "v2_correct": decision.get("correct"),
                "matched": not mismatches,
                "mismatch_fields": ",".join(mismatches),
            }
        )
    return output


def parity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = sum(1 for row in rows if row["matched"])
    return {
        "schema": "l2rar2_r15b_round10_parity_v1",
        "rows": len(rows),
        "matched": matched,
        "mismatched": len(rows) - matched,
        "all_matched": matched == len(rows) and len(rows) == 64,
    }
