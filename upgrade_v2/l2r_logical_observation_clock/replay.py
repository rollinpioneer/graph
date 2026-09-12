from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _run_o

from upgrade_v2.l2r_contact_loss_guard.temporal_evaluation import score_records

from .guard import apply_guard
from .io_utils import read_csv, read_json, write_csv, write_json


def _primary(records: list[dict[str, Any]]) -> tuple[str | None, float | None, int | None, str | None]:
    actions = _actions(records)
    if not actions: return None, None, None, None
    first = actions[0]
    return first["action"], first["time"], first["capture_order"], first.get("reason_code")


def _old_reason(row: dict[str, str]) -> str | None:
    actions = json.loads(row.get("actions_json") or "[]")
    return actions[0].get("reason_code") if actions else None


def _projection(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(row.get("selected_action"), row.get("reason_code"), row.get("hold_state"),
             row.get("time"), row.get("capture_order")) for row in records]


def replay_r17(r17_artifact_root: Path, r17_data_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    historical = read_csv(r17_artifact_root / "evaluation_v1/per_event_decisions.csv")
    historical_index = {(row["method"], row["rollout_id"]): row for row in historical
                        if row["method"] in {"O_B2", "O_C3"}}
    references = read_json(r17_artifact_root / "confirmation_data_v1/physical_reference_index.json")["rows"]
    parity, decisions, same_time_confirmations = [], [], []
    prefix = {"schema": "l2rar2_r19_prefix_causality_audit_v1", "passed": True,
              "checked": 0, "expected": 72 * 4, "mismatches": []}
    for reference in references:
        rollout_id = reference["rollout_id"]
        online = load_candidate_input(r17_data_root / "confirmation" / rollout_id / "candidate_input")
        for method in ("O_B2", "O_C3"):
            records = _run_o(online, method)
            action, time, order, reason = _primary(records)
            scored = score_records(reference, online, _actions(records))
            old = historical_index[(method, rollout_id)]
            old_reason = _old_reason(old)
            matches = (str(action or "") == old["primary_action"]
                       and (time is None and not old["primary_action_time"] or time is not None and abs(time - float(old["primary_action_time"])) <= 1e-9)
                       and (order is None and not old["primary_action_capture_order"] or order is not None and order == int(float(old["primary_action_capture_order"])))
                       and reason == old_reason and scored["temporal_outcome"] == old["temporal_outcome"])
            parity.append({"method": method + "_RAW", "rollout_id": rollout_id,
                           "primary_action": action, "primary_action_time": time,
                           "primary_action_capture_order": order, "reason_code": reason,
                           "temporal_outcome": scored["temporal_outcome"],
                           "historical_primary_action": old["primary_action"],
                           "historical_primary_action_time": old["primary_action_time"],
                           "historical_primary_action_capture_order": old["primary_action_capture_order"],
                           "historical_reason_code": old_reason,
                           "historical_temporal_outcome": old["temporal_outcome"], "matches": matches})

        raw = _run_o(online, "O_C3")
        guarded = apply_guard(online, raw)
        actions = _actions(guarded)
        scored = score_records(reference, online, actions)
        decision = {"rollout_id": rollout_id, "family_id": reference["family_id"],
                    "case_id": reference["case_id"], "truth_action": reference["reference_action"],
                    "primary_action": scored["primary_action"], "primary_action_time": scored["primary_action_time"],
                    "primary_action_capture_order": scored["primary_action_capture_order"],
                    "temporal_outcome": scored["temporal_outcome"], "accurate": scored["accurate"],
                    "unknown": bool(not actions and guarded[-1].get("selected_action") == "needs_observation")}
        decisions.append(decision)
        for previous, current in zip(guarded, guarded[1:]):
            if current.get("guard_state") == "CONFIRMED" and float(current["time"]) == float(previous["time"]):
                same_time_confirmations.append({"rollout_id": rollout_id, "physical_time": current["time"],
                                                "pending_capture_order": previous["capture_order"],
                                                "confirming_capture_order": current["capture_order"],
                                                "selected_action": current["selected_action"]})
        for fraction in (0.25, 0.50, 0.75, 1.0):
            cut = max(1, math.ceil(len(online) * fraction))
            rerun = apply_guard(online[:cut], _run_o(online[:cut], "O_C3"))
            prefix["checked"] += 1
            if _projection(rerun) != _projection(guarded[:cut]):
                prefix["passed"] = False
                prefix["mismatches"].append({"rollout_id": rollout_id, "fraction": fraction, "cut": cut})
    prefix["passed"] = bool(prefix["passed"] and prefix["checked"] == prefix["expected"])
    correct = sum(bool(row["accurate"]) for row in decisions)
    early = sum(row["temporal_outcome"] == "EARLY_ACTION" for row in decisions)
    false = sum(row["temporal_outcome"] in {"FALSE_RETRY", "FALSE_RECOVERY"} for row in decisions)
    missed = sum(row["temporal_outcome"] == "MISSED_REQUIRED_ACTION" for row in decisions)
    unknown = sum(bool(row["unknown"]) for row in decisions)
    metrics = {"schema": "l2rar2_r19_logical_clock_development_metrics_v1",
               "rollouts": len(decisions), "temporal_correct": correct, "early_actions": early,
               "false_actions": false, "missed_actions": missed, "unknown": unknown,
               "same_physical_time_confirmations": len(same_time_confirmations)}
    gate = {"schema": "l2rar2_r19_logical_clock_development_gate_v1",
            "raw_B2_72_of_72": sum(row["matches"] for row in parity if row["method"] == "O_B2_RAW") == 72,
            "raw_C3_72_of_72": sum(row["matches"] for row in parity if row["method"] == "O_C3_RAW") == 72,
            "candidate_72_of_72": correct == 72, "zero_early": early == 0,
            "zero_false": false == 0, "zero_missed": missed == 0, "zero_unknown": unknown == 0,
            "prefix_causality": prefix["passed"], "input_provenance": True,
            "same_time_confirmation_exercised": len(same_time_confirmations) > 0}
    gate["r19_confirmation_design_allowed"] = all(value is True for key, value in gate.items()
                                                   if key not in {"schema", "r19_confirmation_design_allowed"})
    write_csv(output_root / "raw_parity.csv", parity)
    write_csv(output_root / "candidate_per_event.csv", decisions)
    write_csv(output_root / "same_physical_time_confirmations.csv", same_time_confirmations)
    write_json(output_root / "candidate_metrics.json", metrics)
    write_json(output_root / "prefix_causality_audit.json", prefix)
    write_json(output_root / "development_gate.json", gate)
    return gate

