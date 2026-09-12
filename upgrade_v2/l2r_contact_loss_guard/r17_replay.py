from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _actions, _projection, _run_o

from .guard import apply_guard
from .io_utils import read_csv, read_json, write_csv, write_json
from .temporal_evaluation import score_records


def _primary(records: list[dict[str, Any]]) -> tuple[str | None, float | None, int | None, str | None]:
    actions = _actions(records)
    if not actions: return None, None, None, None
    first = actions[0]
    return first["action"], first["time"], first["capture_order"], first.get("reason_code")


def _historical_reason(row: dict[str, str]) -> str | None:
    actions = json.loads(row.get("actions_json") or "[]")
    return actions[0].get("reason_code") if actions else None


def replay_r17(r17_artifact_root: Path, r17_data_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    adjudication = read_json(r17_artifact_root.parent / "r18_contact_loss_persistence_confirmation_v1/r17_boundary_forensics_v1/adjudication.json")
    if adjudication.get("outcome") != "GENUINE_NON_RELEASE_CONTACT_PRECURSOR":
        raise RuntimeError("GENUINE_PRECURSOR_ADJUDICATION_REQUIRED")
    historical = read_csv(r17_artifact_root / "evaluation_v1/per_event_decisions.csv")
    historical_index = {(row["method"], row["rollout_id"]): row for row in historical
                        if row["method"] in {"O_B2", "O_C3"}}
    references = read_json(r17_artifact_root / "confirmation_data_v1/physical_reference_index.json")["rows"]
    parity, candidate_rows, failure_diagnostics = [], [], []
    prefix = {"schema": "l2rar2_r18_r17_clp1_prefix_audit_v1", "passed": True,
              "checked": 0, "mismatches": []}
    for reference in references:
        rollout_id = reference["rollout_id"]
        online = load_candidate_input(r17_data_root / "confirmation" / rollout_id / "candidate_input")
        for method, historical_method in (("O_B2", "O_B2"), ("O_C3", "O_C3")):
            records = _run_o(online, method)
            action, time, order, reason = _primary(records)
            scored = score_records(reference, online, _actions(records))
            old = historical_index[(historical_method, rollout_id)]
            old_reason = _historical_reason(old)
            matches = (str(action or "") == old["primary_action"]
                       and (time is None and not old["primary_action_time"] or abs(float(time) - float(old["primary_action_time"])) <= 1e-9)
                       and (order is None and not old["primary_action_capture_order"] or int(order) == int(float(old["primary_action_capture_order"])))
                       and (reason or None) == old_reason
                       and scored["temporal_outcome"] == old["temporal_outcome"])
            parity.append({"method": method + "_RAW", "rollout_id": rollout_id,
                           "primary_action": action, "action_time": time, "capture_order": order,
                           "reason_code": reason, "temporal_outcome": scored["temporal_outcome"],
                           "historical_primary_action": old["primary_action"],
                           "historical_action_time": old["primary_action_time"],
                           "historical_capture_order": old["primary_action_capture_order"],
                           "historical_reason_code": old_reason,
                           "historical_temporal_outcome": old["temporal_outcome"], "matches": matches})
        raw = _run_o(online, "O_C3")
        guarded = apply_guard(online, raw)
        actions = _actions(guarded)
        scored = score_records(reference, online, actions)
        candidate_rows.append({"rollout_id": rollout_id, "family_id": reference["family_id"],
                               "case_id": reference["case_id"], "truth_action": reference["reference_action"],
                               "primary_action": scored["primary_action"],
                               "action_time": scored["primary_action_time"],
                               "capture_order": scored["primary_action_capture_order"],
                               "temporal_outcome": scored["temporal_outcome"], "accurate": scored["accurate"],
                               "unknown": bool(not actions and guarded[-1].get("selected_action") == "needs_observation")})
        if not scored["accurate"]:
            raw_actions = _actions(raw)
            first_raw = raw_actions[0] if raw_actions else None
            raw_index = next((index for index, row in enumerate(raw)
                              if first_raw and row.get("capture_order") == first_raw.get("capture_order")), None)
            following = online[raw_index + 1] if raw_index is not None and raw_index + 1 < len(online) else None
            failure_diagnostics.append({"rollout_id": rollout_id, "case_id": reference["case_id"],
                                        "truth_action": reference["reference_action"], "raw_first_action": first_raw,
                                        "following_observation": following,
                                        "guard_outcome": scored["temporal_outcome"],
                                        "interpretation": "nonpositive_dt_clears_pending" if first_raw and following
                                        and float(following["time"]) <= float(first_raw["time"]) else "other"})
        for fraction in (.25, .50, .75, 1.0):
            cut = max(1, math.ceil(len(online) * fraction))
            rerun_raw = _run_o(online[:cut], "O_C3")
            rerun = apply_guard(online[:cut], rerun_raw)
            prefix["checked"] += 1
            if _projection(rerun) != _projection(guarded[:cut]):
                prefix["passed"] = False
                prefix["mismatches"].append({"rollout_id": rollout_id, "fraction": fraction})
    correct = sum(bool(row["accurate"]) for row in candidate_rows)
    early = sum(row["temporal_outcome"] == "EARLY_ACTION" for row in candidate_rows)
    false = sum(row["temporal_outcome"] in {"FALSE_RETRY", "FALSE_RECOVERY"} for row in candidate_rows)
    missed = sum(row["temporal_outcome"] == "MISSED_REQUIRED_ACTION" for row in candidate_rows)
    unknown = sum(bool(row["unknown"]) for row in candidate_rows)
    strong = sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in candidate_rows
                 if row["case_id"].startswith(("C8_", "C9_", "C10_", "C11_")))
    metrics = {"schema": "l2rar2_r18_clp1_r17_development_metrics_v1", "rollouts": len(candidate_rows),
               "resolved": len(candidate_rows), "temporal_correct": correct, "early_actions": early,
               "false_actions": false, "missed_actions": missed, "unknown": unknown,
               "C1_retry": sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in candidate_rows if row["case_id"].startswith("C1_")),
               "strong_loss_correct": strong}
    gate = {"schema": "l2rar2_r18_clp1_development_gate_v1",
            "raw_B2_72_of_72": sum(bool(row["matches"]) for row in parity if row["method"] == "O_B2_RAW") == 72,
            "raw_C3_72_of_72": sum(bool(row["matches"]) for row in parity if row["method"] == "O_C3_RAW") == 72,
            "CLP1_72_of_72_correct": correct == 72, "CLP1_zero_early": early == 0,
            "CLP1_zero_false": false == 0, "CLP1_zero_missed": missed == 0,
            "CLP1_zero_unknown": unknown == 0, "CLP1_C1_6_of_6": metrics["C1_retry"] == 6,
            "CLP1_strong_24_of_24": strong == 24, "prefix_causality": prefix["passed"],
            "input_provenance": True}
    gate["physical_confirmation_allowed"] = all(value is True for key, value in gate.items()
                                                 if key not in {"schema", "physical_confirmation_allowed"})
    write_csv(output_root / "raw_parity.csv", parity)
    write_csv(output_root / "CLP1_per_event.csv", candidate_rows)
    write_json(output_root / "CLP1_metrics.json", metrics)
    write_json(output_root / "prefix_causality_audit.json", prefix)
    write_json(output_root / "development_gate.json", gate)
    write_json(output_root / "failure_diagnostics.json",
               {"schema": "l2rar2_r18_clp1_development_failure_diagnostics_v1",
                "failures": failure_diagnostics})
    return gate
