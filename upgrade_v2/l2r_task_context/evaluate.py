from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _online_observation, _predicates
from upgrade_v2.l2r_hold_evidence.event_adapter import infer_history_complete, run_event_interface
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate

from .event_interface import run_m1
from .io import read_csv, read_json, read_jsonl, sha256, write_csv, write_json
from .reference import build_reference


METHODS = ("D0_generic_attempt_end", "M1_requested_effect_gate")
POSITIVE_CASES = (
    "K1_hold_request_ends_without_hold", "K4_regular_hold_loss",
    "K5_brief_hold_loss", "K6_long_gap_after_loss",
)
NEGATIVE_CASES = (
    "K2_touch_request_completes_without_hold", "K3_normal_hold_pause_resume",
    "K7_commanded_release", "K8_acquisition_touch_then_continue",
)


def _load_online(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_jsonl(Path(meta["path"]) / "observations_dense.jsonl")
    observations = [_online_observation(row) for row in raw]
    geometries = build_features(observations)
    predictions: list[dict[str, Any]] = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    evidence = evaluate_candidate(predictions, geometries, "B_count2")
    requests = read_jsonl(Path(meta["path"]) / "controller_requests.jsonl")
    return predictions, evidence, requests


def _audit_request(meta: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = {
        "scenario", "case_id", "root_family_id", "oracle", "weld_state",
        "future_outcome", "expected_action", "retry_grasp", "recover_object", "missed_grasp",
    }
    actions = read_csv(Path(meta["path"]) / "actions.csv")
    first_action_time = min(float(row["start_time"]) for row in actions) if actions else None
    reasons: list[str] = []
    if len(requests) != 1:
        reasons.append("expected_exactly_one_request")
    else:
        request = requests[0]
        if request.get("source") != "controller_dispatch":
            reasons.append("invalid_request_source")
        if request.get("requested_effect") != meta.get("requested_effect"):
            reasons.append("request_effect_metadata_mismatch")
        if first_action_time is None or float(request.get("received_time", float("inf"))) > first_action_time + 1e-9:
            reasons.append("request_not_received_before_first_action")
        if forbidden.intersection(request):
            reasons.append("forbidden_outcome_or_reference_field")
    return {
        "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"], "requested_effect": meta.get("requested_effect"),
        "source": requests[0].get("source") if len(requests) == 1 else "INVALID_COUNT",
        "first_action_time": first_action_time,
        "request_received_time": requests[0].get("received_time") if len(requests) == 1 else None,
        "valid": not reasons,
        "reasons": ";".join(reasons),
        "outcome_encoded": bool(len(requests) == 1 and forbidden.intersection(requests[0])),
    }


def _method_rows(method: str, observations: list[dict[str, Any]], evidence: list[dict[str, Any]], requests: list[dict[str, Any]], history_complete: bool) -> list[dict[str, Any]]:
    if method == "D0_generic_attempt_end":
        return run_event_interface(observations, evidence, history_complete=history_complete)
    if method == "M1_requested_effect_gate":
        return run_m1(observations, evidence, requests, history_complete=history_complete)
    raise ValueError(method)


def _emergencies(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    result = []
    for row in rows:
        guards = row.get("effective_guards", {})
        names = [name for name in ("retry_grasp", "recover_object") if guards.get(name) == "true"]
        if names:
            result.append((row, "conflict" if len(names) > 1 else names[0]))
    return result


def _event_decision(meta: dict[str, Any], method: str, rows: list[dict[str, Any]], evidence: list[dict[str, Any]], event: dict[str, Any]) -> dict[str, Any]:
    start, end = float(event["decision_window_start"]), float(event["decision_window_end"])
    window = [row for row in rows if start - 1e-9 <= float(row.get("time", -1)) <= end + 1e-9]
    emergency = _emergencies(window)
    selected_row, selected_action = emergency[0] if emergency else (None, "none")
    expected = event["expected_action"]
    correct = selected_action == expected if expected != "none" else not emergency
    before = [row for row in rows if float(row.get("time", -1)) < start - 1e-9]
    premature = bool(_emergencies(before))
    conflict = any(action == "conflict" for _, action in emergency)
    evidence_before = [row for row in evidence if float(row.get("time", -1)) <= start + 1e-9]
    hold_before = any(row.get("hold_memory") == "true" for row in evidence_before)
    delay = float(selected_row["time"]) - start if selected_row is not None and correct and expected != "none" else None
    return {
        "method": method,
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "event_id": event["event_id"],
        "reference_type": event["reference_type"],
        "expected_action": expected,
        "selected_action": selected_action,
        "selected_time": selected_row.get("time") if selected_row else None,
        "window_start": start,
        "window_end": end,
        "correct": correct,
        "missed": expected != "none" and not correct,
        "false_emergency": expected == "none" and bool(emergency),
        "premature_emergency": premature,
        "event_window_conflict": conflict,
        "rollout_conflict": any(action == "conflict" for _, action in _emergencies(rows)),
        "hold_evidence_before_event": hold_before,
        "hold_memory_at_last_closed_observation": next((row.get("hold_memory") for row in reversed(evidence) if row.get("closed") == "true"), "unknown"),
        "delay_seconds": delay,
        "reference_status": "reference_labeled",
        "request_source": "controller_dispatch_pre_action",
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def _recall(rows: list[dict[str, Any]]) -> float | None:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None


def _estimable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("reference_status") == "reference_labeled"]


def _estimable_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows or len(_estimable(rows)) != len(rows):
        return None
    return _rate(rows, key)


def _estimable_recall(rows: list[dict[str, Any]]) -> float | None:
    if not rows or len(_estimable(rows)) != len(rows):
        return None
    return _recall(rows)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _masked_diagnostics(meta: dict[str, Any], observations: list[dict[str, Any]], evidence: list[dict[str, Any]], requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    if meta["case_id"] == "K1_hold_request_ends_without_hold":
        for view, history, context in (("history_missing", False, requests), ("context_missing", True, [])):
            rows = run_m1(observations, evidence, context, history_complete=history)
            emergencies = _emergencies(rows)
            result.append({
                "root_family_id": meta["root_family_id"], "rollout_id": meta["rollout_id"],
                "case_id": meta["case_id"], "view": view,
                "retry": any(action in {"retry_grasp", "conflict"} for _, action in emergencies),
                "recover": any(action in {"recover_object", "conflict"} for _, action in emergencies),
                "selected_action": emergencies[0][1] if emergencies else rows[-1]["selected_action"],
                "physical_family_credit": 0,
            })
    if meta["case_id"] == "K4_regular_hold_loss":
        rows = run_m1(observations, evidence, [], history_complete=True)
        emergencies = _emergencies(rows)
        result.append({
            "root_family_id": meta["root_family_id"], "rollout_id": meta["rollout_id"],
            "case_id": meta["case_id"], "view": "context_missing_held_loss",
            "retry": any(action in {"retry_grasp", "conflict"} for _, action in emergencies),
            "recover": any(action in {"recover_object", "conflict"} for _, action in emergencies),
            "selected_action": emergencies[0][1] if emergencies else rows[-1]["selected_action"],
            "physical_family_credit": 0,
        })
    return result


def _metrics(method: str, decisions: list[dict[str, Any]], masked: list[dict[str, Any]]) -> dict[str, Any]:
    own = [row for row in decisions if row["method"] == method]
    by_case = {case: [row for row in own if row["case_id"] == case] for case in POSITIVE_CASES + NEGATIVE_CASES}
    positives = [row for row in own if row["case_id"] in POSITIVE_CASES]
    estimable_positives = _estimable(positives)
    delays = [float(row["delay_seconds"]) for row in estimable_positives if row["delay_seconds"] is not None]
    unresolved_count = sum(row.get("reference_status") != "reference_labeled" for row in own)
    missing_history = [row for row in masked if row["view"] == "history_missing"] if method.startswith("M1") else []
    missing_context = [row for row in masked if row["view"] == "context_missing"] if method.startswith("M1") else []
    row: dict[str, Any] = {
        "method": method,
        "events": len(own),
        "root_families": len({item["root_family_id"] for item in own}),
        "positive_events": len(positives),
        "estimable_positive_events": len(estimable_positives),
        "positive_correct": sum(bool(item["correct"]) for item in estimable_positives),
        "K1_recall": _estimable_recall(by_case[POSITIVE_CASES[0]]),
        "K4_recall": _estimable_recall(by_case[POSITIVE_CASES[1]]),
        "K5_recall": _estimable_recall(by_case[POSITIVE_CASES[2]]),
        "K6_recall": _estimable_recall(by_case[POSITIVE_CASES[3]]),
        "wrong_or_unknown_positive_rate": _estimable_rate(positives, "missed"),
        "K2_false_emergency_rate": _estimable_rate(by_case[NEGATIVE_CASES[0]], "false_emergency"),
        "K3_false_emergency_rate": _estimable_rate(by_case[NEGATIVE_CASES[1]], "false_emergency"),
        "K7_false_emergency_rate": _estimable_rate(by_case[NEGATIVE_CASES[2]], "false_emergency"),
        "K8_false_emergency_rate": _estimable_rate(by_case[NEGATIVE_CASES[3]], "false_emergency"),
        "premature_emergency_rate": _estimable_rate(positives, "premature_emergency"),
        "any_conflict_rollout_rate": len({item["rollout_id"] for item in own if item["rollout_conflict"]}) / len({item["rollout_id"] for item in own}) if own else None,
        "event_window_conflict_rate": _estimable_rate(own, "event_window_conflict"),
        "unjustified_definite_on_missing_history_rate": sum(row["retry"] or row["recover"] for row in missing_history) / len(missing_history) if missing_history else None,
        "unjustified_retry_on_missing_context_rate": sum(row["retry"] for row in missing_context) / len(missing_context) if missing_context else None,
        "regular_hold_evidence_rate": _estimable_rate(by_case[POSITIVE_CASES[1]], "hold_evidence_before_event"),
        "brief_hold_evidence_rate": _estimable_rate(by_case[POSITIVE_CASES[2]], "hold_evidence_before_event"),
        "touch_false_hold_evidence_rate": _estimable_rate(by_case[NEGATIVE_CASES[0]], "hold_evidence_before_event"),
        "pause_hold_retention_rate": _estimable_rate(
            [{**item, "pause_hold_retention": item["hold_memory_at_last_closed_observation"] == "true"} for item in by_case[NEGATIVE_CASES[1]]],
            "pause_hold_retention",
        ),
        "correct_detection_delay_p95_seconds": _p95(delays),
        "delay_observed_count": len(delays),
        "positive_missed_count": sum(bool(item["missed"]) for item in estimable_positives),
        "positive_unresolved_count": sum(row.get("reference_status") != "reference_labeled" for row in positives),
        "reference_unresolved": unresolved_count,
        "per_case_support": {
            case: {
                "events": len(rows),
                "estimable_events": len(_estimable(rows)),
                "reference_unresolved": sum(item.get("reference_status") != "reference_labeled" for item in rows),
                "root_families": len({item["root_family_id"] for item in rows}),
            }
            for case, rows in by_case.items()
        },
    }
    return row


def _unresolved_decision(
    meta: dict[str, Any],
    method: str,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    emergencies = _emergencies(rows)
    selected_row, selected_action = emergencies[0] if emergencies else (None, "none")
    return {
        "method": method,
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "event_id": f"{meta['rollout_id']}:primary",
        "reference_type": "reference_unresolved",
        "expected_action": None,
        "selected_action": selected_action,
        "selected_time": selected_row.get("time") if selected_row else None,
        "window_start": None,
        "window_end": None,
        "correct": None,
        "missed": None,
        "false_emergency": None,
        "premature_emergency": None,
        "event_window_conflict": None,
        "rollout_conflict": any(action == "conflict" for _, action in _emergencies(rows)),
        "hold_evidence_before_event": None,
        "hold_memory_at_last_closed_observation": "unknown",
        "delay_seconds": None,
        "reference_status": "reference_unresolved",
        "reference_reason": reference["reason"],
        "request_source": "controller_dispatch_pre_action",
    }


def gate_metrics(metrics: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    g = protocol["focused_gates"]
    checks: dict[str, bool] = {}
    for name in ("K1_recall", "K4_recall", "K5_recall", "K6_recall"):
        checks[name] = metrics[name] is not None and metrics[name] >= float(g["positive_type_recall_min"])
    checks["wrong_or_unknown_positive_rate"] = metrics["wrong_or_unknown_positive_rate"] is not None and metrics["wrong_or_unknown_positive_rate"] <= float(g["wrong_or_unknown_positive_rate_max"])
    for name in ("K2_false_emergency_rate", "K3_false_emergency_rate", "K7_false_emergency_rate", "K8_false_emergency_rate"):
        checks[name] = metrics[name] is not None and metrics[name] <= float(g["per_negative_stratum_false_emergency_max"])
    checks["premature_emergency_rate"] = metrics["premature_emergency_rate"] is not None and metrics["premature_emergency_rate"] <= float(g["premature_emergency_max"])
    checks["any_conflict_rollout_rate"] = metrics["any_conflict_rollout_rate"] is not None and metrics["any_conflict_rollout_rate"] <= float(g["any_conflict_rollout_max"])
    checks["event_window_conflict_rate"] = metrics["event_window_conflict_rate"] is not None and metrics["event_window_conflict_rate"] <= float(g["event_window_conflict_max"])
    checks["unjustified_definite_on_missing_history_rate"] = metrics["unjustified_definite_on_missing_history_rate"] is not None and metrics["unjustified_definite_on_missing_history_rate"] <= float(g["unjustified_definite_on_missing_history_max"])
    checks["unjustified_retry_on_missing_context_rate"] = metrics["unjustified_retry_on_missing_context_rate"] is not None and metrics["unjustified_retry_on_missing_context_rate"] <= float(g["unjustified_retry_on_missing_context_max"])
    checks["regular_hold_evidence_rate"] = metrics["regular_hold_evidence_rate"] is not None and metrics["regular_hold_evidence_rate"] >= float(g["hold_evidence_regular_and_brief_min"])
    checks["brief_hold_evidence_rate"] = metrics["brief_hold_evidence_rate"] is not None and metrics["brief_hold_evidence_rate"] >= float(g["hold_evidence_regular_and_brief_min"])
    checks["touch_false_hold_evidence_rate"] = metrics["touch_false_hold_evidence_rate"] is not None and metrics["touch_false_hold_evidence_rate"] <= float(g["touch_false_hold_evidence_max"])
    checks["pause_hold_retention_rate"] = metrics["pause_hold_retention_rate"] is not None and metrics["pause_hold_retention_rate"] >= float(g["pause_hold_retention_min"])
    checks["correct_detection_delay_p95_seconds"] = metrics["correct_detection_delay_p95_seconds"] is not None and metrics["correct_detection_delay_p95_seconds"] <= float(g["correct_detection_delay_p95_seconds_max"])
    checks["per_case_support"] = all(
        support["events"] >= int(g["each_required_case_min_events"]) and support["root_families"] >= int(g["each_required_case_min_root_families"])
        for support in metrics["per_case_support"].values()
    )
    checks["request_provenance"] = bool(metrics.get("request_provenance_all"))
    checks["reference_contract"] = int(metrics.get("reference_unresolved", 0)) == 0
    return {"checks": checks, "all_pass": all(checks.values()), "failed": [name for name, passed in checks.items() if not passed]}


def evaluate(data_root: Path, methods: list[str], protocol_path: Path, contract_root: Path, output_root: Path) -> dict[str, Any]:
    if tuple(methods) != METHODS:
        raise ValueError(f"methods must be {METHODS}")
    protocol = read_json(protocol_path)
    if read_json(contract_root / "contract_status.json").get("status") != "CONTRACT_LOCKED_FOR_NEW_COLLECTION":
        raise ValueError("contract is not locked")
    metadata = read_csv(data_root / "rollout_manifest.csv")
    decisions: list[dict[str, Any]] = []
    masked: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for meta in metadata:
        observations, evidence, requests = _load_online(meta)
        provenance.append(_audit_request(meta, requests))
        reference = build_reference(meta)
        if reference["status"] != "reference_labeled":
            unresolved.append({"rollout_id": meta["rollout_id"], "case_id": meta["case_id"], "reason": reference["reason"]})
            history = infer_history_complete(observations)
            for method in methods:
                rows = _method_rows(method, observations, evidence, requests, history)
                decisions.append(_unresolved_decision(meta, method, rows, reference))
            continue
        history = infer_history_complete(observations)
        for method in methods:
            rows = _method_rows(method, observations, evidence, requests, history)
            decisions.append(_event_decision(meta, method, rows, evidence, reference["events"][0]))
        masked.extend(_masked_diagnostics(meta, observations, evidence, requests))
    write_csv(output_root / "event_decisions.csv", decisions)
    write_csv(output_root / "masked_context_diagnostics.csv", masked)
    write_csv(output_root / "reference_unresolved.csv", unresolved, fields=["rollout_id", "case_id", "reason"])
    write_csv(output_root / "context_provenance.csv", provenance)
    metric_rows = [_metrics(method, decisions, masked) for method in methods]
    for row in metric_rows:
        row["request_provenance_all"] = bool(provenance) and all(item["valid"] for item in provenance)
        row["reference_unresolved"] = len(unresolved)
    csv_rows = [{**row, "per_case_support": __import__("json").dumps(row["per_case_support"], sort_keys=True)} for row in metric_rows]
    write_csv(output_root / "method_metrics.csv", csv_rows)
    per_case = []
    for method in methods:
        for case in POSITIVE_CASES + NEGATIVE_CASES:
            rows = [row for row in decisions if row["method"] == method and row["case_id"] == case]
            per_case.append({
                "method": method, "case_id": case, "events": len(rows),
                "root_families": len({row["root_family_id"] for row in rows}),
                "estimable_events": len(_estimable(rows)),
                "reference_unresolved": sum(row.get("reference_status") != "reference_labeled" for row in rows),
                "correct": sum(bool(row["correct"]) for row in rows if row.get("correct") is not None),
                "recall_or_specificity": _estimable_recall(rows),
                "false_emergency": sum(bool(row["false_emergency"]) for row in rows if row.get("false_emergency") is not None),
            })
    write_csv(output_root / "per_case_metrics.csv", per_case)
    selected_metrics = next(row for row in metric_rows if row["method"] == "M1_requested_effect_gate")
    gates = gate_metrics(selected_metrics, protocol)
    if unresolved:
        gates["all_pass"] = False
        gates["failed"].append("reference_unresolved")
    gate_record = {
        "schema": "pathgraph_l2rar2_development_gates_v1",
        "status": "DEVELOPMENT_PASS" if gates["all_pass"] else "DEVELOPMENT_FAILED",
        "candidate_id": "M1_requested_effect_gate",
        **gates,
        "reference_unresolved": len(unresolved),
        "root_families": len({row["root_family_id"] for row in metadata}),
        "physical_rollouts": len(metadata),
    }
    write_json(output_root / "development_gates.json", gate_record)
    return gate_record


def replay_development(inputs_path: Path, contract_root: Path, methods: list[str], output_root: Path) -> dict[str, Any]:
    del inputs_path
    if tuple(methods) != METHODS:
        raise ValueError("cache pilot is frozen to D0 and M1")
    if read_json(contract_root / "contract_status.json")["status"] != "CONTRACT_LOCKED_FOR_NEW_COLLECTION":
        raise ValueError("contract is not frozen")
    from .contract import ControllerRequest, RequestedEffect

    def obs(effect: RequestedEffect | None, *, end: bool, held_loss: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows = [
            {"frame_index": 0, "capture_order": 0, "time": 0.0, "attempt_id": 1, "attempt_phase": "acquiring", "attempt_active": True, "attempt_end": False, "attempt_end_reason": None,
             "predicates": {"contact_present": "true" if held_loss else "false", "gripper_command_closed": "true", "gripper_command_open": "false", "contact_recently_lost": "false"}},
            {"frame_index": 1, "capture_order": 1, "time": 0.1, "attempt_id": 1, "attempt_phase": "ended" if end else "acquiring", "attempt_active": not end, "attempt_end": end, "attempt_end_reason": "segment_complete" if end else None, "attempt_end_sequence": 1,
             "predicates": {"contact_present": "false", "gripper_command_closed": "true", "gripper_command_open": "false", "contact_recently_lost": "true" if held_loss else "false"}},
        ]
        evidence = [{"hold_evidence": "true" if held_loss else "false", "hold_memory": "true" if held_loss else "false"}, {"hold_evidence": "false", "hold_memory": "false"}]
        requests = [] if effect is None else [ControllerRequest("pilot", 1, "tracked_object_001", effect, 0.0, -1, 0.0).as_dict()]
        return rows, evidence, requests

    fixtures = [
        ("hold_end_no_hold", RequestedEffect.HOLD_OBJECT, True, False, "retry_grasp"),
        ("touch_end_no_hold", RequestedEffect.TOUCH_OBJECT, True, False, "none"),
        ("active_transient", RequestedEffect.HOLD_OBJECT, False, False, "none"),
        ("missing_context", None, True, False, "needs_observation"),
        ("held_loss_non_hold", RequestedEffect.TOUCH_OBJECT, False, True, "recover_object"),
    ]
    rows = []
    for fixture_id, effect, end, held_loss, expected in fixtures:
        observations, evidence, requests = obs(effect, end=end, held_loss=held_loss)
        for method in methods:
            decisions = _method_rows(method, observations, evidence, requests, True)
            rows.append({
                "fixture_id": fixture_id, "method": method,
                "requested_effect": effect.value if effect else "MISSING",
                "expected_m1_action": expected,
                "selected_action": decisions[-1]["selected_action"],
                "m1_contract_match": decisions[-1]["selected_action"] == expected if method.startswith("M1") else "NOT_APPLICABLE",
                "physical_rollout_credit": 0,
            })
    write_csv(output_root / "cache_and_logic_pilot.csv", rows)
    m1 = [row for row in rows if row["method"].startswith("M1")]
    result = {
        "schema": "pathgraph_l2rar2_cache_pilot_v1",
        "status": "CACHE_MECHANISM_AND_LOGIC_PASS" if all(row["m1_contract_match"] for row in m1) else "CACHE_MECHANISM_FAILED",
        "historical_context_role": "missing; no retrospective M1 scientific score",
        "logic_fixtures": len(m1),
        "new_physical_rollouts": 0,
    }
    write_json(output_root / "pilot_status.json", result)
    return result


def lock_candidate(development_root: Path, protocol_path: Path, generation_lock_path: Path, output_path: Path) -> dict[str, Any]:
    gates = read_json(development_root / "development_gates.json")
    if gates.get("status") != "DEVELOPMENT_PASS" or not gates.get("all_pass"):
        raise RuntimeError("development gates did not pass; candidate cannot be locked")
    lock = {
        "schema": "pathgraph_l2rar2_selection_lock_v1",
        "status": "LOCKED_FOR_FOCUSED_CONFIRMATION",
        "selected_candidate_id": "M1_requested_effect_gate",
        "baseline_id": "D0_generic_attempt_end",
        "development_gates_sha256": sha256(development_root / "development_gates.json"),
        "development_event_decisions_sha256": sha256(development_root / "event_decisions.csv"),
        "protocol_sha256": sha256(protocol_path),
        "generation_lock_sha256": sha256(generation_lock_path),
        "candidate_modified_after_select": False,
        "standard_l2r_confirmation_status": "NOT_RUN_OUT_OF_SCOPE",
    }
    write_json(output_path, lock)
    return lock


def paired_bootstrap(decisions: list[dict[str, Any]], *, resamples: int, seed: int) -> list[dict[str, Any]]:
    families = sorted({row["root_family_id"] for row in decisions})
    rng = random.Random(seed)
    output = []
    for case in POSITIVE_CASES + NEGATIVE_CASES:
        by_family: dict[str, dict[str, float]] = defaultdict(dict)
        for row in decisions:
            if row["case_id"] == case:
                by_family[row["root_family_id"]][row["method"]] = float(bool(row["correct"]))
        effects = [values.get("M1_requested_effect_gate", 0.0) - values.get("D0_generic_attempt_end", 0.0) for values in by_family.values()]
        samples = []
        for _ in range(resamples):
            drawn = [effects[rng.randrange(len(effects))] for _ in effects]
            samples.append(sum(drawn) / len(drawn))
        samples.sort()
        output.append({
            "case_id": case, "families": len(effects), "metric": "paired_correct_rate_M1_minus_D0",
            "point_estimate": sum(effects) / len(effects),
            "ci95_low": samples[int(0.025 * resamples)],
            "ci95_high": samples[min(resamples - 1, int(0.975 * resamples))],
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        })
    return output
