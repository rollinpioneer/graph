from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_geometry_events.state_machine import GeometryEventStateMachine
from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _predicates
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_task_context.online_interface_repair import FIXED_CANDIDATES, run_repaired_interface

from .audits import parameter_audit, rgb_provenance
from .candidate_inputs import ALLOWED, FORBIDDEN, load_candidate_input
from .io_utils import read_json, read_jsonl, write_csv, write_json
from .protocol import METHODS, O_CANDIDATES, PARAMETERS, RECOVERY_DEADLINE_S, RETRY_DEADLINE_S
from .temporal_scoring import score_temporal


def _actions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, previous = [], None
    for row in records:
        action = row.get("selected_action")
        if action in {"retry_grasp", "recover_object"} and action != previous:
            rows.append({"action": action, "time": float(row["time"]),
                         "capture_order": row.get("capture_order"), "reason_code": row.get("reason_code")})
        previous = action
    return rows


def _run_o(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    observations = []
    for row in rows:
        observation = dict(row)
        observation["frame_index"] = row.get("capture_order")
        observations.append(observation)
    geometry = build_features(observations)
    predictions, previous = [], None
    for observation, feature in zip(observations, geometry):
        predictions.append({**observation, "predicates": _predicates(observation, feature, previous)})
        previous = observation
    candidate_id = O_CANDIDATES[method]
    base, config = FIXED_CANDIDATES[candidate_id]
    evidence = evaluate_candidate(predictions, geometry, base, config)
    first = rows[0]
    requests = [{"request_id": "request_1", "attempt_id": int(first.get("attempt_id") or 1),
                 "target_track_id": "object", "requested_effect": first.get("requested_effect", "UNKNOWN"),
                 "issued_time": float(first["time"]), "issued_capture_order": int(first["capture_order"]),
                 "received_time": float(first["time"]), "source": "controller_dispatch"}]
    return run_repaired_interface(predictions, evidence, requests, history_complete=True)


def _nearest_reference(physics: list[dict[str, Any]], time_value: float) -> dict[str, Any] | None:
    if not physics: return None
    return min(physics, key=lambda row: abs(float(row["time"]) - time_value))


def _run_s(rows: list[dict[str, Any]], physics: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        ref = _nearest_reference(physics, float(row["time"]))
        valid = ref is not None and ref.get("object_world_position") is not None and ref.get("gripper_world_position") is not None
        samples.append({
            "time": row["time"], "capture_order": row["capture_order"],
            "contact_present": row.get("contact_present"), "gripper_command": row.get("gripper_command"),
            "attempt_id": row.get("attempt_id"), "attempt_phase": row.get("attempt_phase"),
            "attempt_active": row.get("attempt_active"), "attempt_end": row.get("attempt_end"),
            "attempt_end_reason": row.get("attempt_end_reason"),
            "requested_effect": row.get("requested_effect"), "context_valid": row.get("context_valid"),
            "object_xyz": ref.get("object_world_position") if valid else None,
            "gripper_xyz": ref.get("gripper_world_position") if valid else None,
            "data_quality": "VALID" if valid else "MISSING_OR_INVALID",
            "data_quality_reason": None if valid else "reference_position_unavailable",
        })
    return GeometryEventStateMachine(method, {"parameters": PARAMETERS}).run(samples)


def _run(method: str, rows: list[dict[str, Any]], physics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    return _run_o(rows, method) if method.startswith("O_") else _run_s(rows, physics or [], method)


def _projection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("selected_action", "reason_code", "hold_state")
    return [{key: row.get(key) for key in fields} for row in records]


def _percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def evaluate(confirmation_root: Path, reference_root: Path, output_root: Path) -> dict[str, Any]:
    gate = read_json(reference_root / "generator_gate.json")
    if not gate.get("candidate_evaluation_allowed"):
        raise RuntimeError("GENERATOR_GATE_REQUIRED")
    output_root.mkdir(parents=True, exist_ok=False)

    # Phase 1: run O-tier predictions from candidate_input only. Labels are not loaded yet.
    online_by_rollout: dict[str, list[dict[str, Any]]] = {}
    o_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metadata in sorted(confirmation_root.glob("*/metadata.json")):
        rollout_id = metadata.parent.name
        rows = load_candidate_input(metadata.parent / "candidate_input")
        online_by_rollout[rollout_id] = rows
        for method in ("O_B2", "O_C3"):
            o_records[(method, rollout_id)] = _run_o(rows, method)

    # Phase 2: attach physical labels and run explicitly state-assisted diagnostics.
    references = read_json(reference_root / "physical_reference_index.json")["rows"]
    decisions: list[dict[str, Any]] = []
    prefix = {"schema": "l2rar2_r17_prefix_causality_audit_v1", "passed": True,
              "checked": 0, "mismatches": []}
    provenance_methods = {}
    for method in METHODS:
        provenance_methods[method] = {
            "input_tier": "ONLINE_TRUE_RGB" if method.startswith("O_") else "STATE_ASSISTED_DIAGNOSTIC",
            "eligible_for_online_selection": method.startswith("O_"),
            "files_read": ["candidate_input/detections_rgb.jsonl", "candidate_input/contact_proxy.jsonl",
                           "candidate_input/attempt_lifecycle.jsonl", "candidate_input/gripper_commands.jsonl",
                           "candidate_input/request_context.jsonl"] + ([] if method.startswith("O_") else ["reference/physics_trace.jsonl"]),
            "fields_read": sorted(ALLOWED) if method.startswith("O_") else ["object_world_position", "gripper_world_position"],
            "forbidden_online_reads": 0,
        }
        for reference in references:
            rollout_id = reference["rollout_id"]
            rows = online_by_rollout[rollout_id]
            physics = read_jsonl((reference_root / reference["physics_trace_path"]).resolve())
            records = o_records[(method, rollout_id)] if method.startswith("O_") else _run_s(rows, physics, method)
            actions = _actions(records)
            truth = reference.get("reference_action", "none")
            rollout_end = float(rows[-1]["time"])
            if truth == "retry_grasp":
                onset_row = next((row for row in rows if row.get("attempt_end") is True), None)
                onset = float(onset_row["time"]) if onset_row else None
                deadline = None if onset is None else min(onset + RETRY_DEADLINE_S, rollout_end)
            elif truth == "recover_object":
                onset = reference.get("loss_onset_time_abs")
                onset = float(onset) if onset is not None else None
                deadline = None if onset is None else min(onset + RECOVERY_DEADLINE_S, rollout_end)
            else:
                onset = deadline = None
            scored = score_temporal(truth_action=truth, event_onset=onset, deadline=deadline,
                                    actions=actions, resolvable=bool(reference.get("resolvable")))
            first_post = next((float(row["time"]) for row in rows if onset is not None and float(row["time"]) >= onset - 1e-9), None)
            observable = None
            if onset is not None:
                previous_contact = None
                for row in rows:
                    contact = row.get("contact_present")
                    if previous_contact is True and contact is False:
                        observable = float(row["time"]); break
                    previous_contact = contact
                if observable is None: observable = first_post
            decision = {
                "method": method, "input_tier": provenance_methods[method]["input_tier"],
                "eligible_for_online_selection": method.startswith("O_"),
                "rollout_id": rollout_id, "family_id": reference["family_id"],
                "case_id": reference["case_id"], "truth_action": truth,
                "reference_state": reference.get("state"), "reference_resolved": bool(reference.get("resolvable")),
                "physical_loss_onset": onset, "decision_deadline": deadline,
                **{key: value for key, value in scored.items() if key != "all_actions"},
                "actions_json": json.dumps(actions, sort_keys=True),
                "unknown": bool(not actions and records and records[-1].get("selected_action") == "needs_observation"),
                "t_first_online_capture_after_onset": first_post,
                "t_method_first_observable": observable,
                "latency_from_physical_onset": None if scored["primary_action_time"] is None or onset is None else scored["primary_action_time"] - onset,
                "latency_from_first_post_onset_capture": None if scored["primary_action_time"] is None or first_post is None else scored["primary_action_time"] - first_post,
                "latency_from_method_observable": None if scored["primary_action_time"] is None or observable is None else scored["primary_action_time"] - observable,
                "time_provenance": "physical reference row.time joined after prediction; online detector row.time drives methods",
            }
            decisions.append(decision)

            for fraction in (0.25, 0.50, 0.75, 1.0):
                cut = max(1, math.ceil(len(rows) * fraction))
                rerun = _run(method, rows[:cut], physics)
                prefix["checked"] += 1
                if _projection(rerun) != _projection(records[:cut]):
                    prefix["passed"] = False
                    prefix["mismatches"].append({"method": method, "rollout_id": rollout_id,
                                                 "fraction": fraction, "cut": cut})

    metrics = []
    for method in METHODS:
        items = [row for row in decisions if row["method"] == method]
        resolved = [row for row in items if row["reference_resolved"]]
        counts = defaultdict(int)
        by_case: dict[str, dict[str, Any]] = {}
        for row in items: counts[row["temporal_outcome"]] += 1
        for case in sorted({row["case_id"].split("_", 1)[0] for row in items}, key=lambda x: int(x[1:])):
            case_rows = [row for row in items if row["case_id"].startswith(case + "_")]
            by_case[case] = {"total": len(case_rows), "correct": sum(bool(row["accurate"]) for row in case_rows),
                             "accuracy": sum(bool(row["accurate"]) for row in case_rows) / len(case_rows),
                             "outcomes": dict((name, sum(row["temporal_outcome"] == name for row in case_rows))
                                              for name in sorted(set(row["temporal_outcome"] for row in case_rows)))}
        latencies = [float(row["latency_from_physical_onset"]) for row in items
                     if row["latency_from_physical_onset"] is not None and row["temporal_outcome"] == "CORRECT_IN_WINDOW"]
        strong = [row for row in items if row["case_id"].startswith(("C8_", "C9_", "C10_", "C11_"))]
        false_group = [row for row in items if row["case_id"].startswith(("C2_", "C3_", "C4_", "C12_"))]
        metric = {
            "method": method, "input_tier": provenance_methods[method]["input_tier"],
            "eligible_for_online_selection": method.startswith("O_"), "total": len(items),
            "resolved": len(resolved), "correct": sum(bool(row["accurate"]) for row in resolved),
            "overall_temporal_accuracy": sum(bool(row["accurate"]) for row in resolved) / len(resolved),
            "unknown_rate": sum(bool(row["unknown"]) for row in items) / len(items),
            "early_action_rate": counts["EARLY_ACTION"] / len(items), "late_action_rate": counts["LATE_ACTION"] / len(items),
            "miss_rate": counts["MISSED_REQUIRED_ACTION"] / len(items),
            "false_retry_rate": counts["FALSE_RETRY"] / len(items),
            "false_recovery_rate": counts["FALSE_RECOVERY"] / len(items),
            "C1_retry_correct": sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in items if row["case_id"].startswith("C1_")),
            "strong_loss_correct_in_window": sum(row["temporal_outcome"] == "CORRECT_IN_WINDOW" for row in strong),
            "false_actions_C2_C3_C4_C12": sum(row["primary_action"] is not None for row in false_group),
            "early_recovery_before_physical_onset": sum(row["temporal_outcome"] == "EARLY_ACTION" and row["primary_action"] == "recover_object" for row in items),
            "latency_median_s": statistics.median(latencies) if latencies else None,
            "latency_p90_s": _percentile(latencies, 0.90), "case_metrics": by_case,
            "outcome_counts": dict(counts),
        }
        metric["confirmation_status"] = "CONFIRMATION_PASS" if (
            method.startswith("O_") and metric["C1_retry_correct"] == 6
            and metric["false_actions_C2_C3_C4_C12"] == 0
            and metric["early_recovery_before_physical_onset"] == 0
            and metric["strong_loss_correct_in_window"] >= 22
            and metric["overall_temporal_accuracy"] >= 0.90
            and metric["unknown_rate"] <= 0.05) else (
                "CONFIRMATION_FAIL" if method.startswith("O_") else "DIAGNOSTIC_ONLY")
        metrics.append(metric)

    input_provenance = {"schema": "l2rar2_r17_input_provenance_audit_v1", "passed": True,
                        "reference_join_after_prediction": True, "methods": provenance_methods,
                        "o_tier_forbidden_fields": sorted(FORBIDDEN)}
    rgb_audit = rgb_provenance(confirmation_root)
    parameters = parameter_audit()
    # Prefix checks are part of every online pass condition.
    if prefix["checked"] < 1440: prefix["passed"] = False
    o_metrics = [row for row in metrics if row["method"].startswith("O_")]
    ranking_key = lambda row: (row["false_actions_C2_C3_C4_C12"], row["early_recovery_before_physical_onset"],
                               -row["strong_loss_correct_in_window"], -row["overall_temporal_accuracy"],
                               float("inf") if row["latency_p90_s"] is None else row["latency_p90_s"], row["unknown_rate"])
    sorted_o = sorted(o_metrics, key=ranking_key)
    tie = len(sorted_o) == 2 and ranking_key(sorted_o[0]) == ranking_key(sorted_o[1])
    ranking = {"schema": "l2rar2_r17_candidate_ranking_v1", "selected_candidate_id": None,
               "comparison": "TIE" if tie else "RANKED_WITHOUT_SELECTION",
               "candidates": sorted_o, "diagnostics": [row for row in metrics if row["method"].startswith("S_")]}
    payload = {"schema": "l2rar2_r17_method_metrics_v1", "methods": metrics,
               "selected_candidate_id": None, "input_provenance": input_provenance,
               "prefix_causality": prefix, "parameter_audit": parameters}
    flat_fields = ("method", "input_tier", "eligible_for_online_selection", "total", "resolved", "correct",
                   "overall_temporal_accuracy", "unknown_rate", "early_action_rate", "late_action_rate",
                   "miss_rate", "false_retry_rate", "false_recovery_rate", "C1_retry_correct",
                   "strong_loss_correct_in_window", "false_actions_C2_C3_C4_C12",
                   "early_recovery_before_physical_onset", "latency_median_s", "latency_p90_s",
                   "confirmation_status")
    write_csv(output_root / "per_event_decisions.csv", decisions)
    write_csv(output_root / "temporal_outcomes.csv", decisions)
    write_csv(output_root / "method_metrics.csv", metrics, flat_fields)
    write_json(output_root / "method_metrics.json", payload)
    write_csv(output_root / "latency_summary.csv", metrics,
              ("method", "latency_median_s", "latency_p90_s", "strong_loss_correct_in_window"))
    write_json(output_root / "input_provenance_audit.json", input_provenance)
    write_json(output_root / "rgb_provenance_audit.json", rgb_audit)
    write_json(output_root / "prefix_causality_audit.json", prefix)
    write_json(output_root / "parameter_audit.json", parameters)
    write_json(output_root / "candidate_ranking.json", ranking)
    recommendation = "TIE; no candidate selected." if tie else "Ranking recorded; no candidate selected automatically."
    (output_root / "candidate_recommendation.md").write_text("# Candidate recommendation\n\n" + recommendation + "\n", encoding="utf-8")
    return payload
