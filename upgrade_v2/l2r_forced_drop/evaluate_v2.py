from __future__ import annotations

import csv
import json
from pathlib import Path

from upgrade_v2.l2r_geometry_events.state_machine import GeometryEventStateMachine
from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _predicates
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_task_context.online_interface_repair import (
    FIXED_CANDIDATES,
    run_repaired_interface,
)

METHODS = ("O_B2", "O_C3", "S_G_H", "S_G_R", "S_G_HR")
O_CANDIDATES = {"O_B2": "B_count2", "O_C3": "C3_vector_rho035"}
S_PARAMETERS = {
    "direction_cosine_min": 0.8,
    "evidence_duration_s": 0.05,
    "height_off_m": 0.01,
    "height_on_m": 0.02,
    "max_gap_s": 0.25,
    "min_motion_3d_m": 0.004,
    "relative_hold_drift_m": 0.01,
    "relative_loss_drift_m": 0.02,
    "relative_rho_max": 0.35,
}
FORBIDDEN_ONLINE = {
    "case_id", "family_id", "physical_loss_confirmed", "reference_action",
    "truth", "selected_level", "weld_active", "xfrc_applied",
    "pre_hold_verified", "numeric_health",
}


def _case_family(row):
    return str(row.get("case_id", "")).split("_", 1)[0]


def _truth_action(reference: dict) -> str:
    family = _case_family(reference)
    if family == "F1":
        return "retry_grasp"
    if family in {"F5", "F6", "F8"} and reference.get("physical_loss_confirmed"):
        return "recover_object"
    return "none"


def _load_online(reference_root: Path, reference: dict) -> list[dict]:
    path = (reference_root / reference["online_trace_path"]).resolve()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _o_inputs(rows: list[dict]):
    observations = []
    for index, row in enumerate(rows):
        obj = row.get("object_world_position")
        grip = row.get("gripper_world_position")
        observations.append({
            "time": row.get("time"), "capture_order": index, "frame_index": index,
            "object_centroid": None if obj is None else [320.0 + 1000.0 * obj[0], 240.0 - 1000.0 * obj[2]],
            "gripper_centroid": None if grip is None else [320.0 + 1000.0 * grip[0], 240.0 - 1000.0 * grip[2]],
            "object_confidence": 1.0, "gripper_confidence": 1.0,
            "width": 640, "height": 480,
            "contact_present": row.get("contact_present"),
            "gripper_command": row.get("gripper_command"),
            "attempt_id": row.get("attempt_id"), "attempt_phase": row.get("attempt_phase"),
            "attempt_active": row.get("attempt_active"), "attempt_end": row.get("attempt_end"),
            "attempt_end_reason": row.get("attempt_end_reason"), "attempt_end_sequence": 1,
        })
    geometry = build_features(observations)
    predictions, previous = [], None
    for observation, feature in zip(observations, geometry):
        predictions.append({**observation, "predicates": _predicates(observation, feature, previous)})
        previous = observation
    return predictions, geometry


def _run_o(rows: list[dict], method: str):
    predictions, geometry = _o_inputs(rows)
    candidate_id = O_CANDIDATES[method]
    base, config = FIXED_CANDIDATES[candidate_id]
    evidence = evaluate_candidate(predictions, geometry, base, config)
    first = rows[0] if rows else {}
    requests = [{"request_id": "request_1", "attempt_id": 1,
                 "target_track_id": "object", "requested_effect": first.get("requested_effect", "UNKNOWN"),
                 "issued_time": float(first.get("time", 0.0)), "issued_capture_order": 0,
                 "received_time": float(first.get("time", 0.0)), "source": "controller_dispatch"}]
    records = run_repaired_interface(predictions, evidence, requests, history_complete=True)
    selected = next((r for r in records if r.get("selected_action") in {"retry_grasp", "recover_object"}), None)
    return (selected.get("selected_action") if selected else "none",
            selected.get("time") if selected else None, records)


def _s_inputs(rows: list[dict]):
    samples = []
    for index, row in enumerate(rows):
        samples.append({
            "time": row.get("time"), "capture_order": index,
            "contact_present": row.get("contact_present"),
            "gripper_command": row.get("gripper_command"),
            "attempt_id": row.get("attempt_id"), "attempt_phase": row.get("attempt_phase"),
            "attempt_active": row.get("attempt_active"), "attempt_end": row.get("attempt_end"),
            "attempt_end_reason": row.get("attempt_end_reason"),
            "object_xyz": row.get("object_world_position"),
            "gripper_xyz": row.get("gripper_world_position"),
            "data_quality": "VALID" if row.get("object_world_position") is not None else "MISSING_OR_INVALID",
            "data_quality_reason": None,
            "requested_effect": row.get("requested_effect"), "context_valid": row.get("context_valid", False),
        })
    return samples


def _run_s(rows: list[dict], method: str):
    machine = GeometryEventStateMachine(method, {"parameters": S_PARAMETERS})
    records = machine.run(_s_inputs(rows))
    selected = next((r for r in records if r.get("selected_action") in {"retry_grasp", "recover_object"}), None)
    return (selected.get("selected_action") if selected else "none",
            selected.get("time") if selected else None, records)


def _run_method(rows: list[dict], method: str):
    return _run_o(rows, method) if method.startswith("O_") else _run_s(rows, method)


def evaluate_v2(reference_root: Path, output_root: Path):
    gate = json.loads((reference_root / "generator_gate.json").read_text(encoding="utf-8"))
    if not gate.get("candidate_evaluation_allowed"):
        raise RuntimeError("GENERATOR_GATE_REQUIRED")
    output_root.mkdir(parents=True, exist_ok=False)
    references = json.loads((reference_root / "physical_reference_index.json").read_text(encoding="utf-8"))["rows"]
    decisions, metrics, time_rows = [], [], []
    leakage = {"schema": "l2rar2_input_leakage_audit_v2", "passed": True, "violations": []}
    causality = {"schema": "l2rar2_prefix_causality_audit_v2", "passed": True, "checked": 0, "mismatches": []}
    for method in METHODS:
        correct = unknown = 0
        per_case = {f"F{i}": {"correct": 0, "total": 0, "unknown": 0} for i in range(1, 9)}
        for reference in references:
            online_rows = _load_online(reference_root, reference)
            for index, row in enumerate(online_rows):
                leaked = sorted(FORBIDDEN_ONLINE.intersection(row))
                if leaked:
                    leakage["passed"] = False
                    leakage["violations"].append({"case_id": reference["case_id"], "index": index, "keys": leaked})
            truth = _truth_action(reference)
            prediction, selected_time, full_records = _run_method(online_rows, method)
            is_correct = prediction == truth
            correct += int(is_correct); unknown += int(prediction == "needs_observation")
            family_case = _case_family(reference)
            per_case[family_case]["total"] += 1
            per_case[family_case]["correct"] += int(is_correct)
            per_case[family_case]["unknown"] += int(prediction == "needs_observation")
            force_start = reference.get("force_start_time")
            observable_delay = None if selected_time is None or force_start is None else float(selected_time) - float(force_start)
            decisions.append({"method": method, "family_id": reference.get("family_id"),
                              "case_id": reference.get("case_id"), "decision": prediction,
                              "truth": truth, "correct": is_correct,
                              "selected_time_abs": selected_time,
                              "observable_latency_from_force_s": observable_delay,
                              "physical_latency_from_force_s": reference.get("loss_confirmed_delay_from_force_s")})
            time_rows.append({"method": method, "family_id": reference.get("family_id"),
                              "case_id": reference.get("case_id"), "selected_time_abs": selected_time,
                              "observable_latency_from_force_s": observable_delay,
                              "physical_latency_from_force_s": reference.get("loss_confirmed_delay_from_force_s"),
                              "time_provenance": "online row.time / reference force_start_time"})
            for cut in sorted({1, max(1, len(online_rows) // 2), len(online_rows)}):
                _, _, prefix_records = _run_method(online_rows[:cut], method)
                fields = ("selected_action", "reason_code", "hold_state")
                prefix_projection = [{k: r.get(k) for k in fields} for r in prefix_records]
                full_projection = [{k: r.get(k) for k in fields} for r in full_records[:cut]]
                causality["checked"] += 1
                if prefix_projection != full_projection:
                    causality["passed"] = False
                    causality["mismatches"].append({"method": method, "case_id": reference["case_id"], "cut": cut})
        total = len(references)
        case_metrics = {case: {"accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
                               "unknown_rate": value["unknown"] / value["total"] if value["total"] else 0.0,
                               **value} for case, value in per_case.items()}
        metrics.append({"method": method, "overall_accuracy": correct / total if total else 0.0,
                        "unknown_rate": unknown / total if total else 0.0,
                        "correct": correct, "total": total, "F1_F8_metrics": case_metrics})
    payload = {"schema": "l2rar2_r16_method_metrics_v2", "methods": metrics,
               "input_leakage": leakage, "prefix_causality": causality,
               "time_provenance": "online row.time; reference labels never enter candidates"}
    with (output_root / "per_event_decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = tuple(decisions[0]) if decisions else ("method",)
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(decisions)
    with (output_root / "method_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("method", "overall_accuracy", "unknown_rate", "correct", "total")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in metrics)
    with (output_root / "time_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = tuple(time_rows[0]) if time_rows else ("method",)
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(time_rows)
    (output_root / "method_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_root / "input_leakage_audit.json").write_text(json.dumps(leakage, indent=2) + "\n")
    (output_root / "prefix_causality_audit.json").write_text(json.dumps(causality, indent=2) + "\n")
    evaluated_status = "EVALUATED"
    ranking = [{"candidate_id": row["method"], "status": evaluated_status, "method_metrics": row} for row in metrics]
    (output_root / "candidate_ranking.json").write_text(json.dumps({"selected_candidate_id": None, "candidates": ranking}, indent=2) + "\n")
    (output_root / "candidate_recommendation.md").write_text("# Candidate recommendation\n\nNo candidate is selected automatically.\n")
    return ranking
