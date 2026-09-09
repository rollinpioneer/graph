"""Evaluate bounded event-semantic candidates and frozen-G2 compatibility."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.execute_graph import predict_branch, expected_branch, graph_coverage
from upgrade_v2.visual_refine_l2.io import read_csv, read_json, read_jsonl

from .event_memory import c1_effective_guard, run_memory


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _observations(item: dict[str, Any]) -> list[dict[str, Any]]:
    return item["observations"]


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _actions_from_guards(guards: list[dict[str, str]]) -> tuple[str, int | None, bool]:
    retry = [i for i, row in enumerate(guards) if row.get("retry_grasp") == "true"]
    recover = [i for i, row in enumerate(guards) if row.get("recover_object") == "true"]
    if not retry and not recover:
        return "none", None, False
    last = max((retry[-1] if retry else -1), (recover[-1] if recover else -1))
    both = retry and recover and retry[-1] == recover[-1]
    if recover and recover[-1] == last:
        return "recover_object", last, bool(both)
    return "retry_grasp", last, bool(both)


def candidate_sequence(item: dict[str, Any], candidate_id: str, hold_confirm: int = 1, loss_confirm: int = 1) -> dict[str, Any]:
    observations = _observations(item)
    if candidate_id == "D_priority_only":
        memory_rows = None
        guards = [{"retry_grasp": row["predicates"].get("grasp_failed_observed", "unknown"),
                   "recover_object": row["predicates"].get("slip_observed", "unknown")}
                  for row in observations]
    elif candidate_id == "C1_current_event_exclusion":
        memory_rows = None
        guards = [c1_effective_guard(row["predicates"]) for row in observations]
    elif candidate_id.startswith("C2_attempt_scoped_event_memory"):
        memory_rows = run_memory(observations, hold_confirm, loss_confirm, bool(item.get("reference", {}).get("history_complete", True)))
        guards = [row["effective_guards"] for row in memory_rows]
    else:
        memory_rows = None
        guards = []
    action, action_frame, conflict = _actions_from_guards(guards) if guards else ("none", None, False)
    needs = any(row.get("semantic_events", {}).get("needs_observation") == "true" for row in (memory_rows or []))
    if action == "none" and needs:
        action = "needs_observation"
    return {"candidate_id": candidate_id, "guards": guards, "memory_rows": memory_rows,
            "selected_action": action, "selected_frame": action_frame, "effective_conflict": conflict,
            "any_conflict": any(row.get("retry_grasp") == "true" and row.get("recover_object") == "true" for row in guards)}


def _event_metrics(items: list[dict[str, Any]], candidate_id: str, hold_confirm: int, loss_confirm: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records, delays = [], []
    for item in items:
        sequence = candidate_sequence(item, candidate_id, hold_confirm, loss_confirm)
        reference = item["reference"]
        expected = reference["action_class"]
        selected = sequence["selected_action"]
        if expected == "release":
            expected = "none"
        if expected == "none":
            correct = selected == "none"
        else:
            correct = selected == expected
        physical_event = reference["event_type"] in {"missed_grasp_retry_required", "held_object_loss_recovery_required"}
        decidable_physical = physical_event and reference.get("label_status") == "reference_labeled" and bool(reference.get("observable_at_decision", False))
        negative = reference["event_type"] in {"release_expected", "needs_observation"}
        records.append({"probe_id": item["probe_id"], "root_family_id": item["root_family_id"], "stratum": item["stratum"],
                        "reference_event_type": reference["event_type"], "reference_action_class": expected,
                        "selected_action": selected, "selected_frame": sequence["selected_frame"], "correct": correct,
                        "effective_conflict": sequence["effective_conflict"], "any_conflict": sequence["any_conflict"],
                        "label_status": reference["label_status"], "physical_event": physical_event,
                        "decidable_physical_event": decidable_physical, "negative_opportunity": negative})
        if decidable_physical:
            decision_frame = reference.get("decision_frame_index")
            expected_guard = "retry_grasp" if reference["event_type"] == "missed_grasp_retry_required" else "recover_object"
            first_correct_frame = next((index for index, guard in enumerate(sequence["guards"]) if guard.get(expected_guard) == "true"), None)
            delay = None if not correct or decision_frame is None or first_correct_frame is None else max(0, first_correct_frame - decision_frame)
            delays.append({"probe_id": item["probe_id"], "root_family_id": item["root_family_id"], "stratum": item["stratum"],
                           "decision_frame_index": decision_frame, "first_correct_frame": first_correct_frame if correct else None,
                           "delay_observation_steps": delay, "delay_seconds": (float(item["observations"][first_correct_frame]["time"]) - float(item["observations"][decision_frame]["time"])) if delay is not None else None,
                           "detected": correct})
    emergency = [row for row in records if row["decidable_physical_event"]]
    physical = [row for row in records if row["physical_event"]]
    undecidable_physical = [row for row in physical if not row["decidable_physical_event"]]
    miss = [row for row in emergency if row["reference_event_type"] == "missed_grasp_retry_required"]
    loss = [row for row in emergency if row["reference_event_type"] == "held_object_loss_recovery_required"]
    negatives = [row for row in records if row["negative_opportunity"]]
    by_stratum = defaultdict(list)
    for row in records: by_stratum[row["stratum"]].append(row)
    false_emergency = []
    for stratum, rows in sorted(by_stratum.items()):
        if stratum in {"touch_without_hold_then_loss", "commanded_release"}:
            false_emergency.append({"candidate_id": candidate_id, "stratum": stratum, "opportunities": len(rows),
                                    "false_emergency": sum(row["selected_action"] in {"retry_grasp", "recover_object"} for row in rows),
                                    "false_emergency_rate": sum(row["selected_action"] in {"retry_grasp", "recover_object"} for row in rows) / len(rows) if rows else None})
    history_rows = [row for row in records if row["stratum"] == "history_or_visual_unavailable"]
    unjustified = sum(row["selected_action"] in {"retry_grasp", "recover_object"} for row in history_rows)
    detected_delays = [row for row in delays if row["delay_observation_steps"] is not None]
    delay_steps = [float(row["delay_observation_steps"]) for row in detected_delays]
    delay_seconds = [float(row["delay_seconds"]) for row in detected_delays]
    raw_overlap = sum(
        any(
            observation["predicates"].get("grasp_failed_observed") == "true"
            and observation["predicates"].get("slip_observed") == "true"
            for observation in item["observations"]
        )
        for item in items
    )
    metrics = {"candidate_id": candidate_id, "miss_events": len(miss), "miss_correct": sum(row["correct"] for row in miss),
               "miss_families": len({row["root_family_id"] for row in miss}),
               "miss_type_recall": sum(row["correct"] for row in miss) / len(miss) if miss else None,
               "loss_events": len(loss), "loss_correct": sum(row["correct"] for row in loss),
               "loss_families": len({row["root_family_id"] for row in loss}),
               "loss_type_recall": sum(row["correct"] for row in loss) / len(loss) if loss else None,
               "physical_event_count": len(emergency), "decidable_physical_event_count": len(emergency),
               "undecidable_physical_event_count": len(undecidable_physical),
               "wrong_or_unknown_rate": sum(not row["correct"] for row in emergency) / len(emergency) if emergency else None,
               "raw_rule_overlap_rollout_rate": raw_overlap / len(records) if records else None,
               "effective_conflict_rollout_rate": sum(row["any_conflict"] for row in records) / len(records) if records else None,
               "event_window_conflict_rate": sum(row["effective_conflict"] for row in emergency) / len(emergency) if emergency else None,
               "unjustified_definite_rate": unjustified / len(history_rows) if history_rows else None,
               "mean_delay_observation_steps": sum(delay_steps) / len(delay_steps) if delay_steps else None,
               "delay_observation_steps_p95": _quantile(delay_steps, .95),
               "mean_delay_seconds": sum(delay_seconds) / len(delay_seconds) if delay_seconds else None,
               "delay_seconds_p95": _quantile(delay_seconds, .95),
               "detected_events": sum(row["detected"] for row in delays), "delay_opportunities": len(delays),
               "undetected_events": sum(not row["detected"] for row in delays),
               "negative_opportunities": len(negatives),
               "false_emergency": sum(row["selected_action"] in {"retry_grasp", "recover_object"} for row in records if row["stratum"] in {"touch_without_hold_then_loss", "commanded_release"}),
               "unjustified_definite": unjustified}
    return metrics, false_emergency, delays


def _compatibility(resolved: dict[str, Any], graph_path: Path, candidate_id: str, hold_confirm: int, loss_confirm: int) -> dict[str, Any]:
    graph = read_json(graph_path)
    records = [row for row in resolved["rollouts"] if row["split"] == "dev_select"]
    rows = []
    for record in records:
        predictions = read_jsonl(Path(record["prediction"]["prediction_path"]))
        old_action, _ = predict_branch(graph, predictions, False)
        item = {"observations": predictions, "reference": {"history_complete": True}}
        sequence = candidate_sequence(item, candidate_id, hold_confirm, loss_confirm)
        selected = sequence["selected_action"] if old_action in {"retry_grasp", "recover_object"} and sequence["selected_action"] in {"retry_grasp", "recover_object"} else old_action
        expected = expected_branch(record["scenario_reference_only"])
        correct = selected == expected if expected != "observe_or_request_view" else selected in {"request_second_view", "request_clarification"}
        actions = [row["action"] for row in read_csv(Path(record["rollout_path"]) / "actions.csv")]
        rows.append({"root_family_id": record["root_family_id"], "rollout_id": record["rollout_id"], "selected": selected,
                     "expected": expected, "correct": correct, "old_action": old_action,
                     "coverage": graph_coverage(graph, predictions, actions)})
    return {"rollouts": len(rows), "families": len({row["root_family_id"] for row in rows}),
            "branch_accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None,
            "coverage": sum(row["coverage"] for row in rows) / len(rows) if rows else None,
            "rows": rows}


def evaluate_development(resolved: dict[str, Any], protocol: dict[str, Any], probe_root: Path, output_root: Path, frozen_graph: Path) -> dict[str, Any]:
    items = [json.loads(line) for line in (probe_root / "probe_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fit_items = [item for item in items if item["split"] == "dev_fit"]
    select_items = [item for item in items if item["split"] == "dev_select"]
    candidates = [("D_priority_only", 0, 0), ("C1_current_event_exclusion", 1, 1)]
    candidates.extend((f"C2_attempt_scoped_event_memory_h{h}_l{l}", h, l) for h in (1, 2) for l in (1, 2))
    metrics, fit_metrics, false_rows, delay_rows = [], [], [], []
    compatibility = []
    for candidate, hold, loss in candidates:
        fit_metric, _, _ = _event_metrics(fit_items, candidate, hold, loss)
        event_metrics, false_candidate, delay_candidate = _event_metrics(select_items, candidate, hold, loss)
        fit_metrics.append(fit_metric)
        metrics.append(event_metrics); false_rows.extend(false_candidate); delay_rows.extend(delay_candidate)
        compatibility.append({"candidate_id": candidate, **{key: value for key, value in _compatibility(resolved, frozen_graph, candidate, hold, loss).items() if key != "rows"}})
    _write_csv(output_root / "candidate_metrics.csv", metrics)
    _write_csv(output_root / "candidate_fit_metrics.csv", fit_metrics)
    _write_csv(output_root / "false_emergency_and_unknown.csv", false_rows)
    _write_csv(output_root / "decision_delay.csv", delay_rows)
    _write_csv(output_root / "legacy_dev_select_compatibility.csv", compatibility)
    compatibility_by_id = {row["candidate_id"]: row for row in compatibility}
    baseline = compatibility_by_id["D_priority_only"]
    thresholds = protocol["new_diagnostic_thresholds"]
    gate_rows = []
    eligible = []
    for row in metrics:
        if row["candidate_id"] == "D_priority_only": continue
        false_by = [item for item in false_rows if item["candidate_id"] == row["candidate_id"]]
        values_present = all(row[name] is not None for name in ("wrong_or_unknown_rate", "event_window_conflict_rate", "unjustified_definite_rate"))
        false_ok = bool(false_by) and all(item["false_emergency_rate"] is not None and item["false_emergency_rate"] <= thresholds["false_emergency_rate_max"] for item in false_by)
        compat = compatibility_by_id[row["candidate_id"]]
        checks = {
            "support_minimum": row["miss_events"] >= 8 and row["loss_events"] >= 8 and row["miss_families"] >= 2 and row["loss_families"] >= 2,
            "event_type_recall": row["miss_type_recall"] is not None and row["loss_type_recall"] is not None and row["miss_type_recall"] >= thresholds["event_type_recall_min"] and row["loss_type_recall"] >= thresholds["event_type_recall_min"],
            "wrong_or_unknown": values_present and row["wrong_or_unknown_rate"] <= thresholds["wrong_or_unknown_fraction_max"],
            "effective_conflict": values_present and row["event_window_conflict_rate"] <= thresholds["event_window_conflict_rate_max"],
            "negative_false_emergency": false_ok,
            "insufficient_history": values_present and row["unjustified_definite_rate"] <= thresholds["unjustified_definite_rate_max"],
            "legacy_branch_noninferiority": compat["branch_accuracy"] + 1e-12 >= baseline["branch_accuracy"] - .02,
            "legacy_coverage_noninferiority": compat["coverage"] + 1e-12 >= baseline["coverage"] - .02,
        }
        gate_rows.append({"candidate_id": row["candidate_id"], **checks, "all_pass": all(checks.values())})
        if all(checks.values()):
            eligible.append(row["candidate_id"])
    _write_csv(output_root / "candidate_gate_checks.csv", gate_rows)
    metrics_by_id = {row["candidate_id"]: row for row in metrics}
    selected = min(
        eligible,
        key=lambda candidate: (
            metrics_by_id[candidate]["wrong_or_unknown_rate"],
            metrics_by_id[candidate]["event_window_conflict_rate"],
            metrics_by_id[candidate]["mean_delay_observation_steps"] if metrics_by_id[candidate]["mean_delay_observation_steps"] is not None else float("inf"),
            0 if candidate == "C1_current_event_exclusion" else 2,
            candidate,
        ),
    ) if eligible else None
    decision = {"schema": "pathgraph_l2ra_development_decision_v1", "status": "DEVELOPMENT_READY" if selected else "DEVELOPMENT_NOT_READY",
                "selected_candidate_id": selected, "eligible_candidates": eligible, "registered_controls_and_candidates": len(candidates), "selectable_candidate_count": 5,
                "selection_rule": "fewest errors/unknown on decidable events, then fewer conflicts, lower delay, fewer parameters",
                "selection_uses_legacy_confirmation": False, "selection_uses_new_select": True,
                "fit_families": len({item["root_family_id"] for item in fit_items}), "fit_rollouts": len(fit_items),
                "select_families": len({item["root_family_id"] for item in select_items}), "select_rollouts": len(select_items),
                "note": "All probe rows are bounded replay/intervention records; intervention strata remain explicitly labeled."}
    registry = {"schema": "pathgraph_l2ra_candidate_registry_v1", "control": "D_priority_only",
                "selectable_candidates": [{"candidate_id": row["candidate_id"], "metrics": row,
                                            "compatibility": compatibility_by_id[row["candidate_id"]],
                                            "gate": next(item for item in gate_rows if item["candidate_id"] == row["candidate_id"])}
                                           for row in metrics if row["candidate_id"] != "D_priority_only"],
                "selected_candidate_id": selected, "legacy_confirmation_used_for_selection": False}
    _write_json(output_root / "candidate_registry.json", registry)
    _write_json(output_root / "development_route.json", decision)
    return {**decision, "metrics": metrics, "compatibility": compatibility}
