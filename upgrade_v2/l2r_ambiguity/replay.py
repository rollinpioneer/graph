"""Replay frozen L2R predicates and expose the hidden overlap timeline."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.execute_graph import _edge_truth, expected_branch, predict_branch
from upgrade_v2.visual_refine_l2.io import read_json, read_jsonl

from .inputs import read_csv
from .reference_events import extract_reference_events


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _bool_pred(frame: dict[str, Any], name: str) -> bool | None:
    value = frame.get("predicates", {}).get(name)
    return True if value == "true" else False if value == "false" else None


def _truths(graph: dict[str, Any], predictions: list[dict[str, Any]]) -> list[dict[str, str]]:
    return _edge_truth(graph, predictions)


def trace_rollout(record: dict[str, Any], graph: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = read_jsonl(Path(record["prediction"]["prediction_path"]))
    action_rows = read_csv(Path(record["rollout_path"]) / "actions.csv")
    if len(action_rows) != len(predictions):
        raise ValueError(f"action/prediction length mismatch: {record['rollout_id']}")
    truths = _truths(graph, predictions)
    selected, ambiguous = predict_branch(graph, predictions, False)
    retry_indices = [i for i, values in enumerate(truths) if values.get("retry_after_missed_grasp") == "true"]
    recover_indices = [i for i, values in enumerate(truths) if values.get("recover_after_contact_loss") == "true"]
    last_event = max(retry_indices + recover_indices) if retry_indices or recover_indices else None
    rows = []
    for index, frame in enumerate(predictions):
        values = truths[index]
        true_ids = [key for key, value in values.items() if value == "true"]
        unknown_ids = [key for key, value in values.items() if value == "unknown"]
        contact = _bool_pred(frame, "contact_present")
        closed = _bool_pred(frame, "gripper_command_closed")
        row = {
            "rollout_id": record["rollout_id"], "root_family_id": record["root_family_id"], "data_role": record["data_role"],
            "scenario_reference_only": record["scenario_reference_only"], "frame_index": index,
            "observation_time": frame.get("time"), "observation_interval_start": float(action_rows[index]["start_time"]),
            "observation_interval_end": float(action_rows[index]["end_time"]),
            "contact_present": contact, "gripper_command_closed": closed,
            "stable_hold_observed": _bool_pred(frame, "stable_hold_observed"),
            "grasp_failed_observed": _bool_pred(frame, "grasp_failed_observed"),
            "slip_observed": _bool_pred(frame, "slip_observed"),
            "recovery_observed": _bool_pred(frame, "recovery_observed"),
            "retry_raw_guard": values.get("retry_after_missed_grasp", "unknown"),
            "recover_raw_guard": values.get("recover_after_contact_loss", "unknown"),
            "raw_true_edge_ids": true_ids, "raw_unknown_edge_ids": unknown_ids,
            "last_event_action_index_used_by_legacy": last_event,
            "selected_branch_legacy": selected, "legacy_ambiguous": ambiguous,
            "retry_recover_both_true_at_this_time": values.get("retry_after_missed_grasp") == "true" and values.get("recover_after_contact_loss") == "true",
        }
        rows.append(row)
    summary = {
        "rollout_id": record["rollout_id"], "root_family_id": record["root_family_id"],
        "scenario": record["scenario_reference_only"], "data_role": record["data_role"],
        "selected_branch_legacy": selected, "legacy_ambiguous": ambiguous,
        "any_raw_overlap": any(row["retry_recover_both_true_at_this_time"] for row in rows),
        "overlap_frames": sum(row["retry_recover_both_true_at_this_time"] for row in rows),
        "valid_observation_frames": len(rows), "raw_retry_true_frames": sum(row["retry_raw_guard"] == "true" for row in rows),
        "raw_recover_true_frames": sum(row["recover_raw_guard"] == "true" for row in rows),
        "expected_branch": expected_branch(record["scenario_reference_only"]),
        "branch_correct": selected == expected_branch(record["scenario_reference_only"]) if expected_branch(record["scenario_reference_only"]) != "observe_or_request_view" else selected in {"request_second_view", "request_clarification"},
    }
    return rows, summary


def trace_split(resolved: dict[str, Any], split: str, graph_path: Path, output_root: Path) -> dict[str, Any]:
    graph = read_json(graph_path)
    resolved_split = {"legacy_dev_fit": "dev_fit", "legacy_dev_select": "dev_select"}.get(split, split)
    records = [row for row in resolved["rollouts"] if row["split"] == resolved_split]
    all_trace, summaries = [], []
    event_windows = []
    for record in records:
        trace, summary = trace_rollout(record, graph)
        all_trace.extend(trace); summaries.append(summary)
        for event in extract_reference_events(record):
            frame_index = event.get("frame_index")
            frame = next((row for row in trace if row["frame_index"] == frame_index), None)
            event_windows.append({"event_id": event["event_id"], "rollout_id": record["rollout_id"],
                                  "root_family_id": record["root_family_id"], "reference_event_type": event["reference_event_type"],
                                  "reference_label_status": event["reference_label_status"], "frame_index": frame_index,
                                  "raw_overlap_at_event": bool(frame and frame["retry_recover_both_true_at_this_time"])})
    trace_path = output_root / "guard_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_trace), encoding="utf-8")
    ambiguous = [row for row in summaries if row["legacy_ambiguous"]]
    _write_csv(output_root / "ambiguous_rollouts.csv", ambiguous,
               ["rollout_id", "root_family_id", "scenario", "data_role", "selected_branch_legacy", "legacy_ambiguous", "any_raw_overlap", "overlap_frames", "valid_observation_frames"])
    by_scenario = defaultdict(list)
    for row in summaries: by_scenario[row["scenario"]].append(row)
    overlap_rows = []
    for scenario, values in sorted(by_scenario.items()):
        denom = len(values)
        overlap_rows.append({"scenario": scenario, "rollouts": denom, "families": len({v["root_family_id"] for v in values}),
                             "legacy_ambiguous_rollouts": sum(v["legacy_ambiguous"] for v in values),
                             "any_raw_overlap_rollouts": sum(v["any_raw_overlap"] for v in values),
                             "overlap_frames": sum(v["overlap_frames"] for v in values),
                             "valid_observation_frames": sum(v["valid_observation_frames"] for v in values),
                             "legacy_ambiguous_rollout_rate": sum(v["legacy_ambiguous"] for v in values) / denom if denom else None,
                             "any_raw_overlap_rollout_rate": sum(v["any_raw_overlap"] for v in values) / denom if denom else None})
    _write_csv(output_root / "overlap_by_scenario.csv", overlap_rows)
    denominator = {
        "split": split, "resolved_split": resolved_split, "rollouts": len(summaries), "families": len({row["root_family_id"] for row in summaries}),
        "valid_observation_frames": len(all_trace), "legacy_ambiguous_rollout_count": sum(row["legacy_ambiguous"] for row in summaries),
        "any_raw_overlap_rollout_count": sum(row["any_raw_overlap"] for row in summaries),
        "raw_overlap_frame_count": sum(row["overlap_frames"] for row in summaries),
        "reference_event_window_count": len(event_windows),
        "reference_event_windows_with_raw_overlap": sum(row["raw_overlap_at_event"] for row in event_windows),
        "legacy_ambiguous_rollout_rate": sum(row["legacy_ambiguous"] for row in summaries) / len(summaries) if summaries else None,
        "any_raw_overlap_rollout_rate": sum(row["any_raw_overlap"] for row in summaries) / len(summaries) if summaries else None,
        "event_window_overlap_is_reported_separately": True,
    }
    _write_json(output_root / "denominator_ledger.json", denominator)
    _write_csv(output_root / "event_window_overlap.csv", event_windows,
               ["event_id", "rollout_id", "root_family_id", "reference_event_type", "reference_label_status", "frame_index", "raw_overlap_at_event"])
    reproduction = {"split": split, "graph_id": graph["graph_id"], "rollouts": len(summaries),
                    "families": len({row["root_family_id"] for row in summaries}),
                    "branch_accuracy": sum(row["branch_correct"] for row in summaries) / len(summaries) if summaries else None,
                    "ambiguous_edge_rate": sum(row["legacy_ambiguous"] for row in summaries) / len(summaries) if summaries else None,
                    "legacy_ambiguous_rollout_count": sum(row["legacy_ambiguous"] for row in summaries),
                    "raw_prediction_replayed": True, "old_predict_branch_used": True,
                    "raw_overlap_rollout_count": sum(row["any_raw_overlap"] for row in summaries)}
    _write_json(output_root / "legacy_reproduction.json", reproduction)
    report = (f"# Replay report: {split}\n\n- Rollouts: `{len(summaries)}`\n- Families: `{len({row['root_family_id'] for row in summaries})}`\n"
              f"- Legacy ambiguous rollouts: `{reproduction['legacy_ambiguous_rollout_count']}`\n"
              f"- Any raw overlap rollouts: `{reproduction['raw_overlap_rollout_count']}`\n"
              "- Raw predicate overlap and legacy final arbitration are reported separately.\n"
              f"- Confirmation data role: `{'POST_HOC_DIAGNOSIS' if split == 'legacy_confirmation' else 'DEVELOPMENT_REFERENCE'}`\n")
    (output_root / "replay_report.md").write_text(report, encoding="utf-8")
    return {"reproduction": reproduction, "summaries": summaries, "trace": all_trace}
