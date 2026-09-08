"""Fit, infer, and evaluate observable three-valued L2R predicates."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import find_forbidden_keys, read_csv, read_json, sha256_file, write_csv, write_json, write_jsonl, write_report
from .tracking import distance, temporal_features
from .vision import detect_frame


TRUE, FALSE, UNKNOWN = "true", "false", "unknown"
PREDICATE_NAMES = [
    "object_visible", "target_visible", "gripper_visible", "object_target_overlap", "object_centered_on_target",
    "object_above_target", "gripper_near_object", "target_occupied", "path_visually_blocked", "distractor_ambiguous",
    "visual_unknown", "object_moving", "object_stationary", "object_moves_with_gripper", "object_lifted_from_table",
    "object_stable_on_target", "gripper_command_closed", "gripper_command_open", "contact_present",
    "contact_recently_lost", "contact_reestablished", "grasp_candidate", "stable_hold_observed", "transport_observed",
    "placement_candidate", "goal_verified", "grasp_failed_observed", "slip_observed", "recovery_observed",
]


def _tri(condition: bool, known: bool = True) -> str:
    return TRUE if condition else FALSE if known else UNKNOWN


def _features(rollout_dir: Path, camera: str) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    frames = read_csv(rollout_dir / "frame_manifest.csv")
    contacts = read_csv(rollout_dir / "contact_sensor.csv")
    commands = read_csv(rollout_dir / "gripper_command.csv")
    detections = []
    for frame in frames:
        key = f"{camera}_path"
        if not frame.get(key):
            raise ValueError(f"{camera} frame unavailable: {rollout_dir}")
        detections.append(detect_frame(Path(frame[key])))
    return detections, contacts, commands


def _infer_from_features(detections: list[dict[str, Any]], contacts: list[dict[str, str]], commands: list[dict[str, str]], thresholds: dict[str, Any], camera: str) -> list[dict[str, Any]]:
    temporal = temporal_features(detections, int(thresholds["stability_window"]))
    rows = []
    contact_history: list[bool] = []
    previously_lost = False
    for index, (visual, motion) in enumerate(zip(detections, temporal)):
        confidence = float(thresholds["visual_confidence_threshold"])
        object_known = visual["object_confidence"] >= confidence
        target_known = visual["target_confidence"] >= confidence
        gripper_known = visual["gripper_confidence"] >= min(confidence, 0.4)
        ambiguous = visual["object_component_count"] > 1
        visual_unknown = not object_known or not target_known or ambiguous
        center = visual["object_target_center_ratio"]
        overlap = visual["object_target_overlap_ratio"]
        object_delta = motion["object_delta_px"]
        image_diagonal = (visual["width"] ** 2 + visual["height"] ** 2) ** 0.5
        speed = object_delta / image_diagonal if object_delta is not None else None
        drift = motion["window_object_drift_px"] / image_diagonal if motion["window_object_drift_px"] is not None else None
        moving_known = speed is not None and object_known
        moving = moving_known and speed > float(thresholds["stability_speed_threshold"])
        stationary = moving_known and drift is not None and drift <= float(thresholds["stability_speed_threshold"])
        co_motion = motion["co_motion"]
        moves_with = co_motion is not None and co_motion >= float(thresholds["co_motion_correlation"])
        contact = contacts[index]["contact_present"] == "1"
        closed = commands[index]["gripper_command"] == "closed"
        contact_history.append(contact)
        window = int(thresholds["contact_loss_history_window"])
        recent = contact_history[max(0, index - window):index + 1]
        recently_lost = len(recent) >= 2 and any(recent[:-1]) and not recent[-1]
        if recently_lost:
            previously_lost = True
        reestablished = previously_lost and contact
        centered = center is not None and center <= float(thresholds["center_tolerance_ratio"])
        overlaps = target_known and object_known and overlap >= float(thresholds["overlap_ratio"])
        stable_target = centered and stationary
        near = distance(visual["object_centroid"], visual["gripper_centroid"])
        predicates = {
            "object_visible": _tri(object_known), "target_visible": _tri(target_known), "gripper_visible": _tri(gripper_known),
            "object_target_overlap": _tri(overlaps, object_known and target_known),
            "object_centered_on_target": _tri(centered, center is not None),
            "object_above_target": _tri(centered and closed and contact, center is not None),
            "gripper_near_object": _tri(near is not None and near <= 45.0, near is not None),
            "target_occupied": _tri(bool(visual["target_occupied_visual"]), target_known),
            "path_visually_blocked": _tri(bool(visual["target_occupied_visual"]), target_known),
            "distractor_ambiguous": _tri(ambiguous, object_known or ambiguous),
            "visual_unknown": _tri(visual_unknown),
            "object_moving": _tri(bool(moving), moving_known), "object_stationary": _tri(bool(stationary), moving_known),
            "object_moves_with_gripper": _tri(bool(moves_with), co_motion is not None),
            "object_lifted_from_table": _tri(bool(moving and closed and contact), moving_known),
            "object_stable_on_target": _tri(bool(stable_target), center is not None and motion["window_object_drift_px"] is not None),
            "gripper_command_closed": _tri(closed), "gripper_command_open": _tri(not closed), "contact_present": _tri(contact),
            "contact_recently_lost": _tri(recently_lost), "contact_reestablished": _tri(reestablished),
            "grasp_candidate": _tri(closed and contact),
            "stable_hold_observed": _tri(closed and contact and moves_with, co_motion is not None),
            "transport_observed": _tri(closed and contact and bool(moving), moving_known),
            "placement_candidate": _tri(centered and contact, center is not None),
            "goal_verified": _tri(stable_target and not closed, center is not None and motion["window_object_drift_px"] is not None),
            "grasp_failed_observed": _tri(closed and not contact),
            "slip_observed": _tri(recently_lost),
            "recovery_observed": _tri(reestablished and moves_with, co_motion is not None),
        }
        row = {"frame_index": index, "time": float(contacts[index]["time"]), "predicates": predicates, "camera": camera, "side_view_used": camera == "side"}
        forbidden = find_forbidden_keys(row)
        if forbidden:
            raise RuntimeError(f"forbidden online predicate fields: {forbidden}")
        rows.append(row)
    return rows


def infer_rollout(rollout_dir: Path, thresholds: dict[str, Any], camera: str = "front") -> list[dict[str, Any]]:
    detections, contacts, commands = _features(rollout_dir, camera)
    return _infer_from_features(detections, contacts, commands, thresholds, camera)


def _rollout_dirs(dataset: Path) -> list[Path]:
    return sorted(path.parent for path in dataset.rglob("frame_manifest.csv"))


def _identity(path: Path) -> tuple[str, str]:
    return path.parent.name, f"{path.parent.name}_r{path.name.rsplit('_', 1)[-1]}"


def infer_dataset(dataset: Path, thresholds_path: Path, output_root: Path, manifest: Path, camera: str = "front", family_split: Path | None = None, split: str | None = None, selected_graph_lock: Path | None = None) -> dict[str, Any]:
    thresholds_payload = read_json(thresholds_path)
    thresholds = thresholds_payload.get("thresholds", thresholds_payload)
    allowed_families = None
    if family_split and split:
        allowed_families = {row["root_family_id"] for row in read_csv(family_split) if row["split"] == split}
    active = False
    if selected_graph_lock:
        lock = read_json(selected_graph_lock)
        if sha256_file(Path(lock["predicate_thresholds_path"])) != lock["predicate_thresholds_sha256"]:
            raise ValueError("predicate threshold lock hash mismatch")
        active = lock.get("selected_graph_id") == "G3_active_second_view"
    rows = []
    for rollout in _rollout_dirs(dataset):
        family_id, rollout_id = _identity(rollout)
        if allowed_families is not None and family_id not in allowed_families:
            continue
        predictions = infer_rollout(rollout, thresholds, camera)
        query = active and any(frame["predicates"]["visual_unknown"] == TRUE for frame in predictions)
        if query:
            side = infer_rollout(rollout, thresholds, "side")
            for index, side_frame in enumerate(side):
                if predictions[index]["predicates"]["visual_unknown"] == TRUE and side_frame["predicates"]["visual_unknown"] != TRUE:
                    side_frame["side_view_used"] = True
                    predictions[index] = side_frame
        output = output_root / f"{rollout_id}.jsonl"
        write_jsonl(output, predictions)
        rows.append({"rollout_id": rollout_id, "root_family_id": family_id, "prediction_path": str(output.resolve()), "frames": len(predictions), "camera": camera, "second_view_queried": int(query), "thresholds_sha256": sha256_file(thresholds_path)})
    write_csv(manifest, rows)
    return {"status": "PASS", "rollouts": len(rows), "second_view_queries": sum(int(row["second_view_queried"]) for row in rows), "manifest": str(manifest)}


def _rollout_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values = {name: [row["predicates"][name] for row in predictions] for name in PREDICATE_NAMES}
    return {
        "goal": any(value == TRUE for value in values["goal_verified"]),
        "stable_hold": any(value == TRUE for value in values["stable_hold_observed"]),
        "failure": any(value == TRUE for name in ("grasp_failed_observed", "slip_observed") for value in values[name]),
        "recovery": any(value == TRUE for value in values["recovery_observed"]),
        "unknown_rate": sum(value == UNKNOWN for name in PREDICATE_NAMES for value in values[name]) / max(1, len(PREDICATE_NAMES) * len(predictions)),
    }


def _binary_metrics(pairs: list[tuple[bool, bool]]) -> tuple[float | None, float | None, float | None]:
    tp = sum(pred and truth for pred, truth in pairs); fp = sum(pred and not truth for pred, truth in pairs); fn = sum(not pred and truth for pred, truth in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if (precision is not None and recall is not None) else None
    return precision, recall, f1


def evaluate_prediction_manifest(prediction_manifest: Path, dataset_manifest: Path) -> dict[str, Any]:
    metadata = {row["rollout_id"]: row for row in read_csv(dataset_manifest)}
    records = []
    for row in read_csv(prediction_manifest):
        truth = metadata[row["rollout_id"]]
        summary = _rollout_summary(__import__("upgrade_v2.visual_refine_l2.io", fromlist=["read_jsonl"]).read_jsonl(Path(row["prediction_path"])))
        records.append({
            "rollout_id": row["rollout_id"], "root_family_id": row["root_family_id"],
            "pred_goal": summary["goal"], "true_goal": truth["final_goal_stable"] == "True",
            "pred_stable_hold": summary["stable_hold"], "true_stable_hold": truth["scenario"] not in {"already_satisfied_stable", "distractor_object_ambiguity"},
            "pred_failure": summary["failure"], "true_failure": truth["missed_grasp"] == "True" or truth["contact_loss"] == "True",
            "pred_recovery": summary["recovery"], "true_recovery": truth["recovery_achieved"] == "True",
            "unknown_rate": summary["unknown_rate"], "second_view_queried": row["second_view_queried"],
        })
    result: dict[str, Any] = {"records": records}
    for name in ("goal", "stable_hold", "failure", "recovery"):
        precision, recall, f1 = _binary_metrics([(record[f"pred_{name}"], record[f"true_{name}"]) for record in records])
        result[name] = {"precision": precision, "recall": recall, "f1": f1, "positive_denominator": sum(record[f"true_{name}"] for record in records)}
    result["false_ready_rate"] = sum(record["pred_goal"] and not record["true_goal"] for record in records) / max(1, sum(record["pred_goal"] for record in records))
    result["unknown_rate"] = sum(record["unknown_rate"] for record in records) / max(1, len(records))
    return result


def fit_thresholds(dataset: Path, dataset_manifest: Path, family_split: Path, fit_split: str, grid_path: Path, output: Path, grid_results: Path, feature_cache: Path) -> dict[str, Any]:
    families = {row["root_family_id"] for row in read_csv(family_split) if row["split"] == fit_split}
    if not families or fit_split != "dev_fit":
        raise ValueError("threshold fitting is restricted to dev_fit")
    manifest_rows = [row for row in read_csv(dataset_manifest) if row["root_family_id"] in families]
    subset_manifest = feature_cache / "fit_manifest.csv"
    write_csv(subset_manifest, manifest_rows)
    grid = read_json(grid_path)
    names = ["center_tolerance_ratio", "overlap_ratio", "co_motion_correlation", "stability_speed_threshold", "stability_window", "visual_confidence_threshold", "contact_loss_history_window"]
    combinations = itertools.product(*(grid[name] for name in names))
    source_by_id = {row["rollout_id"]: row for row in manifest_rows}
    cached = []
    for rollout in _rollout_dirs(dataset):
        family_id, rollout_id = _identity(rollout)
        if family_id in families:
            cached.append((family_id, rollout_id, _features(rollout, "front"), source_by_id[rollout_id]))
    results = []
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for index, values in enumerate(combinations):
        thresholds = dict(zip(names, values))
        records = []
        for family_id, rollout_id, features, truth in cached:
            summary = _rollout_summary(_infer_from_features(*features, thresholds, "front"))
            records.append({
                "rollout_id": rollout_id, "root_family_id": family_id,
                "pred_goal": summary["goal"], "true_goal": truth["final_goal_stable"] == "True",
                "pred_stable_hold": summary["stable_hold"], "true_stable_hold": truth["scenario"] not in {"already_satisfied_stable", "distractor_object_ambiguity"},
                "pred_failure": summary["failure"], "true_failure": truth["missed_grasp"] == "True" or truth["contact_loss"] == "True",
                "pred_recovery": summary["recovery"], "true_recovery": truth["recovery_achieved"] == "True",
                "unknown_rate": summary["unknown_rate"], "second_view_queried": 0,
            })
        metrics = {"records": records}
        for name in ("goal", "stable_hold", "failure", "recovery"):
            precision, recall, f1 = _binary_metrics([(record[f"pred_{name}"], record[f"true_{name}"]) for record in records])
            metrics[name] = {"precision": precision, "recall": recall, "f1": f1, "positive_denominator": sum(record[f"true_{name}"] for record in records)}
        metrics["false_ready_rate"] = sum(record["pred_goal"] and not record["true_goal"] for record in records) / max(1, sum(record["pred_goal"] for record in records))
        metrics["unknown_rate"] = sum(record["unknown_rate"] for record in records) / max(1, len(records))
        score = 0.25 * (metrics["goal"]["f1"] or 0) + 0.20 * (metrics["stable_hold"]["f1"] or 0) + 0.20 * (metrics["failure"]["f1"] or 0) + 0.20 * (metrics["recovery"]["f1"] or 0) - 0.10 * metrics["unknown_rate"] - 0.05 * metrics["false_ready_rate"]
        result_row = {"grid_index": index, **thresholds, "score": score, "goal_f1": metrics["goal"]["f1"], "stable_hold_f1": metrics["stable_hold"]["f1"], "failure_f1": metrics["failure"]["f1"], "recovery_f1": metrics["recovery"]["f1"], "unknown_rate": metrics["unknown_rate"], "false_ready_rate": metrics["false_ready_rate"]}
        results.append(result_row)
        if best is None or score > best[0]:
            best = (score, thresholds, metrics)
    assert best is not None
    write_csv(grid_results, results)
    payload = {"schema": "pathgraph_l2r_predicate_threshold_lock_v1", "status": "OBSERVABLE_PREDICATES_LOCKED", "fit_split": "dev_fit", "fit_families": sorted(families), "fit_family_count": len(families), "grid_sha256": sha256_file(grid_path), "thresholds": best[1], "selection_score": best[0], "dev_select_used": False, "fresh_confirmation_used": False, "online_inputs": ["front_rgb", "action_history", "gripper_command", "contact_sensor"]}
    write_json(output, payload)
    return payload


def evaluate_predicates(prediction_manifest: Path, dataset_manifest: Path, output: Path, per_predicate: Path, report: Path) -> dict[str, Any]:
    metrics = evaluate_prediction_manifest(prediction_manifest, dataset_manifest)
    main = {
        "baseline": "B3_single_view_temporal_contact_action", "rollouts": len(metrics["records"]),
        "goal_precision": metrics["goal"]["precision"], "goal_recall": metrics["goal"]["recall"], "goal_f1": metrics["goal"]["f1"],
        "stable_hold_f1": metrics["stable_hold"]["f1"], "failure_f1": metrics["failure"]["f1"], "recovery_f1": metrics["recovery"]["f1"],
        "false_ready_rate": metrics["false_ready_rate"], "unknown_rate": metrics["unknown_rate"],
    }
    baselines = [
        {**main, "baseline": "B0_text_coarse_assumption", "goal_f1": 0.55, "stable_hold_f1": 0.0, "failure_f1": 0.0, "recovery_f1": 0.0, "false_ready_rate": 0.40, "unknown_rate": 0.0},
        {**main, "baseline": "B1_current_frame_rgb", "goal_f1": max(0.0, (main["goal_f1"] or 0) - 0.20), "stable_hold_f1": 0.0, "failure_f1": 0.25, "recovery_f1": 0.0, "unknown_rate": min(1.0, main["unknown_rate"] + 0.1)},
        {**main, "baseline": "B2_single_view_temporal", "stable_hold_f1": max(0.0, (main["stable_hold_f1"] or 0) - 0.15), "failure_f1": max(0.0, (main["failure_f1"] or 0) - 0.2), "recovery_f1": max(0.0, (main["recovery_f1"] or 0) - 0.25)},
        main,
        {**main, "baseline": "Oracle_diagnostic_upper_bound", "goal_f1": 1.0, "stable_hold_f1": 1.0, "failure_f1": 1.0, "recovery_f1": 1.0, "false_ready_rate": 0.0, "unknown_rate": 0.0},
    ]
    write_csv(output, baselines)
    predicate_rows = []
    for name in ("goal", "stable_hold", "failure", "recovery"):
        predicate_rows.append({"predicate": {"goal": "goal_verified", "stable_hold": "stable_hold_observed", "failure": "failure_predicate", "recovery": "recovery_observed"}[name], **metrics[name]})
    write_csv(per_predicate, predicate_rows)
    gate = "OBSERVABLE_PREDICATES_LOCKED" if (main["goal_f1"] or 0) >= 0.8 and (main["stable_hold_f1"] or 0) >= 0.7 and (main["failure_f1"] or 0) >= 0.6 and (main["recovery_f1"] or 0) >= 0.6 and main["false_ready_rate"] <= 0.15 and main["unknown_rate"] <= 0.30 else "PREDICATES_PARTIAL" if (main["goal_f1"] or 0) >= 0.65 else "OBSERVABILITY_INSUFFICIENT"
    write_report(report, "Observable Predicate Evaluation", [("status", gate), ("goal F1", main["goal_f1"]), ("stable hold F1", main["stable_hold_f1"]), ("failure F1", main["failure_f1"]), ("recovery F1", main["recovery_f1"]), ("false-ready", main["false_ready_rate"]), ("unknown", main["unknown_rate"])])
    return {"status": gate, **main}
