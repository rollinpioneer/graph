from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .event_adapter import run_event_interface
from .hold_features import build_features
from .hold_predicates import evaluate_candidate
from .inputs import read_json, read_jsonl, write_csv, write_json
from .reference_v2 import build_reference


CANDIDATES = {
    "B_count1": {},
    "B_count2": {},
    "C3_vector_rho035": {"relative_rho_max": 0.35},
    "C3_vector_rho055": {"relative_rho_max": 0.55},
    "C4_time010_disp004": {"supported_time_min": 0.10, "supported_displacement_min": 0.004},
    "C4_time010_disp008": {"supported_time_min": 0.10, "supported_displacement_min": 0.008},
    "C4_time020_disp004": {"supported_time_min": 0.20, "supported_displacement_min": 0.004},
    "C4_time020_disp008": {"supported_time_min": 0.20, "supported_displacement_min": 0.008},
}


def _predicates(observation: dict[str, Any], geometry: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if observation.get("observation_masked"):
        return {key: "unknown" for key in ("contact_present", "gripper_command_closed", "gripper_command_open", "stable_hold_observed", "contact_recently_lost")}
    closed = observation.get("gripper_command") == "closed"
    contact = bool(observation.get("contact_present"))
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    co_motion = None
    if object_norm is not None and gripper_norm is not None:
        scale = max(float(object_norm), float(gripper_norm), 0.001)
        co_motion = max(0.0, 1.0 - abs(float(object_norm) - float(gripper_norm)) / scale)
    return {
        "contact_present": "true" if contact else "false",
        "gripper_command_closed": "true" if closed else "false",
        "gripper_command_open": "false" if closed else "true",
        "stable_hold_observed": "true" if closed and contact and co_motion is not None and co_motion >= 0.8 else "false",
        "contact_recently_lost": "true" if previous and previous.get("contact_present") is True and not contact else "false",
    }


def _stream_path(meta: dict[str, Any], sampling: str) -> Path:
    return Path(meta["path"]) / ("observations_dense.jsonl" if sampling == "control_tick_20hz" else "observations_action_end.jsonl")


def _load_rollout(meta: dict[str, Any], sampling: str):
    observations = read_jsonl(_stream_path(meta, sampling))
    geometries = build_features(observations)
    predictions, previous = [], None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    return predictions, geometries, build_reference({**meta, "rollout_path": meta["path"]})


def _first_action(rows: list[dict[str, Any]]) -> tuple[str, int | None, float | None, bool]:
    actions = []
    for index, row in enumerate(rows):
        guards = row.get("effective_guards", {})
        if guards.get("retry_grasp") == "true":
            actions.append((index, "retry_grasp", row.get("time")))
        if guards.get("recover_object") == "true":
            actions.append((index, "recover_object", row.get("time")))
    if not actions:
        return "none", None, None, False
    first_index = min(item[0] for item in actions)
    first = [item for item in actions if item[0] == first_index]
    names = {item[1] for item in first}
    return ("conflict" if len(names) > 1 else first[0][1]), first_index, first[0][2], len(names) > 1


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))]


def _metric_for_candidate(items, candidate_id: str, config: dict[str, Any], sampling: str):
    event_rows, conflict_rollouts = [], set()
    for meta, observations, geometries, reference in items:
        base_id = "B_count1" if candidate_id == "B_count1" else "B_count2" if candidate_id == "B_count2" else "C3_vector" if candidate_id.startswith("C3") else "C4_time"
        evidence = evaluate_candidate(observations, geometries, base_id, dict(config))
        memory = run_event_interface(observations, evidence, history_complete=not meta.get("observation_intervention", False))
        references = reference["decision_reference_events"] or [{"reference_type": "none", "expected_action": "none", "decision_window_start": -float("inf"), "decision_window_end": float("inf"), "observation_quality": "recorded"}]
        for event in references:
            start, end = float(event["decision_window_start"]), float(event["decision_window_end"])
            window = [row for row in memory if row.get("time") is not None and start - 1e-9 <= float(row["time"]) <= end + 1e-9]
            action, frame, action_time, conflict = _first_action(window)
            expected = event["expected_action"]
            before = [row for row in evidence if start != -float("inf") and row.get("time") is not None and float(row["time"]) <= start + 1e-9]
            event_rows.append({
                "candidate_id": candidate_id, "sampling": sampling, "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"],
                "reference_event_type": event["reference_type"], "expected_action": expected, "selected_action": action, "selected_frame": frame, "selected_time": action_time,
                "correct": action == expected, "conflict": conflict, "emergency_any": any(row.get("selected_action") in {"retry_grasp", "recover_object"} for row in window),
                "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in before), "reference_hold": bool(event.get("reference_hold_interval")),
                "delay_seconds": float(action_time) - start if action_time is not None and start != -float("inf") and action == expected else None,
                "observation_quality": event.get("observation_quality", "recorded"),
            })
            if conflict:
                conflict_rollouts.add(meta["rollout_id"])
    positives = [row for row in event_rows if row["expected_action"] in {"retry_grasp", "recover_object"}]
    negatives = [row for row in event_rows if row["expected_action"] == "none"]
    information = [row for row in event_rows if row["expected_action"] == "needs_observation"]
    by_type = defaultdict(list)
    by_stratum = defaultdict(list)
    for row in positives:
        by_type[row["reference_event_type"]].append(row)
    for row in event_rows:
        by_stratum[row["stratum"]].append(row)

    def rate(rows, predicate):
        return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None

    type_recall = {key: rate(rows, lambda row: row["correct"]) for key, rows in sorted(by_type.items())}
    stratum_false = {key: rate(rows, lambda row: row["emergency_any"]) for key, rows in sorted(by_stratum.items())}
    delays = [float(row["delay_seconds"]) for row in positives if row["delay_seconds"] is not None]
    metrics = {
        "candidate_id": candidate_id, "sampling": sampling, "events": len(event_rows), "families": len({row["root_family_id"] for row in event_rows}),
        "positive_events": len(positives), "positive_correct": sum(bool(row["correct"]) for row in positives), "positive_recall": rate(positives, lambda row: row["correct"]),
        "miss_recall": rate(by_type.get("missed_grasp_retry_required", []), lambda row: row["correct"]), "loss_recall": rate(by_type.get("held_object_loss_recovery_required", []), lambda row: row["correct"]),
        "brief_loss_recall": rate([row for row in positives if row["stratum"] == "brief_true_hold_then_loss"], lambda row: row["correct"]), "long_gap_recall": rate([row for row in positives if row["stratum"] == "long_gap_after_loss"], lambda row: row["correct"]),
        "wrong_or_unknown_rate": rate(positives, lambda row: not row["correct"]), "conflict_rate": rate(event_rows, lambda row: row["conflict"]),
        "conflict_rollout_rate": len(conflict_rollouts) / len({row["rollout_id"] for row in event_rows}) if event_rows else None,
        "negative_false_emergency_rate": rate(negatives, lambda row: row["emergency_any"]), "information_false_emergency_rate": rate(information, lambda row: row["emergency_any"]),
        "hold_evidence_rate": rate(positives, lambda row: row["hold_evidence_before_event"]), "brief_hold_evidence_rate": rate([row for row in positives if row["stratum"] == "brief_true_hold_then_loss"], lambda row: row["hold_evidence_before_event"]),
        "delay_p95_seconds": _p95(delays), "delay_observed_count": len(delays), "paired_sampling": all(_stream_path(meta, sampling).is_file() for meta, *_ in items),
        "type_recall_json": json.dumps(type_recall, sort_keys=True), "stratum_false_emergency_json": json.dumps(stratum_false, sort_keys=True), "data_role": "new_dynamic_mujoco_r1",
    }
    return metrics, event_rows


def build_features_for_data(data_root: Path, output_root: Path) -> dict[str, Any]:
    rows = []
    for meta_path in sorted(data_root.rglob("metadata.json")):
        meta = read_json(meta_path)
        for sampling, filename in (("control_tick_20hz", "observations_dense.jsonl"), ("action_end", "observations_action_end.jsonl")):
            observations = read_jsonl(meta_path.parent / filename)
            features = build_features(observations)
            out = output_root / meta["split"] / sampling / f"{meta['rollout_id']}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in features), encoding="utf-8")
            rows.append({"rollout_id": meta["rollout_id"], "split": meta["split"], "sampling": sampling, "feature_path": str(out.resolve()), "observation_count": len(features), "feature_contract": "timestamp_aligned_proxy_hold_v2"})
    write_csv(output_root / "feature_manifest.csv", rows)
    return {"status": "FEATURES_READY", "rollouts": len({row["rollout_id"] for row in rows}), "observations": sum(int(row["observation_count"]) for row in rows), "streams": ["control_tick_20hz", "action_end"]}


def evaluate_development(data_root: Path, protocol: dict[str, Any], output_root: Path, partition: str | None = None) -> dict[str, Any]:
    metas = [read_json(path) for path in sorted(data_root.rglob("metadata.json"))]
    if partition:
        metas = [meta for meta in metas if meta["split"] == partition]
    items = {"control_tick_20hz": {"dev_fit": [], "dev_select": []}, "action_end": {"dev_fit": [], "dev_select": []}}
    for meta in metas:
        for sampling in items:
            predictions, geometries, reference = _load_rollout(meta, sampling)
            items[sampling][meta["split"]].append((meta, predictions, geometries, reference))
    all_metrics, fit_metrics, event_rows = [], [], []
    for candidate_id, config in CANDIDATES.items():
        for sampling in items:
            fit_metric, _ = _metric_for_candidate(items[sampling]["dev_fit"], candidate_id, config, sampling)
            select_metric, rows = _metric_for_candidate(items[sampling]["dev_select"], candidate_id, config, sampling)
            fit_metrics.append(fit_metric); all_metrics.append(select_metric); event_rows.extend(rows)
    write_csv(output_root / "candidate_metrics.csv", all_metrics)
    write_csv(output_root / "candidate_fit_metrics.csv", fit_metrics)
    write_csv(output_root / "event_decisions.csv", event_rows)
    threshold = protocol["thresholds"]
    dense_metrics = [row for row in all_metrics if row["sampling"] == "control_tick_20hz"]
    gates = []
    for row in dense_metrics:
        type_recall = json.loads(row["type_recall_json"])
        required = [value for key, value in type_recall.items() if key in {"missed_grasp_retry_required", "held_object_loss_recovery_required"} and value is not None]
        checks = {
            "paired_recording": row["paired_sampling"], "support_minimum": row["positive_events"] >= 8 and row["families"] >= 2 and len(required) >= 2,
            "positive_recall": bool(required) and min(required) >= threshold["event_type_recall_min"], "wrong_or_unknown": row["wrong_or_unknown_rate"] is not None and row["wrong_or_unknown_rate"] <= threshold["wrong_or_unknown_fraction_max"],
            "conflict": row["conflict_rate"] is not None and row["conflict_rate"] <= threshold["conflict_rate_max"] and row["conflict_rollout_rate"] <= threshold["conflict_rate_max"],
            "negative_false_emergency": row["negative_false_emergency_rate"] is not None and row["negative_false_emergency_rate"] <= threshold["negative_false_emergency_rate_max"],
            "information_false_emergency": row["information_false_emergency_rate"] is None or row["information_false_emergency_rate"] <= threshold["negative_false_emergency_rate_max"],
            "hold_evidence": row["hold_evidence_rate"] is not None and row["hold_evidence_rate"] >= threshold["hold_evidence_recall_min"], "delay_p95": row["delay_p95_seconds"] is not None and row["delay_p95_seconds"] <= threshold["delay_p95_seconds_max"],
        }
        gates.append({"candidate_id": row["candidate_id"], **checks, "all_pass": all(checks.values())})
    write_csv(
        output_root / "development_gates.csv",
        gates,
        fields=[
            "candidate_id",
            "paired_recording",
            "support_minimum",
            "positive_recall",
            "wrong_or_unknown",
            "conflict",
            "negative_false_emergency",
            "information_false_emergency",
            "hold_evidence",
            "delay_p95",
            "all_pass",
        ],
    )
    eligible = [row["candidate_id"] for row in gates if row["all_pass"]]
    selected = min(eligible, key=lambda candidate: (next(row["wrong_or_unknown_rate"] for row in dense_metrics if row["candidate_id"] == candidate), next(row["negative_false_emergency_rate"] for row in dense_metrics if row["candidate_id"] == candidate), candidate)) if eligible else None
    route = {"schema": "pathgraph_l2rar1_development_route_v1", "status": "DEVELOPMENT_READY" if selected else "DEVELOPMENT_NOT_READY", "selected_candidate_id": selected, "eligible_candidates": eligible, "fit_families": len({item[0]["root_family_id"] for item in items["control_tick_20hz"]["dev_fit"]}), "fit_rollouts": len(items["control_tick_20hz"]["dev_fit"]), "select_families": len({item[0]["root_family_id"] for item in items["control_tick_20hz"]["dev_select"]}), "select_rollouts": len(items["control_tick_20hz"]["dev_select"]), "sampling": "control_tick_20hz", "comparison_sampling": "action_end", "selection_uses_future_labels": False, "candidate_count": len(CANDIDATES), "api_calls": 0, "training_jobs": 0}
    write_json(output_root / "development_route.json", route)
    write_json(output_root / "candidate_registry.json", {"schema": "pathgraph_l2rar1_candidate_registry_v1", "candidates": all_metrics, "gates": gates, "selected_candidate_id": selected, "selection_status": route["status"]})
    return route
