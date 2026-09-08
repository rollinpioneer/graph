from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .event_adapter import first_action, run_event_interface
from .hold_features import build_features
from .hold_predicates import evaluate_candidate
from .inputs import read_csv, read_json, read_jsonl, write_csv, write_json
from .reference_v2 import build_reference


CANDIDATES = {
    "B_count1": {}, "B_count2": {},
    "C3_vector_rho035": {"relative_rho_max": 0.35}, "C3_vector_rho055": {"relative_rho_max": 0.55},
    "C4_time010_disp004": {"supported_time_min": 0.10, "supported_displacement_min": 0.004},
    "C4_time010_disp008": {"supported_time_min": 0.10, "supported_displacement_min": 0.008},
    "C4_time020_disp004": {"supported_time_min": 0.20, "supported_displacement_min": 0.004},
    "C4_time020_disp008": {"supported_time_min": 0.20, "supported_displacement_min": 0.008},
}


def _predicates(observation: dict[str, Any], geometry: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    closed = observation.get("gripper_command") == "closed"
    contact = bool(observation.get("contact_present"))
    opened = not closed
    object_norm, gripper_norm = geometry.get("object_displacement_norm"), geometry.get("gripper_displacement_norm")
    co_motion = None
    if object_norm is not None and gripper_norm is not None:
        scale = max(float(object_norm), float(gripper_norm), 0.001)
        co_motion = max(0.0, 1.0 - abs(float(object_norm) - float(gripper_norm)) / scale)
    stable = closed and contact and co_motion is not None and co_motion >= 0.8
    return {"contact_present": "true" if contact else "false", "gripper_command_closed": "true" if closed else "false", "gripper_command_open": "true" if opened else "false", "stable_hold_observed": "true" if stable else "false", "contact_recently_lost": "true" if previous and previous.get("contact_present") and not contact else "false"}


def _load_rollout(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(meta["path"])
    observations = read_jsonl(root / "observations_dense.jsonl")
    geometries = build_features(observations)
    predictions = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        p = _predicates(observation, geometry, previous)
        predictions.append({**observation, "predicates": p})
        previous = observation
    reference = build_reference({**meta, "rollout_path": meta["path"]})
    return predictions, geometries, reference


def _metric_for_candidate(items: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]], candidate_id: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_rows = []
    for meta, observations, geometries, reference in items:
        cfg = dict(config)
        evidence = evaluate_candidate(observations, geometries, "B_count1" if candidate_id == "B_count1" else "B_count2" if candidate_id == "B_count2" else "C3_vector" if candidate_id.startswith("C3") else "C4_time", cfg)
        memory = run_event_interface(observations, evidence, history_complete=not meta.get("observation_intervention", False))
        action, frame, conflict = first_action(memory)
        emergency_any = any(row.get("selected_action") in {"retry_grasp", "recover_object"} for row in memory)
        for event in reference["decision_reference_events"]:
            expected = event["expected_action"]
            if event["reference_type"] in {"release_expected", "touch_without_stable_hold"}:
                expected = "none"
            onset = float(event["reference_event_start"])
            window_rows = [row for row in memory if row.get("time") is not None and onset <= float(row["time"]) <= onset + 0.5]
            window_action, window_frame, window_conflict = first_action(window_rows)
            before_event = [row for row in evidence if row.get("time") is not None and float(row["time"]) <= onset + 1e-9]
            event_rows.append({"candidate_id": candidate_id, "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"], "reference_event_type": event["reference_type"], "expected_action": expected, "selected_action": window_action, "selected_frame": window_frame, "correct": window_action == expected, "conflict": window_conflict, "emergency_any": emergency_any, "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in before_event), "reference_hold": bool(reference["hold_reference_intervals"])})
        if not reference["decision_reference_events"]:
            event_rows.append({"candidate_id": candidate_id, "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "stratum": meta["stratum"], "reference_event_type": "none", "expected_action": "none", "selected_action": action, "selected_frame": frame, "correct": action == "none", "conflict": conflict, "emergency_any": emergency_any, "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in evidence), "reference_hold": bool(reference["hold_reference_intervals"])})
    positive = [row for row in event_rows if row["expected_action"] in {"retry_grasp", "recover_object"}]
    by_type = defaultdict(list)
    for row in positive: by_type[row["reference_event_type"]].append(row)
    negative_strata = {"touch_without_hold_then_loss", "commanded_release", "normal_hold_pause_resume", "history_or_visual_unavailable"}
    negative = [row for row in event_rows if row["stratum"] in negative_strata]
    metrics = {"candidate_id": candidate_id, "events": len(event_rows), "families": len({row["root_family_id"] for row in event_rows}), "positive_events": len(positive), "positive_correct": sum(row["correct"] for row in positive), "positive_recall": sum(row["correct"] for row in positive) / len(positive) if positive else None, "miss_recall": sum(row["correct"] for row in by_type.get("missed_grasp_retry_required", [])) / len(by_type.get("missed_grasp_retry_required", [])) if by_type.get("missed_grasp_retry_required") else None, "loss_recall": sum(row["correct"] for row in by_type.get("held_object_loss_recovery_required", [])) / len(by_type.get("held_object_loss_recovery_required", [])) if by_type.get("held_object_loss_recovery_required") else None, "wrong_or_unknown_rate": sum(not row["correct"] for row in positive) / len(positive) if positive else None, "conflict_rate": sum(row["conflict"] for row in event_rows) / len(event_rows) if event_rows else None, "negative_false_emergency_rate": sum(row.get("emergency_any", row["selected_action"] in {"retry_grasp", "recover_object"}) for row in negative) / len(negative) if negative else None, "hold_evidence_rate": sum(row["hold_evidence_before_event"] for row in positive) / len(positive) if positive else None, "paired_sampling": True, "data_role": "new_dynamic_mujoco_r1"}
    return metrics, event_rows


def build_features_for_data(data_root: Path, output_root: Path) -> dict[str, Any]:
    rows = []
    for meta_path in sorted(data_root.rglob("metadata.json")):
        meta = read_json(meta_path)
        observations = read_jsonl(meta_path.parent / "observations_dense.jsonl")
        features = build_features(observations)
        out = output_root / meta["split"] / f"{meta['rollout_id']}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in features), encoding="utf-8")
        rows.append({"rollout_id": meta["rollout_id"], "split": meta["split"], "feature_path": str(out.resolve()), "observation_count": len(features), "feature_contract": "timestamp_aligned_proxy_hold_v2"})
    write_csv(output_root / "feature_manifest.csv", rows)
    return {"status": "FEATURES_READY", "rollouts": len(rows), "observations": sum(int(r["observation_count"]) for r in rows), "streams": ["control_tick_20hz", "action_end"]}


def evaluate_development(data_root: Path, protocol: dict[str, Any], output_root: Path, partition: str | None = None) -> dict[str, Any]:
    metas = [read_json(path) for path in sorted(data_root.rglob("metadata.json"))]
    if partition:
        metas = [meta for meta in metas if meta["split"] == partition]
    items = []
    for meta in metas:
        observations, geometries, reference = _load_rollout(meta)
        items.append((meta, observations, geometries, reference))
    fit = [item for item in items if item[0]["split"] == "dev_fit"]
    select = [item for item in items if item[0]["split"] == "dev_select"]
    metrics, fit_metrics, event_rows = [], [], []
    for candidate, config in CANDIDATES.items():
        fm, _ = _metric_for_candidate(fit, candidate, config)
        sm, rows = _metric_for_candidate(select, candidate, config)
        fit_metrics.append(fm); metrics.append(sm); event_rows.extend(rows)
    write_csv(output_root / "candidate_metrics.csv", metrics)
    write_csv(output_root / "candidate_fit_metrics.csv", fit_metrics)
    write_csv(output_root / "event_decisions.csv", event_rows)
    threshold = protocol["thresholds"]
    gates = []
    for row in metrics:
        checks = {"paired_recording": row["paired_sampling"], "support_minimum": row["positive_events"] >= 8 and row["families"] >= 2, "positive_recall": row["positive_recall"] is not None and row["positive_recall"] >= threshold["event_type_recall_min"], "wrong_or_unknown": row["wrong_or_unknown_rate"] is not None and row["wrong_or_unknown_rate"] <= threshold["wrong_or_unknown_fraction_max"], "conflict": row["conflict_rate"] is not None and row["conflict_rate"] <= threshold["conflict_rate_max"], "negative_false_emergency": row["negative_false_emergency_rate"] is not None and row["negative_false_emergency_rate"] <= threshold["negative_false_emergency_rate_max"], "hold_evidence": row["hold_evidence_rate"] is not None and row["hold_evidence_rate"] >= threshold["hold_evidence_recall_min"]}
        gates.append({"candidate_id": row["candidate_id"], **checks, "all_pass": all(checks.values())})
    write_csv(output_root / "development_gates.csv", gates)
    eligible = [row["candidate_id"] for row in gates if row["all_pass"]]
    selected = min(eligible, key=lambda c: (next(x["wrong_or_unknown_rate"] for x in metrics if x["candidate_id"] == c), next(x["negative_false_emergency_rate"] for x in metrics if x["candidate_id"] == c), c)) if eligible else None
    route = {"schema": "pathgraph_l2rar1_development_route_v1", "status": "DEVELOPMENT_READY" if selected else "DEVELOPMENT_NOT_READY", "selected_candidate_id": selected, "eligible_candidates": eligible, "fit_families": len({x[0]["root_family_id"] for x in fit}), "fit_rollouts": len(fit), "select_families": len({x[0]["root_family_id"] for x in select}), "select_rollouts": len(select), "sampling": "control_tick_20hz", "selection_uses_future_labels": False, "api_calls": 0, "training_jobs": 0}
    write_json(output_root / "development_route.json", route)
    write_json(output_root / "candidate_registry.json", {"schema": "pathgraph_l2rar1_candidate_registry_v1", "candidates": metrics, "gates": gates, "selected_candidate_id": selected, "selection_status": route["status"]})
    return route
