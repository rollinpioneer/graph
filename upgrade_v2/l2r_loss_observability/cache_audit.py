"""Read-only R11 cache audit for loss-event observability.

This module never passes oracle, event labels, scenario names, or future
observations into the online interface.  Oracle/event records are used only
to align the offline audit and to state what the cached files cannot prove.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from upgrade_v2.l2r_task_context.online_interface_repair import (
    FIXED_CANDIDATES,
    _load_candidate,
    _first_emergency,
    run_repaired_interface,
)
from upgrade_v2.l2r_task_context.reference import build_reference
from upgrade_v2.l2r_task_context.io import read_csv, read_json, read_jsonl, sha256, write_csv, write_json

from .contracts import (
    CASES,
    CONTRAST_CASES,
    FIXED_CANDIDATES as AUDIT_CANDIDATES,
    LOSS_CASES,
    PROXY_MARGIN_M,
    PROXY_OFFSET_Z_M,
    REFERENCE_DRIFT_LIMIT_M,
    REFERENCE_MIN_DISPLACEMENT_M,
    R11_VERSION,
    Rollout,
    bool_text,
    control_variant_text,
    euclidean,
    relative_vector,
    vector3,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rollouts(data_root: Path) -> list[Rollout]:
    rows = read_csv(data_root / "rollout_manifest.csv")
    if len(rows) != 32:
        raise ValueError(f"R11 cache audit requires 32 rollouts, found {len(rows)}")
    rollouts = []
    for row in sorted(rows, key=lambda item: item["rollout_id"]):
        rollouts.append(Rollout(
            rollout_id=row["rollout_id"],
            root_family_id=row["root_family_id"],
            case_id=row["case_id"],
            path=Path(row["path"]),
            family_seed=int(row["family_seed"]),
            rollout_seed=int(row["rollout_seed"]),
            requested_effect=row["requested_effect"],
            control_variant=control_variant_text(row.get("control_variant", "unknown")),
            repair_version=row.get("repair_version", "unknown"),
            collection_version=row.get("collection_version", "unknown"),
        ))
    if {item.case_id for item in rollouts} != set(CASES):
        raise ValueError("cache case coverage differs from frozen R2 case order")
    if len({item.root_family_id for item in rollouts}) != 4:
        raise ValueError("cache must contain four root families")
    return rollouts


def _event(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("event") == name), None)


def _oracle_by_order(path: Path) -> dict[int, dict[str, Any]]:
    rows = read_csv(path / "oracle_timeline.csv")
    return {int(row["capture_order"]): row for row in rows}


def _family_radii(lock_path: Path) -> dict[str, float]:
    lock = read_json(lock_path)
    return {row["root_family_id"]: float(row["physical_spec"]["object_radius"]) for row in lock["families"]}


def _observations(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path / "observations_dense.jsonl")


def _event_window(meta: Rollout, event_time: float, seconds: float = 0.5) -> tuple[float, float]:
    rows = _observations(meta.path)
    end = max(float(row["time"]) for row in rows)
    return event_time, min(event_time + seconds, end)


def _valid_centroid(row: dict[str, Any], key: str) -> tuple[float, float] | None:
    value = row.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _relative_centroid(row: dict[str, Any]) -> tuple[float, float] | None:
    obj = _valid_centroid(row, "object_centroid")
    grip = _valid_centroid(row, "gripper_centroid")
    if obj is None or grip is None:
        return None
    return obj[0] - grip[0], obj[1] - grip[1]


def _pixel_distance(left: tuple[float, float] | None, right: tuple[float, float] | None) -> float | None:
    if left is None or right is None:
        return None
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _visual_rows(meta: Rollout, event_time: float, window_end: float) -> list[dict[str, Any]]:
    rows = [row for row in _observations(meta.path) if event_time - 0.2 - 1e-9 <= float(row["time"]) <= window_end + 1e-9]
    output: list[dict[str, Any]] = []
    previous = None
    for row in rows:
        relative = _relative_centroid(row)
        delta = _pixel_distance(previous, relative)
        width, height = row.get("width"), row.get("height")
        diagonal = math.hypot(float(width), float(height)) if width and height else None
        output.append({
            "rollout_id": meta.rollout_id,
            "root_family_id": meta.root_family_id,
            "case_id": meta.case_id,
            "capture_order": row.get("capture_order"),
            "time": row.get("time"),
            "front_path": row.get("front_path"),
            "front_sha256": row.get("front_sha256"),
            "object_centroid_x_px": _valid_centroid(row, "object_centroid")[0] if _valid_centroid(row, "object_centroid") else None,
            "object_centroid_y_px": _valid_centroid(row, "object_centroid")[1] if _valid_centroid(row, "object_centroid") else None,
            "gripper_centroid_x_px": _valid_centroid(row, "gripper_centroid")[0] if _valid_centroid(row, "gripper_centroid") else None,
            "gripper_centroid_y_px": _valid_centroid(row, "gripper_centroid")[1] if _valid_centroid(row, "gripper_centroid") else None,
            "relative_centroid_x_px": relative[0] if relative else None,
            "relative_centroid_y_px": relative[1] if relative else None,
            "relative_step_delta_px": delta,
            "relative_step_delta_diagonal_ratio": delta / diagonal if delta is not None and diagonal else None,
            "object_confidence": row.get("object_confidence"),
            "gripper_confidence": row.get("gripper_confidence"),
            "contact_present_raw": row.get("contact_present"),
            "gripper_command_raw": row.get("gripper_command"),
            "observation_valid": relative is not None and row.get("front_path") is not None and row.get("front_sha256") is not None,
        })
        previous = relative
    return output


def _visual_summary(rows: list[dict[str, Any]], event_time: float) -> dict[str, Any]:
    post = [row for row in rows if float(row["time"]) >= event_time - 1e-9]
    valid = [row for row in post if row["observation_valid"]]
    deltas = [float(row["relative_step_delta_px"]) for row in valid if row["relative_step_delta_px"] is not None]
    # This is an audit measurement, not a new online candidate.  It records
    # any observed relative image motion above one pixel and never labels it
    # as a task-level loss by itself.
    signal = next((row for row in valid if row["relative_step_delta_px"] is not None and float(row["relative_step_delta_px"]) > 1.0), None)
    return {
        "visual_observable_status": "visual_motion_present_not_loss_specific" if signal else "visual_motion_not_established",
        "t_visual_observable": signal["time"] if signal else None,
        "visual_signal_rule": "valid adjacent relative object-gripper centroid motion > 1.0 px; audit-only, not a loss predicate",
        "valid_post_event_frames": len(valid),
        "max_relative_step_delta_px": max(deltas) if deltas else None,
        "max_relative_step_delta_diagonal_ratio": max((float(row["relative_step_delta_diagonal_ratio"]) for row in valid if row["relative_step_delta_diagonal_ratio"] is not None), default=None),
        "rgb_evidence_available": bool(valid),
    }


def _saved_contact_transition(observations: list[dict[str, Any]], start: float, end: float) -> tuple[float | None, str]:
    window = [row for row in observations if start - 1e-9 <= float(row["time"]) <= end + 1e-9]
    for previous, current in zip(window, window[1:]):
        if bool_text(previous.get("contact_present")) is True and bool_text(current.get("contact_present")) is False:
            return float(current["time"]), "contact_true_to_false_in_event_window"
    if any(bool_text(row.get("contact_present")) is None for row in window):
        return None, "contact_unknown_in_event_window"
    return None, "no_contact_true_to_false_in_event_window"


def _sensor_rows(meta: Rollout, event_time: float, radius: float) -> list[dict[str, Any]]:
    oracle = _oracle_by_order(meta.path)
    rows = []
    for observation in _observations(meta.path):
        if abs(float(observation["time"]) - event_time) > 0.25 and not (event_time <= float(observation["time"]) <= event_time + 0.5):
            continue
        oracle_row = oracle.get(int(observation["capture_order"]))
        if oracle_row is None:
            rows.append({
                "rollout_id": meta.rollout_id,
                "capture_order": observation.get("capture_order"),
                "time": observation.get("time"),
                "sensor_recompute_status": "oracle_row_missing",
            })
            continue
        object_xyz = vector3(oracle_row.get("object_xyz"))
        gripper_xyz = vector3(oracle_row.get("gripper_xyz"))
        proxy_center = tuple(gripper_xyz[index] + (PROXY_OFFSET_Z_M if index == 2 else 0.0) for index in range(3)) if gripper_xyz else None
        distance = euclidean(object_xyz, proxy_center)
        threshold = radius + PROXY_MARGIN_M
        weld = str(oracle_row.get("weld_state", "0")).lower() in {"1", "true"}
        closed = observation.get("gripper_command") == "closed"
        recomputed = None if distance is None else bool(weld or (closed and distance < threshold))
        saved = bool_text(observation.get("contact_present"))
        rows.append({
            "rollout_id": meta.rollout_id,
            "root_family_id": meta.root_family_id,
            "case_id": meta.case_id,
            "capture_order": observation.get("capture_order"),
            "time": observation.get("time"),
            "object_radius_m": radius,
            "proxy_offset_z_m": PROXY_OFFSET_Z_M,
            "proxy_margin_m": PROXY_MARGIN_M,
            "distance_to_proxy_center_m": distance,
            "proxy_threshold_m": threshold,
            "weld_state_offline": weld,
            "gripper_closed_raw": closed,
            "contact_present_saved": saved,
            "contact_present_recomputed": recomputed,
            "saved_equals_recomputed": None if saved is None or recomputed is None else saved == recomputed,
            "sensor_contract": "offline recomputation from oracle geometry; not supplied to online interface",
        })
    return rows


def _method_decisions(meta: Rollout, candidate_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    observations, evidence, requests = _load_candidate({"path": str(meta.path), "rollout_id": meta.rollout_id, "root_family_id": meta.root_family_id, "case_id": meta.case_id, "requested_effect": meta.requested_effect}, candidate_id)
    decisions = run_repaired_interface(observations, evidence, requests, history_complete=True)
    return observations, evidence, decisions


def _build_regression_audit(rollouts: list[Rollout], output_root: Path) -> dict[str, Any]:
    """Re-score all 32 cached opportunities without changing the frozen gate."""
    rows: list[dict[str, Any]] = []
    for meta in rollouts:
        reference = build_reference({
            "path": str(meta.path), "rollout_id": meta.rollout_id,
            "root_family_id": meta.root_family_id, "case_id": meta.case_id,
            "requested_effect": meta.requested_effect,
        })
        for candidate_id in AUDIT_CANDIDATES:
            _, evidence, decisions = _method_decisions(meta, candidate_id)
            labeled = reference.get("status") == "reference_labeled"
            event = reference["events"][0] if labeled else {}
            start = float(event["decision_window_start"]) if labeled else None
            end = float(event["decision_window_end"]) if labeled else None
            selected = _first_emergency(decisions, start, end) if labeled else None
            selected_action = selected.get("selected_action") if selected else "none"
            expected = event.get("expected_action") if labeled else None
            evidence_before = [
                row for row in evidence
                if labeled and row.get("time") is not None and float(row["time"]) <= start + 1e-9
            ]
            rows.append({
                "rollout_id": meta.rollout_id,
                "root_family_id": meta.root_family_id,
                "case_id": meta.case_id,
                "candidate_id": candidate_id,
                "reference_status": "reference_labeled" if labeled else "reference_unresolved",
                "reference_reason": None if labeled else reference.get("reason"),
                "expected_action": expected,
                "selected_action": selected_action,
                "selected_time": selected.get("time") if selected else None,
                "wrong_action": selected_action if labeled and selected_action != expected else "none",
                "emergency_any": selected_action != "none",
                "correct": selected_action == expected if labeled else None,
                "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in evidence_before),
                "pending_event_at_selection": selected.get("pending_event") if selected else "none",
                "online_oracle_used": False,
                "data_role": "repair_dev_cache_only",
            })

    metrics: list[dict[str, Any]] = []
    for candidate_id in AUDIT_CANDIDATES:
        own = [row for row in rows if row["candidate_id"] == candidate_id]
        labeled = [row for row in own if row["reference_status"] == "reference_labeled"]

        def case(case_id: str) -> list[dict[str, Any]]:
            return [row for row in labeled if row["case_id"] == case_id]

        def rate(values: list[dict[str, Any]], key: str) -> float | None:
            return sum(bool(row[key]) for row in values) / len(values) if values else None

        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "opportunity_rows": len(own),
            "reference_labeled_rows": len(labeled),
            "reference_unresolved_rows": len(own) - len(labeled),
            "all_labeled_event_accuracy": rate(labeled, "correct"),
            "wrong_action_rows": sum(row["wrong_action"] != "none" for row in labeled),
            "unknown_or_needs_observation_rows": sum(row["selected_action"] == "needs_observation" for row in labeled),
        }
        for case_id in CASES:
            short = case_id.split("_", 1)[0]
            values = case(case_id)
            if case_id in LOSS_CASES or case_id == "K1_hold_request_ends_without_hold":
                item[f"{short}_recall"] = rate(values, "correct")
            else:
                item[f"{short}_false_emergency_rate"] = rate(values, "emergency_any")
            item[f"{short}_n"] = len(values)
        metrics.append(item)

    write_csv(output_root / "regression_audit.csv", rows)
    payload = {
        "schema": "l2rar2_r11_regression_metrics_v1",
        "status": "REGRESSION_REPLAY_COMPLETE",
        "unique_rollouts": len(rollouts),
        "method_rows": len(rows),
        "current_unique_physical_events": len(rollouts),
        "candidates": metrics,
        "historical_context_included": False,
        "candidate_set_changed": False,
        "reference_contract_changed": False,
    }
    write_json(output_root / "regression_metrics.json", payload)
    return payload


def build_cache_audit(data_root: Path, lock_path: Path, output_root: Path, baseline_root: Path | None = None) -> dict[str, Any]:
    rollouts = load_rollouts(data_root)
    radii = _family_radii(lock_path)
    loss_rollouts = [item for item in rollouts if item.case_id in LOSS_CASES]
    if len(loss_rollouts) != 12:
        raise ValueError("expected 12 unique K4/K5/K6 loss events")

    event_chain: list[dict[str, Any]] = []
    loss_audit: list[dict[str, Any]] = []
    sensor_audit: list[dict[str, Any]] = []
    visual_audit: list[dict[str, Any]] = []
    timing_audit: list[dict[str, Any]] = []
    method_stage_counts: dict[str, Counter[str]] = {candidate: Counter() for candidate in AUDIT_CANDIDATES}
    regression = _build_regression_audit(rollouts, output_root)

    for meta in loss_rollouts:
        events = read_jsonl(meta.path / "events.jsonl")
        contact_loss = _event(events, "contact_lost")
        if contact_loss is None:
            raise ValueError(f"missing contact_lost in {meta.rollout_id}")
        t_detach = float(contact_loss["time"])
        start, end = _event_window(meta, t_detach)
        raw_obs = _observations(meta.path)
        t_contact, contact_status = _saved_contact_transition(raw_obs, start, end)
        visual_rows = _visual_rows(meta, t_detach, end)
        visual_summary = _visual_summary(visual_rows, t_detach)
        visual_audit.extend(visual_rows)
        oracle = _oracle_by_order(meta.path)
        before = [row for row in oracle.values() if float(row["time"]) < t_detach - 1e-9]
        after = [row for row in oracle.values() if float(row["time"]) >= t_detach - 1e-9]
        before_row = before[-1] if before else None
        after_row = after[0] if after else None
        relative_before = relative_vector(before_row.get("object_xyz"), before_row.get("gripper_xyz")) if before_row else None
        relative_after = relative_vector(after_row.get("object_xyz"), after_row.get("gripper_xyz")) if after_row else None
        loss_audit.append({
            "rollout_id": meta.rollout_id,
            "root_family_id": meta.root_family_id,
            "case_id": meta.case_id,
            "source_path": str(meta.path.resolve()),
            "event_name": "contact_lost",
            "t_detach_command": t_detach,
            "t_loss_physical_verified": None,
            "weld_state_before": before_row.get("weld_state") if before_row else None,
            "weld_state_at_event": oracle.get(int(contact_loss.get("capture_order", -1)), {}).get("weld_state"),
            "weld_state_after": after_row.get("weld_state") if after_row else None,
            "relative_position_before_m": json.dumps(relative_before) if relative_before else None,
            "relative_position_after_m": json.dumps(relative_after) if relative_after else None,
            "relative_position_change_m": euclidean(relative_before, relative_after),
            "object_height_before_m": vector3(before_row.get("object_xyz"))[2] if before_row and vector3(before_row.get("object_xyz")) else None,
            "object_height_after_m": vector3(after_row.get("object_xyz"))[2] if after_row and vector3(after_row.get("object_xyz")) else None,
            "object_specific_contact_pairs_recorded": False,
            "contact_force_recorded": False,
            "post_detach_writeback_count_recorded": False,
            "physical_loss_status": "insufficient_data",
            "physical_loss_reason": "cache_has_event_and_qpos-like oracle positions but no qvel/contact-pair/contact-force/writeback trace",
            "reference_contract_status": "legacy_reference_label_preserved",
        })
        sensor_audit.extend(_sensor_rows(meta, t_detach, radii[meta.root_family_id]))

        reference = build_reference({
            "path": str(meta.path), "rollout_id": meta.rollout_id, "root_family_id": meta.root_family_id,
            "case_id": meta.case_id, "requested_effect": meta.requested_effect,
        })
        reference_event = reference["events"][0] if reference.get("events") else {}
        for candidate_id in AUDIT_CANDIDATES:
            observations, evidence, decisions = _method_decisions(meta, candidate_id)
            selected = _first_emergency(decisions, start, end)
            historical_hold = any(row.get("historical_hold_established") for row in decisions if float(row.get("time", 0.0)) < t_detach - 1e-9)
            first_hold = next((row for row in decisions if row.get("historical_hold_established")), None)
            if historical_hold:
                method_stage_counts[candidate_id]["hold_established"] += 1
            else:
                method_stage_counts[candidate_id]["hold_not_established"] += 1
            if t_contact is not None:
                method_stage_counts[candidate_id]["contact_loss_observed"] += 1
            else:
                method_stage_counts[candidate_id]["contact_loss_missing"] += 1
            if selected is not None:
                method_stage_counts[candidate_id]["decision_any"] += 1
            if selected is not None and selected.get("selected_action") == "recover_object":
                method_stage_counts[candidate_id]["decision_correct"] += 1
            first_block = "physical_loss_verification_missing" if loss_audit[-1]["physical_loss_status"] != "verified" else "none"
            if first_block == "none" and t_contact is None:
                first_block = "contact_observable_missing"
            event_chain.append({
                "rollout_id": meta.rollout_id,
                "root_family_id": meta.root_family_id,
                "case_id": meta.case_id,
                "candidate_id": candidate_id,
                "source_path": str(meta.path.resolve()),
                "t_detach_command": t_detach,
                "t_loss_physical_verified": None,
                "t_contact_observable": t_contact,
                "t_visual_observable": visual_summary["t_visual_observable"],
                "historical_hold_established": historical_hold,
                "hold_evidence_first_time": first_hold.get("time") if first_hold else None,
                "t_decision_any": selected.get("time") if selected else None,
                "t_decision_correct": selected.get("time") if selected and selected.get("selected_action") == "recover_object" else None,
                "selected_action": selected.get("selected_action") if selected else "none",
                "last_passing_stage": "historical_hold_established" if historical_hold else "online_hold_evidence",
                "first_blocking_stage": first_block,
                "chain_conclusion": "insufficient_data",
                "reference_event_id": reference_event.get("event_id"),
                "reference_label_preserved": True,
            })
            timing_audit.append({
                "rollout_id": meta.rollout_id,
                "root_family_id": meta.root_family_id,
                "case_id": meta.case_id,
                "candidate_id": candidate_id,
                "t_detach_command": t_detach,
                "t_loss_physical_verified": None,
                "t_contact_observable": t_contact,
                "t_visual_observable": visual_summary["t_visual_observable"],
                "t_decision_any": selected.get("time") if selected else None,
                "t_decision_correct": None,
                "observation_end_time": max(float(row["time"]) for row in raw_obs),
                "contact_observable_status": contact_status,
                "visual_observable_status": visual_summary["visual_observable_status"],
                "delay_status": "physical_loss_not_verified",
                "null_reason": "physical_loss_not_verified; correct detection delay is not estimable",
                "delay_identity": "t_decision_correct - t_loss_verified = (t_observable - t_loss_verified) + (t_decision_correct - t_observable)",
            })

    # Add a compact source/contract record, including the known simulator
    # caveat that contact_present is a distance/attached proxy in this benchmark.
    source_contract = {
        "schema": "l2rar2_r11_source_contract_audit_v1",
        "online_allowed_inputs": [
            "adapted observation prefix", "contact_present", "gripper_command", "attempt lifecycle", "controller request provenance", "candidate hold evidence"
        ],
        "online_forbidden_inputs": ["case_id", "scenario", "events.jsonl", "oracle_timeline.csv", "weld_state", "future_outcome", "reference expected_action"],
        "offline_only_inputs": ["events.jsonl", "oracle_timeline.csv", "weld_state", "reference labels", "object qpos-like positions"],
        "contact_present_implementation_note": "DynamicTabletop.contact_sensor combines attached state and gripper/object distance; it is not an object-specific force sensor.",
        "audit_only_recomputed_sensor": "weld_state OR (closed AND distance(object, gripper + [0,0,-0.13]) < object_radius + 0.045 m)",
        "reference_thresholds_frozen": {"maximum_relative_position_drift_m": REFERENCE_DRIFT_LIMIT_M, "minimum_object_displacement_m": REFERENCE_MIN_DISPLACEMENT_M},
        "candidate_set_frozen": list(AUDIT_CANDIDATES),
        "reference_labels_rewritten": False,
        "online_oracle_used": False,
        "online_future_used": False,
        "audit_scope": "diagnostic regression; not a candidate selection or confirmation result",
    }

    # R11 counts are unique physical events, not method rows.
    unique_loss_count = len(loss_rollouts)
    physical_status_counts = Counter(row["physical_loss_status"] for row in loss_audit)
    route = {
        "schema": "l2rar2_r11_repair_route_decision_v1",
        "audit_status": "CACHE_AUDIT_COMPLETE_CACHE_INSUFFICIENT_FOR_PHYSICAL_LOSS_VERIFICATION",
        "unique_loss_events": unique_loss_count,
        "physical_loss_verified": physical_status_counts.get("verified", 0),
        "physical_loss_not_verified": physical_status_counts.get("not_verified", 0),
        "physical_loss_insufficient_data": physical_status_counts.get("insufficient_data", 0),
        "contact_observable_events": sum(row["t_contact_observable"] is not None for row in timing_audit[::2]),
        "visual_observable_events": sum(row["t_visual_observable"] is not None for row in timing_audit[::2]),
        "first_blocking_stage": "physical_loss_verification_missing",
        "next_repair_route": "INSUFFICIENT_EVIDENCE_STOP",
        "route_status": "PENDING_PHYSICAL_DIAGNOSTIC_REPLAY",
        "reason": "cached files lack object-specific contact pairs/forces, qvel, and post-detach writeback trace; physical replay is allowed only as bounded diagnostic evidence",
        "threshold_tuning": False,
        "candidate_selected": None,
        "confirmation": False,
        "l3_entry_allowed": False,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "event_chain_trace.csv", event_chain)
    write_csv(output_root / "loss_event_audit.csv", loss_audit)
    write_csv(output_root / "sensor_proxy_audit.csv", sensor_audit)
    write_csv(output_root / "visual_loss_evidence_audit.csv", visual_audit)
    write_csv(output_root / "time_audit_v2.csv", timing_audit)
    write_json(output_root / "source_contract_audit.json", source_contract)
    write_json(output_root / "repair_route_decision.json", route)
    write_json(output_root / "cache_audit_summary.json", {
        "schema": "l2rar2_r11_cache_audit_summary_v1",
        "status": "CACHE_AUDIT_COMPLETE",
        "rollouts_read": len(rollouts),
        "unique_loss_events": unique_loss_count,
        "contrast_rollouts": sum(item.case_id in CONTRAST_CASES for item in rollouts),
        "loss_event_status_counts": dict(sorted(physical_status_counts.items())),
        "method_stage_counts": {key: dict(sorted(value.items())) for key, value in method_stage_counts.items()},
        "baseline_replay_root": str(baseline_root.resolve()) if baseline_root else None,
        "raw_rgb_hashes_preserved": True,
        "oracle_or_future_used_online": False,
        "diagnostic_physical_executions": 0,
        "regression_metrics": regression,
    })
    return {
        "status": "CACHE_AUDIT_COMPLETE",
        "rollouts_read": len(rollouts),
        "unique_loss_events": unique_loss_count,
        "event_chain_rows": len(event_chain),
        "timing_rows": len(timing_audit),
        "cache_insufficient_for_physical_verification": True,
        "next_repair_route": route["next_repair_route"],
    }
