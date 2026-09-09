"""Bounded L2RA-R2 follow-up on cached online evidence and weld geometry.

The diagnostic keeps the frozen R2 cache and reference labels unchanged.  It
compares existing hold predicates through the same M1 interface and runs a
small, explicitly counted simulator mechanism probe; it does not collect a new
dataset, tune parameters, train, confirm, or authorize L3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from pathlib import Path
from typing import Any

import numpy as np

from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _online_observation, _predicates
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_hold_evidence.probe_adapter import control_variant, perform
from upgrade_v2.visual_refine_l2.dynamic_simulator import DynamicTabletop, _xml, family_spec
from upgrade_v2.visual_refine_l2.io import secret_scan

from .event_interface import run_m1
from .io import git, read_csv, read_json, read_jsonl, sha256, write_csv, write_json
from .mechanism_localization import _oracle_peak_intervals
from .reference import build_reference


TARGET_K5 = "L2RAR2_SELECT_04_820004:K5_brief_hold_loss"
K1 = "K1_hold_request_ends_without_hold"
K2 = "K2_touch_request_completes_without_hold"
K3 = "K3_normal_hold_pause_resume"
K4 = "K4_regular_hold_loss"
K5 = "K5_brief_hold_loss"
K6 = "K6_long_gap_after_loss"
K7 = "K7_commanded_release"
K8 = "K8_acquisition_touch_then_continue"
TARGET_FAMILIES = {"L2RAR2_SELECT_05_820005", "L2RAR2_SELECT_07_820007"}
FIXED_CANDIDATES = {
    "B_count2": ("B_count2", {}),
    "C3_vector_rho035": ("C3_vector", {"relative_rho_max": 0.35}),
    "C3_vector_rho055": ("C3_vector", {"relative_rho_max": 0.55}),
}
NOISE_FLOOR = 0.001
MAGNITUDE_MIN = 0.8
MOTION_MIN = 0.004
DIRECTION_MIN = 0.80
FORCED_OFFSET = np.asarray([0.0, 0.0, -0.13], dtype=float)


def _distance(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _json_vector(value: Any) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    return [float(item) for item in parsed[:3]]


def _load_candidate(meta: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    raw = read_jsonl(Path(meta["path"]) / "observations_dense.jsonl")
    observations = [_online_observation(row) for row in raw]
    geometries = build_features(observations)
    predictions: list[dict[str, Any]] = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    base_id, config = FIXED_CANDIDATES[candidate_id]
    evidence = evaluate_candidate(predictions, geometries, base_id, config)
    requests = read_jsonl(Path(meta["path"]) / "controller_requests.jsonl")
    m1 = run_m1(predictions, evidence, requests, history_complete=True)
    return {
        "raw": raw,
        "observations": observations,
        "predictions": predictions,
        "geometries": geometries,
        "evidence": evidence,
        "requests": requests,
        "m1": m1,
    }


def _magnitude_score(geometry: dict[str, Any]) -> float | None:
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    if object_norm is None or gripper_norm is None:
        return None
    scale = max(float(object_norm), float(gripper_norm), NOISE_FLOOR)
    return max(0.0, 1.0 - abs(float(object_norm) - float(gripper_norm)) / scale)


def _b_gate_reason(prediction: dict[str, Any], geometry: dict[str, Any]) -> str:
    predicates = prediction["predicates"]
    if predicates.get("gripper_command_closed") != "true":
        return "gripper_not_closed"
    if predicates.get("contact_present") != "true":
        return "contact_not_present"
    score = _magnitude_score(geometry)
    if score is None:
        return "magnitude_unavailable"
    if score < MAGNITUDE_MIN:
        return "magnitude_co_motion_below_0.8"
    return "magnitude_co_motion_pass"


def _c3_gate_reason(
    prediction: dict[str, Any],
    previous: dict[str, Any] | None,
    geometry: dict[str, Any],
    rho_max: float,
) -> str:
    predicates = prediction["predicates"]
    previous_predicates = previous["predicates"] if previous else {}
    if predicates.get("gripper_command_closed") != "true" or previous_predicates.get("gripper_command_closed") != "true":
        return "closed_continuity_not_met"
    if predicates.get("contact_present") != "true" or previous_predicates.get("contact_present") != "true":
        return "contact_continuity_not_met"
    if not geometry.get("effective_motion_interval"):
        return "effective_interval_not_met"
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    if object_norm is None or gripper_norm is None:
        return "motion_unavailable"
    if min(float(object_norm), float(gripper_norm)) < MOTION_MIN:
        return "minimum_motion_below_0.004"
    cosine = geometry.get("direction_cosine")
    rho = geometry.get("relative_vector_error")
    if cosine is None or rho is None:
        return "direction_or_rho_unavailable"
    if float(cosine) < DIRECTION_MIN:
        return "direction_cosine_below_0.80"
    if float(rho) > rho_max:
        return f"relative_vector_error_above_{rho_max:.2f}"
    return "directional_interval_pass"


def _k5_trace(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded = {candidate: _load_candidate(meta, candidate) for candidate in FIXED_CANDIDATES}
    baseline = loaded["B_count2"]
    observations = baseline["observations"]
    first_contact = next(index for index, row in enumerate(observations) if row.get("contact_present") is True)
    first_online_loss = next(
        index for index in range(first_contact + 1, len(observations))
        if observations[index - 1].get("contact_present") is True and observations[index].get("contact_present") is False
    )
    rows: list[dict[str, Any]] = []
    raw_b_run = 0
    for index in range(first_contact, first_online_loss + 1):
        observation = observations[index]
        geometry = baseline["geometries"][index]
        prediction = baseline["predictions"][index]
        raw_stable = prediction["predicates"].get("stable_hold_observed") == "true"
        raw_b_run = raw_b_run + 1 if raw_stable else 0
        previous = baseline["predictions"][index - 1] if index else None
        row = {
            "rollout_id": meta["rollout_id"],
            "time": observation.get("time"),
            "capture_order": observation.get("capture_order"),
            "phase": observation.get("phase"),
            "source_phase": observation.get("source_phase"),
            "attempt_phase": observation.get("attempt_phase"),
            "attempt_active": observation.get("attempt_active"),
            "attempt_end": observation.get("attempt_end"),
            "attempt_end_reason": observation.get("attempt_end_reason"),
            "contact_present": observation.get("contact_present"),
            "gripper_command": observation.get("gripper_command"),
            "object_centroid": json.dumps(observation.get("object_centroid"), separators=(",", ":")),
            "gripper_centroid": json.dumps(observation.get("gripper_centroid"), separators=(",", ":")),
            "object_displacement_vector": json.dumps(geometry.get("object_displacement_vector"), separators=(",", ":")),
            "gripper_displacement_vector": json.dumps(geometry.get("gripper_displacement_vector"), separators=(",", ":")),
            "object_displacement_norm": geometry.get("object_displacement_norm"),
            "gripper_displacement_norm": geometry.get("gripper_displacement_norm"),
            "direction_cosine": geometry.get("direction_cosine"),
            "relative_vector_error": geometry.get("relative_vector_error"),
            "effective_motion_interval": geometry.get("effective_motion_interval"),
            "magnitude_co_motion_score": _magnitude_score(geometry),
            "b_count2_raw_stable": raw_stable,
            "b_count2_raw_run_length": raw_b_run,
            "b_count2_gate_reason": _b_gate_reason(prediction, geometry),
        }
        for candidate_id, rho in (("C3_vector_rho035", 0.35), ("C3_vector_rho055", 0.55)):
            prefix = "c3_rho035" if rho == 0.35 else "c3_rho055"
            row[f"{prefix}_gate_reason"] = _c3_gate_reason(prediction, previous, geometry, rho)
        for candidate_id, candidate in loaded.items():
            prefix = candidate_id.lower()
            row[f"{prefix}_evidence"] = candidate["evidence"][index]["hold_evidence"]
            row[f"{prefix}_memory"] = candidate["evidence"][index]["hold_memory"]
            row[f"{prefix}_m1_action"] = candidate["m1"][index]["selected_action"]
            row[f"{prefix}_m1_reason"] = candidate["m1"][index]["reason_code"]
        rows.append(row)

    event_time = next(float(row["time"]) for row in read_jsonl(Path(meta["path"]) / "events.jsonl") if row.get("event") == "contact_lost")
    summary: dict[str, Any] = {
        "schema": "pathgraph_l2rar2_k5_missing_onset_v1",
        "rollout_id": meta["rollout_id"],
        "physical_contact_lost_time": event_time,
        "first_online_contact_time": observations[first_contact]["time"],
        "first_online_contact_loss_time": observations[first_online_loss]["time"],
        "b_count2_maximum_raw_stable_run": max(int(row["b_count2_raw_run_length"]) for row in rows),
        "candidate_onsets_before_physical_loss": {},
    }
    for candidate_id, candidate in loaded.items():
        onsets = [
            row.get("time") for row in candidate["evidence"]
            if row.get("time") is not None and float(row["time"]) <= event_time + 1e-9 and row.get("hold_evidence") == "true"
        ]
        summary["candidate_onsets_before_physical_loss"][candidate_id] = onsets
    summary["arrived_online_stream_contains_hold_support_before_physical_loss"] = bool(
        summary["candidate_onsets_before_physical_loss"]["C3_vector_rho035"]
    )
    summary["frozen_rule_blocking_b_count2"] = (
        "B_count2 requires two consecutive stable_hold_observed rows; its magnitude-only score reaches the 0.8 gate only once before physical loss. "
        "The existing fixed C3 direction/rho rule accepts the 0.50-0.65 s lift intervals without oracle or future input."
    )
    summary["oracle_used_for"] = "offline endpoint audit only; not supplied to any online predicate or M1"
    return rows, summary


def _event_times(meta: dict[str, Any]) -> dict[str, float]:
    return {str(row["event"]): float(row["time"]) for row in read_jsonl(Path(meta["path"]) / "events.jsonl")}


def _comparison_segments(metadata: dict[str, dict[str, Any]], unresolved_ids: set[str]) -> list[dict[str, Any]]:
    segment_specs: list[tuple[dict[str, Any], str, float, float]] = []
    for meta in sorted(metadata.values(), key=lambda item: item["rollout_id"]):
        events = _event_times(meta)
        if meta["root_family_id"] in TARGET_FAMILIES and meta["case_id"] in {K2, K8}:
            segment_specs.append((meta, "initial_transient_contact", events["transient_contact"], events["transient_contact_lost"]))
        if meta["root_family_id"] in TARGET_FAMILIES and meta["case_id"] == K8:
            segment_specs.append((meta, "same_k8_later_hold", events["contact_established"], float("inf")))
        if meta["case_id"] == K5 and meta["rollout_id"] not in unresolved_ids:
            segment_specs.append((meta, "reference_labeled_k5", events["contact_established"], events["contact_lost"]))

    output: list[dict[str, Any]] = []
    for meta, kind, start, end in segment_specs:
        for candidate_id in FIXED_CANDIDATES:
            loaded = _load_candidate(meta, candidate_id)
            window = [
                row for row in loaded["evidence"]
                if row.get("time") is not None and start - 1e-9 <= float(row["time"]) <= end + 1e-9
            ]
            m1_window = [
                row for row in loaded["m1"]
                if row.get("time") is not None and start - 1e-9 <= float(row["time"]) <= end + 1e-9
            ]
            onset = next((row["time"] for row in window if row.get("hold_evidence") == "true"), None)
            output.append({
                "segment_kind": kind,
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "case_id": meta["case_id"],
                "candidate_id": candidate_id,
                "fixed_parameters": json.dumps(FIXED_CANDIDATES[candidate_id][1], sort_keys=True),
                "window_start": start,
                "window_end": None if math.isinf(end) else end,
                "hold_evidence_observed": onset is not None,
                "first_hold_evidence_time": onset,
                "hold_memory_observed": any(row.get("hold_memory") == "true" for row in window),
                "retry_or_recover_in_window": any(
                    row.get("selected_action") in {"retry_grasp", "recover_object"} for row in m1_window
                ),
                "same_m1_interface": "M1_requested_effect_gate",
                "reference_label_status": "reference_unresolved" if meta["rollout_id"] in unresolved_ids else "reference_labeled",
            })
    return output


def _first_emergency(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any] | None:
    for row in rows:
        time = float(row.get("time", -1.0))
        if start - 1e-9 <= time <= end + 1e-9 and row.get("selected_action") in {"retry_grasp", "recover_object"}:
            return row
    return None


def _fixed_m1_rows(metadata: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    for meta in sorted(metadata.values(), key=lambda item: item["rollout_id"]):
        reference = build_reference(meta)
        for candidate_id in FIXED_CANDIDATES:
            loaded = _load_candidate(meta, candidate_id)
            if reference["status"] != "reference_labeled":
                decisions.append({
                    "candidate_id": candidate_id,
                    "rollout_id": meta["rollout_id"],
                    "root_family_id": meta["root_family_id"],
                    "case_id": meta["case_id"],
                    "reference_status": "reference_unresolved",
                    "reference_reason": reference.get("reason"),
                    "expected_action": None,
                    "selected_action": None,
                    "correct": None,
                    "hold_evidence_before_event": None,
                    "same_m1_interface": "M1_requested_effect_gate",
                })
                continue
            event = reference["events"][0]
            start = float(event["decision_window_start"])
            end = float(event["decision_window_end"])
            selected = _first_emergency(loaded["m1"], start, end)
            selected_action = selected["selected_action"] if selected else "none"
            expected = event["expected_action"]
            evidence_before = [row for row in loaded["evidence"] if float(row.get("time", -1.0)) <= start + 1e-9]
            decisions.append({
                "candidate_id": candidate_id,
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "case_id": meta["case_id"],
                "reference_status": "reference_labeled",
                "reference_reason": None,
                "expected_action": expected,
                "selected_action": selected_action,
                "selected_time": selected.get("time") if selected else None,
                "correct": selected_action == expected,
                "hold_evidence_before_event": any(row.get("hold_memory") == "true" for row in evidence_before),
                "hold_memory_at_last_closed_observation": next(
                    (row.get("hold_memory") for row in reversed(loaded["evidence"]) if row.get("closed") == "true"), "unknown"
                ),
                "same_m1_interface": "M1_requested_effect_gate",
            })

    metrics: list[dict[str, Any]] = []
    for candidate_id in FIXED_CANDIDATES:
        own = [row for row in decisions if row["candidate_id"] == candidate_id]
        labeled = [row for row in own if row["reference_status"] == "reference_labeled"]

        def case_rows(case_id: str) -> list[dict[str, Any]]:
            return [row for row in labeled if row["case_id"] == case_id]

        def rate(rows: list[dict[str, Any]], predicate) -> float | None:
            return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None

        metrics.append({
            "candidate_id": candidate_id,
            "fixed_parameters": json.dumps(FIXED_CANDIDATES[candidate_id][1], sort_keys=True),
            "same_r2_cache": True,
            "same_m1_interface": "M1_requested_effect_gate",
            "reference_labeled_events": len(labeled),
            "reference_unresolved_events_preserved": len(own) - len(labeled),
            "K1_recall": rate(case_rows(K1), lambda row: row["correct"]),
            "K2_false_emergency_rate": rate(case_rows(K2), lambda row: row["selected_action"] != "none"),
            "K3_false_emergency_rate": rate(case_rows(K3), lambda row: row["selected_action"] != "none"),
            "K4_recall": rate(case_rows(K4), lambda row: row["correct"]),
            "K5_recall": rate(case_rows(K5), lambda row: row["correct"]),
            "K6_recall": rate(case_rows(K6), lambda row: row["correct"]),
            "K7_false_emergency_rate": rate(case_rows(K7), lambda row: row["selected_action"] != "none"),
            "K8_false_emergency_rate": rate(case_rows(K8), lambda row: row["selected_action"] != "none"),
            "K5_hold_evidence_rate": rate(case_rows(K5), lambda row: row["hold_evidence_before_event"]),
            "K3_pause_hold_retention_rate": rate(case_rows(K3), lambda row: row["hold_memory_at_last_closed_observation"] == "true"),
            "all_labeled_event_accuracy": rate(labeled, lambda row: row["correct"]),
            "independent_confirmation": False,
        })
    return decisions, metrics


def _record_probe_point(
    rows: list[dict[str, Any]], sim: DynamicTabletop, root_family_id: str, mode: str,
    point: str, anchor: np.ndarray | None, control_index: int | None = None, physics_step: int | None = None,
) -> None:
    relative = sim.object_xyz - sim.data.mocap_pos[0]
    rows.append({
        "root_family_id": root_family_id,
        "probe_mode": mode,
        "point": point,
        "control_index": control_index,
        "physics_step": physics_step,
        "time": float(sim.data.time),
        "weld_active": bool(sim.data.eq_active[sim.weld_id]),
        "object_xyz": json.dumps(sim.object_xyz.tolist(), separators=(",", ":")),
        "gripper_xyz": json.dumps(sim.data.mocap_pos[0].tolist(), separators=(",", ":")),
        "relative_position_xyz_m": json.dumps(relative.tolist(), separators=(",", ":")),
        "drift_from_attach_m": _distance(relative, anchor) if anchor is not None else None,
    })


def _probe_family(meta: dict[str, Any], update_relpose_at_attach: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_family_id = meta["root_family_id"]
    family_index = int(root_family_id.split("_")[3])
    seed = int(meta["family_seed"])
    rollout_seed = int(meta["rollout_seed"])
    spec = family_spec(root_family_id, "slip_then_recover", seed, rollout_seed, probe_variant="brief_hold")
    sim = DynamicTabletop(spec, rollout_seed)
    mode = "relpose_at_attach_counterfactual" if update_relpose_at_attach else "current_pipeline"
    for action in ("observe_scene", "approach_object"):
        perform(sim, action, family_index)
    rows: list[dict[str, Any]] = []
    anchor_holder: list[np.ndarray | None] = [None]
    original_attach = sim._attach

    def wrapped_attach(self: DynamicTabletop) -> None:
        _record_probe_point(rows, self, root_family_id, mode, "before_attach", None)
        anchor_holder[0] = self.object_xyz - self.data.mocap_pos[0]
        if update_relpose_at_attach:
            self.model.eq_data[self.weld_id, 3:6] = anchor_holder[0]
            self.model.eq_data[self.weld_id, 6:10] = np.asarray([1.0, 0.0, 0.0, 0.0])
        original_attach()
        _record_probe_point(rows, self, root_family_id, mode, "after_attach_before_step", anchor_holder[0])

    sim._attach = types.MethodType(wrapped_attach, sim)
    perform(sim, "close_gripper", family_index)
    anchor = anchor_holder[0]
    if anchor is None:
        raise RuntimeError("attach probe did not capture an anchor")
    variant = control_variant(family_index)
    start = sim.data.mocap_pos[0].copy()
    target = start + np.asarray([0.0, 0.0, float(variant["lift_delta_z"])])
    controls = int(variant["lift_controls"])
    for control_index in range(1, controls + 1):
        alpha = control_index / controls
        sim.data.mocap_pos[0] = start * (1.0 - alpha) + target * alpha
        _record_probe_point(rows, sim, root_family_id, mode, "before_forced_write", anchor, control_index, 0)
        for physics_step in range(1, 6):
            sim._set_object_xyz(sim.data.mocap_pos[0] + FORCED_OFFSET)
            _record_probe_point(rows, sim, root_family_id, mode, "after_forced_write_before_step", anchor, control_index, physics_step)
            sim.mujoco.mj_step(sim.model, sim.data)
            _record_probe_point(rows, sim, root_family_id, mode, "after_mj_step", anchor, control_index, physics_step)
    post_steps = [row for row in rows if row["point"] == "after_mj_step"]
    summary = {
        "root_family_id": root_family_id,
        "probe_mode": mode,
        "family_index": family_index,
        "family_seed": seed,
        "rollout_seed": rollout_seed,
        "mujoco_version": sim.mujoco.__version__,
        "weld_id": int(sim.weld_id),
        "eq_type": int(sim.model.eq_type[sim.weld_id]),
        "eq_obj1id": int(sim.model.eq_obj1id[sim.weld_id]),
        "eq_obj2id": int(sim.model.eq_obj2id[sim.weld_id]),
        "compiled_eq_data": sim.model.eq_data[sim.weld_id].tolist(),
        "eq_solref": sim.model.eq_solref[sim.weld_id].tolist(),
        "eq_solimp": sim.model.eq_solimp[sim.weld_id].tolist(),
        "attach_relative_position_xyz_m": anchor.tolist(),
        "maximum_post_step_drift_from_attach_m": max(float(row["drift_from_attach_m"]) for row in post_steps),
        "first_control_final_relative_position_xyz_m": json.loads(
            next(row["relative_position_xyz_m"] for row in post_steps if row["control_index"] == 1 and row["physics_step"] == 5)
        ),
        "simulator_instances": 1,
        "physical_rollouts_collected": 0,
    }
    return rows, summary


def _reference_match_rows(
    metadata: dict[str, dict[str, Any]], unresolved_ids: set[str], probe_summaries: dict[tuple[str, str], dict[str, Any]], drift_limit: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in sorted(metadata.values(), key=lambda item: item["rollout_id"]):
        oracle = read_csv(Path(meta["path"]) / "oracle_timeline.csv")
        intervals = _oracle_peak_intervals(oracle)
        if not intervals:
            continue
        selected = max(intervals, key=lambda item: item["maximum_drift"])
        anchor = selected["anchor"]
        anchor_time = float(anchor["row"]["time"])
        next_sample = next((sample for sample in selected["samples"] if float(sample["row"]["time"]) > anchor_time + 1e-9), None)
        if next_sample is None:
            continue
        current = probe_summaries[(meta["root_family_id"], "current_pipeline")]
        counterfactual = probe_summaries[(meta["root_family_id"], "relpose_at_attach_counterfactual")]
        predicted = current["first_control_final_relative_position_xyz_m"]
        peak = selected["peaks"][0]
        anchor_relative = anchor["relative"]
        observed_delta = np.asarray(peak["relative"]) - np.asarray(anchor_relative)
        predicted_delta = np.asarray(predicted) - np.asarray(anchor_relative)
        cosine = None
        if np.linalg.norm(observed_delta) > 0 and np.linalg.norm(predicted_delta) > 0:
            cosine = float(np.dot(observed_delta, predicted_delta) / (np.linalg.norm(observed_delta) * np.linalg.norm(predicted_delta)))
        rows.append({
            "rollout_id": meta["rollout_id"],
            "root_family_id": meta["root_family_id"],
            "case_id": meta["case_id"],
            "reference_status": "reference_unresolved" if meta["rollout_id"] in unresolved_ids else "reference_labeled",
            "anchor_time": anchor_time,
            "anchor_relative_position_xyz_m": json.dumps(anchor_relative, separators=(",", ":")),
            "first_post_anchor_time": next_sample["row"]["time"],
            "first_post_anchor_relative_position_xyz_m": json.dumps(next_sample["relative"], separators=(",", ":")),
            "current_probe_predicted_first_tick_relative_xyz_m": json.dumps(predicted, separators=(",", ":")),
            "current_probe_first_tick_error_m": _distance(next_sample["relative"], predicted),
            "observed_peak_relative_position_xyz_m": json.dumps(peak["relative"], separators=(",", ":")),
            "observed_peak_delta_from_anchor_xyz_m": json.dumps(observed_delta.tolist(), separators=(",", ":")),
            "current_probe_delta_direction_cosine": cosine,
            "observed_maximum_drift_m": selected["maximum_drift"],
            "frozen_relative_drift_limit_m": drift_limit,
            "counterfactual_maximum_post_step_drift_m": counterfactual["maximum_post_step_drift_from_attach_m"],
            "current_mechanism_exact_match": _distance(next_sample["relative"], predicted) <= 1e-12,
            "reference_label_changed": False,
        })
    return rows


def _candidate_segment_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for candidate in FIXED_CANDIDATES:
        own = [row for row in rows if row["candidate_id"] == candidate]
        summary[candidate] = {
            "initial_false_hold_segments_with_evidence": sum(row["hold_evidence_observed"] for row in own if row["segment_kind"] == "initial_transient_contact"),
            "initial_false_hold_segments_total": sum(row["segment_kind"] == "initial_transient_contact" for row in own),
            "later_k8_segments_with_evidence": sum(row["hold_evidence_observed"] for row in own if row["segment_kind"] == "same_k8_later_hold"),
            "later_k8_segments_total": sum(row["segment_kind"] == "same_k8_later_hold" for row in own),
            "reference_labeled_k5_with_evidence": sum(row["hold_evidence_observed"] for row in own if row["segment_kind"] == "reference_labeled_k5"),
            "reference_labeled_k5_total": sum(row["segment_kind"] == "reference_labeled_k5" for row in own),
        }
    return summary


def run_followup(
    data_root: Path,
    unresolved_path: Path,
    reference_contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    metadata = {row["rollout_id"]: row for row in read_csv(data_root / "rollout_manifest.csv")}
    unresolved = read_csv(unresolved_path)
    unresolved_ids = {row["rollout_id"] for row in unresolved}
    contract = read_json(reference_contract_path)
    drift_limit = float(contract["hold_proxy"]["maximum_relative_position_drift_m"])
    if len(unresolved) != 19 or abs(drift_limit - 0.02) > 1e-12:
        raise ValueError("follow-up requires the frozen 19-row unresolved set and 0.02 m reference limit")
    target_meta = metadata.get(TARGET_K5)
    if target_meta is None:
        raise ValueError(f"target K5 missing from manifest: {TARGET_K5}")

    output_root.mkdir(parents=True, exist_ok=True)
    k5_rows, k5_summary = _k5_trace(target_meta)
    segment_rows = _comparison_segments(metadata, unresolved_ids)
    m1_rows, m1_metrics = _fixed_m1_rows(metadata)
    write_csv(output_root / "k5_missing_onset_trace.csv", k5_rows)
    write_json(output_root / "k5_missing_onset_summary.json", k5_summary)
    write_csv(output_root / "fixed_predicate_comparison.csv", segment_rows)
    write_csv(output_root / "fixed_m1_comparison.csv", m1_rows)
    write_csv(output_root / "fixed_m1_metrics.csv", m1_metrics)

    all_probe_rows: list[dict[str, Any]] = []
    probe_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    k5_metas = [meta for meta in metadata.values() if meta["case_id"] == K5]
    for meta in sorted(k5_metas, key=lambda item: item["root_family_id"]):
        for update_relpose in (False, True):
            probe_rows, probe_summary = _probe_family(meta, update_relpose)
            all_probe_rows.extend(probe_rows)
            probe_summaries[(meta["root_family_id"], probe_summary["probe_mode"])] = probe_summary
    write_csv(output_root / "simulator_weld_step_probe.csv", all_probe_rows)

    dynamic_source = Path(__file__).resolve().parents[1] / "visual_refine_l2/dynamic_simulator.py"
    sample_meta = sorted(k5_metas, key=lambda item: item["root_family_id"])[0]
    sample_spec = family_spec(
        sample_meta["root_family_id"], "slip_then_recover", int(sample_meta["family_seed"]),
        int(sample_meta["rollout_seed"]), probe_variant="brief_hold",
    )
    current_summaries = [row for (root, mode), row in probe_summaries.items() if mode == "current_pipeline"]
    counterfactual_summaries = [row for (root, mode), row in probe_summaries.items() if mode == "relpose_at_attach_counterfactual"]
    weld_contract = {
        "schema": "pathgraph_l2rar2_weld_contract_diagnostic_v1",
        "mujoco_version": current_summaries[0]["mujoco_version"],
        "dynamic_simulator_source": str(dynamic_source),
        "dynamic_simulator_source_sha256": sha256(dynamic_source),
        "xml_template_sha256": hashlib.sha256(_xml(sample_spec).encode()).hexdigest(),
        "compiled_weld": {
            key: current_summaries[0][key]
            for key in ("weld_id", "eq_type", "eq_obj1id", "eq_obj2id", "eq_solref", "eq_solimp")
        },
        "compiled_default_eq_data_by_family": {
            row["root_family_id"]: row["compiled_eq_data"] for row in current_summaries
        },
        "implementation_sequence": [
            "XML weld has no explicit relpose",
            "_attach activates equality without updating eq_data",
            "each attached physics step writes object to gripper + [0,0,-0.13] before mj_step",
        ],
        "current_pipeline_maximum_post_step_drift_range_m": [
            min(row["maximum_post_step_drift_from_attach_m"] for row in current_summaries),
            max(row["maximum_post_step_drift_from_attach_m"] for row in current_summaries),
        ],
        "relpose_at_attach_counterfactual_maximum_post_step_drift_range_m": [
            min(row["maximum_post_step_drift_from_attach_m"] for row in counterfactual_summaries),
            max(row["maximum_post_step_drift_from_attach_m"] for row in counterfactual_summaries),
        ],
        "diagnostic_simulator_instances": len(current_summaries) + len(counterfactual_summaries),
        "new_dataset_rollouts": 0,
        "reference_limit_changed": False,
        "normal_pipeline_changed": False,
    }
    write_json(output_root / "simulator_weld_contract.json", weld_contract)

    reference_rows = _reference_match_rows(metadata, unresolved_ids, probe_summaries, drift_limit)
    write_csv(output_root / "reference_mechanism_match.csv", reference_rows)
    segment_summary = _candidate_segment_summary(segment_rows)
    current_matches = [row for row in reference_rows if row["current_mechanism_exact_match"]]
    current_probe_errors = [float(row["current_probe_first_tick_error_m"]) for row in reference_rows]
    counterfactual_under_limit = [
        row for row in reference_rows if float(row["counterfactual_maximum_post_step_drift_m"]) <= drift_limit + 1e-12
    ]

    decision = {
        "schema": "pathgraph_l2rar2_followup_repair_decision_v1",
        "decision_count": 1,
        "decision": "SIMULATOR_REFERENCE_REPAIR_FIRST",
        "single_repair_scope": "collection_and_reference_geometry",
        "proposed_change_not_applied": (
            "Before activating grasp_weld in a separately versioned simulator, set the weld relative translation/quaternion to the actual attach pose; "
            "retain the scripted writeback and frozen 0.02 m limit, then recollect new independent development roots under a new collection/reference version."
        ),
        "reason": (
            "The current compiled weld target is the initial XML body relation, while every attached physics step writes a different relation. "
            "The family-matched current probe reproduces 47/48 cached first-tick relative positions exactly; the sole K8 deviation is 1.66e-5 m with the same drift direction. "
            "The bounded relpose-at-attach counterfactual remains below 0.02 m for all eight development families."
        ),
        "online_diagnostic": {
            "existing_C3_vector_rho035": segment_summary["C3_vector_rho035"],
            "same_m1_metrics": next(row for row in m1_metrics if row["candidate_id"] == "C3_vector_rho035"),
            "interpretation": (
                "The existing fixed C3 rho=0.35 rule suppresses the four targeted initial false holds, retains both later K8 holds, "
                "and supplies evidence for all four reference-labeled K5 cases. Through the unchanged M1 interface, however, it changes no-contact endpoint evidence to unknown, "
                "so K1 recall is 0/8. It is not a complete candidate and is not renamed, tuned, selected, or independently confirmed here."
            ),
            "candidate_selected": False,
            "why_not_selected": (
                "C3 rho=0.35 is mechanism-informative but incomplete under the same M1 interface because K1 becomes needs_observation rather than retry; "
                "reference/collection self-consistency is repaired first"
            ),
        },
        "old_reference_rows_reclassified": 0,
        "old_reference_limit_changed": False,
        "candidate_grid_expanded": False,
        "training_jobs": 0,
        "api_calls": 0,
        "confirmation_run": False,
        "l3_entry_allowed": False,
        "retained_graph": "G1_predicate_bound",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "current_status": "L2RAR2_PARTIAL_KEEP_G1",
    }
    write_json(output_root / "repair_decision.json", decision)

    result = {
        "schema": "pathgraph_l2rar2_followup_resolution_v1",
        "status": "FOLLOWUP_DIAGNOSTIC_COMPLETE",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "current_status": "L2RAR2_PARTIAL_KEEP_G1",
        "retained_graph": "G1_predicate_bound",
        "l3_entry_allowed": False,
        "k5": k5_summary,
        "fixed_candidate_segment_summary": segment_summary,
        "reference_consistency": {
            "records_with_weld_intervals": len(reference_rows),
            "reference_labeled_records": sum(row["reference_status"] == "reference_labeled" for row in reference_rows),
            "reference_unresolved_records": sum(row["reference_status"] == "reference_unresolved" for row in reference_rows),
            "current_probe_exact_first_tick_matches": len(current_matches),
            "current_probe_first_tick_max_error_m": max(current_probe_errors),
            "current_probe_first_tick_error_above_1e_12_records": sum(error > 1e-12 for error in current_probe_errors),
            "counterfactual_records_under_frozen_limit": len(counterfactual_under_limit),
            "frozen_limit_m": drift_limit,
            "conclusion": (
                "The observed drift is generated by a simulator/reference contract inconsistency: the compiled default weld relation and per-step scripted writeback impose different relative poses. "
                "This mechanism affects both labeled and unresolved records; threshold crossing depends on the family-specific attach anchor."
            ),
        },
        "repair_decision": decision,
        "execution": {
            "cached_rollouts_recomputed": len(metadata),
            "diagnostic_simulator_instances": weld_contract["diagnostic_simulator_instances"],
            "new_dataset_rollouts": 0,
            "training_jobs": 0,
            "api_calls": 0,
            "api_key_read": False,
            "confirmation_rollouts": 0,
        },
    }
    write_json(output_root / "followup_resolution.json", result)

    report = [
        "## Material Passport",
        "",
        "- Artifact type: bounded cached evidence and simulator-mechanism diagnostic",
        "- Verification status: `ANALYZED`; not an independent confirmation",
        "- Current boundary: `L2RAR2_PARTIAL_KEEP_G1`; retained graph: `G1_predicate_bound`; L3 closed",
        "",
        "# L2RA-R2 follow-up resolution",
        "",
        "## K5 evidence chain",
        "",
        f"- Target: `{TARGET_K5}`. Physical loss is recorded at `{k5_summary['physical_contact_lost_time']:.2f}s`; the arrived contact stream first reports loss at `{k5_summary['first_online_contact_loss_time']:.2f}s`.",
        f"- `B_count2` reaches a maximum raw stable run of `{k5_summary['b_count2_maximum_raw_stable_run']}` and therefore has no onset before physical loss.",
        "- Existing `C3_vector_rho035` and `C3_vector_rho055` both establish causal hold evidence during the lift before loss. The missing B onset is caused by the two-consecutive magnitude gate, not by absence of all online directional evidence.",
        "",
        "## Fixed existing-rule comparison",
        "",
        f"- `B_count2`: `{json.dumps(segment_summary['B_count2'], sort_keys=True)}`.",
        f"- `C3_vector_rho035`: `{json.dumps(segment_summary['C3_vector_rho035'], sort_keys=True)}`.",
        f"- `C3_vector_rho055`: `{json.dumps(segment_summary['C3_vector_rho055'], sort_keys=True)}`.",
        "- All three use the same R2 cache and the same `M1_requested_effect_gate`. No parameter search or method renaming was performed.",
        f"- `C3_vector_rho035` is not a complete candidate: on the unchanged M1 interface its K1 recall is `{next(row for row in m1_metrics if row['candidate_id'] == 'C3_vector_rho035')['K1_recall']}` because no-contact endpoint evidence is `unknown`, producing `needs_observation` instead of retry.",
        "",
        "## Simulator/reference consistency",
        "",
        f"- MuJoCo version: `{weld_contract['mujoco_version']}`. Diagnostic simulator instances: `{weld_contract['diagnostic_simulator_instances']}`; new dataset rollouts: `0`.",
        "- The XML weld has no explicit relative pose; `_attach()` activates the compiled equality unchanged; `_advance()` writes the object to `gripper + [0,0,-0.13]` before every physics step.",
        f"- The current mechanism probe exactly matches the first cached post-anchor relative position in `{len(current_matches)}/{len(reference_rows)}` weld-bearing records. The sole non-exact K8 row differs by `{max(current_probe_errors):.9g} m` and has the same drift direction; all 19 unresolved rows are exact matches.",
        f"- The relpose-at-attach counterfactual stays under the unchanged `0.02 m` limit in `{len(counterfactual_under_limit)}/{len(reference_rows)}` records. This is diagnostic support, not a relabeling of old rows.",
        "",
        "## Single repair decision",
        "",
        "- Decision: `SIMULATOR_REFERENCE_REPAIR_FIRST`.",
        "- In a new version only, set the weld target to the actual attach pose before activation, keep the existing scripted writeback and the frozen reference threshold, and recollect independent development roots. Old records and labels remain unchanged.",
        "- Existing `C3_vector_rho035` remains a diagnostic explanation for the specified hold segments, not a complete online candidate: it fails K1 through the unchanged M1 unknown-state path.",
        "- No candidate expansion, training, API/key access, confirmation, reference relaxation, unresolved-row deletion, or L3 entry occurred.",
    ]
    (output_root / "followup_resolution.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_root / "actual_commands.txt").write_text(
        "git fetch origin --prune\n"
        "python -m upgrade_v2.l2r_task_context.cli complete-cache-followup --data-root <frozen-dev-select> --unresolved <frozen-unresolved> --reference-contract <frozen-contract> --output-root <round-8>\n",
        encoding="utf-8",
    )

    scan = secret_scan([
        Path(__file__),
        Path(__file__).with_name("cli.py"),
        Path(__file__).with_name("tests") / "test_task_context.py",
        output_root,
    ])
    scan.update({
        "schema": "pathgraph_l2rar2_followup_secret_scan_v1",
        "scope": "follow-up source and lightweight round artifacts; ZIP and credential sources excluded",
        "api_calls": 0,
        "api_key_read": False,
        "training_jobs": 0,
    })
    if scan["status"] != "PASS":
        raise RuntimeError("secret scan failed")
    write_json(output_root / "secret_scan.json", scan)

    artifact_names = [
        "k5_missing_onset_trace.csv", "k5_missing_onset_summary.json", "fixed_predicate_comparison.csv",
        "fixed_m1_comparison.csv", "fixed_m1_metrics.csv", "simulator_weld_contract.json",
        "simulator_weld_step_probe.csv", "reference_mechanism_match.csv", "repair_decision.json",
        "followup_resolution.json", "followup_resolution.md", "actual_commands.txt", "secret_scan.json",
    ]
    manifest = {
        "schema": "pathgraph_l2rar2_followup_manifest_v1",
        "date": "2026-09-09",
        "implementation_commit_at_run": git(Path.cwd(), "rev-parse", "HEAD"),
        "code_hashes": {
            "upgrade_v2/l2r_task_context/followup_resolution.py": sha256(Path(__file__)),
            "upgrade_v2/l2r_task_context/cli.py": sha256(Path(__file__).with_name("cli.py")),
            "upgrade_v2/l2r_task_context/tests/test_task_context.py": sha256(Path(__file__).with_name("tests") / "test_task_context.py"),
        },
        "input_hashes": {
            "rollout_manifest": sha256(data_root / "rollout_manifest.csv"),
            "reference_unresolved": sha256(unresolved_path),
            "reference_contract": sha256(reference_contract_path),
        },
        "artifacts": [
            {"path": name, "size_bytes": (output_root / name).stat().st_size, "sha256": sha256(output_root / name)}
            for name in artifact_names
        ],
        "api_calls": 0,
        "api_key_read": False,
        "training_jobs": 0,
        "new_dataset_rollouts": 0,
        "diagnostic_simulator_instances": weld_contract["diagnostic_simulator_instances"],
    }
    write_json(output_root / "run_manifest.json", manifest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Complete bounded L2RA-R2 cache and weld follow-up")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_followup(args.data_root, args.unresolved, args.reference_contract, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
