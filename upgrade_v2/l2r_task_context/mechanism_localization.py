"""Cache-only localization of L2RA-R2 hold and reference mechanisms.

This module explains existing ``B_count2`` evidence and the frozen physical
reference failures.  It does not change an online predicate, tune a threshold,
relabel a reference event, collect a rollout, train a model, or run confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_task_context.evaluate import _online_observation, _predicates

from .io import read_csv, read_jsonl, sha256, write_csv, write_json


TARGET_FAMILIES = {
    "L2RAR2_SELECT_05_820005",
    "L2RAR2_SELECT_07_820007",
}
K2 = "K2_touch_request_completes_without_hold"
K5 = "K5_brief_hold_loss"
K8 = "K8_acquisition_touch_then_continue"
NOISE_FLOOR = 0.001
MIN_DIRECTIONAL_MOTION = 0.004
DIRECTION_MIN = 0.80
RELATIVE_RHO_MAX = 0.35
TOLERANCE = 1e-12
RELATIVE_POSITION_DEFINITION = (
    "relative_position_world_m = object_xyz_world_m - gripper_xyz_world_m; "
    "drift_m = L2(relative_position_world_m[t] - relative_position_world_m[first_weld_row])"
)


def _number(value: Any) -> float:
    return float(value)


def _vector(value: Any) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        raise ValueError("expected a three-dimensional vector")
    return [float(item) for item in parsed[:3]]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _same_centroids(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if previous is None:
        return False
    return (
        previous.get("object_centroid") == current.get("object_centroid")
        and previous.get("gripper_centroid") == current.get("gripper_centroid")
    )


def _magnitude_score(geometry: dict[str, Any]) -> float | None:
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    if object_norm is None or gripper_norm is None:
        return None
    scale = max(float(object_norm), float(gripper_norm), NOISE_FLOOR)
    return max(0.0, 1.0 - abs(float(object_norm) - float(gripper_norm)) / scale)


def _classify_interval(geometry: dict[str, Any], exact_repeat: bool) -> str:
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    if object_norm is None or gripper_norm is None:
        return "geometry_unavailable"
    low_motion = max(float(object_norm), float(gripper_norm)) <= NOISE_FLOOR
    if exact_repeat:
        return "exact_static_repeat_accepted_by_magnitude_score"
    if low_motion:
        return "sub_noise_motion_accepted_by_magnitude_score"
    cosine = geometry.get("direction_cosine")
    rho = geometry.get("relative_vector_error")
    if cosine is not None and float(cosine) < DIRECTION_MIN:
        return "similar_magnitude_but_direction_disagrees"
    if rho is not None and float(rho) > RELATIVE_RHO_MAX:
        return "similar_magnitude_but_relative_vector_disagrees"
    if (
        geometry.get("effective_motion_interval")
        and min(float(object_norm), float(gripper_norm)) >= MIN_DIRECTIONAL_MOTION
        and cosine is not None
        and float(cosine) >= DIRECTION_MIN
        and rho is not None
        and float(rho) <= RELATIVE_RHO_MAX
    ):
        return "directionally_consistent_effective_motion"
    return "magnitude_match_without_full_directional_support"


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
    return observations, geometries, evidence


def _raw_same_timestamp_phases(oracle_rows: list[dict[str, Any]], time: float) -> list[str]:
    return sorted({
        str(row.get("phase"))
        for row in oracle_rows
        if row.get("time") is not None and abs(_number(row["time"]) - time) <= 1e-9
    })


def _evidence_onsets(evidence: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(evidence)
        if row.get("hold_evidence") == "true"
        and (index == 0 or evidence[index - 1].get("hold_evidence") != "true")
    ]


def _maximum_stable_run(observations: list[dict[str, Any]], geometries: list[dict[str, Any]]) -> int:
    maximum = 0
    current = 0
    for observation, geometry in zip(observations, geometries):
        score = _magnitude_score(geometry)
        stable = (
            score is not None
            and score >= 0.8
            and observation.get("contact_present") is True
            and observation.get("gripper_command") == "closed"
        )
        current = current + 1 if stable else 0
        maximum = max(maximum, current)
    return maximum


def _interval_row(
    meta: dict[str, Any],
    observations: list[dict[str, Any]],
    geometries: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    index: int,
    segment_id: str,
    segment_kind: str,
    pair_position: int,
) -> dict[str, Any]:
    current = observations[index]
    previous = observations[index - 1] if index > 0 else None
    geometry = geometries[index]
    time = _number(current["time"])
    raw_phases = _raw_same_timestamp_phases(oracle_rows, time)
    object_norm = geometry.get("object_displacement_norm")
    gripper_norm = geometry.get("gripper_displacement_norm")
    exact_repeat = _same_centroids(previous, current)
    score = _magnitude_score(geometry)
    merged_same_time_count = sum(
        abs(_number(row["time"]) - time) <= 1e-9 for row in observations
    )
    return {
        "segment_id": segment_id,
        "segment_kind": segment_kind,
        "pair_position": pair_position,
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "time": time,
        "capture_order": current.get("capture_order"),
        "source_phase": current.get("source_phase", current.get("phase")),
        "previous_time": previous.get("time") if previous else None,
        "previous_capture_order": previous.get("capture_order") if previous else None,
        "contact_present": current.get("contact_present"),
        "gripper_command": current.get("gripper_command"),
        "attempt_phase": current.get("attempt_phase"),
        "attempt_active": current.get("attempt_active"),
        "stable_hold_observed": "true" if score is not None and score >= 0.8 and current.get("contact_present") is True and current.get("gripper_command") == "closed" else "false",
        "magnitude_co_motion_score": score,
        "object_displacement_norm": object_norm,
        "gripper_displacement_norm": gripper_norm,
        "max_displacement_norm": max(float(object_norm), float(gripper_norm)) if object_norm is not None and gripper_norm is not None else None,
        "min_displacement_norm": min(float(object_norm), float(gripper_norm)) if object_norm is not None and gripper_norm is not None else None,
        "effective_motion_interval": geometry.get("effective_motion_interval"),
        "direction_cosine": geometry.get("direction_cosine"),
        "relative_vector_error": geometry.get("relative_vector_error"),
        "both_below_noise_floor": bool(
            object_norm is not None
            and gripper_norm is not None
            and max(float(object_norm), float(gripper_norm)) <= NOISE_FLOOR
        ),
        "exact_visual_geometry_repeat_from_previous": exact_repeat,
        "direction_unavailable_due_low_motion": geometry.get("direction_cosine") is None,
        "direction_disagrees": bool(
            geometry.get("direction_cosine") is not None
            and float(geometry["direction_cosine"]) < DIRECTION_MIN
        ),
        "directional_interval_pass": bool(
            geometry.get("effective_motion_interval")
            and object_norm is not None
            and gripper_norm is not None
            and min(float(object_norm), float(gripper_norm)) >= MIN_DIRECTIONAL_MOTION
            and geometry.get("direction_cosine") is not None
            and float(geometry["direction_cosine"]) >= DIRECTION_MIN
            and geometry.get("relative_vector_error") is not None
            and float(geometry["relative_vector_error"]) <= RELATIVE_RHO_MAX
        ),
        "raw_same_timestamp_phases": ",".join(raw_phases),
        "raw_control_and_action_end_same_timestamp": set(raw_phases) == {"action_end", "control_tick"},
        "same_timestamp_row_count_in_merged_dense": merged_same_time_count,
        "same_timestamp_duplicate_counted_twice_by_bcount2": merged_same_time_count > 1,
        "mechanism_class": _classify_interval(geometry, exact_repeat),
        "online_feature_scope": "arrived dense observation history only; oracle phase audit is offline diagnostic metadata",
    }


def _select_segments(
    metadata: dict[str, dict[str, Any]], unresolved_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interval_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for meta in sorted(metadata.values(), key=lambda row: row["rollout_id"]):
        case_id = meta["case_id"]
        is_target = meta["root_family_id"] in TARGET_FAMILIES and case_id in {K2, K8}
        is_brief_positive = case_id == K5 and meta["rollout_id"] not in unresolved_ids
        if not (is_target or is_brief_positive):
            continue
        observations, geometries, evidence = _load_online(meta)
        oracle_rows = read_csv(Path(meta["path"]) / "oracle_timeline.csv")
        events = read_jsonl(Path(meta["path"]) / "events.jsonl")
        event_times = {str(row.get("event")): _number(row["time"]) for row in events if row.get("time") is not None}
        onsets = _evidence_onsets(evidence)
        selected: list[tuple[int, str]] = []
        if is_target and onsets:
            selected.append((onsets[0], "target_initial_false_hold"))
        if case_id == K8:
            loss_time = event_times.get("transient_contact_lost")
            later = next((index for index in onsets if loss_time is not None and _number(observations[index]["time"]) > loss_time + 1e-9), None)
            if later is not None:
                selected.append((later, "same_k8_later_hold"))
        if is_brief_positive and onsets:
            selected.append((onsets[0], "brief_hold_positive"))
        if is_brief_positive and not onsets:
            segment_rows.append({
                "segment_id": f"{meta['rollout_id']}:brief_hold_positive_no_bcount2_onset",
                "segment_kind": "brief_hold_positive_no_bcount2_onset",
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "case_id": case_id,
                "evidence_time": None,
                "evidence_capture_order": None,
                "contributing_intervals": 0,
                "mechanism_classes": json.dumps({"no_bcount2_evidence_onset": 1}, sort_keys=True),
                "sub_noise_interval_count": 0,
                "exact_static_repeat_count": 0,
                "direction_disagreement_count": 0,
                "direction_unavailable_count": 0,
                "directional_interval_pass_count": 0,
                "action_end_interval_count": 0,
                "raw_same_timestamp_control_action_end_count": 0,
                "merged_same_timestamp_double_count_count": 0,
                "minimum_pair_motion_norm": None,
                "maximum_pair_motion_norm": None,
                "maximum_stable_run_length": _maximum_stable_run(observations, geometries),
                "reference_label_status": "reference_labeled",
            })
        for onset_index, segment_kind in selected:
            if onset_index < 1:
                raise ValueError(f"B_count2 onset lacks two contributing rows: {meta['rollout_id']}")
            segment_id = f"{meta['rollout_id']}:{segment_kind}:{observations[onset_index]['capture_order']}"
            contributing = [
                _interval_row(meta, observations, geometries, oracle_rows, index, segment_id, segment_kind, pair_position)
                for pair_position, index in enumerate((onset_index - 1, onset_index), start=1)
            ]
            interval_rows.extend(contributing)
            classes = Counter(row["mechanism_class"] for row in contributing)
            segment_rows.append({
                "segment_id": segment_id,
                "segment_kind": segment_kind,
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "case_id": case_id,
                "evidence_time": observations[onset_index]["time"],
                "evidence_capture_order": observations[onset_index].get("capture_order"),
                "contributing_intervals": 2,
                "mechanism_classes": json.dumps(dict(sorted(classes.items())), sort_keys=True),
                "sub_noise_interval_count": sum(bool(row["both_below_noise_floor"]) for row in contributing),
                "exact_static_repeat_count": sum(bool(row["exact_visual_geometry_repeat_from_previous"]) for row in contributing),
                "direction_disagreement_count": sum(bool(row["direction_disagrees"]) for row in contributing),
                "direction_unavailable_count": sum(bool(row["direction_unavailable_due_low_motion"]) for row in contributing),
                "directional_interval_pass_count": sum(bool(row["directional_interval_pass"]) for row in contributing),
                "action_end_interval_count": sum(row["source_phase"] == "action_end" for row in contributing),
                "raw_same_timestamp_control_action_end_count": sum(bool(row["raw_control_and_action_end_same_timestamp"]) for row in contributing),
                "merged_same_timestamp_double_count_count": sum(bool(row["same_timestamp_duplicate_counted_twice_by_bcount2"]) for row in contributing),
                "minimum_pair_motion_norm": min(float(row["min_displacement_norm"]) for row in contributing if row["min_displacement_norm"] is not None),
                "maximum_pair_motion_norm": max(float(row["max_displacement_norm"]) for row in contributing if row["max_displacement_norm"] is not None),
                "maximum_stable_run_length": _maximum_stable_run(observations, geometries),
                "reference_label_status": "reference_labeled" if meta["rollout_id"] not in unresolved_ids else "reference_unresolved",
            })
    return interval_rows, segment_rows


def _action_candidates(actions: list[dict[str, Any]], time: float) -> list[str]:
    return [
        str(row["action"])
        for row in actions
        if row.get("start_time") not in (None, "")
        and row.get("end_time") not in (None, "")
        and _number(row["start_time"]) - 1e-9 <= time <= _number(row["end_time"]) + 1e-9
    ]


def _stage_rows(active: list[dict[str, Any]], followed_by_nonweld: bool) -> list[str]:
    distinct_times = sorted({_number(row["time"]) for row in active})
    establishment_end = distinct_times[min(1, len(distinct_times) - 1)]
    last_time = distinct_times[-1]
    stages = []
    for row in active:
        time = _number(row["time"])
        if time <= establishment_end + 1e-9:
            stages.append("constraint_establishment")
        elif followed_by_nonweld and abs(time - last_time) <= 1e-9:
            stages.append("loss_boundary")
        else:
            stages.append("sustained_hold")
    return stages


def _oracle_peak_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_number(row["time"]), int(row.get("capture_order", 0))))
    intervals: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []

    def finish(followed_by_nonweld: bool) -> None:
        if not active:
            return
        anchor_object = _vector(active[0]["object_xyz"])
        anchor_gripper = _vector(active[0]["gripper_xyz"])
        anchor_relative = [anchor_object[index] - anchor_gripper[index] for index in range(3)]
        stages = _stage_rows(active, followed_by_nonweld)
        samples = []
        for row, stage in zip(active, stages):
            object_xyz = _vector(row["object_xyz"])
            gripper_xyz = _vector(row["gripper_xyz"])
            relative = [object_xyz[index] - gripper_xyz[index] for index in range(3)]
            samples.append({
                "row": row,
                "stage": stage,
                "relative": relative,
                "drift": _distance(anchor_relative, relative),
            })
        maximum = max(sample["drift"] for sample in samples)
        peaks = [sample for sample in samples if abs(sample["drift"] - maximum) <= TOLERANCE]
        intervals.append({
            "anchor": samples[0],
            "samples": samples,
            "peaks": peaks,
            "maximum_drift": maximum,
            "followed_by_nonweld": followed_by_nonweld,
        })

    for row in ordered:
        weld = str(row.get("weld_state", "0")).lower() in {"1", "true"}
        if weld:
            active.append(row)
        elif active:
            finish(True)
            active = []
    if active:
        finish(False)
    return intervals


def _reference_peak_row(meta: dict[str, Any], unresolved: dict[str, str], drift_limit: float) -> dict[str, Any]:
    root = Path(meta["path"])
    oracle_rows = read_csv(root / "oracle_timeline.csv")
    actions = read_csv(root / "actions.csv")
    intervals = _oracle_peak_intervals(oracle_rows)
    if not intervals:
        raise ValueError(f"unresolved rollout has no weld interval: {meta['rollout_id']}")
    interval_index, selected = max(enumerate(intervals), key=lambda item: item[1]["maximum_drift"])
    anchor = selected["anchor"]
    peaks = selected["peaks"]
    first_peak, last_peak = peaks[0], peaks[-1]
    anchor_time = _number(anchor["row"]["time"])
    first_peak_time = _number(first_peak["row"]["time"])
    last_peak_time = _number(last_peak["row"]["time"])
    peak_stages = sorted({peak["stage"] for peak in peaks})
    peak_delta = [first_peak["relative"][index] - anchor["relative"][index] for index in range(3)]
    return {
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "original_reference_status": "reference_unresolved",
        "original_reference_reason": unresolved.get("reason"),
        "selected_weld_interval_index": interval_index,
        "weld_interval_count": len(intervals),
        "weld_sample_count": len(selected["samples"]),
        "anchor_time": anchor_time,
        "anchor_capture_order": anchor["row"].get("capture_order"),
        "anchor_phase": anchor["row"].get("phase"),
        "anchor_action_candidates": ",".join(_action_candidates(actions, anchor_time)),
        "anchor_relative_position_xyz_m": json.dumps(anchor["relative"], separators=(",", ":")),
        "peak_first_time": first_peak_time,
        "peak_last_time": last_peak_time,
        "peak_first_capture_order": first_peak["row"].get("capture_order"),
        "peak_last_capture_order": last_peak["row"].get("capture_order"),
        "peak_first_phase": first_peak["row"].get("phase"),
        "peak_last_phase": last_peak["row"].get("phase"),
        "peak_action_candidates": ",".join(_action_candidates(actions, first_peak_time)),
        "peak_relative_position_xyz_m": json.dumps(first_peak["relative"], separators=(",", ":")),
        "peak_relative_delta_from_anchor_xyz_m": json.dumps(peak_delta, separators=(",", ":")),
        "maximum_relative_position_drift_m": selected["maximum_drift"],
        "frozen_relative_drift_limit_m": drift_limit,
        "relative_drift_excess_m": selected["maximum_drift"] - drift_limit,
        "peak_first_stage": first_peak["stage"],
        "peak_last_stage": last_peak["stage"],
        "peak_stages": ",".join(peak_stages),
        "peak_occurrence_count": len(peaks),
        "peak_persists_after_first_attainment": len(peaks) > 1,
        "interval_followed_by_nonweld": selected["followed_by_nonweld"],
        "relative_position_definition": RELATIVE_POSITION_DEFINITION,
        "stage_definition": (
            "constraint_establishment = first two distinct weld-active timestamps; "
            "loss_boundary = final weld-active timestamp before weld becomes false; sustained_hold = intervening rows"
        ),
        "reference_label_changed": False,
        "reference_version_changed": False,
        "action_names_role": "offline stage localization only; never supplied to online hold evidence",
    }


def _mechanism_summary(segment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = Counter(row["segment_kind"] for row in segment_rows)
    early = [row for row in segment_rows if row["segment_kind"] == "target_initial_false_hold"]
    observed_retained = [row for row in segment_rows if row["segment_kind"] in {"same_k8_later_hold", "brief_hold_positive"}]
    missing_brief = [row for row in segment_rows if row["segment_kind"] == "brief_hold_positive_no_bcount2_onset"]
    return {
        "segment_counts": dict(sorted(by_kind.items())),
        "early_false_segments": len(early),
        "early_false_all_have_sub_noise_interval": bool(early) and all(int(row["sub_noise_interval_count"]) > 0 for row in early),
        "early_false_any_direction_disagreement": any(int(row["direction_disagreement_count"]) > 0 for row in early),
        "early_false_all_use_action_end_row": bool(early) and all(int(row["action_end_interval_count"]) > 0 for row in early),
        "early_false_same_timestamp_duplicate_double_counted": any(int(row["merged_same_timestamp_double_count_count"]) > 0 for row in early),
        "observed_valid_onset_segments": len(observed_retained),
        "observed_valid_onsets_all_have_directional_interval": bool(observed_retained) and all(int(row["directional_interval_pass_count"]) > 0 for row in observed_retained),
        "brief_hold_reference_labeled_total": sum(row["segment_kind"].startswith("brief_hold_positive") for row in segment_rows),
        "brief_hold_with_bcount2_onset": sum(row["segment_kind"] == "brief_hold_positive" for row in segment_rows),
        "brief_hold_without_bcount2_onset": len(missing_brief),
        "cached_contrast_establishes_complete_replacement_rule": False,
        "diagnostic_boundary": (
            "The merged dense stream contains one row per timestamp, so same-timestamp control/action-end captures are not counted twice. "
            "The observed valid onsets contain directional motion, but one reference-labeled brief hold has no B_count2 onset. "
            "The cache therefore localizes the false-positive mechanism without establishing a complete replacement rule."
        ),
    }


def run_localization(
    data_root: Path,
    unresolved_path: Path,
    reference_contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    metadata = {row["rollout_id"]: row for row in read_csv(data_root / "rollout_manifest.csv")}
    unresolved_rows = read_csv(unresolved_path)
    unresolved_by_id = {row["rollout_id"]: row for row in unresolved_rows}
    contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    drift_limit = float(contract["hold_proxy"]["maximum_relative_position_drift_m"])

    interval_rows, segment_rows = _select_segments(metadata, set(unresolved_by_id))
    reference_rows = []
    for unresolved in unresolved_rows:
        meta = metadata.get(unresolved["rollout_id"])
        if meta is None:
            raise ValueError(f"unresolved rollout missing from manifest: {unresolved['rollout_id']}")
        reference_rows.append(_reference_peak_row(meta, unresolved, drift_limit))

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "online_bcount2_interval_mechanisms.csv", interval_rows)
    write_csv(output_root / "online_segment_comparison.csv", segment_rows)
    write_csv(output_root / "reference_drift_peaks.csv", reference_rows)

    stage_counts = Counter(row["peak_first_stage"] for row in reference_rows)
    stage_sets = Counter(row["peak_stages"] for row in reference_rows)
    action_counts = Counter(row["peak_action_candidates"] for row in reference_rows)
    persistent_peaks = sum(bool(row["peak_persists_after_first_attainment"]) for row in reference_rows)
    reference_summary = {
        "schema": "pathgraph_l2rar2_reference_drift_stage_summary_v1",
        "status": "REFERENCE_MECHANISM_LOCALIZED",
        "rows": len(reference_rows),
        "original_unresolved_denominator_preserved": len(reference_rows),
        "peak_first_stage_counts": dict(sorted(stage_counts.items())),
        "peak_stage_set_counts": dict(sorted(stage_sets.items())),
        "peak_first_action_counts": dict(sorted(action_counts.items())),
        "peak_persists_after_first_attainment_count": persistent_peaks,
        "single_attainment_peak_count": len(reference_rows) - persistent_peaks,
        "relative_position_definition": RELATIVE_POSITION_DEFINITION,
        "frozen_relative_drift_limit_m": drift_limit,
        "reference_label_changes": 0,
        "reference_version_changes": 0,
        "interpretation": (
            "Peak localization is an offline audit of the existing world-frame proxy. First attainment and later persistence are reported separately. "
            "It does not turn any unresolved event into a pass or attribute reference changes to online algorithm gain."
        ),
    }
    write_json(output_root / "reference_drift_stage_summary.json", reference_summary)

    online_summary = _mechanism_summary(segment_rows)
    result = {
        "schema": "pathgraph_l2rar2_mechanism_localization_v1",
        "status": "MECHANISM_LOCALIZATION_COMPLETE",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "new_status": "L2RAR2_PARTIAL_KEEP_G1",
        "scope": {
            "data_root": str(data_root.resolve()),
            "target_families": sorted(TARGET_FAMILIES),
            "target_k2_k8_rollouts": 4,
            "brief_hold_reference_labeled_rollouts": online_summary["brief_hold_reference_labeled_total"],
            "brief_hold_rollouts_with_bcount2_onset": online_summary["brief_hold_with_bcount2_onset"],
            "online_comparison_segments": len(segment_rows),
            "reference_unresolved_rows": len(reference_rows),
        },
        "online_mechanism": online_summary,
        "reference_mechanism": reference_summary,
        "decision": {
            "retained_graph": "G1_predicate_bound",
            "l3_entry_allowed": False,
            "candidate_grid_expanded": False,
            "online_hold_predicate_change_authorized": False,
            "physical_reference_relaxed": False,
            "reference_version_changed": False,
            "unresolved_rows_removed": 0,
            "new_sampling_authorized": False,
            "training_authorized": False,
            "confirmation_authorized": False,
            "next_step": "inspect the reference-labeled K5 case without a B_count2 onset before formulating one separately versioned online hypothesis; validate reference geometry independently",
        },
        "provenance": {
            "physical_rollouts_reexecuted": 0,
            "training_jobs": 0,
            "api_calls": 0,
            "api_key_read": False,
            "online_actions_used_as_features": False,
            "oracle_and_action_names_used_for": "offline reference-stage localization and raw-capture multiplicity audit only",
            "online_interval_hash": sha256(output_root / "online_bcount2_interval_mechanisms.csv"),
            "online_segment_hash": sha256(output_root / "online_segment_comparison.csv"),
            "reference_peak_hash": sha256(output_root / "reference_drift_peaks.csv"),
        },
    }
    write_json(output_root / "mechanism_localization.json", result)

    report = [
        "## Material Passport",
        "",
        "- Artifact type: cached-experiment mechanism localization",
        "- Verification status: `ANALYZED` (cache-only; no physical re-run)",
        "- Historical boundary: `L2RAR1_PARTIAL_KEEP_G1`",
        "- Current R2 boundary: `L2RAR2_PARTIAL_KEEP_G1`; retained graph: `G1_predicate_bound`; L3 remains closed",
        "",
        "# L2RA-R2 mechanism localization",
        "",
        "## Online mechanism",
        "",
        f"- Compared `{online_summary['segment_counts'].get('target_initial_false_hold', 0)}` initial K2/K8 false-hold segments, "
        f"`{online_summary['segment_counts'].get('same_k8_later_hold', 0)}` later K8 hold segments, and "
        f"`{online_summary['brief_hold_reference_labeled_total']}` reference-labeled brief-hold examples; "
        f"`{online_summary['brief_hold_with_bcount2_onset']}` have a `B_count2` onset and `{online_summary['brief_hold_without_bcount2_onset']}` does not.",
        f"- Every initial false segment contains a sub-noise interval: `{online_summary['early_false_all_have_sub_noise_interval']}`. "
        f"A direction disagreement is present in at least one initial false segment: `{online_summary['early_false_any_direction_disagreement']}`.",
        f"- Same-timestamp control/action-end rows are counted twice by the merged dense stream: `{online_summary['early_false_same_timestamp_duplicate_double_counted']}`. "
        "The collector keeps one post-update row per timestamp; action-end participation and duplicate counting are therefore distinct questions.",
        f"- Every observed later-K8/brief-hold onset contains at least one directionally consistent effective-motion interval: `{online_summary['observed_valid_onsets_all_have_directional_interval']}`.",
        "- This does not establish a complete replacement rule: one frozen reference-labeled brief hold has no `B_count2` onset. The cache localizes the early false-positive mechanism, but a future hypothesis must also recover that missed positive without reopening short-touch false holds.",
        "",
        "## Reference mechanism",
        "",
        f"- Preserved all `{len(reference_rows)}` unresolved rows and the frozen `0.02 m` limit. First peak-stage counts: `{json.dumps(dict(sorted(stage_counts.items())), sort_keys=True)}`.",
        f"- First peak action contexts: `{json.dumps(dict(sorted(action_counts.items())), sort_keys=True)}`; `{persistent_peaks}/{len(reference_rows)}` peaks persist at the maximum after first attainment, so a plateau is not reported as a one-frame spike.",
        f"- Relative position remains `{RELATIVE_POSITION_DEFINITION}`.",
        "- `reference_drift_peaks.csv` records anchor and peak vectors, first/last peak times, peak stages, and offline action context. No row was converted to a reference pass.",
        "- Any future geometry or reference change must use a new version and be reported beside this frozen result; it cannot be counted as online algorithm gain.",
        "",
        "## Decision boundary",
        "",
        "- No candidate expansion, retraining, API/key access, new sampling, reference relaxation, confirmation, or L3 entry was performed or authorized.",
        "- Online predicate work and reference-interface work remain separate validation tracks.",
    ]
    (output_root / "mechanism_localization.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache-only L2RA-R2 mechanism localization")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_localization(args.data_root, args.unresolved, args.reference_contract, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
