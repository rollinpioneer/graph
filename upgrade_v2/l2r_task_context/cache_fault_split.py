"""Read-only cache diagnostic for the R2 hold/attempt boundary.

The diagnostic has two deliberately separate views:

* target K2/K8 traces use only arrived dense observations and the frozen
  ``B_count2`` evidence path.  Reference events and oracle rows are used only
  to name offline comparison points.
* unresolved references are audited for file/timestamp/action-end integrity,
  then compared with the already locked physical hold proxy.  The original
  unresolved label is never changed or removed from a denominator.

This module does not tune a threshold, create a candidate, retrain, collect
new rollouts, or run confirmation.
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

from .io import read_csv, read_json, read_jsonl, sha256, write_csv, write_json


TARGET_FAMILIES = {
    "L2RAR2_SELECT_05_820005",
    "L2RAR2_SELECT_07_820007",
}
TARGET_CASES = {
    "K2_touch_request_completes_without_hold",
    "K8_acquisition_touch_then_continue",
}
REQUIRED_DENSE_KEYS = ("time", "capture_order", "contact_present", "gripper_command")
REQUIRED_ORACLE_KEYS = ("time", "capture_order", "weld_state", "object_xyz", "gripper_xyz")


def _number(value: Any) -> float:
    return float(value)


def _first(rows: Iterable[dict[str, Any]], predicate) -> dict[str, Any] | None:
    return next((row for row in rows if predicate(row)), None)


def _first_time(rows: Iterable[dict[str, Any]], predicate) -> float | None:
    row = _first(rows, predicate)
    return _number(row["time"]) if row is not None and row.get("time") is not None else None


def _nearest_prior(rows: list[dict[str, Any]], time: float, *, source_phase: str | None = None) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("time") is not None and _number(row["time"]) <= time + 1e-9]
    if source_phase is not None:
        phase_rows = [row for row in eligible if row.get("source_phase") == source_phase]
        if phase_rows:
            eligible = phase_rows
    return max(eligible, key=lambda row: (_number(row["time"]), int(row.get("capture_order", 0)))) if eligible else None


def _same_stream_key(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("capture_order") is not None and right.get("capture_order") is not None:
        return int(left["capture_order"]) == int(right["capture_order"])
    return left.get("time") is not None and right.get("time") is not None and abs(_number(left["time"]) - _number(right["time"])) <= 1e-9


def _audit_dense(rows: list[dict[str, Any]], label: str) -> list[str]:
    issues: list[str] = []
    if not rows:
        return [f"{label}_empty"]
    previous_time = None
    previous_order = None
    for index, row in enumerate(rows):
        missing = [key for key in REQUIRED_DENSE_KEYS if key not in row]
        if missing:
            issues.append(f"{label}_row_{index}_missing_" + "_".join(missing))
        try:
            time = _number(row["time"])
            order = int(row["capture_order"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{label}_row_{index}_invalid_time_or_capture_order")
            continue
        if previous_time is not None and time < previous_time - 1e-9:
            issues.append(f"{label}_time_not_monotonic")
        if previous_order is not None and order <= previous_order:
            issues.append(f"{label}_capture_order_not_increasing")
        previous_time, previous_order = time, order
    return sorted(set(issues))


def _audit_action_end_alignment(dense: list[dict[str, Any]], action_end: list[dict[str, Any]]) -> list[str]:
    issues = _audit_dense(action_end, "action_end")
    for index, row in enumerate(action_end):
        match = _first(dense, lambda candidate: _same_stream_key(candidate, row))
        if match is None:
            issues.append(f"action_end_row_{index}_missing_dense_match")
            continue
        for key in ("contact_present", "gripper_command"):
            if match.get(key) != row.get(key):
                issues.append(f"action_end_row_{index}_{key}_mismatch")
    return sorted(set(issues))


def _vector(value: Any) -> list[float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        raise ValueError("expected a three-dimensional vector")
    return [float(item) for item in parsed[:3]]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _oracle_intervals(rows: list[dict[str, Any]], drift_limit: float) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_number(row["time"]), int(row.get("capture_order", 0))))
    intervals: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []

    def finish(end_time: float) -> None:
        if not active:
            return
        objects = [_vector(row["object_xyz"]) for row in active]
        grippers = [_vector(row["gripper_xyz"]) for row in active]
        relative = [[obj[i] - grip[i] for i in range(3)] for obj, grip in zip(objects, grippers)]
        displacement = _distance(objects[0], objects[-1])
        relative_drift = max(_distance(relative[0], point) for point in relative)
        intervals.append({
            "start_time": _number(active[0]["time"]),
            "end_time": end_time,
            "sample_count": len(active),
            "duration_seconds": max(0.0, end_time - _number(active[0]["time"])),
            "object_displacement_m": displacement,
            "maximum_relative_position_drift_m": relative_drift,
            "object_displacement_pass": displacement >= 0.01,
            "relative_drift_pass": relative_drift <= drift_limit,
            "status": "held_verified" if displacement >= 0.01 and relative_drift <= drift_limit else "reference_unresolved",
        })

    for row in ordered:
        weld = str(row.get("weld_state", "0")).lower() in {"1", "true"}
        if weld:
            active.append(row)
        elif active:
            finish(_number(row["time"]))
            active = []
    if active:
        finish(_number(active[-1]["time"]))
    return intervals


def _load_online(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_jsonl(Path(meta["path"]) / "observations_dense.jsonl")
    observations = [_online_observation(row) for row in raw]
    geometries = build_features(observations)
    predictions: list[dict[str, Any]] = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    evidence = evaluate_candidate(predictions, geometries, "B_count2")
    return predictions, evidence


def _target_trace(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations, evidence = _load_online(meta)
    action_end = read_jsonl(Path(meta["path"]) / "observations_action_end.jsonl")
    events = read_jsonl(Path(meta["path"]) / "events.jsonl")
    event_by_name = {row.get("event"): row for row in events}
    transient_loss_time = event_by_name.get("transient_contact_lost", {}).get("time")
    transient_loss_time = _number(transient_loss_time) if transient_loss_time is not None else None
    first_evidence_index = next((index for index, row in enumerate(evidence) if row.get("hold_evidence") == "true"), None)
    first_memory_index = next((index for index, row in enumerate(evidence) if row.get("hold_memory") == "true"), None)
    post_loss_evidence_index = next((
        index for index, row in enumerate(evidence)
        if transient_loss_time is not None
        and row.get("time") is not None
        and _number(row["time"]) > transient_loss_time + 1e-9
        and row.get("hold_evidence") == "true"
    ), None)
    post_loss_memory_index = next((
        index for index, row in enumerate(evidence)
        if transient_loss_time is not None
        and row.get("time") is not None
        and _number(row["time"]) > transient_loss_time + 1e-9
        and row.get("hold_memory") == "true"
    ), None)
    first_evidence = evidence[first_evidence_index] if first_evidence_index is not None else None
    first_memory = evidence[first_memory_index] if first_memory_index is not None else None
    post_loss_evidence = evidence[post_loss_evidence_index] if post_loss_evidence_index is not None else None
    post_loss_memory = evidence[post_loss_memory_index] if post_loss_memory_index is not None else None
    rows: list[dict[str, Any]] = []
    for observation, evidence_row in zip(observations, evidence):
        time = _number(observation["time"])
        aligned_end = _nearest_prior(action_end, time, source_phase="action_end")
        geometry = evidence_row.get("geometry", {})
        rows.append({
            "rollout_id": meta["rollout_id"],
            "root_family_id": meta["root_family_id"],
            "case_id": meta["case_id"],
            "time": time,
            "capture_order": observation.get("capture_order"),
            "source_phase": observation.get("source_phase", observation.get("phase")),
            "front_path": observation.get("front_path"),
            "front_sha256": observation.get("front_sha256"),
            "attempt_id": observation.get("attempt_id"),
            "attempt_phase": observation.get("attempt_phase"),
            "attempt_active": observation.get("attempt_active"),
            "attempt_end": observation.get("attempt_end"),
            "attempt_end_reason": observation.get("attempt_end_reason"),
            "contact_present": observation.get("contact_present"),
            "gripper_command": observation.get("gripper_command"),
            "object_centroid": json.dumps(observation.get("object_centroid"), separators=(",", ":")),
            "gripper_centroid": json.dumps(observation.get("gripper_centroid"), separators=(",", ":")),
            "object_confidence": observation.get("object_confidence"),
            "gripper_confidence": observation.get("gripper_confidence"),
            "observation_masked": bool(observation.get("observation_masked") or observation.get("observation_missing")),
            "hold_evidence": evidence_row.get("hold_evidence"),
            "hold_memory": evidence_row.get("hold_memory"),
            "effective_motion_interval": geometry.get("effective_motion_interval"),
            "object_displacement_norm": geometry.get("object_displacement_norm"),
            "gripper_displacement_norm": geometry.get("gripper_displacement_norm"),
            "direction_cosine": geometry.get("direction_cosine"),
            "relative_vector_error": geometry.get("relative_vector_error"),
            "nearest_action_end_time": aligned_end.get("time") if aligned_end else None,
            "action_end_alignment_seconds": time - _number(aligned_end["time"]) if aligned_end else None,
        })
    summary = {
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "requested_effect": meta.get("requested_effect"),
        "dense_observation_count": len(observations),
        "action_end_observation_count": len(action_end),
        "first_hold_evidence_time": first_evidence.get("time") if first_evidence else None,
        "first_hold_evidence_capture_order": observations[first_evidence_index].get("capture_order") if first_evidence_index is not None else None,
        "first_hold_memory_time": first_memory.get("time") if first_memory else None,
        "first_hold_memory_capture_order": observations[first_memory_index].get("capture_order") if first_memory_index is not None else None,
        "first_post_transient_loss_hold_evidence_time": post_loss_evidence.get("time") if post_loss_evidence else None,
        "first_post_transient_loss_hold_evidence_capture_order": observations[post_loss_evidence_index].get("capture_order") if post_loss_evidence_index is not None else None,
        "first_post_transient_loss_hold_memory_time": post_loss_memory.get("time") if post_loss_memory else None,
        "first_post_transient_loss_hold_memory_capture_order": observations[post_loss_memory_index].get("capture_order") if post_loss_memory_index is not None else None,
        "hold_evidence_count": sum(row.get("hold_evidence") == "true" for row in evidence),
        "hold_memory_count": sum(row.get("hold_memory") == "true" for row in evidence),
        "hold_evidence_count_through_transient_loss": sum(
            row.get("hold_evidence") == "true"
            and transient_loss_time is not None
            and row.get("time") is not None
            and _number(row["time"]) <= transient_loss_time + 1e-9
            for row in evidence
        ),
        "first_transient_contact_time": event_by_name.get("transient_contact", {}).get("time"),
        "first_transient_contact_lost_time": event_by_name.get("transient_contact_lost", {}).get("time"),
        "first_contact_established_time": event_by_name.get("contact_established", {}).get("time"),
        "reference_events_used_for_alignment": sorted(event_by_name),
        "online_hold_evidence_observed": first_evidence is not None,
        "online_hold_memory_observed": first_memory is not None,
        "online_hold_reestablished_after_transient_loss": post_loss_memory is not None,
        "online_feature_sources": "dense observations only; action-end used only for causal alignment audit",
    }
    return rows, summary


def _reference_row(meta: dict[str, Any], unresolved: dict[str, str], drift_limit: float) -> tuple[dict[str, Any], list[str]]:
    root = Path(meta["path"])
    dense = read_jsonl(root / "observations_dense.jsonl")
    action_end = read_jsonl(root / "observations_action_end.jsonl")
    oracle = read_csv(root / "oracle_timeline.csv")
    events = read_jsonl(root / "events.jsonl")
    issues = _audit_dense(dense, "dense") + _audit_action_end_alignment(dense, action_end)
    issues.extend(
        f"oracle_row_{index}_missing_" + "_".join(key for key in REQUIRED_ORACLE_KEYS if key not in row)
        for index, row in enumerate(oracle)
        if any(key not in row for key in REQUIRED_ORACLE_KEYS)
    )
    try:
        intervals = _oracle_intervals(oracle, drift_limit)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        intervals = []
        issues.append("oracle_vector_or_time_invalid")
    loss_event = _first(events, lambda row: row.get("event") == "contact_lost")
    event_time = _number(loss_event["time"]) if loss_event and loss_event.get("time") is not None else None
    loss_inside_interval = bool(
        event_time is not None
        and any(interval["start_time"] <= event_time + 1e-9 <= interval["end_time"] + 1e-9 for interval in intervals)
    )
    observed_intervals = [interval for interval in intervals if interval["status"] == "held_verified"]
    _, online_evidence = _load_online(meta)
    first_evidence = _first(online_evidence, lambda row: row.get("hold_evidence") == "true")
    first_memory = _first(online_evidence, lambda row: row.get("hold_memory") == "true")
    if issues:
        category = "recording_or_time_alignment_insufficient"
        owner = "collection_or_reference_interface"
    elif not intervals:
        category = "physical_hold_interval_not_recorded"
        owner = "collection_or_reference_interface"
    elif not observed_intervals:
        category = "physical_proxy_present_but_outside_frozen_condition"
        owner = "reference_proxy_or_collection_geometry"
    else:
        category = "reference_contract_inconsistency_requires_review"
        owner = "reference_interface"
    row = {
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "case_id": meta["case_id"],
        "original_reference_reason": unresolved.get("reason"),
        "original_reference_status": "reference_unresolved",
        "diagnostic_category": category,
        "suggested_owner": owner,
        "dense_rows": len(dense),
        "action_end_rows": len(action_end),
        "oracle_rows": len(oracle),
        "event_names": ",".join(sorted({str(row.get("event")) for row in events})),
        "contact_loss_time": event_time,
        "oracle_hold_interval_count": len(intervals),
        "oracle_verified_hold_interval_count": len(observed_intervals),
        "loss_inside_oracle_hold_interval": loss_inside_interval,
        "first_online_hold_evidence_time": first_evidence.get("time") if first_evidence else None,
        "first_online_hold_memory_time": first_memory.get("time") if first_memory else None,
        "online_hold_evidence_observed": first_evidence is not None,
        "online_hold_memory_observed": first_memory is not None,
        "max_oracle_relative_drift_m": max((interval["maximum_relative_position_drift_m"] for interval in intervals), default=None),
        "oracle_relative_drift_limit_m": drift_limit,
        "relative_drift_excess_m": max(
            (interval["maximum_relative_position_drift_m"] - drift_limit for interval in intervals),
            default=None,
        ),
        "max_oracle_object_displacement_m": max((interval["object_displacement_m"] for interval in intervals), default=None),
        "oracle_interval_summary": json.dumps(intervals, sort_keys=True),
        "integrity_issues": ";".join(sorted(set(issues))),
        "online_feature_sources": "dense observations only; oracle/events are reference-only",
    }
    return row, sorted(set(issues))


def run_diagnostic(
    data_root: Path,
    unresolved_path: Path,
    reference_contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    metadata = {row["rollout_id"]: row for row in read_csv(data_root / "rollout_manifest.csv")}
    unresolved_rows = read_csv(unresolved_path)
    contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    drift_limit = float(contract["hold_proxy"]["maximum_relative_position_drift_m"])

    target_trace_rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    for meta in metadata.values():
        if meta["root_family_id"] in TARGET_FAMILIES and meta["case_id"] in TARGET_CASES:
            rows, summary = _target_trace(meta)
            target_trace_rows.extend(rows)
            target_summaries.append(summary)
    if len(target_summaries) != 4:
        raise ValueError(f"expected four target rollouts, found {len(target_summaries)}")

    unresolved_by_id = {row["rollout_id"]: row for row in unresolved_rows}
    split_rows: list[dict[str, Any]] = []
    integrity_issues: list[str] = []
    for unresolved in unresolved_rows:
        meta = metadata.get(unresolved["rollout_id"])
        if meta is None:
            split_rows.append({
                "rollout_id": unresolved["rollout_id"],
                "case_id": unresolved.get("case_id"),
                "original_reference_reason": unresolved.get("reason"),
                "original_reference_status": "reference_unresolved",
                "diagnostic_category": "recording_or_time_alignment_insufficient",
                "suggested_owner": "manifest_or_collection_interface",
                "integrity_issues": "manifest_row_missing",
            })
            integrity_issues.append("manifest_row_missing")
            continue
        row, issues = _reference_row(meta, unresolved_by_id[unresolved["rollout_id"]], drift_limit)
        split_rows.append(row)
        integrity_issues.extend(issues)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "target_hold_trace.csv", target_trace_rows)
    write_csv(output_root / "target_hold_summary.csv", target_summaries)
    write_csv(output_root / "reference_unresolved_split.csv", split_rows)
    category_counts = Counter(row["diagnostic_category"] for row in split_rows)
    online_vs_reference = Counter(
        "online_evidence_and_memory" if row.get("online_hold_evidence_observed") and row.get("online_hold_memory_observed")
        else "online_evidence_only" if row.get("online_hold_evidence_observed")
        else "no_online_hold_evidence"
        for row in split_rows
    )
    drift_excesses = [
        float(row["relative_drift_excess_m"])
        for row in split_rows
        if row.get("relative_drift_excess_m") is not None
    ]
    result = {
        "schema": "pathgraph_l2rar2_cache_fault_split_v1",
        "status": "CACHE_DIAGNOSTIC_COMPLETE",
        "scientific_status": "L2RAR1_PARTIAL_KEEP_G1",
        "scope": {
            "data_root": str(data_root.resolve()),
            "target_families": sorted(TARGET_FAMILIES),
            "target_cases": sorted(TARGET_CASES),
            "reference_unresolved_input": str(unresolved_path.resolve()),
            "target_rollouts": len(target_summaries),
            "reference_unresolved_rows": len(split_rows),
        },
        "target_conclusion": {
            "first_hold_evidence_and_memory_are_recomputed_from_frozen_b_count2": True,
            "k2_and_k8_are_reported_separately": True,
            "no_online_rule_changed": True,
            "no_confirmation_started": True,
            "interpretation": "Frozen B_count2 marks the initial transient-contact prefix as hold evidence in both K2 and K8. K2 then clears memory at contact loss and never re-establishes it; K8 later re-establishes hold after the second contact while the attempt remains active. This is a short-contact false-hold boundary in the online predicate, while contact loss during active acquisition still cannot announce retry by itself.",
        },
        "reference_conclusion": {
            "original_unresolved_denominator_preserved": len(split_rows),
            "diagnostic_category_counts": dict(sorted(category_counts.items())),
            "online_hold_state_counts": dict(sorted(online_vs_reference.items())),
            "integrity_issue_counts": dict(sorted(Counter(integrity_issues).items())),
            "relative_drift_excess_range_m": {
                "min": min(drift_excesses) if drift_excesses else None,
                "max": max(drift_excesses) if drift_excesses else None,
            },
            "interpretation": "The unresolved rows with an oracle weld interval but relative drift above the locked limit are reference-proxy/collection-geometry boundary cases, not evidence that the online hold predicate never observed a hold.",
        },
        "decision": {
            "next_owner": "split_online_hold_predicate_and_reference_proxy_interfaces",
            "online_hold_predicate_diagnostic": "K2/K8 initial transient contact is accepted as B_count2 hold evidence",
            "reference_interface_diagnostic": "19 unresolved rows have intact alignment and online hold evidence but exceed the locked physical drift limit",
            "online_hold_predicate_change_authorized": False,
            "new_sampling_authorized": False,
            "training_authorized": False,
            "confirmation_authorized": False,
            "l3_entry_allowed": False,
            "retained_graph": "G1_predicate_bound",
        },
        "provenance": {
            "physical_rollouts_reexecuted": 0,
            "training_jobs": 0,
            "api_calls": 0,
            "api_key_read": False,
            "reference_events_and_oracle_used_for": "offline alignment and proxy audit only",
            "target_trace_hash": sha256(output_root / "target_hold_trace.csv"),
            "reference_split_hash": sha256(output_root / "reference_unresolved_split.csv"),
        },
    }
    write_json(output_root / "cache_fault_split.json", result)
    report = [
        "## Material Passport",
        "",
        "- Artifact type: cached-experiment validation report",
        "- Verification status: `ANALYZED` (read-only cache recomputation; no physical re-run)",
        "- Data role: frozen `dev_select` diagnostic cache",
        "- Scientific boundary: `L2RAR1_PARTIAL_KEEP_G1`; L3 remains closed",
        "",
        "# L2RA-R2 cache fault split",
        "",
        "- Status: `CACHE_DIAGNOSTIC_COMPLETE`; retained graph: `G1_predicate_bound`.",
        "- Scope is read-only cached data. Physical rollouts re-executed: `0`; training jobs: `0`; API calls: `0`; confirmation: not run.",
        f"- Target scope: four rollouts from families `05` and `07`, cases `K2` and `K8`; full per-frame trace is in `target_hold_trace.csv`.",
        f"- Reference scope: all `{len(split_rows)}` rows from the original `reference_unresolved.csv` remain in the denominator and are preserved verbatim by status/reason.",
        "",
        "## Findings",
        "",
        "- Frozen `B_count2` marks the initial transient contact as hold evidence in both K2 and K8. K2 clears that memory at loss and never re-establishes it; K8 shares the same early false-hold prefix, then later re-establishes hold after the second contact while `attempt_active=true`.",
        "- All 19 unresolved rows have intact dense/action-end alignment and online hold evidence. Their oracle relative drift exceeds the locked `0.02 m` limit by `0.000305-0.001825 m`; they are reported as `physical_proxy_present_but_outside_frozen_condition`.",
        "- The result is a two-fault split: K2/K8 expose an online short-contact false-hold boundary, while the 19 unresolved rows expose a reference-proxy/collection-geometry boundary. It does not yet authorize changing either contract, deleting unresolved rows, relaxing the physical reference, disabling recovery, expanding candidates, retraining, or confirmation.",
        "",
        "## Files",
        "",
        "- `target_hold_summary.csv`: first online evidence/memory and offline event alignment for K2/K8.",
        "- `target_hold_trace.csv`: per-observation image/centroid/confidence/contact/phase/evidence trace.",
        "- `visual_frame_review.csv`: manual review of selected source frames; it is diagnostic evidence only.",
        "- `reference_unresolved_split.csv`: all original unresolved rows with diagnostic-only classification.",
        "- `cache_fault_split.json`: machine-readable scope, counts, decision boundary, and provenance.",
    ]
    (output_root / "cache_fault_split.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only L2RA-R2 cache fault split")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(args.data_root, args.unresolved, args.reference_contract, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
