"""Diagnose the observable boundary between a touch and a missed grasp.

This is a read-only post-hoc diagnostic. Reference event files are used only
to align and group cached cases; they are never converted into online
predicates. The command trace is reduced to already-issued low-level target
positions, gripper state, and timing. Action names, scenario names, oracle
state, and future outcomes are deliberately excluded from the diagnostic
features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .hold_features import build_features
from .inputs import read_json, read_jsonl, write_csv, write_json


TARGET_STRATA = {
    "miss_without_prior_hold": "missed_grasp",
    "touch_without_hold_then_loss": "transient_contact_lost",
}


def _float(value: Any) -> float:
    return float(value)


def _bool(value: Any) -> bool:
    return bool(value)


def _event_time(rollout: Path, event_name: str) -> float:
    events = read_jsonl(rollout / "events.jsonl")
    matches = [row for row in events if row.get("event") == event_name]
    if not matches:
        raise ValueError(f"missing reference event {event_name!r}: {rollout}")
    return _float(matches[0]["time"])


def _flatten_controls(rollout: Path) -> list[dict[str, Any]]:
    rows = []
    for record in read_jsonl(rollout / "low_level_controls.jsonl"):
        # Keep only the command state that had already reached the controller.
        # In particular, do not retain action names or action indices.
        for control in record.get("controls", []):
            position = [float(value) for value in control["mocap_position"]]
            rows.append({
                "start_time": _float(control["start_time"]),
                "end_time": _float(control["end_time"]),
                "gripper_command": control.get("gripper_command"),
                "mocap_position": position,
                "physics_steps": int(control.get("physics_steps", 0)),
            })
    return sorted(rows, key=lambda row: (row["end_time"], row["start_time"]))


def _command_at(controls: list[dict[str, Any]], time: float) -> dict[str, Any]:
    candidates = [row for row in controls if abs(row["end_time"] - time) <= 1e-9]
    if not candidates:
        candidates = sorted(controls, key=lambda row: abs(row["end_time"] - time))[:1]
    if not candidates:
        raise ValueError("low-level command trace is empty")
    current = candidates[-1]
    index = controls.index(current)
    previous = controls[index - 1] if index else current
    delta = [current["mocap_position"][i] - previous["mocap_position"][i] for i in range(3)]
    return {
        "control_start_time": current["start_time"],
        "control_end_time": current["end_time"],
        "command_alignment_error_seconds": abs(current["end_time"] - time),
        "command_gripper": current["gripper_command"],
        "command_target_delta": delta,
        "command_target_delta_norm": math.sqrt(sum(value * value for value in delta)),
        "command_target_delta_xy_norm": math.hypot(delta[0], delta[1]),
        "command_target_delta_z": delta[2],
        "command_physics_steps": current["physics_steps"],
        "previous_command_available": index > 0,
    }


def _online_signature(rows: list[dict[str, Any]], event_time: float) -> dict[str, Any]:
    at_event = [row for row in rows if abs(_float(row["time"]) - event_time) <= 1e-9]
    if not at_event:
        at_event = sorted(rows, key=lambda row: abs(_float(row["time"]) - event_time))[:1]
    current = at_event[-1]
    prior = [row for row in rows if _float(row["time"]) < event_time - 1e-9]
    prior_contact = any(_bool(row.get("contact_present")) for row in prior)
    previous = prior[-1] if prior else None
    return {
        "aligned_observation_time": _float(current["time"]),
        "alignment_error_seconds": abs(_float(current["time"]) - event_time),
        "closed": current.get("gripper_command") == "closed",
        "contact_present": _bool(current.get("contact_present")),
        "previous_contact_present": _bool(previous.get("contact_present")) if previous else None,
        "prior_contact_seen": prior_contact,
        "observation_masked": bool(current.get("observation_masked") or current.get("observation_missing")),
        "object_confidence": current.get("object_confidence"),
        "gripper_confidence": current.get("gripper_confidence"),
        "object_centroid_present": current.get("object_centroid") is not None,
        "gripper_centroid_present": current.get("gripper_centroid") is not None,
    }


def _first_online_contact_loss(rows: list[dict[str, Any]]) -> float:
    previous = None
    for row in rows:
        current_contact = row.get("contact_present")
        if (
            previous is not None
            and previous.get("gripper_command") == "closed"
            and row.get("gripper_command") == "closed"
            and previous.get("contact_present") is True
            and current_contact is False
        ):
            return _float(row["time"])
        previous = row
    raise ValueError("no online closed-contact loss edge found")


def _case(meta_path: Path) -> dict[str, Any]:
    meta = read_json(meta_path)
    rollout = meta_path.parent
    event_name = TARGET_STRATA[meta["stratum"]]
    reference_event_time = _event_time(rollout, event_name)
    observations = read_jsonl(rollout / "observations_dense.jsonl")
    event_time = _first_online_contact_loss(observations)
    geometry = build_features(observations)
    online = _online_signature(observations, event_time)
    command = _command_at(_flatten_controls(rollout), event_time)
    geometry_row = min(geometry, key=lambda row: abs(_float(row["time"]) - event_time))
    return {
        "rollout_id": meta["rollout_id"],
        "root_family_id": meta["root_family_id"],
        "split": meta["split"],
        "reference_group": meta["stratum"],
        "online_contact_loss_time": event_time,
        "reference_event_time": reference_event_time,
        "online": online,
        "geometry": {
            "effective_motion_interval": geometry_row.get("effective_motion_interval"),
            "object_displacement_norm": geometry_row.get("object_displacement_norm"),
            "gripper_displacement_norm": geometry_row.get("gripper_displacement_norm"),
            "direction_cosine": geometry_row.get("direction_cosine"),
            "relative_vector_error": geometry_row.get("relative_vector_error"),
        },
        "command": command,
    }


def _range(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "median": sorted(values)[len(values) // 2] if values else None,
    }


def diagnose(data_root: Path, output_root: Path, split: str = "dev_select") -> dict[str, Any]:
    cases = []
    for meta_path in sorted(data_root.rglob("metadata.json")):
        meta = read_json(meta_path)
        if meta.get("split") == split and meta.get("stratum") in TARGET_STRATA:
            cases.append(_case(meta_path))
    if not cases:
        raise FileNotFoundError(f"no target cases found under {data_root} for split={split}")

    grouped = defaultdict(list)
    for case in cases:
        grouped[case["reference_group"]].append(case)

    common_fields = (
        "closed", "contact_present", "previous_contact_present", "prior_contact_seen",
        "observation_masked", "object_centroid_present", "gripper_centroid_present",
    )
    common_signature_counts = Counter(
        tuple(case["online"].get(field) for field in common_fields) for case in cases
    )
    command_ranges = {}
    for group, rows in grouped.items():
        values = [float(row["command"]["command_target_delta_norm"]) for row in rows]
        command_ranges[group] = {
            "count": len(values),
            "target_delta_norm": _range(values),
            "stationary_exact_count": sum(value == 0.0 for value in values),
            "moving_count": sum(value != 0.0 for value in values),
        }

    miss = grouped["miss_without_prior_hold"]
    touch = grouped["touch_without_hold_then_loss"]
    miss_values = [float(row["command"]["command_target_delta_norm"]) for row in miss]
    touch_values = [float(row["command"]["command_target_delta_norm"]) for row in touch]
    command_state_separates = bool(
        miss and touch
        and all(value == 0.0 for value in miss_values)
        and all(value != 0.0 for value in touch_values)
    )
    common_state = {
        "fields": list(common_fields),
        "signature_counts": [
            {"signature": list(signature), "count": count}
            for signature, count in sorted(common_signature_counts.items(), key=lambda item: repr(item[0]))
        ],
        "same_core_signature_across_target_groups": len({
            tuple(case["online"].get(field) for field in common_fields) for case in cases
        }) == 1,
    }
    result = {
        "schema": "pathgraph_l2rar1_attempt_boundary_diagnostic_v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "scientific_status": "L2RAR1_PARTIAL_KEEP_G1",
        "split": split,
        "cases": len(cases),
        "cases_by_reference_group": {key: len(value) for key, value in sorted(grouped.items())},
        "online_feature_contract": {
            "allowed": [
                "dense observation time, contact sensor, gripper command, visual centroids/confidence",
                "already-issued low-level controller target positions and timing",
            ],
            "excluded": [
                "scenario and stratum as decision features",
                "events.jsonl and reference event names as online features",
                "weld_state, qpos/qvel, oracle timeline, future outcome",
                "action names and action indices",
            ],
        },
        "first_contact_loss": {
            "common_online_state": common_state,
            "interpretation": "At first contact loss, the cached online predicate state is shared by missed-grasp and short-touch cases; the common event interface has no independent attempt-end field.",
        },
        "issued_command_at_contact_loss": {
            "per_group": command_ranges,
            "exact_stationary_vs_moving_separation": command_state_separates,
            "interpretation": "The currently arrived low-level target command separates these fixed dev_select cases, but this is a post-hoc diagnostic of the recorded controller phase, not a frozen retry rule or a general guarantee.",
        },
        "decision_boundary": {
            "attempt_end_observable_from_current_common_interface": False,
            "causal_command_phase_signal_observed_in_cache": command_state_separates,
            "recommended_minimal_contract": "Expose an explicit controller-side attempt_phase/attempt_end marker derived from command lifecycle, without action names, oracle state, or future outcome; keep hold_unknown separate from retryable_missed_grasp.",
            "do_not_do": [
                "do not delete short-contact protection",
                "do not lower a hold threshold",
                "do not use the next action or final outcome to label the current event",
                "do not open L3 or R4 from this diagnostic",
            ],
        },
        "provenance": {
            "physical_rollouts_reexecuted": 0,
            "training_jobs": 0,
            "api_calls": 0,
            "reference_files_read_for_alignment_only": ["events.jsonl", "metadata.json"],
        },
        "case_rows": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "attempt_boundary_diagnostic.json", result)
    write_csv(output_root / "attempt_boundary_cases.csv", [
        {
            "rollout_id": case["rollout_id"],
            "root_family_id": case["root_family_id"],
            "reference_group": case["reference_group"],
            "event_time": case["online_contact_loss_time"],
            "reference_event_time": case["reference_event_time"],
            "closed": case["online"]["closed"],
            "contact_present": case["online"]["contact_present"],
            "previous_contact_present": case["online"]["previous_contact_present"],
            "prior_contact_seen": case["online"]["prior_contact_seen"],
            "observation_masked": case["online"]["observation_masked"],
            "command_target_delta_norm": case["command"]["command_target_delta_norm"],
            "command_target_delta_xy_norm": case["command"]["command_target_delta_xy_norm"],
            "command_target_delta_z": case["command"]["command_target_delta_z"],
            "command_gripper": case["command"]["command_gripper"],
        }
        for case in cases
    ])
    report = [
        "# Attempt-boundary diagnostic",
        "",
        "- Status: `L2RAR1_PARTIAL_KEEP_G1`; L3 and R4 remain closed.",
        f"- Scope: `{split}` cached cases only; `{len(cases)}` cases, no physical re-execution.",
        "- At first contact loss, the common online state is identical across the missed-grasp and short-touch groups.",
        f"- The already-arrived low-level command target is stationary for all missed-grasp cases and moving for all short-touch cases in this fixed cache: `{command_state_separates}`.",
        "",
        "## Interpretation",
        "",
        "The existing hold/event interface cannot announce `attempt_end` from its current predicate set alone. The command trace contains a potentially useful causal phase signal, but its perfect separation here is limited to this fixed development cache and must not be promoted directly into a threshold-tuned retry rule.",
        "",
        "The smallest next interface change is to expose an explicit controller-side attempt phase or attempt-end marker derived from command lifecycle. Keep `hold_unknown` separate from `retryable_missed_grasp`; do not remove short-contact protection, lower thresholds, use future action names, or enter L3/R4.",
        "",
        "## Files",
        "",
        "- `attempt_boundary_diagnostic.json`: machine-readable result and case rows.",
        "- `attempt_boundary_cases.csv`: compact per-case comparison table.",
    ]
    (output_root / "attempt_boundary_diagnostic.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="dev_select")
    args = parser.parse_args()
    result = diagnose(args.data_root, args.output_root, args.split)
    print(json.dumps({key: result[key] for key in ("status", "cases", "cases_by_reference_group", "first_contact_loss", "issued_command_at_contact_loss", "decision_boundary")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
