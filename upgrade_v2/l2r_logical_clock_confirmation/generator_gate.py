from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input

from .io_utils import read_csv, read_json, read_jsonl, write_json


def _case(reference: dict[str, Any], number: int) -> bool:
    return str(reference["case_id"]).startswith(f"T{number}_")


def _clock_coverage(confirmation_root: Path, references: list[dict[str, Any]]) -> dict[str, Any]:
    same_positive, later_positive, t4_clear = [], [], []
    by_id = {row["rollout_id"]: row for row in references}
    for output in sorted(confirmation_root.glob("*/candidate_output/O_C3_CLP2_LOGICAL_CLOCK.jsonl")):
        rollout_id = output.parents[1].name
        rows = read_jsonl(output)
        observations = load_candidate_input(output.parents[1] / "candidate_input")
        if len(rows) != len(observations):
            raise RuntimeError(f"CANDIDATE_OUTPUT_ALIGNMENT_ERROR:{rollout_id}")
        reference = by_id[rollout_id]
        for index, (previous, current) in enumerate(zip(rows, rows[1:])):
            previous_observation, current_observation = observations[index:index + 2]
            if current.get("guard_state") == "CONFIRMED":
                item = {"rollout_id": rollout_id, "case_id": reference["case_id"],
                        "pending_time": previous.get("time"), "confirming_time": current.get("time"),
                        "pending_capture_order": previous.get("capture_order"),
                        "confirming_capture_order": current.get("capture_order"),
                        "pending_contact": previous_observation.get("contact_present"),
                        "confirming_contact": current_observation.get("contact_present")}
                if (float(current["time"]) == float(previous["time"])
                        and previous_observation.get("contact_present") is False
                        and current_observation.get("contact_present") is False):
                    same_positive.append(item)
                elif float(current["time"]) > float(previous["time"]): later_positive.append(item)
        if _case(reference, 4):
            pending = next((index for index, row in enumerate(rows) if row.get("guard_state") == "PENDING"), None)
            if pending is not None and pending + 1 < len(rows):
                following = rows[pending + 1]
                if (float(following["time"]) == float(rows[pending]["time"])
                        and observations[pending + 1].get("contact_present") is True
                        and following.get("selected_action") == "none"
                        and following.get("guard_state") == "CLEAR"):
                    t4_clear.append({"rollout_id": rollout_id, "first_order": rows[pending]["capture_order"],
                                     "second_order": following["capture_order"], "time": following["time"]})
    return {"schema": "l2rar2_r20_logical_clock_coverage_v1",
            "same_time_false_false_positive_groups": same_positive,
            "same_time_false_false_positive_count": len(same_positive),
            "strictly_later_confirmation_groups": later_positive,
            "strictly_later_confirmation_count": len(later_positive),
            "same_time_false_true_clear_groups": t4_clear,
            "same_time_false_true_clear_count": len(t4_clear)}


def validate_generator(confirmation_root: Path, reference_root: Path, output: Path) -> dict[str, Any]:
    references = read_json(reference_root / "physical_reference_index.json")["rows"]
    frames = read_csv(reference_root / "frame_manifest.csv")
    detections = read_csv(reference_root / "detection_manifest.csv")
    faults = read_csv(reference_root / "fault_injection_manifest.csv")
    subset = lambda *numbers: [row for row in references if any(_case(row, number) for number in numbers)]
    trace = sum(bool(row.get("trace_complete")) for row in references)
    detector = sum(str(row.get("status")) == "PASS" for row in detections)
    numeric = sum(bool(row.get("numeric_health_pass")) for row in references)
    prehold = sum(bool(row.get("pre_hold_verified")) for row in references)
    strong_7_10 = sum(bool(row.get("physical_loss_confirmed")) for row in subset(7, 8, 9, 10))
    strong_11 = sum(bool(row.get("physical_loss_confirmed")) for row in subset(11))
    no_loss = sum(not bool(row.get("physical_loss_confirmed")) and bool(row.get("resolvable")) for row in subset(1, 2, 3, 4))
    release = sum(row.get("state") == "COMMANDED_RELEASE" and bool(row.get("commanded_release")) for row in subset(12))
    resolved_5_6 = sum(bool(row.get("resolvable")) for row in subset(5, 6))
    phase_counts = {}
    for number, offset in zip((7, 8, 9, 10, 11), (0, 10, 20, 30, 40)):
        group = subset(number)
        phase_counts[str(offset)] = sum(row.get("requested_phase_offset_ms") == offset
                                        and abs(float(row.get("actual_phase_offset_ms")) - offset) <= 1e-6 for row in group)
    t3_faults = [row for row in faults if row.get("fault_type") == "T3_SINGLE_FALSE"]
    t4_faults = [row for row in faults if row.get("fault_type") == "T4_SAME_TIME_FALSE_TRUE"]
    missing = [row for row in frames if str(row.get("frame_missing", "")).lower() == "true"]
    planned = [row for row in missing if row.get("frame_missing_reason") == "T11_SUCCESSOR_PERIODIC_RGB_DROPOUT"]
    detector_errors = sum(int(float(row.get("errors") or 0)) for row in detections)
    coverage = _clock_coverage(confirmation_root, references)
    write_json(reference_root / "logical_clock_coverage.json", coverage)
    gate = {"schema": "l2rar2_r20_generator_gate_v1", "rollouts": len(references),
            "trace_complete_72_of_72": trace == 72, "trace_complete_count": trace,
            "detector_complete_72_of_72": detector == 72, "detector_complete_count": detector,
            "numeric_health_72_of_72": numeric == 72, "numeric_health_count": numeric,
            "prehold_72_of_72": prehold == 72, "prehold_count": prehold,
            "strong_loss_T7_T10_24_of_24": strong_7_10 == 24, "strong_loss_T7_T10_count": strong_7_10,
            "strong_loss_T11_6_of_6": strong_11 == 6, "strong_loss_T11_count": strong_11,
            "no_loss_T1_T4_24_of_24": no_loss == 24, "no_loss_T1_T4_count": no_loss,
            "commanded_release_T12_6_of_6": release == 6, "commanded_release_count": release,
            "resolved_T5_T6_12_of_12": resolved_5_6 == 12, "resolved_T5_T6_count": resolved_5_6,
            "phase_offsets_6_of_6_each": all(value == 6 for value in phase_counts.values()),
            "phase_offset_counts": phase_counts,
            "T3_single_false_6_of_6": len(t3_faults) == 6,
            "T3_next_later_restored_6_of_6": len(t3_faults) == 6 and all(float(row["next_later_source_time"]) > float(row["source_time"]) for row in t3_faults),
            "T4_same_time_false_true_6_of_6": len(t4_faults) == 6,
            "T4_same_jpeg_detection_6_of_6": len(t4_faults) == 6 and all(bool(row.get("same_jpeg_sha256")) for row in t4_faults),
            "fault_reference_unchanged_12_of_12": len(t3_faults) + len(t4_faults) == 12 and all(str(row.get("reference_unchanged")).lower() == "true" for row in t3_faults + t4_faults),
            "T11_planned_missing_18": len(planned) == 18, "planned_missing_count": len(planned),
            "unplanned_missing_0": len(missing) == len(planned), "unplanned_missing_count": len(missing) - len(planned),
            "detector_errors_0": detector_errors == 0, "detector_error_count": detector_errors,
            "same_time_false_false_positive_min_1": coverage["same_time_false_false_positive_count"] >= 1,
            "same_time_false_true_clear_6_of_6": coverage["same_time_false_true_clear_count"] == 6,
            "strictly_later_confirmation_min_1": coverage["strictly_later_confirmation_count"] >= 1}
    gates = [value for key, value in gate.items() if key not in {"schema", "rollouts", "phase_offset_counts"}
             and not key.endswith(("_count", "count")) and isinstance(value, bool)]
    gate["candidate_evaluation_allowed"] = len(references) == 72 and all(gates)
    write_json(output, gate)
    return gate
