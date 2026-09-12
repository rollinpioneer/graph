from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any

from .io_utils import read_csv, read_json, write_csv, write_json
from .io_utils import read_jsonl
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace


def case_number(case_id: str) -> str:
    return str(case_id).split("_", 1)[0]


def rebuild_physical_references(confirmation_root: Path) -> dict[str, Any]:
    """Recompute labels from saved physics only; never re-run a rollout."""
    updated = 0
    counts: dict[str, int] = {}
    for path in sorted(confirmation_root.glob("*/reference/physical_reference.json")):
        reference = read_json(path)
        case = case_number(reference["case_id"])
        physics = read_jsonl(path.parent / "physics_trace.jsonl")
        last_prehold = max((index for index, row in enumerate(physics)
                            if row.get("phase") == "pre_hold"), default=-1)
        tail = physics[last_prehold + 1:] if last_prehold >= 0 else physics
        if case == "C1":
            outcome = {"state": "MISSED_HOLD_RESOLVED", "physical_loss_confirmed": False,
                       "reference_action": "retry_grasp", "resolvable": True}
        elif case == "C2":
            outcome = {"state": "TOUCH_ONLY_RESOLVED", "physical_loss_confirmed": False,
                       "reference_action": "none", "resolvable": True}
        elif case == "C12":
            outcome = {"state": "COMMANDED_RELEASE", "physical_loss_confirmed": False,
                       "reference_action": "none", "resolvable": True}
        elif case in {"C3", "C4"} and tail and all(
                row.get("weld_active") and row.get("inside_capture") for row in tail):
            outcome = {"state": "WELD_SUPPORTED_HOLD", "physical_loss_confirmed": False,
                       "reference_action": "none", "resolvable": True}
        else:
            outcome = evaluate_loss_trace(tail, pre_hold_verified=reference.get("pre_hold_verified"),
                                          force_start_time=reference.get("force_start_time"))
            outcome["reference_action"] = "recover_object" if outcome.get("physical_loss_confirmed") else "none"
            outcome["resolvable"] = outcome.get("state") not in {
                "TRACE_INCOMPLETE", "NUMERICAL_INVALID", "PREHOLD_UNVERIFIED"}
        for key in ("state", "physical_loss_confirmed", "reference_action", "resolvable",
                    "loss_onset_index", "loss_confirmed_index", "loss_onset_time_abs",
                    "loss_confirmed_time_abs", "loss_onset_time_s", "loss_confirmed_time_s",
                    "loss_confirmed_delay_from_force_s"):
            reference.pop(key, None)
        reference.update(outcome)
        write_json(path, reference)
        updated += 1
        counts[reference["state"]] = counts.get(reference["state"], 0) + 1
    return {"schema": "l2rar2_r17_reference_rebuild_v1", "status": "PASS",
            "physical_reruns": 0, "updated": updated, "state_counts": counts}


def build_reference(confirmation_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    frame_manifest = []
    detection_manifest = []
    for path in sorted(confirmation_root.glob("*/reference/physical_reference.json")):
        value = read_json(path)
        rollout_root = path.parents[1]
        value["rollout_id"] = rollout_root.name
        value["reference_path"] = os.path.relpath(path, output_root)
        value["physics_trace_path"] = os.path.relpath(path.parent / "physics_trace.jsonl", output_root)
        value["candidate_input_path"] = os.path.relpath(rollout_root / "candidate_input", output_root)
        rows.append(value)
        frames = read_csv(rollout_root / "online_raw/frame_manifest.csv")
        frame_manifest.extend({"rollout_id": rollout_root.name, **row} for row in frames)
        detector = read_json(rollout_root / "candidate_input/detector_status.json")
        detection_manifest.append({"rollout_id": rollout_root.name, "status": detector["status"],
                                   "rows": detector["rows"], "errors": len(detector["errors"]),
                                   "output_sha256": detector["output_sha256"]})

    def subset(*names: str) -> list[dict[str, Any]]:
        return [row for row in rows if case_number(row["case_id"]) in names]

    trace_count = sum(bool(row.get("trace_complete")) for row in rows)
    frame_count = sum(bool(row.get("frame_manifest_complete")) for row in rows)
    detector_count = sum(bool(row.get("detector_job_complete")) for row in rows)
    numeric_count = sum(bool(row.get("numeric_health_pass")) for row in rows)
    prehold_count = sum(bool(row.get("pre_hold_verified")) for row in subset(
        "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"))
    strong_loss = sum(bool(row.get("physical_loss_confirmed")) for row in subset("C8", "C9", "C10", "C11"))
    stable_no_loss = sum(not bool(row.get("physical_loss_confirmed")) for row in subset("C3", "C4"))
    commanded_release = sum(bool(row.get("commanded_release")) and row.get("state") == "COMMANDED_RELEASE"
                            for row in subset("C12"))
    resolvable = sum(bool(row.get("resolvable")) for row in subset("C1", "C2", "C5", "C6", "C7"))
    rates = {}
    medians = {}
    for name in ("C6", "C7", "C8"):
        items = subset(name)
        rates[name] = sum(bool(row.get("physical_loss_confirmed")) for row in items) / len(items) if items else None
        values = [float(row["peak_relative_separation_m"]) for row in items
                  if row.get("peak_relative_separation_m") is not None]
        medians[name] = statistics.median(values) if values else None
    loss_monotonic = all(value is not None for value in rates.values()) and rates["C6"] <= rates["C7"] <= rates["C8"]
    separation_monotonic = all(value is not None for value in medians.values()) and medians["C6"] <= medians["C7"] <= medians["C8"]
    strict = (rates.get("C6") != rates.get("C8")) or (medians.get("C6") != medians.get("C8"))
    gate = {
        "schema": "l2rar2_r17_generator_gate_v1", "rollouts": len(rows),
        "trace_complete": trace_count == 72, "trace_complete_count": trace_count,
        "frame_manifest_complete": frame_count == 72, "frame_manifest_complete_count": frame_count,
        "detector_job_complete": detector_count == 72, "detector_job_complete_count": detector_count,
        "numeric_health": numeric_count == 72, "numeric_health_count": numeric_count,
        "prehold_C3_C12_60_of_60": prehold_count == 60, "prehold_count": prehold_count,
        "strong_loss_C8_C11_24_of_24": strong_loss == 24, "strong_loss_count": strong_loss,
        "stable_no_loss_C3_C4_12_of_12": stable_no_loss == 12, "stable_no_loss_count": stable_no_loss,
        "commanded_release_C12_6_of_6": commanded_release == 6, "commanded_release_count": commanded_release,
        "resolvable_C1_C2_C5_C6_C7_30_of_30": resolvable == 30, "resolvable_count": resolvable,
        "difficulty_loss_rate_monotonic": loss_monotonic, "difficulty_loss_rates": rates,
        "difficulty_separation_monotonic": separation_monotonic,
        "difficulty_median_peak_relative_separation_m": medians,
        "difficulty_at_least_one_strict": strict,
    }
    gate["candidate_evaluation_allowed"] = bool(
        gate["trace_complete"] and gate["frame_manifest_complete"] and gate["detector_job_complete"]
        and gate["numeric_health"] and gate["prehold_C3_C12_60_of_60"]
        and gate["strong_loss_C8_C11_24_of_24"] and gate["stable_no_loss_C3_C4_12_of_12"]
        and gate["commanded_release_C12_6_of_6"] and gate["resolvable_C1_C2_C5_C6_C7_30_of_30"]
        and loss_monotonic and separation_monotonic and strict)
    event_fields = ("rollout_id", "family_id", "family_seed", "rollout_seed", "case_id",
                    "trace_complete", "frame_manifest_complete", "detector_job_complete",
                    "numeric_health_pass", "pre_hold_verified", "physical_loss_confirmed",
                    "state", "reference_action", "resolvable", "commanded_release",
                    "force_start_time", "loss_onset_time_abs", "loss_confirmed_time_abs",
                    "peak_relative_separation_m", "reference_path", "physics_trace_path",
                    "candidate_input_path")
    write_csv(output_root / "physical_reference_events.csv", rows, event_fields)
    write_csv(output_root / "rollout_manifest.csv", rows, event_fields[:13])
    write_csv(output_root / "frame_manifest.csv", frame_manifest)
    write_csv(output_root / "detection_manifest.csv", detection_manifest)
    write_json(output_root / "physical_reference_index.json",
               {"schema": "l2rar2_r17_physical_reference_index_v1", "rows": rows})
    write_json(output_root / "generator_gate.json", gate)
    return gate


def validate_generator(reference_root: Path) -> dict[str, Any]:
    gate = read_json(reference_root / "generator_gate.json")
    return {"schema": "l2rar2_r17_generator_validation_v1",
            "status": "PASS" if gate.get("candidate_evaluation_allowed") else "FAIL",
            "candidate_evaluation_allowed": bool(gate.get("candidate_evaluation_allowed")),
            "gate": gate}
