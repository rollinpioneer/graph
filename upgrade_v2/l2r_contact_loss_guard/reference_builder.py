from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io_utils import read_csv, read_json, write_csv, write_json


def case_number(case_id: str) -> str:
    return str(case_id).split("_", 1)[0]


def build_reference(confirmation_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    detectors: list[dict[str, Any]] = []
    for path in sorted(confirmation_root.glob("*/reference/physical_reference.json")):
        row = read_json(path)
        rollout = path.parents[1]
        row.update({"rollout_id": rollout.name,
                    "reference_path": os.path.relpath(path, output_root),
                    "physics_trace_path": os.path.relpath(path.parent / "physics_trace.jsonl", output_root),
                    "candidate_input_path": os.path.relpath(rollout / "candidate_input", output_root)})
        rows.append(row)
        frames.extend({"rollout_id": rollout.name, **frame}
                      for frame in read_csv(rollout / "online_raw/frame_manifest.csv"))
        status = read_json(rollout / "candidate_input/detector_status.json")
        detectors.append({"rollout_id": rollout.name, "status": status["status"],
                          "rows": status["rows"], "errors": len(status["errors"]),
                          "output_sha256": status["output_sha256"]})

    subset = lambda *names: [row for row in rows if case_number(row["case_id"]) in names]
    trace = sum(bool(row.get("trace_complete")) for row in rows)
    detector = sum(bool(row.get("detector_job_complete")) for row in rows)
    numeric = sum(bool(row.get("numeric_health_pass")) for row in rows)
    prehold = sum(bool(row.get("pre_hold_verified")) for row in rows)
    strong = sum(bool(row.get("physical_loss_confirmed")) for row in subset("T6", "T7", "T8", "T9", "T10", "T11"))
    stable = sum(not bool(row.get("physical_loss_confirmed")) and bool(row.get("resolvable"))
                 for row in subset("T1", "T2", "T3"))
    release = sum(row.get("state") == "COMMANDED_RELEASE" and bool(row.get("commanded_release"))
                  for row in subset("T12"))
    variable = sum(bool(row.get("resolvable")) for row in subset("T4", "T5"))
    offsets = {offset: sum(bool(row.get("trace_complete")) and row.get("phase_offset_ms") == offset
                           for row in rows) for offset in (0, 10, 20, 30, 40)}
    t11_missing = sum(str(frame.get("frame_missing", "")).lower() == "true"
                      for frame in frames if case_number(frame["rollout_id"].split("__", 1)[1]) == "T11")
    gate = {
        "schema": "l2rar2_r18_generator_gate_v1", "rollouts": len(rows),
        "trace_complete_72_of_72": trace == 72, "trace_complete_count": trace,
        "detector_complete_72_of_72": detector == 72, "detector_complete_count": detector,
        "numeric_health_72_of_72": numeric == 72, "numeric_health_count": numeric,
        "prehold_72_of_72": prehold == 72, "prehold_count": prehold,
        "strong_loss_T6_T11_36_of_36": strong == 36, "strong_loss_count": strong,
        "stable_no_loss_T1_T3_18_of_18": stable == 18, "stable_no_loss_count": stable,
        "commanded_release_T12_6_of_6": release == 6, "commanded_release_count": release,
        "T4_T5_resolved_12_of_12": variable == 12, "T4_T5_resolved_count": variable,
        "phase_offsets_6_of_6_each": all(value == 6 for value in offsets.values()),
        "phase_offset_counts": offsets, "T11_rgb_missing_18_total": t11_missing == 18,
        "T11_rgb_missing_count": t11_missing,
    }
    gate["candidate_evaluation_allowed"] = all(value is True for key, value in gate.items()
                                                if key.endswith(("_of_72", "_of_36", "_of_18", "_of_6", "_of_12", "_each", "_total")))
    fields = ("rollout_id", "family_id", "family_seed", "rollout_seed", "case_id",
              "trace_complete", "detector_job_complete", "numeric_health_pass", "pre_hold_verified",
              "physical_loss_confirmed", "state", "reference_action", "resolvable", "commanded_release",
              "force_start_time", "phase_offset_ms", "loss_onset_time_abs", "loss_confirmed_time_abs",
              "reference_path", "physics_trace_path", "candidate_input_path")
    write_csv(output_root / "physical_reference_events.csv", rows, fields)
    write_csv(output_root / "rollout_manifest.csv", rows, fields[:14])
    write_csv(output_root / "frame_manifest.csv", frames)
    write_csv(output_root / "detection_manifest.csv", detectors)
    write_json(output_root / "physical_reference_index.json",
               {"schema": "l2rar2_r18_physical_reference_index_v1", "rows": rows})
    write_json(output_root / "generator_gate.json", gate)
    return gate


def validate_generator(reference_root: Path) -> dict[str, Any]:
    gate = read_json(reference_root / "generator_gate.json")
    return {"schema": "l2rar2_r18_generator_validation_v1",
            "status": "PASS" if gate.get("candidate_evaluation_allowed") else "FAIL",
            "candidate_evaluation_allowed": bool(gate.get("candidate_evaluation_allowed")), "gate": gate}
