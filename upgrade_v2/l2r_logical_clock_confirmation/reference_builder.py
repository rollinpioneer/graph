from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io_utils import read_csv, read_json, write_csv, write_json


def build_reference(confirmation_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    references, frames, detections, logical, faults = [], [], [], [], []
    for path in sorted(confirmation_root.glob("*/reference/physical_reference.json")):
        rollout = path.parents[1]
        reference = read_json(path)
        reference.update({"rollout_id": rollout.name,
                          "reference_path": os.path.relpath(path, output_root),
                          "physics_trace_path": os.path.relpath(path.parent / "physics_trace.jsonl", output_root),
                          "candidate_input_path": os.path.relpath(rollout / "candidate_input", output_root)})
        references.append(reference)
        frames.extend({"rollout_id": rollout.name, **row}
                      for row in read_csv(rollout / "online_raw/frame_manifest.csv"))
        detector = read_json(rollout / "detector_output/detector_status.json")
        reference["detector_job_complete"] = detector["status"] == "PASS"
        detections.append({"rollout_id": rollout.name, "status": detector["status"],
                           "rows": detector["rows"], "missing_frames": detector["missing_frames"],
                           "errors": len(detector["errors"]), "output_sha256": detector["output_sha256"]})
        logical.extend({"rollout_id": rollout.name, **row}
                       for row in read_csv(rollout / "online_raw/logical_observation_manifest.csv"))
        fault = read_json(rollout / "online_raw/fault_injection_manifest.json")
        if fault["faults"]:
            faults.extend({"rollout_id": rollout.name, "case_id": reference["case_id"], **row}
                          for row in fault["faults"])
        else:
            faults.append({"rollout_id": rollout.name, "case_id": reference["case_id"],
                           "fault_type": None, "reference_unchanged": True})
    fields = ("rollout_id", "family_id", "family_seed", "rollout_seed", "case_id",
              "trace_complete", "frame_manifest_complete", "detector_job_complete", "numeric_health_pass",
              "pre_hold_verified", "physical_loss_confirmed", "state", "reference_action", "resolvable",
              "commanded_release", "force_start_time", "phase_grid_time", "requested_phase_offset_ms",
              "actual_phase_offset_ms", "loss_onset_time_abs", "loss_confirmed_time_abs",
              "peak_relative_separation_m", "reference_path", "physics_trace_path", "candidate_input_path")
    write_csv(output_root / "physical_reference_events.csv", references, fields)
    write_csv(output_root / "rollout_manifest.csv", references, fields[:19])
    write_csv(output_root / "frame_manifest.csv", frames)
    write_csv(output_root / "detection_manifest.csv", detections)
    write_csv(output_root / "logical_observation_manifest.csv", logical)
    write_csv(output_root / "fault_injection_manifest.csv", faults)
    write_json(output_root / "physical_reference_index.json",
               {"schema": "l2rar2_r20_physical_reference_index_v1", "rows": references})
    return {"schema": "l2rar2_r20_reference_build_v1", "status": "PASS" if len(references) == 72 else "FAIL",
            "rollouts": len(references)}
