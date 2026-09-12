from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.vision import detect_frame

from .io_utils import read_csv, read_json, sha256_file, write_json, write_jsonl


def detect_rollout(rollout_root: Path) -> dict[str, Any]:
    manifest = read_csv(rollout_root / "online_raw/frame_manifest.csv")
    rows = []
    errors = []
    for frame in manifest:
        missing = str(frame.get("frame_missing", "")).lower() == "true"
        base = {
            "time": float(frame["time"]), "capture_order": int(frame["capture_order"]),
            "frame_path": frame.get("jpeg_path") or None, "frame_missing": missing,
            "detector_error": None, "width": 192, "height": 144,
        }
        if missing:
            rows.append({**base, "object_centroid": None, "object_area": 0.0,
                         "object_component_count": 0, "object_confidence": 0.0,
                         "gripper_centroid": None, "gripper_area": 0.0,
                         "gripper_confidence": 0.0})
            continue
        path = rollout_root / str(frame["jpeg_path"])
        try:
            result = detect_frame(path)
            rows.append({**base,
                         "object_centroid": result.get("object_centroid"),
                         "object_area": result.get("object_area", 0.0),
                         "object_component_count": result.get("object_component_count", 0),
                         "object_confidence": result.get("object_confidence", 0.0),
                         "gripper_centroid": result.get("gripper_centroid"),
                         "gripper_area": result.get("gripper_area", 0.0),
                         "gripper_confidence": result.get("gripper_confidence", 0.0),
                         "width": result.get("width", 192), "height": result.get("height", 144)})
        except Exception as exc:  # detector failures are recorded, never oracle-filled
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"capture_order": base["capture_order"], "error": error})
            rows.append({**base, "detector_error": error, "object_centroid": None,
                         "object_area": 0.0, "object_component_count": 0,
                         "object_confidence": 0.0, "gripper_centroid": None,
                         "gripper_area": 0.0, "gripper_confidence": 0.0})
    candidate = rollout_root / "candidate_input"
    candidate.mkdir(parents=True, exist_ok=True)
    write_jsonl(candidate / "detections_rgb.jsonl", rows)
    for name in ("contact_proxy.jsonl", "attempt_lifecycle.jsonl", "gripper_commands.jsonl", "request_context.jsonl"):
        shutil.copyfile(rollout_root / "online_raw" / name, candidate / name)
    reference_path = rollout_root / "reference/physical_reference.json"
    reference = read_json(reference_path)
    reference["detector_job_complete"] = not errors and len(rows) == len(manifest)
    reference["detector_rows"] = len(rows)
    reference["missing_frames"] = sum(bool(row["frame_missing"]) for row in rows)
    write_json(reference_path, reference)
    status = {"status": "PASS" if reference["detector_job_complete"] else "FAIL",
              "rows": len(rows), "errors": errors,
              "output_sha256": sha256_file(candidate / "detections_rgb.jsonl")}
    write_json(candidate / "detector_status.json", status)
    return status


def detect_confirmation(confirmation_root: Path) -> list[dict[str, Any]]:
    results = []
    for metadata in sorted(confirmation_root.glob("*/metadata.json")):
        results.append({"rollout": metadata.parent.name, **detect_rollout(metadata.parent)})
    return results
