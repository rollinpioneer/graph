from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

from upgrade_v2.visual_refine_l2.vision import detect_frame

from .io_utils import read_csv, sha256, write_json, write_jsonl


def detect_rollout(rollout_root: Path) -> dict[str, Any]:
    frames = read_csv(rollout_root / "online_raw/frame_manifest.csv")
    rows, errors = [], []
    for frame in frames:
        missing = str(frame.get("frame_missing", "")).lower() == "true"
        base = {"time": float(frame["time"]), "capture_order": int(frame["capture_order"]),
                "frame_path": frame.get("jpeg_path") or None, "frame_missing": missing,
                "detector_error": None, "width": 192, "height": 144}
        if missing:
            rows.append({**base, "object_centroid": None, "object_area": 0.0,
                         "object_component_count": 0, "object_confidence": 0.0,
                         "gripper_centroid": None, "gripper_area": 0.0, "gripper_confidence": 0.0})
            continue
        try:
            result = detect_frame(rollout_root / str(frame["jpeg_path"]))
            rows.append({**base, "object_centroid": result.get("object_centroid"),
                         "object_area": result.get("object_area", 0.0),
                         "object_component_count": result.get("object_component_count", 0),
                         "object_confidence": result.get("object_confidence", 0.0),
                         "gripper_centroid": result.get("gripper_centroid"),
                         "gripper_area": result.get("gripper_area", 0.0),
                         "gripper_confidence": result.get("gripper_confidence", 0.0),
                         "width": result.get("width", 192), "height": result.get("height", 144)})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"; errors.append({"capture_order": base["capture_order"], "error": error})
            rows.append({**base, "detector_error": error, "object_centroid": None, "object_area": 0.0,
                         "object_component_count": 0, "object_confidence": 0.0,
                         "gripper_centroid": None, "gripper_area": 0.0, "gripper_confidence": 0.0})
    output = rollout_root / "detector_output"; output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "detections_rgb.jsonl", rows)
    status = {"schema": "l2rar2_r20_detector_status_v1",
              "status": "PASS" if not errors and len(rows) == len(frames) else "FAIL",
              "rows": len(rows), "missing_frames": sum(row["frame_missing"] for row in rows),
              "errors": errors, "output_sha256": sha256(output / "detections_rgb.jsonl")}
    write_json(output / "detector_status.json", status)
    return {"rollout_id": rollout_root.name, **status}


def detect_confirmation(confirmation_root: Path, workers: int = 4) -> dict[str, Any]:
    roots = [path.parent for path in sorted(confirmation_root.glob("*/metadata.json"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        rows = list(pool.map(detect_rollout, roots))
    result = {"schema": "l2rar2_r20_detection_result_v1",
              "status": "PASS" if len(rows) == 72 and all(row["status"] == "PASS" for row in rows) else "FAIL",
              "rollouts": len(rows), "detector_errors": sum(len(row["errors"]) for row in rows), "rows": rows}
    write_json(confirmation_root / "detection_status.json", result)
    return result
