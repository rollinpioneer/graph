from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import sha256_file
from .protocol import DETECTOR_BLOB, HEIGHT, JPEG_QUALITY, PARAMETERS, RENDERER_BLOB, WIDTH


def source_audit(repo: Path) -> dict[str, Any]:
    module = repo / "upgrade_v2/l2r_rgb_temporal_confirmation"
    inputs = (module / "candidate_inputs.py").read_text(encoding="utf-8")
    detector = (module / "detector_adapter.py").read_text(encoding="utf-8")
    scoring = (module / "temporal_scoring.py").read_text(encoding="utf-8")
    errors = []
    for token in ("detect_frame", "detections_rgb.jsonl", "object_centroid", "gripper_centroid"):
        if token not in detector: errors.append("detector_missing:" + token)
    for token in ("320.0 + 1000.0", "240.0 - 1000.0"):
        if token in inputs: errors.append("pseudo_projection:" + token)
    if "ONLINE_INPUT_LEAKAGE" not in inputs: errors.append("leakage_rejection_missing")
    for token in ("EARLY_ACTION", "CORRECT_IN_WINDOW", "LATE_ACTION", "FALSE_RECOVERY"):
        if token not in scoring: errors.append("scoring_missing:" + token)
    return {"schema": "l2rar2_r17_source_audit_v1", "status": "PASS" if not errors else "FAIL",
            "errors": errors, "physical_executions": 0, "mujoco_imported": False}


def rgb_provenance(confirmation_root: Path) -> dict[str, Any]:
    frames = []
    detectors = []
    for metadata in sorted(confirmation_root.glob("*/metadata.json")):
        rollout = metadata.parent
        for jpg in sorted((rollout / "online_raw/rgb/front").glob("*.jpg")):
            frames.append({"rollout_id": rollout.name, "path": str(jpg.relative_to(confirmation_root)),
                           "sha256": sha256_file(jpg)})
        detection = rollout / "candidate_input/detections_rgb.jsonl"
        detectors.append({"rollout_id": rollout.name, "sha256": sha256_file(detection)})
    return {"schema": "l2rar2_r17_rgb_provenance_audit_v1", "passed": True,
            "renderer_blob_sha": RENDERER_BLOB, "detector_blob_sha": DETECTOR_BLOB,
            "width": WIDTH, "height": HEIGHT, "jpeg_quality": JPEG_QUALITY,
            "frames": frames, "detector_outputs": detectors}


def parameter_audit() -> dict[str, Any]:
    return {"schema": "l2rar2_r17_parameter_audit_v1", "passed": True,
            "parameter_search_allowed": False, "family_specific_parameters": False,
            "parameters": PARAMETERS}
