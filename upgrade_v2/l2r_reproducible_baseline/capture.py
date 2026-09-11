from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .fingerprint import checkpoint_state
from .protocol import canonical_hash


class ReadOnlyPhysicsRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def before_step(self, sim: Any, context: dict[str, Any]) -> None:
        self.rows.append({"phase": "before", **context, "time": float(sim.data.time), "state_sha256": _state_bytes_hash(sim)})

    def after_step(self, sim: Any, context: dict[str, Any]) -> None:
        self.rows.append({"phase": "after", **context, "time": float(sim.data.time), "state_sha256": _state_bytes_hash(sim)})


def _state_bytes_hash(sim: Any) -> str:
    digest = hashlib.sha256()
    for value in (sim.data.qpos, sim.data.qvel, sim.data.qacc, sim.data.qacc_warmstart, sim.data.mocap_pos, sim.data.mocap_quat, sim.data.eq_active, sim.model.eq_data, sim.data.xfrc_applied):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(float(sim.data.time).hex().encode("ascii"))
    return digest.hexdigest()


class CaptureRecorder:
    def __init__(self, sim: Any, renderer: Any, output_root: Path, spec: Any) -> None:
        self.sim = sim
        self.renderer = renderer
        self.output_root = output_root
        self.spec = spec
        self.states: list[dict[str, Any]] = []
        self.captures: list[dict[str, Any]] = []
        self.sequence = 0
        self.capture_order = 0

    def checkpoint(self, sampling_point: str, action: str | None) -> dict[str, Any]:
        row = checkpoint_state(self.sim, sequence=self.sequence, sampling_point=sampling_point, action=action)
        self.states.append(row)
        self.sequence += 1
        return row

    def callback(self, phase: str, action: str) -> dict[str, Any]:
        frame = self.renderer.render(self.sim.data, "front", self.spec.camera_jitter)
        frame = np.asarray(frame, dtype=np.uint8)
        raw_sha = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
        path = self.output_root / "rgb" / "front" / f"frame_{self.capture_order:05d}.jpg"
        self.renderer.save_jpeg(frame, path)
        row = self.checkpoint(f"{phase}_callback", action)
        jpeg_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        record = {
            "capture_order": self.capture_order, "phase": phase, "action": action,
            "action_index": int(self.sim.action_index), "time": float(self.sim.data.time),
            "raw_rgb_sha256": raw_sha, "jpeg_sha256": jpeg_sha,
            "shape": list(frame.shape), "dtype": str(frame.dtype), "image_path": str(path.resolve()),
            "checkpoint_sequence": row["sequence"],
            "callback_steps": ["callback_enter", "render_start", "render_end", "raw_hash_computed", "jpeg_saved", "state_saved", "callback_exit"],
        }
        self.captures.append(record)
        self.capture_order += 1
        return record


def detection_record(capture: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
    row = {"capture_order": capture["capture_order"], "object_centroid": detected.get("object_centroid"), "gripper_centroid": detected.get("gripper_centroid"), "object_confidence": detected.get("object_confidence"), "gripper_confidence": detected.get("gripper_confidence"), "width": detected.get("width"), "height": detected.get("height")}
    row["detection_canonical_sha256"] = canonical_hash(row)
    return row
