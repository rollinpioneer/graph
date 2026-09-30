"""Task-local RGB-D, fact, and independent task-success bindings for Family B."""
from __future__ import annotations

import time
import numpy as np

from cp_disr.facts import FactRecord, Truth
from .family_b_env import MOVABLE, PALETTE, PAD_HALF, RECEIVER_INNER
from .perception import (PerceptionAdapter, PerceptionResult, THRESHOLDS,
                         VERSION, _metric_depth, backproject_mask,
                         camera_calibration, sha)
from .task_evaluator import TaskEvaluator


def _palette_mask(rgb, name):
    image = np.asarray(rgb, dtype=np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    color = PALETTE[name][:3]
    own = np.linalg.norm(image - color, axis=2)
    mask = own < THRESHOLDS["color_tol"]
    for other_name, other in PALETTE.items():
        if other_name != name:
            mask &= own <= np.linalg.norm(image - other[:3], axis=2)
    return mask


class FamilyBPerception(PerceptionAdapter):
    recorder = None
    variant = "FAMILY_B_TASK_LOCAL_PALETTE_V1"

    def infer(self, observation):
        if not isinstance(observation, dict) or "rgb" not in observation:
            observation = self.env.public_observation()
        rgb = np.asarray(observation["rgb"])
        depth = _metric_depth(self.env, observation["depth"])
        calib = camera_calibration(self.env)
        self.calibration_hash = sha(calib)
        blobs, xyz, reasons = {}, {}, {}
        for name in PALETTE:
            rec = backproject_mask(_palette_mask(rgb, name), depth, calib)
            if rec is None:
                blobs[name] = None
                reasons[name] = "insufficient_color_depth_support"
            else:
                blobs[name] = {
                    "xyz": rec["xyz"].tolist(), "pixels": rec["pixels"],
                    "std": rec["std"].tolist(), "source": "rgb_depth",
                }
                xyz[name] = rec["xyz"]
        # The fixed fixtures have registered, public calibrated poses.
        layout = self.env.public_layout()
        for name in ("receiver", "pad_u", "pad_v"):
            if name not in xyz:
                xyz[name] = np.asarray(layout[name], dtype=float)
                blobs[name] = {
                    "xyz": xyz[name].tolist(), "pixels": 0, "std": [0, 0, 0],
                    "source": "static_layout",
                }
                reasons.pop(name, None)
        self.env.last_perception = xyz
        meas = {
            "blobs": blobs, "unknown_reasons": reasons,
            "eef_pos": np.asarray(observation["eef_pos"], dtype=float).tolist(),
            "gripper_qpos": np.asarray(observation["gripper_qpos"], dtype=float).tolist(),
            "calib": {k: calib[k] for k in ("fovy", "width", "height")},
            "depth_stats": {
                "min": float(np.nanmin(depth)), "max": float(np.nanmax(depth)),
                "mean": float(np.nanmean(depth)),
            },
        }
        if self.recorder is not None:
            self.recorder.perception_frame(
                rgb, depth, calib, layout, meas, self.variant,
                self.calibration_hash)
        return PerceptionResult(
            meas, self.checkpoint_hash, self.calibration_hash,
            ("agentview_rgb", "agentview_depth", "camera_calibration"), VERSION,
        )


class FamilyBVerifier:
    """Dynamic facts only from observed RGB-D and gripper; no nominal fill."""

    recorder = None

    def __init__(self, env):
        self.env = env
        self.prev = {}

    def verify(self, measurement, execution=None):
        m = measurement.measurements
        blobs = m.get("blobs", {})
        eef = m.get("eef_pos")
        grip = m.get("gripper_qpos") or [0.04, -0.04]
        grip_open = abs(float(grip[0])) > THRESHOLDS["gripper_open"]
        layout = self.env.public_layout()
        table_z = float(layout["table_top_z"])
        now = time.time()
        recs = []

        def pos(name):
            b = blobs.get(name)
            return None if not b else b.get("xyz")

        def add(fid, value, reason, evidence):
            previous = self.prev.get(fid, Truth.UNKNOWN)
            rec = FactRecord(
                fact_id=fid, value=value, capture_time=now, available_time=now,
                evidence_ids=tuple(evidence),
                last_confirmed_value=value if value != Truth.UNKNOWN else previous,
                last_confirmed_time=now if value != Truth.UNKNOWN else None,
                reason="family-b-verifier-v1:" + reason,
            )
            recs.append(rec)
            self.prev[fid] = rec.last_confirmed_value

        near = []
        for name in MOVABLE:
            p = pos(name)
            if p is not None and eef is not None:
                dxy = float(np.linalg.norm(np.asarray(p[:2]) - np.asarray(eef[:2])))
                if dxy < THRESHOLDS["held_xy"] and abs(p[2]-eef[2]) < THRESHOLDS["held_z"]:
                    near.append(name)
        if grip_open and not near:
            add("p:GripperEmpty", Truth.TRUE, "open_no_nearby_cube", ("gripper", "rgb"))
        elif not grip_open and near:
            add("p:GripperEmpty", Truth.FALSE, "closed_near_cube", ("gripper", "rgb"))
        else:
            add("p:GripperEmpty", Truth.UNKNOWN, "grip_conflict", ("gripper", "rgb"))

        def in_receiver(p):
            c = layout["receiver"]
            return (abs(p[0]-c[0]) <= RECEIVER_INNER[0] + THRESHOLDS["inside_margin"]
                    and abs(p[1]-c[1]) <= RECEIVER_INNER[1] + THRESHOLDS["inside_margin"]
                    and p[2] < table_z + 0.12)

        def on_pad(p, name):
            c = layout[name]
            return (abs(p[0]-c[0]) <= PAD_HALF[0] + THRESHOLDS["buffer_margin"]
                    and abs(p[1]-c[1]) <= PAD_HALF[1] + THRESHOLDS["buffer_margin"]
                    and p[2] < table_z + 0.10)

        for name in MOVABLE:
            p = pos(name)
            ids = {
                "held": f"p:Held:{name}",
                "table": f"p:OnTable:{name}",
                "inside": f"p:Inside:{name}:receiver",
                "u": f"p:AtBuffer:{name}:pad_u",
                "v": f"p:AtBuffer:{name}:pad_v",
            }
            if p is None:
                for fid in ids.values():
                    add(fid, Truth.UNKNOWN, "object_blob_missing", ("rgb",))
                continue
            close = bool(eef is not None and
                         np.linalg.norm(np.asarray(p[:2])-np.asarray(eef[:2]))
                         < THRESHOLDS["held_xy"] and
                         abs(p[2]-eef[2]) < THRESHOLDS["held_z"])
            held = (Truth.TRUE if not grip_open and close else
                    Truth.FALSE if grip_open and not close else Truth.UNKNOWN)
            inside = bool(in_receiver(p))
            at_u, at_v = on_pad(p, "pad_u"), on_pad(p, "pad_v")
            table = (abs(p[2] - (table_z + 0.025)) < THRESHOLDS["table_z_tol"]
                     and not inside and not at_u and not at_v
                     and held != Truth.TRUE)
            add(ids["held"], held, "rgbd_grip_proximity", ("rgb", "depth", "gripper"))
            add(ids["inside"], Truth.TRUE if inside else Truth.FALSE,
                "receiver_geometry", ("rgb", "depth", "public_layout"))
            add(ids["u"], Truth.TRUE if at_u else Truth.FALSE,
                "pad_u_geometry", ("rgb", "depth", "public_layout"))
            add(ids["v"], Truth.TRUE if at_v else Truth.FALSE,
                "pad_v_geometry", ("rgb", "depth", "public_layout"))
            add(ids["table"], Truth.TRUE if table else Truth.FALSE,
                "table_excludes_receiver_and_pads", ("rgb", "depth", "public_layout"))

        manifest = getattr(self.env, "task_manifest", None)
        if manifest is None or manifest.task_id != "T_P_FB":
            raise RuntimeError("STOPPED_STATIC_INVARIANT_PROVENANCE")
        for contract in getattr(self.env, "task_contracts", ()):
            if any(a.id == "p:Open:receiver"
                   for a in contract.effects.add + contract.effects.delete +
                   contract.effects.unknown):
                raise RuntimeError("STOPPED_STATIC_INVARIANT_WRITER")
        add("p:Open:receiver", Truth.TRUE,
            "fixed_open_receiver_asset_and_manifest",
            ("task_manifest", "public_layout"))
        if self.recorder is not None:
            self.recorder.facts(tuple(recs))
        return tuple(recs)


class FamilyBEvaluator(TaskEvaluator):
    def __init__(self, env, deadline_seconds, task_id="T_P_FB"):
        if task_id != "T_P_FB":
            raise ValueError("FamilyBEvaluator requires T_P_FB")
        super().__init__(env, deadline_seconds, task_id)

    def goal_true(self):
        h = self.env.hidden_truth()
        c = h["receiver"]
        inside = all(
            abs(h[name][0]-c[0]) <= RECEIVER_INNER[0] and
            abs(h[name][1]-c[1]) <= RECEIVER_INNER[1] and
            h[name][2] <= h["table_top_z"] + 0.12
            for name in ("obj_b", "obj_c")
        )
        empty = abs(float(h["gripper_qpos"][0])) > THRESHOLDS["gripper_open"]
        return bool(inside and empty)
