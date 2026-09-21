"""Frozen RGB-D frontend using color masks and camera-calibrated depth backprojection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
import numpy as np

from .d0_env import COLORS

VERSION = "cp-disr-d0-rgbd-colorseg-v2"
THRESHOLDS = {
    "min_pixels": 8,
    "depth_valid": (0.05, 4.0),
    "color_tol": 0.32,
    "held_xy": 0.06,
    "held_z": 0.07,
    "table_z_tol": 0.05,
    "inside_margin": 0.02,
    "buffer_margin": 0.03,
    "open_offset": 0.08,
    "gripper_open": 0.022,
}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _mask(rgb, color, tol):
    img = rgb.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    d = np.linalg.norm(img - np.asarray(color[:3])[None, None, :], axis=2)
    return d < tol


def _metric_depth(env, depth):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    try:
        from robosuite.utils.camera_utils import get_real_depth_map

        real = np.asarray(get_real_depth_map(env.sim, depth), dtype=np.float32)
        if np.isfinite(real).mean() > 0.5 and float(np.nanmax(real)) > 0.05:
            return real
    except Exception:
        pass
    # Fallback: if already metric, keep; if normalized 0-1, scale by a conservative far plane.
    if float(np.nanmax(depth)) <= 1.01:
        extent = float(getattr(env.sim.model, "stat", None).extent) if hasattr(env.sim.model, "stat") else 3.0
        near = 0.1
        far = max(2.0, extent)
        return near / (1.0 - depth * (1.0 - near / far) + 1e-8)
    return depth


def camera_calibration(env):
    cam_id = env.sim.model.camera_name2id("agentview")
    pos = np.array(env.sim.data.cam_xpos[cam_id], dtype=float)
    mat = np.array(env.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    fovy = float(env.sim.model.cam_fovy[cam_id])
    return {"pos": pos.tolist(), "mat": mat.tolist(), "fovy": fovy, "width": 128, "height": 128}


def backproject_mask(mask, depth, calib):
    ys, xs = np.where(mask)
    if len(xs) < THRESHOLDS["min_pixels"]:
        return None
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > THRESHOLDS["depth_valid"][0]) & (z < THRESHOLDS["depth_valid"][1])
    if int(valid.sum()) < THRESHOLDS["min_pixels"]:
        return None
    xs, ys, z = xs[valid], ys[valid], z[valid]
    h, w = int(calib["height"]), int(calib["width"])
    fovy = np.deg2rad(calib["fovy"])
    f = 0.5 * h / np.tan(fovy / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x_cam = (xs - cx) * z / f
    y_cam = (ys - cy) * z / f
    pts_cam = np.stack([x_cam, -y_cam, -z], axis=1)
    R = np.array(calib["mat"], dtype=float)
    t = np.array(calib["pos"], dtype=float)
    pts = pts_cam @ R.T + t
    return {"xyz": pts.mean(0), "pixels": int(len(xs)), "std": pts.std(0)}


@dataclass
class PerceptionResult:
    measurements: dict
    checkpoint_hash: str
    calibration_hash: str
    evidence_ids: tuple
    version: str = VERSION


class PerceptionAdapter:
    def __init__(self, env):
        self.env = env
        self.checkpoint_hash = sha({"version": VERSION, "thresholds": THRESHOLDS})
        self.calibration_hash = sha(camera_calibration(env))

    def infer(self, observation) -> PerceptionResult:
        if not isinstance(observation, dict) or "rgb" not in observation:
            observation = self.env.public_observation()
        rgb = np.asarray(observation["rgb"])
        depth = _metric_depth(self.env, observation["depth"])
        calib = camera_calibration(self.env)
        self.calibration_hash = sha(calib)
        blobs = {}
        table_xyz = {}
        reasons = {}
        for name, color in COLORS.items():
            m = _mask(rgb, color, THRESHOLDS["color_tol"])
            rec = backproject_mask(m, depth, calib)
            if rec is None:
                blobs[name] = None
                reasons[name] = "insufficient_color_depth_support"
            else:
                blobs[name] = {"xyz": rec["xyz"].tolist(), "pixels": rec["pixels"], "std": rec["std"].tolist()}
                table_xyz[name] = rec["xyz"]
        layout = self.env.public_layout()
        if "container" not in table_xyz:
            table_xyz["container"] = np.array(layout["container"], dtype=float)
            blobs["container"] = {"xyz": table_xyz["container"].tolist(), "pixels": 0, "std": [0, 0, 0], "source": "static_layout"}
            reasons.pop("container", None)
        if "buffer" not in table_xyz:
            table_xyz["buffer"] = np.array(layout["buffer"], dtype=float)
            blobs["buffer"] = {"xyz": table_xyz["buffer"].tolist(), "pixels": 0, "std": [0, 0, 0], "source": "static_layout"}
            reasons.pop("buffer", None)
        self.env.last_perception = table_xyz
        meas = {
            "blobs": blobs,
            "unknown_reasons": reasons,
            "eef_pos": np.asarray(observation["eef_pos"], dtype=float).tolist(),
            "gripper_qpos": np.asarray(observation["gripper_qpos"], dtype=float).tolist(),
            "calib": {k: calib[k] for k in ("fovy", "width", "height")},
            "depth_stats": {
                "min": float(np.nanmin(depth)),
                "max": float(np.nanmax(depth)),
                "mean": float(np.nanmean(depth)),
            },
        }
        return PerceptionResult(
            meas,
            self.checkpoint_hash,
            self.calibration_hash,
            ("agentview_rgb", "agentview_depth", "camera_calibration"),
            VERSION,
        )
