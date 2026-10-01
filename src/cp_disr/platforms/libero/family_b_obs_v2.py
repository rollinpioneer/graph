"""FAMILY_B_PUBLIC_OBS_V2: fixed two-camera public RGB-D observation for Family B.

Cameras are fixed robosuite arena cameras, identical for every candidate, layout,
context and method.  No camera motion, no extra robot action.  Dynamic facts are
derived only from public RGB-D evidence; a missing blob is UNKNOWN, never filled
from a previous frame, a nominal effect, hidden state or initial coordinates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from .family_b_adapters import FamilyBPerception, _palette_mask
from .family_b_env import MOVABLE, PALETTE, FamilyBEnv
from .perception import (THRESHOLDS, VERSION, PerceptionResult, _metric_depth,
                         backproject_mask)

PROFILE_VERSION = "FAMILY_B_PUBLIC_OBS_V2"
CAMERAS = ("agentview", "sideview")
IMAGE_SIZE = 128
FIXED_ATOL = 1e-9
# Pre-registered fusion contract (frozen before any v2 reset).
FUSION = {
    "rule": "PER_OBJECT_MAX_SUPPORT_RATIO",
    "support": "number of finite depth-valid pixels inside the object's palette mask",
    "reference": "support of the same object in the same view at the first perception call of the episode",
    "ratio": "support/reference if reference>0 else (1.0 if support>=min_pixels else 0.0)",
    "valid_view": "support>=min_pixels and estimator succeeded",
    "select": "valid view with the larger ratio; exact tie -> earlier camera in CAMERAS order",
    "none_valid": "UNKNOWN (blob absent); no carry-over from any other frame or source",
    "estimators": {
        "agentview": "ALL_SURFACE_MEAN (identical to the validated v1 estimator)",
        "sideview": "EXTENTS_MIDPOINT_XY_MEAN_Z (midpoint of min/max world x and y of valid surface points)",
    },
}
_PROFILE = {"path": None, "data": None}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def set_profile(path):
    data = json.loads(Path(path).read_text())
    if data.get("profile_version") != PROFILE_VERSION:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:version")
    if tuple(data["cameras"]) != CAMERAS:
        raise ValueError("STOPPED_OBSERVATION_PROFILE:cameras")
    _PROFILE.update(path=str(path), data=data)
    return data


def profile():
    if _PROFILE["data"] is None:
        raise RuntimeError("STOPPED_OBSERVATION_PROFILE:not loaded")
    return _PROFILE["data"]


def quat_to_mat(quat):
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def live_camera(env, name):
    cid = env.sim.model.camera_name2id(name)
    return {
        "name": name, "mode": int(env.sim.model.cam_mode[cid]),
        "parent_body": int(env.sim.model.cam_bodyid[cid]),
        "pos": np.array(env.sim.model.cam_pos[cid], dtype=float).tolist(),
        "quat": np.array(env.sim.model.cam_quat[cid], dtype=float).tolist(),
        "fovy": float(env.sim.model.cam_fovy[cid]),
    }


def calibration(env, name):
    cid = env.sim.model.camera_name2id(name)
    return {
        "pos": np.array(env.sim.data.cam_xpos[cid], dtype=float).tolist(),
        "mat": np.array(env.sim.data.cam_xmat[cid], dtype=float).reshape(3, 3).tolist(),
        "fovy": float(env.sim.model.cam_fovy[cid]),
        "width": IMAGE_SIZE, "height": IMAGE_SIZE,
    }


def verify_frozen_cameras(env, frozen):
    """Live fixed cameras must match the frozen manifest exactly (before reset)."""
    issues = []
    for name in CAMERAS:
        live, ref = live_camera(env, name), frozen["cameras"][name]
        if live["mode"] != ref["mode"]:
            issues.append(f"{name}:mode")
        for key in ("pos", "quat"):
            if not np.allclose(live[key], ref[key], atol=FIXED_ATOL, rtol=0):
                issues.append(f"{name}:{key}")
        if abs(live["fovy"] - ref["fovy"]) > FIXED_ATOL:
            issues.append(f"{name}:fovy")
    if issues:
        raise RuntimeError("STOPPED_OBSERVATION_PROFILE:live camera drift:" + ",".join(issues))
    return True


class FamilyBObsV2Env(FamilyBEnv):
    """Same task as FamilyBEnv; only the public camera set changes."""

    def __init__(self, case, render_gpu_device_id=0):
        self.case = case
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 0.005, 0.0001)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.use_object_obs = False
        self.last_perception = {}
        self.refresh_perception = None
        self.observation_profile_version = PROFILE_VERSION
        SingleArmEnv.__init__(
            self, robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            gripper_types="default", initialization_noise=None,
            use_camera_obs=True, has_renderer=False, has_offscreen_renderer=True,
            render_camera="agentview", render_collision_mesh=False,
            render_visual_mesh=True, render_gpu_device_id=render_gpu_device_id,
            control_freq=20, horizon=2000, ignore_done=True, hard_reset=True,
            camera_names=list(CAMERAS), camera_heights=[IMAGE_SIZE] * len(CAMERAS),
            camera_widths=[IMAGE_SIZE] * len(CAMERAS),
            camera_depths=[True] * len(CAMERAS), camera_segmentations=None,
        )
        if _PROFILE["data"] is not None:
            verify_frozen_cameras(self, _PROFILE["data"])

    def public_observation(self):
        """Same public fields as D0 (agentview kept as rgb/depth) plus all fixed views."""
        raw = (SingleArmEnv._get_observations(self) if hasattr(SingleArmEnv, "_get_observations")
               else SingleArmEnv._get_observation(self))
        views = {
            name: {"rgb": np.flipud(np.asarray(raw[f"{name}_image"])).copy(),
                   "depth": np.flipud(np.asarray(raw[f"{name}_depth"])).copy()}
            for name in CAMERAS
        }
        proprio = np.concatenate([
            np.asarray(raw["robot0_eef_pos"], dtype=float),
            np.asarray(raw["robot0_eef_quat"], dtype=float),
            np.asarray(raw["robot0_gripper_qpos"], dtype=float),
            np.asarray(raw["robot0_joint_pos_cos"], dtype=float),
            np.asarray(raw["robot0_joint_pos_sin"], dtype=float),
        ])
        return {
            "rgb": views["agentview"]["rgb"], "depth": views["agentview"]["depth"],
            "views": views, "proprio": proprio,
            "eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=float),
            "eef_quat": np.asarray(raw["robot0_eef_quat"], dtype=float),
            "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=float),
            "sim_time": float(self.sim.data.time),
        }


def make_family_b_obs_v2_env(case, gpu=0):
    return FamilyBObsV2Env(case, render_gpu_device_id=gpu)


# ----------------------------------------------------------------- pure perception
def surface_points(mask, depth, calib):
    ys, xs = np.where(mask)
    if len(xs) < THRESHOLDS["min_pixels"]:
        return None
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > THRESHOLDS["depth_valid"][0]) & (z < THRESHOLDS["depth_valid"][1])
    if int(valid.sum()) < THRESHOLDS["min_pixels"]:
        return None
    xs, ys, z = xs[valid], ys[valid], z[valid]
    h, w = int(calib["height"]), int(calib["width"])
    f = 0.5 * h / np.tan(np.deg2rad(calib["fovy"]) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    pts_cam = np.stack([(xs - cx) * z / f, -((ys - cy) * z / f), -z], axis=1)
    return pts_cam @ np.array(calib["mat"], dtype=float).T + np.array(calib["pos"], dtype=float)


def estimate(view, pts):
    if view == "agentview":
        return pts.mean(0)
    mid = (pts.min(0) + pts.max(0)) / 2.0
    return np.array([mid[0], mid[1], pts[:, 2].mean()])


def analyze_view(view, rgb, depth, calib):
    """Per-object public evidence in one view. Never sees anything but RGB-D."""
    result = {}
    for name in PALETTE:
        mask = _palette_mask(rgb, name)
        mask_pixels = int(mask.sum())
        pts = surface_points(mask, depth, calib)
        if pts is None:
            result[name] = {"mask_pixels": mask_pixels, "depth_support": 0, "blob_present": False,
                            "xyz": None, "reason": "insufficient_color_depth_support"}
            continue
        xyz = estimate(view, pts)
        result[name] = {"mask_pixels": mask_pixels, "depth_support": int(len(pts)), "blob_present": True,
                        "xyz": xyz.tolist(), "std": pts.std(0).tolist(), "reason": None}
    return result


def fuse(per_view, reference):
    """Deterministic pre-registered fusion; `reference` is episode-initial support."""
    min_pixels = THRESHOLDS["min_pixels"]
    fused = {}
    for name in PALETTE:
        rows, best = {}, None
        for order, view in enumerate(CAMERAS):
            ev = per_view[view][name]
            support = int(ev["depth_support"])
            ref = int(reference.get(view, {}).get(name, 0))
            ratio = (support / ref) if ref > 0 else (1.0 if support >= min_pixels else 0.0)
            valid = bool(ev["blob_present"]) and support >= min_pixels
            rows[view] = {"support": support, "reference": ref, "ratio": ratio, "valid": valid}
            if valid and (best is None or ratio > best[0]):
                best = (ratio, view)
        if best is None:
            fused[name] = {"selected_view": None, "blob": None, "inputs": rows,
                           "reason": "no_view_with_public_support"}
        else:
            ev = per_view[best[1]][name]
            fused[name] = {"selected_view": best[1], "inputs": rows, "reason": None,
                           "blob": {"xyz": ev["xyz"], "pixels": ev["depth_support"],
                                    "std": ev["std"], "source": "rgb_depth", "view": best[1]}}
    return fused


class FamilyBPerceptionV2(FamilyBPerception):
    variant = PROFILE_VERSION

    def __init__(self, env):
        super().__init__(env)
        self.reference = {}
        self.profile_hash = profile()["profile_sha256"]
        self.calibration_hash = digest({v: calibration(env, v) for v in CAMERAS})

    def analyze(self, observation):
        """Pure function of the public observation; hidden keys are never read."""
        views = observation["views"]
        calibs, per_view, depths = {}, {}, {}
        for view in CAMERAS:
            depth = _metric_depth(self.env, views[view]["depth"])
            calibs[view], depths[view] = calibration(self.env, view), depth
            per_view[view] = analyze_view(view, np.asarray(views[view]["rgb"]), depth, calibs[view])
        return per_view, calibs, depths

    def infer(self, observation):
        observation = self.env.public_observation()
        per_view, calibs, depths = self.analyze(observation)
        first = not self.reference
        if first:
            self.reference = {v: {n: int(per_view[v][n]["depth_support"]) for n in PALETTE} for v in CAMERAS}
        fused = fuse(per_view, self.reference)
        blobs, xyz, reasons = {}, {}, {}
        for name in PALETTE:
            blob = fused[name]["blob"]
            blobs[name] = blob
            if blob is None:
                reasons[name] = "insufficient_color_depth_support"
            else:
                xyz[name] = np.asarray(blob["xyz"], dtype=float)
        layout = self.env.public_layout()
        for name in ("receiver", "pad_u", "pad_v"):
            if name not in xyz:
                xyz[name] = np.asarray(layout[name], dtype=float)
                blobs[name] = {"xyz": xyz[name].tolist(), "pixels": 0, "std": [0, 0, 0],
                               "source": "static_layout"}
                reasons.pop(name, None)
        self.env.last_perception = xyz
        agent_depth = depths["agentview"]
        meas = {
            "blobs": blobs, "unknown_reasons": reasons,
            "eef_pos": np.asarray(observation["eef_pos"], dtype=float).tolist(),
            "gripper_qpos": np.asarray(observation["gripper_qpos"], dtype=float).tolist(),
            "calib": {k: calibs["agentview"][k] for k in ("fovy", "width", "height")},
            "depth_stats": {"min": float(np.nanmin(agent_depth)), "max": float(np.nanmax(agent_depth)),
                            "mean": float(np.nanmean(agent_depth))},
            "profile_version": PROFILE_VERSION, "profile_sha256": self.profile_hash,
        }
        if self.recorder is not None and hasattr(self.recorder, "perception_views"):
            self.recorder.perception_views({
                "views": observation["views"], "calibs": calibs, "per_view": per_view,
                "fused": fused, "reference_initial": first, "layout": layout, "meas": meas,
                "profile_sha256": self.profile_hash, "calibration_sha256": self.calibration_hash,
                "depths": depths,
            })
        return PerceptionResult(
            meas, self.checkpoint_hash, self.calibration_hash,
            tuple(f"{v}_{k}" for v in CAMERAS for k in ("rgb", "depth")) + ("camera_calibration",),
            VERSION)
