"""Versioned simulator variant for the L2RA-R2 attach-relpose repair.

The historical ``DynamicTabletop`` remains untouched.  This class fixes only
the weld target at attach time, preserving the existing scripted object
writeback, controller program, physics parameters, and observation contract.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .dynamic_simulator import DynamicTabletop


ATTACH_RELPOSE_VERSION = "l2rar2_attach_relpose_v1"


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-15:
        raise ValueError("quaternion must be non-zero")
    return np.asarray(quat, dtype=float) / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    normalized = _quat_normalize(quat)
    return normalized * np.asarray((1.0, -1.0, -1.0, -1.0))


def _quat_multiply_raw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=float)
    rw, rx, ry, rz = np.asarray(right, dtype=float)
    return np.asarray((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _quat_normalize(_quat_multiply_raw(_quat_normalize(left), _quat_normalize(right)))


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_quat = np.asarray((0.0, *np.asarray(vector, dtype=float)))
    return _quat_multiply_raw(_quat_multiply_raw(_quat_normalize(quat), vector_quat), _quat_conjugate(quat))[1:]


class AttachRelposeDynamicTabletop(DynamicTabletop):
    """DynamicTabletop with an attach-time weld pose equal to actual geometry."""

    repair_version = ATTACH_RELPOSE_VERSION

    def _attach(self) -> None:
        # MuJoCo weld relpose is body2 expressed in body1's local frame.
        gripper_quat = _quat_normalize(self.data.mocap_quat[0])
        object_quat = _quat_normalize(self.data.qpos[self.object_qpos + 3:self.object_qpos + 7])
        gripper_inverse = _quat_conjugate(gripper_quat)
        relative_position = _quat_rotate(gripper_inverse, self.object_xyz - self.data.mocap_pos[0])
        relative_quat = _quat_multiply(gripper_inverse, object_quat)
        self.model.eq_data[self.weld_id, 3:6] = relative_position
        self.model.eq_data[self.weld_id, 6:10] = relative_quat
        super()._attach()

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot["model_eq_data"] = self.model.eq_data.copy()
        snapshot["repair_version"] = self.repair_version
        return snapshot

    def restore(self, snapshot: dict[str, Any]) -> None:
        super().restore(snapshot)
        if snapshot.get("repair_version") not in (None, self.repair_version):
            raise ValueError("snapshot was created by a different simulator version")
        if "model_eq_data" in snapshot:
            self.model.eq_data[:] = snapshot["model_eq_data"]
            self.mujoco.mj_forward(self.model, self.data)
