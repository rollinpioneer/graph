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


class AttachRelposeDynamicTabletop(DynamicTabletop):
    """DynamicTabletop with an attach-time weld pose equal to actual geometry."""

    repair_version = ATTACH_RELPOSE_VERSION

    def _attach(self) -> None:
        relative_position = self.object_xyz - self.data.mocap_pos[0]
        self.model.eq_data[self.weld_id, 3:6] = relative_position
        self.model.eq_data[self.weld_id, 6:10] = np.asarray((1.0, 0.0, 0.0, 0.0))
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
