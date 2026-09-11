from __future__ import annotations

from typing import Any

import numpy as np

from upgrade_v2.visual_refine_l2.repaired_simulator import AttachRelposeDynamicTabletop

from . import SIM_VERSION


class ControlledForcedDropTabletop(AttachRelposeDynamicTabletop):
    """R16 simulator: intervention and post-hoc loss outcome stay separate."""

    sim_version = SIM_VERSION
    physics_hz = 100

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pre_hold_verified = False
        self.intervention_phase = "none"
        self.scripted_writeback_count = 0
        self._intervention_started = False

    def _set_object_xyz(self, xyz):
        if self._intervention_started:
            raise RuntimeError("BLOCKED_SCRIPTED_WRITEBACK_AFTER_WELD_OFF")
        self.scripted_writeback_count += 1
        return super()._set_object_xyz(xyz)

    def verify_pre_hold(self, *, height_rise_m: float, relative_drift_m: float, sustain_s: float) -> bool:
        self.pre_hold_verified = bool(
            bool(self.data.eq_active[self.weld_id])
            and height_rise_m >= 0.03
            and relative_drift_m <= 0.01
            and sustain_s >= 0.10
        )
        return self.pre_hold_verified

    def disable_weld_for_intervention(self) -> None:
        self._intervention_started = True
        self.attached = False
        self.data.eq_active[self.weld_id] = 0
        self.intervention_phase = "weld_off_applied"
        self._record_event("weld_disable_applied", observable=False)

    def set_object_force(self, force_world_n: np.ndarray) -> None:
        force = np.asarray(force_world_n, dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force must be a finite 3-vector")
        object_body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        self.data.xfrc_applied[object_body, :3] = force
        self.data.xfrc_applied[object_body, 3:] = 0.0
        self.intervention_phase = "force_pulse"

    def clear_object_force(self) -> None:
        object_body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        self.data.xfrc_applied[object_body, :] = 0.0
        self.intervention_phase = "post_force_clear"

    def physics_step(self) -> None:
        if not np.isfinite(self.data.xfrc_applied).all():
            raise RuntimeError("NUMERICAL_INVALID")
        self.mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        self.mujoco.mj_forward(self.model, self.data)

    def reference_snapshot(self, phase: str) -> dict[str, Any]:
        object_body = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        return {
            "schema": SIM_VERSION,
            "time": float(self.data.time),
            "phase": phase,
            "object_qpos": self.data.qpos[self.object_qpos:self.object_qpos + 7].tolist(),
            "object_qvel": self.data.qvel[self.object_dof:self.object_dof + 6].tolist(),
            "object_xyz": self.data.xpos[object_body].tolist(),
            "eq_active": bool(self.data.eq_active[self.weld_id]),
            "xfrc_applied": self.data.xfrc_applied[object_body].tolist(),
            "scripted_writeback_count": self.scripted_writeback_count,
        }
