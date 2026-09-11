from __future__ import annotations

import copy
from typing import Any

import numpy as np

from upgrade_v2.visual_refine_l2.repaired_simulator import AttachRelposeDynamicTabletop

from . import SIM_VERSION
from .protocol import canonical_hash


class InstrumentationMutationDetected(RuntimeError):
    pass


def _runtime_state_digest(sim: Any) -> str:
    state = {
        "qpos": np.asarray(sim.data.qpos).tobytes().hex(), "qvel": np.asarray(sim.data.qvel).tobytes().hex(),
        "qacc": np.asarray(sim.data.qacc).tobytes().hex(), "qacc_warmstart": np.asarray(sim.data.qacc_warmstart).tobytes().hex(),
        "mocap_pos": np.asarray(sim.data.mocap_pos).tobytes().hex(), "mocap_quat": np.asarray(sim.data.mocap_quat).tobytes().hex(),
        "eq_active": np.asarray(sim.data.eq_active).tobytes().hex(), "eq_data": np.asarray(sim.model.eq_data).tobytes().hex(),
        "xfrc_applied": np.asarray(sim.data.xfrc_applied).tobytes().hex(), "time": float(sim.data.time),
        "rng": copy.deepcopy(sim.rng.bit_generator.state), "events": copy.deepcopy(sim.events),
        "lifecycle": sim.attempt_lifecycle.snapshot(),
        "flags": {key: getattr(sim, key) for key in ("gripper_closed", "attached", "failed_once", "recovered", "contact_lost", "action_index")},
    }
    return canonical_hash(state)


class ReproducibleBaselineTabletop(AttachRelposeDynamicTabletop):
    """One physical code path for ordinary and read-only instrumented runs."""

    simulator_version = SIM_VERSION

    def __init__(self, spec: Any, rollout_seed: int, physics_observer: Any = None) -> None:
        self.physics_observer = physics_observer
        super().__init__(spec, rollout_seed)

    def _step_once(self, context: dict[str, Any]) -> None:
        if self.physics_observer is not None:
            before = _runtime_state_digest(self)
            self.physics_observer.before_step(self, context)
            if _runtime_state_digest(self) != before:
                raise InstrumentationMutationDetected("R14B_INSTRUMENTATION_MUTATION_DETECTED before_step")
        self.mujoco.mj_step(self.model, self.data)
        if self.physics_observer is not None:
            before = _runtime_state_digest(self)
            self.physics_observer.after_step(self, context)
            if _runtime_state_digest(self) != before:
                raise InstrumentationMutationDetected("R14B_INSTRUMENTATION_MUTATION_DETECTED after_step")

    def _advance(self, target: np.ndarray | None = None, controls: int = 4) -> None:
        start = self.data.mocap_pos[0].copy()
        for control in range(controls):
            control_start = float(self.data.time)
            alpha = (control + 1) / controls
            if target is not None:
                self.data.mocap_pos[0] = start * (1 - alpha) + target * alpha
            commanded_position = self.data.mocap_pos[0].copy()
            for step in range(5):
                if self.attached:
                    self._set_object_xyz(self.data.mocap_pos[0] + np.array([0.0, 0.0, -0.13]))
                self._step_once({"control_index": len(self._active_control_sequence), "physics_index": len(self._active_control_sequence) * 5 + step})
            self._active_control_sequence.append({
                "control_step": len(self._active_control_sequence), "start_time": round(control_start, 6),
                "end_time": round(float(self.data.time), 6), "mocap_position": [round(float(x), 9) for x in commanded_position],
                "gripper_command": "closed" if self.gripper_closed else "open", "physics_steps": 5,
            })
            if self._r1_control_callback is not None:
                self._r1_control_callback(self, self._active_control_sequence[-1])
