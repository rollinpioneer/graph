from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import SIM_VERSION
from .protocol import CALIBRATION_DURATION_S, PulseLevel


@dataclass(frozen=True)
class ForcePulse:
    level_id: str
    duration_s: float
    target_delta_v_local_mps: tuple[float, float, float]
    force_local_n: tuple[float, float, float]
    force_world_n: tuple[float, float, float]


def force_from_delta_v(mass_kg: float, delta_v_local_mps: Any, duration_s: float = CALIBRATION_DURATION_S) -> np.ndarray:
    if mass_kg <= 0 or duration_s <= 0:
        raise ValueError("mass and duration must be positive")
    delta = np.asarray(delta_v_local_mps, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("delta_v must be a finite 3-vector")
    return mass_kg * delta / duration_s


def local_to_world(rotation_world_from_local: Any, vector_local: Any) -> np.ndarray:
    rotation = np.asarray(rotation_world_from_local, dtype=float)
    vector = np.asarray(vector_local, dtype=float)
    if rotation.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("rotation must be 3x3 and vector must be length 3")
    return rotation @ vector


def make_force_pulse(mass_kg: float, gripper_rotation, level: PulseLevel, duration_s: float = CALIBRATION_DURATION_S) -> ForcePulse:
    local = force_from_delta_v(mass_kg, level.target_delta_v_local_mps, duration_s)
    world = local_to_world(gripper_rotation, local)
    return ForcePulse(level.level_id, duration_s, level.target_delta_v_local_mps, tuple(local), tuple(world))


def run_force_pulse(sim: Any, pulse: ForcePulse, trace: Callable[[Any, str], None] | None = None,
                    *, steps: int | None = None) -> list[dict[str, Any]]:
    """Apply a pulse and always clear the six-vector force.

    The frozen calibration path uses the default five 100 Hz steps.  Explicit
    ``steps`` is reserved for a separately labelled intervention-design probe.
    """
    if not getattr(sim, "pre_hold_verified", False):
        raise RuntimeError("PREHOLD_UNVERIFIED")
    expected_steps = int(round(pulse.duration_s * sim.physics_hz))
    if steps is None:
        steps = expected_steps
    if steps <= 0 or expected_steps != steps:
        raise ValueError("pulse duration and physics-step count mismatch")
    rows: list[dict[str, Any]] = []
    sim.disable_weld_for_intervention()
    if trace:
        trace(sim, "post_weld_off_pre_force")
    try:
        sim.set_object_force(np.asarray(pulse.force_world_n, dtype=float))
        if trace:
            trace(sim, "force_pulse_started")
        for step in range(steps):
            sim.physics_step()
            row = sim.reference_snapshot("force_pulse")
            row["pulse_step"] = step + 1
            rows.append(row)
            if trace:
                trace(sim, "physics_step")
    finally:
        sim.clear_object_force()
        if trace:
            trace(sim, "force_pulse_ended")
    sim.forward()
    return rows
