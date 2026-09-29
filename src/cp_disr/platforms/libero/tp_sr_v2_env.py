"""Task-specific environments for the T_P_SR V2 pilot. D0ManipulationEnv and the V1 runtime are not modified.

InstrumentedD0Env    : V1 geometry (cube interferer) + passive recording mixin  -> Wave D
TPSRV2BarEnv         : same, with a graspable rectangular bar interferer          -> Wave P
The bar has the same half height as the supported cube (so the unchanged controller grasps at the same z) and only a
different footprint / density; no collision is disabled and nothing is teleported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from .d0_env import COLORS, OBJECT_HALF, CaseSpec, D0ManipulationEnv
from .tp_sr_instrumentation import RecordingEnvMixin


@dataclass
class CaseSpecV2(CaseSpec):
    second_half: tuple | None = None          # (hx, hy, hz) of the interferer; None = V1 cube
    second_density: float = 200.0
    second_friction: tuple = (1.2, 0.005, 0.0001)


class InstrumentedD0Env(RecordingEnvMixin, D0ManipulationEnv):
    """V1 geometry, passive recording only."""


class TPSRV2BarEnv(RecordingEnvMixin, D0ManipulationEnv):
    def _load_model(self):
        # Mirrors D0ManipulationEnv._load_model; the only differences are the interferer size/density/friction.
        half = getattr(self.case, "second_half", None)
        if half is None or abs(float(half[2]) - float(OBJECT_HALF[2])) > 1e-9:
            raise RuntimeError("TPSRV2BarEnv requires second_half with the supported cube half height")
        super(D0ManipulationEnv, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self._inject_static_geoms(arena)
        from .d0_env import LID_HALF
        self.second_role = getattr(self.case, "second_role", "interferer")
        rgba = COLORS.get(self.second_role, COLORS["second_object"])
        self.target = BoxObject(name="target", size=OBJECT_HALF, rgba=COLORS["target"], friction=(1.2, 0.005, 0.0001), density=200)
        self.second_half = np.asarray(half, dtype=float)
        self.second_object = BoxObject(name=self.second_role, size=self.second_half, rgba=rgba,
                                       friction=tuple(getattr(self.case, "second_friction", (1.2, 0.005, 0.0001))),
                                       density=float(getattr(self.case, "second_density", 200.0)))
        self.lid = BoxObject(name="lid", size=LID_HALF, rgba=COLORS["lid"], friction=(1.8, 0.005, 0.0001), density=80)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot.robot_model for robot in self.robots],
                                      mujoco_objects=[self.target, self.second_object, self.lid])
        self._static_names = ("container_floor", "container_front", "container_back", "container_left", "container_right", "buffer_pad")


def make_v2_env(spec, gpu=0):
    if getattr(spec, "second_half", None) is not None:
        return TPSRV2BarEnv(spec, render_gpu_device_id=gpu)
    return InstrumentedD0Env(spec, render_gpu_device_id=gpu)
