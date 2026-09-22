"""D0 robosuite environment: Panda OSC_POSE plus project-owned tabletop assets."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_body, new_geom


COLORS = {
    "target": np.array([0.85, 0.20, 0.15, 1.0]),
    "second_object": np.array([0.92, 0.72, 0.12, 1.0]),
    "interferer": np.array([0.10, 0.78, 0.78, 1.0]),
    "container": np.array([0.18, 0.35, 0.75, 1.0]),
    "lid": np.array([0.55, 0.20, 0.85, 1.0]),
    "buffer": np.array([0.18, 0.72, 0.30, 1.0]),
}

# Cube must fit inside the container opening; lid must sit on the rim and
# still be narrower than the Panda gripper opening (~0.08 m).
OBJECT_HALF = np.array([0.020, 0.020, 0.020])
LID_HALF = np.array([0.036, 0.036, 0.012])
CONTAINER_INNER = np.array([0.030, 0.030])
CONTAINER_WALL = 0.012
CONTAINER_H = 0.048
BUFFER_HALF = np.array([0.08, 0.08, 0.004])
RUNTIME_VERSION = "d0-runtime-v2.1-p0"


@dataclass
class CaseSpec:
    case_id: str
    split: str
    seed: int
    target_xy: tuple[float, float]
    second_xy: tuple[float, float]
    container_xy: tuple[float, float]
    buffer_xy: tuple[float, float]
    lid_closed: bool = True
    task_id: str = "D0"
    second_role: str = "second_object"
    deadline: float = 90.0


class D0ManipulationEnv(SingleArmEnv):
    """Deterministic D0 tabletop with two cubes, openable container, and buffer pad."""

    def __init__(self, case: CaseSpec, render_gpu_device_id: int = 0):
        self.case = case
        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_friction = (1.0, 0.005, 0.0001)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.use_object_obs = False
        self.last_perception = {}
        self.refresh_perception = None
        controller_configs = load_controller_config(default_controller="OSC_POSE")
        super().__init__(
            robots="Panda",
            controller_configs=controller_configs,
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=20,
            horizon=2000,
            ignore_done=True,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=128,
            camera_widths=128,
            camera_depths=True,
            camera_segmentations=None,
        )

    def reward(self, action=None):
        return 0.0

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self._inject_static_geoms(arena)
        self.second_role = getattr(self.case, "second_role", "second_object")
        second_rgba = COLORS.get(self.second_role, COLORS["second_object"])
        self.target = BoxObject(name="target", size=OBJECT_HALF, rgba=COLORS["target"], friction=(1.2, 0.005, 0.0001), density=200)
        self.second_object = BoxObject(name=self.second_role, size=OBJECT_HALF, rgba=second_rgba, friction=(1.2, 0.005, 0.0001), density=200)
        self.lid = BoxObject(name="lid", size=LID_HALF, rgba=COLORS["lid"], friction=(1.8, 0.005, 0.0001), density=80)
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.target, self.second_object, self.lid],
        )
        self._static_names = (
            "container_floor",
            "container_front",
            "container_back",
            "container_left",
            "container_right",
            "buffer_pad",
        )

    def _inject_static_geoms(self, arena):
        cx, cy = self.case.container_xy
        bx, by = self.case.buffer_xy
        z0 = float(self.table_offset[2] + 0.025)
        wall = CONTAINER_WALL
        inner = CONTAINER_INNER
        h = CONTAINER_H
        world = arena.worldbody

        def add_box(parent, name, pos, size, rgba):
            parent.append(
                new_geom(
                    name=name,
                    type="box",
                    size=size,
                    pos=pos,
                    rgba=rgba,
                    group="0",
                    contype="1",
                    conaffinity="1",
                    condim="4",
                    friction="1.5 0.1 0.1",
                    solimp="0.99 0.95 0.001",
                    solref="0.02 1",
                )
            )

        container = new_body(name="d0_container", pos=[0, 0, 0])
        add_box(container, "container_floor", [cx, cy, z0 + 0.004], [inner[0] + wall, inner[1] + wall, 0.004], COLORS["container"])
        add_box(container, "container_back", [cx, cy + inner[1] + wall, z0 + h / 2], [inner[0] + wall, wall, h / 2], COLORS["container"])
        add_box(container, "container_front", [cx, cy - inner[1] - wall, z0 + h / 2], [inner[0] + wall, wall, h / 2], COLORS["container"])
        add_box(container, "container_left", [cx - inner[0] - wall, cy, z0 + h / 2], [wall, inner[1], h / 2], COLORS["container"])
        add_box(container, "container_right", [cx + inner[0] + wall, cy, z0 + h / 2], [wall, inner[1], h / 2], COLORS["container"])
        world.append(container)
        table_skin = new_body(name="d0_table_skin", pos=[0, 0, 0])
        add_box(table_skin, "table_skin", [0.0, 0.0, z0 + 0.0015], [0.36, 0.36, 0.0015], np.array([0.62, 0.55, 0.45, 1.0]))
        world.append(table_skin)
        buffer = new_body(name="d0_buffer", pos=[0, 0, 0])
        add_box(buffer, "buffer_pad", [bx, by, z0 + BUFFER_HALF[2]], list(BUFFER_HALF), COLORS["buffer"])
        world.append(buffer)
        self.container_center = np.array([cx, cy, z0 + 0.004])
        self.buffer_center = np.array([bx, by, z0 + BUFFER_HALF[2]])
        self.table_top_z = z0
        self.container_rim_z = z0 + h

    def _setup_references(self):
        super()._setup_references()
        role = getattr(self, "second_role", getattr(self.case, "second_role", "second_object"))
        self.second_role = role
        self.obj_body_id = {
            "target": self.sim.model.body_name2id(self.target.root_body),
            role: self.sim.model.body_name2id(self.second_object.root_body),
            "lid": self.sim.model.body_name2id(self.lid.root_body),
        }

    def _reset_internal(self):
        super()._reset_internal()
        self._apply_case_poses()

    def _open_gripper_reset(self):
        """Drive fingers to the open stop without using hidden object poses."""
        action = np.zeros(self.action_dim, dtype=float)
        action[-1] = -1.0
        for _ in range(20):
            super().step(action)

    def _apply_case_poses(self):
        z_cube = self.table_top_z + OBJECT_HALF[2] + 0.001
        z_lid = self.table_top_z + CONTAINER_H + LID_HALF[2] + 0.001
        poses = {
            "target": (*self.case.target_xy, z_cube),
            "second_object": (*self.case.second_xy, z_cube),
            "lid": (*self.case.container_xy, z_lid) if self.case.lid_closed else (self.case.container_xy[0] + 0.26, self.case.container_xy[1], z_cube + 0.01),
        }
        objs = {"target": self.target, "second_object": self.second_object, "lid": self.lid}
        for name, obj in objs.items():
            joint = obj.joints[0]
            pos = poses[name]
            qpos = np.array(list(pos) + [1, 0, 0, 0], dtype=float)
            self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()

    def _check_success(self):
        return False

    def public_layout(self):
        return {
            "container": np.array(self.container_center, dtype=float),
            "buffer": np.array(self.buffer_center, dtype=float),
            "table_top_z": float(self.table_top_z),
            "container_rim_z": float(self.container_rim_z),
        }

    def public_observation(self):
        raw = super()._get_observations() if hasattr(self, "_get_observations") else super()._get_observation()
        obs = {
            k: v
            for k, v in raw.items()
            if ("object-state" not in k)
            and (not k.endswith("_pos") or k.startswith("robot0_"))
            and (not k.endswith("_quat") or k.startswith("robot0_"))
            and "cube" not in k
        }
        rgb = np.flipud(np.asarray(obs["agentview_image"])).copy()
        depth = np.flipud(np.asarray(obs["agentview_depth"])).copy()
        proprio = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=float),
                np.asarray(obs["robot0_eef_quat"], dtype=float),
                np.asarray(obs["robot0_gripper_qpos"], dtype=float),
                np.asarray(obs["robot0_joint_pos_cos"], dtype=float),
                np.asarray(obs["robot0_joint_pos_sin"], dtype=float),
            ]
        )
        return {
            "rgb": rgb,
            "depth": depth,
            "proprio": proprio,
            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=float),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=float),
            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=float),
            "sim_time": float(self.sim.data.time),
        }

    def hidden_truth(self):
        def pos(name):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[name]], dtype=float)

        role = getattr(self, "second_role", "second_object")
        return {
            "target": pos("target"),
            role: pos(role),
            "lid": pos("lid"),
            "container": np.array(self.container_center, dtype=float),
            "buffer": np.array(self.buffer_center, dtype=float),
            "table_top_z": float(self.table_top_z),
            "gripper_qpos": np.array(self.public_observation()["gripper_qpos"], dtype=float),
            "eef_pos": np.array(self.public_observation()["eef_pos"], dtype=float),
        }

    def step_osc(self, dpos, dgrip, n=1):
        traces = []
        for _ in range(int(n)):
            action = np.zeros(self.action_dim, dtype=float)
            action[:3] = np.clip(dpos, -1.0, 1.0)
            action[3:6] = 0.0
            action[-1] = float(np.clip(dgrip, -1.0, 1.0))
            if not np.all(np.isfinite(action)):
                raise RuntimeError("NaN/Inf OSC action")
            obs, reward, done, info = super().step(action)
            traces.append(
                {
                    "eef": np.array(obs["robot0_eef_pos"], dtype=float),
                    "t": float(self.sim.data.time),
                    "action": action.copy(),
                }
            )
        return traces


def make_env(case: CaseSpec, gpu=0):
    return D0ManipulationEnv(case, render_gpu_device_id=gpu)
