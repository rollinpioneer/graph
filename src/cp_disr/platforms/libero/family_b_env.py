"""Three-cube, two-pad Family B tabletop. No hidden state enters control inputs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_body, new_geom

from .d0_env import CaseSpec, D0ManipulationEnv, OBJECT_HALF, CONTAINER_H
from .tp_sr_instrumentation import RecordingEnvMixin

MOVABLE = ("carrier", "obj_b", "obj_c")
PALETTE = {
    "carrier": np.array([0.85, 0.20, 0.15, 1.0]),
    "obj_b": np.array([0.92, 0.72, 0.12, 1.0]),
    "obj_c": np.array([0.10, 0.78, 0.78, 1.0]),
    "receiver": np.array([0.18, 0.35, 0.75, 1.0]),
    "pad_u": np.array([0.18, 0.72, 0.30, 1.0]),
    "pad_v": np.array([0.55, 0.20, 0.85, 1.0]),
}
RECEIVER_INNER = np.array([0.070, 0.035])
RECEIVER_WALL = 0.012
PAD_HALF = np.array([0.045, 0.045, 0.004])
RECEIVER_SLOT_OFFSET = 0.035


@dataclass
class FamilyBCaseSpec(CaseSpec):
    obj_c_xy: tuple[float, float] = (0.0, -0.20)
    pad_v_xy: tuple[float, float] = (0.20, -0.02)


class FamilyBEnv(RecordingEnvMixin, D0ManipulationEnv):
    """The three movable cubes use identical BoxObject dynamics."""

    def _add_static(self, parent, name, pos, size, rgba):
        parent.append(new_geom(
            name=name, type="box", pos=pos, size=size, rgba=rgba,
            group="0", contype="1", conaffinity="1", condim="4",
            friction="1.5 0.1 0.1", solimp="0.99 0.95 0.001",
            solref="0.02 1",
        ))

    def _inject_static_geoms(self, arena):
        rx, ry = self.case.container_xy
        z0 = float(self.table_offset[2] + 0.025)
        inner, wall, h = RECEIVER_INNER, RECEIVER_WALL, CONTAINER_H
        receiver = new_body(name="fb_receiver", pos=[0, 0, 0])
        self._add_static(receiver, "receiver_floor",
                         [rx, ry, z0 + 0.004],
                         [inner[0] + wall, inner[1] + wall, 0.004],
                         PALETTE["receiver"])
        for name, x, y, sx, sy in (
            ("back", rx, ry + inner[1] + wall, inner[0] + wall, wall),
            ("front", rx, ry - inner[1] - wall, inner[0] + wall, wall),
            ("left", rx - inner[0] - wall, ry, wall, inner[1]),
            ("right", rx + inner[0] + wall, ry, wall, inner[1]),
        ):
            self._add_static(receiver, "receiver_" + name,
                             [x, y, z0 + h / 2], [sx, sy, h / 2],
                             PALETTE["receiver"])
        arena.worldbody.append(receiver)
        self.pad_centers = {}
        for name, xy in (("pad_u", self.case.buffer_xy),
                         ("pad_v", self.case.pad_v_xy)):
            x, y = xy
            pad = new_body(name="fb_" + name, pos=[0, 0, 0])
            self._add_static(pad, name, [x, y, z0 + PAD_HALF[2]],
                             list(PAD_HALF), PALETTE[name])
            arena.worldbody.append(pad)
            self.pad_centers[name] = np.array([x, y, z0 + PAD_HALF[2]])
        self.receiver_center = np.array([rx, ry, z0 + 0.004])
        self.receiver_slots = {
            "obj_b": np.array([rx - RECEIVER_SLOT_OFFSET, ry, z0 + 0.004]),
            "obj_c": np.array([rx + RECEIVER_SLOT_OFFSET, ry, z0 + 0.004]),
        }
        self.container_center = self.receiver_center
        self.buffer_center = self.pad_centers["pad_u"]
        self.table_top_z = z0
        self.container_rim_z = z0 + h

    def _load_model(self):
        super(D0ManipulationEnv, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self._inject_static_geoms(arena)
        self.movable_objects = {
            name: BoxObject(name=name, size=OBJECT_HALF, rgba=PALETTE[name],
                            friction=(1.2, 0.005, 0.0001), density=200)
            for name in MOVABLE
        }
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.movable_objects.values()),
        )
        self._static_names = (
            "receiver_floor", "receiver_back", "receiver_front",
            "receiver_left", "receiver_right", "pad_u", "pad_v",
        )

    def _setup_references(self):
        # D0's reference setup requires target/second_object/lid. Family B has no lid.
        super(D0ManipulationEnv, self)._setup_references()
        self.obj_body_id = {
            name: self.sim.model.body_name2id(obj.root_body)
            for name, obj in self.movable_objects.items()
        }

    def _apply_case_poses(self):
        z = self.table_top_z + OBJECT_HALF[2] + 0.001
        xy = {
            "carrier": self.case.target_xy,
            "obj_b": self.case.second_xy,
            "obj_c": self.case.obj_c_xy,
        }
        for name, obj in self.movable_objects.items():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.array([*xy[name], z, 1, 0, 0, 0], dtype=float),
            )
        self.sim.forward()

    def resolve_skill_destination(self, skill_name, object_id, destination_id):
        if (skill_name, object_id, destination_id) in (
            ("PLACE_BUFFER", "carrier", "pad_u"),
            ("PLACE_BUFFER", "carrier", "pad_v"),
        ):
            return self.pad_centers[destination_id].copy()
        if skill_name == "PLACE" and destination_id == "receiver":
            if object_id in self.receiver_slots:
                return self.receiver_slots[object_id].copy()
        raise ValueError("STOPPED_DESTINATION_BINDING:" +
                         ":".join((skill_name, object_id, destination_id)))

    def public_layout(self):
        return {
            "receiver": self.receiver_center.copy(),
            "pad_u": self.pad_centers["pad_u"].copy(),
            "pad_v": self.pad_centers["pad_v"].copy(),
            "receiver_slots": {k: v.copy() for k, v in self.receiver_slots.items()},
            "table_top_z": float(self.table_top_z),
            "container_rim_z": float(self.container_rim_z),
        }

    def hidden_truth(self):
        obs = self.public_observation()
        result = {
            name: np.array(self.sim.data.body_xpos[bid], dtype=float)
            for name, bid in self.obj_body_id.items()
        }
        result.update({
            "receiver": self.receiver_center.copy(),
            "pad_u": self.pad_centers["pad_u"].copy(),
            "pad_v": self.pad_centers["pad_v"].copy(),
            "table_top_z": float(self.table_top_z),
            "gripper_qpos": np.array(obs["gripper_qpos"], dtype=float),
            "eef_pos": np.array(obs["eef_pos"], dtype=float),
        })
        return result


def make_family_b_env(case, gpu=0):
    return FamilyBEnv(case, render_gpu_device_id=gpu)
