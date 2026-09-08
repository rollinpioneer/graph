"""Dynamic MuJoCo primitive tabletop with explicit online/oracle separation."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


SCENARIOS = (
    "normal_pick_place",
    "already_satisfied_stable",
    "already_on_target_offcenter",
    "missed_grasp_then_retry",
    "slip_then_recover",
    "target_occupied",
    "distractor_object_ambiguity",
    "primary_view_occlusion",
)


@dataclass(frozen=True)
class FamilySpec:
    root_family_id: str
    scenario: str
    object_radius: float
    friction: float
    camera_jitter: float
    object_x: float
    object_y: float
    target_x: float
    target_y: float
    rollout_seed_base: int


def family_spec(
    root_family_id: str,
    scenario: str,
    seed: int,
    rollout_seed_base: int,
    *,
    camera_jitter: bool = True,
    object_size_jitter: bool = True,
    friction_jitter: bool = True,
) -> FamilySpec:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    return FamilySpec(
        root_family_id=root_family_id,
        scenario=scenario,
        object_radius=float(rng.uniform(0.060, 0.078)) if object_size_jitter else 0.069,
        friction=float(rng.uniform(0.55, 0.95)) if friction_jitter else 0.75,
        camera_jitter=float(rng.uniform(-1.8, 1.8)) if camera_jitter else 0.0,
        object_x=float(rng.uniform(-0.38, -0.25)),
        object_y=float(rng.uniform(-0.10, 0.04)),
        target_x=float(rng.uniform(0.27, 0.38)),
        target_y=float(rng.uniform(0.05, 0.17)),
        rollout_seed_base=rollout_seed_base,
    )


def _xml(spec: FamilySpec) -> str:
    obstacle = spec.scenario == "target_occupied"
    distractor = spec.scenario == "distractor_object_ambiguity"
    occluder = spec.scenario == "primary_view_occlusion"
    obstacle_xml = '<body name="obstacle" pos=".33 .11 .57"><freejoint/><geom name="obstacle_geom" type="box" size=".055 .055 .055" rgba="1 .72 .04 1" mass=".12"/></body>' if obstacle else ""
    distractor_xml = '<body name="distractor" pos="-.17 .10 .575"><geom name="distractor_geom" type="cylinder" size=".068 .065" rgba=".78 .03 .16 1"/></body>' if distractor else ""
    occluder_xml = f'<body name="occluder" pos="{spec.object_x:.6f} {spec.object_y-.22:.6f} .72"><geom name="occluder_geom" type="box" size=".15 .035 .22" rgba=".38 .32 .28 1"/></body>' if occluder else ""
    return f"""
<mujoco model="l2r_dynamic_tabletop">
  <compiler angle="degree"/>
  <option timestep=".01" gravity="0 0 -9.81" integrator="RK4"/>
  <visual><quality shadowsize="1024"/></visual>
  <worldbody>
    <light directional="true" pos="0 -1 3" dir="0 0 -1" diffuse="1 1 1"/>
    <light directional="true" pos="-2 -1 2" dir="1 1 -1" diffuse=".5 .5 .5"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".16 .18 .20 1"/>
    <geom name="table" type="box" size=".92 .66 .05" pos="0 0 .45" rgba=".48 .31 .16 1"/>
    <body name="target" pos="{spec.target_x:.6f} {spec.target_y:.6f} .515">
      <geom name="target_geom" type="cylinder" size=".145 .015" rgba=".92 .93 .88 1" contype="1" conaffinity="1"/>
    </body>
    <body name="object" pos="{spec.object_x:.6f} {spec.object_y:.6f} {0.5+spec.object_radius:.6f}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="cylinder" size="{spec.object_radius:.6f} {spec.object_radius*.82:.6f}" rgba=".88 .025 .035 1" mass=".18" friction="{spec.friction:.4f} .01 .001"/>
    </body>
    <body name="gripper" mocap="true" pos="{spec.object_x:.6f} {spec.object_y-.25:.6f} .78">
      <geom name="palm" type="box" size=".055 .035 .025" rgba=".02 .72 .82 1" contype="0" conaffinity="0"/>
      <geom name="finger_left" type="box" size=".018 .025 .075" pos="-.080 0 -.075" rgba=".02 .72 .82 1"/>
      <geom name="finger_right" type="box" size=".018 .025 .075" pos=".080 0 -.075" rgba=".02 .72 .82 1"/>
    </body>
    {obstacle_xml}{distractor_xml}{occluder_xml}
  </worldbody>
  <equality><weld name="grasp_weld" body1="gripper" body2="object" active="false" solref=".002 1"/></equality>
</mujoco>
"""


class DynamicTabletop:
    """A deterministic scripted benchmark, not a physical robot simulator."""

    physics_hz = 100
    control_hz = 20
    render_hz = 10
    horizon_seconds = 12.0

    def __init__(self, spec: FamilySpec, rollout_seed: int) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mujoco = mujoco
        self.spec = spec
        self.rng = np.random.default_rng(rollout_seed)
        self.model = mujoco.MjModel.from_xml_string(_xml(spec))
        self.data = mujoco.MjData(self.model)
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos = int(self.model.jnt_qposadr[self.object_joint])
        self.object_dof = int(self.model.jnt_dofadr[self.object_joint])
        self.weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.obstacle_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "") if False else -1
        if spec.scenario == "target_occupied":
            obstacle_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
            self.obstacle_joint = int(self.model.body_jntadr[obstacle_body])
        self.gripper_closed = False
        self.attached = False
        self.failed_once = False
        self.recovered = False
        self.contact_lost = False
        self.action_index = 0
        self.events: list[dict[str, Any]] = []
        self._active_control_sequence: list[dict[str, Any]] = []
        # Optional read-only hook used by L2RA-R1.  It is invoked after the
        # same five physics steps as the ordinary control loop and cannot
        # alter actions, state, or RNG.
        self._r1_control_callback = None
        self._initialize_scenario()
        mujoco.mj_forward(self.model, self.data)

    def _initialize_scenario(self) -> None:
        if self.spec.scenario == "already_satisfied_stable":
            self._set_object_xyz((self.spec.target_x, self.spec.target_y, 0.5 + self.spec.object_radius * 1.65))
        elif self.spec.scenario == "already_on_target_offcenter":
            self._set_object_xyz((self.spec.target_x + 0.105, self.spec.target_y, 0.5 + self.spec.object_radius * 1.65))

    def _set_object_xyz(self, xyz: tuple[float, float, float] | np.ndarray) -> None:
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = xyz
        self.data.qpos[self.object_qpos + 3:self.object_qpos + 7] = (1, 0, 0, 0)
        self.data.qvel[self.object_dof:self.object_dof + 6] = 0

    @property
    def object_xyz(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos:self.object_qpos + 3].copy()

    def snapshot(self) -> dict[str, Any]:
        return {
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "mocap_pos": self.data.mocap_pos.copy(),
            "mocap_quat": self.data.mocap_quat.copy(), "eq_active": self.data.eq_active.copy(), "time": float(self.data.time),
            "gripper_closed": self.gripper_closed, "attached": self.attached, "failed_once": self.failed_once,
            "recovered": self.recovered, "contact_lost": self.contact_lost, "action_index": self.action_index,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state), "events": copy.deepcopy(self.events),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        for key in ("qpos", "qvel", "mocap_pos", "mocap_quat", "eq_active"):
            getattr(self.data, key)[:] = snapshot[key]
        self.data.time = snapshot["time"]
        for key in ("gripper_closed", "attached", "failed_once", "recovered", "contact_lost", "action_index"):
            setattr(self, key, snapshot[key])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        self.events = copy.deepcopy(snapshot["events"])
        self.mujoco.mj_forward(self.model, self.data)

    def _record_event(self, event: str, observable: bool = True) -> None:
        self.events.append({"time": round(float(self.data.time), 6), "event": event, "observable_online": observable})

    def _attach(self) -> None:
        self.attached = True
        self.data.eq_active[self.weld_id] = 1
        self._record_event("contact_established")

    def _detach(self, event: str) -> None:
        self.attached = False
        self.data.eq_active[self.weld_id] = 0
        if event == "contact_lost":
            self.contact_lost = True
        self._record_event(event)

    def _advance(self, target: np.ndarray | None = None, controls: int = 4) -> None:
        start = self.data.mocap_pos[0].copy()
        for control in range(controls):
            control_start = float(self.data.time)
            alpha = (control + 1) / controls
            if target is not None:
                self.data.mocap_pos[0] = start * (1 - alpha) + target * alpha
            commanded_position = self.data.mocap_pos[0].copy()
            for _ in range(5):
                if self.attached:
                    self._set_object_xyz(self.data.mocap_pos[0] + np.array([0.0, 0.0, -0.13]))
                self.mujoco.mj_step(self.model, self.data)
            self._active_control_sequence.append({
                "control_step": len(self._active_control_sequence),
                "start_time": round(control_start, 6),
                "end_time": round(float(self.data.time), 6),
                "mocap_position": [round(float(value), 9) for value in commanded_position],
                "gripper_command": "closed" if self.gripper_closed else "open",
                "physics_steps": 5,
            })
            if self._r1_control_callback is not None:
                self._r1_control_callback(self, self._active_control_sequence[-1])

    def contact_sensor(self) -> bool:
        if self.spec.scenario == "missed_grasp_then_retry" and self.failed_once and not self.attached and not self.recovered:
            return False
        distance = float(np.linalg.norm(self.object_xyz - (self.data.mocap_pos[0] + np.array([0.0, 0.0, -0.13]))))
        return bool(self.attached or (self.gripper_closed and distance < self.spec.object_radius + 0.045))

    def perform(self, action: str) -> dict[str, Any]:
        self.action_index += 1
        self._active_control_sequence = []
        before = float(self.data.time)
        obj = self.object_xyz
        target = np.array([self.spec.target_x, self.spec.target_y, 0.80])
        if action in {"observe_scene", "verify", "stop"}:
            self._advance(controls=3)
        elif action in {"approach_object", "retry", "recover"}:
            self._advance(obj + np.array([0.0, 0.0, 0.13]))
            if action in {"retry", "recover"}:
                self.gripper_closed = True
                self._attach()
                self.recovered = True
                self._record_event("recovery_achieved")
        elif action == "close_gripper":
            self.gripper_closed = True
            self._advance(controls=2)
            if self.spec.scenario == "missed_grasp_then_retry" and not self.failed_once:
                self.failed_once = True
                self._record_event("missed_grasp")
            else:
                self._attach()
        elif action == "lift":
            self._advance(self.data.mocap_pos[0] + np.array([0.0, 0.0, 0.22]))
        elif action == "transport_to_target":
            midway = (self.data.mocap_pos[0] + target) / 2
            self._advance(midway)
            if self.spec.scenario == "slip_then_recover" and not self.failed_once:
                self.failed_once = True
                self._detach("contact_lost")
                self._advance(controls=8)
            else:
                self._advance(target)
        elif action == "align":
            self._advance(target)
        elif action == "lower":
            self._advance(np.array([self.spec.target_x, self.spec.target_y, 0.5 + self.spec.object_radius * 2.1]))
        elif action == "open_gripper":
            self.gripper_closed = False
            if self.attached:
                self._detach("released")
            self._set_object_xyz((self.spec.target_x, self.spec.target_y, 0.5 + self.spec.object_radius * 1.65))
            self._advance(controls=10)
        elif action == "clear_target":
            if self.obstacle_joint >= 0:
                address = int(self.model.jnt_qposadr[self.obstacle_joint])
                self.data.qpos[address:address + 3] = (0.62, 0.38, 0.56)
                self.data.qvel[int(self.model.jnt_dofadr[self.obstacle_joint]):int(self.model.jnt_dofadr[self.obstacle_joint]) + 6] = 0
            self._record_event("target_cleared")
            self._advance(controls=3)
        else:
            raise ValueError(f"unknown high-level action: {action}")
        if self.contact_lost and self.contact_sensor():
            self._record_event("contact_reestablished")
            self.contact_lost = False
        return {
            "action_index": self.action_index, "action": action, "start_time": round(before, 6), "end_time": round(float(self.data.time), 6),
            "gripper_command": "closed" if self.gripper_closed else "open", "contact_present": self.contact_sensor(),
            "termination_reason": None, "low_level_control_sequence": self._active_control_sequence,
        }

    def oracle_snapshot(self) -> dict[str, Any]:
        delta = self.object_xyz[:2] - np.array([self.spec.target_x, self.spec.target_y])
        return {
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(), "weld_state": bool(self.data.eq_active[self.weld_id]),
            "ncon": int(self.data.ncon), "object_target_distance": float(np.linalg.norm(delta)),
            "goal_stable": bool(np.linalg.norm(delta) <= 0.055 and self.object_xyz[2] < 0.66 and not self.gripper_closed),
        }
