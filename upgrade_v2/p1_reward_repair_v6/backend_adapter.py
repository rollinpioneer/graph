"""Constraint-assisted MuJoCo backend for V6 loop verification.

Reuses the tabletop XML structure and weld-relpose idea from
upgrade_v2/visual_refine_l2 and upgrade_v2/l2r_forced_drop, but does NOT
call historical perform()/campaign collectors and does NOT write object
qpos/qvel after episode start.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .util import git_blob_sha1, sha256_bytes, sha256_file


PHYSICS_DT = 0.002
CONTROL_DT = 0.020
STEPS_PER_CONTROL = int(round(CONTROL_DT / PHYSICS_DT))  # 10
HOLD_STABLE_S = 0.10
LOSS_CONFIRM_S = 0.10
REL_HOLD_M = 0.02
REL_LOSS_M = 0.04
GOAL_XY_M = 0.06
GOAL_Z_MAX = 0.62
VEL_STABLE = 0.01
RETURN_TIMEOUT_S = 2.0


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-15:
        raise ValueError("quaternion must be non-zero")
    return np.asarray(q, dtype=float) / n


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    return q * np.asarray((1.0, -1.0, -1.0, -1.0))


def _quat_multiply_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.asarray((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ))


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _quat_normalize(_quat_multiply_raw(_quat_normalize(a), _quat_normalize(b)))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.asarray((0.0, *np.asarray(v, dtype=float)))
    qc = _quat_conjugate(q)
    out = _quat_multiply_raw(_quat_multiply_raw(_quat_normalize(q), qv), qc)
    return np.asarray(out[1:], dtype=float)


@dataclass
class FamilyPhysics:
    family_id: int
    object_radius: float
    friction: float
    mass: float
    object_xy: tuple[float, float]
    target_xy: tuple[float, float]
    disturb_xy: tuple[float, float]
    disturb_force_n: float
    table_half_z: float = 0.05
    table_z: float = 0.45

    @property
    def object_z0(self) -> float:
        return self.table_z + self.table_half_z + self.object_radius

    @property
    def target_z(self) -> float:
        return self.table_z + self.table_half_z + 0.015


def family_physics(family_id: int) -> FamilyPhysics:
    rng = np.random.default_rng(int(family_id))
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    return FamilyPhysics(
        family_id=int(family_id),
        object_radius=float(rng.uniform(0.060, 0.078)),
        friction=float(rng.uniform(0.55, 0.95)),
        mass=float(rng.uniform(0.12, 0.24)),
        object_xy=(float(rng.uniform(-0.38, -0.25)), float(rng.uniform(-0.10, 0.04))),
        target_xy=(float(rng.uniform(0.27, 0.38)), float(rng.uniform(0.05, 0.17))),
        disturb_xy=(math.cos(ang), math.sin(ang)),
        disturb_force_n=float(rng.uniform(10.0, 16.0)),
    )


def tabletop_xml(spec: FamilyPhysics) -> str:
    ox, oy = spec.object_xy
    tx, ty = spec.target_xy
    oz = spec.object_z0
    return f"""
<mujoco model="p1_v6_constraint_assisted_tabletop">
  <compiler angle="radian"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" integrator="Euler" noslip_iterations="2" iterations="50"/>
  <worldbody>
    <light directional="true" pos="0 -1 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba=".16 .18 .20 1"/>
    <geom name="table" type="box" size=".92 .66 {spec.table_half_z}" pos="0 0 {spec.table_z}" rgba=".48 .31 .16 1"/>
    <body name="target" pos="{tx:.6f} {ty:.6f} {spec.target_z:.6f}">
      <geom name="target_geom" type="cylinder" size=".145 .015" rgba=".92 .93 .88 1" contype="1" conaffinity="1"/>
    </body>
    <body name="object" pos="{ox:.6f} {oy:.6f} {oz:.6f}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="cylinder" size="{spec.object_radius:.6f} {spec.object_radius*0.82:.6f}"
            rgba=".88 .025 .035 1" mass="{spec.mass:.6f}" friction="{spec.friction:.4f} .01 .001"/>
    </body>
    <body name="gripper" mocap="true" pos="{ox:.6f} {oy-0.25:.6f} {oz+0.28:.6f}">
      <geom name="palm" type="box" size=".055 .035 .025" rgba=".02 .72 .82 1" contype="0" conaffinity="0"/>
      <geom name="finger_left" type="box" size=".018 .025 .075" pos="-.080 0 -.075" rgba=".02 .72 .82 1" contype="0" conaffinity="0"/>
      <geom name="finger_right" type="box" size=".018 .025 .075" pos=".080 0 -.075" rgba=".02 .72 .82 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grasp_weld" body1="gripper" body2="object" active="false" solref="0.002 1" solimp="0.9 0.95 0.001"/>
  </equality>
</mujoco>
"""


class PhysicalBackend:
    """Real mj_step integration. Object qpos is written only inside reset_episode."""

    backend_id = "MUJOCO_CONSTRAINT_ASSISTED_TABLETOP_V6"
    evidence_tier = "MUJOCO_CONSTRAINT_ASSISTED_STATE_VERIFICATION"

    def __init__(self, repo: Path, *, max_speed_m_s: float = 0.45):
        os.environ.pop("MUJOCO_GL", None)
        import mujoco
        self.mujoco = mujoco
        self.repo = Path(repo)
        self.max_speed = float(max_speed_m_s)
        self.model = None
        self.data = None
        self.spec: FamilyPhysics | None = None
        self.started = False
        self.finished = False
        self._direct_pose_writes_after_start = 0
        self._velocity_zeroing_after_start = 0
        self._checkpoint_resets_inside_episode = 0
        self._init_writes = 0
        self.mj_steps = 0
        self.control_steps = 0
        self.gripper_closed = False
        self.release_commanded = False
        self.recovery_command = None
        self.controller_mode = "IDLE"
        self.weld_id = None
        self.object_body = None
        self.object_qpos = None
        self.object_dof = None
        self.xml = None
        self.xml_sha256 = None
        self.mutation_ledger: list[dict] = []
        self.command_log: list[dict] = []
        self.intervention_log: list[dict] = []
        self.weld_log: list[dict] = []
        self.control_log: list[dict] = []
        self.reference_log: list[dict] = []
        self._target_pose = None
        self._ever_held = False
        self._hold_stable_s = 0.0
        self._loss_stable_s = 0.0
        self._goal_stable_s = 0.0
        self._open_loss_id = None
        self._loss_counter = 0
        self._held_now = False
        self._last_hold_snapshot = None
        self._force = np.zeros(3)
        self.auto_weld = True
        self._episode_meta = {}

    def _require_active(self):
        if not self.started or self.finished:
            raise RuntimeError("episode is not active")

    def _object_xyz(self) -> np.ndarray:
        return np.array(self.data.xpos[self.object_body], dtype=float)

    def _object_quat(self) -> np.ndarray:
        return np.array(self.data.xquat[self.object_body], dtype=float)

    def _object_vel(self) -> np.ndarray:
        # cvel is 6D in the body frame mixed; use qvel of free joint (lin, ang)
        return np.array(self.data.qvel[self.object_dof:self.object_dof+3], dtype=float)

    def _object_angvel(self) -> np.ndarray:
        return np.array(self.data.qvel[self.object_dof+3:self.object_dof+6], dtype=float)

    def _rel_grasp(self) -> float:
        grip = np.array(self.data.mocap_pos[0], dtype=float)
        offset = np.array([0.0, 0.0, -0.13])
        return float(np.linalg.norm(self._object_xyz() - (grip + offset)))

    def _set_weld(self, active: bool, reason: str):
        was = bool(self.data.eq_active[self.weld_id])
        if active and not was:
            gq = _quat_normalize(self.data.mocap_quat[0])
            oq = _quat_normalize(self.data.qpos[self.object_qpos+3:self.object_qpos+7])
            rel_p = _quat_rotate(_quat_conjugate(gq), self._object_xyz() - self.data.mocap_pos[0])
            rel_q = _quat_multiply(_quat_conjugate(gq), oq)
            self.model.eq_data[self.weld_id, 3:6] = rel_p
            self.model.eq_data[self.weld_id, 6:10] = rel_q
        self.data.eq_active[self.weld_id] = int(bool(active))
        if was != bool(active):
            self.weld_log.append({
                "t": float(self.data.time), "active": bool(active), "reason": reason,
                "mj_steps": self.mj_steps,
            })

    def _write_object_qpos_init(self, xyz, quat=(1.0,0.0,0.0,0.0)):
        if self.started and not self.finished:
            self._direct_pose_writes_after_start += 1
            raise RuntimeError("direct object qpos write after episode start is forbidden")
        self.data.qpos[self.object_qpos:self.object_qpos+3] = np.asarray(xyz, dtype=float)
        self.data.qpos[self.object_qpos+3:self.object_qpos+7] = np.asarray(quat, dtype=float)
        self.data.qvel[self.object_dof:self.object_dof+6] = 0.0
        self._init_writes += 1
        self.mutation_ledger.append({
            "when": "reset_episode", "kind": "object_qpos_qvel_init",
            "xyz": [float(x) for x in xyz], "quat": [float(x) for x in quat],
        })

    def reset_episode(self, family_spec: dict, case_spec: dict) -> dict:
        if self.started and not self.finished:
            raise RuntimeError("reset_episode during an active episode is forbidden")
        # A finished episode may be replaced by a new one; init writes are allowed only while inactive.
        self.started = False
        self.finished = False
        self.spec = family_physics(int(family_spec["family_id"]))
        # lock sampled physics into the actual XML fields
        self.xml = tabletop_xml(self.spec)
        self.xml_sha256 = sha256_bytes(self.xml.encode())
        self.model = self.mujoco.MjModel.from_xml_string(self.xml)
        self.data = self.mujoco.MjData(self.model)
        self.weld_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"))
        self.object_body = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object"))
        jid = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "object_free"))
        self.object_qpos = int(self.model.jnt_qposadr[jid])
        self.object_dof = int(self.model.jnt_dofadr[jid])
        self.mj_steps = 0
        self.control_steps = 0
        self.gripper_closed = False
        self.release_commanded = False
        self.recovery_command = None
        self.controller_mode = "RESET"
        self.mutation_ledger = []
        self.command_log = []
        self.intervention_log = []
        self.weld_log = []
        self.control_log = []
        self.reference_log = []
        self._direct_pose_writes_after_start = 0
        self._velocity_zeroing_after_start = 0
        self._checkpoint_resets_inside_episode = 0
        self._init_writes = 0
        self._ever_held = False
        self._hold_stable_s = 0.0
        self._loss_stable_s = 0.0
        self._goal_stable_s = 0.0
        self._open_loss_id = None
        self._loss_counter = 0
        self._held_now = False
        self._last_hold_snapshot = None
        self._force[:] = 0
        self.auto_weld = True
        self._target_pose = np.array(self.data.mocap_pos[0], dtype=float)
        self._write_object_qpos_init((self.spec.object_xy[0], self.spec.object_xy[1], self.spec.object_z0))
        self.data.mocap_pos[0] = np.array([self.spec.object_xy[0], self.spec.object_xy[1]-0.25, self.spec.object_z0+0.28])
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
        self.mujoco.mj_forward(self.model, self.data)
        self.started = True
        self.finished = False
        self._episode_meta = {
            "family_id": int(family_spec["family_id"]),
            "case_id": case_spec.get("case_id"),
            "rollout_seed": family_spec.get("rollout_seed"),
            "physics": {
                "object_radius": self.spec.object_radius,
                "friction": self.spec.friction,
                "mass": self.spec.mass,
                "object_xy": list(self.spec.object_xy),
                "target_xy": list(self.spec.target_xy),
                "disturb_xy": list(self.spec.disturb_xy),
                "disturb_force_n": self.spec.disturb_force_n,
                "entered_xml_fields": ["geom/object_geom/size", "geom/object_geom/friction", "geom/object_geom/mass", "body/object/pos", "body/target/pos"],
            },
        }
        self.mutation_ledger.append({"when": "reset_episode", "kind": "model_compile", "xml_sha256": self.xml_sha256})
        return dict(self._episode_meta, t=0.0, xml_sha256=self.xml_sha256)

    def set_controller_target(self, target_pose, gripper_intent: str, *, mode: str = "MOVE") -> None:
        self._require_active()
        self._target_pose = np.asarray(target_pose, dtype=float).reshape(3)
        intent = str(gripper_intent)
        if intent not in {"open", "closed", "hold_wait"}:
            raise ValueError(intent)
        prev = self.gripper_closed
        if intent == "closed":
            self.gripper_closed = True
            if mode in {"CLOSE", "DESCEND", "APPROACH", "LIFT"}:
                self.auto_weld = True
        elif intent == "open":
            self.gripper_closed = False
            self.auto_weld = False
            if mode in {"RELEASE", "RETREAT", "SETTLE_PLACE", "GOAL_WAIT"}:
                self.release_commanded = True
        self.controller_mode = mode
        self.command_log.append({
            "t": float(self.data.time), "issued": True, "start": True, "end": False,
            "target_pose": [float(x) for x in self._target_pose],
            "gripper_intent": intent, "mode": mode, "prev_closed": prev,
        })

    def issue_recovery_command(self, *, execute: bool) -> dict:
        self._require_active()
        rec = {
            "t": float(self.data.time), "command": "recover", "execute": bool(execute),
            "mode": "RECOVER" if execute else "WAIT",
        }
        self.recovery_command = rec
        self.controller_mode = rec["mode"]
        self.command_log.append(dict(rec, issued=True, start=bool(execute), end=False))
        return rec

    def apply_declared_disturbance(self, disturbance_spec: dict) -> dict:
        self._require_active()
        force = np.asarray(disturbance_spec.get("force_n", [0.0, 0.0, 0.0]), dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force_n must be a finite 3-vector")
        self._force = force
        self.data.xfrc_applied[self.object_body, :3] = force
        self.data.xfrc_applied[self.object_body, 3:] = 0.0
        row = {
            "t": float(self.data.time), "kind": "xfrc_pulse",
            "force_n": [float(x) for x in force],
            "duration_s": disturbance_spec.get("duration_s"),
            "breaks_weld": bool(disturbance_spec.get("breaks_weld", False)),
        }
        if disturbance_spec.get("breaks_weld"):
            self.auto_weld = False
            self._set_weld(False, "disturbance_constraint_break")
        self.intervention_log.append(row)
        return row

    def clear_disturbance(self) -> None:
        self._require_active()
        self._force[:] = 0
        self.data.xfrc_applied[self.object_body, :] = 0.0
        self.intervention_log.append({"t": float(self.data.time), "kind": "xfrc_clear"})

    def _update_reference(self) -> dict:
        rel = self._rel_grasp()
        weld = bool(self.data.eq_active[self.weld_id])
        vel = float(np.linalg.norm(self._object_vel()))
        ang = float(np.linalg.norm(self._object_angvel()))
        xyz = self._object_xyz()
        goal_d = float(np.linalg.norm(xyz[:2] - np.asarray(self.spec.target_xy)))
        support = {
            "weld_active": weld,
            "relative_grasp_m": rel,
            "ncon": int(self.data.ncon),
            "gripper_closed": self.gripper_closed,
            "object_speed": vel,
            "object_ang_speed": ang,
        }
        hold_candidate = bool(self.gripper_closed and weld and rel <= REL_HOLD_M and self.auto_weld)
        if hold_candidate:
            self._hold_stable_s += CONTROL_DT
        else:
            self._hold_stable_s = 0.0
        held = bool(hold_candidate and self._hold_stable_s + 1e-12 >= HOLD_STABLE_S)
        if held:
            self._ever_held = True
            self._held_now = True
            self._loss_stable_s = 0.0
            self._last_hold_snapshot = self.read_integration_state()
        detached = bool(self._ever_held and (not weld or rel >= REL_LOSS_M) and not self.release_commanded)
        if detached and not held:
            self._loss_stable_s += CONTROL_DT
        else:
            self._loss_stable_s = 0.0
        events = []
        if detached and self._loss_stable_s + 1e-12 >= LOSS_CONFIRM_S and self._open_loss_id is None and self._held_now:
            self._loss_counter += 1
            self._open_loss_id = f"L{self._loss_counter}"
            self._held_now = False
            events.append({"kind": "LOSS", "object_id": "obj", "loss_id": self._open_loss_id})
        if held and self._open_loss_id is not None:
            events.append({"kind": "HOLD_REESTABLISHED", "object_id": "obj", "loss_id": self._open_loss_id})
            self._open_loss_id = None
        in_goal = bool(goal_d <= GOAL_XY_M and xyz[2] <= GOAL_Z_MAX and not held and vel <= 0.05)
        if in_goal:
            self._goal_stable_s += CONTROL_DT
        else:
            self._goal_stable_s = 0.0
        valid = bool(in_goal and self._goal_stable_s + 1e-12 >= HOLD_STABLE_S)
        rec_started = bool(self.recovery_command and self._open_loss_id and not held)
        if rec_started and self.recovery_command and not self.recovery_command.get("started_logged"):
            events.append({"kind": "RECOVERY_STARTED", "object_id": "obj", "loss_id": self._open_loss_id})
            self.recovery_command["started_logged"] = True
        phase = "WAIT"
        if valid:
            phase = "VALID"
        elif self._open_loss_id and not held:
            phase = "RECOVERING" if rec_started else "LOST"
        elif held and xyz[2] >= self.spec.table_z + self.spec.table_half_z + self.spec.object_radius + 0.04:
            phase = "TRANSPORT"
        elif held:
            phase = "HELD"
        elif in_goal and not held:
            phase = "PLACED"
        ref = {
            "t": float(self.data.time),
            "held": held,
            "ever_held": self._ever_held,
            "valid": valid,
            "phase": phase,
            "open_loss_id": self._open_loss_id,
            "release_commanded": self.release_commanded,
            "recovery_command_issued": bool(self.recovery_command),
            "recovery_executing": bool(self.recovery_command and self.recovery_command.get("execute")),
            "goal_xy_error": goal_d,
            "events": events,
            "support": support,
            "controller_mode": self.controller_mode,
        }
        self.reference_log.append(ref)
        return ref

    def advance_control_interval(self) -> dict:
        self._require_active()
        start_t = float(self.data.time)
        start_pos = np.array(self.data.mocap_pos[0], dtype=float)
        target = start_pos if self._target_pose is None or self.controller_mode in {"WAIT", "IDLE"} else self._target_pose
        delta = target - start_pos
        dist = float(np.linalg.norm(delta))
        max_step = self.max_speed * CONTROL_DT
        if dist > max_step and dist > 1e-9:
            dest = start_pos + delta * (max_step / dist)
        else:
            dest = target
        for i in range(STEPS_PER_CONTROL):
            a = (i + 1) / STEPS_PER_CONTROL
            self.data.mocap_pos[0] = start_pos * (1.0 - a) + dest * a
            if self.auto_weld and self.gripper_closed and not bool(self.data.eq_active[self.weld_id]) and self._rel_grasp() <= REL_HOLD_M:
                self._set_weld(True, "close_near_object")
            if (not self.gripper_closed) and bool(self.data.eq_active[self.weld_id]):
                self._set_weld(False, "open_release")
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("NUMERICAL_INVALID")
            self.mujoco.mj_step(self.model, self.data)
            self.mj_steps += 1
        self.control_steps += 1
        ref = self._update_reference()
        row = {
            "control_step": self.control_steps,
            "start_time": start_t,
            "end_time": float(self.data.time),
            "physics_steps": STEPS_PER_CONTROL,
            "dt_s": PHYSICS_DT,
            "expected_time": self.mj_steps * PHYSICS_DT,
            "mocap": [float(x) for x in self.data.mocap_pos[0]],
            "gripper_closed": self.gripper_closed,
            "mode": self.controller_mode,
        }
        self.control_log.append(row)
        return dict(row, reference=ref, state=self.read_current_state())

    def read_current_state(self) -> dict:
        self._require_active()
        xyz = self._object_xyz()
        return {
            "t": float(self.data.time),
            "physical_time_ns": int(round(float(self.data.time) * 1e9)),
            "eef": [float(x) for x in self.data.mocap_pos[0]],
            "eef_quat": [float(x) for x in self.data.mocap_quat[0]],
            "gripper_closed": self.gripper_closed,
            "objects": {
                "obj": {
                    "pos": [float(x) for x in xyz],
                    "vel": [float(x) for x in self._object_vel()],
                    "quat": [float(x) for x in self._object_quat()],
                    "angular_vel": [float(x) for x in self._object_angvel()],
                    "target_xy": [float(x) for x in self.spec.target_xy],
                    "target_z": float(self.spec.target_z),
                }
            },
            "weld_active": bool(self.data.eq_active[self.weld_id]),
            "ncon": int(self.data.ncon),
            "controller_mode": self.controller_mode,
        }

    def read_integration_state(self) -> dict:
        self._require_active()
        return {
            "t": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy() if self.data.act is not None else np.zeros(0),
            "mocap_pos": self.data.mocap_pos.copy(),
            "mocap_quat": self.data.mocap_quat.copy(),
            "eq_active": self.data.eq_active.copy(),
            "xfrc_applied": self.data.xfrc_applied.copy(),
            "time_s": float(self.data.time),
        }

    def restore_checkpoint_forbidden(self, _snapshot: dict) -> None:
        self._checkpoint_resets_inside_episode += 1
        raise RuntimeError("checkpoint restore inside episode is forbidden for closure")

    def finish_episode(self) -> dict:
        self._require_active()
        self.finished = True
        return {
            "mj_steps": self.mj_steps,
            "expected_time_s": self.mj_steps * PHYSICS_DT,
            "actual_time_s": float(self.data.time),
            "direct_pose_overwrites_after_start": self._direct_pose_writes_after_start,
            "velocity_zeroing_after_start": self._velocity_zeroing_after_start,
            "checkpoint_resets_inside_episode": self._checkpoint_resets_inside_episode,
            "init_state_writes": self._init_writes,
            "weld_changes": len(self.weld_log),
            "xml_sha256": self.xml_sha256,
            "mujoco_version": getattr(self.mujoco, "__version__", None),
        }

    def source_audit(self) -> dict:
        files = [
            self.repo/"upgrade_v2/visual_refine_l2/dynamic_simulator.py",
            self.repo/"upgrade_v2/visual_refine_l2/repaired_simulator.py",
            self.repo/"upgrade_v2/l2r_forced_drop/simulator.py",
        ]
        src = {}
        for f in files:
            if f.is_file():
                src[str(f)] = {"sha256": sha256_file(f), "git_blob": git_blob_sha1(f.read_bytes()), "bytes": f.stat().st_size}
        return {
            "backend_id": self.backend_id,
            "evidence_tier": self.evidence_tier,
            "reused": "XML tabletop + weld relpose; NOT DynamicTabletop.perform writeback/campaign",
            "forbidden_writeback_in_this_backend": True,
            "sources": src,
            "physics_dt_s": PHYSICS_DT,
            "control_dt_s": CONTROL_DT,
            "steps_per_control": STEPS_PER_CONTROL,
        }

