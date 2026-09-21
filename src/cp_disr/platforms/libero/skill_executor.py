"""Finite-state OSC skill controllers. No teleport, no fact writes, no VLM prior."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .d0_env import OBJECT_HALF, LID_HALF, CONTAINER_H


# This robosuite Panda mapping: +1 closes, -1 opens.
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0


@dataclass
class SkillTrace:
    skill: str
    arguments: tuple[str, ...]
    states: list[str] = field(default_factory=list)
    exit_reason: str = "IN_PROGRESS"
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    steps: int = 0
    rejected: bool = False
    interrupted: bool = False
    timeout: bool = False
    nan_blocked: bool = False


def _toward(current, target, gain=6.0, clip=0.06):
    err = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    step = np.clip(gain * err, -clip, clip)
    scale = 20.0  # 0.05m -> 1.0
    return np.clip(step * scale, -1.0, 1.0), float(np.linalg.norm(err))


class SkillExecutor:
    def __init__(self, env, safety, clock):
        self.env = env
        self.safety = safety
        self.clock = clock

    def _hover_z(self):
        return float(self.env.table_top_z + 0.22)

    def _refresh_perception(self):
        cb = getattr(self.env, "refresh_perception", None)
        if cb is not None:
            cb()

    def execute(self, candidate_id: str, timeout_seconds: float) -> dict:
        start = self.clock.now_seconds()
        sim_start = float(self.env.sim.data.time)
        parts = candidate_id.split(":")
        skill = parts[1]
        args = tuple(parts[2:-1])
        trace = SkillTrace(skill=skill, arguments=args, start_seconds=start)
        try:
            self._refresh_perception()
            obs = self.env.public_observation()
            if not self.safety.can_execute(candidate_id, obs):
                trace.rejected = True
                trace.exit_reason = "REJECTED_UNSAFE_OR_INVALID"
                return self._finish(trace, start, sim_start)
            if skill == "OPEN":
                self._open(trace, timeout_seconds, sim_start)
            elif skill == "PICK":
                if not args:
                    trace.rejected = True
                    trace.exit_reason = "REJECTED_INVALID_ARGUMENTS"
                else:
                    self._pick(trace, args[0], timeout_seconds, sim_start)
            elif skill == "PLACE":
                if len(args) < 2:
                    trace.rejected = True
                    trace.exit_reason = "REJECTED_INVALID_ARGUMENTS"
                else:
                    self._place(trace, args[0], args[1], timeout_seconds, sim_start)
            elif skill == "PLACE_BUFFER":
                if len(args) < 2:
                    trace.rejected = True
                    trace.exit_reason = "REJECTED_INVALID_ARGUMENTS"
                else:
                    self._place_buffer(trace, args[0], args[1], timeout_seconds, sim_start)
            elif skill == "MOVE":
                trace.rejected = True
                trace.exit_reason = "REJECTED_UNKNOWN_SKILL"
            else:
                trace.rejected = True
                trace.exit_reason = "REJECTED_UNKNOWN_SKILL"
        except Exception as exc:  # noqa: BLE001
            if "NaN" in str(exc) or "Inf" in str(exc):
                trace.nan_blocked = True
                trace.exit_reason = "INTERRUPT_NAN_ACTION"
            else:
                trace.interrupted = True
                trace.exit_reason = "INTERRUPT_CONTROLLER_EXCEPTION:" + type(exc).__name__
        return self._finish(trace, start, sim_start)

    def _finish(self, trace, start, sim_start):
        trace.end_seconds = self.clock.now_seconds()
        if trace.exit_reason == "IN_PROGRESS":
            if trace.timeout:
                trace.exit_reason = "TIMEOUT"
            elif trace.rejected:
                trace.exit_reason = "REJECTED_UNSAFE_OR_INVALID"
            else:
                trace.exit_reason = "INCOMPLETE_MOTION"
        elapsed = float(self.env.sim.data.time) - sim_start
        if elapsed <= 0:
            elapsed = max(0.05, trace.steps / 20.0)
        return {
            "execution_id": f"{trace.skill}-{int(start * 1000)}",
            "controller_exit": trace.exit_reason,
            "start_seconds": start,
            "end_seconds": trace.end_seconds,
            "evidence_ids": (f"trace:{trace.skill}",),
            "skill": trace.skill,
            "arguments": trace.arguments,
            "states": list(trace.states),
            "sim_duration": elapsed,
            "steps": trace.steps,
            "rejected": trace.rejected,
            "timeout": trace.timeout,
            "interrupted": trace.interrupted,
            "nan_blocked": trace.nan_blocked,
        }

    def _move_to(self, trace, target_xyz, grip, timeout, sim_start, tol=0.018, max_steps=220, tag="APPROACH"):
        trace.states.append(tag)
        for _ in range(max_steps):
            if (float(self.env.sim.data.time) - sim_start) >= timeout:
                trace.timeout = True
                trace.exit_reason = "TIMEOUT"
                return False
            obs = self.env.public_observation()
            dpos, err = _toward(obs["eef_pos"], target_xyz)
            self.env.step_osc(dpos, grip, n=1)
            trace.steps += 1
            if err < tol:
                return True
        trace.exit_reason = "INCOMPLETE_MOTION"
        return False

    def _hold(self, trace, grip, n, timeout, sim_start):
        for _ in range(n):
            if (float(self.env.sim.data.time) - sim_start) >= timeout:
                trace.timeout = True
                trace.exit_reason = "TIMEOUT"
                return False
            self.env.step_osc(np.zeros(3), grip, n=1)
            trace.steps += 1
        return True

    def _object_xyz(self, name):
        self._refresh_perception()
        est = self.env.last_perception.get(name)
        if est is None:
            raise RuntimeError("perception_unavailable")
        return np.array(est, dtype=float)

    def _grasp_z(self, perc_z, half_z):
        table = float(self.env.table_top_z)
        # Color blob z is biased high (visible top). Grasp at object center.
        return float(min(perc_z - 0.012, table + half_z + 0.002))

    def _open(self, trace, timeout, sim_start):
        # Grasp the free lid and park it on-table, still in camera view.
        lid = self._object_xyz("lid")
        hover = np.array([lid[0], lid[1], self._hover_z()])
        grasp = np.array([lid[0], lid[1], float(lid[2]) - 0.008])
        park_xy = np.array([0.02, 0.18])
        park = np.array([park_xy[0], park_xy[1], self._hover_z()])
        drop = np.array([park_xy[0], park_xy[1], float(self.env.table_top_z + LID_HALF[2] + 0.012)])
        if not self._move_to(trace, hover, GRIP_OPEN, timeout, sim_start, tag="APPROACH", tol=0.025, max_steps=200):
            return
        reached = self._move_to(trace, grasp, GRIP_OPEN, timeout, sim_start, tag="INTERACT", tol=0.016, max_steps=300)
        eef = np.asarray(self.env.public_observation()["eef_pos"], dtype=float)
        if (not reached) and float(np.linalg.norm(eef - grasp)) > 0.035:
            return
        if not self._hold(trace, GRIP_CLOSE, 36, timeout, sim_start):
            return
        lift = np.array([lid[0], lid[1], float(self.env.table_top_z + 0.16)])
        if not self._move_to(trace, lift, GRIP_CLOSE, timeout, sim_start, tag="RETREAT", tol=0.02, max_steps=180):
            return
        if not self._move_to(trace, park, GRIP_CLOSE, timeout, sim_start, tag="INTERACT", tol=0.025, max_steps=220):
            return
        if not self._move_to(trace, drop, GRIP_CLOSE, timeout, sim_start, tag="INTERACT", tol=0.015, max_steps=180):
            return
        if not self._hold(trace, GRIP_OPEN, 24, timeout, sim_start):
            return
        if self._move_to(trace, park, GRIP_OPEN, timeout, sim_start, tag="RETREAT", tol=0.03, max_steps=160):
            trace.exit_reason = "NORMAL_TERMINATION"

    def _pick(self, trace, obj, timeout, sim_start):
        xyz = self._object_xyz(obj)
        hover = np.array([xyz[0], xyz[1], self._hover_z()])
        grasp = np.array([xyz[0], xyz[1], self._grasp_z(float(xyz[2]), OBJECT_HALF[2])])
        press = np.array([grasp[0], grasp[1], grasp[2] - 0.008])
        if not self._move_to(trace, hover, GRIP_OPEN, timeout, sim_start, tag="APPROACH", tol=0.02, max_steps=180):
            return
        if not self._move_to(trace, grasp, GRIP_OPEN, timeout, sim_start, tag="INTERACT", tol=0.008, max_steps=260):
            return
        if not self._move_to(trace, press, GRIP_OPEN, timeout, sim_start, tag="INTERACT", tol=0.01, max_steps=80):
            pass
        if not self._hold(trace, GRIP_CLOSE, 36, timeout, sim_start):
            return
        # Present the held object to agentview so RGB-D postcondition is observable.
        show = np.array([0.02, -0.08, float(self.env.table_top_z + 0.16)])
        if not self._move_to(trace, show, GRIP_CLOSE, timeout, sim_start, tag="RETREAT", tol=0.03, max_steps=200):
            if not self._move_to(trace, hover, GRIP_CLOSE, timeout, sim_start, tag="RETREAT", tol=0.04, max_steps=120):
                return
        self._refresh_perception()
        trace.exit_reason = "NORMAL_TERMINATION"

    def _place(self, trace, obj, container, timeout, sim_start):
        c = np.array(self.env.container_center, dtype=float)
        hover = np.array([c[0], c[1], self._hover_z()])
        drop = np.array([c[0], c[1], self.env.table_top_z + CONTAINER_H + 0.03])
        if not self._move_to(trace, hover, GRIP_CLOSE, timeout, sim_start, tag="APPROACH"):
            return
        if not self._move_to(trace, drop, GRIP_CLOSE, timeout, sim_start, tag="INTERACT"):
            return
        if not self._hold(trace, GRIP_OPEN, 16, timeout, sim_start):
            return
        if self._move_to(trace, hover, GRIP_OPEN, timeout, sim_start, tag="RETREAT"):
            trace.exit_reason = "NORMAL_TERMINATION"

    def _place_buffer(self, trace, obj, buffer, timeout, sim_start):
        b = np.array(self.env.buffer_center, dtype=float)
        hover = np.array([b[0], b[1], self._hover_z()])
        drop = np.array([b[0], b[1], self.env.table_top_z + OBJECT_HALF[2] + 0.05])
        if not self._move_to(trace, hover, GRIP_CLOSE, timeout, sim_start, tag="APPROACH"):
            return
        if not self._move_to(trace, drop, GRIP_CLOSE, timeout, sim_start, tag="INTERACT"):
            return
        if not self._hold(trace, GRIP_OPEN, 16, timeout, sim_start):
            return
        if self._move_to(trace, hover, GRIP_OPEN, timeout, sim_start, tag="RETREAT"):
            trace.exit_reason = "NORMAL_TERMINATION"
