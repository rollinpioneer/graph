"""Finite-state OSC skill controllers. No teleport, no fact writes, no VLM prior."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .d0_env import OBJECT_HALF, LID_HALF, CONTAINER_H
from .clock import CONTROL_DT


# This robosuite Panda mapping: +1 closes, -1 opens.
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0
GRIP_OPEN_QPOS = 0.022


class PerceptionMissing(Exception):
    """Object is not currently localizable. Normal runtime failure, not a crash."""

    def __init__(self, name, reason="object_not_in_perception"):
        self.name = name
        self.reason = reason
        super().__init__("perception_missing:%s:%s" % (name, reason))


class CriticalFactLost(Exception):
    """A previously observed required object is no longer localizable."""

    def __init__(self, name, reason="previously_seen_object_missing"):
        self.name = name
        self.reason = reason
        super().__init__("critical_fact_lost:%s:%s" % (name, reason))


class PerceptionBindingError(RuntimeError):
    """Perception callback, binding, or coordinate payload is technically invalid."""


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
    needs_confirm: bool = False
    confirm_grip: float | None = None


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
        self._last_xyz = {}

    def _hover_z(self):
        return float(self.env.table_top_z + 0.22)

    def _refresh_perception(self):
        cb = getattr(self.env, "refresh_perception", None)
        if cb is None:
            raise PerceptionBindingError("refresh_perception_unbound")
        try:
            return cb()
        except PerceptionMissing:
            raise
        except CriticalFactLost:
            raise
        except PerceptionBindingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PerceptionBindingError("refresh_perception_failed:%s:%s" % (type(exc).__name__, str(exc).splitlines()[0][:120])) from exc

    def execute(self, candidate_id: str, timeout_seconds: float) -> dict:
        start = self.clock.now_seconds()
        sim_start = float(self.env.sim.data.time)
        parts = candidate_id.split(":")
        skill = parts[1] if len(parts) > 1 else ""
        args = tuple(parts[2:-1]) if len(parts) > 3 else tuple(parts[2:])
        trace = SkillTrace(skill=skill, arguments=args, start_seconds=start)
        try:
            self._refresh_perception()
            obs = self.env.public_observation()
            if not self.safety.can_execute(candidate_id, obs):
                trace.rejected = True
                trace.exit_reason = "REJECTED_UNSAFE_OR_INVALID"
                return self._finish(trace, start, sim_start, timeout_seconds)
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
        except PerceptionMissing as exc:
            trace.exit_reason = "EXECUTION_FAILED"
            trace.needs_confirm = True
            trace.states.append("PERCEPTION_MISSING:%s" % exc.name)
        except CriticalFactLost as exc:
            trace.exit_reason = "CRITICAL_FACT_LOST"
            trace.needs_confirm = True
            trace.states.append("CRITICAL_FACT_LOST:%s" % exc.name)
        except PerceptionBindingError as exc:
            trace.interrupted = True
            detail = str(exc).splitlines()[0][:120]
            trace.exit_reason = "INTERRUPT_CONTROLLER_EXCEPTION:%s:%s" % (type(exc).__name__, detail)
            trace.states.append("EXCEPTION")
        except Exception as exc:  # noqa: BLE001
            if "NaN" in str(exc) or "Inf" in str(exc):
                trace.nan_blocked = True
                trace.exit_reason = "INTERRUPT_NAN_ACTION"
            else:
                trace.interrupted = True
                detail = str(exc).splitlines()[0][:120]
                trace.exit_reason = "INTERRUPT_CONTROLLER_EXCEPTION:%s:%s" % (type(exc).__name__, detail)
            trace.states.append("EXCEPTION")
        return self._finish(trace, start, sim_start, timeout_seconds)

    def _current_grip_command(self):
        try:
            obs = self.env.public_observation()
        except Exception:
            return None
        if not isinstance(obs, dict) or "gripper_qpos" not in obs:
            return None
        grip = np.asarray(obs["gripper_qpos"], dtype=float)
        if grip.size == 0 or not np.all(np.isfinite(grip)):
            return None
        return GRIP_OPEN if abs(float(grip.reshape(-1)[0])) > GRIP_OPEN_QPOS else GRIP_CLOSE

    def _maybe_confirm_tick(self, trace, remaining):
        """Hold/confirm only with independent safety permission. Never invent time."""
        if trace.steps != 0 or trace.interrupted or trace.nan_blocked:
            return
        if not (trace.rejected or trace.needs_confirm):
            return
        hold_fn = getattr(self.safety, "can_hold", None)
        if hold_fn is None:
            trace.states.append("CONFIRM_TICK_DENIED_NO_HOLD_API")
            return
        try:
            obs = self.env.public_observation()
            allowed = bool(hold_fn(obs))
        except Exception:
            allowed = False
        if not allowed:
            trace.states.append("CONFIRM_TICK_DENIED_NO_HOLD_PERMISSION")
            return
        if remaining is None or (not math.isfinite(float(remaining))) or float(remaining) < CONTROL_DT:
            trace.states.append("CONFIRM_TICK_DENIED_INSUFFICIENT_REMAINING")
            return
        grip = self._current_grip_command()
        if grip is None:
            trace.states.append("CONFIRM_TICK_DENIED_UNKNOWN_GRIP")
            return
        try:
            self.env.step_osc(np.zeros(3), float(grip), n=1)
            trace.steps += 1
            trace.confirm_grip = float(grip)
            trace.states.append("CONFIRM_TICK_HOLD")
        except Exception as exc:  # noqa: BLE001
            trace.interrupted = True
            detail = str(exc).splitlines()[0][:120]
            if "NaN" in str(exc) or "Inf" in str(exc):
                trace.nan_blocked = True
                trace.exit_reason = "INTERRUPT_NAN_ACTION"
            else:
                trace.exit_reason = "INTERRUPT_CONTROLLER_EXCEPTION:%s:%s" % (type(exc).__name__, detail)
            trace.states.append("CONFIRM_TICK_FAILED")

    def _finish(self, trace, start, sim_start, timeout_seconds):
        remaining = float(timeout_seconds) - (float(self.env.sim.data.time) - float(sim_start))
        self._maybe_confirm_tick(trace, remaining=remaining)
        sim_end = float(self.env.sim.data.time)
        elapsed = sim_end - float(sim_start)
        trace.end_seconds = self.clock.now_seconds()
        if trace.exit_reason == "IN_PROGRESS":
            if trace.timeout:
                trace.exit_reason = "TIMEOUT"
            elif trace.rejected:
                trace.exit_reason = "REJECTED_UNSAFE_OR_INVALID"
            else:
                trace.exit_reason = "INCOMPLETE_MOTION"
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
            "raw_sim_start": float(sim_start),
            "raw_sim_end": sim_end,
            "steps": trace.steps,
            "rejected": trace.rejected,
            "timeout": trace.timeout,
            "interrupted": trace.interrupted,
            "nan_blocked": trace.nan_blocked,
            "needs_confirm": trace.needs_confirm,
            "confirm_grip": trace.confirm_grip,
        }

    def _move_to(self, trace, target_xyz, grip, timeout, sim_start, tol=0.018, max_steps=220, tag="APPROACH"):
        trace.states.append(tag)
        for _ in range(max_steps):
            if (float(self.env.sim.data.time) - sim_start) >= timeout:
                trace.timeout = True
                trace.exit_reason = "TIMEOUT"
                return False
            remaining = float(timeout) - (float(self.env.sim.data.time) - float(sim_start))
            if remaining < CONTROL_DT:
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
            remaining = float(timeout) - (float(self.env.sim.data.time) - float(sim_start))
            if remaining < CONTROL_DT:
                trace.timeout = True
                trace.exit_reason = "TIMEOUT"
                return False
            self.env.step_osc(np.zeros(3), grip, n=1)
            trace.steps += 1
        return True

    def _object_xyz(self, name):
        self._refresh_perception()
        perc = getattr(self.env, "last_perception", None)
        if perc is None:
            raise PerceptionBindingError("last_perception_unbound")
        if not isinstance(perc, dict):
            raise PerceptionBindingError("last_perception_not_dict:%s" % type(perc).__name__)
        if name not in perc or perc.get(name) is None:
            if name in self._last_xyz:
                raise CriticalFactLost(name)
            raise PerceptionMissing(name)
        try:
            xyz = np.array(perc[name], dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            raise PerceptionBindingError("malformed_xyz:%s:%s" % (name, type(exc).__name__)) from exc
        if xyz.size < 3 or not np.all(np.isfinite(xyz[:3])):
            raise PerceptionBindingError("nonfinite_or_short_xyz:%s" % name)
        out = xyz[:3].copy()
        self._last_xyz[name] = out
        return out

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