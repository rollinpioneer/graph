"""Eventful explicit-state simulator for the U2 stochastic-boundary prototype."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .event_schema import EVENT_IDS, MODE_IDS


SCENARIOS = (
    "nominal_success", "grazing_contact", "slip_recovery", "obstacle_detour", "terminal_collision", "stagnation",
)


@dataclass(frozen=True)
class FamilySpec:
    root_family_id: str
    scenario: str
    side: int
    obstacle_margin: float
    process_noise: float
    slip_step: int
    stagnation_steps: int


class StochasticBoundarySimulator:
    """2-D gripper/object proxy with sensors, gold events, and exact snapshots.

    Scenario and controller state are hidden generation variables.  The public
    observation is the fixed 17-dimensional sensor/action history described in
    the U2 protocol.
    """

    observation_dim = 17
    action_dim = 2
    horizon = 48
    goal = np.asarray([0.86, 0.50], dtype=np.float64)
    obstacle = np.asarray([0.50, 0.50], dtype=np.float64)
    obstacle_radius = 0.115
    goal_radius = 0.070
    snapshot_schema = "u2_stochastic_boundary_snapshot_v1"

    def __init__(self, family: FamilySpec, rollout_seed: int):
        if family.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {family.scenario}")
        self.family = family
        self.rollout_seed = int(rollout_seed)
        self.rng = np.random.default_rng(self.rollout_seed)
        self.reset()

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else np.zeros(2, dtype=np.float64)

    def reset(self) -> np.ndarray:
        side = float(self.family.side)
        self.agent_pos = np.asarray([0.10, 0.28 + 0.04 * side], dtype=np.float64)
        self.agent_vel = np.zeros(2, dtype=np.float64)
        self.object_pos = np.asarray([0.26, 0.30 + 0.04 * side], dtype=np.float64)
        self.object_vel = np.zeros(2, dtype=np.float64)
        self.contact = False
        self.collision = False
        self.object_in_goal = False
        self.stable_goal_steps = 0
        self.step_index = 0
        self.done = False
        self.success = False
        self.terminal_reason = ""
        self.phase = "approach"
        self.ever_transport = False
        self.transport_pending = False
        self.ever_lost = False
        self.ever_recovery_started = False
        self.ever_detour = False
        self.ever_stagnation = False
        self.loss_pending = False
        self.recovery_start_pending = False
        self.grazing_recovery_blocked = False
        self.recovery_wait = 0
        self.path_length = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.goal_distance_history = [self.goal_distance]
        self.event_log: list[dict[str, Any]] = []
        return self.observable()

    @property
    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.object_pos))

    @property
    def agent_object_distance(self) -> float:
        return float(np.linalg.norm(self.object_pos - self.agent_pos))

    @property
    def object_clearance(self) -> float:
        return float(np.linalg.norm(self.object_pos - self.obstacle) - self.obstacle_radius)

    def observable(self) -> np.ndarray:
        # 17 fields: relative geometry, velocities, binary sensors, previous action.
        return np.asarray([
            *(self.agent_pos - self.object_pos), *self.agent_vel,
            *(self.goal - self.object_pos), *self.object_vel,
            *(self.agent_pos - self.obstacle), *(self.object_pos - self.obstacle),
            float(self.contact), float(self.collision), float(self.object_in_goal), *self.prev_action,
        ], dtype=np.float32)

    def state_vector(self) -> np.ndarray:
        return np.asarray([
            *self.agent_pos, *self.agent_vel, *self.object_pos, *self.object_vel,
            float(self.contact), float(self.collision), float(self.object_in_goal),
            float(self.stable_goal_steps), float(self.step_index), float(self.loss_pending),
            float(self.transport_pending), float(self.recovery_start_pending),
            float(self.recovery_wait), float(self.path_length), *self.prev_action,
        ], dtype=np.float64)

    def gold_events(self) -> list[dict[str, Any]]:
        return list(self.event_log)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.snapshot_schema, "family": self.family.__dict__, "rollout_seed": self.rollout_seed,
            "agent_pos": self.agent_pos.tolist(), "agent_vel": self.agent_vel.tolist(),
            "object_pos": self.object_pos.tolist(), "object_vel": self.object_vel.tolist(),
            "contact": self.contact, "collision": self.collision, "object_in_goal": self.object_in_goal,
            "stable_goal_steps": self.stable_goal_steps, "step_index": self.step_index, "done": self.done,
            "success": self.success, "terminal_reason": self.terminal_reason, "phase": self.phase,
            "ever_transport": self.ever_transport, "transport_pending": self.transport_pending,
            "ever_lost": self.ever_lost,
            "ever_recovery_started": self.ever_recovery_started, "ever_detour": self.ever_detour,
            "ever_stagnation": self.ever_stagnation, "loss_pending": self.loss_pending,
            "recovery_start_pending": self.recovery_start_pending,
            "grazing_recovery_blocked": self.grazing_recovery_blocked,
            "recovery_wait": self.recovery_wait, "path_length": self.path_length,
            "prev_action": self.prev_action.tolist(), "goal_distance_history": list(self.goal_distance_history),
            "event_log": copy.deepcopy(self.event_log),
            "rng_state": json.loads(json.dumps(copy.deepcopy(self.rng.bit_generator.state))),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != self.snapshot_schema:
            raise ValueError("unsupported U2 simulator snapshot")
        self.agent_pos = np.asarray(snapshot["agent_pos"], dtype=np.float64)
        self.agent_vel = np.asarray(snapshot["agent_vel"], dtype=np.float64)
        self.object_pos = np.asarray(snapshot["object_pos"], dtype=np.float64)
        self.object_vel = np.asarray(snapshot["object_vel"], dtype=np.float64)
        for key in ("contact", "collision", "object_in_goal", "done", "success", "ever_transport", "transport_pending", "ever_lost", "ever_recovery_started", "ever_detour", "ever_stagnation", "loss_pending", "recovery_start_pending", "grazing_recovery_blocked"):
            setattr(self, key, bool(snapshot[key]))
        for key in ("stable_goal_steps", "step_index", "recovery_wait"):
            setattr(self, key, int(snapshot[key]))
        self.terminal_reason = str(snapshot["terminal_reason"])
        self.phase = str(snapshot["phase"])
        self.path_length = float(snapshot["path_length"])
        self.prev_action = np.asarray(snapshot["prev_action"], dtype=np.float64)
        self.goal_distance_history = [float(value) for value in snapshot["goal_distance_history"]]
        self.event_log = list(snapshot["event_log"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = snapshot["rng_state"]

    def policy_action(self) -> np.ndarray:
        """Hidden controller used only to generate trajectories and gold transitions."""
        scenario = self.family.scenario
        if self.phase in {"recovery_approach", "contact_lost"}:
            target = self.object_pos
        elif scenario == "stagnation" and self.step_index < self.family.stagnation_steps:
            return np.zeros(2, dtype=np.float64)
        else:
            if not self.contact:
                target = self.object_pos
            else:
                push_direction = self.goal - self.object_pos
                obstacle_distance = float(np.linalg.norm(self.object_pos - self.obstacle))
                if scenario == "terminal_collision" and not self.collision:
                    push_direction = self.obstacle - self.object_pos
                elif (scenario != "terminal_collision" and self.object_pos[0] < 0.66
                      and obstacle_distance < 0.31):
                    # The controller's hidden route bends around the obstacle;
                    # the learner gets only the resulting observable geometry.
                    push_direction = np.asarray([1.0, 0.86 * self.family.side], dtype=np.float64)
                jitter = self.rng.normal(0.0, self.family.process_noise * 0.18, size=2)
                return np.clip(0.78 * self._unit(push_direction) + jitter, -1.0, 1.0)
        jitter = self.rng.normal(0.0, self.family.process_noise * 0.18, size=2)
        return np.clip(self._unit(target - self.agent_pos) + jitter, -1.0, 1.0)

    def _choose_event(self, names: list[str]) -> str:
        priority = ("terminal_failure", "stable_success", "goal_enter", "contact_reestablished", "contact_off_failure", "recovery_start", "detour_start", "stagnation_onset", "contact_on", "transport_on")
        for name in priority:
            if name in names:
                return name
        return "none"

    def step(self, action: np.ndarray) -> tuple[np.ndarray, int, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step called after terminal state")
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        contact_before, goal_before = self.contact, self.object_in_goal
        previous_object = self.object_pos.copy()
        noise = self.rng.normal(0.0, self.family.process_noise, size=4)
        self.agent_vel = 0.45 * self.agent_vel + 0.050 * action + noise[:2]
        self.agent_pos = np.clip(self.agent_pos + self.agent_vel, 0.0, 1.0)
        events: list[str] = []
        # An explicit loss must first emit recovery_start; automatic contact is
        # therefore disabled while the loss transition is pending.
        if not self.contact and not self.loss_pending and self.agent_object_distance < 0.105:
            self.contact = True
            self.phase = "contact_established" if not self.ever_lost else "contact_reestablished"
            events.append("contact_reestablished" if self.ever_lost else "contact_on")
        force_loss = self.family.scenario == "slip_recovery" and self.step_index == self.family.slip_step and self.contact
        grazing_loss = (self.family.scenario == "grazing_contact" and self.contact and not self.ever_lost
                        and self.step_index == 4 and self.rng.random() < 0.50)
        if force_loss or grazing_loss:
            self.contact = False
            self.ever_lost = True
            self.loss_pending = True
            self.recovery_start_pending = True
            if grazing_loss:
                # Some grazing contacts lead to an unrecoverable separation;
                # this is determined before rollout and remains hidden state.
                self.grazing_recovery_blocked = self.rollout_seed % 3 == 0
            self.recovery_wait = 0
            self.phase = "contact_lost"
            events.append("contact_off_failure")
        recovery_started_now = False
        if self.contact:
            self.object_vel = 0.40 * self.object_vel + 0.34 * self.agent_vel + noise[2:]
            self.object_pos = np.clip(self.object_pos + self.object_vel, 0.0, 1.0)
        else:
            self.object_vel = 0.80 * self.object_vel + noise[2:] * 0.6
            self.object_pos = np.clip(self.object_pos + self.object_vel, 0.0, 1.0)
            if self.loss_pending and "contact_off_failure" not in events:
                if not self.ever_recovery_started:
                    self.ever_recovery_started = True
                    self.phase = "recovery_approach"
                    events.append("recovery_start")
                    self.recovery_start_pending = False
                    recovery_started_now = True
                self.recovery_wait += 1
        if (self.ever_recovery_started and not recovery_started_now and not self.contact
                and not self.grazing_recovery_blocked and self.agent_object_distance < 0.108
                and self.recovery_wait >= 2):
            self.contact = True
            self.loss_pending = False
            self.phase = "contact_reestablished"
            events.append("contact_reestablished")
        self.path_length += float(np.linalg.norm(self.object_pos - previous_object))
        self.collision = bool(self.object_clearance <= 0.0)
        if self.collision:
            if self.family.scenario == "terminal_collision":
                self.done = True
                self.terminal_reason = "terminal_collision"
                self.phase = "terminal_failure"
                events.append("terminal_failure")
            else:
                # A nonterminal collision is deflected; it is available for detour/recovery evidence.
                normal = self._unit(self.object_pos - self.obstacle)
                self.object_vel = self.object_vel - 1.4 * np.dot(self.object_vel, normal) * normal
                if not self.ever_detour:
                    self.ever_detour = True
                    self.phase = "detour"
                    events.append("detour_start")
        motion = float(np.linalg.norm(self.object_vel))
        if self.transport_pending and "contact_on" not in events:
            self.transport_pending = False
            self.ever_transport = True
            self.phase = "transport"
            events.append("transport_on")
        elif self.contact and motion > 0.018 and not self.ever_transport:
            # Keep contact_on and transport_on on distinct transitions.
            if "contact_on" in events:
                self.transport_pending = True
            else:
                self.ever_transport = True
                self.phase = "transport"
                events.append("transport_on")
        self.object_in_goal = bool(self.goal_distance <= self.goal_radius)
        if self.object_in_goal and not goal_before:
            self.phase = "goal_entered"
            events.append("goal_enter")
        self.stable_goal_steps = self.stable_goal_steps + 1 if self.object_in_goal and self.contact else 0
        if self.stable_goal_steps >= 3:
            self.done = True
            self.success = True
            self.phase = "stable_success"
            events.append("stable_success")
        self.goal_distance_history.append(self.goal_distance)
        if len(self.goal_distance_history) >= 7 and not self.ever_stagnation:
            progress = self.goal_distance_history[-7] - self.goal_distance_history[-1]
            if progress < 0.003 and motion < 0.007:
                self.ever_stagnation = True
                self.phase = "stagnation"
                events.append("stagnation_onset")
        self.step_index += 1
        if self.step_index >= self.horizon and not self.done:
            self.done = True
            self.terminal_reason = "horizon"
            self.phase = "terminal_failure"
            events.append("terminal_failure")
        chosen = self._choose_event(events)
        boundary = chosen != "none"
        self.prev_action = action.copy()
        info = {"events": events, "gold_event_name": chosen, "gold_mode": self.phase,
                "contact_before": contact_before, "contact_after": self.contact,
                "goal_distance": self.goal_distance, "object_clearance": self.object_clearance,
                "success": self.success, "terminal_reason": self.terminal_reason}
        self.event_log.append({"t": self.step_index - 1, "event": chosen, "all_events": events})
        return self.observable(), EVENT_IDS[chosen], boundary, info


def make_family_specs(count: int, seed: int) -> list[FamilySpec]:
    """Create equal scenario support and varied hidden parameters per root family."""
    if count <= 0:
        raise ValueError("root family count must be positive")
    rng = np.random.default_rng(seed)
    specs: list[FamilySpec] = []
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        specs.append(FamilySpec(
            root_family_id=f"u2_family_{index:03d}", scenario=scenario,
            side=1 if (index // len(SCENARIOS)) % 2 == 0 else -1,
            obstacle_margin=float(rng.uniform(0.012, 0.035)),
            process_noise=float(rng.uniform(0.0015, 0.0050)),
            slip_step=int(rng.integers(3, 6)), stagnation_steps=int(rng.integers(6, 11)),
        ))
    return specs
