"""Stateful stochastic simulator used for U2 boundary-data construction.

The generator may select a behavior stratum to obtain coverage, but that
stratum is deliberately absent from :meth:`features`.  Boundary labels are
derived from the resulting state transitions: proximity, loss/recovery of
contact, path inefficiency, and lack of goal progress.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np


class U2BoundarySimulator:
    """A continuous object transport simulator with stochastic contact loss."""

    snapshot_schema = "u2_boundary_snapshot_v1"
    goal = np.asarray([0.85, 0.50], dtype=np.float64)
    obstacle = np.asarray([0.50, 0.50], dtype=np.float64)
    obstacle_radius = 0.11
    goal_radius = 0.065
    horizon = 45

    def __init__(self, position: np.ndarray, seed: int):
        self.rng = np.random.default_rng(int(seed))
        self.position = np.asarray(position, dtype=np.float64).reshape(2).copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.contact = True
        self.contact_quality = 1.0
        self.recovery_progress = 0
        self.step_index = 0
        self.path_length = 0.0
        self.goal_distance_history = [self.goal_distance]
        self.done = False
        self.success = False
        self.failure_reason: str | None = None

    @property
    def goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    @property
    def clearance(self) -> float:
        return float(np.linalg.norm(self.position - self.obstacle) - self.obstacle_radius)

    def features(self, previous_action: np.ndarray) -> list[float]:
        """Observable inputs; behavior stratum and interventions are excluded."""
        action = np.asarray(previous_action, dtype=np.float64).reshape(3)
        return [*self.position.tolist(), *self.velocity.tolist(), *(self.goal - self.position).tolist(),
                float(self.contact), float(self.contact_quality), *action.tolist()]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.snapshot_schema, "position": self.position.tolist(), "velocity": self.velocity.tolist(),
            "contact": self.contact, "contact_quality": self.contact_quality,
            "recovery_progress": self.recovery_progress, "step_index": self.step_index,
            "path_length": self.path_length, "goal_distance_history": self.goal_distance_history,
            "done": self.done, "success": self.success, "failure_reason": self.failure_reason,
            "rng_state": json.loads(json.dumps(copy.deepcopy(self.rng.bit_generator.state))),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != self.snapshot_schema:
            raise ValueError("unsupported U2 snapshot schema")
        self.position = np.asarray(snapshot["position"], dtype=np.float64)
        self.velocity = np.asarray(snapshot["velocity"], dtype=np.float64)
        self.contact = bool(snapshot["contact"])
        self.contact_quality = float(snapshot["contact_quality"])
        self.recovery_progress = int(snapshot["recovery_progress"])
        self.step_index = int(snapshot["step_index"])
        self.path_length = float(snapshot["path_length"])
        self.goal_distance_history = [float(x) for x in snapshot["goal_distance_history"]]
        self.done = bool(snapshot["done"])
        self.success = bool(snapshot["success"])
        self.failure_reason = snapshot["failure_reason"]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = snapshot["rng_state"]

    def step(self, action: np.ndarray, *, force_contact_loss: bool = False) -> tuple[list[float], dict[str, Any]]:
        if self.done:
            raise RuntimeError("step called after terminal state")
        command = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        previous_position = self.position.copy()
        contact_before = self.contact
        loss_event = False
        recovery_event = False
        if self.contact and force_contact_loss:
            self.contact = False
            self.contact_quality = 0.0
            self.recovery_progress = 0
            loss_event = True
        if not self.contact and command[2] > 0.60:
            self.recovery_progress += 1
            if self.recovery_progress >= 2:
                self.contact = True
                self.contact_quality = 0.70
                recovery_event = True
        if self.contact:
            noise = self.rng.normal(0.0, 0.003, size=2)
            self.velocity = 0.68 * self.velocity + 0.060 * command[:2] + noise
            self.position = np.clip(self.position + self.velocity, 0.0, 1.0)
            grazing_hazard = self.clearance <= 0.018
            if grazing_hazard:
                self.contact_quality = max(0.0, self.contact_quality - 0.04)
                if self.rng.random() < 0.08:
                    self.contact = False
                    self.contact_quality = 0.0
                    self.recovery_progress = 0
                    loss_event = True
        else:
            self.velocity = 0.55 * self.velocity
            self.position = np.clip(self.position + self.velocity, 0.0, 1.0)
        self.path_length += float(np.linalg.norm(self.position - previous_position))
        self.step_index += 1
        self.goal_distance_history.append(self.goal_distance)
        grazing_event = bool(0.0 <= self.clearance <= 0.018)
        stagnation_event = False
        if len(self.goal_distance_history) >= 5:
            progress = self.goal_distance_history[-5] - self.goal_distance_history[-1]
            stagnation_event = bool(progress < 0.004)
        self.success = bool(self.contact and self.goal_distance <= self.goal_radius)
        if self.step_index >= self.horizon and not self.success:
            self.failure_reason = "horizon"
        self.done = bool(self.success or self.failure_reason is not None)
        events = {
            "grazing": grazing_event, "contact_loss": loss_event or (contact_before and not self.contact),
            "contact_recovery": recovery_event, "stagnation": stagnation_event,
        }
        return self.features(command), {"events": events, "success": self.success, "done": self.done,
                                         "goal_distance": self.goal_distance, "clearance": self.clearance,
                                         "contact": self.contact, "path_length": self.path_length}
