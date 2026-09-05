"""Small continuous stochastic simulator with a complete serializable state.

This is a D1 bridge backend, not a task-performance benchmark.  Its purpose is
to establish the missing infrastructure invariant: an anchor contains every
dynamic quantity and the random-generator state necessary to restore and
continue an actual state-changing rollout exactly.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np


class StochasticPushSimulator:
    """A bounded 2-D agent/object simulator with contact impulses and process noise."""

    snapshot_schema = "stochastic_push_snapshot_v1"

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    @staticmethod
    def _bound(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for index in range(2):
            if position[index] < 0.0:
                position[index] = -position[index]
                velocity[index] *= -0.55
            elif position[index] > 1.0:
                position[index] = 2.0 - position[index]
                velocity[index] *= -0.55
        return position, velocity

    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.agent_position = np.asarray([0.20, 0.20], dtype=np.float64)
        self.agent_velocity = np.zeros(2, dtype=np.float64)
        self.object_position = np.asarray([0.36, 0.20], dtype=np.float64)
        self.object_velocity = np.zeros(2, dtype=np.float64)
        self.contact_count = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate((self.agent_position, self.agent_velocity,
                               self.object_position, self.object_velocity)).copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        control = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        process_noise = self.rng.normal(0.0, 0.004, size=4)
        self.agent_velocity = 0.78 * self.agent_velocity + 0.080 * control + process_noise[:2]
        self.agent_position = self.agent_position + self.agent_velocity
        self.agent_position, self.agent_velocity = self._bound(self.agent_position, self.agent_velocity)
        offset = self.object_position - self.agent_position
        distance = float(np.linalg.norm(offset))
        contact = distance < 0.16
        if contact:
            normal = offset / max(distance, 1e-12)
            approach = max(float(np.dot(self.agent_velocity - self.object_velocity, normal)), 0.0)
            self.object_velocity = self.object_velocity + 0.60 * approach * normal
            self.contact_count += 1
        self.object_velocity = 0.91 * self.object_velocity + process_noise[2:]
        self.object_position = self.object_position + self.object_velocity
        self.object_position, self.object_velocity = self._bound(self.object_position, self.object_velocity)
        self.step_index += 1
        return self.observation(), {"contact": contact, "process_noise": process_noise.tolist()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.snapshot_schema,
            "seed": self.seed,
            "step_index": self.step_index,
            "agent_position": self.agent_position.tolist(),
            "agent_velocity": self.agent_velocity.tolist(),
            "object_position": self.object_position.tolist(),
            "object_velocity": self.object_velocity.tolist(),
            "contact_count": self.contact_count,
            # JSON round-trip prevents aliases into NumPy's mutable generator state.
            "rng_state": json.loads(json.dumps(copy.deepcopy(self.rng.bit_generator.state))),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != self.snapshot_schema:
            raise ValueError("unsupported stochastic simulator snapshot")
        self.seed = int(snapshot["seed"])
        self.step_index = int(snapshot["step_index"])
        self.agent_position = np.asarray(snapshot["agent_position"], dtype=np.float64)
        self.agent_velocity = np.asarray(snapshot["agent_velocity"], dtype=np.float64)
        self.object_position = np.asarray(snapshot["object_position"], dtype=np.float64)
        self.object_velocity = np.asarray(snapshot["object_velocity"], dtype=np.float64)
        self.contact_count = int(snapshot["contact_count"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = snapshot["rng_state"]

    def state_vector(self) -> np.ndarray:
        return np.concatenate((self.agent_position, self.agent_velocity,
                               self.object_position, self.object_velocity)).copy()
