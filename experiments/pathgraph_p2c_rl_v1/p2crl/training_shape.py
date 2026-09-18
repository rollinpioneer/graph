"""Immutable training shapes. Formal PPO hyperparameters stay frozen."""
from __future__ import annotations
from dataclasses import dataclass
from .errors import ProtocolViolation

@dataclass(frozen=True)
class TrainingShape:
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int

    @property
    def rollout_quantum(self) -> int:
        return int(self.n_envs) * int(self.n_steps)

    @property
    def minibatches_per_rollout(self) -> int:
        q = self.rollout_quantum
        if q % int(self.batch_size) != 0:
            raise ProtocolViolation("rollout_quantum not divisible by batch_size")
        return q // int(self.batch_size)

    def expected_rollouts(self, timesteps: int) -> int:
        q = self.rollout_quantum
        if int(timesteps) % q != 0:
            raise ProtocolViolation("timesteps not divisible by rollout_quantum")
        return int(timesteps) // q

    def expected_ppo_epochs(self, timesteps: int) -> int:
        return self.expected_rollouts(timesteps) * int(self.n_epochs)

    def expected_optimizer_steps(self, timesteps: int) -> int:
        return self.expected_rollouts(timesteps) * self.minibatches_per_rollout * int(self.n_epochs)

    def as_dict(self):
        return {
            "n_envs": int(self.n_envs),
            "n_steps": int(self.n_steps),
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "rollout_quantum": self.rollout_quantum,
        }

    def validate_target(self, target_steps: int) -> None:
        if int(target_steps) % self.rollout_quantum != 0:
            raise ProtocolViolation("target_steps % rollout_quantum != 0")
        if self.rollout_quantum % int(self.batch_size) != 0:
            raise ProtocolViolation("rollout_quantum % batch_size != 0")


SMOKE_SHAPE = TrainingShape(8, 64, 256, 10)
FORMAL_SHAPE = TrainingShape(8, 256, 256, 10)


def shape_for_kind(kind: str) -> TrainingShape:
    if kind == "smoke":
        return SMOKE_SHAPE
    if kind == "formal":
        return FORMAL_SHAPE
    raise ProtocolViolation(f"unknown job kind: {kind}")


def shape_for_job(job) -> TrainingShape:
    return shape_for_kind((job or {}).get("kind") or "formal")
