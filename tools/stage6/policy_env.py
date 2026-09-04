"""Small deterministic graph transport environment used by the Stage 6 collector.

The controller is scripted only to obtain demonstrations.  Every action used for
supervision is the clipped action actually passed to :meth:`step`, never a label
reconstructed from a graph trace.  This environment intentionally exposes branch
order and recovery state in a 14-D low-dimensional observation compatible with the
frozen PathGraph model's input width.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    scenario: str
    order: str = ""
    seed: int = 0


class TransportGraphEnv:
    action_dim = 4
    obs_dim = 14
    horizon = 22

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.a = 0.0
        self.b = 0.0
        self.transport = 0.0
        self.grasped = True
        self.dropped = False
        self.reopened = False
        self.failed = False
        self.recovery_count = 0
        self.intervention_done = False
        self.done = False
        self.last_action = np.zeros(4, dtype=np.float32)
        return self.obs()

    def obs(self) -> np.ndarray:
        # These are exclusively numeric raw-state fields in the Stage-4 schema:
        # eef_pos(3), object_pos(3), target_pos(3), gripper_state(2), last action(3).
        # Scenario/order IDs and outcome metadata never enter the model features.
        if self.spec.task_id == "transport_dual_order":
            current = self.a if self.spec.order == "A_first" else self.b
            other = self.b if self.spec.order == "A_first" else self.a
            eef = (0.05 + .76 * current, .10 + .18 * other, .78)
            obj = (.08 + .70 * current, .10 + .18 * other, .65)
            target = (.80, .10 if self.spec.order == "A_first" else .35, .65)
        elif self.dropped:
            eef, obj, target = ((.30,.20,.90),(.80,.10,.55),(.80,.10,.65))
        elif self.reopened:
            eef, obj, target = ((.45,.11,.76),(.44,.10,.64),(.80,.10,.65))
        else:
            eef = (.02 + .79*self.transport, .05 + .06*self.transport, .97-.20*self.transport)
            obj = (.02 + .78*self.transport, .02 + .09*self.transport, .82-.17*self.transport)
            target = (.80,.10,.65)
        return np.asarray([
            *eef, *obj, *target, float(self.grasped)*.8, float(self.grasped)*.8,
            *self.last_action[:3],
        ], dtype=np.float32)

    def _intervene(self) -> None:
        if self.intervention_done or self.spec.task_id != "transport_recovery" or self.t != 7:
            return
        scenario = self.spec.scenario
        if scenario == "drop_regrasp":
            self.grasped = False; self.dropped = True; self.transport = max(0.0, self.transport - 0.18)
        elif scenario == "gripper_reopen":
            self.grasped = False; self.reopened = True; self.transport = max(0.0, self.transport - 0.10)
        elif scenario == "controlled_failure":
            self.grasped = False; self.dropped = True; self.failed = True; self.transport = max(0.0, self.transport - 0.35)
        self.intervention_done = True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("step called after terminal state")
        applied = np.clip(np.asarray(action, dtype=np.float32).reshape(self.action_dim), -1.0, 1.0)
        if self.spec.task_id == "transport_dual_order":
            self.a = float(np.clip(self.a + max(0.0, applied[0]) * 0.55, 0.0, 1.0))
            self.b = float(np.clip(self.b + max(0.0, applied[1]) * 0.55, 0.0, 1.0))
        else:
            # Reopening and regrasping are real action effects, not labels.
            if self.reopened and applied[3] > 0.35:
                self.reopened = False; self.grasped = True; self.recovery_count += 1
            if self.dropped and not self.failed and applied[2] > 0.35:
                self.dropped = False; self.grasped = True; self.recovery_count += 1
            if self.grasped and not self.failed:
                self.transport = float(np.clip(self.transport + max(0.0, applied[1]) * 0.35, 0.0, 1.0))
        self.last_action = applied.copy()
        self.t += 1
        self._intervene()
        success = (self.a >= 0.99 and self.b >= 0.99) if self.spec.task_id == "transport_dual_order" else (self.transport >= 0.99 and not self.failed)
        self.done = bool(success or self.t >= self.horizon or self.failed)
        reward = 1.0 if success else (-1.0 if self.failed else -0.01)
        info = {
            "action_applied": applied.copy(), "success": bool(success), "failed": bool(self.failed),
            "recovery_count": int(self.recovery_count), "t": int(self.t),
        }
        return self.obs(), reward, self.done, info


def scripted_controller(obs: np.ndarray, spec: EpisodeSpec, rng: np.random.Generator) -> np.ndarray:
    """Return an executable control; jitter is bounded before env.step receives it."""
    a = np.zeros(4, dtype=np.float32)
    if spec.task_id == "transport_dual_order":
        current = (float(obs[0])-.05)/.76
        other = (float(obs[1])-.10)/.18
        if current < .99:
            a[0 if spec.order == "A_first" else 1] = .24
        elif other < .99:
            a[1 if spec.order == "A_first" else 0] = .24
        else: a[:2] = .08
    elif spec.scenario == "controlled_failure" and abs(float(obs[0])-.30)<.03:
        a[:] = 0.0
    elif abs(float(obs[0])-.30)<.03:
        a[2] = 0.85; a[1] = 0.08
    elif abs(float(obs[0])-.45)<.03:
        a[3] = 0.85; a[1] = 0.08
    else:
        a[1] = 0.25; a[2] = 0.65
    a += rng.normal(0.0, 0.008, size=4).astype(np.float32)
    return np.clip(a, -1.0, 1.0)


def rollout_demo(spec: EpisodeSpec) -> Dict[str, object]:
    env = TransportGraphEnv(spec)
    rng = np.random.default_rng(spec.seed + 991)
    obs = env.reset()
    observations, actions, passed_actions, rewards = [], [], [], []
    while not env.done:
        command = scripted_controller(obs, spec, rng)
        next_obs, reward, _, info = env.step(command)
        observations.append(obs.copy()); actions.append(np.asarray(info["action_applied"], dtype=np.float32))
        passed_actions.append(command.copy()); rewards.append(float(reward))
        obs = next_obs
    success = bool((env.a >= .99 and env.b >= .99) if spec.task_id == "transport_dual_order" else (env.transport >= .99 and not env.failed))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "action_applied": np.asarray(actions, dtype=np.float32),
        "action_passed": np.asarray(passed_actions, dtype=np.float32),
        "env_rewards": np.asarray(rewards, dtype=np.float32),
        "success": success, "failed": bool(env.failed), "recovery_count": int(env.recovery_count),
        "path_signature": "A>B" if spec.order == "A_first" else ("B>A" if spec.order == "B_first" else "transport_recovery"),
    }
