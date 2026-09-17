"""Gymnasium wrapper over frozen SkillEnv with method-independent masks."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from p2cq_research.environment import SkillEnv
from p2cq_research.observations import LAYOUT_DIM, encode_observation
from p2cq_research.potentials import evaluate_all

from .mask_contract import legal_mask_bool, mask_sha256
from .potentials_v2 import METHOD_V2, evaluate_all_with_v2, shaped_training_reward
from .remaining_work_model import RemainingWorkPlanner


class MaskableSkillGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, contracts, method, *, gamma=0.99, beta=1.0):
        super().__init__()
        if not contracts:
            raise ValueError("contracts")
        self.contracts = list(contracts)
        self.method = method
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.action_space = spaces.Discrete(37)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(LAYOUT_DIM,), dtype=np.float32)
        self.env = SkillEnv(self.contracts[0])
        self._i = 0
        self._planner = None
        self._phi = 0.0
        self.last_mask_sha256 = None
        self.invalid_selected = 0
        self.nonfinite = 0

    def action_masks(self):
        # Deliberately ignores self.method.
        mask = legal_mask_bool(self.env)
        self.last_mask_sha256 = mask_sha256(mask)
        return mask

    def _phi_of(self, dyn, terminated=False, success=False):
        if terminated:
            return 0.0
        if self.method == "TASK_ONLY_ZERO_V1":
            return 0.0
        if self.method == METHOD_V2:
            if self._planner is None or self._planner.contract is not self.env.contract:
                self._planner = RemainingWorkPlanner(self.env.contract)
            return self._planner.potential(dyn)[0]
        table = evaluate_all(self.env.contract, dyn)
        return float(table[self.method])

    def _obs(self):
        return np.asarray(encode_observation(self.env.contract, self.env), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = 0
        if options and "contract_index" in options:
            idx = int(options["contract_index"])
        elif seed is not None:
            idx = int(seed) % len(self.contracts)
        else:
            idx = self._i % len(self.contracts)
            self._i += 1
        contract = self.contracts[idx % len(self.contracts)]
        self.env.reset(contract)
        self._planner = RemainingWorkPlanner(self.env.contract) if self.method == METHOD_V2 else None
        self._phi = self._phi_of(self.env.dynamic_public())
        obs = self._obs()
        if not np.isfinite(obs).all():
            self.nonfinite += 1
        return obs, {"method": self.method, "mask_sha256": mask_sha256(self.action_masks())}

    def step(self, action):
        action = int(action)
        mask = self.action_masks()
        legal = bool(mask[action]) if 0 <= action < 37 else False
        if not legal:
            self.invalid_selected += 1
        phi_before = self._phi
        _obs, task_r, terminated, truncated, info = self.env.step(action)
        dyn = self.env.dynamic_public()
        phi_after_raw = self._phi_of(dyn, terminated=False)
        train_r, bonus = shaped_training_reward(
            task_r, phi_before, phi_after_raw, gamma=self.gamma, beta=self.beta, terminated=bool(terminated)
        )
        self._phi = 0.0 if terminated else phi_after_raw
        obs = self._obs()
        if not (np.isfinite(obs).all() and np.isfinite(train_r)):
            self.nonfinite += 1
        info = dict(info)
        info.update({
            "task_reward": float(task_r),
            "training_reward": float(train_r),
            "shaping": float(bonus),
            "selected_action": action,
            "selected_action_was_legal": legal,
            "available_action_count": int(mask.sum()),
            "mask_sha256": self.last_mask_sha256,
            "method": self.method,
        })
        return obs, float(train_r), bool(terminated), bool(truncated), info