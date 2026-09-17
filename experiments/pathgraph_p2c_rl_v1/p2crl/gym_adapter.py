"""Thin Gymnasium adapter over frozen SkillEnv. Mask ignores method."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import LAYOUT_DIM, encode_observation
from p2cq_research.potentials import evaluate_all
from p2crm.mask_contract import legal_mask_bool, mask_sha256
from p2crm.potentials_v2 import METHOD_V2, shaped_training_reward
from p2crm.remaining_work_model import RemainingWorkPlanner
from .sampler import BlockSampler, sampler_seed

class RLTaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, contracts, method, sampler_seed_value, rank, *, gamma=0.99, beta=1.0):
        super().__init__()
        if not contracts:
            raise ValueError("contracts")
        self.contracts = list(contracts)
        self.method = method
        self.rank = int(rank)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.action_space = spaces.Discrete(37)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(LAYOUT_DIM,), dtype=np.float32)
        self.sampler = BlockSampler(self.contracts, int(sampler_seed_value))
        self.env = SkillEnv(self.contracts[0])
        self._planners = {}
        self._phi = 0.0
        self.last_mask_sha256 = None
        self.invalid_selected = 0
        self.nonfinite = 0
        self.episode_count = 0
        self.contract_index = 0

    def action_masks(self):
        mask = legal_mask_bool(self.env)
        self.last_mask_sha256 = mask_sha256(mask)
        return mask

    def _planner(self):
        key = self.env.contract.task_hash()
        planner = self._planners.get(key)
        if planner is None:
            planner = RemainingWorkPlanner(self.env.contract)
            self._planners[key] = planner
        return planner

    def _phi_of(self, dyn, terminated=False):
        if terminated:
            return 0.0
        if self.method == "TASK_ONLY_ZERO_V1":
            return 0.0
        if self.method == METHOD_V2:
            return float(self._planner().potential(dyn)[0])
        table = evaluate_all(self.env.contract, dyn)
        return float(table[self.method])

    def _obs(self):
        return np.asarray(encode_observation(self.env.contract, self.env), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "contract_index" in options:
            idx = int(options["contract_index"])
        else:
            idx = self.sampler.next_index()
        self.contract_index = idx
        contract = self.contracts[idx]
        self.env.reset(contract)
        self.episode_count += 1
        self._phi = self._phi_of(self.env.dynamic_public())
        obs = self._obs()
        if not np.isfinite(obs).all():
            self.nonfinite += 1
        return obs, {"method": self.method, "contract_index": idx, "mask_sha256": mask_sha256(self.action_masks())}

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
            "raw_phi_after": float(phi_after_raw),
            "effective_phi_after": 0.0 if terminated else float(phi_after_raw),
            "selected_action": action,
            "selected_action_was_legal": legal,
            "available_action_count": int(mask.sum()),
            "mask_sha256": self.last_mask_sha256,
            "method": self.method,
            "terminated_task": bool(terminated),
            "truncated_task": bool(truncated),
        })
        return obs, float(train_r), bool(terminated), bool(truncated), info


def make_masked_env(contracts, method, run_seed, rank, gamma=0.99, beta=1.0):
    from sb3_contrib.common.wrappers import ActionMasker
    env = RLTaskEnv(contracts, method, sampler_seed(run_seed, rank), rank, gamma=gamma, beta=beta)
    return ActionMasker(env, lambda e: e.action_masks())
