"""Mask-aware evaluation. Success is task goal, not shaped return."""
from __future__ import annotations
import hashlib
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import encode_observation
from p2cq_research.task_contract import canonical
from p2crm.mask_contract import legal_mask_bool
import numpy as np

def evaluate_episode(model, contract, *, deterministic=True, sampling_seed=None):
    env = SkillEnv(contract)
    env.reset()
    invalid = 0
    nonfinite = 0
    steps = 0
    if sampling_seed is not None:
        import torch, random
        random.seed(int(sampling_seed))
        np.random.seed(int(sampling_seed))
        torch.manual_seed(int(sampling_seed))
    while not env.terminated():
        obs = np.asarray(encode_observation(contract, env), dtype=np.float32)
        if not np.isfinite(obs).all():
            nonfinite += 1
        mask = legal_mask_bool(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        action = int(action)
        if not bool(mask[action]):
            invalid += 1
            raise RuntimeError("illegal action selected")
        _o, _r, term, trunc, info = env.step(action)
        steps += 1
        if trunc:
            raise RuntimeError("unexpected truncation")
    return {
        "success": 1 if env.success() else 0,
        "steps": steps,
        "horizon": contract.horizon,
        "invalid_actions": invalid,
        "nonfinite": nonfinite,
        "terminated": 1,
        "truncated": 0,
        "motif": contract.motif,
        "family_id": contract.family_id,
        "contract_sha256": hashlib.sha256(canonical(contract.public_dict()).encode("utf-8")).hexdigest(),
    }
