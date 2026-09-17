"""Legal-skill masks. No reward, oracle, pair-side, or future fields."""
from __future__ import annotations

import hashlib

import numpy as np

from p2cq_research.environment import SkillEnv
from p2cq_research.task_contract import ACTION_NAMES


def legal_mask_bool(env: SkillEnv, st=None):
    mask = env.legal_mask(st)
    if len(mask) != 37:
        raise RuntimeError("mask width")
    if not mask[0]:
        raise RuntimeError("WAIT must remain legal")
    if not any(mask):
        raise RuntimeError("all-false mask")
    return np.asarray(mask, dtype=np.bool_)


def mask_sha256(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def assert_method_independent(env, methods):
    """Mask must ignore reward method identity."""
    m0 = legal_mask_bool(env)
    for _ in methods:
        m = legal_mask_bool(env)
        if not np.array_equal(m0, m):
            raise RuntimeError("method-dependent mask")
    return m0


ACTION_NAMES  # re-export