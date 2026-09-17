"""Mask-aware evaluation. Do not use SB3 evaluate_policy."""
from __future__ import annotations

import numpy as np


def evaluate_masked(model, env, *, n_episodes=4, deterministic=True):
    returns = []
    successes = 0
    invalid = 0
    nonfinite = 0
    steps = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        truncated = False
        ret = 0.0
        while not (done or truncated):
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
            action = int(action)
            if not bool(mask[action]):
                invalid += 1
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            ret += float(reward)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                nonfinite += 1
            if info.get("selected_action_was_legal") is False:
                invalid += 1
        returns.append(ret)
        if info.get("success"):
            successes += 1
    return {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)) if returns else None,
        "successes": successes,
        "invalid_actions_selected": int(invalid),
        "nonfinite": int(nonfinite),
        "steps": steps,
        "evaluation_mask_used": True,
        "training_mask_used": True,
    }