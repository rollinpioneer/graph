"""Gated MaskablePPO runner. Prepare/dry-run never call learn()."""
from __future__ import annotations
import importlib.metadata
import os
import random
from pathlib import Path
import numpy as np
from .checkpointing import save_and_hash, state_dict_digest
from .gym_adapter import make_masked_env
from .io_utils import write_new
from .sampler import run_seed

EXPECTED = {"stable-baselines3": "2.7.1", "sb3-contrib": "2.7.1", "gymnasium": "1.2.2"}


def assert_versions():
    observed = {}
    for name, exp in EXPECTED.items():
        got = importlib.metadata.version(name)
        observed[name] = got
        if got != exp:
            raise RuntimeError(f"{name}: expected {exp}, got {got}")
    return observed


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))


def make_model(contracts, method, seed, n_envs, n_steps, batch_size, n_epochs, device="cpu"):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    factories = [
        (lambda rank=rank: make_masked_env(contracts, method, seed, rank))
        for rank in range(n_envs)
    ]
    env = DummyVecEnv(factories)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        seed=int(seed),
        device=device,
        verbose=0,
        policy_kwargs={
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
            "optimizer_kwargs": {"eps": 1e-5},
        },
    )
    return model, env


def assert_release_active(release):
    if not release or release.get("schema") not in ("P2CRL_CAMPAIGN_RELEASE_V1", "P2CRL_CAMPAIGN_RELEASE_V2"):
        raise RuntimeError("release schema")
    if release.get("status") != "ACTIVE" or release.get("training_release") is not True:
        raise RuntimeError("campaign not released")


def dry_run(contracts, methods, out_dir, seed=2026091801, env_step_budget=5120):
    """Construct models and take a few env steps. Zero gradient updates."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    versions = assert_versions()
    import torch
    seed_all(seed)
    used_steps = 0
    inits = {}
    for method in methods:
        model, venv = make_model(contracts[:16] if len(contracts) >= 16 else contracts, method, seed, n_envs=2, n_steps=32, batch_size=64, n_epochs=2)
        digest = state_dict_digest(model)
        inits[method] = digest
        inner = venv.envs[0]
        while hasattr(inner, "env") and not hasattr(inner, "action_masks"):
            inner = inner.env
        obs, _ = inner.reset()
        mask = inner.action_masks()
        for _ in range(4):
            legal = [i for i, bit in enumerate(mask) if bit]
            a = int(legal[0])
            obs, r, term, trunc, info = inner.step(a)
            used_steps += 1
            mask = inner.action_masks()
            if term:
                obs, _ = inner.reset()
                mask = inner.action_masks()
        if int(model.num_timesteps) != 0:
            raise RuntimeError("dry-run mutated timesteps")
        # fake milestone schedule without learn
        for target in [0, 32768, 65536, 131072, 262144, 524288]:
            if int(model.num_timesteps) != 0:
                raise RuntimeError("fake schedule advanced")
            if target < 0:
                raise RuntimeError("milestone")
        venv.close()
        if used_steps > env_step_budget:
            raise RuntimeError("dry-run step budget")
    parity = len(set(inits.values())) == 1
    rec = {
        "schema": "P2CRL_DRY_RUN_V1",
        "gradient_updates": 0,
        "learn_called": False,
        "env_steps": used_steps,
        "versions": dict(versions, torch=torch.__version__, numpy=np.__version__),
        "initial_policy_sha256": inits,
        "initial_policy_parity": parity,
        "deterministic_algorithms": True,
    }
    write_new(out / "dry_run.json", rec)
    if not parity:
        raise RuntimeError("init parity failed")
    return rec


def execute_job(*args, **kwargs):
    from .train_job import execute_job as _execute_job
    return _execute_job(*args, **kwargs)
