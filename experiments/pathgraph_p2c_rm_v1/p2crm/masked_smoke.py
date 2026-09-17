"""Four-job MaskablePPO smoke. Not a research comparison."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np

from .generator_v2 import iter_split
from .gym_bridge import MaskableSkillGym
from .io_utils import dump_replace, sha256_file
from .masked_evaluate import evaluate_masked
from . import METHOD_V2, SMOKE_METHODS

SMOKE_SEED = 2026091701
SMOKE_STEPS = 512


def state_dict_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def assert_versions():
    observed = {}
    expected = {
        "stable-baselines3": "2.7.1",
        "sb3-contrib": "2.7.1",
        "gymnasium": "1.2.2",
    }
    for name, exp in expected.items():
        got = importlib.metadata.version(name)
        observed[name] = got
        if got != exp:
            raise RuntimeError(f"{name}: expected {exp}, got {got}; isolate environment, do not silently upgrade")
    return observed


def smoke_contracts():
    out = []
    for spec in iter_split("smoke"):
        out.append(spec["left"])
        out.append(spec["right"])
    return out


def _make_env(contracts, method, rank=0):
    from sb3_contrib.common.wrappers import ActionMasker

    env = MaskableSkillGym(contracts, method, gamma=0.99, beta=1.0)
    return ActionMasker(env, lambda e: e.action_masks())


def run_smoke_job(method, out_dir, *, seed=SMOKE_SEED):
    versions = assert_versions()
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contracts = smoke_contracts()
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_envs = 2
    n_steps = 32
    factories = [
        (lambda rank=rank: _make_env(contracts, method, rank=rank))
        for rank in range(n_envs)
    ]
    env = DummyVecEnv(factories)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
            "optimizer_kwargs": {"eps": 1e-5},
        },
    )
    model.set_logger(configure(str(out / "ppo_logs"), ["csv"]))
    init_hash = state_dict_digest(model)
    dump_replace(out / "initialization.json", {
        "method": method,
        "policy_and_value_sha256": init_hash,
        "same_seed_must_match_across_methods": True,
        "versions": dict(versions, torch=torch.__version__, numpy=np.__version__),
    })
    model.learn(total_timesteps=SMOKE_STEPS, reset_num_timesteps=False, log_interval=1, progress_bar=False)
    if int(model.num_timesteps) != SMOKE_STEPS:
        raise RuntimeError(f"timesteps {model.num_timesteps}")
    ckpt = out / f"policy_{SMOKE_STEPS}.zip"
    model.save(str(ckpt))

    # unwrap action masker to accumulate counters
    invalid = 0
    nonfinite = 0
    for wrapped in env.envs:
        inner = wrapped
        while hasattr(inner, "env") and not hasattr(inner, "invalid_selected"):
            inner = inner.env
        invalid += int(getattr(inner, "invalid_selected", 0))
        nonfinite += int(getattr(inner, "nonfinite", 0))

    eval_env = _make_env(contracts, method)
    ev = evaluate_masked(model, eval_env, n_episodes=2, deterministic=True)
    invalid += int(ev["invalid_actions_selected"])
    nonfinite += int(ev["nonfinite"])
    inner = eval_env
    while hasattr(inner, "env") and not hasattr(inner, "invalid_selected"):
        inner = inner.env
    invalid += int(getattr(inner, "invalid_selected", 0))
    nonfinite += int(getattr(inner, "nonfinite", 0))
    eval_env.close()
    env.close()

    status = "SMOKE_PASS_NOT_RESEARCH_RESULT"
    if invalid or nonfinite:
        status = "MASKABLE_PPO_INTEGRATION_FAILED"
    rec = {
        "method": method,
        "seed": seed,
        "total_timesteps": SMOKE_STEPS,
        "status": status,
        "initial_policy_sha256": init_hash,
        "invalid_actions_selected": int(invalid),
        "nonfinite": int(nonfinite),
        "training_mask_used": True,
        "evaluation_mask_used": True,
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_path": str(ckpt),
        "eval": ev,
    }
    dump_replace(out / "job.json", rec)
    return rec


def run_all(out_root):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for method in SMOKE_METHODS:
        rec = run_smoke_job(method, out_root / method)
        jobs.append(rec)
    hashes = {j["initial_policy_sha256"] for j in jobs}
    parity = len(hashes) == 1 and None not in hashes
    summary = {
        "schema": "P2C_RM_MASKABLE_SMOKE_REGISTRY_V1",
        "formal_training_jobs": 0,
        "jobs": jobs,
        "initial_policy_parity": parity,
        "status": "SMOKE_PASS_NOT_RESEARCH_RESULT" if parity and all(j["status"] == "SMOKE_PASS_NOT_RESEARCH_RESULT" for j in jobs) else "MASKABLE_PPO_INTEGRATION_FAILED",
    }
    dump_replace(out_root / "smoke_registry.json", summary)
    dump_replace(out_root / "smoke_summary.json", {
        "jobs": len(jobs),
        "parity": parity,
        "status": summary["status"],
        "invalid": sum(j["invalid_actions_selected"] for j in jobs),
        "nonfinite": sum(j["nonfinite"] for j in jobs),
    })
    return summary