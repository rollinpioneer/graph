"""Frozen re-evaluation of saved GFP checkpoints (no training, no parameter change).

Differences from the official evaluate_parallel, on purpose:
  * exactly n_rounds * n_envs episodes are counted (the official loop also counts
    episodes finished by envs that already met their quota, which over-weights
    short, i.e. successful, episodes);
  * the same eval seed is used for every checkpoint of a run (common random
    numbers), so checkpoint-to-checkpoint differences are less noisy;
  * xy is recorded every 20 steps for endpoint / route analysis.
The policy is the official one: agent.sample_actions (one-step actor, N(0,I) noise).

usage: python frozen_eval.py OUT_JSON EVAL_SEED N_ROUNDS NAME:STEP [NAME:STEP ...]
"""
import os
import sys
import json
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import gymnasium
import ogbench

from common import load_agent, run_dir, xy_to_ij

N_ENVS = 10
XY_EVERY = 20


def make_creator(env_name):
    def creator():
        e = ogbench.make_env_and_datasets(env_name, env_only=True)
        e.reset()
        return e
    return creator


def run_ckpt(envs, agent, eval_seed, n_rounds):
    key = jax.random.PRNGKey(eval_seed)
    eps = []
    for r in range(n_rounds):
        obs, _ = envs.reset(seed=int(eval_seed * 1000 + r * N_ENVS))
        active = np.ones(N_ENVS, bool)
        t = 0
        paths = [[obs[k, :2].tolist()] for k in range(N_ENVS)]
        while active.any():
            key, sk = jax.random.split(key)
            act = np.clip(np.asarray(agent.sample_actions(obs, seed=sk)), -1, 1)
            nobs, rew, term, trunc, info = envs.step(act)
            t += 1
            done = np.logical_or(term, trunc)
            for k in range(N_ENVS):
                if not active[k]:
                    continue
                if t % XY_EVERY == 0 or done[k]:
                    paths[k].append(nobs[k, :2].tolist())
                if done[k]:
                    i, j = xy_to_ij(nobs[k, :2])
                    eps.append(dict(
                        round=r, env=k, success=float(info["success"][k]),
                        length=t, terminated=bool(term[k]),
                        final_xy=nobs[k, :2].tolist(), final_ij=[int(i), int(j)],
                        path=[[round(a, 2), round(b, 2)] for a, b in paths[k]],
                    ))
                    active[k] = False
            if done.any() and active.any():
                nobs, _ = envs.reset(options={"reset_mask": done})
            obs = nobs
    return eps


def main():
    out, eval_seed, n_rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    specs = [s.split(":") for s in sys.argv[4:]]
    results = json.load(open(out)) if os.path.exists(out) else {}
    # Build all envs before JAX touches the GPU, in spawned workers (forking a
    # process that already holds a CUDA/JAX context breaks MuJoCo's EGL init).
    envs = {}
    for name, _ in specs:
        env_name = json.load(open(f"{run_dir(name)}/config.json"))["env_name"]
        if env_name not in envs:
            envs[env_name] = gymnasium.vector.AsyncVectorEnv(
                [make_creator(env_name)] * N_ENVS,
                autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED,
                context="spawn",
            )
    for name, step in specs:
        k = f"{name}:{step}"
        if k in results:
            continue
        t0 = time.time()
        agent, cfg, path = load_agent(name, int(step))
        env_name = cfg["env_name"]
        eps = run_ckpt(envs[env_name], agent, eval_seed, n_rounds)
        sr = 100 * np.mean([e["success"] for e in eps])
        results[k] = dict(name=name, step=int(step), env_name=env_name, params=path,
                          eval_seed=eval_seed, n_episodes=len(eps), success=sr,
                          seconds=time.time() - t0, episodes=eps)
        json.dump(results, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{k} n={len(eps)} success={sr:.1f} {time.time() - t0:.0f}s", flush=True)
    for e in envs.values():
        e.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
