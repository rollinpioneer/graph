"""c2: controlled replay separating candidate count, randomness and Q selection (no training).

usage: python c2_replay.py OUT_JSON EVAL_SEED N_ROUNDS NAME:STEP:ARM [...]
Arms (see ANALYSIS_CARD_C_P0REF.md):
  A0 actor K=1 random z (official)   A1 actor z=0           A4 actor K=16 medoid (no Q)
  A2 actor K=16 argmax Q             A3 actor K=16 argmin Q  A5 actor K=64 argmax Q
  F0 flow K=1                        F2 flow K=16 argmax Q   F3 flow K=16 argmin Q
Q = official critic_eval (target critic, ensemble mean). Episodes/seeds as in frozen_eval.py.
"""
import os
import sys
import json
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium

from frozen_eval import make_creator, N_ENVS, XY_EVERY
from common import run_dir, xy_to_ij, bfs_dist, TASKS

ARMS = {
    "A0": ("actor", 1, "single"), "A1": ("actor", 1, "zero"), "A4": ("actor", 16, "medoid"),
    "A2": ("actor", 16, "argmax"), "A3": ("actor", 16, "argmin"), "A5": ("actor", 64, "argmax"),
    "F0": ("flow", 1, "single"), "F2": ("flow", 16, "argmax"), "F3": ("flow", 16, "argmin"),
}


def make_policy(source, K, mode):
    import jax
    import jax.numpy as jnp
    from common import ACT_DIM

    def pol(agent, obs, key):
        B, D = obs.shape
        if mode == "zero":
            z = jnp.zeros((B, 1, ACT_DIM))
        else:
            z = jax.random.normal(key, (B, K, ACT_DIM))
        k = z.shape[1]
        o = jnp.repeat(obs[:, None, :], k, axis=1).reshape(-1, D)
        zz = z.reshape(-1, ACT_DIM)
        if source == "actor":
            a = jnp.clip(agent.network.select("actor")(o, zz), -1, 1)
        else:
            a = agent.compute_flow_actions(o, zz)
        a3 = a.reshape(B, k, ACT_DIM)
        if k == 1:
            return a3[:, 0]
        if mode == "medoid":
            idx = jnp.argmin(((a3 - a3.mean(1, keepdims=True)) ** 2).sum(-1), axis=1)
        else:
            q = agent.critic_eval(o, a).reshape(B, k)
            idx = jnp.argmax(q, 1) if mode == "argmax" else jnp.argmin(q, 1)
        return a3[jnp.arange(B), idx]

    return jax.jit(pol)


def run(envs, agent, pol, eval_seed, n_rounds, task):
    import jax
    bfs = bfs_dist(TASKS[task]["goal_ij"])
    key = jax.random.PRNGKey(eval_seed)
    eps = []
    for r in range(n_rounds):
        obs, _ = envs.reset(seed=int(eval_seed * 1000 + r * N_ENVS))
        active = np.ones(N_ENVS, bool)
        t = 0
        paths = [[obs[k, :2].tolist()] for k in range(N_ENVS)]
        while active.any():
            key, sk = jax.random.split(key)
            act = np.clip(np.asarray(pol(agent, obs.astype(np.float32), sk)), -1, 1)
            nobs, rew, term, trunc, info = envs.step(act)
            t += 1
            done = np.logical_or(term, trunc)
            for k in range(N_ENVS):
                if not active[k]:
                    continue
                if t % XY_EVERY == 0 or done[k]:
                    paths[k].append(nobs[k, :2].tolist())
                if done[k]:
                    pi, pj = xy_to_ij(np.array(paths[k]))
                    eps.append(dict(success=float(info["success"][k]), length=t,
                                    final_xy=nobs[k, :2].tolist(),
                                    min_bfs=int(bfs[pi, pj].min()), start_bfs=int(bfs[pi[0], pj[0]])))
                    active[k] = False
            if done.any() and active.any():
                nobs, _ = envs.reset(options={"reset_mask": done})
            obs = nobs
    return eps


def main():
    out, eval_seed, n_rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    specs = [s.split(":") for s in sys.argv[4:]]
    results = json.load(open(out)) if os.path.exists(out) else {}
    envs = {}
    for name, _, _ in specs:
        env_name = json.load(open(f"{run_dir(name)}/config.json"))["env_name"]
        if env_name not in envs:
            envs[env_name] = gymnasium.vector.AsyncVectorEnv(
                [make_creator(env_name)] * N_ENVS,
                autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED, context="spawn")
    from common import load_agent
    pols, agents = {}, {}
    for name, step, arm in specs:
        k = f"{name}:{step}:{arm}"
        if k in results:
            continue
        t0 = time.time()
        if (name, step) not in agents:
            agents.clear()
            agents[(name, step)] = load_agent(name, int(step))
        agent, cfg, path = agents[(name, step)]
        if arm not in pols:
            pols[arm] = make_policy(*ARMS[arm])
        task = int(name.split("_")[1][4:])
        eps = run(envs[cfg["env_name"]], agent, pols[arm], eval_seed, n_rounds, task)
        sr = 100 * np.mean([e["success"] for e in eps])
        left = 100 * np.mean([e["min_bfs"] < e["start_bfs"] for e in eps])
        mb = float(np.mean([e["min_bfs"] for e in eps]))
        results[k] = dict(name=name, step=int(step), arm=arm, params=path, eval_seed=eval_seed,
                          n=len(eps), success=sr, left_start=left, min_bfs_mean=mb,
                          seconds=time.time() - t0, episodes=eps)
        json.dump(results, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{k} n={len(eps)} success={sr:.1f} left_start={left:.1f} min_bfs={mb:.2f} {time.time() - t0:.0f}s", flush=True)
    for e in envs.values():
        e.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
