"""c1: at the same task4 start state, does the critic prefer wall-ward (west) or detour (east) candidates?

usage: python c1_rank.py OUT_JSON N_START K H NAME:STEP [NAME:STEP ...]
Candidates per start state (z fixed, shared by all models):
  actor : a = clip(actor(s0, z_k)); label = dx after H closed-loop steps of actor(., z_k)
  flow  : a = flow(s0, z_k);        label = dx after H closed-loop steps of flow(., z_k)
  data  : first of H dataset actions from the K nearest dataset states (obs without xy);
          label = dx after replaying those H actions open-loop from s0 (model independent)
Score: official critic_eval (target critic, ensemble mean) at s0.
"""
import os
import sys
import json
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from envpool import EnvPool

ENV = "humanoidmaze-medium-navigate-singletask-task4-v0"
W = 32
THR = 0.5


def auc(qw, qe):
    if len(qw) == 0 or len(qe) == 0:
        return None
    d = np.asarray(qw)[:, None] - np.asarray(qe)[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def summarize(q, dx):
    q, dx = np.concatenate(q), np.concatenate(dx)
    w, e = dx <= -THR, dx >= THR
    return dict(n_west=int(w.sum()), n_east=int(e.sum()), n_still=int((~w & ~e).sum()),
                auc_west_gt_east=auc(q[w], q[e]),
                q_west=float(q[w].mean()) if w.any() else None,
                q_east=float(q[e].mean()) if e.any() else None,
                dx_mean=float(dx.mean()), dx_quartiles=[float(x) for x in np.quantile(dx, [0.25, 0.5, 0.75])])


def main():
    out, n_start, K, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    specs = [(a, int(b)) for a, b in (s.split(":") for s in sys.argv[5:])]
    pool = EnvPool(ENV, W)
    starts = [pool.reset(i) for i in range(n_start)]

    import jax
    import jax.numpy as jnp
    import ogbench
    from common import load_agent, ACT_DIM

    @jax.jit
    def act_actor(agent, obs, z):
        return jnp.clip(agent.network.select("actor")(obs, z), -1, 1)

    @jax.jit
    def act_flow(agent, obs, z):
        return agent.compute_flow_actions(obs, z)

    @jax.jit
    def qscore(agent, obs, a):
        return agent.critic_eval(obs, a)

    def rollout(states, step_fn):
        """states: list of (qpos, qvel); step_fn(obs_batch, idx_batch) -> actions. Returns dx, dy."""
        dx, dy = np.zeros(len(states)), np.zeros(len(states))
        for b in range(0, len(states), W):
            idx = np.arange(b, min(b + W, len(states)))
            obs = pool.set([states[i] for i in idx])
            x0 = obs[:, :2].copy()
            for t in range(H):
                obs = pool.step(step_fn(obs, idx, t))
            dx[idx], dy[idx] = obs[:, 0] - x0[:, 0], obs[:, 1] - x0[:, 1]
        return dx, dy

    res = dict(env=ENV, n_start=n_start, K=K, H=H, thr=THR,
               starts=[dict(xy=s[0][:2].tolist()) for s in starts], models={})
    t0 = time.time()

    # --- data-transplant candidates (model independent) ---
    _, ds, _ = ogbench.make_env_and_datasets(ENV)
    obs_all = ds["observations"].astype(np.float32)
    act_all = np.clip(ds["actions"], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
    N = len(obs_all)
    valid = (np.arange(N) % 2000) <= 2000 - H - 1  # fixed 2000-step episodes (checked in coverage.json)
    feat = obs_all[:, 2:]
    mu, sd = feat.mean(0), feat.std(0) + 1e-6
    data_first, data_seq, data_src = [], [], []
    for (ob, qp, qv) in starts:
        f = (ob[2:] - mu) / sd
        d = np.zeros(N, np.float32)
        for c in range(0, N, 200000):
            d[c:c + 200000] = (((feat[c:c + 200000] - mu) / sd - f) ** 2).sum(1)
        d[~valid] = np.inf
        nn = np.argsort(d)[:K]
        data_src.append(nn)
        data_seq.append(np.stack([act_all[i:i + H] for i in nn]))  # K,H,A
        data_first.append(act_all[nn])
    del feat
    states = [(starts[i][1], starts[i][2]) for i in range(n_start) for _ in range(K)]
    seqs = np.concatenate(data_seq)  # n_start*K, H, A
    ddx, ddy = rollout(states, lambda obs, idx, t: seqs[idx, t])
    res["data_labels"] = dict(dx=ddx.tolist(), dy=ddy.tolist(),
                              src_idx=np.concatenate(data_src).tolist(),
                              src_dx=[float(obs_all[i + H, 0] - obs_all[i, 0]) for i in np.concatenate(data_src)],
                              src_xy=[obs_all[i, :2].tolist() for i in np.concatenate(data_src)])
    print(f"data candidates labelled {time.time() - t0:.0f}s; west={int((ddx <= -THR).sum())} east={int((ddx >= THR).sum())}", flush=True)
    del obs_all

    Z = np.random.default_rng(11).standard_normal((K, ACT_DIM)).astype(np.float32)
    zz = np.tile(Z, (n_start, 1))
    s0 = np.stack([s[0] for s in starts]).astype(np.float32)
    s0_rep = np.repeat(s0, K, axis=0)
    for name, step in specs:
        t1 = time.time()
        agent, cfg, path = load_agent(name, step)
        r = dict(params=path)
        q_data = np.asarray(qscore(agent, s0_rep, np.concatenate(data_first)))
        r["data"] = dict(q=q_data.tolist(), **summarize([q_data], [ddx]))
        for src, fn in (("actor", act_actor), ("flow", act_flow)):
            a0 = np.asarray(fn(agent, s0_rep, zz))
            q = np.asarray(qscore(agent, s0_rep, a0))
            dx, dy = rollout(states, lambda obs, idx, t: np.asarray(fn(agent, obs.astype(np.float32), zz[idx])))
            r[src] = dict(q=q.tolist(), dx=dx.tolist(), dy=dy.tolist(), **summarize([q], [dx]))
        res["models"][f"{name}:{step}"] = r
        json.dump(res, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(name, step, {k: (r[k]["n_west"], r[k]["n_east"], r[k]["auc_west_gt_east"]) for k in ("actor", "flow", "data")},
              f"{time.time() - t1:.0f}s", flush=True)
    pool.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
