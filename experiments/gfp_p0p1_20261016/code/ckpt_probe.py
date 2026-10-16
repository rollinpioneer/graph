"""Probe saved checkpoints on a fixed sample of dataset states (no env, no training).

For every NAME:STEP it computes, on the same N dataset states s with the same
noise z (z is shared by the actor and the flow, as in the official distillation):
  a_pi   = clip(actor(s, z))            one-step actor action
  a_flow = flow(s, z) (10 Euler steps)  VaBC flow action
  Qt_pi, Qt_flow, Qt_data               target critic, ensemble mean (critic_eval)
  Qo_pi, Qo_data, Qo_data_sd            online critic, mean / ensemble sd
  g      = sigmoid((lam/eta) * (Qt_data - min(Qt_pi, Qt_flow))), eta = 1e-3,
           lam = 1 / mean|Qo_pi|  (official formula; for F runs it is hypothetical)
and the change of a_pi / a_flow / Qt_pi relative to the previous probed step of
the same run when that step is exactly 100k earlier.

usage: python ckpt_probe.py OUT_DIR N_STATES NAME:STEP [NAME:STEP ...]
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import ogbench

from common import load_agent, xy_to_ij, ACT_DIM

ETA = 1e-3
CHUNK = 5000


@jax.jit
def probe(agent, obs, act, z):
    a_pi = jnp.clip(agent.network.select("actor")(obs, z), -1, 1)
    a_flow = agent.compute_flow_actions(obs, z)
    qo_pi = agent.network.select("critic")(obs, actions=a_pi)
    qo_data = agent.network.select("critic")(obs, actions=act)
    return dict(
        a_pi=a_pi, a_flow=a_flow,
        Qt_pi=agent.critic_eval(obs, a_pi),
        Qt_flow=agent.critic_eval(obs, a_flow),
        Qt_data=agent.critic_eval(obs, act),
        Qo_pi=qo_pi.mean(0), Qo_data=qo_data.mean(0), Qo_data_sd=qo_data.std(0),
    )


def main():
    out_dir, n = sys.argv[1], int(sys.argv[2])
    specs = [(a, int(b)) for a, b in (s.split(":") for s in sys.argv[3:])]
    os.makedirs(out_dir, exist_ok=True)

    meta_path = f"{out_dir}/states.npz"
    if not os.path.exists(meta_path):
        data = {}
        for t in (1, 4):
            _, tr, _ = ogbench.make_env_and_datasets(
                f"humanoidmaze-medium-navigate-singletask-task{t}-v0")
            data[t] = tr
        N = len(data[1]["observations"])
        idx = np.sort(np.random.default_rng(20261016).choice(N, n, replace=False))
        obs = data[1]["observations"][idx].astype(np.float32)
        assert np.array_equal(obs, data[4]["observations"][idx]), "task datasets differ"
        act = np.clip(data[1]["actions"][idx], -1 + 1e-5, 1 - 1e-5).astype(np.float32)
        z = np.random.default_rng(7).standard_normal((n, ACT_DIM)).astype(np.float32)
        i, j = xy_to_ij(obs[:, :2])
        np.savez_compressed(
            meta_path, idx=idx, obs=obs, act=act, z=z, cell_i=i, cell_j=j,
            next_xy=data[1]["next_observations"][idx][:, :2],
            rew1=data[1]["rewards"][idx], mask1=data[1]["masks"][idx],
            rew4=data[4]["rewards"][idx], mask4=data[4]["masks"][idx],
            n_dataset=N)
        del data
    m = np.load(meta_path)
    obs, act, z = m["obs"], m["act"], m["z"]

    prev = {}
    for name, step in specs:
        f = f"{out_dir}/{name}_{step}.npz"
        t0 = time.time()
        agent, cfg, path = load_agent(name, step)
        parts = [jax.device_get(probe(agent, obs[k:k + CHUNK], act[k:k + CHUNK], z[k:k + CHUNK]))
                 for k in range(0, len(obs), CHUNK)]
        r = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        lam = 1.0 / np.abs(r["Qo_pi"]).mean()
        ref = np.minimum(r["Qt_pi"], r["Qt_flow"])
        x = (lam / ETA) * (r["Qt_data"] - ref)
        g = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
        save = {k: r[k].astype(np.float32) for k in
                ("Qt_pi", "Qt_flow", "Qt_data", "Qo_pi", "Qo_data", "Qo_data_sd")}
        save.update(g=g.astype(np.float32), lam=lam, params=path,
                    pi_flow_gap=np.linalg.norm(r["a_pi"] - r["a_flow"], axis=1).astype(np.float32),
                    pi_data_gap=np.linalg.norm(r["a_pi"] - act, axis=1).astype(np.float32))
        p = prev.get(name)
        if p is not None and p["step"] == step - 100000:
            save["d_a_pi"] = np.linalg.norm(r["a_pi"] - p["a_pi"], axis=1).astype(np.float32)
            save["d_a_flow"] = np.linalg.norm(r["a_flow"] - p["a_flow"], axis=1).astype(np.float32)
            save["d_Qt_pi"] = (r["Qt_pi"] - p["Qt_pi"]).astype(np.float32)
        prev[name] = dict(step=step, a_pi=r["a_pi"], a_flow=r["a_flow"], Qt_pi=r["Qt_pi"])
        np.savez_compressed(f, **save)
        print(f"{name}:{step} lam={lam:.4g} Qt_pi={r['Qt_pi'].mean():.1f} g={g.mean():.3f} "
              f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
