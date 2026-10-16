"""Start-region value scan over saved checkpoints (no env, no training).

usage: python island_scan.py PROBE_NPZ OUT_JSON NAME:STEP [NAME:STEP ...]
On the fixed probe states / noise of P0 (task1 start region (1,1),(2,1),(1,2),(2,2)), per checkpoint:
online and target V = Q(s, clip(actor(s,z))), ensemble mean, start region and all states;
lambda = 1/mean|Q_online|; start-region mean ||dQ_online/da||. Used only to pick the fork point.
"""
import os
import sys
import json

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from common import load_agent

START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]


def main():
    probe = np.load(sys.argv[1])
    out = sys.argv[2]
    obs, z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)

    @jax.jit
    def f(ag):
        a = jnp.clip(ag.network.select("actor")(obs, z), -1, 1)
        q_on = ag.network.select("critic")(obs, actions=a).mean(0)
        q_tg = ag.critic_eval(obs, a)
        g = jax.grad(lambda aa: ag.network.select("critic")(obs, actions=aa).mean(0).sum())(a)
        return q_on, q_tg, jnp.linalg.norm(g, axis=1)

    res = json.load(open(out)) if os.path.exists(out) else {}
    for spec in sys.argv[3:]:
        name, step = spec.split(":")
        ag, _, path = load_agent(name, int(step))
        q_on, q_tg, gn = map(np.asarray, f(ag))
        res[spec] = dict(path=path, V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                         V_online_all=float(q_on.mean()), lam=float(1 / np.abs(q_on).mean()),
                         dqda_norm_start=float(gn[reg].mean()), n_start=int(reg.sum()))
        print(spec, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[spec].items() if k != "path"}), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
