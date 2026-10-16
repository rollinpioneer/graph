"""Post-hoc size check for P2's target_agg=actor (no training): how much would the TD target move?

usage: python target_gap.py STATES_NPZ OUT_JSON NAME:STEP [...]
Official GFP target with target_agg=mean uses (Q(s', a_actor) + Q(s', a_flow)) / 2; target_agg=actor uses Q(s', a_actor).
The per-state difference is (Q(s', a_actor) - Q(s', a_flow)) / 2. Evaluated on the probe's fixed dataset states
(next observations are not stored there, so s is used as a stand-in for s'), with the probe's fixed noise z.
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

CHUNK = 5000
START_REGION = {"task1": [(1, 1), (2, 1), (1, 2), (2, 2)], "task4": [(6, 5), (6, 6)]}


@jax.jit
def gap(agent, obs, z):
    a_act = jnp.clip(agent.network.select("actor")(obs, z), -1, 1)
    a_flow = agent.compute_flow_actions(obs, z)
    q_act, q_flow = agent.critic_eval(obs, a_act), agent.critic_eval(obs, a_flow)
    return q_act, q_flow, jnp.linalg.norm(a_act - a_flow, axis=-1)


def main():
    m = np.load(sys.argv[1])
    obs, z, ci, cj = m["obs"], m["z"], m["cell_i"], m["cell_j"]
    res = {}
    for spec in sys.argv[3:]:
        name, step = spec.split(":")
        agent, _, _ = load_agent(name, int(step))
        out = [gap(agent, obs[k:k + CHUNK], z[k:k + CHUNK]) for k in range(0, len(obs), CHUNK)]
        qa, qf, dl = (np.concatenate([np.asarray(o[i]) for o in out]) for i in range(3))
        half = (qa - qf) / 2
        reg = np.zeros(len(obs), bool)
        for i, j in START_REGION[name.split("_")[1]]:
            reg |= (ci == i) & (cj == j)
        res[spec] = dict(mean_abs_Q=float(np.abs(qa).mean()), action_l2_actor_flow=float(dl.mean()),
                         target_shift_mean=float(half.mean()), target_shift_abs=float(np.abs(half).mean()),
                         start_region=dict(n=int(reg.sum()), target_shift_mean=float(half[reg].mean()),
                                           target_shift_abs=float(np.abs(half[reg]).mean())))
        print(spec, json.dumps(res[spec]), flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
