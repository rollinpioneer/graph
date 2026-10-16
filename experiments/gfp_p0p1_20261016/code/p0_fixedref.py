"""P0: does V(s) rise because the critic changed or because the actor's action changed?

usage: python p0_fixedref.py STATES_NPZ OUT_JSON NAME:STEP_A:STEP_B [...]
On the probe's fixed dataset states and fixed noise z:
  total  = Q_B(a_B) - Q_A(a_A)
  critic = Q_B(a_A) - Q_A(a_A)     (actions fixed to A's actor)
  actor  = Q_A(a_B) - Q_A(a_A)     (critic fixed to A)
  inter  = total - critic - actor
  data   = Q_B(a_data) - Q_A(a_data)
Q = official critic_eval (target critic, ensemble mean); a = clip(actor(s, z)).
"""
import os
import sys
import json

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from common import load_agent, MAZE_MAP

CHUNK = 5000
START_REGION = ["1,1", "2,1", "1,2", "2,2"]


@jax.jit
def actor_act(agent, obs, z):
    return jnp.clip(agent.network.select("actor")(obs, z), -1, 1)


@jax.jit
def q(agent, obs, a):
    return agent.critic_eval(obs, a)


def batched(f, *arrs):
    return np.concatenate([np.asarray(f(*[x[k:k + CHUNK] for x in arrs])) for k in range(0, len(arrs[0]), CHUNK)])


def main():
    m = np.load(sys.argv[1])
    obs, act, z, ci, cj = m["obs"], m["act"], m["z"], m["cell_i"], m["cell_j"]
    res = {}
    for spec in sys.argv[3:]:
        name, sa, sb = spec.split(":")
        A, _, _ = load_agent(name, int(sa))
        B, _, _ = load_agent(name, int(sb))
        aA = batched(lambda o, zz: actor_act(A, o, zz), obs, z)
        aB = batched(lambda o, zz: actor_act(B, o, zz), obs, z)
        QA_aA = batched(lambda o, a: q(A, o, a), obs, aA)
        QA_aB = batched(lambda o, a: q(A, o, a), obs, aB)
        QB_aA = batched(lambda o, a: q(B, o, a), obs, aA)
        QB_aB = batched(lambda o, a: q(B, o, a), obs, aB)
        QA_d = batched(lambda o, a: q(A, o, a), obs, act)
        QB_d = batched(lambda o, a: q(B, o, a), obs, act)
        parts = dict(total=QB_aB - QA_aA, critic=QB_aA - QA_aA, actor=QA_aB - QA_aA, data=QB_d - QA_d)
        parts["inter"] = parts["total"] - parts["critic"] - parts["actor"]
        cells = {}
        for (i, j) in zip(*np.nonzero(MAZE_MAP == 0)):
            sel = (ci == i) & (cj == j)
            if sel.sum() < 20:
                continue
            cells[f"{i},{j}"] = dict(n=int(sel.sum()), V_A=float(QA_aA[sel].mean()), V_B=float(QB_aB[sel].mean()),
                                     **{k: float(v[sel].mean()) for k, v in parts.items()})
        reg = np.zeros(len(obs), bool)
        for c in START_REGION:
            i, j = map(int, c.split(","))
            reg |= (ci == i) & (cj == j)
        res[spec] = dict(
            all={k: float(v.mean()) for k, v in parts.items()},
            all_abs={k: float(np.abs(v).mean()) for k, v in parts.items()},
            start_region=dict(n=int(reg.sum()), **{k: float(v[reg].mean()) for k, v in parts.items()}),
            action_shift_l2=float(np.linalg.norm(aB - aA, axis=1).mean()),
            cells=cells)
        sr = res[spec]["start_region"]
        print(spec, "start_region", {k: round(sr[k], 2) for k in ("total", "critic", "actor", "inter", "data")},
              "all", {k: round(v, 2) for k, v in res[spec]["all"].items()}, flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
