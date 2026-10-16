"""X3 card: read-only actor-motion reference from existing checkpoints (no training).

usage: python x3_motion.py PROBE_NPZ OUT_JSON
Run with cwd = official GFP source (source/gfp, 1abbd5f). For each fork (s0@600k, s3@300k) and each existing
branch (B0, B1a from experiments B/C/D; E, P10, P25, M1, M5, R from EGRA stage 1) it loads the 10k-spaced
checkpoints at fork + 50k/100k/150k/200k (where they exist) and measures, on the fixed evaluation probe
with its fixed noises z:
  shift_fork   mean ||a_t - a_fork||          (actor action, clipped to [-1, 1])
  shift_50k    mean ||a_t - a_(t-50k)||       (late-window motion; a_fork is used for t-50k = fork)
  gap_vabc     mean ||a_t - f_t||             (f = VaBC flow action, same z, clipped)
  vabc_shift   mean ||f_t - f_fork||
  qadv         mean Q_t(s, a_t) - Q_t(s, f_t) (online critic, q_agg over the ensemble)
each over all probe states and over the start region. The same quantities are logged by x3_fork.py.
"""
import json
import os
import pickle
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")
import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent

A = "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1"
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]
FORKS = {"G_task1_s0": 600000, "G_task1_s3": 300000}
DIRS = {
    ("G_task1_s0", "B0"): f"{A}/exp_ab/b_B0",
    ("G_task1_s0", "B1a"): f"{A}/exp_ab/c/G_task1_s0_B1a",
    ("G_task1_s3", "B0"): f"{A}/exp_ab/c/G_task1_s3_B0",
    ("G_task1_s3", "B1a"): f"{A}/exp_ab/c/G_task1_s3_B1a",
}
for r in FORKS:
    for arm in ("E", "P10", "P25", "M1", "M5", "R"):
        DIRS[(r, arm)] = f"{A}/egra/s1/{r}_{arm}"


def measure(agent, obs, z):
    a = jnp.clip(agent.network.select("actor")(obs, z), -1, 1)
    f = jnp.clip(agent.compute_flow_actions(obs, noises=z), -1, 1)
    agg = getattr(jnp, agent.config["q_agg"])
    qa = agg(agent.network.select("critic")(obs, actions=a), axis=0)
    qf = agg(agent.network.select("critic")(obs, actions=f), axis=0)
    return np.asarray(a), np.asarray(f), np.asarray(qa - qf)


def main():
    probe_npz, out = sys.argv[1], sys.argv[2]
    probe = np.load(probe_npz)
    obs, z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)

    def stat(x):
        return dict(all=float(x.mean()), start=float(x[reg].mean()))

    res = {}
    for run, fork in FORKS.items():
        template, _, _ = load_agent(run, fork)
        a0, f0, q0 = measure(template, obs, z)
        res[f"{run}:fork"] = dict(step=fork, gap_vabc=stat(np.linalg.norm(a0 - f0, axis=1)), qadv=stat(q0))
        for (r, arm), d in DIRS.items():
            if r != run:
                continue
            prev = {fork: a0}
            for k in (50000, 100000, 150000, 200000):
                p = f"{d}/params_{fork + k}.pkl"
                if not os.path.exists(p):
                    continue
                ag = flax.serialization.from_state_dict(template, pickle.load(open(p, "rb"))["agent"])
                a, f, qd = measure(ag, obs, z)
                prev[fork + k] = a
                row = dict(step=fork + k, shift_fork=stat(np.linalg.norm(a - a0, axis=1)),
                           gap_vabc=stat(np.linalg.norm(a - f, axis=1)), vabc_shift=stat(np.linalg.norm(f - f0, axis=1)),
                           qadv=stat(qd))
                if fork + k - 50000 in prev:
                    row["shift_50k"] = stat(np.linalg.norm(a - prev[fork + k - 50000], axis=1))
                res[f"{run}:{arm}:+{k // 1000}k"] = row
                print(run, arm, json.dumps(row), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
