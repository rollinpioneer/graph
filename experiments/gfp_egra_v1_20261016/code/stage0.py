"""EGRA stage 0: read-only calibration of the gate statistic on existing 10k-spaced fork checkpoints. No training.

usage: python stage0.py OUT_JSON
For each fork trajectory (run, fork step, checkpoint dir, branch), with the reference actor fixed to the fork actor:
  v_t(i) = Q_target_t(s_i, clip(actor_fork(s_i, z_i)))            (critic_eval: target critic, q_agg)
  d_t    = quantile_0.95 |v_t - v_{t-10k}|                         (gate statistic, W = 10k)
  shock_t = Q_target_t(s_i, clip(actor_t(s_i, z_i))) - v_t(i)      (what a swap at t would inject; 0.95 quantile of |.|, mean)
Start-region (cells (1,1),(2,1),(1,2),(2,2)) breakdowns are reported for interpretation only.
"""
import os
import sys
import json
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")

import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent, run_dir
from gate_probe import make_gate_probe, gate_probe_cells

EXP = "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/exp_ab"
TRAJ = [
    ("s0_B1a", "G_task1_s0", 600000, f"{EXP}/c/G_task1_s0_B1a"),
    ("s3_B1a", "G_task1_s3", 300000, f"{EXP}/c/G_task1_s3_B1a"),
    ("s0_B0", "G_task1_s0", 600000, f"{EXP}/b_B0"),
    ("s3_B0", "G_task1_s3", 300000, f"{EXP}/c/G_task1_s3_B0"),
    ("s0_B1", "G_task1_s0", 600000, f"{EXP}/b_B1"),
    ("s3_B1", "G_task1_s3", 300000, f"{EXP}/c/G_task1_s3_B1"),
]
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]


def main():
    out = sys.argv[1]
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    cfg = json.load(open(f"{run_dir('G_task1_s0')}/config.json"))
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    idx, P_obs, P_z = make_gate_probe(np.asarray(train["observations"]))
    ci, cj = gate_probe_cells(P_obs)
    reg = np.zeros(len(idx), bool)
    for i, j in START_REGION:
        reg |= (ci == i) & (cj == j)
    P_obs, P_z = jnp.asarray(P_obs), jnp.asarray(P_z)

    @jax.jit
    def acts(ag):
        return jnp.clip(ag.network.select("actor")(P_obs, P_z), -1, 1)

    @jax.jit
    def qt(ag, a):
        return ag.critic_eval(P_obs, a)

    res = dict(n_probe=int(len(idx)), n_start=int(reg.sum()), probe_idx_sha=str(hash(idx.tobytes())), traj={})
    for name, run, fork, d in TRAJ:
        base, _, _ = load_agent(run, fork)
        a_ref = acts(base)
        rows, prev = [], None
        for step in range(fork, fork + 100001, 10000):
            if step == fork:
                ag = base
            else:
                p = f"{d}/params_{step}.pkl"
                ag = flax.serialization.from_state_dict(base, pickle.load(open(p, "rb"))["agent"])
            v = np.asarray(qt(ag, a_ref))
            sh = np.asarray(qt(ag, acts(ag))) - v
            row = dict(step=step, v_mean=float(v.mean()), v_start=float(v[reg].mean()),
                       shock_q95=float(np.quantile(np.abs(sh), 0.95)), shock_mean=float(sh.mean()),
                       shock_start_mean=float(sh[reg].mean()))
            if prev is not None:
                dv = np.abs(v - prev)
                row.update(d_q95=float(np.quantile(dv, 0.95)), d_q90=float(np.quantile(dv, 0.90)),
                           d_q99=float(np.quantile(dv, 0.99)), d_mean=float(dv.mean()),
                           d_start_mean=float(dv[reg].mean()), d_rest_q95=float(np.quantile(dv[~reg], 0.95)))
            prev = v
            rows.append(row)
            print(name, json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in row.items()}), flush=True)
        res["traj"][name] = rows
        json.dump(res, open(out, "w"), indent=1)
    b1a = [r["d_q95"] for n in ("s0_B1a", "s3_B1a") for r in res["traj"][n] if "d_q95" in r]
    res["eps_rule"] = "eps = 1.5 * max d_q95 over s0_B1a and s3_B1a"
    res["eps"] = 1.5 * max(b1a)
    json.dump(res, open(out, "w"), indent=1)
    print("EPS", res["eps"], flush=True)


if __name__ == "__main__":
    main()
