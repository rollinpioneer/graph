"""Stage-1 confirmation, fork selection only: start-region value scan over saved formal checkpoints of any task
(no env, no training). Same quantities as analysis_p0p1/code/island_scan.py, but the start region is taken from the
run's task: every free cell within REGION_STEPS maze steps (4-neighbour BFS) of the task's init cell.
For task1 this gives exactly the screening region (1,1),(2,1),(1,2),(2,2) (checked at start-up).

usage: python island_scan_t4.py PROBE_NPZ OUT_JSON TASK NAME:STEP [NAME:STEP ...]
"""
import os
import sys
import json

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")

import numpy as np
import jax
import jax.numpy as jnp

from common import load_agent, bfs_dist, TASKS

REGION_STEPS = 2


def start_region(task):
    d = bfs_dist(TASKS[task]["init_ij"])
    return sorted((int(i), int(j)) for i, j in zip(*np.where((d >= 0) & (d <= REGION_STEPS))))


assert start_region(1) == sorted([(1, 1), (2, 1), (1, 2), (2, 2)]), start_region(1)


def main():
    probe = np.load(sys.argv[1])
    out, task = sys.argv[2], int(sys.argv[3])
    region = start_region(task)
    print("task", task, "start region", region, flush=True)
    obs, z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in region:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    gd = bfs_dist(TASKS[task]["goal_ij"])[probe["cell_i"], probe["cell_j"]]

    @jax.jit
    def f(ag):
        a = jnp.clip(ag.network.select("actor")(obs, z), -1, 1)
        q_on = ag.network.select("critic")(obs, actions=a).mean(0)
        q_tg = ag.critic_eval(obs, a)
        g = jax.grad(lambda aa: ag.network.select("critic")(obs, actions=aa).mean(0).sum())(a)
        return q_on, q_tg, jnp.linalg.norm(g, axis=1)

    res = json.load(open(out)) if os.path.exists(out) else {}
    for spec in sys.argv[4:]:
        name, step = spec.split(":")
        ag, cfg, path = load_agent(name, int(step))
        assert cfg["env_name"].endswith(f"task{task}-v0"), cfg["env_name"]
        q_on, q_tg, gn = map(np.asarray, f(ag))
        ok = gd >= 0
        res[spec] = dict(path=path, region=region, n_start=int(reg.sum()),
                         V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                         V_online_all=float(q_on.mean()), contrast=float(q_on[reg].mean() - q_on.mean()),
                         lam=float(1 / np.abs(q_on).mean()), dqda_norm_start=float(gn[reg].mean()),
                         corr_V_bfs=float(np.corrcoef(q_on[ok], gd[ok])[0, 1]))
        print(spec, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[spec].items()
                                if k not in ("path", "region")}), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
