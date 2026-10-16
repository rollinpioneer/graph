"""Stage-2 confirmation (card section 4.2): start-region value scan over the 100k checkpoints of every from-scratch
run, both arms, with the same probe, start region and quantities as confirm_t4/code/island_scan_t4.py (task 1).
I-arm checkpoints carry an extra 'value' entry (the IQL V net); only the agent network is loaded, onto the official
GFPAgent template, so the critic / actor forward passes are the official ones.

usage: python scan_scratch.py PROBE_NPZ OUT_JSON RUN_DIR:NAME [RUN_DIR:NAME ...]   (RUN_DIR = .../out/*/date_*)
"""
import os
import sys
import json
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/confirm_t4/code")

import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent, bfs_dist, TASKS
from island_scan_t4 import start_region

STEPS = list(range(100000, 1000001, 100000))


def main():
    probe = np.load(sys.argv[1])
    out = sys.argv[2]
    region = start_region(1)
    obs, z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in region:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    gd = bfs_dist(TASKS[1]["goal_ij"])[probe["cell_i"], probe["cell_j"]]
    ok = gd >= 0
    template, _, _ = load_agent("G_task1_s0", 100000)

    @jax.jit
    def f(ag):
        a = jnp.clip(ag.network.select("actor")(obs, z), -1, 1)
        q_on = ag.network.select("critic")(obs, actions=a).mean(0)
        g = jax.grad(lambda aa: ag.network.select("critic")(obs, actions=aa).mean(0).sum())(a)
        return q_on, jnp.linalg.norm(g, axis=1)

    res = json.load(open(out)) if os.path.exists(out) else {}
    for spec in sys.argv[3:]:
        rd, name = spec.split(":")
        cfg = json.load(open(f"{rd}/config.json"))
        assert cfg["env_name"].endswith("task1-v0")
        for step in STEPS:
            k = f"{name}:{step}"
            if k in res:
                continue
            sd = dict(pickle.load(open(f"{rd}/params_{step}.pkl", "rb"))["agent"])
            has_v = sd.pop("value", None) is not None
            ag = flax.serialization.from_state_dict(template, sd)
            q_on, gn = map(np.asarray, f(ag))
            res[k] = dict(run=name, step=step, has_value_net=has_v, V_start=float(q_on[reg].mean()),
                          V_all=float(q_on.mean()), contrast=float(q_on[reg].mean() - q_on.mean()),
                          dqda_norm_start=float(gn[reg].mean()), corr_V_bfs=float(np.corrcoef(q_on[ok], gd[ok])[0, 1]))
            print(k, json.dumps({a: round(b, 3) if isinstance(b, float) else b for a, b in res[k].items()}), flush=True)
            json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
