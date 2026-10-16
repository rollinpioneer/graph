"""Step 1 (no training): do both critic heads raise the island together, and how do critic values compare
with independent return references?

usage: python critic_check.py OUT_JSON [EVAL_JSON ...]   (with eval files: only those checkpoints)
Run with cwd = official GFP source (source/gfp, 1abbd5f). Read-only: loads existing checkpoints, resets
evaluation environments (no stepping) and reads existing evaluation episodes and the training dataset.

For every checkpoint that already has a frozen evaluation (same eval seed 4242, same reset seeds):
  start states   s0 of every evaluated episode, rebuilt by env.reset(seed=4242*1000 + 10*round) (no step)
  V_head[k](s0)  online critic head k at (s0, clip(actor(s0, z))), averaged over 16 fixed z per state
  V_tgt(s0)      target critic, q_agg=mean over heads (the quantity the TD target uses)
  G_real         realized discounted return of the same episode under the deployed policy
                 (reward per step from the dataset convention, see REWARD below; 2000-step failures are
                 within 0.01 of the infinite-horizon value)
  gap            V(s0) - G_real, per episode and averaged
On the fixed evaluation probe (start region and all states):
  head k Q at the actor action and at the dataset action; head disagreement |Q1 - Q2|
  G_data         discounted return of the dataset trajectory from that state (behaviour policy, truncated at
                 the trajectory end; rewards are <= 0, so truncation can only make G_data too high)
"""
import glob
import json
import os
import pickle
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
A = "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1"
sys.path.insert(0, f"{A}/code")

import numpy as np
import gymnasium

from frozen_eval import make_creator, N_ENVS
from common import load_agent, run_dir

EVAL_SEED, N_Z = 4242, 16
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]
ORIG_EVALS = [f"{A}/out/p0a_s0.json", f"{A}/out/p0a_ctrl.json", f"{A}/exp_ab/c/eval_orig_s3.json"]
PKL_EVALS = (sorted(glob.glob(f"{A}/exp_ab/eval_B[0-9].json")) + sorted(glob.glob(f"{A}/exp_ab/c/eval_c.json.*"))
             + sorted(glob.glob(f"{A}/egra/out/eval_s1_*.json")) + sorted(glob.glob(f"{A}/x3/out/eval_x3_*.json")))
PKL_EVALS = [f for f in PKL_EVALS if "summary" not in f]   # summaries duplicate the per-arm files


def collect_pairs(only=None):
    pairs = []
    for f in (ORIG_EVALS if only is None else []):
        for k, v in json.load(open(f)).items():
            run, step = k.split(":")[:2]
            pairs.append(dict(run=run, branch="orig", step=int(step), pkl=None, eval_file=f, key=k))
    for f in (PKL_EVALS if only is None else only):
        for k, v in json.load(open(f)).items():
            if "pkl" not in v or not os.path.exists(v["pkl"]):
                continue
            step = int(os.path.basename(v["pkl"]).split("_")[1].split(".")[0])
            m = re.search(r"G_task1_s\d", v["pkl"])
            run = v.get("run") or (m.group(0) if m else "G_task1_s0")   # exp_ab/b_B* are the s0 600k forks
            pairs.append(dict(run=run, branch=v.get("branch", k.split(":")[1]), step=step,
                              pkl=v["pkl"], eval_file=f, key=k))
    return pairs


def main():
    out = sys.argv[1]
    pairs = collect_pairs(sys.argv[2:] or None)
    env_name = json.load(open(f"{run_dir('G_task1_s0')}/config.json"))["env_name"]
    envs = gymnasium.vector.AsyncVectorEnv([make_creator(env_name)] * N_ENVS,
                                           autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED, context="spawn")
    max_rounds = 20
    init = {}
    for r in range(max_rounds):
        obs, _ = envs.reset(seed=int(EVAL_SEED * 1000 + r * N_ENVS))
        for k in range(N_ENVS):
            init[(r, k)] = obs[k].copy()
    envs.close()

    import jax
    import jax.numpy as jnp
    import flax
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH

    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, env_name)
    rew, masks = np.asarray(train["rewards"]), np.asarray(train["masks"])
    terms = np.asarray(train["terminals"]) if "terminals" in train else None
    REWARD = dict(values=np.unique(rew).tolist(), rew_at_mask0=np.unique(rew[masks == 0]).tolist(),
                  rew_at_mask1=np.unique(rew[masks == 1]).tolist())
    gamma = 0.995
    G = np.zeros(len(rew))
    nxt = 0.0
    for t in range(len(rew) - 1, -1, -1):
        if terms is not None and terms[t]:
            nxt = 0.0
        G[t] = rew[t] + gamma * masks[t] * nxt
        nxt = G[t]
    # per-step reward of the deployed policy in the eval env: -1 per step, the success transition gets
    # the dataset's mask-0 reward
    r_succ = float(REWARD["rew_at_mask0"][0]) if REWARD["rew_at_mask0"] else -1.0
    r_step = float(REWARD["rew_at_mask1"][0]) if len(REWARD["rew_at_mask1"]) == 1 else -1.0

    def realized(ep):
        T = ep["length"]
        if ep["terminated"] and ep["success"] > 0:
            return r_step * (1 - gamma ** (T - 1)) / (1 - gamma) + r_succ * gamma ** (T - 1)
        return r_step * (1 - gamma ** T) / (1 - gamma)

    probe = np.load(f"{A}/out/probe/states.npz")
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    p_obs, p_act, p_z = jnp.asarray(probe["obs"]), jnp.asarray(probe["act"]), jnp.asarray(probe["z"])
    G_probe = G[probe["idx"]]

    templates = {}
    zkey = jax.random.PRNGKey(20261016)

    def get_template(run):
        if run not in templates:
            templates[run] = load_agent(run, 600000 if run == "G_task1_s0" else 300000 if run == "G_task1_s3" else 600000)[0]
        return templates[run]

    @jax.jit
    def start_values(ag, s0, z):
        # s0 (N, ob), z (N_Z, N, act)
        def one(zz):
            a = jnp.clip(ag.network.select("actor")(s0, zz), -1, 1)
            return ag.network.select("critic")(s0, actions=a), ag.network.select("target_critic")(s0, actions=a)
        q_on, q_tg = jax.vmap(one)(z)          # (N_Z, 2, N)
        return q_on.mean(0), q_tg.mean(0)

    @jax.jit
    def probe_values(ag):
        a = jnp.clip(ag.network.select("actor")(p_obs, p_z), -1, 1)
        return ag.network.select("critic")(p_obs, actions=a), ag.network.select("critic")(p_obs, actions=p_act)

    res = dict(REWARD=REWARD, gamma=gamma, n_dataset=int(len(rew)),
               G_data_probe=dict(start_mean=float(G_probe[reg].mean()), start_max=float(G_probe[reg].max()),
                                 start_frac_reach=float((G_probe[reg] > -150).mean()), all_mean=float(G_probe.mean())),
               rows=[])
    for p in pairs:
        ev = json.load(open(p["eval_file"]))[p["key"]]
        eps = [e for e in ev["episodes"] if (e["round"], e["env"]) in init]
        if p["pkl"] is None:
            ag = load_agent(p["run"], p["step"])[0]
        else:
            ag = flax.serialization.from_state_dict(get_template(p["run"]), pickle.load(open(p["pkl"], "rb"))["agent"])
        s0 = jnp.asarray(np.stack([init[(e["round"], e["env"])] for e in eps]))
        z = jax.random.normal(zkey, (N_Z, len(eps), p_act.shape[1]))
        q_on, q_tg = map(np.asarray, start_values(ag, s0, z))
        g = np.array([realized(e) for e in eps])
        qa, qd = map(np.asarray, probe_values(ag))
        succ = np.array([e["success"] for e in eps]) > 0
        row = dict(run=p["run"], branch=p["branch"], step=p["step"], n_eps=len(eps), success=float(100 * succ.mean()),
                   V_on_head=[float(q_on[0].mean()), float(q_on[1].mean())], V_on=float(q_on.mean()),
                   V_tgt=float(q_tg.mean()), head_gap_s0=float(np.abs(q_on[0] - q_on[1]).mean()),
                   G_real=float(g.mean()), gap_on=float(q_on.mean() - g.mean()), gap_tgt=float(q_tg.mean() - g.mean()),
                   gap_head=[float(q_on[0].mean() - g.mean()), float(q_on[1].mean() - g.mean())],
                   corr_V_G=float(np.corrcoef(q_on.mean(0), g)[0, 1]) if g.std() > 0 and q_on.mean(0).std() > 0 else None,
                   probe_start_actor_head=[float(qa[0][reg].mean()), float(qa[1][reg].mean())],
                   probe_start_data_head=[float(qd[0][reg].mean()), float(qd[1][reg].mean())],
                   probe_all_actor_head=[float(qa[0].mean()), float(qa[1].mean())],
                   probe_head_gap_start=float(np.abs(qa[0] - qa[1])[reg].mean()),
                   probe_head_gap_all=float(np.abs(qa[0] - qa[1]).mean()),
                   probe_start_data_minus_Gdata=float(qd.mean(0)[reg].mean() - G_probe[reg].mean()))
        res["rows"].append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
