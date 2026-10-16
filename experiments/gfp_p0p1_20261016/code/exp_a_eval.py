"""Frozen evaluation of experiment-A actors with the official execution (agent.sample_actions, no Q selection).

usage: python exp_a_eval.py OUT_JSON EVAL_SEED N_ROUNDS ACTOR_PKL [ACTOR_PKL ...]
Same protocol as frozen_eval.py (10 envs, exactly N_ROUNDS*10 episodes, reset seed = EVAL_SEED*1000 + r*10).
The agent is G_task1_s0 at 600k with only modules_actor replaced; sample_actions uses only the actor.
"""
import os
import sys
import json
import time
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium

from frozen_eval import make_creator, run_ckpt, N_ENVS
from common import run_dir

RUN, STEP = "G_task1_s0", 600000
START_REGION = {(1, 1), (2, 1), (1, 2), (2, 2)}


def main():
    out, eval_seed, n_rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    env_name = json.load(open(f"{run_dir(RUN)}/config.json"))["env_name"]
    envs = gymnasium.vector.AsyncVectorEnv([make_creator(env_name)] * N_ENVS,
                                           autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED, context="spawn")
    from common import load_agent
    base, _, _ = load_agent(RUN, STEP)
    results = json.load(open(out)) if os.path.exists(out) else {}
    for pkl in sys.argv[4:]:
        k = f"{os.path.basename(pkl)}:{eval_seed}:{n_rounds}"
        if k in results:
            continue
        t0 = time.time()
        d = pickle.load(open(pkl, "rb"))
        params = dict(base.network.params)
        params["modules_actor"] = d["actor"]
        agent = base.replace(network=base.network.replace(params=params))
        eps = run_ckpt(envs, agent, eval_seed, n_rounds)
        sr = 100 * np.mean([e["success"] for e in eps])
        stay = 100 * np.mean([tuple(e["final_ij"]) in START_REGION for e in eps])
        results[k] = dict(pkl=pkl, branch=d["branch"], it=d["it"], eval_seed=eval_seed, n_episodes=len(eps),
                          success=sr, final_in_start_region=stay, seconds=time.time() - t0, episodes=eps)
        json.dump(results, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{k} n={len(eps)} success={sr:.1f} final_in_start_region={stay:.1f} {time.time() - t0:.0f}s", flush=True)
    envs.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
