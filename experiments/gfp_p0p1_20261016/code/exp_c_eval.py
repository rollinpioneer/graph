"""Frozen evaluation of experiment-C checkpoints with the official execution (agent.sample_actions, no Q selection).

usage: python exp_c_eval.py OUT_JSON RUN STEP EVAL_SEED N_ROUNDS PARAMS_PKL [PARAMS_PKL ...]
Same protocol as frozen_eval.py (10 envs, exactly N_ROUNDS*10 episodes, reset seed = EVAL_SEED*1000 + r*10).
Each PARAMS_PKL is written by exp_c.py (full GFPAgent state dict); it is loaded onto the RUN:STEP template agent
(same run, so the static config is that run's own).
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
import flax

from frozen_eval import make_creator, run_ckpt, N_ENVS
from common import run_dir, load_agent

START_REGION = {(1, 1), (2, 1), (1, 2), (2, 2)}


def main():
    out, RUN, STEP, eval_seed, n_rounds = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    env_name = json.load(open(f"{run_dir(RUN)}/config.json"))["env_name"]
    envs = gymnasium.vector.AsyncVectorEnv([make_creator(env_name)] * N_ENVS,
                                           autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED, context="spawn")
    template, _, _ = load_agent(RUN, STEP)
    results = json.load(open(out)) if os.path.exists(out) else {}
    for pkl in sys.argv[6:]:
        d = pickle.load(open(pkl, "rb"))
        assert d.get("run", RUN) == RUN, (d.get("run"), RUN)
        k = f"{RUN}:{d['branch']}:{os.path.basename(pkl)}:{eval_seed}:{n_rounds}"
        if k in results:
            continue
        t0 = time.time()
        agent = flax.serialization.from_state_dict(template, d["agent"])
        eps = run_ckpt(envs, agent, eval_seed, n_rounds)
        sr = 100 * np.mean([e["success"] for e in eps])
        stay = 100 * np.mean([tuple(e["final_ij"]) in START_REGION for e in eps])
        results[k] = dict(pkl=pkl, run=RUN, branch=d["branch"], eval_seed=eval_seed, n_episodes=len(eps),
                          success=sr, final_in_start_region=stay, seconds=time.time() - t0, episodes=eps)
        json.dump(results, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        print(f"{k} n={len(eps)} success={sr:.1f} final_in_start_region={stay:.1f} {time.time() - t0:.0f}s", flush=True)
    envs.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
