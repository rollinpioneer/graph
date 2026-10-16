"""Experiment C: experiment B's fork-and-continue, generalized to any run / fork step, with B1 split.

usage: python exp_c.py PROBE_NPZ OUT_DIR RUN START BRANCH [N_UPDATES]
Run with cwd = official GFP source (source/gfp). Same code path as exp_b.py except RUN/START are arguments
and the frozen TD-target generator can be restricted to one of the two policies.

B0   official continuation (same agent rng, Adam state and fused 4-step sampler as main.py)
B1   critic TD target: next actions from the frozen START actor AND frozen START VaBC (as exp_b.py B1)
B1a  critic TD target: only the actor's next action comes from the frozen START actor; VaBC's from the current VaBC
B1b  critic TD target: only VaBC's next action comes from the frozen START VaBC; the actor's from the current actor
Both generators still share one noise draw (sample_rng), as in GFPAgent.critic_loss; target_agg is unchanged.
Everything else is the unmodified official update. Writes only into OUT_DIR.
"""
import os
import sys
import json
import time
import pickle
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent

PROBE_EVERY = 5000
SAVE_EVERY = 10000
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]


def make_classes():
    from agents.gfp import GFPAgent

    class B1Agent(GFPAgent):
        frozen: Any = None  # dict(actor=params or None, vabc=params or None) of the START checkpoint

        def _generator(self):
            p = dict(self.network.params)
            if self.frozen["actor"] is not None:
                p["modules_actor"] = self.frozen["actor"]
            if self.frozen["vabc"] is not None:
                p["modules_vabc"] = self.frozen["vabc"]
            return GFPAgent(rng=self.rng, network=self.network.replace(params=p), config=self.config)

        def critic_loss(self, batch, grad_params, rng):
            # identical to GFPAgent.critic_loss except that actions come from the frozen generator
            rng, sample_rng = jax.random.split(rng)
            next_obs = batch["next_observations"]
            vabc_obs = next_obs if self.config["target_use_next"] else batch["observations"]
            tgt_agg = self.config["target_agg"]
            gen = self._generator()
            vabc_acts = gen.sample_flow_actions(vabc_obs, sample_rng) if tgt_agg != "actor" else None
            actor_acts = gen.sample_actions(next_obs, sample_rng) if tgt_agg != "vabc" else None
            next_q = self.value_with_all_options(tgt_agg, next_obs, actor_acts, vabc_acts)
            target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
            q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
            critic_loss = jnp.square(q - target_q).mean()
            info = {"critic_loss": critic_loss, "q_mean": q.mean(), "tgt_q_mean": next_q.mean()} if self.config["log_metrics"] else {}
            return critic_loss, info

    return GFPAgent, B1Agent


def main():
    probe_npz, out_dir, RUN, START, branch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    n_updates = int(sys.argv[6]) if len(sys.argv) > 6 else 100000
    assert branch in ("B0", "B1", "B1a", "B1b") and n_updates <= 100000 and n_updates % 4 == 0
    os.makedirs(out_dir, exist_ok=True)
    GFPAgent, B1Agent = make_classes()

    base, cfg, path0 = load_agent(RUN, START)
    # main.py adds log_metrics to the agent config at run time; config.json["agent"] does not store it
    base = base.replace(config=base.config.copy({"log_metrics": bool(cfg["log_metrics"])}))
    fields = dict(rng=base.rng, network=base.network, config=base.config)
    if branch == "B0":
        agent = base
    else:
        fa = base.network.params["modules_actor"] if branch in ("B1", "B1a") else None
        fv = base.network.params["modules_vabc"] if branch in ("B1", "B1b") else None
        agent = B1Agent(**fields, frozen=dict(actor=fa, vabc=fv))

    # dataset exactly as main.py builds it (ReplayBuffer over the processed OGBench dataset, on GPU)
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    from utils.datasets import ReplayBuffer
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    buf = ReplayBuffer.create_from_initial_dataset(dict(train), size=max(cfg["buffer_size"], train.size + 1))
    data = {k: jnp.asarray(v) for k, v in buf.items()}
    dsize = buf.size
    fused, bs = cfg["n_steps_fused"], cfg["agent"]["batch_size"]
    assert not cfg.get("balanced_sampling", False) and cfg["dataset_on_gpu"]

    # probe states for the island / gradient metrics
    probe = np.load(probe_npz)
    p_obs, p_z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    a0 = None
    g0 = None

    @jax.jit
    def probe_fn(ag):
        a = jnp.clip(ag.network.select("actor")(p_obs, p_z), -1, 1)
        q_on = ag.network.select("critic")(p_obs, actions=a).mean(0)
        q_tg = ag.critic_eval(p_obs, a)
        return a, q_on, q_tg

    @jax.jit
    def dqda(ag, a):
        return jax.grad(lambda aa: ag.network.select("critic")(p_obs, actions=aa).mean(0).sum())(a)

    log, t0, it = [], time.time(), 0
    while it <= n_updates:
        if it % PROBE_EVERY == 0:
            a, q_on, q_tg = map(np.asarray, probe_fn(agent))
            if a0 is None:
                a0 = a
                g0 = np.asarray(dqda(agent, jnp.asarray(a0)))
            g = np.asarray(dqda(agent, jnp.asarray(a0)))
            cos = (g * g0).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(g0, axis=1) + 1e-12)
            row = dict(step=START + it, V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                       V_online_all=float(q_on.mean()), lam=float(1 / np.abs(q_on).mean()),
                       dqda_cos_vs600_start=float(cos[reg].mean()), dqda_norm_start=float(np.linalg.norm(g, axis=1)[reg].mean()),
                       action_shift_start=float(np.linalg.norm(a - a0, axis=1)[reg].mean()), sec=time.time() - t0)
            if it > 0:
                row.update({k: float(v) for k, v in info.items() if k.startswith(("critic/", "actor/"))})
            log.append(row)
            print(branch, json.dumps(row), flush=True)
        if it > 0 and it % SAVE_EVERY == 0:
            plain = GFPAgent(rng=agent.rng, network=agent.network, config=agent.config)
            with open(f"{out_dir}/params_{START + it}.pkl", "wb") as f:
                pickle.dump(dict(agent=flax.serialization.to_state_dict(plain), branch=branch, run=RUN, start=path0), f)
        if it == n_updates:
            break
        agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
        it += fused
    json.dump(dict(branch=branch, run=RUN, start=path0, n_updates=n_updates, log=log), open(f"{out_dir}/train_{branch}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
