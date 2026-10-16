"""EGRA stage 1: fork-and-continue with a replaced TD-target actor generator (only the actor branch changes).

usage: python egra_fork.py PROBE_NPZ OUT_DIR RUN START ARM N_UPDATES EPS
Run with cwd = official GFP source (source/gfp, 1abbd5f, unmodified); the subclass below overrides only
GFPAgent.critic_loss (actor-branch generator) and, for the EMA arms, multi_sample_and_update (adds the EMA step).

ARM  TD-target actor branch a'_act = clip(mu_gen(s', z))     (z shared with the VaBC branch, as in the official code)
B0   gen = current actor (official; not part of the stage-1 matrix, smoke only)
E    EGRA: hard snapshot; swap to current actor when (since>=K_MIN and d<=EPS) or since>=K_MAX
P10  hard snapshot, swap every 10k          P25  every 25k
M1   EMA of the actor, tau_a=1e-4           M5   tau_a=5e-3 (TD3/ReBRAC default), updated every step after the update
R    SMTA: ring of M=4 snapshots (push every 10k, newest first), per-sample uniform pick
At START every generator equals the START actor (so the first steps equal B1a for E/P/R).
VaBC branch, guidance, actor loss, rewards, discount, target_use_next semantics: unchanged.
Gate (E only): every C=1000 updates, v = Q_target(s_i, clip(mu_ref(s_i, z_i))) on the 4096-state label-free gate probe;
d = quantile_0.95 |v - v_(t-10k)| (same reference). After a swap v is recomputed under the new reference.
Writes only into OUT_DIR.
"""
import os
import sys
import json
import time
import pickle
from functools import partial
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")

import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent
from gate_probe import make_gate_probe

PROBE_EVERY, SAVE_EVERY, CHECK_EVERY = 5000, 10000, 1000
K_MIN, K_MAX, W, Q = 10000, 50000, 10000, 0.95
RING_M, RING_DELTA = 4, 10000
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]
ARMS = dict(B0=("current", None), E=("ref", None), P10=("ref", 10000), P25=("ref", 25000),
            M1=("ema", 1e-4), M5=("ema", 5e-3), R=("ring", None))


def make_classes():
    from agents.gfp import GFPAgent
    from utils.datasets import GPU_dataset_sample

    class GenAgent(GFPAgent):
        ref: Any = None  # actor params (E/P/M) or stacked actor params with leading dim RING_M (R)

        def _actor_with(self, pa, obs, z):
            p = dict(self.network.params)
            p["modules_actor"] = pa
            return self.network.select("actor")(obs, z, params=p)

        def gen_actions(self, obs, z, pick_rng):
            src = self.config["td_actor_source"]
            if src == "current":
                return jnp.clip(self.network.select("actor")(obs, z), -1, 1)
            if src == "ring":
                allacts = jax.vmap(lambda pa: self._actor_with(pa, obs, z))(self.ref)       # (M, B, A)
                j = jax.random.randint(pick_rng, (obs.shape[0],), 0, allacts.shape[0])
                return jnp.clip(allacts[j, jnp.arange(obs.shape[0])], -1, 1)
            return jnp.clip(self._actor_with(self.ref, obs, z), -1, 1)

        def critic_loss(self, batch, grad_params, rng):
            # identical to GFPAgent.critic_loss except the actor-branch generator
            rng, sample_rng = jax.random.split(rng)
            pick_rng = jax.random.fold_in(rng, 1)
            next_obs = batch["next_observations"]
            vabc_obs = next_obs if self.config["target_use_next"] else batch["observations"]
            tgt_agg = self.config["target_agg"]
            vabc_acts = self.sample_flow_actions(vabc_obs, sample_rng) if tgt_agg != "actor" else None
            if tgt_agg != "vabc":
                z = self.sample_noises(next_obs, sample_rng)  # same z as sample_actions / sample_flow_actions
                actor_acts = self.gen_actions(next_obs, z, pick_rng)
            else:
                actor_acts = None
            next_q = self.value_with_all_options(tgt_agg, next_obs, actor_acts, vabc_acts)
            target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
            q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
            critic_loss = jnp.square(q - target_q).mean()
            info = {"critic_loss": critic_loss, "q_mean": q.mean(), "tgt_q_mean": next_q.mean()} if self.config["log_metrics"] else {}
            return critic_loss, info

        @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
        def multi_sample_and_update(self, n_steps_fused, dataset, dataset_size, batch_size):
            # official loop + (EMA arms only) actor EMA after each update
            _agent = self
            for _ in range(n_steps_fused):
                batch, rng = GPU_dataset_sample(_agent.rng, dataset, batch_size, dataset_size)
                rng, loss_rng = jax.random.split(rng)

                def loss_fn(grad_params):
                    return _agent.total_loss(batch, grad_params, rng=loss_rng)

                new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
                _agent.target_update(new_network, "critic")
                ref = _agent.ref
                if _agent.config["td_actor_source"] == "ema":
                    ta = _agent.config["tau_actor"]
                    ref = jax.tree_util.tree_map(lambda p, tp: p * ta + tp * (1 - ta), new_network.params["modules_actor"], ref)
                _agent = _agent.replace(network=new_network, rng=rng, ref=ref)
            return _agent, info

    return GFPAgent, GenAgent


def main():
    probe_npz, out_dir, RUN, START, ARM = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    n_updates, EPS = int(sys.argv[6]), float(sys.argv[7])
    assert ARM in ARMS and n_updates <= 200000 and n_updates % 4 == 0
    os.makedirs(out_dir, exist_ok=True)
    GFPAgent, GenAgent = make_classes()
    src, par = ARMS[ARM]

    base, cfg, path0 = load_agent(RUN, START)
    conf = base.config.copy({"log_metrics": bool(cfg["log_metrics"]), "td_actor_source": src,
                             "tau_actor": float(par) if src == "ema" else 0.0})
    actor0 = base.network.params["modules_actor"]
    ref0 = (jax.tree_util.tree_map(lambda x: jnp.stack([x] * RING_M), actor0) if src == "ring"
            else actor0 if src != "current" else None)
    agent = GenAgent(rng=base.rng, network=base.network, config=conf, ref=ref0)

    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    from utils.datasets import ReplayBuffer
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    buf = ReplayBuffer.create_from_initial_dataset(dict(train), size=max(cfg["buffer_size"], train.size + 1))
    data = {k: jnp.asarray(v) for k, v in buf.items()}
    dsize = buf.size
    fused, bs = cfg["n_steps_fused"], cfg["agent"]["batch_size"]
    assert not cfg.get("balanced_sampling", False) and cfg["dataset_on_gpu"]

    # evaluation probe (same as experiments B/C/D; start-region labels used only for the window-rule metrics)
    probe = np.load(probe_npz)
    p_obs, p_z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    # label-free gate probe
    _, g_obs, g_z = make_gate_probe(np.asarray(train["observations"]))
    g_obs, g_z = jnp.asarray(g_obs), jnp.asarray(g_z)

    @jax.jit
    def probe_fn(ag):
        a = jnp.clip(ag.network.select("actor")(p_obs, p_z), -1, 1)
        return a, ag.network.select("critic")(p_obs, actions=a).mean(0), ag.critic_eval(p_obs, a)

    @jax.jit
    def dqda(ag, a):
        return jax.grad(lambda aa: ag.network.select("critic")(p_obs, actions=aa).mean(0).sum())(a)

    @jax.jit
    def gate_v(ag):
        if src == "ring":
            vs = jax.vmap(lambda pa: ag.critic_eval(g_obs, jnp.clip(ag._actor_with(pa, g_obs, g_z), -1, 1)))(ag.ref)
            return vs.mean(0)
        if src == "current":
            return ag.critic_eval(g_obs, jnp.clip(ag.network.select("actor")(g_obs, g_z), -1, 1))
        return ag.critic_eval(g_obs, jnp.clip(ag._actor_with(ag.ref, g_obs, g_z), -1, 1))

    @jax.jit
    def v_cur(ag):
        return ag.critic_eval(g_obs, jnp.clip(ag.network.select("actor")(g_obs, g_z), -1, 1))

    def swap(ag):
        a = ag.network.params["modules_actor"]
        if src == "ring":
            return ag.replace(ref=jax.tree_util.tree_map(lambda r, x: jnp.concatenate([x[None], r[:-1]], 0), ag.ref, a))
        return ag.replace(ref=a)

    gate = dict(t_s=START, buf={START: np.asarray(gate_v(agent))}, n_brake=0, swaps=[], d_last=None)
    log, glog, t0, it, a0, g0 = [], [], time.time(), 0, None, None
    info = {}
    while it <= n_updates:
        step = START + it
        if it > 0 and it % CHECK_EVERY == 0 and src in ("ref", "ring"):
            since = step - gate["t_s"]
            do = False
            if ARM == "E":
                v = np.asarray(gate_v(agent))
                gate["buf"][step] = v
                gate["buf"] = {k: x for k, x in gate["buf"].items() if k >= step - W}
                d = float(np.quantile(np.abs(v - gate["buf"][step - W]), Q)) if (step - W) in gate["buf"] else float("inf")
                gate["d_last"] = d
                if since >= K_MAX or (since >= K_MIN and d <= EPS):
                    do, why = True, ("forced" if since >= K_MAX else "gate")
                elif since >= K_MIN:
                    gate["n_brake"] += 1
                glog.append(dict(step=step, d=d, since=since))
            elif ARM in ("P10", "P25"):
                do, why = since >= par, "period"
            elif ARM == "R":
                do, why = since >= RING_DELTA, "ring_push"
            if do:
                agent = swap(agent)
                gate["swaps"].append(dict(step=step, d=gate["d_last"], why=why))
                gate["t_s"] = step
                gate["buf"] = {step: np.asarray(gate_v(agent))}
                print(ARM, "SWAP", json.dumps(gate["swaps"][-1]), flush=True)
        if it % PROBE_EVERY == 0:
            a, q_on, q_tg = map(np.asarray, probe_fn(agent))
            if a0 is None:
                a0 = a
                g0 = np.asarray(dqda(agent, jnp.asarray(a0)))
            g = np.asarray(dqda(agent, jnp.asarray(a0)))
            cos = (g * g0).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(g0, axis=1) + 1e-12)
            vg, vc = np.asarray(gate_v(agent)), np.asarray(v_cur(agent))
            row = dict(step=step, V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                       V_online_all=float(q_on.mean()), lam=float(1 / np.abs(q_on).mean()),
                       dqda_cos_vs_fork_start=float(cos[reg].mean()), dqda_norm_start=float(np.linalg.norm(g, axis=1)[reg].mean()),
                       action_shift_start=float(np.linalg.norm(a - a0, axis=1)[reg].mean()),
                       action_shift_all=float(np.linalg.norm(a - a0, axis=1).mean()),
                       value_lag_gap=float((vc - vg).mean()), n_swaps=len(gate["swaps"]), n_brake=gate["n_brake"],
                       gate_d=gate["d_last"], sec=time.time() - t0)
            if it > 0:
                row.update({k: float(v) for k, v in info.items() if k.startswith(("critic/", "actor/"))})
            log.append(row)
            print(ARM, json.dumps(row), flush=True)
        if it > 0 and it % SAVE_EVERY == 0:
            plain = GFPAgent(rng=agent.rng, network=agent.network, config=base.config)
            with open(f"{out_dir}/params_{step}.pkl", "wb") as f:
                pickle.dump(dict(agent=flax.serialization.to_state_dict(plain), branch=ARM, run=RUN, start=path0,
                                 ref=jax.device_get(agent.ref), gate=dict(t_s=gate["t_s"], swaps=gate["swaps"],
                                 n_brake=gate["n_brake"], buf={k: v for k, v in gate["buf"].items()})), f)
        if it == n_updates:
            break
        agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
        it += fused
    json.dump(dict(arm=ARM, run=RUN, start=path0, n_updates=n_updates, eps=EPS, K_MIN=K_MIN, K_MAX=K_MAX, W=W, Q=Q,
                   swaps=gate["swaps"], n_brake=gate["n_brake"], gate_log=glog, log=log),
              open(f"{out_dir}/train_{ARM}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
