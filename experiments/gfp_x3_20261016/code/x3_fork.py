"""X3: fork-and-continue with the actor's direct Q term removed (only the actor loss changes).

usage: python x3_fork.py PROBE_NPZ OUT_DIR RUN START N_UPDATES          (training; needs the user's approval)
       python x3_fork.py --check PROBE_NPZ RUN START                    (one batch, forward/grad only, no update)
Run with cwd = official GFP source (source/gfp, 1abbd5f, unmodified). The subclass below overrides only
GFPAgent.actor_loss:

  official  actor_loss = VaBC_loss(guidance) + alpha * distill_loss + lam * (-Q_online(s, clip(mu(s, z))))
  X3        actor_loss = VaBC_loss(guidance) + alpha * distill_loss

Everything else is the official code path: critic loss and TD target (current actor + VaBC, target_agg=mean,
same z), target-critic EMA, guidance (still computed from the target critic, the stop-gradient actor action,
the VaBC action and lam = 1/mean|Q|), VaBC loss, distillation, optimizer, batch sampling. q_loss is still
computed and logged, but it is not added to the loss. The deployed policy is the current actor.

Probe rows every 5k (same as egra_fork.py, plus actor-motion metrics, see x3_motion.py):
  V_online_start/all, dqda norm/cos vs fork, action_shift_start/all (vs fork),
  shift_50k (vs the actor 50k earlier), gap_vabc, vabc_shift, qadv, training losses.
Writes only into OUT_DIR.
"""
import os
import sys
import json
import time
import pickle
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")

import numpy as np
import jax
import jax.numpy as jnp
import flax

from common import load_agent

PROBE_EVERY, SAVE_EVERY = 5000, 10000
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]


def make_classes():
    from agents.gfp import GFPAgent

    class X3Agent(GFPAgent):
        def actor_loss(self, batch, grad_params, rng):
            # identical to GFPAgent.actor_loss (1abbd5f) except that q_loss is not added to the total
            batch_size, action_dim = batch["actions"].shape
            rng, distill_rng, flow_x_rng, flow_t_rng = jax.random.split(rng, 4)

            noises = jax.random.normal(distill_rng, (batch_size, action_dim))
            target_vabc_acts = self.compute_flow_actions(batch["observations"], noises=noises)
            actor_acts = self.network.select("actor")(batch["observations"], noises, params=grad_params)
            distill_loss = jnp.mean((actor_acts - target_vabc_acts) ** 2)

            actor_acts = jnp.clip(actor_acts, -1, 1)
            qs = self.network.select("critic")(batch["observations"], actions=actor_acts)
            q = getattr(jnp, self.config["q_agg"])(qs, axis=0)
            q_loss = -q.mean()
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            if self.config["normalize_q_loss"]:
                q_loss = lam * q_loss

            x_0 = jax.random.normal(flow_x_rng, (batch_size, action_dim))
            x_1 = batch["actions"]
            t = jax.random.uniform(flow_t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0
            pred = self.network.select("vabc")(batch["observations"], x_t, t, params=grad_params)
            flow_error = jnp.mean((pred - vel) ** 2, axis=1)
            if self.config["eta_temperature"] == 0:
                VaBC_loss = jnp.mean(flow_error)
                g_info = dict()
            else:
                guidance, g_info = self.compute_guidance(
                    batch, jax.lax.stop_gradient(actor_acts), target_vabc_acts, lam
                )
                VaBC_loss = jnp.mean(guidance * flow_error)

            actor_loss = VaBC_loss + self.config["alpha"] * distill_loss  # X3: no "+ q_loss"
            if self.config["log_metrics"]:
                info = dict(actor_loss=actor_loss, VaBC_loss=VaBC_loss, distill_loss=distill_loss,
                            q_loss=jax.lax.stop_gradient(q_loss), **g_info)
            else:
                info = dict()
            return actor_loss, info

    return GFPAgent, X3Agent


def load_probe(probe_npz):
    probe = np.load(probe_npz)
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    return jnp.asarray(probe["obs"]), jnp.asarray(probe["z"]), reg


def sample_batch(RUN, base, cfg):
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    from utils.datasets import ReplayBuffer
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    buf = ReplayBuffer.create_from_initial_dataset(dict(train), size=max(cfg["buffer_size"], train.size + 1))
    return buf


def check(probe_npz, RUN, START):
    """Forward/grad equivalence on one batch; no parameter update is applied or saved."""
    GFPAgent, X3Agent = make_classes()
    base, cfg, _ = load_agent(RUN, START)
    base = base.replace(config=base.config.copy({"log_metrics": True}))  # as egra_fork.py: the key is not in the saved config
    x3 = X3Agent(rng=base.rng, network=base.network, config=base.config)
    buf = sample_batch(RUN, base, cfg)
    rs = np.random.RandomState(0)
    idx = rs.randint(0, buf.size, cfg["agent"]["batch_size"])
    batch = {k: jnp.asarray(v[idx]) for k, v in buf.items()}
    rng = jax.random.PRNGKey(123)
    p = base.network.params

    def off(gp):
        return GFPAgent.actor_loss(base, batch, gp, rng)

    def new(gp):
        return X3Agent.actor_loss(x3, batch, gp, rng)

    (lo, io), go = jax.value_and_grad(off, has_aux=True)(p)
    (ln, in_), gn = jax.value_and_grad(new, has_aux=True)(p)
    co, _ = GFPAgent.critic_loss(base, batch, p, rng)
    cn, _ = X3Agent.critic_loss(x3, batch, p, rng)
    gnorm = lambda g: float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))))
    diff = jax.tree_util.tree_map(lambda a, b: a - b, go, gn)
    res = dict(
        loss_off=float(lo), loss_x3=float(ln), q_loss=float(io["q_loss"]),
        loss_identity_err=float(abs(lo - io["q_loss"] - ln)),
        same_vabc_distill=bool(np.allclose(io["VaBC_loss"], in_["VaBC_loss"]) and np.allclose(io["distill_loss"], in_["distill_loss"])),
        critic_loss_equal=bool(np.allclose(co, cn)),
        grad_diff_actor=gnorm(diff["modules_actor"]), grad_actor_off=gnorm(go["modules_actor"]), grad_actor_x3=gnorm(gn["modules_actor"]),
        grad_diff_vabc=gnorm(diff["modules_vabc"]), grad_diff_critic=gnorm(diff["modules_critic"]),
    )
    print(json.dumps(res, indent=1), flush=True)
    assert res["loss_identity_err"] < 1e-4 and res["same_vabc_distill"] and res["critic_loss_equal"]
    assert res["grad_diff_vabc"] < 1e-6 and res["grad_diff_critic"] < 1e-6 and res["grad_diff_actor"] > 0
    print("CHECK OK", flush=True)


def main():
    if sys.argv[1] == "--check":
        return check(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    probe_npz, out_dir, RUN, START, n_updates = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    assert n_updates <= 200000 and n_updates % 4 == 0
    os.makedirs(out_dir, exist_ok=True)
    GFPAgent, X3Agent = make_classes()
    base, cfg, path0 = load_agent(RUN, START)
    base = base.replace(config=base.config.copy({"log_metrics": bool(cfg["log_metrics"])}))  # as egra_fork.py
    assert base.config["eta_temperature"] != 0 and not base.config["target_use_next"] and base.config["target_agg"] == "mean"
    agent = X3Agent(rng=base.rng, network=base.network, config=base.config)

    buf = sample_batch(RUN, base, cfg)
    data = {k: jnp.asarray(v) for k, v in buf.items()}
    dsize = buf.size
    fused, bs = cfg["n_steps_fused"], cfg["agent"]["batch_size"]
    assert not cfg.get("balanced_sampling", False) and cfg["dataset_on_gpu"]

    p_obs, p_z, reg = load_probe(probe_npz)
    agg = getattr(jnp, base.config["q_agg"])

    @jax.jit
    def probe_fn(ag):
        a = jnp.clip(ag.network.select("actor")(p_obs, p_z), -1, 1)
        f = jnp.clip(ag.compute_flow_actions(p_obs, noises=p_z), -1, 1)
        qa_all = ag.network.select("critic")(p_obs, actions=a)
        qf = agg(ag.network.select("critic")(p_obs, actions=f), axis=0)
        return a, f, qa_all.mean(0), ag.critic_eval(p_obs, a), agg(qa_all, axis=0) - qf

    @jax.jit
    def dqda(ag, a):
        return jax.grad(lambda aa: ag.network.select("critic")(p_obs, actions=aa).mean(0).sum())(a)

    def st(x):
        return float(x.mean()), float(x[reg].mean())

    log, t0, it, a0, f0, g0, hist, info = [], time.time(), 0, None, None, None, {}, {}
    while it <= n_updates:
        step = START + it
        if it % PROBE_EVERY == 0:
            a, f, q_on, q_tg, qd = map(np.asarray, probe_fn(agent))
            if a0 is None:
                a0, f0 = a, f
                g0 = np.asarray(dqda(agent, jnp.asarray(a0)))
            hist[step] = a
            hist = {k: v for k, v in hist.items() if k >= step - 50000}
            g = np.asarray(dqda(agent, jnp.asarray(a0)))
            cos = (g * g0).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(g0, axis=1) + 1e-12)
            sh_all, sh_start = st(np.linalg.norm(a - a0, axis=1))
            row = dict(step=step, V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                       V_online_all=float(q_on.mean()), lam=float(1 / np.abs(q_on).mean()),
                       dqda_cos_vs_fork_start=float(cos[reg].mean()), dqda_norm_start=float(np.linalg.norm(g, axis=1)[reg].mean()),
                       action_shift_start=sh_start, action_shift_all=sh_all,
                       gap_vabc=st(np.linalg.norm(a - f, axis=1)), vabc_shift=st(np.linalg.norm(f - f0, axis=1)),
                       qadv=st(qd), sec=time.time() - t0)
            if step - 50000 in hist:
                row["shift_50k"] = st(np.linalg.norm(a - hist[step - 50000], axis=1))
            if it > 0:
                row.update({k: float(v) for k, v in info.items() if k.startswith(("critic/", "actor/"))})
            log.append(row)
            print("X3", json.dumps(row), flush=True)
        if it > 0 and it % SAVE_EVERY == 0:
            plain = GFPAgent(rng=agent.rng, network=agent.network, config=base.config)
            with open(f"{out_dir}/params_{step}.pkl", "wb") as fh:
                pickle.dump(dict(agent=flax.serialization.to_state_dict(plain), branch="X3", run=RUN, start=path0), fh)
        if it == n_updates:
            break
        agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
        it += fused
    json.dump(dict(arm="X3", run=RUN, start=path0, n_updates=n_updates, log=log), open(f"{out_dir}/train_X3.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
