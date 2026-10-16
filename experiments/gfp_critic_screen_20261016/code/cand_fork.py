"""Step 2 screening: fork-and-continue with one candidate critic change (development screening only).

usage: python cand_fork.py train PROBE_NPZ OUT_DIR RUN START N_UPDATES ARM PARAM
       python cand_fork.py check RUN START ARM PARAM          (one batch, forward/grad only, nothing saved)
       python cand_fork.py time  RUN START ARM PARAM N        (N updates for timing only, nothing saved)
ARM / PARAM:
  IQL   expectile    "IQL-inspired GFP": the critic's TD target uses a state value V(s') learned by expectile
                     regression on the target critic at dataset actions (IQL), instead of the current actor's and
                     VaBC's actions. V is a new network (same MLP as one critic head, own Adam, lr as the agent).
                     Before the fork continues, V alone is fitted for 50k updates with every agent parameter
                     frozen (extended in 10k chunks up to 100k until |mean(V - Q_data)| <= 10; else the arm is invalid).
                     The actor loss (VaBC guidance + alpha * distill + lambda * (-Q)) is unchanged.
  CHAIN beta         CHAIN-style value churn reduction on the critic (Tang & Berseth 2024): official critic loss
                     + lam_c * mean_k[(Q_k(s~, a~) - Q_k^prev(s~, a~))^2] on a separately sampled reference batch,
                     Q^prev = the critic parameters before the previous update. Reference pairs are the dataset
                     action and the current actor's action (stop-gradient) at each reference state, because the
                     value island is measured at actor actions. lam_c = beta * EMA(td loss) / EMA(churn loss)
                     (bias-corrected EMAs, decay 0.99), i.e. the churn term is kept at a fixed fraction of the TD
                     loss. Actor loss unchanged.
  CQL   alpha        CQL(H)-style conservatism on the critic (Kumar et al. 2020), simplified: official critic loss
                     + alpha * mean_k[logsumexp_j Q_k(s, a_j) - log 8 - Q_k(s, a_data)] over 8 actions per state
                     (4 uniform in [-1, 1], 4 current-actor actions with fresh noise, stop-gradient), no importance
                     weights. Actor loss unchanged. (Added after the I1/C1/C2 round, CARD_SCREEN2_20261016.md.)
  OFF   -            official GFP update (timing reference only).
Run with cwd = official GFP source (source/gfp, 1abbd5f, unmodified). Probe rows every 5k, same metrics as
x3_fork.py. Writes only into OUT_DIR.
"""
import os
import sys
import json
import time
import pickle
from functools import partial
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")

import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax

from common import load_agent

PROBE_EVERY, SAVE_EVERY = 5000, 10000
V_WARMUP, V_WARMUP_MAX, V_GATE = 50000, 100000, 10.0   # V-only warm-up; gate |mean(V - Q_data)| <= V_GATE
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]
EMA_DECAY = 0.99


def make_classes():
    from agents.gfp import GFPAgent
    from utils.datasets import GPU_dataset_sample
    from utils.flax_utils import ModuleDict, TrainState
    from utils.networks import Value

    class IQLAgent(GFPAgent):
        value: Any = None                       # TrainState of ModuleDict(value=Value(num_ensembles=1))

        def value_loss(self, batch, vparams):
            q = jax.lax.stop_gradient(self.critic_eval(batch["observations"], batch["actions"]))
            v = self.value.select("value")(batch["observations"], params=vparams)
            diff = q - v
            w = jnp.where(diff > 0, self.config["expectile"], 1 - self.config["expectile"])
            loss = jnp.mean(w * diff ** 2)
            return loss, dict(value_loss=loss, v_mean=v.mean(), q_data_mean=q.mean(), v_minus_q=(v - q).mean())

        def critic_loss(self, batch, grad_params, rng):
            # IQL target: no actor / VaBC action in the TD target
            next_v = self.value.select("value")(batch["next_observations"])
            target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v
            q = self.network.select("critic")(batch["observations"], actions=batch["actions"], params=grad_params)
            critic_loss = jnp.square(q - target_q).mean()
            info = dict(critic_loss=critic_loss, q_mean=q.mean(), tgt_q_mean=next_v.mean(), q_max=q.max(), q_min=q.min())
            return critic_loss, info

        def update_value(self, batch):
            vstate, vinfo = self.value.apply_loss_fn(loss_fn=lambda vp: self.value_loss(batch, vp))
            return self.replace(value=vstate), vinfo

        @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
        def warmup_value(self, n_steps_fused, dataset, dataset_size, batch_size):
            _agent, info = self, {}
            for _ in range(n_steps_fused):
                batch, rng = GPU_dataset_sample(_agent.rng, dataset, batch_size, dataset_size)
                _agent, info = _agent.update_value(batch)
                _agent = _agent.replace(rng=rng)
            return _agent, info

        @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
        def multi_sample_and_update(self, n_steps_fused, dataset, dataset_size, batch_size):
            _agent, info = self, {}
            for _ in range(n_steps_fused):
                batch, rng = GPU_dataset_sample(_agent.rng, dataset, batch_size, dataset_size)
                rng, loss_rng = jax.random.split(rng)
                _agent, vinfo = _agent.update_value(batch)       # IQL order: V, then Q and actor

                def loss_fn(grad_params):
                    return _agent.total_loss(batch, grad_params, rng=loss_rng)

                new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
                _agent.target_update(new_network, "critic")
                _agent = _agent.replace(network=new_network, rng=rng)
                info = dict(info, **{f"value/{k}": v for k, v in vinfo.items()})
            return _agent, info

    class ChainAgent(GFPAgent):
        prev_critic: Any = None                 # critic params before the previous update
        ema: Any = None                         # dict(td, churn, t)

        def critic_loss(self, batch, grad_params, rng):
            td_loss, info = GFPAgent.critic_loss(self, batch, grad_params, rng)
            ro, ra, rz = batch["ref_observations"], batch["ref_actions"], batch["ref_noises"]
            a_act = jax.lax.stop_gradient(jnp.clip(self.network.select("actor")(ro, rz), -1, 1))
            prev = dict(self.network.params, modules_critic=self.prev_critic)
            churn = 0.0
            for a in (ra, a_act):
                q_cur = self.network.select("critic")(ro, actions=a, params=grad_params)
                q_prev = jax.lax.stop_gradient(self.network.select("critic")(ro, actions=a, params=prev))
                churn = churn + jnp.mean((q_cur - q_prev) ** 2)
            # t = updates done so far; the td EMA has t samples, the churn EMA t-1 (churn is 0 on the first update)
            t = self.ema["t"]
            td_hat = self.ema["td"] / (1 - EMA_DECAY ** jnp.maximum(t, 1.0))
            ch_hat = self.ema["churn"] / (1 - EMA_DECAY ** jnp.maximum(t - 1, 1.0))
            lam_c = jnp.where(t > 1, self.config["chain_beta"] * td_hat / jnp.maximum(ch_hat, 1e-8), 0.0)
            lam_c = jax.lax.stop_gradient(lam_c)
            info = dict(info, td_loss=td_loss, churn_loss=churn, chain_lam=lam_c)
            return td_loss + lam_c * churn, info

        @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
        def multi_sample_and_update(self, n_steps_fused, dataset, dataset_size, batch_size):
            _agent, info = self, {}
            for _ in range(n_steps_fused):
                batch, rng = GPU_dataset_sample(_agent.rng, dataset, batch_size, dataset_size)
                ref, rng = GPU_dataset_sample(rng, dataset, batch_size, dataset_size)
                rng, loss_rng, z_rng = jax.random.split(rng, 3)
                batch = dict(batch, ref_observations=ref["observations"], ref_actions=ref["actions"],
                             ref_noises=jax.random.normal(z_rng, ref["actions"].shape))

                def loss_fn(grad_params):
                    return _agent.total_loss(batch, grad_params, rng=loss_rng)

                old = _agent.network.params["modules_critic"]
                new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
                _agent.target_update(new_network, "critic")
                e = _agent.ema
                ema = dict(td=EMA_DECAY * e["td"] + (1 - EMA_DECAY) * info["critic/td_loss"],
                           churn=jnp.where(e["t"] > 0, EMA_DECAY * e["churn"] + (1 - EMA_DECAY) * info["critic/churn_loss"],
                                           e["churn"]),
                           t=e["t"] + 1.0)
                # churn is identically 0 on the first update (prev = current), so its EMA starts one update later
                _agent = _agent.replace(network=new_network, rng=rng, prev_critic=old, ema=ema)
            return _agent, info

    class CQLAgent(GFPAgent):
        def critic_loss(self, batch, grad_params, rng):
            td_loss, info = GFPAgent.critic_loss(self, batch, grad_params, rng)
            obs = batch["observations"]
            u_rng, z_rng = jax.random.split(jax.random.fold_in(rng, 7))
            shape = (4,) + batch["actions"].shape
            a_u = jax.random.uniform(u_rng, shape, minval=-1.0, maxval=1.0)
            z = jax.random.normal(z_rng, shape)
            a_pi = jax.lax.stop_gradient(jax.vmap(lambda zz: jnp.clip(self.network.select("actor")(obs, zz), -1, 1))(z))
            qf = lambda a: self.network.select("critic")(obs, actions=a, params=grad_params)
            q_cat = jnp.concatenate([jax.vmap(qf)(a_u), jax.vmap(qf)(a_pi)], axis=0)       # (8, heads, B)
            lse = jax.scipy.special.logsumexp(q_cat, axis=0) - jnp.log(8.0)
            q_d = qf(batch["actions"])
            pen = jnp.mean(lse - q_d)
            info = dict(info, td_loss=td_loss, cql_pen=pen, q_unif=jnp.mean(q_cat[:4]), q_pi=jnp.mean(q_cat[4:]))
            return td_loss + self.config["cql_alpha"] * pen, info

    return GFPAgent, IQLAgent, ChainAgent, CQLAgent, ModuleDict, TrainState, Value


def build(RUN, START, ARM, PARAM):
    GFPAgent, IQLAgent, ChainAgent, CQLAgent, ModuleDict, TrainState, Value = make_classes()
    base, cfg, path0 = load_agent(RUN, START)
    conf = base.config.copy({"log_metrics": True})
    assert conf["eta_temperature"] != 0 and not conf["target_use_next"] and conf["target_agg"] == "mean"
    if ARM == "IQL":
        conf = conf.copy({"expectile": float(PARAM)})
        vdef = ModuleDict(dict(value=Value(hidden_dims=conf["value_hidden_dims"], layer_norm=conf["layer_norm"],
                                           num_ensembles=1)))
        vparams = vdef.init(jax.random.PRNGKey(20261016), value=(np.zeros((1, 69), np.float32),))["params"]
        vstate = TrainState.create(vdef, vparams, tx=optax.adam(learning_rate=conf["lr"]))
        agent = IQLAgent(rng=base.rng, network=base.network, config=conf, value=vstate)
    elif ARM == "CHAIN":
        conf = conf.copy({"chain_beta": float(PARAM)})
        agent = ChainAgent(rng=base.rng, network=base.network, config=conf,
                           prev_critic=base.network.params["modules_critic"],
                           ema=dict(td=jnp.float32(0), churn=jnp.float32(0), t=jnp.float32(0)))
    elif ARM == "CQL":
        conf = conf.copy({"cql_alpha": float(PARAM)})
        agent = CQLAgent(rng=base.rng, network=base.network, config=conf)
    else:
        agent = GFPAgent(rng=base.rng, network=base.network, config=conf)
    return GFPAgent, base, cfg, path0, agent


def dataset(cfg):
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    from utils.datasets import ReplayBuffer
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    buf = ReplayBuffer.create_from_initial_dataset(dict(train), size=max(cfg["buffer_size"], train.size + 1))
    assert not cfg.get("balanced_sampling", False) and cfg["dataset_on_gpu"]
    return {k: jnp.asarray(v) for k, v in buf.items()}, buf.size


def check(RUN, START, ARM, PARAM):
    """One batch: the actor loss and VaBC gradients equal the official ones; only the critic loss changes."""
    GFPAgent, base, cfg, _, agent = build(RUN, START, ARM, PARAM)
    data, dsize = dataset(cfg)
    rs = np.random.RandomState(0)
    idx = rs.randint(0, dsize, cfg["agent"]["batch_size"])
    ridx = rs.randint(0, dsize, cfg["agent"]["batch_size"])
    batch = {k: v[idx] for k, v in data.items()}
    if ARM == "CHAIN":
        batch = dict(batch, ref_observations=data["observations"][ridx], ref_actions=data["actions"][ridx],
                     ref_noises=jax.random.normal(jax.random.PRNGKey(7), data["actions"][ridx].shape))
    rng = jax.random.PRNGKey(123)
    p = base.network.params
    off = GFPAgent(rng=base.rng, network=base.network, config=agent.config)
    (ao, _), gao = jax.value_and_grad(lambda gp: GFPAgent.actor_loss(off, batch, gp, rng), has_aux=True)(p)
    (an, _), gan = jax.value_and_grad(lambda gp: type(agent).actor_loss(agent, batch, gp, rng), has_aux=True)(p)
    (co, _), gco = jax.value_and_grad(lambda gp: GFPAgent.critic_loss(off, batch, gp, rng), has_aux=True)(p)
    (cn, ci), gcn = jax.value_and_grad(lambda gp: type(agent).critic_loss(agent, batch, gp, rng), has_aux=True)(p)
    gnorm = lambda g: float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g))))
    sub = lambda a, b: jax.tree_util.tree_map(lambda x, y: x - y, a, b)
    res = dict(arm=ARM, param=PARAM, actor_loss_off=float(ao), actor_loss_new=float(an),
               actor_grad_diff=gnorm(sub(gao, gan)), critic_loss_off=float(co), critic_loss_new=float(cn),
               critic_grad_off=gnorm(gco["modules_critic"]), critic_grad_diff=gnorm(sub(gco, gcn)["modules_critic"]),
               critic_grad_outside_critic=gnorm({k: v for k, v in gcn.items() if k != "modules_critic"}))
    if ARM == "IQL":
        vinfo = agent.value_loss(batch, agent.value.params)[1]
        res.update({k: float(v) for k, v in vinfo.items()})
    if ARM == "CQL":
        res.update(cql_pen=float(ci["cql_pen"]), q_unif=float(ci["q_unif"]), q_pi=float(ci["q_pi"]),
                   td_loss=float(ci["td_loss"]))
    if ARM == "CHAIN":
        res.update(churn_at_fork=float(ci["churn_loss"]), lam_at_fork=float(ci["chain_lam"]))
        a2, info2 = agent.multi_sample_and_update(cfg["n_steps_fused"], data, dsize, cfg["agent"]["batch_size"])
        res.update(after_4_updates=dict(churn=float(info2["critic/churn_loss"]), lam=float(info2["critic/chain_lam"]),
                                        td=float(info2["critic/td_loss"]), ema_t=float(a2.ema["t"])))
    print(json.dumps(res, indent=1), flush=True)
    assert abs(res["actor_loss_off"] - res["actor_loss_new"]) < 1e-5 and res["actor_grad_diff"] < 1e-6
    assert res["critic_grad_outside_critic"] < 1e-6
    if ARM == "CQL":
        assert abs(res["td_loss"] - res["critic_loss_off"]) < 1e-5
        assert abs(res["critic_loss_new"] - res["critic_loss_off"] - float(PARAM) * res["cql_pen"]) < 1e-3
    if ARM == "CHAIN":
        assert abs(res["critic_loss_off"] - res["critic_loss_new"]) < 1e-5 and res["churn_at_fork"] == 0.0
        assert res["after_4_updates"]["churn"] > 0 and res["after_4_updates"]["lam"] > 0
    print("CHECK OK", flush=True)


def time_only(RUN, START, ARM, PARAM, N):
    _, base, cfg, _, agent = build(RUN, START, ARM, PARAM)
    data, dsize = dataset(cfg)
    fused, bs = cfg["n_steps_fused"], cfg["agent"]["batch_size"]
    agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
    jax.block_until_ready(info)
    t0 = time.time()
    for _ in range(N // fused):
        agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
    jax.block_until_ready(info)
    dt = time.time() - t0
    print(json.dumps(dict(arm=ARM, param=PARAM, n=N, sec=dt, ms_per_update=1000 * dt / N)), flush=True)


def train(probe_npz, out_dir, RUN, START, n_updates, ARM, PARAM):
    assert n_updates <= 200000 and n_updates % 4 == 0 and ARM in ("IQL", "CHAIN", "CQL")
    os.makedirs(out_dir, exist_ok=True)
    GFPAgent, base, cfg, path0, agent = build(RUN, START, ARM, PARAM)
    data, dsize = dataset(cfg)
    fused, bs = cfg["n_steps_fused"], cfg["agent"]["batch_size"]

    probe = np.load(probe_npz)
    reg = np.zeros(len(probe["obs"]), bool)
    for i, j in START_REGION:
        reg |= (probe["cell_i"] == i) & (probe["cell_j"] == j)
    p_obs, p_z = jnp.asarray(probe["obs"]), jnp.asarray(probe["z"])
    agg = getattr(jnp, base.config["q_agg"])

    @jax.jit
    def probe_fn(ag):
        a = jnp.clip(ag.network.select("actor")(p_obs, p_z), -1, 1)
        f = jnp.clip(ag.compute_flow_actions(p_obs, noises=p_z), -1, 1)
        qa_all = ag.network.select("critic")(p_obs, actions=a)
        qf = agg(ag.network.select("critic")(p_obs, actions=f), axis=0)
        v = ag.value.select("value")(p_obs) if ARM == "IQL" else qa_all.mean(0)
        return a, f, qa_all.mean(0), ag.critic_eval(p_obs, a), agg(qa_all, axis=0) - qf, v

    @jax.jit
    def dqda(ag, a):
        return jax.grad(lambda aa: ag.network.select("critic")(p_obs, actions=aa).mean(0).sum())(a)

    def st(x):
        return float(x.mean()), float(x[reg].mean())

    t_start = time.time()
    warm = []
    if ARM == "IQL":   # V alone, agent frozen
        w = 0
        while True:
            agent, vinfo = agent.warmup_value(fused, data, dsize, bs)
            w += fused
            if w % 5000 == 0:
                row = dict(w=w, **{k: float(v) for k, v in vinfo.items()}, sec=time.time() - t_start)
                warm.append(row)
                print("WARM", json.dumps(row), flush=True)
                if w >= V_WARMUP and w % 10000 == 0 and abs(row["v_minus_q"]) <= V_GATE:
                    break
                if w >= V_WARMUP_MAX:
                    json.dump(dict(arm=ARM, param=PARAM, run=RUN, warmup=warm, invalid="V warm-up gate failed"),
                              open(f"{out_dir}/train.json", "w"), indent=1)
                    print("INVALID warm-up gate", flush=True)
                    return

    log, t0, it, a0, f0, g0, hist, info = [], time.time(), 0, None, None, None, {}, {}
    while it <= n_updates:
        step = START + it
        if it % PROBE_EVERY == 0:
            a, f, q_on, q_tg, qd, v = map(np.asarray, probe_fn(agent))
            if a0 is None:
                a0, f0 = a, f
                g0 = np.asarray(dqda(agent, jnp.asarray(a0)))
            hist[step] = a
            hist = {k: x for k, x in hist.items() if k >= step - 50000}
            g = np.asarray(dqda(agent, jnp.asarray(a0)))
            cos = (g * g0).sum(1) / (np.linalg.norm(g, axis=1) * np.linalg.norm(g0, axis=1) + 1e-12)
            sh_all, sh_start = st(np.linalg.norm(a - a0, axis=1))
            row = dict(step=step, V_online_start=float(q_on[reg].mean()), V_target_start=float(q_tg[reg].mean()),
                       V_online_all=float(q_on.mean()), lam=float(1 / np.abs(q_on).mean()),
                       dqda_cos_vs_fork_start=float(cos[reg].mean()), dqda_norm_start=float(np.linalg.norm(g, axis=1)[reg].mean()),
                       action_shift_start=sh_start, action_shift_all=sh_all,
                       gap_vabc=st(np.linalg.norm(a - f, axis=1)), vabc_shift=st(np.linalg.norm(f - f0, axis=1)),
                       qadv=st(qd), sec=time.time() - t0)
            if ARM == "IQL":
                row["Vnet"] = st(v)
            if step - 50000 in hist:
                row["shift_50k"] = st(np.linalg.norm(a - hist[step - 50000], axis=1))
            if it > 0:
                row.update({k: float(x) for k, x in info.items() if k.startswith(("critic/", "actor/", "value/"))})
            log.append(row)
            print(ARM, json.dumps(row), flush=True)
        if it > 0 and it % SAVE_EVERY == 0:
            plain = GFPAgent(rng=agent.rng, network=agent.network, config=base.config)
            extra = dict(value=flax.serialization.to_state_dict(agent.value.params)) if ARM == "IQL" else {}
            with open(f"{out_dir}/params_{step}.pkl", "wb") as fh:
                pickle.dump(dict(agent=flax.serialization.to_state_dict(plain), branch=f"{ARM}{PARAM}", run=RUN,
                                 start=path0, **extra), fh)
        if it == n_updates:
            break
        agent, info = agent.multi_sample_and_update(fused, data, dsize, bs)
        it += fused
    json.dump(dict(arm=ARM, param=PARAM, run=RUN, start=path0, n_updates=n_updates, warmup=warm,
                   total_sec=time.time() - t_start, log=log), open(f"{out_dir}/train.json", "w"), indent=1)
    print("DONE", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "check":
        return check(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    if mode == "time":
        return time_only(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]))
    a = sys.argv[2:]
    return train(a[0], a[1], a[2], int(a[3]), int(a[4]), a[5], a[6])


if __name__ == "__main__":
    main()
