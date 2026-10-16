"""Experiment A: does the later critic teach the same actor worse?  (actor-only continuation, diagnostic)

usage: python exp_a.py PROBE_NPZ OUT_DIR BRANCH [N_UPDATES]
Run with cwd = official GFP source (source/gfp).

Start: G_task1_s0 at 600k -- actor params, the actor's slice of the Adam state (mu, nu, count), VaBC.
Frozen: VaBC (600k) and the chosen online critic. Only the actor is updated, with the official actor
objective restricted to the terms that reach the actor (GFPAgent.actor_loss):
    L = alpha * mean((actor(s,z) - VaBC_600(s,z))^2)  +  q_term
    q_term = -lambda * mean_i Q_online(s_i, clip(actor(s_i, z_i))),  lambda = stop_grad(1 / mean|Q_lam(s, clip(actor))|)
Branches:
    A0  Q = online critic 600k, lambda from online critic 600k
    A1  Q = online critic 700k, lambda from online critic 700k
    A2  Q = online critic 700k, lambda from online critic 600k (old normalization rule)
    A3  no Q term (distillation to the fixed VaBC only)
Identical across branches: initial actor and Adam state, batch indices, noise z (one fixed key sequence).
Saves the actor at 10k and 20k into OUT_DIR (new directory; nothing under runs/ is touched).
"""
import os
import sys
import json
import time
import pickle

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax

from common import load_agent, xy_to_ij

RUN = "G_task1_s0"
STEP_OLD, STEP_NEW = 600000, 700000
SAVE_AT = (10000, 20000)
LOG_EVERY = 1000
DATA_SEED = 20261016
START_REGION = [(1, 1), (2, 1), (1, 2), (2, 2)]
BRANCHES = {"A0": ("old", "old"), "A1": ("new", "new"), "A2": ("new", "old"), "A3": (None, None)}


def main():
    probe_npz, out_dir, branch = sys.argv[1], sys.argv[2], sys.argv[3]
    n_updates = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    assert branch in BRANCHES and n_updates <= 20000
    os.makedirs(out_dir, exist_ok=True)
    q_src, lam_src = BRANCHES[branch]

    old, cfg, path_old = load_agent(RUN, STEP_OLD)
    new, _, path_new = load_agent(RUN, STEP_NEW)
    critics = {"old": old.network.params["modules_critic"], "new": new.network.params["modules_critic"]}
    alpha, lr, bs = old.config["alpha"], cfg["agent"]["lr"], cfg["agent"]["batch_size"]
    assert old.config["q_agg"] == "mean" and old.config["normalize_q_loss"]

    # official processed dataset (observations only are needed by the actor objective)
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, cfg["env_name"])
    obs_all = jnp.asarray(train["observations"])
    n_data = obs_all.shape[0]

    # actor slice of the saved Adam state: opt_state = (ScaleByAdamState(count, mu, nu), EmptyState())
    adam_state = old.network.opt_state[0]
    tx = optax.adam(lr)
    actor_p = old.network.params["modules_actor"]
    opt_state = tx.init(actor_p)
    opt_state = (opt_state[0]._replace(count=adam_state.count, mu=adam_state.mu["modules_actor"],
                                       nu=adam_state.nu["modules_actor"]),) + tuple(opt_state[1:])

    pa_dim = int(old.config["action_dim"])
    net = old.network
    base_params = dict(net.params)

    def actor_fwd(pa, o, z):
        p = dict(base_params); p["modules_actor"] = pa
        return net.select("actor")(o, z, params=p)

    def q_on(critic_p, o, a):
        p = dict(base_params); p["modules_critic"] = critic_p
        return net.select("critic")(o, actions=a, params=p).mean(0)

    def losses(pa, o, z):
        a_raw = actor_fwd(pa, o, z)
        target = old.compute_flow_actions(o, noises=z)  # frozen VaBC 600k
        distill = jnp.mean((a_raw - target) ** 2)
        a = jnp.clip(a_raw, -1, 1)
        if q_src is None:
            q_mean = q_on(critics["old"], o, a).mean()
            qterm, lam = 0.0 * jax.lax.stop_gradient(q_mean), jnp.nan
        else:
            q = q_on(critics[q_src], o, a)
            lam = jax.lax.stop_gradient(1 / jnp.abs(q_on(critics[lam_src], o, a)).mean())
            qterm, q_mean = -lam * q.mean(), q.mean()
        return alpha * distill, qterm, dict(distill=distill, q_mean=q_mean, lam=lam)

    def total(pa, o, z):
        d, q, info = losses(pa, o, z)
        return d + q, info

    @jax.jit
    def step(pa, st, key, data):
        key, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (bs,), 0, n_data)
        o = data[idx]
        z = jax.random.normal(k2, (bs, pa_dim))
        (loss, info), g = jax.value_and_grad(total, has_aux=True)(pa, o, z)
        upd, st = tx.update(g, st, pa)
        return optax.apply_updates(pa, upd), st, key, info

    @jax.jit
    def term_grads(pa, key, data):
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (bs,), 0, n_data)
        o = data[idx]
        z = jax.random.normal(k2, (bs, pa_dim))
        gd = jax.grad(lambda p: losses(p, o, z)[0])(pa)
        gq = jax.grad(lambda p: losses(p, o, z)[1])(pa)
        flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
        gd, gq = flat(gd), flat(gq)
        return jnp.linalg.norm(gd), jnp.linalg.norm(gq), jnp.dot(gd, gq) / (jnp.linalg.norm(gd) * jnp.linalg.norm(gq) + 1e-12)

    # fixed probe inputs for the action-change metric
    probe = np.load(probe_npz)
    sel = np.arange(0, len(probe["obs"]), 10)
    p_obs, p_z = jnp.asarray(probe["obs"][sel]), jnp.asarray(probe["z"][sel])
    ci, cj = probe["cell_i"][sel], probe["cell_j"][sel]
    reg = np.zeros(len(sel), bool)
    for i, j in START_REGION:
        reg |= (ci == i) & (cj == j)
    act0 = np.asarray(jnp.clip(actor_fwd(actor_p, p_obs, p_z), -1, 1))
    v_fn = jax.jit(lambda pa, cp: q_on(cp, p_obs, jnp.clip(actor_fwd(pa, p_obs, p_z), -1, 1)))

    key = jax.random.PRNGKey(DATA_SEED)
    gkey = jax.random.PRNGKey(DATA_SEED + 1)
    pa, st = actor_p, opt_state
    log, t0 = [], time.time()
    for it in range(1, n_updates + 1):
        pa, st, key, info = step(pa, st, key, obs_all)
        if it % LOG_EVERY == 0 or it == 1:
            gkey, gk = jax.random.split(gkey)
            gd, gq, cs = term_grads(pa, gk, obs_all)
            a = np.asarray(jnp.clip(actor_fwd(pa, p_obs, p_z), -1, 1))
            sh = np.linalg.norm(a - act0, axis=1)
            v_old, v_new = np.asarray(v_fn(pa, critics["old"])), np.asarray(v_fn(pa, critics["new"]))
            row = dict(it=it, distill=float(info["distill"]), q_mean=float(info["q_mean"]), lam=float(info["lam"]),
                       grad_distill=float(gd), grad_q=float(gq), cos_distill_q=float(cs),
                       shift_all=float(sh.mean()), shift_start=float(sh[reg].mean()),
                       V_oldcritic_start=float(v_old[reg].mean()), V_newcritic_start=float(v_new[reg].mean()),
                       V_oldcritic_all=float(v_old.mean()), V_newcritic_all=float(v_new.mean()),
                       sec=time.time() - t0)
            log.append(row)
            print(branch, json.dumps(row), flush=True)
        if it in SAVE_AT:
            with open(f"{out_dir}/actor_{branch}_{it}.pkl", "wb") as f:
                pickle.dump(dict(actor=jax.device_get(pa), branch=branch, it=it, start=path_old,
                                 q_critic={"old": path_old, "new": path_new}.get(q_src),
                                 lam_critic={"old": path_old, "new": path_new}.get(lam_src)), f)
    json.dump(dict(branch=branch, q_src=q_src, lam_src=lam_src, start=path_old, new=path_new, n_updates=n_updates,
                   data_seed=DATA_SEED, lr=lr, batch_size=bs, alpha=alpha, log=log),
              open(f"{out_dir}/train_{branch}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
