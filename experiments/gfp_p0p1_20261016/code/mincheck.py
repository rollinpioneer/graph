"""Minimal checks before experiment A (no training, no parameter change).

usage: python mincheck.py STATES_NPZ OUT_JSON
Run with cwd = official GFP source (source/gfp).

M1  online vs target critic on the same probe states and actions as P0:
    V = Q(s, clip(actor(s, z))) with both critics, P0 decomposition repeated with the online critic.
M2  what the actor's Q term actually sees (online critic, q_agg = mean, as in GFPAgent.actor_loss):
    lambda = 1 / mean|Q_online(s, a_actor)|, the action gradient dQ/da at the same (s, a),
    cosine of that gradient between checkpoints, and the actor-parameter gradient norms of the
    normalized Q term and of the distillation term (alpha * distill) on fixed batches.
M3  TD-target difference on real transitions (s, a, r, s', mask) of the official processed dataset,
    following GFPAgent.critic_loss exactly (target_use_next=False: VaBC acts from s, scored at s';
    actor and VaBC share one noise draw): official target_agg=mean vs target_agg=actor, and the
    paper variant (VaBC acts from s') registered only, not adopted.
"""
import os
import sys
import json

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp

from common import load_agent, run_dir

START_REGION = {"task1": [(1, 1), (2, 1), (1, 2), (2, 2)], "task4": [(6, 5), (6, 6)]}
PAIRS = [("G_task1_s0", 600000, 700000), ("G_task1_s0", 500000, 600000), ("G_task1_s0", 700000, 800000),
         ("G_task1_s1", 600000, 700000), ("F_task1_s0", 600000, 700000)]
M3_SPECS = [("G_task1_s0", 600000), ("G_task1_s0", 700000), ("G_task1_s1", 700000), ("F_task1_s0", 700000),
            ("G_task4_s0", 1000000), ("G_task4_s3", 1000000)]
CHUNK = 5000
N_TRANS = 50000
N_GRAD_BATCH = 20
BATCH = 256


def q_online(agent, obs, a):
    return agent.network.select("critic")(obs, actions=a).mean(0)


def q_target(agent, obs, a):
    return agent.critic_eval(obs, a)


@jax.jit
def actor_act(agent, obs, z):
    return jnp.clip(agent.network.select("actor")(obs, z), -1, 1)


@jax.jit
def both_q(agent, obs, a):
    return q_online(agent, obs, a), q_target(agent, obs, a)


@jax.jit
def dq_da(agent, obs, a):
    return jax.grad(lambda aa: q_online(agent, obs, aa).sum())(a)


def batched(f, *arrs):
    outs = [f(*[x[k:k + CHUNK] for x in arrs]) for k in range(0, len(arrs[0]), CHUNK)]
    if isinstance(outs[0], tuple):
        return tuple(np.concatenate([np.asarray(o[i]) for o in outs]) for i in range(len(outs[0])))
    return np.concatenate([np.asarray(o) for o in outs])


def region_mask(ci, cj, task):
    m = np.zeros(len(ci), bool)
    for i, j in START_REGION[task]:
        m |= (ci == i) & (cj == j)
    return m


def actor_term_grads(agent, q_agent, lam_agent, obs, z):
    """Actor-parameter gradients of alpha*distill and of the normalized Q term, as in GFPAgent.actor_loss.

    q_agent supplies the online critic whose gradient is followed; lam_agent supplies lambda."""
    alpha = agent.config["alpha"]
    target = agent.compute_flow_actions(obs, noises=z)
    p = agent.network.params["modules_actor"]

    def actor_out(pa):
        params = dict(agent.network.params)
        params["modules_actor"] = pa
        return agent.network.select("actor")(obs, z, params=params)

    def distill(pa):
        return alpha * jnp.mean((actor_out(pa) - target) ** 2)

    def qterm(pa):
        a = jnp.clip(actor_out(pa), -1, 1)
        q = q_online(q_agent, obs, a)
        lam = jax.lax.stop_gradient(1 / jnp.abs(q_online(lam_agent, obs, a)).mean())
        return -lam * q.mean()

    gd, gq = jax.grad(distill)(p), jax.grad(qterm)(p)
    flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
    return flat(gd), flat(gq)


actor_term_grads_jit = jax.jit(actor_term_grads)


def m1_m2(states):
    obs, z, ci, cj = states["obs"], states["z"], states["cell_i"], states["cell_j"]
    reg = region_mask(ci, cj, "task1")
    rng = np.random.default_rng(7)
    gb = [rng.choice(len(obs), BATCH, replace=False) for _ in range(N_GRAD_BATCH)]
    out = {}
    for name, sa, sb in PAIRS:
        A, _, _ = load_agent(name, sa)
        B, _, _ = load_agent(name, sb)
        aA = batched(lambda o, zz: actor_act(A, o, zz), obs, z)
        aB = batched(lambda o, zz: actor_act(B, o, zz), obs, z)
        onA_aA, tgA_aA = batched(lambda o, a: both_q(A, o, a), obs, aA)
        onB_aB, tgB_aB = batched(lambda o, a: both_q(B, o, a), obs, aB)
        onB_aA, tgB_aA = batched(lambda o, a: both_q(B, o, a), obs, aA)
        onA_aB, tgA_aB = batched(lambda o, a: both_q(A, o, a), obs, aB)
        r = {}
        for tag, (qa_a, qb_b, qb_a, qa_b) in dict(online=(onA_aA, onB_aB, onB_aA, onA_aB),
                                                   target=(tgA_aA, tgB_aB, tgB_aA, tgA_aB)).items():
            parts = dict(total=qb_b - qa_a, critic=qb_a - qa_a, actor=qa_b - qa_a)
            r[tag] = dict(V_A_start=float(qa_a[reg].mean()), V_B_start=float(qb_b[reg].mean()),
                          V_A_all=float(qa_a.mean()), V_B_all=float(qb_b.mean()),
                          start={k: float(v[reg].mean()) for k, v in parts.items()},
                          all={k: float(v.mean()) for k, v in parts.items()})
        r["online_minus_target_start"] = dict(A=float((onA_aA - tgA_aA)[reg].mean()), B=float((onB_aB - tgB_aB)[reg].mean()))
        # M2: lambda on the probe states (uniform dataset states) and action gradients at a_A
        r["lambda"] = dict(A=float(1 / np.abs(onA_aA).mean()), B=float(1 / np.abs(onB_aB).mean()),
                           B_at_aA=float(1 / np.abs(onB_aA).mean()))
        gA = batched(lambda o, a: dq_da(A, o, a), obs, aA)
        gB = batched(lambda o, a: dq_da(B, o, a), obs, aA)
        cos = (gA * gB).sum(1) / (np.linalg.norm(gA, axis=1) * np.linalg.norm(gB, axis=1) + 1e-12)
        nA, nB = np.linalg.norm(gA, axis=1), np.linalg.norm(gB, axis=1)
        r["dq_da"] = dict(cos_start=float(cos[reg].mean()), cos_all=float(cos.mean()),
                          norm_A_start=float(nA[reg].mean()), norm_B_start=float(nB[reg].mean()),
                          norm_A_all=float(nA.mean()), norm_B_all=float(nB.mean()),
                          lam_norm_A_all=float(nA.mean() / np.abs(onA_aA).mean()),
                          lam_norm_B_all=float(nB.mean() / np.abs(onB_aA).mean()))
        # actor-parameter gradients of the two terms, actor A fixed, critic A vs critic B
        g = dict(distill=[], q_A=[], q_B=[], q_B_lamA=[], cos_qA_qB=[], cos_d_qA=[], cos_d_qB=[])
        for idx in gb:
            o, zz = jnp.asarray(obs[idx]), jnp.asarray(z[idx])
            gd, gqa = actor_term_grads_jit(A, A, A, o, zz)
            _, gqb = actor_term_grads_jit(A, B, B, o, zz)
            _, gqbl = actor_term_grads_jit(A, B, A, o, zz)
            nrm = lambda v: float(jnp.linalg.norm(v))
            cs = lambda u, v: float(jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-12))
            g["distill"].append(nrm(gd)); g["q_A"].append(nrm(gqa)); g["q_B"].append(nrm(gqb)); g["q_B_lamA"].append(nrm(gqbl))
            g["cos_qA_qB"].append(cs(gqa, gqb)); g["cos_d_qA"].append(cs(gd, gqa)); g["cos_d_qB"].append(cs(gd, gqb))
        r["actor_param_grads"] = {k: float(np.mean(v)) for k, v in g.items()}
        r["action_shift_l2"] = dict(all=float(np.linalg.norm(aB - aA, axis=1).mean()),
                                    start=float(np.linalg.norm(aB - aA, axis=1)[reg].mean()))
        out[f"{name}:{sa}:{sb}"] = r
        print(name, sa, sb, json.dumps({k: r[k] for k in ("lambda", "dq_da", "actor_param_grads")}),
              "online.start", json.dumps(r["online"]["start"]), "target.start", json.dumps(r["target"]["start"]), flush=True)
    return out


def m3():
    from envs.env_utils import make_env_and_datasets
    from utils.log_utils import BENCHMARK_OGBENCH
    from common import xy_to_ij
    out = {}
    ds_cache = {}
    for name, step in M3_SPECS:
        agent, cfg, _ = load_agent(name, step)
        env_name = cfg["env_name"]
        if env_name not in ds_cache:
            _, _, train, _ = make_env_and_datasets(BENCHMARK_OGBENCH, env_name)
            idx = np.random.default_rng(3).choice(train.size, N_TRANS, replace=False)
            ds_cache[env_name] = {k: np.asarray(train[k][idx]) for k in ("observations", "next_observations", "rewards", "masks")}
        d = ds_cache[env_name]
        task = name.split("_")[1]
        ci, cj = xy_to_ij(d["observations"][:, :2])
        reg = region_mask(ci, cj, task)
        assert agent.config["target_use_next"] is False and agent.config["target_agg"] == "mean"
        key = jax.random.PRNGKey(5)

        @jax.jit
        def targets(ag, s, s2, rngk):
            noise = ag.sample_noises(s2, rngk)  # one draw, shared by actor and VaBC as in critic_loss
            a_act = jnp.clip(ag.network.select("actor")(s2, noise), -1, 1)
            a_flow_src = ag.compute_flow_actions(s, noise)    # official: VaBC acts from s (target_use_next=False)
            a_flow_paper = ag.compute_flow_actions(s2, noise)  # paper: VaBC acts from s'
            qa = ag.critic_eval(s2, a_act)
            return qa, ag.critic_eval(s2, a_flow_src), ag.critic_eval(s2, a_flow_paper)

        qa, qf, qfp = [], [], []
        for k in range(0, N_TRANS, CHUNK):
            key, sk = jax.random.split(key)
            r = targets(agent, jnp.asarray(d["observations"][k:k + CHUNK]), jnp.asarray(d["next_observations"][k:k + CHUNK]), sk)
            qa.append(np.asarray(r[0])); qf.append(np.asarray(r[1])); qfp.append(np.asarray(r[2]))
        qa, qf, qfp = map(np.concatenate, (qa, qf, qfp))
        g, m, rew = agent.config["discount"], d["masks"], d["rewards"]
        y_mean = rew + g * m * (qa + qf) / 2         # official
        y_actor = rew + g * m * qa                   # candidate target_agg=actor
        y_paper = rew + g * m * (qa + qfp) / 2       # paper formula, registered only
        res = {}
        for tag, sel in (("all", np.ones(N_TRANS, bool)), ("start", reg)):
            dd, dp = (y_actor - y_mean)[sel], (y_paper - y_mean)[sel]
            res[tag] = dict(n=int(sel.sum()), target_mean=float(y_mean[sel].mean()),
                            actor_minus_mean=float(dd.mean()), actor_minus_mean_abs=float(np.abs(dd).mean()),
                            actor_minus_mean_frac_pos=float((dd > 0).mean()),
                            paper_minus_source=float(dp.mean()), paper_minus_source_abs=float(np.abs(dp).mean()),
                            q_actor_minus_q_flow_src=float((qa - qf)[sel].mean()))
        res["mask_mean"] = float(m.mean())
        res["reward_values"] = sorted(set(np.round(np.unique(rew), 3).tolist()))[:5]
        out[f"{name}:{step}"] = res
        print("M3", name, step, json.dumps(res), flush=True)
    return out


def main():
    states = np.load(sys.argv[1])
    res = dict(m1_m2=m1_m2(states))
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    res["m3"] = m3()
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
