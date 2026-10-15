"""Checks for the BC-reference option (patch p2). Usage: python test_bcref.py SRC_ROOT"""
import sys, json
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, sys.argv[1])
from agents.rebrac import ReBRACAgent

HERE = sys.argv[0].rsplit("/", 1)[0]

def make_config(**over):
    cfg = dict(
        agent_name="rebrac", lr=3e-4, batch_size=256,
        actor_hidden_dims=[512, 512, 512, 512], value_hidden_dims=[512, 512, 512, 512],
        layer_norm=True, actor_layer_norm=False, discount=0.995, tau=0.005, encoder=None,
        alpha_actor=0.01, alpha_critic=0.01, q_agg="mean", actor_freq=2,
        actor_noise=0.2, actor_noise_clip=0.5, actor_fc_scale=0.01, tanh_squash=True,
        log_metrics=True, guidance=False, eta_temperature=1e-3, guidance_normalize=True,
        guidance_ref="actor",
    )
    cfg.update(over)
    return cfg

rng = np.random.default_rng(0)
B, OD, AD = 256, 69, 21
batch = dict(
    observations=jnp.asarray(rng.normal(size=(B, OD)), jnp.float32),
    actions=jnp.asarray(rng.uniform(-1, 1, size=(B, AD)), jnp.float32),
    next_observations=jnp.asarray(rng.normal(size=(B, OD)), jnp.float32),
    next_actions=jnp.asarray(rng.uniform(-1, 1, size=(B, AD)), jnp.float32),
    rewards=jnp.asarray(-np.ones(B), jnp.float32),
    masks=jnp.ones(B, jnp.float32), terminals=jnp.zeros(B, jnp.float32),
)

def flat(params, skip=()):
    keys = sorted(k for k in params if k not in skip)
    return np.concatenate([np.asarray(x).ravel() for k in keys for x in jax.tree_util.tree_leaves(params[k])])

def run(cfg, steps=6):
    agent = ReBRACAgent.create(0, batch["observations"][:1], batch["actions"][:1], cfg)
    init = dict(agent.network.params)
    infos = []
    for i in range(steps):
        agent, info = agent.update(batch, full_update=(i % 2 == 0))
        infos.append({k: float(v) for k, v in info.items()})
    return init, dict(agent.network.params), infos

res = {}
_, p_off, _ = run(make_config())
res["off_equals_p1_off"] = bool(np.array_equal(flat(p_off), np.load(f"{HERE}/patched_off.npy")))
res["off_equals_orig"] = bool(np.array_equal(flat(p_off), np.load(f"{HERE}/orig_off.npy")))
_, p_l1, _ = run(make_config(guidance=True))
res["actor_ref_equals_p1_on_norm"] = bool(np.array_equal(flat(p_l1), np.load(f"{HERE}/patched_on_norm.npy")))

init_hot, p_hot, _ = run(make_config(guidance=True, guidance_ref="bc", eta_temperature=1e9))
init_off = ReBRACAgent.create(0, batch["observations"][:1], batch["actions"][:1], make_config()).network.params
res["bc_module_present"] = "modules_bc_ref" in p_hot
res["init_other_modules_unchanged"] = bool(np.array_equal(flat(init_hot, skip=("modules_bc_ref",)), flat(dict(init_off))))
res["hot_eta_bc_ref_equals_off_except_bc_module"] = bool(np.array_equal(flat(p_hot, skip=("modules_bc_ref",)), flat(p_off)))

_, p_bc, infos = run(make_config(guidance=True, guidance_ref="bc"), steps=40)
res["bc_finite"] = bool(np.all(np.isfinite(flat(p_bc))))
actor_infos = [x for x in infos if "bc_ref/mse" in x]
res["bc_ref_mse_first_last"] = [round(actor_infos[0]["bc_ref/mse"], 3), round(actor_infos[-1]["bc_ref/mse"], 3)]
res["guidance_first_last"] = [round(actor_infos[0]["actor/guidance"], 4), round(actor_infos[-1]["actor/guidance"], 4)]
res["bc_ref_params_moved"] = bool(not np.array_equal(flat({"m": p_bc["modules_bc_ref"]}), flat({"m": init_hot["modules_bc_ref"]})))

# Fused path and validation info (as main.py uses them).
N = 5000
r = np.random.default_rng(1)
ds = dict(observations=jnp.asarray(r.normal(size=(N, OD)), jnp.float32),
          actions=jnp.asarray(r.uniform(-1, 1, size=(N, AD)), jnp.float32),
          next_observations=jnp.asarray(r.normal(size=(N, OD)), jnp.float32),
          rewards=-jnp.ones(N), masks=jnp.ones(N), terminals=jnp.zeros(N))
ds["next_actions"] = jnp.concatenate([ds["actions"][1:], ds["actions"][-1:]], 0)
agent = ReBRACAgent.create(0, ds["observations"][:1], ds["actions"][:1], make_config(guidance=True, guidance_ref="bc"))
for _ in range(3):
    agent, info = agent.multi_sample_and_update(4, ds, N, 256)
_, vinfo = agent.total_loss({k: v[:256] for k, v in ds.items()}, grad_params=None)
res["validation_keys_have_bc_ref_and_guidance"] = ("bc_ref/mse" in vinfo) and ("actor/guidance" in vinfo)
res["validation_finite"] = all(bool(jnp.isfinite(v)) for v in vinfo.values())
print(json.dumps(res, indent=1))
