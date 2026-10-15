"""Checks for the deterministic-teacher option (patch p3). Usage: python test_teacher.py SRC_ROOT [save_bc_ref]"""
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

TEACHER = dict(guidance=True, guidance_ref="teacher", guidance_normalize=False)
res = {}
_, p_bc, _ = run(make_config(guidance=True, guidance_ref="bc"))
if len(sys.argv) > 2 and sys.argv[2] == "save_bc_ref":
    np.save(f"{HERE}/p2_bc_ref.npy", flat(p_bc))
    print("saved p2_bc_ref.npy")
    sys.exit(0)

# 1. Old modes unchanged.
_, p_off, _ = run(make_config())
res["off_equals_orig"] = bool(np.array_equal(flat(p_off), np.load(f"{HERE}/orig_off.npy")))
_, p_l1, _ = run(make_config(guidance=True))
res["actor_ref_equals_p1_on_norm"] = bool(np.array_equal(flat(p_l1), np.load(f"{HERE}/patched_on_norm.npy")))
res["bc_ref_equals_p2"] = bool(np.array_equal(flat(p_bc), np.load(f"{HERE}/p2_bc_ref.npy")))

# 2. Teacher mode: module and init.
init_t, _, _ = run(make_config(**TEACHER), steps=0)
init_off = ReBRACAgent.create(0, batch["observations"][:1], batch["actions"][:1], make_config()).network.params
res["teacher_module_present"] = "modules_teacher" in init_t
res["init_other_modules_unchanged"] = bool(np.array_equal(flat(init_t, skip=("modules_teacher",)), flat(dict(init_off))))

# 3. Gradient isolation (one full update's gradients at init).
def grads(cfg):
    agent = ReBRACAgent.create(0, batch["observations"][:1], batch["actions"][:1], cfg)
    g = jax.grad(lambda p: agent.total_loss(batch, p, True, rng=jax.random.PRNGKey(1))[0])(agent.network.params)
    return dict(g)
g_base = grads(make_config(**TEACHER))
g_alpha = grads(make_config(**TEACHER, alpha_actor=5.0))
g_eta = grads(make_config(**TEACHER, eta_temperature=1e9))
res["teacher_grad_independent_of_alpha_actor"] = bool(np.array_equal(flat({"m": g_base["modules_teacher"]}), flat({"m": g_alpha["modules_teacher"]})))
res["actor_critic_grad_independent_of_eta"] = bool(np.array_equal(flat(g_base, skip=("modules_teacher",)), flat(g_eta, skip=("modules_teacher",))))
res["teacher_grad_depends_on_eta"] = bool(not np.array_equal(flat({"m": g_base["modules_teacher"]}), flat({"m": g_eta["modules_teacher"]})))
res["actor_grad_depends_on_alpha_actor"] = bool(not np.array_equal(flat({"m": g_base["modules_actor"]}), flat({"m": g_alpha["modules_actor"]})))

# 4. Short training.
init_t, p_t, infos = run(make_config(**TEACHER), steps=40)
res["teacher_finite"] = bool(np.all(np.isfinite(flat(p_t))))
ai = [x for x in infos if "actor/teacher_mse" in x]
res["teacher_mse_first_last"] = [round(ai[0]["actor/teacher_mse"], 3), round(ai[-1]["actor/teacher_mse"], 3)]
res["mse_teacher_first_last"] = [round(ai[0]["actor/mse_teacher"], 4), round(ai[-1]["actor/mse_teacher"], 4)]
res["guidance_first_last"] = [round(ai[0]["actor/guidance"], 4), round(ai[-1]["actor/guidance"], 4)]
res["weight_max_last"] = round(ai[-1]["actor/weight_max"], 4)
res["teacher_params_moved"] = bool(not np.array_equal(flat({"m": p_t["modules_teacher"]}), flat({"m": init_t["modules_teacher"]})))

# 5. Fused path and validation info (as main.py uses them).
N = 5000
r = np.random.default_rng(1)
ds = dict(observations=jnp.asarray(r.normal(size=(N, OD)), jnp.float32),
          actions=jnp.asarray(r.uniform(-1, 1, size=(N, AD)), jnp.float32),
          next_observations=jnp.asarray(r.normal(size=(N, OD)), jnp.float32),
          rewards=-jnp.ones(N), masks=jnp.ones(N), terminals=jnp.zeros(N))
ds["next_actions"] = jnp.concatenate([ds["actions"][1:], ds["actions"][-1:]], 0)
agent = ReBRACAgent.create(0, ds["observations"][:1], ds["actions"][:1], make_config(**TEACHER))
for _ in range(3):
    agent, info = agent.multi_sample_and_update(4, ds, N, 256)
_, vinfo = agent.total_loss({k: v[:256] for k, v in ds.items()}, grad_params=None)
res["validation_keys"] = all(k in vinfo for k in ("actor/teacher_mse", "actor/mse_teacher", "actor/guidance", "actor/mse", "actor/teacher_loss"))
res["validation_finite"] = all(bool(jnp.isfinite(v)) for v in vinfo.values())
print(json.dumps(res, indent=1))
