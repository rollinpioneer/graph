"""Unit checks for the G-ReBRAC flag: run once per source tree (orig / patched)."""
import importlib, sys, json
import numpy as np
import jax, jax.numpy as jnp

root = sys.argv[1]
sys.path.insert(0, root)
from agents.rebrac import ReBRACAgent
from utils.datasets import GPU_dataset_sample  # noqa: F401  (import check)

def make_config(**over):
    cfg = dict(
        agent_name="rebrac", lr=3e-4, batch_size=256,
        actor_hidden_dims=[512, 512, 512, 512], value_hidden_dims=[512, 512, 512, 512],
        layer_norm=True, actor_layer_norm=False, discount=0.995, tau=0.005, encoder=None,
        alpha_actor=0.01, alpha_critic=0.01, q_agg="mean", actor_freq=2,
        actor_noise=0.2, actor_noise_clip=0.5, actor_fc_scale=0.01, tanh_squash=True,
        log_metrics=True,
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

def run(cfg, steps=6):
    agent = ReBRACAgent.create(0, batch["observations"][:1], batch["actions"][:1], cfg)
    infos = []
    for i in range(steps):
        agent, info = agent.update(batch, full_update=(i % 2 == 0))
        infos.append({k: float(v) for k, v in info.items()})
    flat = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(agent.network.params)])
    return agent, infos, flat

out = {}
has_flag = "guidance" in open(f"{root}/config/agent/rebrac.yaml").read()
cfgs = {"off": make_config(**({"guidance": False, "eta_temperature": 1e-3, "guidance_normalize": True} if has_flag else {}))}
if has_flag:
    cfgs["on_norm"] = make_config(guidance=True, eta_temperature=1e-3, guidance_normalize=True)
    cfgs["on_raw"] = make_config(guidance=True, eta_temperature=1e-3, guidance_normalize=False)
    cfgs["on_hot"] = make_config(guidance=True, eta_temperature=1e9, guidance_normalize=True)
for name, cfg in cfgs.items():
    agent, infos, flat = run(cfg)
    out[name] = dict(param_sum=float(flat.sum()), param_sha=str(np.asarray(flat).tobytes().__hash__()),
                     finite=bool(jnp.all(jnp.isfinite(flat))), last_actor_info=infos[-2])
    np.save(f"/tmp/claude-0/-home-claude-graph/e17622b5-ab70-59a5-b92c-5b918929ee05/scratchpad/tests/{sys.argv[2]}_{name}.npy", np.asarray(flat))
print(json.dumps(out, indent=1))
