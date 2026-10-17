"""Stage-2 confirmation (CARD_CONFIRM_SCRATCH_20261016.md): IQL-inspired GFP trained from scratch.

Identical update to the IQL arm of experiments/gfp_critic_screen_20261016/code/cand_fork.py (the screened and
confirmed I2 arm): the critic's TD target is r + gamma * m * V(s'), V is a separate state-value network fitted by
expectile regression on the target critic at dataset actions; per batch V is updated first, then the critic and the
actor with the unchanged official GFP losses. Only create() is new:
  - every official GFP network is created exactly as GFPAgent.create (same seed, same config);
  - V = ModuleDict(value=Value(value_hidden_dims, layer_norm, num_ensembles=1)), initialised with
    PRNGKey(seed + 20261016), own Adam with the agent's lr;
  - no V-only warm-up: V, critic and actor start training together at step 1 (card section 3).
"""
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax

from agents.gfp import GFPAgent
from utils.datasets import GPU_dataset_sample
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Value

V_SEED_OFFSET = 20261016


class IQLScratchAgent(GFPAgent):
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

    def update(self, batch):
        raise NotImplementedError("stage 2 runs with dataset_on_gpu=True (fused updates only)")

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        assert 0.5 < float(config["expectile"]) < 1.0
        base = GFPAgent.create(seed, ex_observations, ex_actions, config)
        conf = base.config
        assert conf["eta_temperature"] != 0 and not conf["target_use_next"] and conf["target_agg"] == "mean"
        vdef = ModuleDict(dict(value=Value(hidden_dims=conf["value_hidden_dims"], layer_norm=conf["layer_norm"],
                                           num_ensembles=1)))
        vparams = vdef.init(jax.random.PRNGKey(int(seed) + V_SEED_OFFSET), value=(ex_observations,))["params"]
        vstate = TrainState.create(vdef, vparams, tx=optax.adam(learning_rate=conf["lr"]))
        print(f"==> IQL-inspired GFP from scratch: expectile={conf['expectile']}, V seed={int(seed) + V_SEED_OFFSET}",
              flush=True)
        return cls(base.rng, network=base.network, config=conf, value=vstate)
