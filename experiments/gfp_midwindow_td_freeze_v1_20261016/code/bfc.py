from typing import Any
from functools import partial
import jax
import jax.numpy as jnp
from agents.gfp import GFPAgent
from utils.datasets import GPU_dataset_sample
from utils.flax_utils import nonpytree_field


def target(r, m, gamma, qc, qr, qf, epsilon):
    ref = r + gamma * m * (qr + qf) / 2
    delta = gamma * m * (qc - qr) / 2
    return jax.lax.stop_gradient(ref + jnp.clip(delta, -epsilon, epsilon)), ref, delta


class BFCAgent(GFPAgent):
    reference: Any = None
    tau_pi: float = nonpytree_field(default=0.0005)
    epsilon: float = nonpytree_field(default=0.0)

    def td_parts(self, batch, rng):
        _, key = jax.random.split(rng)
        obs = batch['next_observations']
        vobs = obs if self.config['target_use_next'] else batch['observations']
        current = self.sample_actions(obs, key)
        p = dict(self.network.params)
        p['modules_actor'] = self.reference
        ref_agent = GFPAgent(self.rng, self.network.replace(params=p), self.config)
        reference = ref_agent.sample_actions(obs, key)
        flow = self.sample_flow_actions(vobs, key)
        qc, qr, qf = [self.critic_eval(obs, a) for a in (current, reference, flow)]
        y, yr, d = target(batch['rewards'], batch['masks'], self.config['discount'], qc, qr, qf, self.epsilon)
        return y, yr, d, qc, qr, qf

    def critic_loss(self, batch, grad_params, rng):
        y, yr, d, qc, qr, qf = self.td_parts(batch, rng)
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        loss = jnp.square(q-y).mean()
        return loss, dict(critic_loss=loss, q_mean=q.mean(), q_max=q.max(), q_min=q.min(),
                         tgt_q_mean=((qr+qf)/2).mean(), td_mean=y.mean(),
                         delta_abs=jnp.abs(d).mean(), delta_max=jnp.abs(d).max(),
                         clipped_fraction=jnp.mean(jnp.abs(d)>self.epsilon),
                         accepted_abs=jnp.abs(y-yr).mean())

    @partial(jax.jit, static_argnames=('n_steps_fused','dataset_size','batch_size'))
    def multi_sample_and_update(self, n_steps_fused, dataset, dataset_size, batch_size):
        ag = self
        healthy = jnp.array(True)
        for _ in range(n_steps_fused):
            batch, rng = GPU_dataset_sample(ag.rng, dataset, batch_size, dataset_size)
            rng, loss_rng = jax.random.split(rng)
            def loss_fn(p):
                return ag.total_loss(batch, p, rng=loss_rng)
            net, info = ag.network.apply_loss_fn(loss_fn=loss_fn)
            ag.target_update(net, 'critic')  # preserve upstream OLD-critic EMA
            ref = jax.tree_util.tree_map(lambda old, new: (1-ag.tau_pi)*old+ag.tau_pi*new,
                                         ag.reference, net.params['modules_actor'])
            healthy = healthy & jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in info.values()]))
            ag = ag.replace(network=net, rng=rng, reference=ref)
        return ag, dict(info, fused_all_finite=healthy)
