"""Expected-return floq subsystem, transplanted from 3d60e638.

Host policy, actor objective, and next-action generation remain host-owned.
The source's noise SUM, ensemble-first backup and cos(t) encoding are preserved.
"""
import copy
import flax
import jax
import jax.numpy as jnp
import optax
from .upstream_networks import CriticVectorField, Value


def repeat_batch(x, n):
    return x[:, None].repeat(n, axis=1).reshape((x.shape[0] * n, *x.shape[1:]))


def aggregate_ensemble(values, rule):
    return values.min(axis=0) if rule == 'min' else values.mean(axis=0)


def aggregate_sources(actor, vabc, rule):
    if rule == 'actor': return actor
    if rule == 'vabc': return vabc
    if rule == 'min': return jnp.minimum(actor, vabc)
    if rule == 'max': return jnp.maximum(actor, vabc)
    if rule == 'mean': return (actor + vabc) / 2
    raise ValueError(rule)


def euler_integrate(velocity, observations, actions, initial, steps):
    returns = initial
    for i in range(steps):
        t = jnp.full((*observations.shape[:-1], 1), i / steps)
        returns = returns + velocity(observations, actions, returns=returns, times=t) / steps
    return returns


def support_config(config, r_min, r_max):
    cfg = copy.deepcopy(dict(config))
    if not (0 < cfg['discount'] < 1): raise ValueError('discount outside (0,1)')
    cfg.update(r_min=float(r_min), r_max=float(r_max))
    cfg['noise_min'] = cfg['noise_coverage'] * r_min / (1 - cfg['discount'])
    cfg['noise_max'] = cfg['noise_coverage'] * r_max / (1 - cfg['discount'])
    cfg['q_min'] = (r_min - cfg['reward_offset']) / (1 - cfg['discount'])
    cfg['q_max'] = (r_max + cfg['reward_offset']) / (1 - cfg['discount'])
    return cfg


def network_entries(config, obs, actions, guidance_target=False):
    """New modules only; never overwrites host network utilities."""
    if config['encoder'] is not None or config['use_resnets']:
        raise ValueError('This card freezes state inputs, no encoder, non-residual floq')
    scalar = Value(hidden_dims=tuple(config['value_hidden_dims']),
                   layer_norm=config['layer_norm'], num_ensembles=config['distill_num_ensembles'])
    flow = CriticVectorField(
        hidden_dims=tuple([config['block_width']] * config['block_depth']),
        layer_norm=config['layer_norm'], num_ensembles=config['flow_num_ensembles'],
        embed_time=config['embed_time'], time_embed_dim=config['time_embed_dim'],
        use_prob_embed=config['use_prob_embed'], q_min=config['q_min'], q_max=config['q_max'],
        num_bins=config['num_bins'], sigma=config['sigma'])
    returns = jnp.tile(actions[None, ..., :1], (config['flow_num_ensembles'], 1, 1))
    entries = dict(critic=(scalar, (obs, actions)),
                   floq=(flow, (obs, actions, returns, actions[..., :1])),
                   target_floq=(copy.deepcopy(flow), (obs, actions, returns, actions[..., :1])))
    if guidance_target:
        entries['target_critic'] = (copy.deepcopy(scalar), (obs, actions))
    return entries


def make_network(module_dict, train_state, entries, init_rng, config):
    definition = module_dict({k: v[0] for k, v in entries.items()})
    params = definition.init(init_rng, **{k: v[1] for k, v in entries.items()})['params']
    params['modules_target_floq'] = params['modules_floq']
    if 'modules_target_critic' in params:
        params['modules_target_critic'] = params['modules_critic']
    # Initialization creates zero optimizer state; tx.update is never called.
    return train_state.create(definition, params, tx=optax.adam(learning_rate=config['lr']))


def integrate(agent, obs, actions, ratios, target=False):
    c = agent.config
    initial = c['noise_min'] * (1-ratios) + c['noise_max'] * ratios
    initial = jnp.tile(initial[None], (c['flow_num_ensembles'], 1, 1))
    return euler_integrate(agent.network.select('target_floq' if target else 'floq'),
                          obs, actions, initial, c['critic_flow_steps'])


def terminal_backup(rewards, masks, discount, next_values):
    return rewards + discount * masks * next_values


def backup_value(agent, obs, actions, ratios):
    # Ensemble aggregation precedes all sample/source averaging.
    return aggregate_ensemble(integrate(agent, obs, actions, ratios, target=True), agent.config['q_agg'])


def next_value_gfp(agent, batch, action_rng, ratios):
    c = agent.config; repeat = c['noise_samples']; rule = c['target_agg']
    next_obs = repeat_batch(batch['next_observations'], repeat)
    vabc_obs = repeat_batch(batch['next_observations'] if c['target_use_next']
                            else batch['observations'], repeat)
    actor = vabc = None
    if rule != 'vabc':
        actor_actions = agent.sample_actions(next_obs, action_rng)
        actor = backup_value(agent, next_obs, actor_actions, ratios)
    if rule != 'actor':
        vabc_actions = agent.sample_flow_actions(vabc_obs, action_rng)
        vabc = backup_value(agent, next_obs, vabc_actions, ratios)
    values = aggregate_sources(actor, vabc, rule)
    return values.reshape(batch['rewards'].shape[0], repeat)


def next_value_dfql(agent, batch, action_rng, ratios):
    obs = repeat_batch(batch['next_observations'], agent.config['noise_samples'])
    # Explicitly current DFQL multi-step policy, never floq's one-step policy.
    actions = jnp.clip(agent.sample_actions(obs, seed=action_rng), -1, 1)
    values = backup_value(agent, obs, actions, ratios)
    return values.reshape(batch['rewards'].shape[0], agent.config['noise_samples'])


def reduce_losses(pred, velocity, scalar, current_returns):
    # Source order: SUM initial noises, then mean ensemble and batch.
    flow_matrix = jnp.sum(jnp.square(pred-velocity), axis=-1)
    target = jax.lax.stop_gradient(current_returns.mean(axis=-1).mean(axis=0))
    distill_matrix = jnp.square(scalar-target)
    return flow_matrix.mean(), distill_matrix.mean(), (flow_matrix+distill_matrix).mean()


def value_loss(agent, batch, grad_params, rng, next_value_fn):
    c = agent.config; b = batch['actions'].shape[0]; r = c['noise_samples']; e = c['flow_num_ensembles']
    obs = repeat_batch(batch['observations'], r); actions = repeat_batch(batch['actions'], r)
    rng, action_rng = jax.random.split(rng)
    rng, next_noise_rng = jax.random.split(rng)
    next_ratios = jax.random.uniform(next_noise_rng, (b*r, 1))
    nxt = next_value_fn(agent, batch, action_rng, next_ratios)
    target = jax.lax.stop_gradient(terminal_backup(batch['rewards'][:,None], batch['masks'][:,None],
                                                  c['discount'], nxt).mean(axis=-1))
    rng, noise_rng = jax.random.split(rng)
    ratios = jax.random.uniform(noise_rng, (b*r, 1))
    current = jax.lax.stop_gradient(integrate(agent, obs, actions, ratios)).reshape(e, b, r)
    x0 = c['noise_min']*(1-ratios) + c['noise_max']*ratios
    x1 = jnp.tile(target.reshape(1,b,1), (e,1,r)).reshape(e,b*r,1)
    rng, t_rng = jax.random.split(rng)
    t = jnp.zeros((b*r,1)) if c['train_at_zero_only'] else jax.random.uniform(t_rng, (b*r,1))
    xt = (1-t[None])*x0[None] + t[None]*x1
    velocity = (x1-x0[None]).reshape(e,b,r)
    pred = agent.network.select('floq')(obs, actions, returns=xt, times=t, params=grad_params).reshape(e,b,r)
    scalar = agent.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
    flow, distill, total = reduce_losses(pred, velocity, scalar, current)
    return total, dict(floq_loss=flow, distilled_critic_loss=distill, critic_loss=total,
                      q=scalar.mean(), td_error=jnp.square(current.mean(-1).mean(0)-target).mean(),
                      var_flow_returns=current.mean(0).var(-1).mean(), target_q=target.mean())


def ema_specification(old_online, old_target, tau):
    """Pure definition: source pre-optimizer online state, one-update lag.

Pure function; the controlled T0 update writes its output to a new state.
No input Agent/network/state is mutated.
"""
    return jax.tree_util.tree_map(lambda p,tp: p*tau + tp*(1-tau), old_online, old_target)


TARGET_KEYS = {'gfp': {'floq': 'target_floq', 'critic': 'target_critic'},
               'dfql': {'floq': 'target_floq'}}


def target_parameter_specification(previous_params, tau, host):
    """Complete pure target-key definition, shared by T0 real updates and fixtures."""
    return {'modules_'+target: ema_specification(previous_params['modules_'+online],
                 previous_params['modules_'+target], tau)
            for online,target in TARGET_KEYS[host].items()}


def reject_update(*args, **kwargs):
    raise RuntimeError('NOT_AUTHORIZED: this card allows 0 optimizer/target updates')
