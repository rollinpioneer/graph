"""Install as agents/floq_adapter.py in each isolated host workspace."""
import copy
import flax
import jax
from floq_value_core.core import (network_entries, make_network, value_loss,
    next_value_gfp, next_value_dfql, support_config)
from agents.update_runtime import ControlledUpdateMixin
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import ActorVectorField


def create_agent(host, seed, observations, actions, config):
    # Original route is the upstream create call, including identical RNG/module order.
    if host == 'gfp':
        from agents.gfp import GFPAgent as Original
    elif host == 'dfql':
        from agents.dfql import DFQLAgent as Original
    else: raise ValueError(host)
    if config.get('critic_type', 'scalar') == 'scalar':
        clean = copy.deepcopy(dict(config))
        clean.pop('critic_type', None)
        return Original.create(seed, observations, actions, clean)
    if config['critic_type'] != 'floq': raise ValueError(config['critic_type'])
    c = support_config(config, config['r_min'], config['r_max'])
    c.setdefault('training_authorized',False)
    c.setdefault('optimizer_steps_allowed',0)
    c.setdefault('online_learning_allowed',False)
    c.setdefault('initial_optimizer_step',1)
    if c['encoder'] is not None: raise ValueError('This card excludes pixels/encoders')
    if host == 'dfql' and c['actor_type'] != 'DFQL': raise ValueError('DFQL multi-step policy required')
    rng, init_rng = jax.random.split(jax.random.PRNGKey(seed), 2)
    entries = network_entries(c, observations, actions, guidance_target=(host=='gfp'))
    if host == 'gfp':
        entries.update(vabc=(ActorVectorField(hidden_dims=tuple(c['actor_hidden_dims']),
            action_dim=actions.shape[-1],layer_norm=c['actor_layer_norm'],need_time=True,
            encode_time_dim=c['encode_time_dim']), (observations,actions,actions[...,:1])),
            actor=(ActorVectorField(hidden_dims=tuple(c['actor_hidden_dims']),action_dim=actions.shape[-1],
                layer_norm=c['actor_layer_norm'],need_time=False),(observations,actions)))
        if c['eta_temperature_online'] is None: c['eta_temperature_online']=c['eta_temperature']
    else:
        entries['actor_bc_flow']=(ActorVectorField(hidden_dims=tuple(c['actor_hidden_dims']),
            action_dim=actions.shape[-1], layer_norm=c['actor_layer_norm'],tanh_output=True),
            (observations,actions,actions[...,:1]))
    network = make_network(ModuleDict, TrainState, entries, init_rng, c)
    c.update(ob_dims=observations.shape[1:], action_dim=actions.shape[-1])

    class CombinedAgent(ControlledUpdateMixin,Original):
        _wiring_host=host
        def critic_loss(self, batch, grad_params, rng):
            return value_loss(self, batch, grad_params, rng,
                              next_value_gfp if host=='gfp' else next_value_dfql)

    # Actor loss, policy, guidance and sampling are inherited verbatim.
    return CombinedAgent(rng=rng, network=network, config=flax.core.FrozenDict(c))
