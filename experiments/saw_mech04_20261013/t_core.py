"""Four-arm actor-only intervention; full frozen reference remains immutable."""
from common import *
import optax, jax.numpy as jnp, functools
ACTOR_KEYS=('modules_actor','modules_actor_goal_rep_def')

def subset(params):return {k:params[k] for k in ACTOR_KEYS}
def actor_optimizer(agent):
    adam=agent.network.opt_state[0]
    return (adam._replace(mu=subset(adam.mu),nu=subset(adam.nu)),agent.network.opt_state[1])

def policy(reference,params):
    full=dict(reference.network.params);full.update(params)
    return reference.replace(network=reference.network.replace(params=full))

def terms(reference,params,batch):
    p=dict(reference.network.params);p.update(params)
    s=batch['observations'];g=batch['high_actor_goals'];n=batch['next_observations'];w=batch['high_actor_targets']
    v1,v2=reference.network.select('value')(s,g);n1,n2=reference.network.select('value')(n,g);w1,w2=reference.network.select('value')(w,g)
    wa=jnp.minimum(jnp.exp((n1+n2-v1-v2)*.5*reference.config['awr_alpha']),100.)
    ww=jnp.minimum(jnp.exp((w1+w2-v1-v2)*.5*reference.config['kl_alpha']),100.)
    dist=reference.network.select('actor')(s,g,params=p);teacher=reference.network.select('low_actor')(s,w)
    return -wa*dist.log_prob(batch['actions']),ww*jnp.sum((dist.mode()-jax.lax.stop_gradient(teacher.mode()))**2,axis=-1)

def loss(reference,params,batch,arm):
    awr,wp=terms(reference,params,batch)
    mask=jnp.ones(1024) if arm in (0,3) else jnp.concatenate([jnp.ones(512),jnp.zeros(512)])
    denominator=512 if arm==2 else 1024
    aa=jnp.sum(awr*mask)/denominator;ww=jnp.sum(wp*mask)/denominator
    return aa+ww,dict(awr_loss=aa,waypoint_loss=ww)

@functools.partial(jax.jit,static_argnames=('arm',))
def update(reference,params,opt_state,batch,arm):
    (l,info),grad=jax.value_and_grad(loss,argnums=1,has_aux=True)(reference,params,batch,arm)
    tx=optax.adam(reference.config['lr']);delta,new_state=tx.update(grad,opt_state,params);new_params=optax.apply_updates(params,delta)
    info=dict(info,loss=l,gradient_finite=all_finite(grad),parameter_finite=all_finite(new_params))
    return new_params,new_state,info

def all_finite(tree):return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))

def make_batch(data,indices,arm):
    i,fg,fw,rg,rw=indices;goal=fg.copy();wp=fw.copy()
    if arm==3:goal[512:]=rg[512:];wp[512:]=rw[512:]
    b=data.dataset.sample(1024,i)
    b.update(high_actor_goals=data.get_observations(goal),high_actor_targets=data.get_observations(wp))
    return {k:b[k] for k in ('observations','actions','next_observations','high_actor_goals','high_actor_targets')}
