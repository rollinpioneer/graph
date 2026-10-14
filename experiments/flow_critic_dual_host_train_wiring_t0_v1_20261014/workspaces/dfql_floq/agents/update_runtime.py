"""Controlled optimizer/EMA update APIs. No loop starts on import."""
from functools import partial
import jax
import jax.numpy as jnp
from floq_value_core.core import target_parameter_specification


def write_target_state(previous, optimized, config, host):
    # Both sources use the PRE-optimizer online parameters (one-update lag).
    params=dict(optimized.params)
    params.update(target_parameter_specification(previous.params,config['tau'],host))
    return optimized.replace(params=params)


class ControlledUpdateMixin:
    def _check_update_permission(self, count=1):
        if self.config.get('training_authorized',False) is not True:
            raise RuntimeError('NOT_AUTHORIZED: explicit execution gate required')
        if self.config.get('online_learning_allowed',False):
            raise RuntimeError('Online learning is outside this wiring card')
        allowed=int(self.config.get('optimizer_steps_allowed',0))
        used=int(self.network.step)-int(self.config.get('initial_optimizer_step',1))
        if count<1 or used<0 or used+count>allowed:
            raise RuntimeError(f'UPDATE_BUDGET_EXHAUSTED: {used}+{count}>{allowed}')

    def _with_loss_rng(self, next_rng, loss_rng, batch):
        def loss_fn(params):
            return self.total_loss(batch,params,rng=loss_rng)
        optimized,info=self.network.apply_loss_fn(loss_fn=loss_fn)
        network=write_target_state(self.network,optimized,self.config,self._wiring_host)
        return self.replace(network=network,rng=next_rng),info

    @jax.jit
    def _update_impl(self,batch):
        next_rng,loss_rng=jax.random.split(self.rng)
        return self._with_loss_rng(next_rng,loss_rng,batch)

    def update(self,batch):
        self._check_update_permission()
        return self._update_impl(batch)

    @jax.jit
    def _batch_update_impl(self,batches):
        def step(agent,batch):return agent._update_impl(batch)
        agent,infos=jax.lax.scan(step,self,batches)
        return agent,jax.tree_util.tree_map(lambda x:x[-1],infos)

    def batch_update(self,batches):
        count=batches['actions'].shape[0]
        self._check_update_permission(count)
        return self._batch_update_impl(batches)

    @partial(jax.jit,static_argnames=('n_steps_fused','dataset_size','batch_size'))
    def _multi_sample_impl(self,n_steps_fused,dataset,dataset_size,batch_size):
        agent=self
        for _ in range(n_steps_fused):
            if self._wiring_host=='gfp':
                # Exactly the pinned GFP dataset RNG and loss RNG convention.
                from utils.datasets import GPU_dataset_sample
                batch,rng=GPU_dataset_sample(agent.rng,dataset,batch_size,dataset_size)
            else:
                rng,sample_rng=jax.random.split(agent.rng)
                idx=jax.random.randint(sample_rng,(batch_size,),0,dataset_size)
                batch={k:v[idx] for k,v in dataset.items()}
            rng,loss_rng=jax.random.split(rng)
            agent,info=agent._with_loss_rng(rng,loss_rng,batch)
        return agent,info

    def multi_sample_and_update(self,n_steps_fused,dataset,dataset_size,batch_size):
        self._check_update_permission(n_steps_fused)
        if dataset_size>dataset['actions'].shape[0] or dataset_size<1:
            raise ValueError('Invalid dataset_size')
        return self._multi_sample_impl(n_steps_fused,dataset,dataset_size,batch_size)

    def target_update(self,network,module_name=None):
        """Return a new TrainState; callers must use the return value.

        Public standalone calls are gated; actual optimizer updates atomically
        write the full target map with write_target_state, without mutating self.
        """
        self._check_update_permission()
        valid=('floq','critic') if self._wiring_host=='gfp' else ('floq',)
        if module_name is not None and module_name not in valid:raise ValueError(module_name)
        result=write_target_state(self.network,network,self.config,self._wiring_host)
        if module_name is None:return result
        params=dict(network.params)
        params['modules_target_'+module_name]=result.params['modules_target_'+module_name]
        return network.replace(params=params)
