"""Z0-Z8 pure loss/gradient verification. No training main or optimizer update."""
import argparse, ast, copy, hashlib, importlib.metadata as md, json, os, sys, time, traceback
from pathlib import Path
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--host',choices=['gfp','dfql'],required=True)
args=parser.parse_args();HOST=args.host
WORK=ROOT/'workspaces'/f'{HOST}_floq'
sys.path.insert(0,str(WORK))
from agents.floq_adapter import create_agent
from utils.flax_utils import TrainState
from floq_value_core.core import (euler_integrate,aggregate_ensemble,aggregate_sources,terminal_backup,
    next_value_gfp,next_value_dfql,reduce_losses,ema_specification,TARGET_KEYS,support_config,integrate)
from floq_value_core.upstream_networks import to_probs,compute_support

RESULT={'host':HOST,'optimizer_steps':0,'online_learning':False,'tests':[],
        'devices':[str(d) for d in jax.devices()], 'jax':jax.__version__,
        'precision':{'enable_x64':jax.config.jax_enable_x64,'matmul_precision':str(jax.config.jax_default_matmul_precision)},
        'forbidden_call_count':0}

def dump():
    (ROOT/'reports'/f'zero_train_{HOST}.json').write_text(json.dumps(RESULT,indent=2,sort_keys=True))

def check(name,fn):
    start=time.time()
    try:
        details=fn() or {}
        RESULT['tests'].append(dict(test=name,status='PASS',seconds=time.time()-start,details=details))
        print('PASS',name,flush=True)
    except Exception:
        RESULT['tests'].append(dict(test=name,status='FAIL',seconds=time.time()-start,traceback=traceback.format_exc()))
        print('FAIL',name,traceback.format_exc(),flush=True);dump();raise
    dump()

def trap(*a,**kw):
    RESULT['forbidden_call_count']+=1
    raise RuntimeError('Forbidden parameter/optimizer update attempted')

# Install hard update guards before any Agent is initialized.
TrainState.apply_loss_fn=trap;TrainState.apply_gradients=trap
adam=optax.adam
def guarded_adam(*a,**kw):
    tx=adam(*a,**kw)
    return optax.GradientTransformation(tx.init,trap)
optax.adam=guarded_adam
if HOST=='gfp':
    from agents.gfp import GFPAgent as Original
else:
    from agents.dfql import DFQLAgent as Original
for name in ['update','batch_update','multi_sample_and_update','target_update']:
    if hasattr(Original,name):setattr(Original,name,trap)

def fingerprint(agent):
    return hashlib.sha256(flax.serialization.to_bytes(agent)).hexdigest()

def finite(tree):
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))

def equal(a,b,atol=0,rtol=0):
    aa,ta=jax.tree_util.tree_flatten(a);bb,tb=jax.tree_util.tree_flatten(b)
    assert ta==tb
    for x,y in zip(aa,bb):np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=atol,rtol=rtol)

def gradient_norms(tree):
    return {k:float(np.sqrt(sum(np.square(np.asarray(v,dtype=np.float64)).sum()
                              for v in jax.tree_util.tree_leaves(t)))) for k,t in tree.items()}

def route(agent,batch,rng,metric,allowed,actor=False):
    fn=agent.actor_loss if actor else agent.critic_loss
    grad=jax.jit(jax.grad(lambda p:fn(batch,p,rng)[1][metric]))(agent.network.params)
    finite(grad);norms=gradient_norms(grad)
    for key,norm in norms.items():
        if key not in allowed:assert norm==0,(metric,key,norm)
    assert any(norms[k]>0 for k in allowed),(metric,norms)
    return norms

def fixtures():
    obs=jnp.zeros((2,3));act=jnp.zeros((2,1));initial=jnp.ones((2,2,1))*3
    const=lambda o,a,returns,times: jnp.ones_like(returns)*2
    np.testing.assert_allclose(euler_integrate(const,obs,act,initial,8),5,atol=1e-6)
    # min before noise average differs from min after average: 0 versus 5.
    values=jnp.array([[[0.],[10.]],[[10.],[0.]]])
    assert float(aggregate_ensemble(values,'min').mean())==0
    assert float(values.mean(axis=1).min())==5
    a=jnp.array([1.,9.]);v=jnp.array([5.,3.])
    equal(aggregate_sources(a,v,'mean'),jnp.array([3.,6.]));equal(aggregate_sources(a,v,'min'),jnp.array([1.,3.]))
    equal(terminal_backup(jnp.array([-1.,-2.]),jnp.zeros(2),.995,jnp.ones(2)*100),jnp.array([-1.,-2.]))
    equal(terminal_backup(jnp.array([-1.,-2.]),jnp.ones(2),.5,jnp.ones(2)*4),jnp.array([1.,0.]))
    pred=jnp.ones((2,2,8));vel=jnp.zeros_like(pred);scalar=jnp.ones((2,2));current=jnp.ones_like(pred)*2
    flow,dist,total=reduce_losses(pred,vel,scalar,current)
    assert float(flow)==8 and float(dist)==1
    # Scalar supervision never differentiates through online flow.
    assert np.all(np.asarray(jax.grad(lambda x:reduce_losses(pred,vel,scalar,x)[1])(current))==0)
    tau=.005
    equal(ema_specification({'x':jnp.array([2.,4.])},{'x':jnp.array([1.,3.])},tau),
          {'x':jnp.array([1.005,3.005])},atol=1e-6)
    assert TARGET_KEYS['gfp']=={'floq':'target_floq','critic':'target_critic'}
    assert TARGET_KEYS['dfql']=={'floq':'target_floq'}
    # Non-flat Q: endpoint is clipped before taking dQ/da, not clip's derivative.
    raw=jnp.array([[2.,-.3]]);endpoint=jnp.clip(raw,-1,1)
    g=jax.grad(lambda a:jnp.square(a).sum())(endpoint)
    equal(g,jnp.array([[2.,-.6]]),atol=1e-6)
    composite=jax.grad(lambda a:jnp.square(jnp.clip(a,-1,1)).sum())(raw)
    assert float(composite[0,0])==0 and float(g[0,0])==2
    return dict(integral_atol=1e-6,noise_sum=8,distillation_target_stop_gradient=True,
                backup_ensemble_before_sample=True,ema_arrays_only=True,endpoint_clip_before_gradient=True)

def target_sources():
    import floq_value_core.core as core
    previous=core.backup_value
    calls=[]
    class Fake:
        config={'noise_samples':2,'target_agg':'mean','target_use_next':False,'q_agg':'mean'}
        def sample_actions(self,o,seed):calls.append(('actor',np.asarray(o).tolist()));return o[:,:1]+10
        def sample_flow_actions(self,o,seed):calls.append(('vabc',np.asarray(o).tolist()));return o[:,:1]+20
    def value(agent,o,a,ratios):calls.append(('value_state',np.asarray(o).tolist()));return a
    core.backup_value=value
    batch=dict(observations=jnp.array([[1.],[2.]]),next_observations=jnp.array([[3.],[4.]]),rewards=jnp.zeros(2))
    fake=Fake();ratios=jnp.zeros((4,1));rng=jax.random.PRNGKey(7)
    try:
        got=next_value_gfp(fake,batch,rng,ratios)
        equal(got,jnp.array([[17.,17.],[18.,18.]]))
        assert calls[1][0]=='value_state' and calls[1][1]==[[3.],[3.],[4.],[4.]]
        assert calls[2][0]=='vabc' and calls[2][1]==[[1.],[1.],[2.],[2.]]
        calls.clear();fake.config=dict(fake.config,target_use_next=True)
        equal(next_value_gfp(fake,batch,rng,ratios),jnp.array([[18.,18.],[19.,19.]]))
        assert calls[2][1]==[[3.],[3.],[4.],[4.]]
        calls.clear()
        class Direct(Fake):
            def sample_actions(self,o,seed):
                # Three explicit Euler steps fixture; one-step would yield a different value.
                a=jnp.zeros_like(o[:,:1])
                for i in range(3):a=a+(o[:,:1]+i)/3
                calls.append(('multistep',3));return a
        equal(next_value_dfql(Direct(),batch,rng,ratios),jnp.array([[1.,1.],[1.,1.]])) # final host clip
        assert calls[0]==('multistep',3)
    finally:core.backup_value=previous
    return dict(current_state_vabc_preserved=True,next_state_value=True,source_mean_order=True,dfql_multistep=True)

def main():
    check('Z2_Z3_Z4_Z5_Z6_synthetic_fixtures',fixtures)
    check('Z3_host_action_and_state_sources',target_sources)
    index=json.loads((ROOT/'configs/index.json').read_text())
    originals=[x for x in index if x.startswith('G0' if HOST=='gfp' else 'D0')]
    for filename in originals:
        meta=json.loads((ROOT/'configs'/filename).read_text());task=meta['task'];cfg=meta['agent']
        assert meta['execution_stage']=='prepare_and_verify'
        assert meta['training_authorized'] is False and meta['optimizer_steps_allowed']==0
        assert meta['online_learning_allowed'] is False
        combined=json.loads((ROOT/'configs'/filename.replace('0_','1_',1)).read_text())['agent']
        with np.load(ROOT/'data'/f'{task}.batch.npz') as z:
            batch={k:jnp.asarray(z[k]) for k in z.files}
        rng=jax.random.PRNGKey(81214)
        base_cfg=copy.deepcopy(cfg);base_cfg.pop('critic_type')
        baseline=Original.create(14014,batch['observations'],batch['actions'],base_cfg)
        scalar=create_agent(HOST,14014,batch['observations'],batch['actions'],cfg)
        before=[fingerprint(baseline),fingerprint(scalar)]
        def z0():
            assert type(scalar) is Original
            equal(baseline.network.params,scalar.network.params)
            equal(baseline.rng,scalar.rng)
            equal(baseline.sample_actions(batch['observations'],rng),scalar.sample_actions(batch['observations'],rng))
            for fn in ['critic_loss','actor_loss','total_loss']:
                left=jax.jit(jax.value_and_grad(lambda p:getattr(baseline,fn)(batch,p,rng),has_aux=True))(baseline.network.params)
                right=jax.jit(jax.value_and_grad(lambda p:getattr(scalar,fn)(batch,p,rng),has_aux=True))(scalar.network.params)
                equal(left,right);finite(left)
            assert before==[fingerprint(baseline),fingerprint(scalar)]
            return dict(atol=0,rtol=0,identical_rng=True,no_floq_params=not any('floq' in k for k in scalar.network.params))
        check(f'Z0_{task}',z0)
        del baseline,scalar; jax.clear_caches()
        agent=create_agent(HOST,14014,batch['observations'],batch['actions'],combined)
        state_before=fingerprint(agent);step_before=int(agent.network.step)
        def z1():
            out=agent.network.select('critic')(batch['observations'],batch['actions'])
            assert out.shape==(2,256)
            val,grad=jax.jit(jax.value_and_grad(lambda p:agent.total_loss(batch,p,rng),has_aux=True))(agent.network.params)
            finite((out,val,grad))
            actions=agent.sample_actions(batch['observations'],rng)
            assert actions.shape==batch['actions'].shape
            finite(actions)
            returns=integrate(agent,batch['observations'],batch['actions'],jnp.ones((256,1))*.5)
            assert returns.shape==(2,256,1);finite(returns)
            return dict(critic_shape=list(out.shape),flow_shape=list(returns.shape),loss=float(val[0]),
                        batch_sha256=hashlib.sha256((ROOT/'data'/f'{task}.batch.npz').read_bytes()).hexdigest(),
                        gradient_norms=gradient_norms(grad),device=str(out.device))
        check(f'Z1_{task}_full_loss_and_gradient_JIT',z1)
        def z2():
            c=agent.config;s=compute_support(c['q_min'],c['q_max'],c['num_bins'])
            targets=jnp.array([c['q_min'],c['q_max'],c['noise_min'],c['noise_max'],(c['q_min']+c['q_max'])/2])
            probs=to_probs(targets,s,c['sigma']*(s[1]-s[0]));finite(probs)
            np.testing.assert_allclose(np.asarray(probs.sum(-1)),1,atol=2e-6)
            return dict(q_support=[c['q_min'],c['q_max']],noise_range=[c['noise_min'],c['noise_max']],
                        host_discount=c['discount'],HL_Gauss_atol=2e-6,time_encoding_actual='cos(t), 1 channel; time_embed_dim unused')
        check(f'Z2_{task}_encoding',z2)
        if HOST=='dfql' or agent.config['target_agg']=='actor':
            def source_value_loss():
                # Execute the pinned source critic loss with the host action callback.
                # This checks the core port independently from our implementation.
                node=next(n for n in ast.parse((ROOT/'upstream_snapshots/floq/agents/floq.py').read_text()).body
                          if isinstance(n,ast.ClassDef) and n.name=='FloQAgent')
                fn=copy.deepcopy(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='critic_loss'))
                scope={'jax':jax,'jnp':jnp};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_floq_critic_loss','exec'),scope)
                from types import SimpleNamespace
                carrier=SimpleNamespace(config=agent.config,network=agent.network,sample_actions=agent.sample_actions,
                    compute_current_flow_returns=lambda o,a,noise_ratios:integrate(agent,o,a,noise_ratios),
                    compute_target_flow_returns=lambda o,a,noise_ratios:aggregate_ensemble(
                        integrate(agent,o,a,noise_ratios,target=True),agent.config['q_agg']))
                ref=jax.jit(jax.value_and_grad(lambda p:scope['critic_loss'](carrier,batch,p,rng),has_aux=True))(agent.network.params)
                port=jax.jit(jax.value_and_grad(lambda p:agent.critic_loss(batch,p,rng),has_aux=True))(agent.network.params)
                equal(ref[0][0],port[0][0],atol=2e-5,rtol=2e-5)
                equal(ref[1],port[1],atol=2e-5,rtol=2e-5)
                for key in ['floq_loss','distilled_critic_loss']:
                    equal(ref[0][1][key],port[0][1][key],atol=2e-5,rtol=2e-5)
                return dict(atol=2e-5,rtol=2e-5,reference='AST-extracted pinned FloQAgent.critic_loss',
                            host_next_action_callback=True,loss_and_all_parameter_gradients_compared=True)
            check(f'Z4_{task}_pinned_floq_loss_control',source_value_loss)
        def z4():
            routes={}
            routes['Flow Critic']=route(agent,batch,rng,'floq_loss',{'modules_floq'})
            routes['Scalar distillation']=route(agent,batch,rng,'distilled_critic_loss',{'modules_critic'})
            if HOST=='gfp':
                routes['Actor Q']=route(agent,batch,rng,'q_loss',{'modules_actor'},actor=True)
                routes['Actor endpoint distillation']=route(agent,batch,rng,'distill_loss',{'modules_actor'},actor=True)
                routes['VaBC']=route(agent,batch,rng,'VaBC_loss',{'modules_vabc'},actor=True)
                # References and weight computation close over frozen stored parameters.
                guide_grad=jax.grad(lambda p:agent.compute_guidance(batch,
                    jax.lax.stop_gradient(agent.network.select('actor')(batch['observations'],batch['actions'])),
                    agent.compute_flow_actions(batch['observations'],batch['actions']),jnp.array(1.))[0].sum())(agent.network.params)
                assert all(x==0 for x in gradient_norms(guide_grad).values())
                routes['Guidance frozen']=gradient_norms(guide_grad)
            else:
                routes['DFQL BC']=route(agent,batch,rng,'bc_flow_loss',{'modules_actor_bc_flow'},actor=True)
                routes['DFQL terminal Q guidance']=route(agent,batch,rng,'q_loss',{'modules_actor_bc_flow'},actor=True)
                assert 'modules_actor_onestep_flow' not in agent.network.params
                assert 'modules_target_critic' not in agent.network.params
            return routes
        check(f'Z4_{task}_parameter_gradient_routes',z4)
        def z5():
            # The methods used in the combination are the exact original host functions.
            for method in ['actor_loss','sample_actions','compute_flow_actions']:
                assert getattr(type(agent),method) is getattr(Original,method)
            if HOST=='gfp':
                assert type(agent).compute_guidance is Original.compute_guidance
                assert agent.config['target_use_next']==False
            else:assert agent.config['actor_type']=='DFQL'
            return dict(host_methods_inherited_verbatim=True,actor_type='one-step + VaBC' if HOST=='gfp' else 'multi-step tanh velocity')
        check(f'Z5_{task}_host_actor_preserved',z5)
        def z7():
            encoded=flax.serialization.to_bytes(agent)
            dest=ROOT/'tests_zero_train'/f'{HOST}_{task}.msgpack';dest.write_bytes(encoded)
            restored=flax.serialization.from_bytes(agent,dest.read_bytes())
            assert fingerprint(restored)==state_before
            equal(restored.sample_actions(batch['observations'],rng),agent.sample_actions(batch['observations'],rng))
            return dict(path=str(dest),sha256=hashlib.sha256(encoded).hexdigest(),random_init_only=True,old_whole_agent_compatible=False)
        check(f'Z7_{task}_new_combination_roundtrip',z7)
        def z8():
            assert state_before==fingerprint(agent);assert step_before==int(agent.network.step)
            assert RESULT['forbidden_call_count']==0
            return dict(state_sha256_before=state_before,state_sha256_after=fingerprint(agent),
                        initialized_state_step=step_before,final_state_step=int(agent.network.step),optimizer_steps=0,
                        params_rng_optimizer_unchanged=True)
        check(f'Z8_{task}_zero_update_state_proof',z8)
        del agent,batch;jax.clear_caches()
    RESULT['status']='ZERO_TRAIN_VERIFIED';dump();print('HOST_VERIFIED',HOST,flush=True)

if __name__=='__main__':main()
