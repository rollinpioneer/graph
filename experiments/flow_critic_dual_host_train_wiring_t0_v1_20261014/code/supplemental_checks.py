"""Post-T0 guard/compiler/full-RNG IO checks. Performs zero optimizer updates."""
import argparse,ast,copy,hashlib,json,random,sys,time
from pathlib import Path
import flax,jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--host',choices=['gfp','dfql'],required=True);a=parser.parse_args();host=a.host
sys.path.insert(0,str(ROOT/'workspaces'/f'{host}_floq'))
from agents.floq_adapter import create_agent
from training_wiring import save_state,restore_state,state_digest,read_execution_gate
runtime=json.loads((ROOT/'receipts'/f'{host}_runtime.json').read_text());cfg=runtime['config'];task=runtime['gate']['task']
with np.load(ROOT/'data'/f'{task}.batch.npz') as z:batch={k:jnp.asarray(z[k][:2]) for k in z.files}
results=[]
def record(name,details):results.append(dict(test=name,status='PASS',details=details));print('SUPPLEMENT_PASS',host,name,flush=True)
def equal(x,y):
 aa,ta=jax.tree_util.tree_flatten(x);bb,tb=jax.tree_util.tree_flatten(y);assert ta==tb
 for av,bv in zip(aa,bb):np.testing.assert_array_equal(np.asarray(av),np.asarray(bv))

agent=create_agent(host,0,batch['observations'],batch['actions'],cfg)
agent=restore_state(agent,runtime['output_dir'],2);before=state_digest(agent)
assert int(agent.network.step)==3 and int(agent.network.opt_state[0].count)==2
for name,call in [
 ('update',lambda:agent.update(batch)),
 ('batch_update',lambda:agent.batch_update({k:v[None] for k,v in batch.items()})),
 ('multi_sample_and_update',lambda:agent.multi_sample_and_update(1,batch,2,2)),
 ('target_update',lambda:agent.target_update(agent.network))]:
 try:call()
 except RuntimeError as e:assert 'UPDATE_BUDGET_EXHAUSTED' in str(e)
 else:raise AssertionError(name+' not blocked')
 assert state_digest(agent)==before
record('lifetime_budget_guard_all_four_APIs',{'completed_updates':2,'blocked_without_extra_update':True})
disabled=agent.replace(config=flax.core.FrozenDict(dict(agent.config,training_authorized=False)))
try:disabled.update(batch)
except RuntimeError as e:assert 'NOT_AUTHORIZED' in str(e)
else:raise AssertionError('Closed gate not enforced')
record('explicit_authorization_guard',{'state_unchanged':state_digest(agent)==before})

# Save/restore exact full state plus the CPU dataset and Python RNG.
np.random.seed(1014);random.seed(1014)
folder=ROOT/'runs'/host/'supplemental_full_rng_checkpoint'
save_state(agent,folder,2)
expected_np=np.random.randint(0,1000000,size=32);expected_py=[random.random() for _ in range(8)]
restored=restore_state(agent,folder,2)
np.testing.assert_array_equal(expected_np,np.random.randint(0,1000000,size=32))
assert expected_py==[random.random() for _ in range(8)]
assert state_digest(restored)==before
record('full_checkpoint_optimizer_model_and_all_offline_RNG',{'state_step':3,'adam_count':2,'numpy_and_python_rng_exact':True,
    'path':str(folder),'legacy_native_checkpoint_model_state_restored':True})

# Pure baseline checks: the original class AST is unchanged, scalar constructor path exact.
original_module=__import__('agents.'+host,fromlist=['dummy'])
Original=getattr(original_module,'GFPAgent' if host=='gfp' else 'DFQLAgent')
upstream=ROOT/'upstream_snapshots'/host/'agents'/f'{host}.py'
actual=ROOT/'workspaces'/f'{host}_floq'/'agents'/f'{host}.py'
def klass(p):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==Original.__name__)
assert ast.dump(klass(upstream))==ast.dump(klass(actual))
prefix='G0' if host=='gfp' else 'D0'
base_cfg=json.loads((ROOT/'configs'/f'{prefix}_{task}.json').read_text())['agent'];base_cfg.pop('critic_type')
base=Original.create(1014,batch['observations'],batch['actions'],copy.deepcopy(base_cfg))
scalar=create_agent(host,1014,batch['observations'],batch['actions'],dict(base_cfg,critic_type='scalar'))
assert type(scalar) is Original;equal(base.network.params,scalar.network.params);equal(base.rng,scalar.rng)
key=jax.random.PRNGKey(1014)
equal(base.sample_actions(batch['observations'],key),scalar.sample_actions(batch['observations'],key))
equal(base.total_loss(batch,base.network.params,key),scalar.total_loss(batch,scalar.network.params,key))
record('original_baseline_class_and_scalar_path_preserved',{'AST_identical':True,'atol':0,'rtol':0,'baseline_optimizer_updates':0})
del base,scalar;jax.clear_caches()

# Compile only: no returned updated Agent is executed or assigned.
stack={k:jnp.stack([v,v]) for k,v in batch.items()}
agent._batch_update_impl.lower(agent,stack).compile()
agent._multi_sample_impl.lower(agent,2,batch,2,2).compile()
assert state_digest(agent)==before
record('batch_and_fused_optimizer_graphs_compile_only',{'steps_symbolic':2,'extra_optimizer_updates':0,'device':[str(d) for d in jax.devices()]})

for steps in [3,1000000]:
 try:read_execution_gate(ROOT/'configs'/f'{host}_t0_gate.json',host,task,'floq',steps,0,0,0)
 except RuntimeError as e:assert 'UPDATE_BUDGET_EXHAUSTED' in str(e)
 else:raise AssertionError('Unapproved step count accepted')
record('short_and_full_training_not_authorized_by_T0_gate',{'blocked_update_counts':[3,1000000]})
(ROOT/'reports'/f'{host}_supplemental_checks.json').write_text(json.dumps(dict(status='PASS',host=host,tests=results,
 actual_optimizer_updates_in_supplement=0,verified_historical_T0_updates=2),indent=2))
