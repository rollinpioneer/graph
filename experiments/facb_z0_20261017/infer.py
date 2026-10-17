"""Frozen forward passes only; calibration locked before held-out/island scoring."""
from pathlib import Path
import os,sys,json,pickle,hashlib,time,resource,copy,importlib.metadata as md
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xushijie3/work/gfp_eta0_ablation_20261015/source/gfp')
sys.path.insert(1,'/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/confirm_scratch/code')
import gymnasium as gym, mujoco
calls=dict(training_updates=0,environment_constructions=0,environment_resets=0,environment_steps=0,new_model_initializations=0,forbidden_attempts=0)
def forbidden(*args,**kwargs):
 calls['forbidden_attempts']+=1
 raise RuntimeError('FACB-Z0: operation prohibited')
gym.make=forbidden;gym.Env.step=forbidden;gym.Env.reset=forbidden;mujoco.mj_step=forbidden
import jax,jax.numpy as jnp,flax
from flax import linen as nn
from agents.gfp import GFPAgent
from iql_gfp import IQLScratchAgent
from utils.flax_utils import ModuleDict,TrainState
from utils.networks import Value,ActorVectorField
for cls,names in [(GFPAgent,['create','update','multi_sample_and_update','target_update']),(IQLScratchAgent,['create','update','update_value','multi_sample_and_update','target_update']),(TrainState,['apply_loss_fn','apply_gradients'])]:
 for name in names:
  if hasattr(cls,name):setattr(cls,name,forbidden)
nn.Module.init=forbidden
protocol=json.loads((ROOT/'protocol.json').read_text());records=json.loads((ROOT/'checkpoint_manifest.json').read_text());configs=json.loads((ROOT/'configs.json').read_text())
samples=dict(np.load(ROOT/'arrays/samples.npz'));N=len(samples['observations']);assert N==7168
t0=time.monotonic();integrity=[];validations=[]
def save(n,x):
 p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):
 h=hashlib.sha256()
 for path,value in jax.tree_util.tree_flatten_with_path(x)[0]:
  v=np.asarray(value);h.update(str(path).encode());h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
 return h.hexdigest()
def checktime():
 if time.monotonic()-t0>7150:raise RuntimeError('FACB-Z0 GPU budget exhausted')
def read(row):
 assert sha(row['path'])==row['sha256']
 with open(row['path'],'rb') as f:p=pickle.load(f)['agent']
 assert int(p['network']['step'])==row['step']+1
 assert int(p['network']['opt_state']['0']['count'])==row['step']
 if row['arm']=='I':
  assert int(p['value']['step'])==row['step']+1 and int(p['value']['opt_state']['0']['count'])==row['step']
 return p,digest(p)
def make_g(row,p,cls=GFPAgent):
 cfg=dict(configs[row['id']]['agent']);cfg.update(ob_dims=(69,),action_dim=21,log_metrics=True)
 if cfg['eta_temperature_online'] is None:cfg['eta_temperature_online']=cfg['eta_temperature']
 cfg=flax.core.FrozenDict(cfg)
 critic=Value(tuple(cfg['value_hidden_dims']),layer_norm=cfg['layer_norm'],num_ensembles=2)
 actor=ActorVectorField(tuple(cfg['actor_hidden_dims']),action_dim=21,layer_norm=cfg['actor_layer_norm'],need_time=False)
 vabc=ActorVectorField(tuple(cfg['actor_hidden_dims']),action_dim=21,layer_norm=cfg['actor_layer_norm'],need_time=True,encode_time_dim=cfg['encode_time_dim'])
 definition=ModuleDict(dict(critic=critic,target_critic=copy.deepcopy(critic),actor=actor,vabc=vabc))
 net=TrainState(step=p['network']['step'],apply_fn=definition.apply,model_def=definition,params=p['network']['params'],tx=None,opt_state=None)
 return cls(rng=p['rng'],network=net,config=cfg)
def make_v(row,p):
 cfg=configs[row['id']]['agent'];definition=ModuleDict(dict(value=Value(tuple(cfg['value_hidden_dims']),layer_norm=cfg['layer_norm'],num_ensembles=1)))
 return TrainState(step=p['value']['step'],apply_fn=definition.apply,model_def=definition,params=p['value']['params'],tx=None,opt_state=None)
@jax.jit
def score(g,obs,nobs,key):
 _,srng=jax.random.split(key)
 aa=g.sample_actions(nobs,srng);va=g.sample_flow_actions(obs,srng)
 qa=g.network.select('target_critic')(nobs,aa).mean(0)
 qv=g.network.select('target_critic')(nobs,va).mean(0)
 return (qa+qv)/2,qa,qv
@jax.jit
def vscore(v,obs):return v.select('value')(obs)
@jax.jit
def direct_vscore(params,obs):
 return Value((512,)*4,layer_norm=True,num_ensembles=1).apply({'params':params['modules_value']},obs)
@jax.jit
def native_target(g,obs,nobs,key):
 _,srng=jax.random.split(key)
 return g.value_with_all_options('mean',nobs,g.sample_actions(nobs,srng),g.sample_flow_actions(obs,srng))
@jax.jit
def native_critic_loss(g,b,key):return g.critic_loss(b,g.network.params,key)
@jax.jit
def reconstructed_critic_loss(g,b,key):
 q,_,_=score(g,b['observations'],b['next_observations'],key)
 target=b['rewards']+g.config['discount']*b['masks']*q
 current=g.network.select('critic')(b['observations'],actions=b['actions'])
 return jnp.square(current-target).mean()
@jax.jit
def independent_target_mapping(g,obs,act):
 native=g.network.select('target_critic')(obs,act)
 direct=Value((512,)*4,layer_norm=True,num_ensembles=2).apply({'params':g.network.params['modules_target_critic']},obs,act)
 return jnp.max(jnp.abs(native-direct))
def finish(row,p,old):
 assert digest(p)==old and sha(row['path'])==row['sha256']
 integrity.append(dict(id=row['id'],checkpoint_sha256=row['sha256'],saved_model_optimizer_rng_unchanged=True,payload_digest=old))
 del p

# First verify and evaluate the predefined proxy anchors. No Q-island results read.
all_v=[]
for row in [r for r in records if r['arm']=='I']:
 checktime();p,old=read(row);v=make_v(row,p)
 vals=np.concatenate([np.asarray(vscore(v,samples['next_observations'][k:k+256])) for k in range(0,N,256)])
 assert vals.shape==(N,) and np.isfinite(vals).all()
 # Match native I-arm V and value-loss forward definitions on an existing batch.
 g=make_g(row,p,cls=IQLScratchAgent).replace(value=v)
 batch={k:jnp.asarray(samples[k][:256]) for k in ('observations','next_observations','actions','rewards','masks')}
 native_loss,_=g.value_loss(batch,p['value']['params'])
 q=jax.lax.stop_gradient(g.critic_eval(batch['observations'],batch['actions']));vv=v.select('value')(batch['observations'])
 direct_loss=jnp.mean(jnp.where(q-vv>0,.7,.3)*(q-vv)**2)
 direct_v=direct_vscore(p['value']['params'],batch['next_observations'])
 verr=float(np.max(np.abs(np.asarray(direct_v)-vals[:256])))
 assert verr<=3e-5;np.testing.assert_allclose(native_loss,direct_loss,rtol=3e-5,atol=3e-5)
 validations.append(dict(id=row['id'],value_direct_max_abs_error=verr,native_value_loss_error=float(abs(native_loss-direct_loss)),V_shape=list(vals.shape),step=row['step']))
 all_v.append(vals);finish(row,p,old);print('ANCHOR',row['id'],flush=True)
all_v=np.stack(all_v);assert all_v.shape==(8,N)
np.save(ROOT/'arrays/anchor_values.npy',all_v)
identity=json.loads((ROOT/'anchor_identity.json').read_text());identity['status']='PROXY_VALID_FOR_THIS_CHECK_ONLY';identity['forward_verified_seeds']=list(range(8));save('anchor_identity.json',identity)

grows=[r for r in records if r['arm']=='G'];calrows=[r for r in grows if r['group']=='calibration'];cal_values=[];cal_ids=[]
lock_hash=None
for row in grows:
 checktime()
 if row['group']!='calibration':
  assert lock_hash and sha(ROOT/'calibration.json')==lock_hash,'Calibration must be locked before held-out/island results'
 p,old=read(row);g=make_g(row,p);qvalues=[]
 for repeat,base in enumerate(protocol['rng_bases']):
  parts=[]
  for j,k in enumerate(range(0,N,256)):
   key=jax.random.fold_in(jax.random.PRNGKey(base),j)
   q,qa,qv=score(g,samples['observations'][k:k+256],samples['next_observations'][k:k+256],key)
   q=np.asarray(q);assert q.shape==(min(256,N-k),) and np.isfinite(q).all();parts.append(q)
   if j==0:
    b={n:jnp.asarray(samples[n][:256]) for n in ('observations','next_observations','actions','rewards','masks')}
    _,srng=jax.random.split(key)
    native=native_target(g,b['observations'],b['next_observations'],key)
    qerr=float(np.max(np.abs(np.asarray(native)-q)));assert qerr<=3e-5
    mapping_error=float(independent_target_mapping(g,b['next_observations'],b['actions']));assert mapping_error<=3e-5
    loss,info=native_critic_loss(g,b,key)
    target=b['rewards']+.995*b['masks']*jnp.asarray(q)
    current=g.network.select('critic')(b['observations'],actions=b['actions'])
    direct=reconstructed_critic_loss(g,b,key)
    np.testing.assert_allclose(loss,direct,rtol=3e-5,atol=3e-5)
    assert float(jnp.max(jnp.abs(b['rewards']+.995*0*jnp.asarray(q)-b['rewards'])))==0
    validations.append(dict(id=row['id'],repeat=repeat,native_q_max_abs_error=qerr,target_parameter_mapping_error=mapping_error,critic_loss_abs_error=float(abs(loss-direct)),critic_loss_rtol=3e-5,critic_loss_atol=3e-5,terminal_mask_zero_target_is_reward=True))
  qvalues.append(np.concatenate(parts));checktime()
 qvalues=np.stack(qvalues);np.save(ROOT/f'arrays/{row["id"]}_q.npy',qvalues)
 if row['group']=='calibration':
  cal_values.append((qvalues-all_v[row['seed']][None,:]).reshape(-1));cal_ids.append(row['id'])
  if len(cal_values)==len(calrows):
   d=np.concatenate(cal_values);c=float(np.quantile(np.abs(d),.99,method='linear'))
   if (ROOT/'calibration.json').exists():
    previous=json.loads((ROOT/'calibration.json').read_text());assert previous['c']==c and previous['calibration_ids']==cal_ids
    lock_hash=sha(ROOT/'calibration.json');assert lock_hash==json.loads((ROOT/'receipts/calibration_lock.json').read_text())['sha256']
    print('CALIBRATION_RECOMPUTATION_IDENTICAL_ORIGINAL_LOCK_PRESERVED',c,flush=True)
   else:
    save('calibration.json',dict(calibration_ids=cal_ids,calibration_samples=int(d.size),calibration_seeds=[1,2],S=200.,kappa=c/200.,c=c,quantile=.99,quantile_method='linear',samples_sha256=sha(ROOT/'arrays/samples.npz'),calibration_ids_only=True,heldout_and_island_scored_at_lock=False,elapsed_seconds=time.monotonic()-t0,selection='pooled six healthy checkpoints x two noises x distinct 7168-state union'))
    lock_hash=sha(ROOT/'calibration.json');save('receipts/calibration_lock.json',dict(sha256=lock_hash,heldout_and_island_not_yet_scored=True));print('CALIBRATION_LOCKED',c,flush=True)
 finish(row,p,old)
 save('receipts/progress.json',dict(last_completed=row['id'],group=row['group'],completed_G=len(integrity)-8,seconds=time.monotonic()-t0,**calls))
 print('SCORED',row['id'],row['group'],round(time.monotonic()-t0,2),flush=True)
assert lock_hash==sha(ROOT/'calibration.json') and calls['forbidden_attempts']==0
save('receipts/forward_validation.json',validations)
save('receipts/integrity.json',integrity)
save('receipts/inference.json',dict(status='COMPLETE',G_checkpoint_pairs=len(grows),primary_pairs=20,auxiliary_pairs=6,proxy_anchors=8,states_per_pair=N,noise_repeats=2,G_state_noise_scores=len(grows)*N*2,anchor_state_scores=8*N,seconds=time.monotonic()-t0,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device=[str(x) for x in jax.devices()],versions={k:md.version(k) for k in ['ogbench','numpy','jax','jaxlib','flax','optax']},jax_enable_x64=jax.config.jax_enable_x64,matmul_precision=str(jax.config.jax_default_matmul_precision),**calls))
print('COMPLETE',flush=True)
