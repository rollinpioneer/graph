"""Frozen-checkpoint inference only. No environment creation or optimizer calls."""
import os,sys,time,json,pickle,hashlib,resource,subprocess,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016')
sys.path.insert(0,str(SOURCE/'code'))
from common import make_agent,sha
import jax,jax.numpy as jnp,flax
from agents.gfp import GFPAgent
from bfc import target

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def digest(tree):
 h=hashlib.sha256()
 for leaf in jax.tree_util.tree_leaves(tree):
  a=np.asarray(leaf);h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
 return h.hexdigest()
def stats(a):
 a=np.asarray(a,dtype=np.float64)
 return dict(mean=float(a.mean()),mean_abs=float(np.abs(a).mean()),rms=float(np.sqrt((a*a).mean())),q05=float(np.quantile(a,.05)),median=float(np.median(a)),q95=float(np.quantile(a,.95)))
@jax.jit
def reference_actions(ag,obs,key):
 p=dict(ag.network.params);p['modules_actor']=ag.reference
 ref=GFPAgent(ag.rng,ag.network.replace(params=p),ag.config)
 return ref.sample_actions(obs,key)
@jax.jit
def q_eval(ag,obs,action):return ag.critic_eval(obs,action)
@jax.jit
def native(ag,batch,key):return ag.td_parts(batch,key)

started=time.time();cpu0=resource.getrusage(resource.RUSAGE_SELF)
assets=json.loads((SOURCE/'receipts/assets.json').read_text());n=assets['dataset_size']
data={k:np.load(SOURCE/'data'/f'{k}.npy',mmap_mode='r') for k in ('observations','next_observations','actions','rewards','masks')}
data_hashes={k:sha(SOURCE/'data'/f'{k}.npy') for k in data}
assert all(data_hashes[k]==assets['data'][k]['sha256'] for k in data)
ij=np.floor((data['observations'][:n,:2]+2)/4).astype(int)+1
is_start=np.array([False]*n)
for i,j in ((1,1),(2,1),(1,2),(2,2)):is_start|=(ij[:,0]==i)&(ij[:,1]==j)
rng=np.random.default_rng(20261016)
indices=np.concatenate([rng.choice(np.flatnonzero(is_start),2048,replace=False),rng.choice(np.flatnonzero(~is_start),2048,replace=False)])
batch_np={k:np.asarray(v[indices]).copy() for k,v in data.items()}
np.savez_compressed(ROOT/'arrays/fixed_inputs.npz',indices=indices,**batch_np)
input_info=dict(indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),file_sha256=sha(ROOT/'arrays/fixed_inputs.npz'),path=str(ROOT/'arrays/fixed_inputs.npz'),available_start=int(is_start.sum()),available_outside=int((~is_start).sum()),selected_per_region=2048,region_definition='current observation cells; outside is not verified normal',noise_key_bases=[202610160,202610161],batch_size=256,data_hashes=data_hashes,terminal_selected=int((batch_np['masks']==0).sum()))
write(ROOT/'receipts/inputs.json',input_info)
gamma=.995;epsilon_specs=json.loads((SOURCE/'protocol.json').read_text())['arms']
rows=[];validation=[];resources=[];identities=[]
max_native_error=0.;max_reconstruction_error=0.
# A synthetic terminal check is pure arithmetic, no simulator.
yy,_,_=target(jnp.array([-1.]),jnp.array([0.]),gamma,jnp.array([12.]),jnp.array([-8.]),jnp.array([4.]),.1)
assert float(yy[0])==-1.
for seed in (0,3):
 for arm in ('R2','R3','R4','R5'):
  arm_start=time.time();models=[];hashes=[];paths=[]
  for offset in (0,50000,100000):
   ag,cfg,source_path=make_agent(seed,arm)
   assert ag.config['target_agg']=='mean' and ag.config['q_agg']=='mean' and not ag.config['target_use_next'] and ag.config['discount']==gamma
   path=source_path if offset==0 else SOURCE/'runs'/f's{seed}_{arm}'/f'checkpoint_{offset}.pkl'
   if offset:
    with open(path,'rb') as f:p=pickle.load(f)
    ag=flax.serialization.from_state_dict(ag,p['agent'])
    assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
   else:assert sha(path)==assets['sources'][str(seed)]['sha256']
   assert int(ag.network.opt_state[0].count)=={0:600000,3:300000}[seed]+offset
   h=digest(ag);models.append(ag);hashes.append(h);paths.append((path,sha(path)))
   identities.append(dict(seed=seed,arm=arm,offset=offset,path=str(path),file_sha256=paths[-1][1],full_state_digest=h))
  # q[noise repetition, critic checkpoint, action checkpoint, type, state]
  # types: reference Actor, VaBC, current Actor.
  q=np.empty((2,3,3,3,4096),np.float64)
  ep_epsilon=epsilon_specs[arm]['epsilon']
  for rep,keybase in enumerate((202610160,202610161)):
   for bi,start in enumerate(range(0,4096,256)):
    sl=slice(start,start+256);batch={k:jnp.asarray(v[sl]) for k,v in batch_np.items()};key=jax.random.fold_in(jax.random.PRNGKey(keybase),bi);_,sk=jax.random.split(key)
    generated=[]
    for a in models:
     generated.append((reference_actions(a,batch['next_observations'],sk),a.sample_flow_actions(batch['observations'],sk),a.sample_actions(batch['next_observations'],sk)))
    for c,a in enumerate(models):
     for t in range(3):
      for typ in range(3):q[rep,c,t,typ,sl]=np.asarray(q_eval(a,batch['next_observations'],generated[t][typ]))
    for t,a in enumerate(models):
     ref=batch_np['rewards'][sl]+gamma*batch_np['masks'][sl]*(q[rep,t,t,0,sl]+q[rep,t,t,1,sl])/2
     delta=gamma*batch_np['masks'][sl]*(q[rep,t,t,2,sl]-q[rep,t,t,0,sl])/2
     actual=np.asarray(native(a,batch,key)[0]);err=float(np.max(np.abs(ref+np.clip(delta,-ep_epsilon,ep_epsilon)-actual)))
     max_native_error=max(max_native_error,err);assert err<1e-4,(seed,arm,t,err)
    assert np.isfinite(q[rep,:,:,:,sl]).all()
    if rep==0 and bi==0:
     write(ROOT/'receipts/first_batch_validation.json',dict(seed=seed,arm=arm,max_native_abs_error=max_native_error,passed=True,environment_steps=0,optimizer_updates=0))
  assert np.max(np.abs(q[:,0,0,0]-q[:,0,0,2]))<1e-6
  # Three factors: A reference Actor, V VaBC, C target Critic.
  for rep in range(2):
   for old,new in ((0,1),(1,2),(0,2)):
    y={}
    for bits in itertools.product((0,1),repeat=3):
     at,vt,ct=[new if b else old for b in bits]
     y[bits]=batch_np['rewards']+gamma*batch_np['masks']*(q[rep,ct,at,0]+q[rep,ct,vt,1])/2
    effects={}
    names=['A','V','C']
    for mask in range(1,8):
     value=np.zeros(4096)
     for sub in range(8):
      if sub&~mask:continue
      bits=tuple((sub>>k)&1 for k in range(3))
      value+=(-1)**(mask.bit_count()-sub.bit_count())*y[bits]
     effects[''.join(names[k] for k in range(3) if (mask>>k)&1)]=value
    total=y[1,1,1]-y[0,0,0]
    err=float(np.max(np.abs(sum(effects.values())-total)));max_reconstruction_error=max(max_reconstruction_error,err);assert err<1e-8
    assert max(np.abs(effects['AV']).max(),np.abs(effects['AVC']).max())<1e-8
    d=[];accepted=[]
    for t in (old,new):
     z=gamma*batch_np['masks']*(q[rep,t,t,2]-q[rep,t,t,0])/2;d.append(z);accepted.append(np.clip(z,-ep_epsilon,ep_epsilon))
    for region,sl in [('start',slice(0,2048)),('outside',slice(2048,4096))]:
     metrics=dict(reference_target_drift=stats(total[sl]),accepted_increment_drift=stats((accepted[1]-accepted[0])[sl]),actual_target_drift=stats((total+accepted[1]-accepted[0])[sl]),
      raw_delta_before=stats(d[0][sl]),raw_delta_after=stats(d[1][sl]),accepted_before=stats(accepted[0][sl]),accepted_after=stats(accepted[1][sl]),
      clipped_fraction_before=float((np.abs(d[0][sl])>ep_epsilon).mean()),clipped_fraction_after=float((np.abs(d[1][sl])>ep_epsilon).mean()),
      main_and_interactions={k:stats(v[sl]) for k,v in effects.items()},shapley={k:stats(sum(v/len(name) for name,v in effects.items() if k in name)[sl]) for k in names})
     rows.append(dict(seed=seed,arm=arm,noise_repeat=rep,old_offset=(0,50000,100000)[old],new_offset=(0,50000,100000)[new],region=region,**metrics))
  np.savez_compressed(ROOT/'arrays'/f'q_cross_s{seed}_{arm}.npz',q=q)
  for a,h,(path,filehash) in zip(models,hashes,paths):assert digest(a)==h and sha(path)==filehash
  validation.append(dict(seed=seed,arm=arm,unchanged_full_state=True,unchanged_checkpoint_files=True,optimizer_updates=0,environment_steps=0))
  try:gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True);own=[line for line in gpu.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
  except Exception:own=[]
  resources.append(dict(seed=seed,arm=arm,seconds=time.time()-arm_start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,own_gpu_memory_sample=own))
  write(ROOT/'reports/decomposition.json',rows);write(ROOT/'receipts/progress.json',dict(completed=len(validation),total=8,last=[seed,arm],seconds=time.time()-started))
  print('COMPLETE',seed,arm,round(time.time()-arm_start,2),flush=True)
write(ROOT/'receipts/identity.json',identities)
write(ROOT/'receipts/validation.json',dict(passed=True,branches=validation,max_native_target_abs_error=max_native_error,max_reconstruction_abs_error=max_reconstruction_error,
  additive_actor_vabc_and_triple_interactions_zero=True,synthetic_terminal_pass=True,paired_noise_repeats=2,total_sampled_transitions=4096,region_counts=[2048,2048],optimizer_updates=0,environment_steps=0))
write(ROOT/'receipts/cost.json',dict(seconds=time.time()-started,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime-cpu0.ru_utime-cpu0.ru_stime,
 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),samples=resources,optimizer_updates=0,environment_steps=0))
print('ALL_DONE',time.time()-started,flush=True)
