"""Bounded takeover diagnostic; reuses exact saved states and global-time RNG."""
import os,sys,ast,json,time,random,pickle,copy,hashlib,traceback,resource
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R.parent/'gfp_task4_same_state_v1_20261016';P=R.parent/'gfp_task4_baseline_pilot_v1_20261015'
os.environ.update(COMPARATIVE_ROOT=str(P),COMPARATIVE_JOB='GFP_seed1')
sys.path.insert(0,str(P/'workspaces/gfp'))
import numpy as np,jax,mujoco,ogbench
from importlib.metadata import version
from agents import agents
from utils.flax_utils import restore_agent,TrainState
from training_wiring import restore_state,state_digest,digest
from ogbench.locomaze.humanoid import HumanoidEnv

assert (R/'receipts/protocol_frozen.json').exists()
with (R/'receipts/started.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
t0=time.time();counts=dict(branch_external_steps=0,validation_external_steps=0,physical_control_steps=0,mujoco_integration_frames=0,optimizer_updates=0);stage='initialization';results=[];env=None
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def status():save(R/'receipts/progress.json',dict(stage=stage,branches=len(results),elapsed=time.time()-t0,**counts))
def forbidden(*a,**k):raise RuntimeError('OPTIMIZER_UPDATE_FORBIDDEN')
for cls in [agents['gfp'],agents['rebrac']]:
 for n in ['update','batch_update','multi_sample_and_update','target_update']:
  if hasattr(cls,n):setattr(cls,n,forbidden)
for n in ['apply_loss_fn','apply_gradients']:
 if hasattr(TrainState,n):setattr(TrainState,n,forbidden)

# Use the previously verified implementation verbatim, without its executable main.
source=S/'code/probe.py';assert digest(source)=='c7620a08de43a90dddf145c4c39bfb05dfc6cdadf9b13ae20f9f5dac644d3392'
names={'verify_assets','action','wrappers','restore','physical_record'}
tree=ast.parse(source.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==len(names)
exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'),globals())

physical=HumanoidEnv._step_mujoco_simulation
def counted_sim(self,ctrl,n_frames):
 if counts['physical_control_steps']>=3320:raise RuntimeError('PHYSICAL_CAP')
 physical(self,ctrl,n_frames);counts['physical_control_steps']+=1;counts['mujoco_integration_frames']+=int(n_frames)
HumanoidEnv._step_mujoco_simulation=counted_sim
def step(a,kind):
 key='branch_external_steps' if kind=='branch' else 'validation_external_steps';cap=3200 if kind=='branch' else 20
 if counts[key]>=cap:raise RuntimeError('EXTERNAL_CAP')
 counts[key]+=1
 return env.step(a)

def compare(z,h,ob,a,r,te,tr):
 errors={n:float(np.max(np.abs(np.asarray(v)-z[n][h+1]))) for n,v in [('observations',ob),('qpos',u.data.qpos),('qvel',u.data.qvel)]}
 errors['actions']=float(np.max(np.abs(a-z['actions'][h])))
 assert all(v<=(1e-7 if n=='actions' else 1e-8) for n,v in errors.items()),errors
 assert r==z['rewards'][h] and te==z['terminated'][h] and tr==z['truncated'][h]
 return errors

try:
 assert version('ogbench')=='1.1.0' and mujoco.__version__=='3.3.5' and jax.__version__=='0.6.2'
 assert not jax.config.jax_enable_x64
 c=json.loads((R/'configs/protocol.json').read_text());state=c['state'];snapshot=S/'states'/f'{state}.pkl'
 inputs=json.loads((R/'configs/input_manifest.json').read_text())
 for x in inputs:assert digest(Path(x['path']))==x['sha256'],x['path']
 models={};assets=[];samples=np.load(P/'arrays/train_samples.npz')
 for x in json.loads((S/'receipts/assets_s1.json').read_text()):
  host=x['host'];cfg=x['config'];a=agents[host].create(1,samples['observations'][:1],samples['actions'][:1],cfg);directory=Path(x['checkpoint']).parent
  a=restore_state(a,directory,1000000) if host=='gfp' else restore_agent(a,str(directory),1000000)
  assert digest(Path(x['checkpoint']))==x['checkpoint_sha256'] and state_digest(a)==x['state_sha256'] and int(a.network.step)==1000001
  models[host]=a;assets.append(x)
 save(R/'receipts/assets.json',assets)
 save(R/'receipts/device.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),devices=[str(d) for d in jax.devices()],versions={k:version(k) for k in ['ogbench','mujoco','jax','jaxlib','flax','optax','numpy','gymnasium']},x64=bool(jax.config.jax_enable_x64),precision=str(jax.config.jax_default_matmul_precision)))
 snap=pickle.loads(snapshot.read_bytes())
 env=ogbench.make_env_and_datasets(c['task'],env_only=True);u=env.unwrapped
 assert u._teleport_info is None and int(u._reward_task_id)==4
 geom=[mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(u.model.ngeom)]
 np.random.seed(c['reset_seed']);random.seed(c['reset_seed']);env.reset(seed=c['reset_seed'])
 init_steps=counts['physical_control_steps'];assert init_steps<=100
 refs={}
 for x in c['references']:
  refs[(x['mode'],x['candidate'],x['repeat'])]=dict(np.load(x['path']))
 follow=[refs[('first_action','original_actor',rep)]['policy_keys'] for rep in range(4)]
 baseline=refs[('short_policy','rebrac',0)]
 for rep in range(4):
  assert np.array_equal(follow[rep],refs[('first_action','rebrac',rep)]['policy_keys'])
  for n in ['observations','qpos','qvel','actions']:assert np.array_equal(baseline[n],refs[('short_policy','rebrac',rep)][n])
 stage='restore_validation';status();ob=restore(snap);original=refs[('first_action','original_actor',0)];maxerr=dict(observations=0.,qpos=0.,qvel=0.,actions=0.)
 assert np.array_equal(ob,original['observations'][0])
 for h in range(20):
  a=action('actor',ob,follow[0][h]);ob,r,te,tr,info=step(a,'validation');e=compare(original,h,ob,a,r,te,tr)
  maxerr={k:max(maxerr[k],e[k]) for k in maxerr}
 save(R/'receipts/restore_admission.json',dict(status='PASS',steps=20,max_errors=maxerr,initialization_physical_steps=init_steps,**counts))
 verify_assets();stage='branches';status()
 for length in c['new_lengths']:
  for rep in range(4):
   ob=restore(snap);assert np.array_equal(ob,original['observations'][0]);seq=[physical_record(ob)];aa=[];rr=[];ss=[];tt=[];trr=[];pe=dict(observations=0.,qpos=0.,qvel=0.,actions=0.)
   for h in range(200):
    a=action('rebrac' if h<length else 'actor',ob,follow[rep][h])
    ob,r,te,tr,info=step(a,'branch');seq.append(physical_record(ob));aa.append(a);rr.append(r);ss.append(float(info.get('success',0)));tt.append(te);trr.append(tr)
    if h<length:
     e=compare(baseline,h,ob,a,r,te,tr);pe={k:max(pe[k],e[k]) for k in pe}
    if te or tr:break
   arr={n:np.asarray([s[n] for s in seq]) for n in seq[0]};arr.update(actions=np.asarray(aa),rewards=np.asarray(rr),success=np.asarray(ss),terminated=np.asarray(tt),truncated=np.asarray(trr),policy_keys=follow[rep][:len(aa)],controller=np.asarray([0 if h<length else 1 for h in range(len(aa))],np.int8))
   assert all(np.isfinite(v).all() for v in arr.values())
   dest=R/'branches'/f'L{length:03d}_r{rep}.npz';assert not dest.exists();np.savez_compressed(dest,**arr)
   results.append(dict(length=length,repeat=rep,steps=len(aa),path=str(dest),sha256=digest(dest),prefix_max_errors=pe,source_key_sha256=hashlib.sha256(follow[rep].tobytes()).hexdigest()))
   save(R/'reports/new_branches.json',results);status()
  print('COMPLETED_LENGTH',length,counts,flush=True)
 assert len(results)==16 and counts['branch_external_steps']<=3200
 verify_assets()
 for x in inputs:assert digest(Path(x['path']))==x['sha256'],x['path']
 stage='COMPLETE';status()
 save(R/'receipts/completed.json',dict(status='PASS',branches=len(results),old_assets_unchanged=True,elapsed_seconds=time.time()-t0,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,initialization_reset_control_steps=init_steps,**counts))
except Exception as e:
 save(R/'receipts/failure.json',dict(error=repr(e),traceback=traceback.format_exc(),stage=stage,branches=len(results),**counts));raise
finally:
 if env is not None:env.close()
