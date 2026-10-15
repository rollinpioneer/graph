"""Frozen task4 interventions. Every simulator call is counted; no optimizer updates."""
import os,sys,json,time,random,pickle,copy,hashlib,traceback,resource
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R.parent/'gfp_task4_baseline_pilot_v1_20261015';seed=int(sys.argv[1])
os.environ.update(COMPARATIVE_ROOT=str(P),COMPARATIVE_JOB=f'GFP_seed{seed}')
sys.path.insert(0,str(P/'workspaces/gfp'))
import numpy as np,jax,mujoco,ogbench
from agents import agents
from utils.flax_utils import restore_agent,TrainState
from training_wiring import restore_state,state_digest,digest

def forbidden(*a,**k):raise RuntimeError('OPTIMIZER_UPDATE_FORBIDDEN')
for cls in [agents['gfp'],agents['rebrac']]:
 for n in ['update','batch_update','multi_sample_and_update','target_update']:
  if hasattr(cls,n):setattr(cls,n,forbidden)
for n in ['apply_loss_fn','apply_gradients']:
 if hasattr(TrainState,n):setattr(TrainState,n,forbidden)
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def arrayhash(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def plain(x):return float(np.asarray(x))
states=[s for s in json.loads((R/'configs/states.json').read_text())['states'] if s['seed']==seed]
assert len(states)<=4 and states
assert (R/'receipts/protocol_frozen.json').exists()
with (R/'receipts'/f'started_s{seed}.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
t0=time.time();counts={'physical_environment_steps':0,'external_environment_steps':0,'optimizer_updates':0};results=[];stage='initialization';current=None
def progress():save(R/'receipts'/f'progress_s{seed}.json',dict(stage=stage,current=current,branches=len(results),elapsed=time.time()-t0,**counts))
def wait_gate(name):
 while not (R/'receipts'/name).exists():
  if list((R/'receipts').glob('failure_s*.json')):raise RuntimeError('OTHER_SEED_FAILED')
  progress();time.sleep(2)

samples=np.load(P/'arrays/train_samples.npz');models={};assets=[]
for host,name in [('gfp','GFP'),('rebrac','ReBRAC')]:
 rt=json.loads((P/'runs'/f'{name}_seed{seed}'/'runtime.json').read_text());cfg=dict(rt['cfg']['agent'] if rt.get('adopted') else rt['agent_config'],log_metrics=True)
 a=agents[host].create(seed,samples['observations'][:1],samples['actions'][:1],cfg);directory=Path(rt['output_dir'])
 a=restore_agent(a,str(directory),1000000) if rt.get('adopted') else restore_state(a,directory,1000000)
 assert int(a.network.step)==1000001
 models[host]=a;cp=directory/'params_1000000.pkl'
 assets.append(dict(host=host,checkpoint=str(cp),checkpoint_sha256=digest(cp),state_sha256=state_digest(a),step=int(a.network.step),config=cfg))
save(R/'receipts'/f'assets_s{seed}.json',assets)
def verify_assets():
 for x in assets:
  assert digest(Path(x['checkpoint']))==x['checkpoint_sha256'];assert state_digest(models[x['host']])==x['state_sha256'];assert int(models[x['host']].network.step)==x['step']
@jax.jit
def scores(a,ob,act):return a.network.select('critic')(ob,act),a.network.select('target_critic')(ob,act)
def action(host,ob,key):
 a=models['rebrac' if host=='rebrac' else 'gfp']
 out=a.sample_flow_actions(ob,seed=key,temperature=0) if host=='vabc' else a.sample_actions(ob,seed=key,temperature=0)
 out=np.clip(np.asarray(out),-1,1);assert np.isfinite(out).all();return out
def keys(base,n=2000):
 k=jax.random.PRNGKey(base);out=[]
 for _ in range(n):k,sub=jax.random.split(k);out.append(np.asarray(sub))
 return out

env=ogbench.make_env_and_datasets('humanoidmaze-medium-navigate-singletask-task4-v0',env_only=True);u=env.unwrapped
assert u._teleport_info is None and int(u._reward_task_id)==4
physical=u._step_mujoco_simulation
def counted_sim(ctrl,n_frames):
 if counts['physical_environment_steps']>=45000:raise RuntimeError('SEED_PHYSICAL_STEP_CAP')
 physical(ctrl,n_frames);counts['physical_environment_steps']+=1
u._step_mujoco_simulation=counted_sim
def step(a):
 counts['external_environment_steps']+=1
 return env.step(a)
def wrappers():
 out=[];e=env
 while True:
  out.append(e)
  if not hasattr(e,'env'):return out
  e=e.env
geom=[mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(u.model.ngeom)]
def capture(ob,k):
 integration=np.empty(mujoco.mj_stateSize(u.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(u.model,u.data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
 return dict(data=copy.copy(u.data),integration=integration,observation=np.asarray(ob).copy(),goal=u.cur_goal_xy.copy(),geom_pos=u.model.geom_pos.copy(),task_id=u.cur_task_id,task_info=copy.deepcopy(u.cur_task_info),env_rng=copy.deepcopy(u.np_random.bit_generator.state),action_space_rng=copy.deepcopy(env.action_space.np_random.bit_generator.state),numpy_rng=np.random.get_state(),python_rng=random.getstate(),policy_key=np.asarray(k),wrappers=[{n:copy.deepcopy(vars(e)[n]) for n in ['_elapsed_steps','_has_reset','_checked_reset','_checked_step'] if n in vars(e)} for e in wrappers()])
def restore(s):
 u.data=copy.copy(s['data']);u.cur_goal_xy=s['goal'].copy();u.model.geom_pos[:]=s['geom_pos'];u.cur_task_id=s['task_id'];u.cur_task_info=copy.deepcopy(s['task_info'])
 u.np_random.bit_generator.state=copy.deepcopy(s['env_rng']);env.action_space.np_random.bit_generator.state=copy.deepcopy(s['action_space_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng'])
 for e,vals in zip(wrappers(),s['wrappers']):
  for n,v in vals.items():setattr(e,n,copy.deepcopy(v))
 ob=u.get_ob();assert np.max(np.abs(ob-s['observation']))<=1e-8
 return ob
def physical_record(ob):
 contacts=sum(geom[int(c.geom1)].startswith('block_') or geom[int(c.geom2)].startswith('block_') for c in u.data.contact)
 return dict(observations=np.asarray(ob).copy(),qpos=u.data.qpos.copy(),qvel=u.data.qvel.copy(),wall_contacts=int(contacts),time=float(u.data.time))

def admit(c):
 global stage,current
 stage='restore_admission';current=c['id'];progress();ref=c['source'];p=Path(ref['trajectory']);assert digest(p)==ref['trajectory_sha256'];old=dict(np.load(p));t=c['step'];ks=keys(ref['policy_seed'])
 np.random.seed(ref['global_seed']);random.seed(ref['global_seed']);ob,_=env.reset(seed=ref['reset_seed']);assert arrayhash(ob)==ref['initial_observation_sha256']
 maxobs=0.;maxact=0.
 for i in range(t):
  a=action('actor',ob,ks[i]);err=float(np.max(np.abs(a-old['actions'][i])));maxact=max(maxact,err);assert err<=1e-7,('prefix_action',i,err)
  ob,r,te,tr,info=step(a);err=float(np.max(np.abs(ob-old['observations'][i+1])));maxobs=max(maxobs,err)
  assert err<=1e-8 and r==old['rewards'][i] and not te and not tr,('prefix_state',i,err)
 snap=capture(ob,ks[t]);path=R/'states'/f'{c["id"]}.pkl'
 with path.open('xb') as f:pickle.dump(snap,f,protocol=5)
 original=[]
 for h in range(100):
  a=action('actor',ob,ks[t+h]);maxact=max(maxact,float(np.max(np.abs(a-old['actions'][t+h]))));assert maxact<=1e-7
  ob,r,te,tr,info=step(a);maxobs=max(maxobs,float(np.max(np.abs(ob-old['observations'][t+h+1]))));assert maxobs<=1e-8 and r==old['rewards'][t+h] and not te and not tr
  original.append(dict(**physical_record(ob),action=a,reward=r,terminated=te,truncated=tr))
 with path.open('rb') as f:snap=pickle.load(f)
 ob=restore(snap);errors=dict(observation=0.,qpos=0.,qvel=0.,action=0.)
 for h in range(100):
  a=action('actor',ob,ks[t+h]);ob,r,te,tr,info=step(a);v=original[h]
  for n,x in [('observation',ob),('qpos',u.data.qpos),('qvel',u.data.qvel),('action',a)]:errors[n]=max(errors[n],float(np.max(np.abs(x-v['observations' if n=='observation' else n]))))
  assert all(v<=(1e-7 if n=='action' else 1e-8) for n,v in errors.items()) and (r,te,tr)==(v['reward'],v['terminated'],v['truncated']),('RESTORE_MISMATCH',errors)
 row=dict(state=c['id'],status='PASS',checkpoint=assets[0]['checkpoint_sha256'],snapshot_sha256=digest(path),prefix_max_observation_error=maxobs,prefix_max_action_error=maxact,restore_max_errors=errors,observations_compared=200+t,complete_mjdata_pickle=True,**counts)
 save(R/'receipts'/f'admission_{c["id"]}.json',row);print('ADMISSION',c['id'],errors,flush=True)
 return row

def summarize(c,label,rep,mode,arr,q,qt,steps_before):
 o=arr['observations'];stable=(o[:,23]>=1.2)&(o[:,38]>=.7);cc=((o[:,:2]+6)/4).astype(int)[:,::-1];out=~np.all(cc==[6,5],axis=1);east=(o[:,0]>=18);n=len(arr['actions'])
 return dict(state=c['id'],seed=seed,candidate=label,repeat=rep,mode=mode,steps=n,exit_initial_cell=bool(out.any()),east_exit=bool((east&out).any()),upright_east_exit=bool((east&out&stable).any()),first_exit=int(np.flatnonzero(out)[0]) if out.any() else None,dx={str(h):float(o[min(h,n),0]-o[0,0]) for h in [20,50,100,200]},stable_fraction=float(stable[1:].mean()),stable_first50=float(stable[1:51].mean()),east_progress50=bool(o[min(50,n),0]-o[0,0]>=.5 and stable[1:51].mean()>=.5),minimum_height=float(o[:,23].min()),wall_contact_fraction=float((arr['wall_contacts'][1:]>0).mean()),west_wall_near_fraction=float((o[1:,0]<=14.75).mean()),success=float(arr['success'].max()),return_=float(arr['rewards'].sum()),online_q=np.asarray(q).tolist(),target_q=np.asarray(qt).tolist(),physical_steps=counts['physical_environment_steps']-steps_before)

def branches(c):
 global stage,current
 stage='intervention';current=c['id'];progress();idx=c['index'];ref=c['source'];t=c['step'];old=dict(np.load(ref['trajectory']));ks=keys(ref['policy_seed']);path=R/'states'/f'{c["id"]}.pkl';snap=pickle.loads(path.read_bytes());ob=snap['observation']
 cand=[('original_actor',action('actor',ob,ks[t]))]
 assert np.max(np.abs(cand[0][1]-old['actions'][t]))<=1e-7
 candidate_keys={}
 for draw in range(4):
  k=jax.random.PRNGKey(2026101600+idx*100+draw);candidate_keys[str(draw)]=np.asarray(k).tolist()
  cand.extend([(f'actor_{draw}',action('actor',ob,k)),(f'vabc_{draw}',action('vabc',ob,k))])
 cand.append(('rebrac',action('rebrac',ob,jax.random.PRNGKey(0))))
 values={label:tuple(np.asarray(x) for x in scores(models['gfp'],ob,a)) for label,a in cand}
 save(R/'reports'/f'candidates_{c["id"]}.json',dict(state=c['id'],keys=candidate_keys,candidates=[dict(label=label,action=a.tolist(),online_q=values[label][0].tolist(),target_q=values[label][1].tolist()) for label,a in cand]))
 for mode in ['first_action','short_policy']:
  for label,first in (cand if mode=='first_action' else [('vabc',None),('rebrac',None)]):
   for rep in range(4):
    follow=ks[t:t+200] if rep==0 else keys(2026101700+idx*10+rep,200)
    ob=restore(snap);seq=[physical_record(ob)];aa=[];rr=[];ss=[];teall=[];trall=[];before=counts['physical_environment_steps']
    initial=first if mode=='first_action' else action(label,ob,follow[0]);q,qt=values[label] if mode=='first_action' else scores(models['gfp'],ob,initial)
    for h in range(min(200,2000-t)):
     a=initial if h==0 else action('actor' if mode=='first_action' else label,ob,follow[h])
     ob,r,te,tr,info=step(a);seq.append(physical_record(ob));aa.append(a);rr.append(r);ss.append(float(info.get('success',0)));teall.append(te);trall.append(tr)
     if mode=='first_action' and label=='original_actor' and rep==0:
      assert np.max(np.abs(ob-old['observations'][t+h+1]))<=1e-8 and np.max(np.abs(a-old['actions'][t+h]))<=1e-7 and r==old['rewards'][t+h]
     if te or tr:break
    arr={n:np.asarray([v[n] for v in seq]) for n in seq[0]};arr.update(actions=np.asarray(aa),rewards=np.asarray(rr),success=np.asarray(ss),terminated=np.asarray(teall),truncated=np.asarray(trall),policy_keys=np.asarray(follow[:len(aa)]))
    assert all(np.isfinite(x).all() for x in arr.values())
    dest=R/'branches'/f'{c["id"]}_{mode}_{label}_r{rep}.npz';assert not dest.exists();np.savez_compressed(dest,**arr)
    row=summarize(c,label,rep,mode,arr,q,qt,before);row.update(path=str(dest),sha256=digest(dest));results.append(row)
    save(R/'reports'/f'branches_s{seed}.json',results);progress()
   print('BRANCH_GROUP',c['id'],mode,label,counts['physical_environment_steps'],flush=True)

try:
 save(R/'receipts'/f'device_s{seed}.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),devices=[str(d) for d in jax.devices()],jax=jax.__version__,mujoco=mujoco.__version__,x64=bool(jax.config.jax_enable_x64),precision=str(jax.config.jax_default_matmul_precision)))
 admit(states[0]);save(R/'receipts'/f'pilot_s{seed}.json',dict(status='PASS'));stage='waiting_pilot_gate';wait_gate('pilot_gate.json')
 for c in states[1:]:admit(c)
 verify_assets();save(R/'receipts'/f'admission_s{seed}.json',dict(status='PASS',states=len(states),**counts));stage='waiting_branch_gate';wait_gate('branch_gate.json')
 for c in states:branches(c)
 verify_assets();stage='COMPLETE';progress()
 save(R/'receipts'/f'completed_s{seed}.json',dict(status='PASS',states=len(states),branches=len(results),assets_unchanged=True,elapsed_seconds=time.time()-t0,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**counts))
except Exception as e:
 save(R/'receipts'/f'failure_s{seed}.json',dict(error=repr(e),traceback=traceback.format_exc(),stage=stage,current=current,branches=len(results),**counts));raise
finally:env.close()
