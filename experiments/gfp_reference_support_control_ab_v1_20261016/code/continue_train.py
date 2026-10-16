import os,time,json,pickle,random,resource,traceback
import numpy as np
import jax,jax.numpy as jnp,flax
from common import ROOT,SOURCE,SUPPORT,FORMAL,make_agent,load_base,digest,sha,write
from reference import Updater,nll

def main():
 t0=time.time();protocol=json.loads((ROOT/'protocol.json').read_text())
 assert json.loads((ROOT/'receipts/admission_result.json').read_text())['passed']
 assert not (ROOT/'receipts/continuation_started.json').exists(),'No automatic retry'
 write(ROOT/'receipts/continuation_started.json',dict(pid=os.getpid(),time=t0,gpu=os.environ['CUDA_VISIBLE_DEVICES'],resume_updates=500,end_updates=50000))
 template,cfg,_=make_agent();agents={};initial={}
 for arm in ('A','B'):
  p=ROOT/'runs'/arm/'checkpoint_500.pkl';m=json.loads(p.with_suffix('.json').read_text());assert sha(p)==m['sha256']
  with open(p,'rb') as f:payload=pickle.load(f)
  a=flax.serialization.from_state_dict(template,payload['agent']);assert digest(a)==m['state_sha256']
  assert payload['updates']==500 and payload['reference_events']==5
  assert payload['calibration_sha256']==sha(ROOT/'receipts/calibration.json')
  assert int(a.network.step)==350501 and int(a.network.opt_state[0].count)==350500
  agents[arm]=a
  np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
  with open(ROOT/'runs'/arm/'checkpoint_0.pkl','rb') as f:initial[arm]=flax.serialization.from_state_dict(template,pickle.load(f)['agent'])
 write(ROOT/'receipts/continuation_restore.json',dict(passed=True,updates_resumed=500,agent_adam_reference_rng=True,source_unchanged=True))
 raw={p.stem:np.load(p,mmap_mode='r') for p in (SOURCE/'data').glob('*.npy')};data={k:jnp.asarray(v) for k,v in raw.items()}
 size=json.loads((SOURCE/'receipts/assets.json').read_text())['dataset_size'];valid=(np.arange(len(raw['next_observations']))<size)&(np.asarray(raw['masks'])>0)
 updater=Updater(template,raw['next_observations'],valid);cal=json.loads((ROOT/'receipts/calibration.json').read_text());updater.rho=cal['rho0'];updater.regional=np.asarray(cal['regional_start_outside'])
 events=json.loads((ROOT/'receipts/reference_events.json').read_text());assert len(events)==5
 probe=np.load(FORMAL/'analysis_p0p1/out/probe/states.npz');obs,z=jnp.asarray(probe['obs']),jnp.asarray(probe['z'])
 reg=np.isin(probe['cell_i'],[1,2])&np.isin(probe['cell_j'],[1,2])
 historical,_,_=load_base(3);oldactions=jnp.clip(historical.network.select('actor')(obs,z),-1,1)
 @jax.jit
 def assess(a):
  act=jnp.clip(a.network.select('actor')(obs,z),-1,1)
  q=a.network.select('critic')(obs,actions=act).mean(0);qt=a.critic_eval(obs,act)
  grad=jax.grad(lambda x:a.network.select('critic')(obs,actions=x).mean(0).sum())(oldactions)
  return act,q,qt,grad
 old=list(map(np.asarray,assess(historical)));histgap=float(old[1][reg].mean()-old[1].mean());histnorm=float(np.linalg.norm(old[3],axis=1)[reg].mean())
 write(ROOT/'receipts/anchors.json',dict(step=300000,historical_gap=histgap,historical_gradient_norm=histnorm,probe_sha256=sha(FORMAL/'analysis_p0p1/out/probe/states.npz'),historical_action_sha256=digest(oldactions)))
 query=np.load(SUPPORT/'arrays/queries.npz');noise=np.load(SUPPORT/'arrays/noise.npz');qo=query['obs'];qa=query['actions'];qr=query['region'];qs=query['split']
 models={1701:updater.mdn}
 with open(SUPPORT/'models/mdn_1702.pkl','rb') as f:models[1702]=jax.tree.map(jnp.asarray,pickle.load(f)['params'])
 reliability=json.loads((SUPPORT/'reports/reliability.json').read_text());cutoffs={x['seed']:np.asarray(x['cutoffs']) for x in reliability}
 def actions(a):
  from jax.flatten_util import ravel_pytree
  flat,_=ravel_pytree(a.reference);out=[]
  for key in ('z0','z1'):
   out.append(np.concatenate([np.asarray(updater.action(flat,jnp.asarray(qo[i:i+256]),jnp.asarray(noise[key][i:i+256]))) for i in range(0,len(qo),256)]))
  return np.stack(out)
 initialacts=actions(template);previous={k:initialacts for k in agents};rows={k:[] for k in agents};audits={k:[] for k in agents};info={}
 def scores(model,aa):
  return np.concatenate([np.asarray(nll(model,(qo[i:i+256]-updater.mean)/updater.std,jnp.asarray(aa[i:i+256]))) for i in range(0,len(qo),256)])
 actual_scores={seed:scores(model,qa) for seed,model in models.items()}
 def snapshot(arm,a,it):
  act,q,qt,g=map(np.asarray,assess(a));assert all(np.isfinite(x).all() for x in (act,q,qt,g))
  gn=np.linalg.norm(g,axis=1);cos=(g*old[3]).sum(1)/(gn*np.linalg.norm(old[3],axis=1)+1e-12)
  row=dict(updates=it,global_step=350000+it,V_online_start=float(q[reg].mean()),V_online_all=float(q.mean()),V_online_outside=float(q[~reg].mean()),V_target_start=float(qt[reg].mean()),V_target_outside=float(qt[~reg].mean()),historical_gap_increase=float(q[reg].mean()-q.mean()-histgap),historical_gradient_ratio=float(gn[reg].mean()/histnorm),dqda_norm_start=float(gn[reg].mean()),dqda_norm_outside=float(gn[~reg].mean()),dqda_cos_start=float(cos[reg].mean()),action_shift_start=float(np.linalg.norm(act-old[0],axis=1)[reg].mean()),seconds=time.time()-t0)
  row['joint_anomaly']=row['historical_gap_increase']>=15 and row['historical_gradient_ratio']>=2
  row['training_metrics']=info.get(arm,{})
  rows[arm].append(row);write(ROOT/f'runs/{arm}/probes.json',rows[arm])
  aa=actions(a);densities={seed:np.stack([scores(model,v) for v in aa]) for seed,model in models.items()}
  groups=[]
  for split in np.unique(qs):
   for region in (0,1):
    mask=(qs==split)&(qr==region)
    group=dict(split=str(split),region='start' if region else 'outside',states=int(mask.sum()),adjacent_checkpoint_rms=float(np.sqrt(np.mean((aa[:,mask]-previous[arm][:,mask])**2))),net_initial_rms=float(np.sqrt(np.mean((aa[:,mask]-initialacts[:,mask])**2))))
    for seed,dd in densities.items():
     group[str(seed)]=dict(nll_mean=float(dd[:,mask].mean()),nll_quantiles=np.quantile(dd[:,mask],[.05,.5,.95]).tolist(),tail_fraction=float((dd[:,mask]>cutoffs[seed][region]).mean()),actual_data_nll=float(actual_scores[seed][mask].mean()),actual_data_tail=float((actual_scores[seed][mask]>cutoffs[seed][region]).mean()))
    groups.append(group)
  audits[arm].append(dict(updates=it,groups=groups));previous[arm]=aa
  write(ROOT/f'runs/{arm}/support_audit.json',audits[arm])
 def save(arm,a,it):
  p=dict(agent=flax.serialization.to_state_dict(a),updates=it,seed=3,arm=arm,numpy_rng=np.random.get_state(),python_rng=random.getstate(),reference_events=it//100,reference_rng_seed=6824100,source_sha256=protocol['source_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),calibration_sha256=sha(ROOT/'receipts/calibration.json'))
  path=ROOT/f'runs/{arm}/checkpoint_{it}.pkl'
  assert not path.exists()
  with open(str(path)+'.tmp','wb') as f:pickle.dump(p,f)
  os.replace(str(path)+'.tmp',path)
  with open(path,'rb') as f:back=pickle.load(f)
  restored=flax.serialization.from_state_dict(a,back['agent']);assert digest(a)==digest(restored)
  assert int(a.network.step)==350001+it and int(a.network.opt_state[0].count)==350000+it
  assert back['python_rng']==random.getstate() and pickle.dumps(back['numpy_rng'])==pickle.dumps(np.random.get_state())
  write(path.with_suffix('.json'),dict(updates=it,step=350000+it,sha256=sha(path),state_sha256=digest(a),agent_rng=np.asarray(a.rng).tolist(),reference_sha256=digest(a.reference),full_restore='PASS'))
  return restored
 for arm in agents:snapshot(arm,initial[arm],0)
 for it in range(504,50001,4):
  for arm,a in list(agents.items()):
   a,inf=a.multi_sample_and_update(4,data,size,256)
   assert bool(inf['fused_all_finite']) and all(np.isfinite(float(x)) for x in inf.values())
   agents[arm]=a;info[arm]={k:float(v) for k,v in inf.items()}
  assert np.array_equal(agents['A'].rng,agents['B'].rng)
  if it%100==0:
   agents,event=updater.paired(agents,it//100);events.append(event)
   write(ROOT/'receipts/reference_events.json',events)
   write(ROOT/'receipts/progress.json',dict(phase='training',updates_per_arm=it,updates_total=2*it,reference_events=len(events)*2,seconds=time.time()-t0,last_match=event['relative_diff_global_start_outside'],last_radius_fraction=event['rho_fraction']))
  if it%5000==0:
   for arm,a in list(agents.items()):snapshot(arm,a,it);agents[arm]=save(arm,a,it)
   print('CHECKPOINT',it,'seconds',time.time()-t0,flush=True)
 for item in json.loads((ROOT/'receipts/source_identity.json').read_text())['assets']:assert sha(item['path'])==item['sha256']
 write(ROOT/'receipts/training_complete.json',dict(updates_per_arm=50000,updates_total=100000,reference_events=1000,environment_steps=0,density_updates=0,source_hashes_unchanged=True,seconds=time.time()-t0,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'receipts/continuation_error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
