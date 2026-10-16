import os,sys,time,json,pickle,random,resource,traceback
import numpy as np
import jax,jax.numpy as jnp,flax
from common import ROOT,SOURCE,SUPPORT,make_agent,digest,sha,write
from reference import Updater,project

def main():
 started=time.time();protocol=json.loads((ROOT/'protocol.json').read_text())
 assert protocol['training_authorized'] and protocol['parameter_updates_allowed_this_turn']==100000
 assert not (ROOT/'receipts/admission_started.json').exists(),'No automatic retry'
 write(ROOT/'receipts/admission_started.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],time=started,devices=list(map(str,jax.devices()))))
 identity=json.loads((ROOT/'receipts/source_identity.json').read_text())
 for item in identity['assets']:assert sha(item['path'])==item['sha256']
 ag,cfg,payload=make_agent();original=ag.replace(tau_pi=.00005)
 assert digest(ag.network)==digest(original.network) and digest(ag.reference)==digest(original.reference)
 np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
 initial_reference=digest(ag.reference)
 size=json.loads((SOURCE/'receipts/assets.json').read_text())['dataset_size']
 raw={p.stem:np.load(p,mmap_mode='r') for p in (SOURCE/'data').glob('*.npy')}
 data={k:jnp.asarray(v) for k,v in raw.items()}
 batch={k:v[:256] for k,v in data.items()};key=jax.random.PRNGKey(20261016)
 y0=original.td_parts(batch,key)[0];y1=ag.td_parts(batch,key)[0]
 assert np.array_equal(y0,y1)
 loss0,g0=jax.value_and_grad(lambda p:original.total_loss(batch,p,key)[0])(original.network.params)
 loss1,g1=jax.value_and_grad(lambda p:ag.total_loss(batch,p,key)[0])(ag.network.params)
 assert np.isclose(loss0,loss1,rtol=3e-5,atol=3e-5)
 diffs=[float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree.leaves(g0),jax.tree.leaves(g1))]
 assert all(np.allclose(a,b,rtol=3e-5,atol=3e-5) for a,b in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)))
 # Active set projection fixtures: none, one, two, correlated constraints.
 fixtures=[([1.,2.],[[-1.,0.],[0.,-1.]],[1.,2.]),([1.,-2.],[[1.,0.],[0.,1.]],[0.,-2.]),([1.,2.],[[1.,0.],[0.,1.]],[0.,0.]),([1.,2.],[[1.,0.],[1.,1.]],[ -.5,.5])]
 for d,g,expected in fixtures:
  out,_=project(jnp.array(d),jnp.array(g));assert np.allclose(out,expected,atol=1e-6)
 write(ROOT/'receipts/zero_update_admission.json',dict(passed=True,td_exact=True,loss=float(loss1),gradient_max_abs=max(diffs),projection_fixtures=4,source_restore=True,gfp_updates=0,environment_steps=0))
 del original,g0,g1
 valid=np.arange(len(raw['next_observations']))<size
 valid &= np.asarray(raw['masks'])>0
 updater=Updater(ag,raw['next_observations'],valid)
 calibration=updater.calibrate(ag);assert digest(ag.reference)==initial_reference
 agents={'A':ag,'B':ag};events=[];info={};runpaths={}
 for arm in agents:
  run=ROOT/'runs'/arm;run.mkdir(parents=True,exist_ok=False);runpaths[arm]=run
  write(run/'identity.json',dict(arm=arm,config=cfg,source_sha256=protocol['source_sha256'],initial_state_sha256=digest(ag),initial_reference_sha256=initial_reference,protocol_sha256=sha(ROOT/'protocol.json')))
 def save(it):
  for arm,a in list(agents.items()):
   p=dict(agent=flax.serialization.to_state_dict(a),updates=it,seed=3,arm=arm,numpy_rng=np.random.get_state(),python_rng=random.getstate(),source_sha256=protocol['source_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),reference_events=it//100,reference_rng_seed=6824100,calibration_sha256=sha(ROOT/'receipts/calibration.json'))
   path=runpaths[arm]/f'checkpoint_{it}.pkl'
   with open(str(path)+'.tmp','wb') as f:pickle.dump(p,f)
   os.replace(str(path)+'.tmp',path)
   with open(path,'rb') as f:back=pickle.load(f)
   restored=flax.serialization.from_state_dict(a,back['agent'])
   assert digest(restored)==digest(a)
   assert int(a.network.step)==350001+it and int(a.network.opt_state[0].count)==350000+it
   assert back['python_rng']==random.getstate() and pickle.dumps(back['numpy_rng'])==pickle.dumps(np.random.get_state())
   write(path.with_suffix('.json'),dict(updates=it,step=350000+it,sha256=sha(path),state_sha256=digest(a),agent_rng=np.asarray(a.rng).tolist(),reference_sha256=digest(a.reference),full_restore='PASS'))
   agents[arm]=restored
 save(0)
 for it in range(4,501,4):
  for arm,a in list(agents.items()):
   before_ref=digest(a.reference) if it in (4,96,100) else None
   a,inf=a.multi_sample_and_update(4,data,size,256)
   assert bool(inf['fused_all_finite']) and all(np.isfinite(float(x)) for x in inf.values())
   if before_ref is not None:assert digest(a.reference)==before_ref,'Hidden internal EMA'
   agents[arm]=a;info[arm]={k:float(v) for k,v in inf.items()}
  assert np.array_equal(agents['A'].rng,agents['B'].rng)
  if it==4:
   for arm,a in agents.items():
    assert all(digest(a.network.params[k])!=digest(ag.network.params[k]) for k in ('modules_actor','modules_vabc','modules_critic','modules_target_critic'))
    assert digest(a.reference)==initial_reference
   save(it)
   write(ROOT/'receipts/first_four.json',dict(passed=True,updates_per_arm=4,counted_in_budget=True,online_networks_changed=True,reference_unchanged=True,full_restore=True))
  if it%100==0:
   agents,event=updater.paired(agents,it//100);events.append(event)
   write(ROOT/'receipts/reference_events.json',events)
   print('EVENT',it,event['rho_fraction'],event['relative_diff_global_start_outside'],flush=True)
  if it%100==0 or it==4:
   write(ROOT/'receipts/progress.json',dict(phase='real_update_admission',updates_per_arm=it,updates_total=2*it,reference_events=len(events)*2,seconds=time.time()-started,training_metrics=info))
 save(500)
 passed=all(e['global_match'] for e in events) and sum(e['regional_match'] for e in events)>=4 and sum(e['continuous'] for e in events)>=4
 for item in identity['assets']:assert sha(item['path'])==item['sha256']
 result=dict(passed=bool(passed),status='ADMITTED' if passed else 'STOPPED_MATCHING_ADMISSION',updates_per_arm=500,updates_total=1000,reference_events_total=10,global_matches=sum(e['global_match'] for e in events),regional_matches=sum(e['regional_match'] for e in events),continuous_events=sum(e['continuous'] for e in events),environment_steps=0,density_updates=0,seconds=time.time()-started,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_hashes_unchanged=True,calibration=calibration)
 write(ROOT/'receipts/admission_result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'receipts/execution_error.json',dict(traceback=traceback.format_exc(),time=time.time()))
  raise
