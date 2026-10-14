"""Read-only final audit and aggregation; no Agent/optimizer imports or updates."""
from pathlib import Path
import json,csv,hashlib,statistics,math,time,shlex,datetime
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,data):
 (R/(name+'.json')).write_text(json.dumps(data,indent=2))
 if isinstance(data,list) and data:
  keys=sorted(set(k for x in data for k in x))
  with (R/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
jobs=read(ROOT/'configs/jobs.json');audit=read(ROOT/'receipts/monitor/audit_latest.json')
assert not audit['issues'] and not audit['own_gpu_processes']
assert all(x['terminal_work_receipts_verified'] for x in audit['controllers'].values())
points=[];index=[];losses=[];commands=[];costs=[];starts=[];ends=[];evalends=[];matched={};curve=[]
for j in jobs:
 name=j['job'];d=ROOT/'runs'/name;s=read(d/'status.json')
 assert s['status']=='COMPLETED' and s['updates']==1000000 and s['exit_code']==0
 for f in sorted((d/'output').rglob('params_*.identity.json')):
  x=read(f);step=x['completed_updates'];w=f.with_name(f.name.replace('.identity.json','.pkl'));rng=f.with_name(f.name.replace('.identity.json','.data_rng.pkl'))
  # Determine sidecar spelling from the saved-state implementation, not a guess.
  candidates=list(f.parent.glob(f'params_{step}*rng*'))
  actual=digest(w);assert actual==x['checkpoint_sha256'],str(w)
  assert x['state_step']==step+1
  if step>=50000:assert candidates,str(f)
  index.append(dict(job=name,step=step,path=str(w),bytes=w.stat().st_size,sha256=actual,state_sha256=x['state_sha256'],config_sha256=x['config_sha256'],rng_sidecars=[dict(path=str(z),sha256=digest(z)) for z in candidates]))
 for step in [800000,900000,1000000]:
  x=read(d/'matched_verified'/f'{step}.json');es=x['episodes'];assert len(es)==100 and x['status']=='PASS' and x['optimizer_updates']==0 and x['global_rng_control_verified']
  assert abs(statistics.mean(e['success'] for e in es)-x['success_rate'])<1e-12
  assert abs(statistics.mean(e['return_'] for e in es)-x['mean_return'])<1e-9
  for k,e in enumerate(es):
   assert e['episode']==k and all(e[t]==20261014+k for t in ['reset_seed','numpy_seed','python_seed']) and e['policy_seed']==420000+k
   assert e['terminated'] or e['truncated']
  ids=[z for z in index if z['job']==name and z['step']==step];assert len(ids)==1
  assert ids[0]['sha256']==x['checkpoint_sha256'] and ids[0]['state_sha256']==x['state_sha256']
  matched[name,step]=x
  for proto,comp,obj in [('matched_verified','Actor',x)]+[('native',f.stem.split('_')[-1],read(f)) for f in sorted((d/'native').glob(f'{step}_*.json'))]:
   ep=obj['episodes'];st=obj.get('stats',{})
   success=obj['success_rate'] if proto=='matched_verified' else st['success']
   ret=obj['mean_return'] if proto=='matched_verified' else st.get('acc_reward',st.get('episode.return'))
   points.append(dict(job=name,model=name[:2],seed=j['seed'],task=j['task'],step=step,protocol=proto,component=comp,success=success,return_mean=ret,raw_episode_count=len(ep),raw_success=statistics.mean(e['success'] for e in ep),raw_return=statistics.mean(e['return_'] for e in ep),official_episode_return=st.get('episode.return'),reported_raw_success_gap=success-statistics.mean(e['success'] for e in ep)))
 for f in sorted((d/'native').glob('*.json')):
  x=read(f);curve.append(dict(job=name,step=x['step'],component=x['component'],success=x['stats']['success'],return_mean=x['stats'].get('acc_reward',x['stats'].get('episode.return'))))
 for f in sorted((d/'output').rglob('train.csv')):
  rows=list(csv.DictReader(f.open()));
  for row in rows:
   assert all(math.isfinite(float(v)) for v in row.values() if v!='')
   losses.append(dict(job=name,phase_file=str(f.relative_to(ROOT)),**row))
 exits=[]
 for phase in ['T1','T2','T3']:
  if not (d/f'{phase}_exit.json').exists():continue
  x=read(d/f'{phase}_exit.json');assert x['exit_code']==0 and x['status']=='PASS';exits.append(x);starts.append(x['start_epoch']);ends.append(x['end_epoch'])
  c=read(d/f'{phase}_command.json');commands.append(dict(job=name,phase=phase,**c))
 e=read(d/'matched_verified_exit.json');assert e['exit_code']==0 and e['optimizer_updates']==0
 assert read(d/'matched_verified/completed.json')['status']=='PASS';evalends.append(e['end_epoch'])
 costs.append(dict(job=name,model=name[:2],seed=j['seed'],native_gpu_resident_hours=sum(x['gpu_resident_hours'] for x in exits),matched_gpu_reserved_hours=e['gpu_reserved_hours'],sampled_cpu_seconds=sum(x['resources']['cpu_seconds'] for x in exits),main_pid_gpu_peak_mib=max(x['resources']['gpu_memory_mib'] for x in exits),main_pid_rss_peak_gib=max(x['resources']['rss_kb'] for x in exits)/1024**2,matched_gpu=e['gpu']))
pairs=[]
for host in ['G','D']:
 for seed in [0,1]:
  for step in [800000,900000,1000000]:
   a=matched[f'{host}0_seed{seed}',step];b=matched[f'{host}1_seed{seed}',step]
   ah=[e['initial_observation_sha256'] for e in a['episodes']];bh=[e['initial_observation_sha256'] for e in b['episodes']];assert ah==bh
   pairs.append(dict(host=host,seed=seed,step=step,initial_conditions_equal=True,episodes=100,initial_hash_list_sha256=hashlib.sha256(json.dumps(ah).encode()).hexdigest()))
seed_rows=[];model_rows=[]
for proto in ['native','matched_verified']:
 for name in [j['job'] for j in jobs]:
  xs=[x for x in points if x['job']==name and x['protocol']==proto and x['component']=='Actor'];assert len(xs)==3
  seed_rows.append(dict(job=name,model=name[:2],seed=xs[0]['seed'],task=xs[0]['task'],protocol=proto,success_last3_mean=statistics.mean(x['success'] for x in xs),success_checkpoint_sample_sd=statistics.stdev(x['success'] for x in xs),return_last3_mean=statistics.mean(x['return_mean'] for x in xs),return_checkpoint_sample_sd=statistics.stdev(x['return_mean'] for x in xs)))
 for model in ['G0','G1','D0','D1']:
  xs=[x for x in seed_rows if x['model']==model and x['protocol']==proto];assert len(xs)==2
  model_rows.append(dict(model=model,task=xs[0]['task'],protocol=proto,success_mean=statistics.mean(x['success_last3_mean'] for x in xs),success_seed_sample_sd=statistics.stdev(x['success_last3_mean'] for x in xs),return_mean=statistics.mean(x['return_last3_mean'] for x in xs),return_seed_sample_sd=statistics.stdev(x['return_last3_mean'] for x in xs)))
old=read(ROOT/'receipts/old_assets_and_data_unchanged.json');print('OLD_SCHEMA',old['files'][0],flush=True)
old_checks=[]
for x in old['files']:
 p=Path(x['path']);h=digest(p);expected=x.get('sha256',x.get('current_sha256'));assert h==expected,str(p)
 old_checks.append(dict(path=str(p),sha256=h,unchanged=True))
save('checkpoint_index',index);save('final_checkpoint_results',points);save('final_seed_results',seed_rows);save('final_model_results',model_rows);save('final_costs',costs);save('native_learning_curves',curve);save('loss_learning_curves',losses);save('actual_commands',commands)
summary=dict(status='ALL_REGISTERED_JOBS_ENDED',epoch=time.time(),formal_updates=8000000,new_updates_this_analysis=0,native_jobs=8,native_exit_codes=[0]*8,primary_matched_checkpoints=24,primary_matched_episodes=2400,matched_optimizer_updates=0,paired_start_checks=pairs,checkpoint_files_rehashed=len(index),old_assets_rehashed=old_checks,issues=audit['issues'],own_gpu_processes=audit['own_gpu_processes'],native_gpu_resident_hours=sum(x['native_gpu_resident_hours'] for x in costs),matched_gpu_reserved_hours=sum(x['matched_gpu_reserved_hours'] for x in costs),successful_native_wall_hours=(max(ends)-min(starts))/3600,successful_native_and_primary_eval_wall_hours=(max(evalends)-min(starts))/3600,first_successful_start_epoch=min(starts),last_native_end_epoch=max(ends),last_primary_eval_end_epoch=max(evalends))
save('final_completion_audit',summary)
print('FINAL_MODELS',model_rows,flush=True);print('FINAL_COST', {k:v for k,v in summary.items() if k not in ['paired_start_checks','old_assets_rehashed']},flush=True)
