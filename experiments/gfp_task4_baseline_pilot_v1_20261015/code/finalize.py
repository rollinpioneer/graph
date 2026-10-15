"""Read-only final accounting; no model calls, optimizer calls, or environment steps."""
from pathlib import Path
import collections, csv, hashlib, json, re, time
import numpy as np

R = Path(__file__).resolve().parents[1]
Z = Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015')
E = Path('/home/xushijie3/work/grebrac_l1_v1_20261015/runs/formal')
def read(p): return json.loads(Path(p).read_text())
def save(name, x): (R/'reports'/name).write_text(json.dumps(x, indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

start=time.time()
matched=read(R/'reports/matched_results.json')
assert matched['complete'] and matched['initial_hashes_matched']==100
finished=read(R/'receipts/evaluation_finished.json')
assert len(finished['completed'])==4 and not finished['failed']
native=[]; costs=[]; checkpoints=[]; health=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  job=f'{alg}_seed{seed}'; d=R/'runs'/job; rt=read(d/'runtime.json'); out=Path(rt['output_dir'])
  erows=list(csv.DictReader((out/'eval.csv').open()))
  srows=list(csv.DictReader((out/'success.csv').open()))
  assert len(erows)==len(srows)
  for row,srow in zip(erows,srows):
   cp=int(float(row['step']))
   x=dict(algorithm=alg,seed=seed,checkpoint=cp,success_percent=float(srow['success_Actor']),return_=float(row['Actor/episode.return']))
   if 'success_Flow' in srow: x['vabc_success_percent']=float(srow['success_Flow'])
   native.append(x)
  train=list(csv.DictReader((out/'train.csv').open())); vals={}
  for row in train:
   for k,v in row.items():
    if v is None or not v.strip(): continue
    try: val=float(v)
    except ValueError: continue
    assert np.isfinite(val),(job,k,v)
    vals.setdefault(k,[]).append(val)
  assert int(vals['step'][-1])==1000000
  health.append(dict(job=job,logged_rows=len(train),logged_values_finite=True,last_update=1000000,
   scope='Logged values only; not proof every intermediate value was finite',
   metrics={k:dict(min=min(v),max=max(v),last=v[-1]) for k,v in vals.items() if any(t in k for t in ['loss','guidance','q_mean','q_data','q_ref','grad/'])}))
  if alg=='GFP':
   ex=read(d/'exit.json');assert ex['exit_code']==0
   c=dict(job=job,adopted=False,wall_seconds=ex['wall_seconds'],cpu_seconds=ex['cpu_seconds'],peak_rss_kib=ex['max_rss_kib'])
   rows=[json.loads(x) for x in (out/'checkpoint_index.jsonl').read_text().splitlines()]
   peaks=[]
   for line in (d/'resources.jsonl').read_text().splitlines():
    rec=json.loads(line)
    for proc in rec.get('own_gpu_processes','').splitlines():
     fields=proc.split(',')
     if len(fields)==2 and fields[0].strip()==str(rt['pid']): peaks.append(float(fields[1]))
   c['sampled_main_gpu_peak_mib']=max(peaks) if peaks else None
   c['gpu_peak_scope']='Main training PID only, excluding native evaluator subprocesses and shared users'
   for row in rows:
    cp=row['epoch'];p=out/f'params_{cp}.pkl';assert row['completed_updates']==cp and row['state_step']==cp+1
    assert sha(p)==row['checkpoint_sha256']
    checkpoints.append(dict(job=job,step=cp,path=str(p),sha256=row['checkpoint_sha256'],state_sha256=row['state_sha256'],config_sha256=row['config_sha256'],verified_now=True))
  else:
   old=E/f'R_task4_s{seed}';assert read(old/'exit.json')['exit']==0
   txt=(old/'time_v.txt').read_text()
   def val(key): return float(re.search(re.escape(key)+r':\s*([\d.]+)',txt).group(1))
   wall=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d:.]+)',txt).group(1)
   seconds=0
   for part in wall.split(':'): seconds=seconds*60+float(part)
   c=dict(job=job,adopted=True,wall_seconds=seconds,cpu_seconds=val('User time (seconds)')+val('System time (seconds)'),peak_rss_kib=val('Maximum resident set size (kbytes)'),sampled_main_gpu_peak_mib=None,gpu_peak_scope='Historical per-process GPU telemetry unavailable')
   for row in read(d/'checkpoint_index.json'):
    assert sha(row['path'])==row['sha256'];checkpoints.append(dict(job=job,verified_now=True,**row))
  c['gpu_residence_hours_proxy']=c['wall_seconds']/3600
  ev=read(d/'matched_exit.json');assert ev['exit_code']==0
  c['matched_wall_seconds']=ev['wall_seconds'];c['matched_gpu']=ev['gpu'];costs.append(c)
save('native_results.json',native);save('health_summary.json',health);save('checkpoint_manifest.json',checkpoints)
z2=read(R/'receipts/z2_completed.json');probe=read(R/'receipts/probe_result.json')
assert z2['status']=='PASS' and z2['optimizer_updates']==z2['environment_steps']==0
cost=dict(jobs=costs,new_training_gpu_residence_hours=sum(c['wall_seconds'] for c in costs if not c['adopted'])/3600,
 adopted_training_gpu_residence_hours=sum(c['wall_seconds'] for c in costs if c['adopted'])/3600,
 matched_gpu_residence_hours=sum(c['matched_wall_seconds'] for c in costs)/3600,
 z2_gpu_reserved_pipeline_seconds=z2['wall_seconds'],z2_receipt=probe,
 scope='One GPU per job. Residence/wall allocation proxy, includes native evaluation and shared-device waiting; not exclusive GPU compute. CPU-only final analyses separate. Historical ReBRAC training charged as reused cost, not newly executed.')
launch=read(R/'receipts/launch.json');cost['new_training_start_to_all_matched_wall_seconds']=finished['time']-min(x['start_unix'] for x in launch)
save('cost_summary.json',cost)

effects=[]
for task,root in [('task1',Z),('task4',R)]:
 for x in read(root/'reports/controlled_effects.json'):
  effects.append(dict(task=task,seed=x['seed'],checkpoint=x['checkpoint'],split=x['split'],
   near_neutral=x['near_neutral'],far_neutral=x['far_neutral'],far_minus_near_ci95=x['far_minus_near_ci95'],
   adjusted_neutral_bfs_slope=x['adjusted_neutral_per_bfs_cell']['coef'][0],adjusted_neutral_bfs_ci95=x['adjusted_neutral_per_bfs_cell']['ci95'][0],
   near_proxy_auc=x['proxy']['near']['rank_auc'],far_proxy_auc=x['proxy']['far']['rank_auc'],far_proxy_gap_ci95=x['proxy']['far']['gap_ci95']))
save('task1_task4_comparison.json',effects)
sp=read(R/'reports/evaluation_spatial.json');sg=read(R/'reports/spatial_guidance.json');cover=read(R/'reports/data_coverage.json')['train'];overlap=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  for cp in [800000,900000,1000000]:
   rows=[x for x in sp if x['algorithm']==alg and x['seed']==seed and x['checkpoint']==cp];fails=[x for x in rows if not x['success']]
   term=collections.Counter(x['terminal_cell'] for x in fails)
   for cell,n in term.most_common():
    g=next((v for v in sg if v['seed']==seed and v['checkpoint']==cp and v['cell']==cell),None) if alg=='GFP' else None
    overlap.append(dict(algorithm=alg,seed=seed,checkpoint=cp,cell=cell,failed_terminal_episodes=n,total_failed_episodes=len(fails),
     episodes_visited=sum(cell in x['occupancy'] for x in rows),mean_episode_occupancy_fraction=float(np.mean([x['occupancy'].get(cell,0)/sum(x['occupancy'].values()) for x in rows])),
     data_transitions=cover['cell_transitions'].get(cell,0),data_unique_trajectories=cover['cell_unique_trajectories'].get(cell,0),
     data_right_angle_turns=cover['right_angle_turns'].get(cell,0),frozen_data_guidance=g))
save('failure_guidance_coverage_overlap.json',overlap)
escape=[]
for alg in ['GFP','ReBRAC']:
 for seed in [0,1]:
  for cp in [800000,900000,1000000]:
   v=read(R/'runs'/f'{alg}_seed{seed}'/'matched'/f'{cp}.json');details=[]
   for ep in v['episodes']:
    with np.load(ep['trajectory']) as f: obs=f['observations']
    cells=((obs[:,:2]+6)/4).astype(int)[:,::-1];initial=cells[0];outside=np.any(cells!=initial,axis=1)
    details.append(dict(episode=ep['episode'],success=ep['success'],initial_cell=initial.tolist(),ever_left_initial_cell=bool(outside.any()),
     first_exit_index=int(np.flatnonzero(outside)[0]) if outside.any() else None,initial_cell_fraction=float((~outside).mean())))
   escape.append(dict(algorithm=alg,seed=seed,checkpoint=cp,never_left_initial_cell=sum(not x['ever_left_initial_cell'] for x in details),
    mean_episode_initial_cell_fraction=float(np.mean([x['initial_cell_fraction'] for x in details])),episodes=details))
save('initial_cell_escape.json',escape)
save('finalization_receipt.json',dict(status='PASS',time=time.time(),cpu_analysis_wall_seconds=time.time()-start,optimizer_updates=0,environment_steps=0,
 checkpoint_files_verified=len(checkpoints),matched_episode_records=1200,initial_hashes_matched=100,
 decision='DIAGNOSE; NO-GO for guidance-temperature/module training on present evidence'))
print(json.dumps(dict(cost={k:v for k,v in cost.items() if k not in ['jobs','z2_receipt','scope']},
 escape=[{k:v for k,v in x.items() if k!='episodes'} for x in escape if x['checkpoint']==1000000],checkpoints=len(checkpoints))))
