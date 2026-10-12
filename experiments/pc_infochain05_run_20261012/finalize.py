"""Final lightweight receipts, conservative resource ledger and explicit archive allowlist."""
from common import *
import shutil,collections,time
start=time.time();jobs=[];peak=0;calls=0;raw=[]
def jr(p):return json.loads(pathlib.Path(p).read_text())
dest=R/'receipts/jobs';dest.mkdir(parents=True,exist_ok=True)
for folder in sorted((R/'runs').iterdir()):
 gr=folder/'guard_result.json'
 if not gr.exists():raise RuntimeError('job still active: '+folder.name)
 q=jr(gr);logs=folder/'gpu_memory.jsonl'
 if logs.exists():
  for line in logs.read_text().splitlines():
   x=json.loads(line);peak=max(peak,x.get('aggregate_task_memory_mib',x.get('task_memory_mib',0)))
 place=dest/folder.name;place.mkdir(exist_ok=True)
 for name in ['launch.json','guard_result.json','monitor_takeover.json']:
  if (folder/name).exists():shutil.copy2(folder/name,place/name)
 for name in ['completed.json','pilot_receipt.json','renderer_selfcheck.json','config.json','noise_schedule.json']:
  if (folder/'data'/name).exists():shutil.copy2(folder/'data'/name,place/name)
 cp=folder/'data/completed.json';c=jr(cp) if cp.exists() else {}
 idx=folder/'data/sampling_index.json';samples=jr(idx) if idx.exists() else [];calls+=len(samples)
 for f in (folder/'data').glob('renderer_case_*.json'):shutil.copy2(f,place/f.name)
 if samples:
  slim=[{k:v for k,v in x.items() if k not in ['draw_types','actions']} for x in samples]
  (place/'sampling_receipts.json').write_text(json.dumps(slim,indent=2))
 state='executed' if q['exit_code']==0 else ('stopped' if q.get('stop_reason') else 'failed')
 valid=folder.name!='offline_resume'
 note='INVALID goal contrast: identical renderedG0G1G2 images;72 calls excluded;6C world recreation segfault' if not valid else ''
 if state!='executed':
  log=folder/'process.log'
  excerpt=log.read_text(errors='replace')[-2600:] if log.exists() else ''
  (place/'failure_excerpt.json').write_text(json.dumps({'log_path':str(log),'log_sha256':sha(log) if log.exists() else None,'last2600chars':excerpt,'scientifically_valid':valid,'note':note},indent=2))
 jobs.append({'job':folder.name,'state':state,'exit_code':q['exit_code'],'stop_reason':q.get('stop_reason'),'charged_gpu_seconds':q['charged_gpu_seconds'],'peak_task_mib':q['task_peak_mib'],'gpu':q['gpu'],'pid':q['pid'],'model_calls':len(samples),'role':c.get('role','offline_or_selfcheck'),'full_episodes':c.get('full_episodes',0),'partial_episodes':c.get('partial_episodes',0),'scientifically_valid':valid,'note':note})
 for f in folder.rglob('*'):
  if f.is_file() and f.suffix in ['.npz','.pkl','.log','.jsonl']:
   raw.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f),'storage':'server only'})
for f in (R/'logs').glob('*.log'):
 if f.name=='finalize.log':continue
 raw.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f),'storage':'server only'})
for name in ['P2_gripper_collision_geometry.json','P2_full_selected_traces.json','new_full_window_facts.json']:
 f=R/'results'/name
 if f.exists():raw.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f),'storage':'server only'})
dump('server_artifact_manifest.json',raw);dump('job_status.json',jobs)
total=sum(x['charged_gpu_seconds'] for x in jobs);assert total<7200,total;assert peak<12288,peak
ledger={'UTC':utc(),'charged_gpu_seconds':total,'charged_gpu_hours':total/3600,'aggregate_task_peak_mib':peak,'aggregate_task_peak_gib':peak/1024,'hard_gpu_seconds':7200,'hard_memory_mib':12288,'all_failures_and_contexts_included':True,'accounting':'per-job completewalltime x max(1,task-resident GPU contexts);continuous monitor takeover retains original worker and telemetry; conservative sharedGPUcontext charging','full_episodes':sum(x['full_episodes'] for x in jobs),'partial_episodes':sum(x['partial_episodes'] for x in jobs),'total_recorded_policy_calls':calls,'valid_goal_ABCD_calls':348,'invalid_identical_goal_calls':72,'training_steps':0,'GPU_job_states':dict(collections.Counter(x['state'] for x in jobs)),'CPU_analysis':'separate CPU receipts, including ownNPZanalysis restart; no GPU claims for CPUonly code'}
dump('resource_ledger.json',ledger);table('job_status.csv',[{k:v for k,v in x.items() if k!='note'} for x in jobs])
status=jr(R/'receipts/validation_execution_status.json');dump('condition_status.json',{'supplement':[x for x in jobs if x['role']=='supplement'],'validation':status,'unexecuted':'no training, no sparse/crowded matrix, no independent P3/contact-interface repair; conditionalstop outcomes preserved','source_card_sha256':sha(R/'REQUESTED_CARD.md'),'run_root':str(R),'branch':'codex/pc-infochain-05-20261012-card'})
# Explicit file categories; raw run trees, full logs and full tensor states stay outside Git.
sources=['common.py','prepare.py','runtime.py','probe.py','guard.py','rollout.py','geometry.py','roi.py','stationary.py','post.py','new_figures.py','takeover_guard.py','validation_launch.py','geometry_selfcheck.py','horizon_metrics.py','actual_readout.py','actual_readout_figures.py','renderer_repair.py','graphics_repair.py','finalize.py','REQUESTED_CARD.md','REPRODUCE.md','RESULTS.md','official_compatibility.patch','.gitignore']
excluded={'P2_gripper_collision_geometry.json','P2_full_selected_traces.json','new_full_window_facts.json'}
files=[R/x for x in sources if (R/x).exists()]
files += [f for f in (R/'results').iterdir() if f.is_file() and f.name not in excluded and f.suffix in ['.json','.csv']]
files += [f for f in (R/'figures').iterdir() if f.is_file() and f.suffix=='.png']
files += [f for f in (R/'receipts').rglob('*') if f.is_file() and f.suffix in ['.json','.py','.txt']]
files=sorted(set(files));assert all(f.stat().st_size<12*1024**2 for f in files)
rel=[str(f.relative_to(R)) for f in files];(R/'light_files.txt').write_text('\n'.join(rel)+'\n')
manifest=[{'relative_path':str(f.relative_to(R)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
(R/'light_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FINALIZED',json.dumps(ledger),'lightfiles',len(files),'lightbytes',sum(q['bytes'] for q in manifest),'wall',time.time()-start,flush=True)

