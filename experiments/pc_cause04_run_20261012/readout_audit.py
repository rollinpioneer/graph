"""Complete readout receipts and per-window chain audit without new GPU trajectories."""
from common import *
import time
begin=time.time();es=json.loads((R/'results/episode_index.json').read_text());cache={};rows=[]
for e in es:
 if not e['no_progress_ends']:continue
 if e['trajectory'] not in cache:cache[e['trajectory']]=np.load(e['trajectory'])
 z=cache[e['trajectory']];i=e['slot']
 for end in e['no_progress_ends']:
  c=chain(z,i,end-20,end);before=chain(z,i,max(0,end-40),end-20) if end>20 else None;after=chain(z,i,end,min(e['steps'],end+20)) if end<e['steps'] else None
  rows.append({'episode_id':e['episode_id'],'window_start':end-20,'online_detection':end,'eef_max_displacement_m':c['eef_max_displacement_m'],'eef_path_m':c['eef_path_m'],
  'cube_max_displacement_m':c['cube_max_displacement_m'],'max_mean_progress_m':c['max_mean_goal_improvement_m'],'per_object_progress_m':json.dumps(c['per_object_goal_improvement_m']),
  'action_abs_mean':json.dumps(c['action_abs_mean']),'action_clip_fraction':c['action_clip_fraction'],'q_max_change_rad':c.get('q_max_change_rad'),
  'target_transform_max_error_m':c.get('target_transform_max_error_m'),'pre_context_steps':min(20,end-20),'post_context_steps':min(20,e['steps']-end),
  'pre_context_eef_path_m':before['eef_path_m'] if before else None,'post_context_eef_path_m':after['eef_path_m'] if after else None,
  'contact_counterpart':'unknown','normality':'unestablished','unit':'overlapping window, not independent sample'})
table('P2_all_window_chains.csv',rows)
gpu=json.loads((R/'runs/contact_gpu_retry/data/receipt.json').read_text());cpu=json.loads((R/'runs/contact_cpu_tensor/data/receipt.json').read_text())
cpu_samples=json.loads((R/'runs/contact_cpu_tensor/data/samples.json').read_text());pairs=[c for s in cpu_samples for c in s['pairs'][0]]
known=bool(pairs) and all({c['body0'],c['body1']}=={0,1} and c['env0']==c['env1']==0 for c in pairs)
positions=bool(pairs) and all(abs(c['normal'][2]-1)<1e-6 for c in pairs)
measure={'utc':utc(),'G_C':'FAILED for original GPU task pipeline','original_gpu_positive_net_force_N':gpu['positive_max_cube_net_force_N'],
'original_gpu_negative_net_force_N':gpu['negative_max_cube_net_force_N'],'original_positive_contact_records':gpu['positive_contact_records'],
'original_gpu_api_error':'GymGetEnvRigidContacts/Count cannot be used with GPU pipeline after simulation starts',
'cpu_tensor_fixture':{'contact_records':len(pairs),'known_body_pair_valid':known,'negative_records':cpu['negative_contact_records'],'position_normal_fixture_check':positions,
'body_map':{'0':'fixed table','1':'cube'},'full_task_comparability_passed':False},
'fields':{'both_identities':'absent from original valid readout; present in CPU fixture','position':'CPU body-local contact points only verified on table fixture',
'normal':'CPU fixture z normal, no dynamic propagation validation','force':'net tensor valid on fixture; pair lambda units not calibrated as force',
'timestamps':'get_sim_time per dt 0.01667s, two internal substeps','coverage':'last internal substep; transient events may be lost',
'dropouts':'original pair API has 100% positive dropout/unavailability; CPU fixture 12/12 positive, negative0/12','ordering':'same dt events cannot be ordered'},
'repair_cycles_used':2,'engineering_start_utc':json.loads((R/'receipts/freeze_measurement.json').read_text())['utc'],
'engineering_stop_utc':json.loads((R/'runs/contact_cpu_tensor/guard_result.json').read_text())['end_utc'],
'decision':'stop contact supplements and all contact-dependent mechanism experiments; do not treat geometry as contact',
'positive_fixture_scope':'table/cube isolated static reaction, no dynamic robot-object chain or full policy comparability'}
assert known and positions and not gpu['original_backend_gate']
dump('G_C_receipt.json',measure)
jobs=[];peak=0;observed_gpus=set()
for p in sorted((R/'runs').glob('*/guard_result.json')):
 d=json.loads(p.read_text());d['run']=p.parent.name;jobs.append(d)
 for line in (p.parent/'gpu_memory.jsonl').read_text().splitlines():
  q=json.loads(line);peak=max(peak,q['aggregate_task_memory_mib']);observed_gpus.update(q['active_gpu_uuids'])
total=sum(j['charged_gpu_seconds'] for j in jobs);wall=sum(j['wall_seconds'] for j in jobs)
resource={'utc':utc(),'jobs':jobs,'charged_gpu_seconds':total,'charged_gpu_hours':total/3600,'total_child_wall_seconds':wall,
'charge_with_2s_per_job_exit_margin_gpu_hours':(total+2*len(jobs))/3600,
'eight_gpu_all_wall_plus_margin_upper_gpu_hours':8*(wall+2*len(jobs))/3600,
'upper_bound_reason':'at most8 installed GPUs; all child wall even before CUDA plus2s/job; protects brief contexts not sampled',
'aggregate_task_peak_sampled_mib':peak,'memory_safety_stop_mib':11264,'memory_hard_cap_mib':12288,'sampling_s':.5,
'memory_precision':'sampled peak, no claim of continuous exact maximum; short CPU-tensor job CUDA residency not observed',
'observed_gpu_uuids':sorted(observed_gpus),'hard_gpu_hours_cap':2,'new_full_discovery_episodes':0,'new_validation_episodes':0,
'failures':[j['run'] for j in jobs if j['exit_code']!=0],'budget_exhausted':False,'phase2_reason':'scientific gates, not resource exhaustion'}
assert resource['eight_gpu_all_wall_plus_margin_upper_gpu_hours']<2 and peak<12288 and all(j['stop_reason'] is None for j in jobs)
dump('resource_summary.json',resource)
dump('all_windows_audit_receipt.json',{'utc':utc(),'overlapping_windows':len(rows),'episode_units':sum(bool(e['no_progress_ends']) for e in es),'cpu_wall_seconds':time.time()-begin})
print('AUDIT_COMPLETE',len(rows),'windows, GPUh upper',resource['eight_gpu_all_wall_plus_margin_upper_gpu_hours'],flush=True)
