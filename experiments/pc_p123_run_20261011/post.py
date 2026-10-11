import pathlib,json,numpy as np,csv,hashlib,datetime
from prepare import events
R=pathlib.Path(__file__).resolve().parent
def dump(n,x):(R/'results'/n).write_text(json.dumps(x,indent=2))
def analysis(name):
 F=R/'runs'/name/'data';z=np.load(F/'trajectory.npz');ph=np.load(F/'physical.npz');meta=json.loads((F/'contact_source.json').read_text());cubeids=[b['index_env'] for b in meta['body_map'] if b['actor']>=3]
 rows=[json.loads(l) for l in (F/'endpoints.jsonl').read_text().splitlines()] if (F/'endpoints.jsonl').exists() else []
 for r in rows:
  i=r['slot'];t=r['step'];st=z['states'][:t+1,i];dd=z['distances'][:t+1,i];ee=z['eef'][:t+1,i]
  r['events']=events(st,dd,ee);r['input_clip_fraction']=float((np.abs(z['actions'][:t,i])>=1).mean())
  mask=ph['policy_step']<=t;force=ph['net_force'][mask,i][:,cubeids,:2]
  r['lateral_net_force_gt_0_1N_substep_samples_per_object']=(np.linalg.norm(force,axis=-1)>.1).sum(0).tolist()
  r['effective_physical_dt_s']=float(np.median(np.diff(ph['time'])[np.diff(ph['time'])>0]));r['contact_counterparts_known']=False
  r['contact_force_usable']=json.loads((F/'environment_config.json').read_text())['sim']['physx']['contact_collection']!=0
  if not r['contact_force_usable']:r['lateral_net_force_gt_0_1N_substep_samples_per_object']=None
 return rows
rows=[]
for name in ['P1_sparse8','P1_crowded8','validation_P3_8']:
 if (R/'runs'/name/'data/completed.json').exists():rows+=analysis(name)
dump('episode_metrics.json',rows)
if rows:
 flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
 keys=sorted(set().union(*(set(x) for x in flat)))
 with (R/'results/episode_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
P1=[r for r in rows if r['condition'] in ['6_sparse','6_crowded'] and r['step']==200 and r['group']<8]
if len(P1)==16:
 paired=[]
 for g in range(8):
  p={r['condition']:r for r in P1 if r['group']==g};a=p['6_sparse'];b=p['6_crowded'];paired.append({'group':g,'block':g%2,'sparse_fraction':a['fraction'],'crowded_fraction':b['fraction'],'sparse_minus_crowded':a['fraction']-b['fraction']})
 effects=np.array([r['sparse_minus_crowded'] for r in paired]);rng=np.random.RandomState(2611125);boot=effects[rng.randint(0,8,(10000,8))].mean(1)
 summary={'pairs':paired,'mean_effect':float(effects.mean()),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'block_mean_effect':{str(b):float(effects[[r['block']==b for r in paired]].mean()) for b in [0,1]},'conditions':{}}
 for c in ['6_sparse','6_crowded']:
  rr=[r for r in P1 if r['condition']==c];summary['conditions'][c]={'episodes':8,'mean_fraction':float(np.mean([r['fraction'] for r in rr])),'all_success':sum(r['success'] for r in rr),'mean_distance_m':float(np.mean([r['mean_distance'] for r in rr])),'falls':sum(r['events']['first_fall'] is not None for r in rr),'stalls':sum(r['events']['first_no_progress'] is not None for r in rr)}
 dump('P1_summary.json',summary);print('P1_SUMMARY',json.dumps(summary))
 # Residual decision behavior observations are proximity proxies, not selected-object outputs or causal errors.
 residual=[]
 for name in ['P1_sparse8','P1_crowded8']:
  F=R/'runs'/name/'data';z=np.load(F/'trajectory.npz')
  for i in range(8):
   ee=z['eef'][:,i];st=z['states'][:,i];dd=z['distances'][:,i];ds=np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1);near=ds.argmin(1);close=ds.min(1)<.05
   residual.append({'condition':name,'group':i,'failed_object_ids':np.flatnonzero(dd[-1]>=.04).tolist(),'nearest_cube_switches':int((np.diff(near)!=0).sum()),'steps_with_eef_center_distance_lt_0_05':int(close.sum()),'approach_count_per_identity':[(int(((near==j)&close).sum())) for j in range(6)],'caveat':'Proximity label only; no selected-object action output and no contact counterparts. DLP particles not identity matched.'})
 dump('P1B_behavior.json',{'entry_gate_failed_sparse_scenes_at_least_two':sum(bool(r['failed_object_ids']) for r in residual if r['condition']=='P1_sparse8')>=2,'observations':residual})
 # New scene visual evidence.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 for name in ['P1_sparse8','P1_crowded8']:
  F=R/'runs'/name/'data';ini=np.load(F/'initial.npz');z=np.load(F/'trajectory.npz');folder=R/'results/cases';folder.mkdir(exist_ok=True)
  for i in range(8):
   fig,axs=plt.subplots(2,3,figsize=(9,6))
   for view in range(2):
    imgs=[ini['initial_image'][i,view],ini['goal_image'][i,view],z['images'][-1,i,view]]
    for col,img in enumerate(imgs):
     if img.shape[0]==3:img=np.transpose(img,(1,2,0))
     axs[view,col].imshow(img);axs[view,col].axis('off');axs[view,col].set_title(['initial','goal','step200'][col])
   fig.suptitle(name+' group '+str(i));fig.tight_layout();fig.savefig(folder/(name+'_g%d.png'%i),dpi=100);plt.close(fig)
V=[r for r in rows if r['condition']=='6_crowded' and r['group']>=8 and r['step']==200]
if len(V)==8:
 result={'arms':{},'paired_effects':[],'estimand':'complete policy rule effect, all episodes included, not identical restored states'}
 for arm in ['B0','B1']:
  rr=[r for r in V if r['arm']==arm];result['arms'][arm]={'episodes':len(rr),'mean_fraction':float(np.mean([r['fraction'] for r in rr])),'all_success':sum(r['success'] for r in rr),'falls':sum(r['events']['first_fall'] is not None for r in rr),'stalls':sum(r['events']['first_no_progress'] is not None for r in rr),'mean_distance_m':float(np.mean([r['mean_distance'] for r in rr]))}
 for g in range(8,12):
  d={r['arm']:r for r in V if r['group']==g};result['paired_effects'].append({'group':g,'B1_minus_B0':d['B1']['fraction']-d['B0']['fraction'],'B0_fall_step':d['B0']['events']['first_fall'],'B1_fall_step':d['B1']['events']['first_fall']})
 effects=np.array([x['B1_minus_B0'] for x in result['paired_effects']]);rng=np.random.RandomState(2611126);boot=effects[rng.randint(0,4,(10000,4))].mean(1);result['mean_fraction_effect_B1_minus_B0']=float(effects.mean());result['paired_bootstrap95']=np.quantile(boot,[.025,.975]).tolist();result['block_mean_effect']={str(b):float(effects[[g%2==b for g in range(8,12)]].mean()) for b in [0,1]};dump('P3_validation_summary.json',result);print('P3_VALIDATION',json.dumps(result))
receipts=[json.loads(f.read_text()) for f in (R/'runs').glob('*/guard_result.json')];tele=[]
for f in (R/'runs').glob('*/gpu_memory.jsonl'):
 tele.extend(json.loads(l) for l in f.read_text().splitlines())
resource={'jobs':receipts,'conservative_gpu_seconds_including_2s_per_job_margin':sum(x['charged_gpu_seconds']+2 for x in receipts),'aggregate_memory_peak_mib':max([x.get('aggregate_task_memory_mib',0) for x in tele] or [0]),'accounting':'Each process full wall time charged per GPU with own CUDA/graphics context, setup included; parallel jobs summed; 2 s receipt margin each.'}
resource['gpu_hours']=resource['conservative_gpu_seconds_including_2s_per_job_margin']/3600
dump('resource_summary.json',resource)
print('RESOURCE',resource['gpu_hours'],resource['aggregate_memory_peak_mib'])
