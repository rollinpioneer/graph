from core import *
import csv,shutil,platform
out=R/'results';status=json.loads((out/'pipeline_status.json').read_text())
assert status['status'] in ['COMPLETED_AT_S1_STOP','COMPLETED_CONFIRMATION']
rows=[];strata=[]
for stage in ['development','transfer','confirmation']:
 for p in sorted((R/f'evaluation/{stage}').glob('*/*.json')):
  if not p.stem.isdecimal():continue
  for x in json.loads(p.read_text()):
   rows.append({k:x.get(k) for k in ['scene','frame','configuration','top_identity','original_top_success','original_top_mu','selected_action','selected_top_mu','selected_top_collision','selected_top_empty','R','D']}|{'stage':stage})
   for st in ['high','low']:strata.append(dict(stage=stage,scene=x['scene'],frame=x['frame'],configuration=x['configuration'],stratum=st,**x[st]))
for name,items in [('paired_transitions.csv',rows),('score_strata_per_frame.csv',strata)]:
 with open(out/name,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
paired_scene=[]
for key in sorted(set((x['stage'],x['scene'],x['configuration']) for x in rows)):
 sub=[x for x in rows if (x['stage'],x['scene'],x['configuration'])==key];valid=[x for x in sub if x['top_identity'] is not None];ns=sum(x['original_top_success'] for x in valid);rr=sum(x['R'] for x in valid);dd=sum(x['D'] for x in valid)
 paired_scene.append(dict(zip(['stage','scene','configuration'],key))|dict(N=len(valid),original_success=ns,R=rr,D=dd,Net=(rr-dd)/len(valid) if valid else None,Damage=dd/ns if ns else None))
with open(out/'paired_transition_scene.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired_scene[0]));w.writeheader();w.writerows(paired_scene)
agg=[]
for key in sorted(set((x['stage'],x['configuration'],x['stratum']) for x in strata)):
 sub=[x for x in strata if (x['stage'],x['configuration'],x['stratum'])==key];n=sum(x['n'] for x in sub);m=sum(x['modified'] for x in sub)
 agg.append(dict(zip(['stage','configuration','stratum'],key))|dict(n=n,modified=m,improved=sum(x['improved'] for x in sub),degraded=sum(x['degraded'] for x in sub),improved_given_modified=sum(x['improved'] for x in sub)/m if m else None,degraded_given_modified=sum(x['degraded'] for x in sub)/m if m else None,oracle_coverage=float(np.mean([x['oracle'] for x in sub])),regret=float(np.mean([x['regret'] for x in sub]))))
write(out/'score_strata_summary.json',agg)
# Cases selected by first seed, then scene/frame ID, never by gain size.
cases=[]
for stage,method in [(stage,method) for stage in sorted(set(x['stage'] for x in rows)) for method in ['S4','S4-zero']]:
 sub=[x for x in rows if x['stage']==stage and x['configuration']==f'{method}_{SEEDS[0]}_CD']
 for typ,pred in [('rescued',lambda x:x['R']==1),('damaged',lambda x:x['D']==1),('unchanged_failed',lambda x:not x['original_top_success'] and x['selected_action']==0)]:
  found=next((x for x in sorted(sub,key=lambda x:(x['scene'],x['frame'])) if pred(x)),None)
  if found is None:cases.append(dict(stage=stage,method=method,category=typ,status='NO_CASE'));continue
  s,f,i,j=found['scene'],found['frame'],found['top_identity'],found['selected_action'];folder=R/f'bad_cases/{stage}_{method}_{typ}';folder.mkdir(parents=True,exist_ok=True)
  original,hyp,*_=export(s,f);np.savez_compressed(folder/'poses.npz',original=original[i],selected=hyp[i,j])
  for part in ['rgb','depth']:shutil.copyfile(DATA/f'scenes/scene_{s:04d}/kinect/{part}/{f:04d}.png',folder/f'{part}.png')
  write(folder/'case.json',dict(category=typ,**found,selection='earliest scene/frame; first preregistered seed; fixed original R0 no-CD Top-1 identity'))
  # Scientific plot: observed RGB/depth and pose axes, not a rendering of exact finger geometry.
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from PIL import Image
  from scipy.io import loadmat
  rgb=np.array(Image.open(folder/'rgb.png'));dep=np.array(Image.open(folder/'depth.png'));K=loadmat(DATA/f'scenes/scene_{s:04d}/kinect/meta/{f:04d}.mat')['intrinsic_matrix']
  fig,axs=plt.subplots(1,3,figsize=(15,5))
  for ax,g,title in [(axs[0],original[i],'Original'),(axs[1],hyp[i,j],'Selected')]:
   ax.imshow(rgb);rot=g[4:13].reshape(3,3);center=g[13:16];points=np.stack([center,center+rot[:,0]*.04,center+rot[:,1]*g[1]/2,center-rot[:,1]*g[1]/2]);uv=points@K.T;uv=uv[:,:2]/uv[:,2:]
   ax.plot(uv[:2,0],uv[:2,1],'r-',lw=2,label='approach axis (4 cm)');ax.plot(uv[2:,0],uv[2:,1],'c-',lw=2,label='width segment');ax.scatter(*uv[0],s=15,c='yellow');ax.set_title(f'{title}: depth={g[3]:.3f} m, width={g[1]:.3f} m');ax.set_xlim(max(0,uv[0,0]-180),min(rgb.shape[1],uv[0,0]+180));ax.set_ylim(min(rgb.shape[0],uv[0,1]+180),max(0,uv[0,1]-180));ax.legend(fontsize=7)
  axs[2].imshow(np.ma.masked_equal(dep,0),cmap='viridis');axs[2].set_title('Observed depth (raw units)')
  fig.suptitle(f'{stage} {typ}: scene {s:04d} frame {f:04d}, candidate {i}; mu {found["original_top_mu"]} -> {found["selected_top_mu"]}')
  fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
  cases.append(dict(method=method,category=typ,status='SAVED',path=str(folder.relative_to(R)),**found))
write(out/'bad_case_index.json',cases)
# Resource accounting separates elapsed wall time from measured process CPU.
label=[json.loads(p.read_text()) for p in (R/'results').glob('labels_*.json')]
evalr=[json.loads(p.read_text()) for p in (R/'evaluation').rglob('*.json') if p.name in ['MICRO.json','COMPLETE.json']]
fit=json.loads((out/'fit_complete.json').read_text());smoke=json.loads((out/'smoke_hook.json').read_text())
source_paths={'reused_NQS/code/common.py':N/'code/common.py'}
for name in ['models/economicgrasp.py','models/modules_economicgrasp.py','dataset/graspnet_dataset.py','utils/collision_detector.py']:
 source_paths['EconomicGrasp/'+name]=CODE/name
write(out/'runtime_sources.json',{name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in source_paths.items() if p.exists()})
files=[p for sub in ['labels','heads','results','evaluation','logs','code','bad_cases','exports','confirmation_inputs'] for p in (R/sub).rglob('*') if p.is_file()]
resources=dict(status='MEASURED',cpu_initial_workers=4,cpu_max_geometry_workers=8,short_auxiliary_overlap="Latency and gate-bootstrap helpers briefly overlapped eight evaluation workers: up to ten Python compute processes including auxiliaries. Geometry worker limit stayed eight; this is a disclosed process-count scheduling deviation.",gpu_device=0,gpu_shared=True,label_worker_wall_seconds=sum(x['seconds'] for x in label),label_cpu_seconds=sum(x.get('cpu_seconds',0) for x in label),evaluation_worker_wall_seconds=sum(x['seconds'] for x in evalr),evaluation_cpu_seconds=sum(x.get('cpu_seconds',0) for x in evalr),training_process_wall_seconds=fit['seconds'],training_peak_gpu_allocated_bytes=fit['gpu_peak_bytes'],smoke_gpu_phase_seconds=smoke['seconds'],new_owned_artifact_bytes=sum(p.stat().st_size for p in files),backbone_trainings=0,heads=3,label_unique_pose_count=196608,label_S0_extra_repeat_checks=128,known_cost_scope='Worker wall includes IO/loading. CPU seconds are per-process counters. Existing shared input/features/data excluded from new bytes. S0 GPU allocation may have remained reserved while CPU micro workers ran; its conservative upper bound is hook seconds plus sum micro worker wall.',io_separate_seconds=None,io_note='IO included in measured phase wall times; no separate IO timer was installed, do not infer zero.')
resources['smoke_gpu_reservation_upper_seconds']=smoke['seconds']+sum(x['seconds'] for p,x in [(p,json.loads(p.read_text())) for p in (out).glob('labels_*_micro.json')])
io_start=time.perf_counter();io_bytes=0;io_files=sorted((R/'labels').glob('*.npz'))[::8]
for p in io_files:
 with np.load(p) as z:
  for k in z.files:io_bytes+=z[k].nbytes
resources['warm_label_read_probe']=dict(files=len(io_files),materialized_bytes=io_bytes,seconds=time.perf_counter()-io_start,scope='Post-run warm filesystem-cache read probe; not original cold-IO cost and not excluded from phase wall times.')
resources['symmetry_check']=json.loads((out/'symmetry_check.json').read_text())
resources['additional_geometry_recheck_calls']=132
resources['gpu_hours_upper_bound']=(fit['seconds']+resources['smoke_gpu_reservation_upper_seconds'])/3600
resources['measured_worker_wall_hours']=(resources['label_worker_wall_seconds']+resources['evaluation_worker_wall_seconds'])/3600
write(out/'resource_summary.json',resources)
manifest=[]
for p in sorted(files):
 if p.suffix in ['.pt','.npz'] or p.parent.name=='code':manifest.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)))
write(out/'artifact_manifest.json',manifest)
print('FINAL_OUTPUTS',len(rows),len(cases),resources)
