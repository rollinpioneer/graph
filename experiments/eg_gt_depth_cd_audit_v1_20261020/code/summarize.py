import os,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
R=Path(os.environ['R'])
def safe(v):
 if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [safe(x) for x in v]
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.bool_):return bool(v)
 return v
def write(p,x):Path(p).write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n')
files=sorted((R/'results').glob('shard_*.json'))
ds=[json.load(open(p)) for p in files]
assert len(ds)==24 and all(d['status']=='PASS' for d in ds),[(p.name,d['status'],d['frames'],d['planned_frames']) for p,d in zip(files,ds)]
rows=pd.DataFrame([r for d in ds for r in d['rows']]);pairs=pd.DataFrame([r for d in ds for r in d['pairs']]);objects=pd.DataFrame([r for d in ds for r in d['objects']])
assert sum(d['frames'] for d in ds)==len(json.load(open(R/'selection.json'))['jobs'])
assert len(pairs)==1177 and not rows.duplicated(['scene','frame','candidate']).any()
failure=rows[rows.role=='failure'].copy()
events=pairs[['scene','frame','head','failure']].merge(failure,left_on=['scene','frame','failure'],right_on=['scene','frame','candidate'],validate='many_to_one')
assert len(events)==1177
for df in (failure,events):
 df['few_points']=df.gt_points<=5
 df['all_uncovered']=df.cd_covered_count==0
 df['any_uncovered']=df.cd_covered_count<df.gt_points
 df['all_covered']=df.cd_covered_count==df.gt_points
 df['depth_occupancy_positive']=df.cd_depth_points>0
 df['only_table']=df.table_count==df.gt_points
 df['any_table']=df.table_count>0
 df['free80']=df.free_fraction>=.8
 df['free_smooth_same_segment_fraction']=df.free_smooth_same_segment_count/df.gt_points
 df['within5x5_surface_fraction']=df.within5x5_surface_count/df.gt_points
 df['required_depth_points']=np.floor(.01*(df.cd_equivalent_volume+1e-6))+1
 df['points_to_trigger']=df.required_depth_points-df.cd_depth_points
metrics=['few_points','all_uncovered','any_uncovered','all_covered','depth_occupancy_positive','only_table','any_table','free80','left_fraction','right_fraction','base_fraction','target_fraction','other_fraction','table_fraction','cd_covered_fraction','cd_body_fraction','cd_sweep_fraction','boundary2mm_fraction','free_fraction','surface_fraction','occluded_fraction','unknown_fraction','hole_near_fraction','depth_jump_fraction','seg_edge_fraction','same_segment_fraction','workspace_fraction','free_smooth_same_segment_fraction','within5x5_surface_fraction']
scene=events.groupby('scene')[metrics].mean()
rng=np.random.default_rng(20261020);boot=rng.integers(0,len(scene),(10000,len(scene)))
ci={k:dict(scene_mean=scene[k].mean(),ci95=np.quantile(scene[k].values[boot].mean(1),[.025,.975]).tolist(),event_mean=events[k].mean(),unique_mean=failure[k].mean()) for k in metrics}
qcols=['gt_points','gt_inner','boundary_mm_median','residual_mm_median','nearest_depth_mm_median','cd_iou','cd_depth_points','required_depth_points','points_to_trigger']
quant={k:events[k].quantile([0,.25,.5,.75,.9,1]).to_dict() for k in qcols}
paired=pairs[pairs.success.notna()].merge(rows,left_on=['scene','frame','failure'],right_on=['scene','frame','candidate']).merge(rows,left_on=['scene','frame','success'],right_on=['scene','frame','candidate'],suffixes=('_fail','_success'))
paircols=['cd_iou','cd_depth_points','gt_inner']
pdiff=pd.DataFrame({k:paired[k+'_fail']-paired[k+'_success'] for k in paircols});pdiff['scene']=paired.scene
pairsummary={k:dict(failure_mean=paired[k+'_fail'].mean(),success_mean=paired[k+'_success'].mean(),scene_mean_difference=pdiff.groupby('scene')[k].mean().mean()) for k in paircols}
objgroups=[]
for (s,o),d in objects.groupby(['scene','object_id']):
 v=dict(scene=int(s),object_id=int(o),frames=len(d),surface_distance_mm_median=d.surface_distance_mm_median.median())
 for axis in ['dx_mm','dy_mm','dz_mm','du_px','dv_px']:
  val=d[axis].median();v[axis]=val;v[axis+'_same_sign']=float((np.sign(d[axis])==np.sign(val)).mean())
 v['consistent_gt_depth_offset_flag']=bool(len(d)>=8 and any(abs(v[a])>10 and v[a+'_same_sign']>=.75 for a in ['dx_mm','dy_mm','dz_mm']))
 objgroups.append(v)
og=pd.DataFrame(objgroups)
ident=[r for d in ds for r in d['identities']]
checks=[r for d in ds for r in d['checks']]
summary=dict(status='PASS',frames=len(ident),collision_events=len(events),unique_failures=len(failure),unique_audited_candidates=len(rows),pairing=pairs.tier.value_counts().to_dict(),metrics=ci,quantiles=quant,paired_comparison=pairsummary,depth_bins=events.groupby('depth').agg(n=('candidate','size'),coverage=('cd_covered_fraction','mean'),base=('base_fraction','mean')).reset_index().to_dict('records'),registration=dict(object_frames=len(objects),object_groups=len(og),eligible_groups=int((og.frames>=8).sum()),consistent_offset_groups=og[og.consistent_gt_depth_offset_flag].to_dict('records'),median_object_frame_surface_distance_mm=objects.surface_distance_mm_median.median(),p90_object_frame_surface_distance_mm=objects.surface_distance_mm_median.quantile(.9),within4mm_mean=objects.within4mm.mean(),within10mm_mean=objects.within10mm.mean()),validation=dict(cd_candidate_checks=sum(x['cd_verified'] for x in checks),official_collision_checks=len(rows),max_xml_meta_pose_difference=max(x['xml_meta_pose_max_abs'] for x in ident),original_width_above_100mm=int((rows.raw_width>.1).sum()),principal_point_truncation_pixels=ident[0]['principal_truncation_px']),resource=dict(worker_wall_hours=sum(d['wall_seconds'] for d in ds)/3600,worker_cpu_hours=sum(d['cpu_seconds'] for d in ds)/3600,GPU_hours=0,max_workers=24,new_quality_labels=0,training=0))
write(R/'results/audit_summary.json',summary)
rows.to_csv(R/'results/candidates.csv',index=False);pairs.to_csv(R/'results/pairs.csv',index=False);events.to_csv(R/'results/collision_events.csv',index=False)
objects.to_csv(R/'results/object_frames.csv',index=False);og.to_csv(R/'results/object_consistency.csv',index=False)
scene.to_csv(R/'results/scene_metrics.csv');paired.to_csv(R/'results/paired_comparison.csv',index=False)
write(R/'results/identities.json',ident)
# Representative examples are explicitly post-hoc illustrations, never an unbiased sample.
examples={}
for tag,cond in [('uncovered',failure.all_uncovered),('occupied_below_threshold',failure.depth_occupancy_positive),('smooth_same_object_front',failure.free_smooth_same_segment_fraction>=.8),('depth_edge',failure.depth_jump_fraction>=.8),('table',failure.any_table)]:
 examples[tag]=failure[cond].sort_values(['scene','frame','candidate']).head(3).to_dict('records')
write(R/'results/examples.json',examples)
print(json.dumps(safe(summary),indent=2))
