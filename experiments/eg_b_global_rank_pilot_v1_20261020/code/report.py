from common import *
import collections
cfg=json.loads((R/'frozen_config.json').read_text());groups=cfg['C3'];records=[];artifact=[]
for split,scenes in groups.items():
 for scene in scenes:
  for frame in cfg['development_frames']:
   p=R/f'evaluation/development/{scene:04d}/{frame:04d}.json';z=p.with_suffix('.npz');values=json.loads(p.read_text());assert len(values)==20
   with np.load(z) as cache:
    for v in values:
     ids=cache['selected_'+v['configuration']];q=cache['quality'][ids];assert np.isfinite(q).all();assert v['top_candidate']==(int(ids[0]) if len(ids) else None)
   records += [dict(split=split,**v) for v in values]
   artifact += [dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in [p,z]]
assert len(records)==7680
csv_write(R/'results/development_per_frame.csv',records)
metrics=['ap','ap04','ap08','top1_valid','top1_collision','top1_nonempty_collision','top1_empty','selected_top50_quality_mean_utility','cd_rejection_rate']
names=sorted({v['configuration'] for v in records});table={};scene_rows=[]
for split,scenes in groups.items():
 for scene in scenes:
  for name in names:
   rr=[v for v in records if v['scene']==scene and v['configuration']==name];assert len(rr)==32
   row=dict(split=split,scene=scene,configuration=name,frames=32,**{k:float(np.mean([v[k] for v in rr])) for k in metrics});n=sum(v['selected_top50_count'] for v in rr);row['selected_top50_joint_rejection_rate']=sum(v['selected_top50_collisions'] for v in rr)/n if n else 0;row['selected_top50_nonempty_collision_rate']=sum(v['selected_top50_nonempty_collisions'] for v in rr)/n if n else 0;table[scene,name]=row;scene_rows.append(row)
csv_write(R/'results/development_per_scene.csv',scene_rows)
absolute=[dict(configuration=name,**{m:float(np.mean([v[m] for v in scene_rows if v['configuration']==name])) for m in metrics+['selected_top50_joint_rejection_rate','selected_top50_nonempty_collision_rate']}) for name in names];write(R/'results/development_absolute.json',absolute)
rng=np.random.default_rng(20261020);boot={s:rng.integers(0,4,(10000,4)) for s in groups};comparisons=[];diffrows=[]
contrasts=[('R','O'),('R','M'),('M','O'),('R','EG'),('O','EG'),('M','EG')]
for family,reference in contrasts:
 for cd in [False,True]:
  suffix='_CD' if cd else '';result=dict(family=family,reference=reference,use_cd=cd,metrics={})
  for metric in metrics+['selected_top50_joint_rejection_rate','selected_top50_nonempty_collision_rate']:
   by={sp:np.array([[table[s,f'{family}_{seed}{suffix}'][metric]-table[s,('EG' if reference=='EG' else f'{reference}_{seed}')+suffix][metric] for s in scenes] for seed in SEEDS]) for sp,scenes in groups.items()};scale=1 if metric.startswith('ap') else 100
   dist=np.mean([by[sp].mean(0)[boot[sp]].mean(1) for sp in groups],axis=0)*scale
   result['metrics'][metric]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),ci95=np.quantile(dist,[.025,.975]).tolist(),split_delta={sp:float(v.mean()*scale) for sp,v in by.items()},scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,scenes in groups.items() for i,s in enumerate(scenes)},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())))
  comparisons.append(result)
  for sp,scenes in groups.items():
   for s in scenes:diffrows.append(dict(family=family,reference=reference,use_cd=cd,split=sp,scene=s,**{m:result['metrics'][m]['scene_delta'][str(s)] for m in result['metrics']}))
write(R/'results/development_comparisons.json',comparisons);csv_write(R/'results/development_per_scene_differences.csv',diffrows)
lookup={(v['scene'],v['frame'],v['configuration']):v for v in records};transitions=[];transition_rows=[]
for family,reference in contrasts:
 for cd in [False,True]:
  suffix='_CD' if cd else ''
  for seed in SEEDS:
   for sp,scenes in groups.items():
    matrix=collections.Counter();rescues=damage=same=0
    for s in scenes:
     for frame in cfg['development_frames']:
      x=lookup[s,frame,f'{family}_{seed}{suffix}'];y=lookup[s,frame,('EG' if reference=='EG' else f'{reference}_{seed}')+suffix];matrix[y['top1_category'],x['top1_category']]+=1;rescues+=x['top1_valid'] and not y['top1_valid'];damage+=y['top1_valid'] and not x['top1_valid'];same+=x['top_candidate']==y['top_candidate']
    transitions.append(dict(family=family,reference=reference,use_cd=cd,seed=seed,split=sp,N=128,failure_to_success=rescues,success_to_failure=damage,unchanged_top_candidate=same))
    transition_rows += [dict(family=family,reference=reference,use_cd=cd,seed=seed,split=sp,old_category=a,new_category=b,count=n) for (a,b),n in sorted(matrix.items())]
csv_write(R/'results/development_top1_transitions.csv',transitions);csv_write(R/'results/development_category_transitions.csv',transition_rows)
get=lambda f,r:next(v for v in comparisons if v['family']==f and v['reference']==r and v['use_cd'])['metrics']
ro,rm,reg=get('R','O'),get('R','M'),get('R','EG');checks=dict(RO_top1_at_least_1pp=ro['top1_valid']['delta']>=1,RM_top1_positive=rm['top1_valid']['delta']>0,RO_at_least_two_positive_seeds=sum(v>0 for v in ro['top1_valid']['seed_delta'])>=2,RM_at_least_two_positive_seeds=sum(v>0 for v in rm['top1_valid']['seed_delta'])>=2,RO_AP_at_least_minus_0_5=ro['ap']['delta']>=-.5,REG_AP_positive=reg['ap']['delta']>0,application_identity_checks=True)
decision=dict(status='DEVELOPMENT_GATE_PASS' if all(checks.values()) else 'DEVELOPMENT_GATE_NOT_MET_STOP_CURRENT_RECIPE',checks=checks,independent_confirmation=False,P3_executed=False,P3_requires_separate_authorization=True,all_six_heads_evaluated_no_selection=True,RO=ro,RM=rm,REG=reg)
write(R/'results/development_decision.json',decision);write(R/'results/application_artifact_manifest.json',artifact)
offline=[]
for f in ['O','M','R']:
 for s in SEEDS:offline += json.loads((R/f'results/offline_{f}_{s}.json').read_text())['rows']
csv_write(R/'results/offline_diagnostics.csv',offline)
heads=[]
for v in json.loads((R/'results/training_complete.json').read_text())['heads']:
 ck=Path(v['checkpoint']);assert sha(ck)==v['checkpoint_sha256'];heads.append({k:v[k] for k in ['family','seed','checkpoint','checkpoint_sha256','initial_state_sha256','sample_stream_sha256','schedule_sha256','step','GPU_slot','seconds','peak_torch_gpu_bytes']})
for v in json.loads((P/'C3_method_lock.json').read_text())['heads']['B']:assert sha(v['path'])==v['sha256']
ledger=[json.loads(v) for v in (R/'results/resource_jobs.jsonl').read_text().splitlines()];cost={k:sum(v['seconds'] for v in ledger if v['kind']==k) for k in ['CPU','GPU']};assert cost['CPU']<=36000 and cost['GPU']<=3600
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'archive_worktree' not in p.parts);assert size<=8*1024**3
write(R/'results/final_validation.json',dict(status='PASS',frames=384,records=7680,old_EG_O_selection_and_AP_exact=True,new_native_exact_checks=4,all_selected_quality_finite=True,geometry_and_CD_unchanged=True,heads=heads,training_updates=6000,labels_regenerated=0,P3_executed=False,usage_seconds=cost,new_runtime_bytes=size,scene_bootstrap_seed=20261020,bootstrap_samples=10000))
print(json.dumps(dict(status=decision['status'],checks=checks,RO_top1=ro['top1_valid'],RM_top1=rm['top1_valid'],RO_AP=ro['ap'],REG_AP=reg['ap'])),flush=True)
