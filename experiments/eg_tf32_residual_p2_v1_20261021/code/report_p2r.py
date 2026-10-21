from p2_residual import *
from collections import defaultdict,Counter
cfg=json.loads((R/'frozen_config.json').read_text());groups=cfg['C3'];metrics=['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty','no_output'];records=[];manifest=[];trans={(f,cd):Counter() for f in FAMILIES for cd in [False,True]}
for sp,scenes in groups.items():
 for s in scenes:
  for f in range(256):
   p=R/f'evaluation/development/{s:04d}/{f:04d}.json';zp=p.with_suffix('.npz');vv=json.loads(p.read_text());table={v['configuration']:v for v in vv};assert len(table)==len(vv)==32
   with np.load(zp) as z:
    for v in vv:
     ids=z['selected_'+v['configuration']];mu=z['quality'][ids];assert np.isfinite(mu).all();assert v['top_candidate']==(int(ids[0]) if len(ids) else None);assert bool(len(ids) and 0<mu[0]<=.4000001)==v['top1_valid']
   for cd in [False,True]:
    suffix='_CD' if cd else ''
    for fam in FAMILIES:
     for seed in SEEDS:
      g=table[f'G_{seed}{suffix}'];w=table[f'{fam}_{seed}{suffix}'];t=trans[fam,cd];t['units']+=1;t['changed']+=g['top_candidate']!=w['top_candidate'];t['recovered']+=not g['top1_valid'] and w['top1_valid'];t['regressed']+=g['top1_valid'] and not w['top1_valid']
   records.extend(dict(split=sp,**v) for v in vv);manifest.extend(dict(path=str(x.relative_to(R)),bytes=x.stat().st_size,sha256=sha(x)) for x in [p,zp])
assert len(records)==98304
buckets=defaultdict(list)
for v in records:buckets[v['scene'],v['configuration']].append(v)
scene_rows=[];table={}
for (s,name),vv in sorted(buckets.items()):
 assert len(vv)==256;row=dict(scene=s,split=vv[0]['split'],configuration=name,**{m:float(np.mean([v[m] for v in vv])) for m in metrics});table[s,name]=row;scene_rows.append(row)
csv_write(R/'results/per_scene_results.csv',scene_rows);family=[]
for fam in ['EG','O','G']+FAMILIES:
 for cd in [False,True]:
  for sp in ['overall',*groups]:
   rr=[v for v in scene_rows if v['configuration'].split('_')[0]==fam and v['configuration'].endswith('_CD')==cd and (sp=='overall' or v['split']==sp)];family.append(dict(family=fam,use_cd=cd,split=sp,**{m:float(np.mean([v[m] for v in rr])) for m in metrics}))
csv_write(R/'results/family_results.csv',family);rng=np.random.default_rng(20261021);boot={sp:rng.integers(0,len(ss),(10000,len(ss))) for sp,ss in groups.items()};comparisons=[];diffs=[]
for fam,ref in [(f,'G') for f in FAMILIES]+[('Eact','Epot'),('Eact','Gcap65')]:
 for cd in [False,True]:
  suffix='_CD' if cd else '';res=dict(family=fam,reference=ref,use_cd=cd,metrics={})
  for m in metrics:
   by={sp:np.array([[table[s,f'{fam}_{seed}{suffix}'][m]-table[s,f'{ref}_{seed}{suffix}'][m] for s in ss] for seed in SEEDS]) for sp,ss in groups.items()};scale=1 if m.startswith('ap') else 100;dist=np.mean([v.mean(0)[boot[sp]].mean(1) for sp,v in by.items()],axis=0)*scale
   res['metrics'][m]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(dist,[.025,.975]).tolist(),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())),scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,ss in groups.items() for i,s in enumerate(ss)})
  comparisons.append(res)
  for sp,ss in groups.items():
   for s in ss:diffs.append(dict(family=fam,reference=ref,scene=s,split=sp,use_cd=cd,**{m:res['metrics'][m]['scene_delta'][str(s)] for m in metrics}))
  if ref=='G':
   t=trans[fam,cd];t['net']=t['recovered']-t['regressed'];assert abs(t['net']/9216*100-res['metrics']['top1_valid']['delta'])<1e-10
write(R/'results/comparisons.json',comparisons);csv_write(R/'results/per_scene_differences.csv',diffs);write(R/'results/transitions.json',{f'{f}_{cd}':dict(v) for (f,cd),v in trans.items()})
def get(fam,ref='G'):return next(x['metrics'] for x in comparisons if x['family']==fam and x['reference']==ref and x['use_cd'])
def gate(v):
 ap=v['ap'];top=v['top1_valid'];ar=ap['delta']>=.5 and ap['ci95'][0]>0 and top['ci95'][0]>=-.5 and sum(d>0 for d in ap['seed_delta'])>=2;tr=top['delta']>=1 and top['ci95'][0]>0 and ap['ci95'][0]>=-.2 and sum(d>0 for d in top['seed_delta'])>=2;risk=v['top1_nonempty_collision']['delta']>.5
 return dict(AP_route=bool(ar),Top1_route=bool(tr),collision_protection_triggered=bool(risk),passed=bool((ar or tr) and not risk))
gates={fam:gate(get(fam)) for fam in FAMILIES};eg=gates['Eact'];route='AP' if eg['AP_route'] else 'Top1' if eg['Top1_route'] else None
def mechanism(ref):
 v=get('Eact',ref);primary='ap' if route=='AP' else 'top1_valid';other='top1_valid' if route=='AP' else 'ap';floor=-.5 if route=='AP' else -.2
 return bool(v[primary]['ci95'][0]>0 and v[other]['ci95'][0]>=floor)
status='E_ACT_FAIL';step2=step3=None
if eg['passed']:
 step2=mechanism('Epot');status='E_ACT_TARGET' if step2 else 'E_ACT_PERF_ONLY'
 if step2:step3=mechanism('Gcap65');status='E_ACT_FULL' if step3 else 'E_ACT_TARGET'
if eg['collision_protection_triggered']:status='RISK_REVIEW'
decision=dict(status=status,route=route,performance_gates=gates,sequential_Eact_vs_Epot=step2,sequential_Eact_vs_Gcap65=step3,untested_sequential_steps_are_not_claimed=True,only_E_ACT_FULL_counts=True,overall='E_ACT_FULL_DEVELOPMENT_CANDIDATE' if status=='E_ACT_FULL' else 'STOP_DIVISION_KEEP_G',controls_not_promoted=True,C3_is_development=True,C4_read=False,redesign='frozen TF32 score plus 65-parameter affine residual; not original pure affine model')
write(R/'results/sequential_decision.json',decision);write(R/'results/evaluation_artifact_manifest.json',manifest)
receipts=[json.loads(p.read_text()) for p in (R/'results').glob('eval_*.json')];full=[v for v in receipts if not v['pilot']];assert len(full)==96 and sum(len(v['frames']) for v in full)==3072
events=sorted([(v['start'],1) for v in full]+[(v['end'],-1) for v in full]);cur=peak=0
for _,change in events:cur+=change;peak=max(peak,cur)
train=json.loads((R/'results/training_summary.json').read_text());assert len(train['heads'])==9 and all(x['steps']==1000 and x['parameters']==65 and x['sample_stream_matches_G'] for x in train['heads'])
invariants=json.loads((R/'results/invariance_checks.json').read_text());assert invariants['status']=='PASS' and all(x['start']>invariants['completed_at'] for x in train['heads'])
assert sha(R/'code/eval_p2r_preflight.py')==invariants['preflight_code_sha256']
for h in cfg['protected_G']:assert sha(Path(h['path']))==h['sha256']
assert sha(RP/'prepared.npz')==cfg['protected_prepared_sha256'] and sha(L/'prepared_geometry.npz')==cfg['protected_geometry_sha256']
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size<cfg['storage_cap_GiB']*1024**3
validation=dict(status='PASS',frames=3072,configurations=32,records=98304,old_EG_O_G_selection_and_metrics_exact=True,selected_quality_finite=True,transitions_net_matches_metric=True,protected_G_and_feature_caches_unchanged=True,training_began_after_all_invariants_passed=True,C4_read=False,new_training_labels=0,new_evaluation_quality_calls=sum(v['new_evaluation_labels'] for v in receipts),CPU_worker_hours=sum(v['seconds'] for v in receipts)/3600,CPU_process_hours=sum(v['cpu_seconds'] for v in receipts)/3600,actual_evaluation_peak_processes=peak,GPU_process_hours=train['GPU_process_seconds']/3600,new_bytes=size,outside_01_fraction={fam:sum(v['outside_01'][fam] for v in full)/sum(v['score_candidates_per_family'] for v in full) for fam in ['Eact','Epot']})
write(R/'results/final_validation.json',validation)
lines=['# P2 frozen TF32 plus affine residual: complete C3 report','',f'Decision: {status}; overall {decision["overall"]}.','','| Method (CD) | AP | Top-1 % | Nonempty collision % |','|---|---:|---:|---:|']
for x in family:
 if x['use_cd'] and x['split']=='overall':lines.append(f"| {x['family']} | {x['ap']:.4f} | {x['top1_valid']*100:.4f} | {x['top1_nonempty_collision']*100:.4f} |")
lines+=['','| Comparison (CD) | delta AP [95% CI] | delta Top-1 pp [95% CI] |','|---|---|---|']
for fam,ref in [(f,'G') for f in FAMILIES]+[('Eact','Epot'),('Eact','Gcap65')]:
 v=get(fam,ref);ap=v['ap'];top=v['top1_valid'];lines.append(f"| {fam}-{ref} | {ap['delta']:+.4f} {ap['ci95']} | {top['delta']:+.4f} {top['ci95']} |")
lines+=['','All nine heads ran 1000 AdamW steps, lr0.001/wd0.0001, 128 a0 groups per step. Each has exactly 65 trainable residual parameters, initialized to zero; sample-stream hashes match corresponding protected G. Eact/Epot use alpha*(cached historical TF32 score + residual)+beta; alpha/beta use fit-only target moments. Gcap65 uses sigmoid(frozen G logit + residual). All residual arithmetic is float64; frozen G forward remains float32. No clipping, tuning, extra seeds, extended training or method combinations.','', 'This is an explicitly authorized change of function class, not a numerical repair of the original pure-affine hypothesis. The original pure-affine F1 remains failed. Prior exact TF32 replay evidence was reused by verified hash; all accessed export identities were checked again. Zero-training I1-I4 passed before any training. See invariance_checks.json.','', 'Full C3: 12 scenes x256 frames, CD/noCD. Official selection/quality pilot checked. Historical EG/O/G selections and metrics exactly reproduced every frame. P0 union and existing P1 labels are hash-verified, overlap-checked and reused; newly needed GT is evaluation-only. No C4/C5/reserve data or new training labels.','', 'Same-seed differences; average seeds within scene, then Seen/Similar/Novel stratified paired bootstrap10000, seed20261021. Repeatedly used C3 is development evidence, not independent confirmation. Only preregistered sequential comparisons reached by earlier passing gates support mechanism claims; controls are never promoted.','',json.dumps(decision,indent=2),'',json.dumps(validation,indent=2),'', 'Offline fit/held-out MSE, decision AUROC and OLS convergence gaps are diagnostic only, with no refitting or deployment of OLS. Training/evaluation caches and checkpoints remain on servers.']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(decision=decision,main_comparison=get('Eact'),validation=validation),indent=2))
