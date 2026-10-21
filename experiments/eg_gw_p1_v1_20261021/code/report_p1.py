from common import *
from collections import defaultdict,Counter
cfg=json.loads((R/'frozen_config.json').read_text());groups=cfg['C3'];metrics=['ap','ap04','ap08','top1_valid','top1_nonempty_collision','top1_empty','no_output'];records=[];manifest=[];trans={False:Counter(),True:Counter()};categories={False:Counter(),True:Counter()}
for sp,scenes in groups.items():
 for s in scenes:
  for f in range(256):
   p=R/f'evaluation/development/{s:04d}/{f:04d}.json';zpath=p.with_suffix('.npz');vv=json.loads(p.read_text());table={v['configuration']:v for v in vv};assert len(table)==len(vv)==20
   with np.load(zpath) as z:
    for v in vv:
     ids=z['selected_'+v['configuration']];mu=z['quality'][ids];assert np.isfinite(mu).all();assert v['top_candidate']==(int(ids[0]) if len(ids) else None);assert bool(len(ids) and 0<mu[0]<=.4000001)==v['top1_valid']
   for cd in [False,True]:
    suffix='_CD' if cd else ''
    for seed in SEEDS:
     g=table[f'G_{seed}{suffix}'];w=table[f'GW_{seed}{suffix}'];t=trans[cd];t['units']+=1;t['changed']+=g['top_candidate']!=w['top_candidate'];t['recovered']+=not g['top1_valid'] and w['top1_valid'];t['regressed']+=g['top1_valid'] and not w['top1_valid']
     if g['top1_valid'] and not w['top1_valid']:categories[cd][w['top1_category']]+=1
   records.extend(dict(split=sp,**v) for v in vv);manifest.extend(dict(path=str(x.relative_to(R)),bytes=x.stat().st_size,sha256=sha(x)) for x in [p,zpath])
assert len(records)==61440
buckets=defaultdict(list)
for v in records:buckets[v['scene'],v['configuration']].append(v)
scene_rows=[];table={}
for (s,name),vv in sorted(buckets.items()):
 assert len(vv)==256;row=dict(scene=s,split=vv[0]['split'],configuration=name,**{m:float(np.mean([v[m] for v in vv])) for m in metrics});table[s,name]=row;scene_rows.append(row)
csv_write(R/'results/per_scene_results.csv',scene_rows)
family=[]
for fam in ['EG','O','G','GW']:
 for cd in [False,True]:
  for sp in ['overall',*groups]:
   rr=[v for v in scene_rows if v['configuration'].split('_')[0]==fam and v['configuration'].endswith('_CD')==cd and (sp=='overall' or v['split']==sp)];family.append(dict(family=fam,use_cd=cd,split=sp,**{m:float(np.mean([v[m] for v in rr])) for m in metrics}))
csv_write(R/'results/family_results.csv',family);rng=np.random.default_rng(20261021);boot={sp:rng.integers(0,len(ss),(10000,len(ss))) for sp,ss in groups.items()};comparisons=[];diffs=[]
for cd in [False,True]:
 suffix='_CD' if cd else '';res=dict(family='GW',reference='G',use_cd=cd,metrics={})
 for m in metrics:
  by={sp:np.array([[table[s,f'GW_{seed}{suffix}'][m]-table[s,f'G_{seed}{suffix}'][m] for s in ss] for seed in SEEDS]) for sp,ss in groups.items()};scale=1 if m.startswith('ap') else 100;dist=np.mean([v.mean(0)[boot[sp]].mean(1) for sp,v in by.items()],axis=0)*scale
  res['metrics'][m]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(dist,[.025,.975]).tolist(),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())),scene_delta={str(s):float(by[sp][:,i].mean()*scale) for sp,ss in groups.items() for i,s in enumerate(ss)})
 comparisons.append(res)
 for sp,ss in groups.items():
  for s in ss:diffs.append(dict(scene=s,split=sp,use_cd=cd,**{m:res['metrics'][m]['scene_delta'][str(s)] for m in metrics}))
 t=trans[cd];t['net']=t['recovered']-t['regressed'];assert abs(t['net']/9216*100-res['metrics']['top1_valid']['delta'])<1e-10
write(R/'results/comparisons.json',comparisons);csv_write(R/'results/per_scene_differences.csv',diffs);write(R/'results/transitions.json',{str(k):dict(v,regression_categories=categories[k]) for k,v in trans.items()})
v=comparisons[1]['metrics'];ap=v['ap'];top=v['top1_valid'];ap_route=ap['delta']>=.5 and ap['ci95'][0]>0 and top['ci95'][0]>=-.5 and sum(d>0 for d in ap['seed_delta'])>=2;top_route=top['delta']>=1 and top['ci95'][0]>0 and ap['ci95'][0]>=-.2 and sum(d>0 for d in top['seed_delta'])>=2;risk=v['top1_nonempty_collision']['delta']>.5
decision=dict(status='RISK_REVIEW' if risk else 'PASS' if ap_route or top_route else 'NOT_PASSED',AP_route=bool(ap_route),Top1_route=bool(top_route),collision_protection_triggered=bool(risk),C3_is_development=True,C4_read=False,P2='BLOCKED_F1',overall='GW_CANDIDATE_P2_BLOCKED' if (ap_route or top_route) and not risk else 'STOP_DIVISION_KEEP_G',P2_is_not_a_tested_negative=True)
write(R/'results/development_decision.json',decision);write(R/'results/evaluation_artifact_manifest.json',manifest)
receipts=[json.loads(p.read_text()) for p in (R/'results').glob('eval_*.json')];full=[v for v in receipts if not v['pilot']];assert len(full)==96 and sum(len(v['frames']) for v in full)==3072
events=sorted([(v['start'],1) for v in full]+[(v['end'],-1) for v in full]);cur=peak=0
for _,change in events:cur+=change;peak=max(peak,cur)
train=json.loads((R/'results/training_summary.json').read_text());rep=json.loads((R/'results/reproduction.json').read_text());assert all(v['status']=='REPRODUCED' for v in rep['results'])
size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink() and 'geometry' not in p.relative_to(R).parts);assert size<4*1024**3
validation=dict(status='PASS',frames=3072,configurations=20,records=61440,old_EG_O_G_selection_and_metrics_exact=True,selected_quality_finite=True,transitions_net_matches_metric=True,three_G_reruns_byte_exact=True,initial_and_sample_stream_match=True,C4_read=False,new_training_labels=0,new_evaluation_quality_calls=sum(v['new_evaluation_labels'] for v in receipts),CPU_worker_hours=sum(v['seconds'] for v in receipts)/3600,CPU_process_hours=sum(v['cpu_seconds'] for v in receipts)/3600,actual_evaluation_peak_processes=peak,GPU_process_hours=train['GPU_process_seconds']/3600,new_bytes=size)
write(R/'results/final_validation.json',validation)
lines=['# G-W P1 complete C3 report','',f"Decision: {decision['status']}; overall {decision['overall']}. P2 blocked at F1, not trained or evaluated.",'','| Method (CD) | AP | Top-1 % | Nonempty collision % |','|---|---:|---:|---:|']
for x in family:
 if x['use_cd'] and x['split']=='overall':lines.append(f"| {x['family']} | {x['ap']:.4f} | {x['top1_valid']*100:.4f} | {x['top1_nonempty_collision']*100:.4f} |")
lines+=['',f"GW-G AP {ap['delta']:+.4f}, 95% CI {ap['ci95']}; Top-1 {top['delta']:+.4f}pp, 95% CI {top['ci95']}.",f"AP route {ap_route}; Top-1 route {top_route}; collision protection {risk}.",'',f"Top-1 paired units 9216: changed {trans[True]['changed']}, recovered {trans[True]['recovered']}, regressed {trans[True]['regressed']}, net {trans[True]['net']}. Three seeds are not independent scene samples.",'','All six fixed-step training tasks completed on A100. Original G reruns are tensor-byte identical to protected G; no rerun C3 needed. GW initialization and group sample streams match G. Only the prescribed loss weights changed. Checkpoint step1000 only; no tuning, extra seeds or steps.','', 'C3 evaluation: 12 scenes x256 frames, CD and noCD, official selector/quality checks in pilot; original EG/O/G selections and metrics exactly reproduced for every frame. Newly needed quality labels used only for evaluation; union checked first. Same-seed comparisons, seed-average within scene, Seen/Similar/Novel stratified bootstrap 10000, seed20261021.','', 'C3 has been repeatedly used for development. These intervals are developmental evidence, not independent confirmation. P2 F1 max error 0.00321946 exceeds frozen1e-4; the mathematical affine readout premise does not guarantee this numerical equivalence in existing exports. No redesign or tolerance relaxation performed.','', 'Resources and validation:',json.dumps(validation,indent=2),'', 'Stop the tested loss-weight recipe if not passed. Preserve G. Do not combine methods or open C4/C5. P2 can only resume after separately approved numerical redesign.']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(decision=decision,comparison=v,validation=validation)))
