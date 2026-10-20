from common import *

cfg=json.load(open(R/'frozen_config.json'));assert json.load(open(R/'results/evaluation_complete.json'))['status']=='PASS'
splits=cfg['evaluation']['C3'];scenes=sum(splits.values(),[]);metrics={'AP':'ap','AP04':'ap04','AP08':'ap08','Top1_pct':'top1_valid','Top1_nonempty_collision_pct':'top1_nonempty_collision','Top1_empty_pct':'top1_empty'};rows=[];artifacts=[]
for scene in scenes:
 frames=[]
 for frame in range(256):
  p=R/f'evaluation/development/{scene:04d}/{frame:04d}.json';v=json.load(open(p));assert len(v)==24;assert all(x['scene']==scene and x['frame']==frame for x in v);frames.extend(v);artifacts.append(dict(path=str(p),sha256=sha(p)))
 for family in ['O','G','C','S']:
  for seed in SEEDS:
   for cd in [False,True]:
    selected=[x for x in frames if x['family']==family and x['seed']==seed and x['use_cd']==cd];assert len(selected)==256
    row=dict(scene=scene,split=next(k for k,s in splits.items() if scene in s),family=family,seed=seed,use_cd=cd,frames=256)
    row.update({k:float(np.mean([x[key] for x in selected]))*(100 if k.endswith('pct') else 1) for k,key in metrics.items()});rows.append(row)
csv_write(R/'results/per_head_per_scene.csv',rows);write(R/'results/evaluation_artifact_manifest.json',artifacts)
scene_rows=[]
for cd in [False,True]:
 for family in ['O','G','C','S']:
  for scene in scenes:
   selected=[x for x in rows if x['use_cd']==cd and x['family']==family and x['scene']==scene];assert len(selected)==3
   scene_rows.append(dict(scene=scene,split=selected[0]['split'],family=family,use_cd=cd,**{k:float(np.mean([x[k] for x in selected])) for k in metrics}))
csv_write(R/'results/per_scene_results.csv',scene_rows)
rng=np.random.default_rng(20261020);indices=np.concatenate([rng.integers(4*i,4*i+4,size=(10000,4)) for i in range(3)],axis=1);comparisons=[];summaries=[];per_seed=[]
for cd in [False,True]:
 for family in ['O','G','C','S']:
  selected=[v for v in scene_rows if v['family']==family and v['use_cd']==cd]
  for split in ['all']+list(splits):
   subset=selected if split=='all' else [v for v in selected if v['split']==split]
   summaries.append(dict(family=family,use_cd=cd,split=split,scenes=len(subset),**{k:float(np.mean([x[k] for x in subset])) for k in metrics}))
  for seed in SEEDS:
   selected_seed=[v for v in rows if v['family']==family and v['use_cd']==cd and v['seed']==seed]
   per_seed.append(dict(family=family,seed=seed,use_cd=cd,**{k:float(np.mean([x[k] for x in selected_seed])) for k in metrics}))
 for left,right in [('C','G'),('S','G'),('C','O'),('S','O'),('S','C')]:
  l={v['scene']:v for v in scene_rows if v['family']==left and v['use_cd']==cd};rr={v['scene']:v for v in scene_rows if v['family']==right and v['use_cd']==cd};detail={}
  for metric in metrics:
   delta=np.array([l[s][metric]-rr[s][metric] for s in scenes]);boot=delta[indices].mean(-1);detail[metric]=dict(delta=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist(),positive_scenes=int((delta>0).sum()),negative_scenes=int((delta<0).sum()),per_scene=[dict(scene=s,delta=float(x)) for s,x in zip(scenes,delta)])
  for metric in ['AP','Top1_pct']:
   detail[metric]['positive_seeds']=sum(next(x for x in per_seed if x['family']==left and x['seed']==seed and x['use_cd']==cd)[metric]>next(x for x in per_seed if x['family']==right and x['seed']==seed and x['use_cd']==cd)[metric] for seed in SEEDS)
  comparisons.append(dict(comparison=f'{left}-{right}',use_cd=cd,metrics=detail))
eligible=[];preliminary={}
for family in ['C','S']:
 comp=next(v for v in comparisons if v['comparison']==family+'-G' and v['use_cd']);aa=comp['metrics']['AP'];tt=comp['metrics']['Top1_pct']
 if aa['delta']>=-1 and tt['delta']>=-2:eligible.append(family)
 ap_pass=aa['delta']>=.75 and aa['CI95'][0]>0 and tt['delta']>=-.5 and tt['CI95'][0]>=-1 and aa['positive_seeds']>=2
 top_pass=tt['delta']>=1.5 and tt['CI95'][0]>0 and aa['delta']>=-.25 and aa['CI95'][0]>=-.5 and tt['positive_seeds']>=2
 efficient=aa['delta']>=-.25 and aa['CI95'][0]>=-.5 and tt['delta']>=-.5 and tt['CI95'][0]>=-1
 preliminary[family]=dict(A_performance_only=ap_pass or top_pass,B_performance_only=efficient,full_decision_pending_P3=True)
result=dict(status='PASS',development_only=True,C4_run=False,frames=3072,records=73728,scene_count=12,seeds_per_scene=3,metrics=summaries,per_seed=per_seed,comparisons=comparisons,P3_eligible=eligible,preliminary=preliminary,bootstrap=dict(iterations=10000,seed=20261020,stratified=True,unit='scene; seeds averaged within scene first'))
write(R/'results/C3_summary.json',result);csv_write(R/'results/per_head_metrics.csv',per_seed);csv_write(R/'results/group_metrics.csv',summaries)
print('C3_COMPLETE',json.dumps(dict(eligible=eligible,primary=[x for x in summaries if x['split']=='all' and x['use_cd']],preliminary=preliminary)),flush=True)
