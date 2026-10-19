import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['exploration','confirmation'],required=True);a=p.parse_args();explore=a.phase=='exploration';groups=CFG['exploration'] if explore else json.loads((R/'C2_method_lock.json').read_text())['selected'];records=[]
for split in SPLITS:
 for scene in groups[split]:
  folder=R/f'evaluation/{a.phase}/{scene:04d}';done=json.loads((folder/'COMPLETE.json').read_text());assert done['status']=='PASS' and done['frames']==list(range(256))
  for frame in range(256):
   values=json.loads((folder/f'{frame:04d}.json').read_text());assert len(values)==(14 if explore else 8);records.extend([dict(split=split,**x) for x in values])
csv_write(f'results/{a.phase}_per_frame.csv',records)
metrics=['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','cd_rejection_rate','selected_top50_collision_rate'];table={};scene_rows=[]
for split in SPLITS:
 for scene in groups[split]:
  for name in sorted({x['configuration'] for x in records}):
   vs=[x for x in records if x['scene']==scene and x['configuration']==name];assert len(vs)==256
   row=dict(split=split,scene=scene,configuration=name,frames=256,**{m:float(np.mean([x[m] for x in vs])) for m in metrics if m!='selected_top50_collision_rate'});den=sum(x['selected_top50_count'] for x in vs);row['selected_top50_collision_rate']=sum(x['selected_top50_collisions'] for x in vs)/den if den else 0.;row['F05_fallback_frames']=sum(x['F05_fallback'] for x in vs);table[scene,name]=row;scene_rows.append(row)
csv_write(f'results/{a.phase}_per_scene.csv',scene_rows)
rng=np.random.default_rng(20261019);boot={s:rng.integers(0,len(groups[s]),(10000,len(groups[s]))) for s in SPLITS};comparisons=[];diffrows=[]
pairs=[('F05','R0')]+([('S','R0'),('F05','S')] if explore else [])
for family,reference in pairs:
 for cd in [False,True]:
  suffix='_CD' if cd else '';result=dict(family=family,reference=reference,use_cd=cd,metrics={})
  for metric in metrics:
   by={split:np.array([[table[scene,f'{family}_{seed}{suffix}'][metric]-table[scene,(f'{reference}_{seed}' if reference!='R0' else 'R0')+suffix][metric] for scene in groups[split]] for seed in SEEDS]) for split in SPLITS};scale=1 if metric.startswith('ap') else 100
   distribution=np.mean([by[s].mean(0)[boot[s]].mean(1) for s in SPLITS],axis=0)*scale
   result['metrics'][metric]=dict(delta=float(np.mean([v.mean() for v in by.values()])*scale),ci95=np.quantile(distribution,[.025,.975]).tolist(),ci90=np.quantile(distribution,[.05,.95]).tolist(),seed_delta=(np.mean([v.mean(1) for v in by.values()],axis=0)*scale).tolist(),split_delta={s:float(v.mean()*scale) for s,v in by.items()},scene_delta={str(scene):float(by[s][:,j].mean()*scale) for s in SPLITS for j,scene in enumerate(groups[s])},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by.values())))
  comparisons.append(result)
  for split in SPLITS:
   for scene in groups[split]:diffrows.append(dict(family=family,reference=reference,use_cd=cd,split=split,scene=scene,**{m:result['metrics'][m]['scene_delta'][str(scene)] for m in metrics}))
write(f'results/{a.phase}_comparisons.json',comparisons);csv_write(f'results/{a.phase}_per_scene_differences.csv',diffrows)
lookup={(x['scene'],x['frame'],x['configuration']):x for x in records};transitions=[]
for family,reference in pairs:
 for cd in [False,True]:
  suffix='_CD' if cd else ''
  for seed in SEEDS:
   for split in SPLITS:
    rescue=damage=same=0
    for scene in groups[split]:
     for frame in range(256):
      x=lookup[scene,frame,f'{family}_{seed}{suffix}'];y=lookup[scene,frame,(f'{reference}_{seed}' if reference!='R0' else 'R0')+suffix];rescue+=x['top1_valid'] and not y['top1_valid'];damage+=y['top1_valid'] and not x['top1_valid'];same+=x['top_candidate']==y['top_candidate']
    transitions.append(dict(family=family,reference=reference,use_cd=cd,seed=seed,split=split,N=256*len(groups[split]),failure_to_success=rescue,success_to_failure=damage,unchanged_top_candidate=same))
csv_write(f'results/{a.phase}_top1_transitions.csv',transitions)
x=next(c for c in comparisons if c['family']=='F05' and c['reference']=='R0' and c['use_cd'])['metrics']['ap']
if explore:
 s=next(c for c in comparisons if c['family']=='S' and c['reference']=='R0' and c['use_cd'])['metrics']['ap'];f_to_s=next(c for c in comparisons if c['family']=='F05' and c['reference']=='S' and c['use_cd'])['metrics']['ap']
 checks=dict(mean_positive=x['delta']>0,at_least_two_heads_positive=sum(v>0 for v in x['seed_delta'])>=2,scene0136_better_than_S=f_to_s['scene_delta']['136']>0,worst_scene_better_than_S=min(x['scene_delta'].values())>min(s['scene_delta'].values()))
 decision=dict(status='PASS_FOR_C2' if all(checks.values()) else 'STOP_F05_EXPLORATORY_GATE_FAILED',checks=checks,primary=x,S_primary=s,F05_minus_S=f_to_s,C1_is_exploratory=True,independent_confirmation=False)
else:
 target=x['delta']>=1 and x['ci95'][0]>0 and all(v>0 for v in x['seed_delta']) and x['positive_scenes']>=8
 classification='AP_CONFIRMED_TARGET' if target else 'LIMITED_GAIN' if x['delta']>0 and x['ci95'][0]>0 else 'NEGATIVE' if x['ci95'][1]<0 else 'SMALL_DIFFERENCE_IN_THIS_SETTING' if x['ci90'][0]>=-.5 and x['ci90'][1]<=.5 else 'INCONCLUSIVE_OR_NOT_TARGET_REACHED'
 decision=dict(status=classification,primary=x,family='F05',ensemble=False,geometry_unchanged=True)
write(f'results/{a.phase}_decision.json',decision);print(json.dumps(decision),flush=True)
