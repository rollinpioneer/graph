"""Predeclared split-equal paired statistics and S-first family lock."""
import os,json,csv,argparse,hashlib
from pathlib import Path
import numpy as np
from scoring import sha256

R=Path(os.environ['LQR_ROOT']);cfg=json.loads((R/'frozen_config.json').read_text());p=argparse.ArgumentParser();p.add_argument('--stage',choices=['P1','C1'],required=True);args=p.parse_args();stage=cfg[args.stage];splits=['seen','similar','novel'];seeds=[h['seed'] for h in cfg['heads']];families=['S','F05'] if args.stage=='P1' else [json.loads((R/'method_lock.json').read_text())['family']];frames=list(range(0,256,8)) if args.stage=='P1' else list(range(256));records=[];sources=[]
for split in splits:
 for scene in stage[split]:
  folder=R/f'evaluation/{args.stage}/{scene:04d}';done=json.loads((folder/'COMPLETE.json').read_text());assert done['status']=='PASS' and done['frames']==frames
  for frame in frames:
   path=folder/f'{frame:04d}.json';v=json.loads(path.read_text());assert len(v)==2*(1+3*len(families));records.extend([dict(split=split,**x) for x in v]);sources.append(dict(path=str(path.relative_to(R)),sha256=sha256(path)))
def write(name,value):(R/'results'/name).write_text(json.dumps(value,indent=2)+'\n')
def csv_write(name,rows):
 with open(R/'results'/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csv_write(f'{args.stage}_per_frame.csv',records);metrics=['ap','ap04','ap08','top1_valid','top1_collision','top1_empty','cd_rejection_rate'];scene_rows=[];table={};lookup={(x['scene'],x['frame'],x['configuration']):x for x in records}
for split in splits:
 for scene in stage[split]:
  for name in sorted({x['configuration'] for x in records}):
   v=[x for x in records if x['scene']==scene and x['configuration']==name];assert len(v)==len(frames);row=dict(split=split,scene=scene,configuration=name,frames=len(v),**{m:float(np.mean([x[m] for x in v])) for m in metrics});count=sum(x['selected_top50_count'] for x in v);row['selected_top50_collision_rate']=sum(x['selected_top50_collisions'] for x in v)/count if count else None;row['F05_fallback_frames']=sum(x['F05_fallback'] for x in v);scene_rows.append(row);table[scene,name]=row
csv_write(f'{args.stage}_per_scene.csv',scene_rows);metrics+=['selected_top50_collision_rate'];rng=np.random.default_rng(20261019);boot={split:rng.integers(0,len(stage[split]),(10000,len(stage[split]))) for split in splits};comparisons=[];differences=[];transitions=[]
for family in families:
 for cd in [False,True]:
  suffix='_CD' if cd else '';result=dict(family=family,use_cd=cd,metrics={})
  for metric in metrics:
   by_split={split:np.array([[table[scene,f'{family}_{seed}{suffix}'][metric]-table[scene,'R0'+suffix][metric] for scene in stage[split]] for seed in seeds]) for split in splits};scale=1 if metric.startswith('ap') else 100
   estimate=np.mean([v.mean() for v in by_split.values()])*scale;distribution=np.mean([v.mean(0)[boot[split]].mean(1) for split,v in by_split.items()],axis=0)*scale;seed_delta=np.mean([v.mean(1) for v in by_split.values()],axis=0)*scale
   result['metrics'][metric]=dict(delta=float(estimate),ci95=np.quantile(distribution,[.025,.975]).tolist(),ci90=np.quantile(distribution,[.05,.95]).tolist(),seed_delta=seed_delta.tolist(),split_delta={split:float(v.mean()*scale) for split,v in by_split.items()},scene_delta={str(scene):float(by_split[split][:,j].mean()*scale) for split in splits for j,scene in enumerate(stage[split])},positive_scenes=int(sum((v.mean(0)>0).sum() for v in by_split.values())),unit='AP percentage points' if metric.startswith('ap') else 'percentage points')
  comparisons.append(result)
  for split in splits:
   for scene in stage[split]:differences.append(dict(family=family,use_cd=cd,split=split,scene=scene,**{m:result['metrics'][m]['scene_delta'][str(scene)] for m in metrics}))
  for seed in seeds:
   for split in splits:
    better=worse=unchanged=0
    for scene in stage[split]:
     for frame in frames:
      a=lookup[scene,frame,f'{family}_{seed}{suffix}'];b=lookup[scene,frame,'R0'+suffix];better+=int(a['top1_valid'] and not b['top1_valid']);worse+=int(b['top1_valid'] and not a['top1_valid']);unchanged+=int(a['top_candidate']==b['top_candidate'])
    transitions.append(dict(family=family,use_cd=cd,seed=seed,split=split,N=len(stage[split])*len(frames),failure_to_success=better,success_to_failure=worse,unchanged_top_candidate=unchanged))
write(f'{args.stage}_comparisons.json',comparisons);csv_write(f'{args.stage}_per_scene_differences.csv',differences);csv_write(f'{args.stage}_top1_transitions.csv',transitions)
if args.stage=='P1':
 gates={}
 for family in families:
  x=next(x for x in comparisons if x['family']==family and x['use_cd'])['metrics']['ap'];gates[family]=dict(eligible=x['delta']>0 and sum(v>0 for v in x['seed_delta'])>=2,**x)
 chosen='S' if gates['S']['eligible'] else 'F05' if gates['F05']['eligible'] else None
 decision=dict(status='SELECTED_FOR_CONFIRMATION' if chosen else 'STOP_BOTH_FAMILIES_FAILED_MIGRATION_GATE',selected_family=chosen,gates=gates,rule='S priority, mean>0 and at least2/3 seeds>0; budget gate, not independent evidence',C1_opened=False)
 write('P1_decision.json',decision)
 if chosen:
  lock=dict(family=chosen,heads=cfg['heads'],formula='q' if chosen=='S' else cfg['F05'],config_sha256=sha256(R/'frozen_config.json'),P1_result_identities=sources,selection_rule=decision['rule'],C1_performance_read=False,ensemble=False)
  dest=R/'method_lock.json';assert not dest.exists(),'Do not overwrite a family lock';dest.write_text(json.dumps(lock,indent=2)+'\n')
 print(json.dumps(decision),flush=True)
else:
 x=next(x for x in comparisons if x['use_cd'])['metrics']['ap'];target=x['delta']>=1 and x['ci95'][0]>0 and all(v>0 for v in x['seed_delta']) and x['positive_scenes']>=8
 classification='AP_CONFIRMED_TARGET' if target else 'LIMITED_GAIN' if x['delta']>0 and x['ci95'][0]>0 else 'NEGATIVE' if x['ci95'][1]<0 else 'SMALL_DIFFERENCE_IN_THIS_SETTING' if x['ci90'][0]>=-.5 and x['ci90'][1]<=.5 else 'INCONCLUSIVE_OR_NOT_TARGET_REACHED';decision=dict(status=classification,family=families[0],primary=x,geometry_unchanged=True,ensemble=False)
 write('C1_decision.json',decision);print(json.dumps(decision),flush=True)
if args.stage=='C1':
 import subprocess,sys
 subprocess.run([sys.executable,str(R/'code/build_final_report.py')],check=True)
