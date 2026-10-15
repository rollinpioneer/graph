import json,csv,statistics,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
F=read(R/'configs/frozen_protocol.json');S=read(R/'configs/states.json')['states'];B=Path(F['parent_root']);new=read(R/'reports/branch_results.json');old=read(B/'reports/branch_results.json');result=read(R/'receipts/probe_result.json');ex=read(R/'receipts/exit.json')
assert result['status']=='COMPLETE' and result['assets_unchanged'] and ex['exit_code']==0
assert len(new)==252 and len({(x['state'],x['candidate'],x['sequence_seed']) for x in new})==252
assert sum(x['steps'] for x in new)==result['environment_steps']<=75600
assert max(max(x['first_transition_errors'].values()) for x in new)<=1e-8
assert result['optimizer_updates']==0
labels=['original_actor','resampled_actor','vabc'];names={'original_actor':'原 Actor','resampled_actor':'Actor 重采样','vabc':'VaBC'}
metrics=['progress_final','progress_min','wall_contact_fraction','stagnation_fraction','path_length','low_head_fraction']
for x in old:
 with np.load(x['path']) as a:
  x=dict(x);x['sequence_seed']=0;x['progress_final']=x['initial_distance']-x['final_distance'];x['progress_min']=x['initial_distance']-x['minimum_distance'];x['wall_contact_fraction']=x['wall_contact_steps']/x['steps'];x['path_length']=float(np.linalg.norm(np.diff(a['xy'],axis=0),axis=-1).sum());w=[np.linalg.norm(a['xy'][i:i+101]-a['xy'][i],axis=-1).max()<.5 for i in range(x['steps']-99)];x['stagnation_fraction']=float(np.mean(w)) if w else None
  # Store derived old metrics separately; parent files remain unchanged.
  if 'olds' not in globals():olds=[]
  olds.append(x)
lookup={(x['state'],x['candidate'],x['sequence_seed']):x for x in new+olds}
def mean(v):
 v=[x for x in v if x is not None];return float(np.mean(v)) if v else None
def pair(state,a,b,seeds):
 aa=[lookup[state,a,k] for k in seeds];bb=[lookup[state,b,k] for k in seeds]
 wins=sum(x['success']>y['success'] for x,y in zip(aa,bb));losses=sum(x['success']<y['success'] for x,y in zip(aa,bb))
 return dict(wins=wins,losses=losses,ties=len(seeds)-wins-losses,success_difference=(sum(x['success'] for x in aa)-sum(x['success'] for x in bb))/len(seeds),secondary_mean_differences={m:mean([x[m]-y[m] for x,y in zip(aa,bb) if x[m] is not None and y[m] is not None]) for m in metrics})
rows=[]
for s in S:
 sid=s['id'];ks=F['new_sequence_seeds'];pairs={a+'_vs_'+b:pair(sid,a,b,ks) for a,b in [('resampled_actor','original_actor'),('vabc','original_actor'),('vabc','resampled_actor')]}
 rows.append(dict(state=sid,category=s['category'],model_seed=s['seed'],new_successes={a:int(sum(lookup[sid,a,k]['success'] for k in ks)) for a in labels},old_successes={a:int(lookup[sid,a,0]['success']) for a in labels},pooled_successes={a:int(sum(lookup[sid,a,k]['success'] for k in [0]+ks)) for a in labels},paired_new=pairs,paired_pooled={a+'_vs_'+b:pair(sid,a,b,[0]+ks) for a,b in [('resampled_actor','original_actor'),('vabc','original_actor'),('vabc','resampled_actor')]},secondary_means={a:{m:mean([lookup[sid,a,k][m] for k in ks]) for m in metrics} for a in labels}))
rep=[]
for t in F['replication_targets']:
 p=next(x for x in rows if x['state']==t['state'])['paired_new'][t['candidate']+'_vs_original_actor'];rep.append(dict(**t,**p,passes=p['wins']>=4 and p['losses']<=1))
summary=dict(primary_new_suffix_count=7,secondary_pooled_suffix_count=8,states=rows,predeclared_replication=rep,replication_passes=sum(x['passes'] for x in rep),operational_continue=sum(x['passes'] for x in rep)>=2,environment_steps=result['environment_steps'],optimizer_updates=0,integrity=dict(first_transition_max_error=max(max(x['first_transition_errors'].values()) for x in new),checkpoint_agent_unchanged=True,exit_code=ex['exit_code']))
save(R/'reports/summary.json',summary);save(R/'reports/old_suffix_derived.json',olds)
flat=[{**{k:v for k,v in x.items() if not isinstance(v,(dict,list))},'dataset':'new'} for x in new]
with (R/'reports/new_branch_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
with (R/'reports/state_successes.csv').open('w') as f:
 fields=['state','category','model_seed']+[f'{a}_{d}' for d in ['new7','old1','pooled8'] for a in labels];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in rows:w.writerow({**{k:x[k] for k in fields[:3]},**{f'{a}_{d}':x[key][a] for d,key in [('new7','new_successes'),('old1','old_successes'),('pooled8','pooled_successes')] for a in labels}})
print(json.dumps({'replication':rep,'counts':[{k:x[k] for k in ['state','new_successes','old_successes']} for x in rows],'result':result,'exit':ex},indent=2))
