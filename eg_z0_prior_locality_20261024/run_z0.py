"""Preregistered Z0, CPU-only lookup of existing direct-pose evaluations."""
import json,csv,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text())
ANGLE=R.parent/'eg_i1_angle_physics_20261024/evaluation';CONTROL=R.parent/'eg_i1_collision_controls_20261024/results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows):
 with (R/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load():
 data=defaultdict(list);manifest=[]
 for s in P['scenes']:
  for f in P['frames']:
   ap=ANGLE/f'{s:04d}_{f:04d}.npz';cp=CONTROL/ap.name;a=np.load(ap);c=np.load(cp)
   ar=json.loads(ap.with_suffix('.json').read_text());cr=json.loads(cp.with_suffix('.json').read_text())
   assert sha(ap)==ar['result_sha256']==cr['prior_variants_sha256'];assert sha(cp)==cr['result_sha256']
   assert np.array_equal(a['candidate_id'],c['candidate_id']) and np.array_equal(a['original_angle'],c['original_angle'])
   n=len(a['candidate_id']);assert a['angle_logits'].shape==(13,n)
   for k in ['candidate_id','original_angle','object_id','mu','collision','empty','I1']:data[k].append(a[k])
   data['logits'].append(a['angle_logits'][:12].T);data['EG_cd'].append(c['EG_cd']);data['scene'].append(np.full(n,s));data['frame'].append(np.full(n,f))
   manifest.extend([dict(path=str(p),sha256=sha(p)) for p in [ap,cp]])
 a={k:np.concatenate(v) for k,v in data.items()};n=len(a['candidate_id']);assert n==1229 and a['mu'].shape==(n,12)
 assert np.isfinite(a['mu']).all() and np.isfinite(a['logits']).all()
 ix=np.arange(n);nom=a['original_angle'];assert (~a['collision'][ix,nom]&~a['empty'][ix,nom]).all()
 oldmu=a['mu'][ix,nom];assert ((oldmu>0)&(oldmu<=.4)).sum()==641 and (oldmu>.4).sum()==588
 (R/'input_manifest.json').write_text(json.dumps(dict(inputs=manifest,protocol_sha256=sha(R/'protocol.json'),prereg_sha256=sha(R/'preregistered_plan.md')),indent=2))
 return a
START=time.perf_counter();A=load();N=len(A['candidate_id']);IX=np.arange(N);NOM=A['original_angle'];CLASSES=np.arange(12)
DIST=np.minimum((CLASSES[None,:]-NOM[:,None])%12,(NOM[:,None]-CLASSES[None,:])%12)
ORDER=np.argsort(-A['logits'],axis=1,kind='stable');RANK=rankdata(-A['logits'],axis=1,method='average')
MISMATCH=np.flatnonzero(ORDER[:,0]!=NOM);OLDMU=A['mu'][IX,NOM];GOOD=(OLDMU>0)&(OLDMU<=.4);WEAK=OLDMU>.4
TIEKEY=DIST*12+CLASSES[None,:];TIEKEY[IX,NOM]=-1
SCENES=np.array(sorted(P['scenes']));SCENE_INDEX=np.searchsorted(SCENES,A['scene']);COUNTS=np.bincount(SCENE_INDEX,minlength=12)
DEN=np.stack([COUNTS,np.bincount(SCENE_INDEX,weights=WEAK,minlength=12),np.bincount(SCENE_INDEX,weights=GOOD,minlength=12),COUNTS,COUNTS,COUNTS],axis=1)
METRICS=['feasible','recovery','harm','collision','empty','no_FC']
rng_boot=np.random.default_rng(P['seed']);BOOT=np.concatenate([rng_boot.choice(np.flatnonzero((SCENES>=lo)&(SCENES<hi)),(10000,4),replace=True) for lo,hi in [(100,130),(130,160),(160,190)]],axis=1)
def interval(x):
 x=np.asarray(x,float);b=x[BOOT].mean(axis=1);return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)))
def choose(allowed):
 valid=allowed&np.isfinite(A['I1']);minimum=np.min(np.where(valid,A['I1'],np.inf),axis=1)
 ties=valid&np.isclose(A['I1'],minimum[:,None],rtol=0,atol=1e-7)
 out=np.argmin(np.where(ties,TIEKEY,999),axis=1);fallback=~valid.any(axis=1);out[fallback]=NOM[fallback]
 return out,fallback
def scalar_choose(i,allowed):
 valid=allowed&np.isfinite(A['I1'][i])
 if not valid.any():return int(NOM[i])
 ix=np.flatnonzero(valid);ties=ix[np.isclose(A['I1'][i,ix],A['I1'][i,ix].min(),rtol=0,atol=1e-7)];d=DIST[i,ties]
 return int(NOM[i] if NOM[i] in ties else ties[np.lexsort((ties,d))][0])
def outcomes(angles):
 mu=A['mu'][IX,angles];co=A['collision'][IX,angles];em=A['empty'][IX,angles];feas=~co&~em&(mu>0)&(mu<=.4)
 return np.stack([feas,WEAK&feas,GOOD&~feas,co&~em,em,~co&~em&(mu<=0)],axis=1)
def scene_totals(o):return np.stack([np.bincount(SCENE_INDEX,weights=o[:,k],minlength=12) for k in range(len(METRICS))],axis=1)
selected=[];set_rows=[];summaries=[];scene_rows=[];paired=[];outcome_mean={};choice_archive={};random_draw_rows=[];random_range={};effective=[]
for k in [3,5]:
 prior=np.zeros((N,12),bool);prior[IX[:,None],ORDER[:,:k]]=True
 missing=~prior[IX,NOM];prior[IX[missing],ORDER[missing,k-1]]=False;prior[IX,NOM]=True
 local=DIST<=k//2
 for name,base in [('P',prior),('L',local)]:
  assert (base.sum(1)==k).all() and base[IX,NOM].all()
  for filtered in [False,True]:
   filt='EG_cd' if filtered else 'none';key=f'{name}{k}_{filt}';allowed=base&~A['EG_cd'] if filtered else base
   angles,fb=choose(allowed);assert all(angles[i]==scalar_choose(i,allowed[i]) for i in range(N))
   o=outcomes(angles);outcome_mean[key]=o.astype(float);choice_archive[key]=angles.astype(np.uint8)
   for i in range(N):
    selected.append(dict(scene=int(A['scene'][i]),frame=int(A['frame'][i]),candidate_id=int(A['candidate_id'][i]),object_id=int(A['object_id'][i]),group=key,k=k,filter=filt,original_angle=int(NOM[i]),chosen_angle=int(angles[i]),cyclic_distance=int(DIST[i,angles[i]]),original_mu=float(OLDMU[i]),mu=float(A['mu'][i,angles[i]]),fallback=int(fb[i]),**{m:int(o[i,j]) for j,m in enumerate(METRICS)}))
    effective.append(dict(group=key,scene=int(A['scene'][i]),frame=int(A['frame'][i]),candidate_id=int(A['candidate_id'][i]),set_size=k,finite_I1=int((base[i]&np.isfinite(A['I1'][i])).sum()),eligible=int((allowed[i]&np.isfinite(A['I1'][i])).sum())))
   choice_archive[key+'_mask']=base
for k in [3,5]:
 for filt in ['none','EG_cd']:
  key=f'R{k}_{filt}';choice_archive[key]=np.zeros((1000,N),np.uint8);outcome_mean[key]=np.zeros((N,len(METRICS)))
rng=np.random.default_rng(P['seed']);random_counts=defaultdict(list)
for draw in range(1000):
 u=rng.random((N,12));u[IX,NOM]=np.inf;permutation=np.argsort(u,axis=1,kind='stable')
 for k in [3,5]:
  base=np.zeros((N,12),bool);base[IX[:,None],permutation[:,:k-1]]=True;base[IX,NOM]=True
  for filt in ['none','EG_cd']:
   key=f'R{k}_{filt}';allowed=base&~A['EG_cd'] if filt=='EG_cd' else base;angles,fb=choose(allowed);o=outcomes(angles)
   choice_archive[key][draw]=angles;outcome_mean[key]+=o/1000;cnt=o.sum(0);random_counts[key].append(cnt)
   random_draw_rows.append(dict(group=key,draw=draw,fallback=int(fb.sum()),**dict(zip(METRICS,map(int,cnt)))))
   if draw==0:assert all(angles[i]==scalar_choose(i,allowed[i]) for i in range(N))
 for key in ['R3_none','R5_none']:assert np.all(np.isin(choice_archive[key][draw],CLASSES))
for key,o in outcome_mean.items():
 totals=scene_totals(o);rates=totals/DEN;summary=dict(group=key,**{m:float(o[:,j].sum()) for j,m in enumerate(METRICS)})
 if key.startswith('R'):
  arr=np.array(random_counts[key]);summary['monte_carlo_95_range']={m:[float(np.quantile(arr[:,j],.025)),float(np.quantile(arr[:,j],.975))] for j,m in enumerate(METRICS)}
 for j,m in enumerate(METRICS):
  est=interval(rates[:,j]);summary[m+'_scene_rate']=est
  for si,s in enumerate(SCENES):scene_rows.append(dict(group=key,scene=int(s),metric=m,count=float(totals[si,j]),denominator=int(DEN[si,j]),rate=float(rates[si,j])))
 summaries.append(summary)
for k in [3,5]:
 for filt in ['none','EG_cd']:
  for first,second in [(f'P{k}_{filt}',f'L{k}_{filt}'),(f'P{k}_{filt}',f'R{k}_{filt}'),(f'L{k}_{filt}',f'R{k}_{filt}')]:
   diff=scene_totals(outcome_mean[first]-outcome_mean[second])/DEN
   for j,m in enumerate(METRICS):paired.append(dict(first=first,second=second,metric=m,**interval(diff[:,j])))
# Mechanism: keep I1-missing variants; collision knowledge should not be conditional on contact recovery.
non_nominal=DIST>0;valid=non_nominal&~A['empty']&np.isfinite(A['mu']);y=A['collision']&~A['empty']
auc_scene=[];auc_summary=[]
for name,score in [('logit_rank',RANK),('cyclic_distance',DIST)]:
 vals=[]
 for s in SCENES:
  mask=valid&(A['scene'][:,None]==s);v=float(roc_auc_score(y[mask],score[mask]));vals.append(v);auc_scene.append(dict(scene=int(s),predictor=name,n=int(mask.sum()),collisions=int(y[mask].sum()),AUROC=v))
 auc_summary.append(dict(predictor=name,pooled_AUROC=float(roc_auc_score(y[valid],score[valid])),scene_equal=interval(vals)))
delta=np.array([x['AUROC'] for x in auc_scene if x['predictor']=='logit_rank'])-np.array([x['AUROC'] for x in auc_scene if x['predictor']=='cyclic_distance'])
auc_difference=interval(delta)
tertiles=[];tertile_scenes=[];tertile_contrasts=[]
for dist in range(1,7):
 mask=valid&(DIST==dist);q=np.quantile(RANK[mask],[1/3,2/3]);binid=np.where(RANK<=q[0],0,np.where(RANK<=q[1],1,2));rates_by_bin=[]
 for b in range(3):
  bm=mask&(binid==b);vals=[]
  for s in SCENES:
   sm=bm&(A['scene'][:,None]==s);rate=float(y[sm].mean()) if sm.any() else np.nan;vals.append(rate)
   tertile_scenes.append(dict(distance=dist,tertile=b+1,scene=int(s),n=int(sm.sum()),collision=int(y[sm].sum()),collision_rate=rate))
  assert np.isfinite(vals).all(),'Empty scene tertile needs explicit missing-data handling'
  rates_by_bin.append(vals);tertiles.append(dict(distance=dist,degrees=dist*15,tertile=b+1,rank_cut1=float(q[0]),rank_cut2=float(q[1]),n=int(bm.sum()),collision=int(y[bm].sum()),pooled_collision_rate=float(y[bm].mean()),**interval(vals)))
 tertile_contrasts.append(dict(distance=dist,contrast='low_logit_minus_high_logit_collision_rate',**interval(np.array(rates_by_bin[2])-rates_by_bin[0])))
pr=next(x for x in summaries if x['group']=='P3_none');lo=next(x for x in summaries if x['group']=='L3_none');ci=next(x for x in paired if x['first']=='P3_none' and x['second']=='L3_none' and x['metric']=='collision')
ratio=pr['collision']/lo['collision'];recovery_ratio=pr['recovery']/lo['recovery'];conditions=dict(collision_ratio_at_most_0p7=ratio<=.7,collision_CI_upper_below_zero=ci['high']<0,recovery_ratio_at_least_0p9=recovery_ratio>=.9)
decision='PRIOR' if all(conditions.values()) else 'LOCALITY' if .7<=ratio<=1.3 else 'NO-PRIOR' if ratio>1.3 else 'INCONCLUSIVE'
result=dict(decision=decision,primary_conditions=conditions,primary_collision_ratio=ratio,primary_recovery_ratio=recovery_ratio,primary_collision_rate_difference=ci,summaries=summaries,paired=paired,
 mechanism=dict(AUROC=auc_summary,paired_AUROC_difference=auc_difference,non_nominal=int(non_nominal.sum()),evaluated=int(valid.sum()),empty_excluded=int((non_nominal&A['empty']).sum()),unknown_excluded=int((non_nominal&~np.isfinite(A['mu'])).sum()),tertile_contrasts=tertile_contrasts),
 assertions=dict(frames=48,candidates=N,variants=N*12,nominal_argmax_mismatch=len(MISMATCH),nominal_clean=True,original_feasible=int(GOOD.sum()),original_weak=int(WEAK.sum()),scalar_choose_agreement=True,input_SHA_verified=96),
 CPU_seconds=time.process_time(),wall_seconds=time.perf_counter()-START,training=False,inference=False,new_FC_evaluations=0,C4_C5=False)
writecsv('selected_candidates.csv',selected);writecsv('effective_set_sizes.csv',effective);writecsv('random_draw_counts.csv',random_draw_rows);writecsv('scene_statistics.csv',scene_rows);writecsv('paired_intervals.csv',paired);writecsv('AUROC_by_scene.csv',auc_scene);writecsv('distance_rank_tertiles.csv',tertiles);writecsv('distance_rank_tertiles_by_scene.csv',tertile_scenes);writecsv('distance_rank_contrasts.csv',tertile_contrasts)
np.savez_compressed(R/'selected_angles.npz',**choice_archive,scene=A['scene'],frame=A['frame'],candidate_id=A['candidate_id'],original_angle=NOM)
np.savez_compressed(R/'mechanism_inputs.npz',scene=A['scene'],frame=A['frame'],candidate_id=A['candidate_id'],original_angle=NOM,logit_rank=RANK,cyclic_distance=DIST,non_nominal_valid=valid,collision=A['collision'],empty=A['empty'],I1_defined=np.isfinite(A['I1']))
(R/'statistics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(decision=decision,conditions=conditions,collision_ratio=ratio,recovery_ratio=recovery_ratio,primary_ci=ci,groups=[{k:v for k,v in x.items() if k=='group' or k in METRICS} for x in summaries],mechanism=result['mechanism'],assertions=result['assertions'],wall_seconds=result['wall_seconds']),indent=2))
