"""Frozen rule comparison; no fitting. Scene-cluster bootstrap and exact random controls."""
from core import *
import csv,math
cohort=json.loads((R/'results/cohort.json').read_text());assert json.loads((R/'results/replay_receipt.json').read_text())['status']=='PASS'
for s in SCENES:assert json.loads((R/f'results/quality_{s:04d}.json').read_text())['status']=='PASS'
names=['EG','top3x2','top2x3','joint6','random6','keep1_random5','joint12','random12','keep1_random11','all48'];det=['EG','top3x2','top2x3','joint6','joint12','all48'];records=[]
def random_hit(n_good,n,k):return 1.-(math.comb(n-n_good,k)/math.comb(n,k) if n-n_good>=k else 0.)
for row in cohort:
 s,f=row['scene'],row['frame'];z=np.load(R/f'replay/{s:04d}_{f:04d}.npz');qr=np.load(R/f'quality/{s:04d}_{f:04d}.npz')['q'].reshape(12,4)
 al=z['angle_logits'];dl=z['depth_logits'];a0=int(z['angle_index']);d0=int(z['depth_index']);orig=a0*4+d0
 q=np.roll(qr,a0,axis=0).ravel();good=(q>0)&(q<=.4000001);assert bool(good[orig])==row['success'] and q[orig]==row['original_quality']
 ar=np.argsort(-al,kind='stable');dr=np.argsort(-dl,kind='stable');arank=np.argsort(ar)+1;drank=np.argsort(dr)+1
 joint=(al[:,None].astype(float)+dl[None,:]).ravel();order=np.argsort(-joint,kind='stable');assert order[0]==orig
 sets={'EG':np.array([orig]),'top3x2':np.array([a*4+d for a in ar[:3] for d in dr[:2]]),'top2x3':np.array([a*4+d for a in ar[:2] for d in dr[:3]]),'joint6':order[:6],'joint12':order[:12],'all48':np.arange(48)}
 assert all(orig in ids for ids in sets.values())
 hit={k:float(good[v].any()) for k,v in sets.items()};risk={k:float(1-good[v].mean()) for k,v in sets.items()};forced={k:float(not good[next(i for i in order if i in ids and i!=orig)]) for k,ids in sets.items() if len(ids)>1}
 ng=int(good.sum());g0=int(good[orig])
 for k in [6,12]:
  hit[f'random{k}']=random_hit(ng,48,k);hit['keep1_random'+str(k-1)]=1. if g0 else random_hit(ng,47,k-1)
  risk[f'random{k}']=1-ng/48;risk['keep1_random'+str(k-1)]=((1-g0)+(k-1)*(47-ng+g0)/47)/k
 feasible=np.flatnonzero(good);best=feasible[q[feasible]==q[feasible].min()] if len(feasible) else np.array([],int)
 canonical=next((int(i) for i in order if i in best),None)
 angle_only=bool(good.reshape(12,4)[:,d0].any());depth_only=bool(good.reshape(12,4)[a0,:].any())
 category='original_success' if g0 else ('both_single' if angle_only and depth_only else 'angle_only_suffices' if angle_only else 'depth_only_suffices' if depth_only else 'joint_change_required' if ng else 'unrecoverable48')
 rec=dict(**row,original_angle=a0,original_depth=d0,good_cells=ng,category=category,hits=hit,uniform_within_set_failure_risk=risk,forced_top_alternative_failure=forced,best_quality=float(q[best[0]]) if len(best) else None,best_rank_pairs=[[int(arank[i//4]),int(drank[i%4])] for i in best],canonical_best_rank=[int(arank[canonical//4]),int(drank[canonical%4])] if canonical is not None else None,feasible_rank_pairs=[[int(arank[i//4]),int(drank[i%4])] for i in feasible])
 records.append(rec)
fail=[x for x in records if not x['success']];success=[x for x in records if x['success']];assert len(fail)==187 and sum(x['good_cells']>0 for x in fail)==170 and len(success)==453
rng=np.random.default_rng(20261018);draw=rng.integers(0,len(SCENES),size=(20000,len(SCENES)));counts=np.array([sum(x['scene']==s for x in fail) for s in SCENES]);den=counts[draw].sum(1)
def bootstrap(values):
 sums=np.array([sum(v for x,v in zip(fail,values) if x['scene']==s) for s in SCENES]);return sums[draw].sum(1)/den
summaries=[]
for name in names:
 vals=np.array([x['hits'][name] for x in fail]);boot=bootstrap(vals)
 summaries.append(dict(strategy=name,recovered_or_expected=float(vals.sum()),failure_coverage=float(vals.mean()),recoverable_coverage=float(vals.sum()/170),failure_coverage_ci95=np.quantile(boot,[.025,.975]).tolist(),success_oracle_retention=float(np.mean([x['hits'][name] for x in success])),success_uniform_choice_failure=float(np.mean([x['uniform_within_set_failure_risk'][name] for x in success])),success_forced_top_alternative_failure=float(np.mean([x['forced_top_alternative_failure'][name] for x in success])) if name in ['top3x2','top2x3','joint6','joint12','all48'] else None))
tests=[]
for name,control in [('top3x2','keep1_random5'),('top2x3','keep1_random5'),('joint6','keep1_random5'),('joint12','keep1_random11')]:
 diff=np.array([x['hits'][name]-x['hits'][control] for x in fail]);boot=bootstrap(diff);lower=float(np.quantile(boot,.0125));tests.append(dict(strategy=name,control=control,difference_pp=float(diff.mean()*100),ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),family_corrected_lower_pp=lower*100,reliable_advantage=bool(lower>0)))
cats={k:sum(x['category']==k for x in fail) for k in ['angle_only_suffices','depth_only_suffices','both_single','joint_change_required','unrecoverable48']}
rankhist=np.zeros((12,4),int);canonical_hist=rankhist.copy()
for x in fail:
 if x['canonical_best_rank']:
  a,d=x['canonical_best_rank'];canonical_hist[a-1,d-1]+=1
 for a,d in set(map(tuple,x['best_rank_pairs'])):rankhist[a-1,d-1]+=1
write(R/'results/per_case.json',records)
with open(R/'results/per_case.csv','w',newline='') as f:
 fields=['scene','frame','success','original_quality','good_cells','category']+names;w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in records:w.writerow({**{k:x[k] for k in fields[:6]},**x['hits']})
scene_rows=[]
for s in SCENES:
 a=[x for x in fail if x['scene']==s]
 for name in names:scene_rows.append(dict(scene=s,strategy=name,failures=len(a),recoverable=sum(x['good_cells']>0 for x in a),recovered_or_expected=sum(x['hits'][name] for x in a)))
with open(R/'results/per_scene.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(scene_rows[0]));w.writeheader();w.writerows(scene_rows)
write(R/'results/summary.json',dict(status='COMPLETE',failures=187,recoverable=170,successes=453,strategies=summaries,paired_tests=tests,categories=cats,canonical_best_rank_histogram=canonical_hist.tolist(),all_best_rank_case_counts=rankhist.tolist(),reliable_strategies=[x['strategy'] for x in tests if x['reliable_advantage']],decision='RANKING_INFORMATION_SUPPORTED_ONLY' if any(x['reliable_advantage'] for x in tests) else 'PAUSE_RANK_DEPENDENT_TRAINING',bootstrap=dict(unit='scene',replicates=20000,seed=20261018,confidence='case weighted paired cluster bootstrap; four-test Bonferroni one-sided lower'),new_training=0))
print(json.dumps(dict(strategies=summaries,tests=tests,categories=cats),indent=2))
