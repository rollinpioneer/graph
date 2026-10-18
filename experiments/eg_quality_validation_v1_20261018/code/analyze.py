import json,csv,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
N=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
sys.path.insert(0,'/home/xushijie3/work/eg_sub25_v1_20261015/d2d4_20261017')
from d2_lookup import SceneLabels
labels={}
groups=defaultdict(list); aps=defaultdict(list); allrows=[]; completes=[]
for p in sorted((N/'results').glob('*/COMPLETE.json')):
    comp=json.loads(p.read_text()); completes.append(comp)
    rows=json.loads((p.parent/'records.json').read_text())
    if comp['scene']<100 and comp['scene'] not in labels:labels[comp['scene']]=SceneLabels(comp['scene'])
    for r in rows:
        r['label_best_bucket_representative']=r['label_best']
        if r['scene']<100:
            ob=labels[r['scene']].obj[r['object_id']+1];pt=r['label_point'];v=r['view']
            mu=np.asarray(ob['scores'][pt,v]).ravel();col=np.asarray(ob['collision'][pt,v]).ravel();w=np.asarray(ob['offsets'][pt,v])[...,2].ravel()
            valid=(mu>0)&~col&(w<=.1)
            r['label_best']=float(mu[valid].min()) if valid.any() else -1.
            r['label_collision_source']='released_scene_cache'
        else:
            r['label_best']=r['label_best_exact']
            assert r['label_collision_source']=='native_geometry_requested_cells'
    allrows+=rows
    groups[comp['role']+'/'+comp['partition']]+=rows
    aps[comp['role']+'/'+comp['partition']]+=json.loads((p.parent/'ap.json').read_text())
assert len(completes)==18 and all(x['frames']==64 for x in completes)
FR=np.array([.2,.4,.6,.8,1.,1.2])
def ap_values(q):
    q=np.asarray(q);z=np.zeros((50,6))
    for j,f in enumerate(FR):
        c=np.cumsum((q>0)&(q<=f+1e-7))
        if len(c):z[:,j]=c[np.minimum(np.arange(50),len(c)-1)]/np.arange(1,51)
    return dict(ap=float(z.mean()*100),ap04=float(z[:,1].mean()*100),ap08=float(z[:,3].mean()*100))
pool_frame_count=sum(len(v) for v in aps.values())//6
for group,rows in groups.items():
    aps[group]=[r for r in aps[group] if r['variant'] not in ['label_best','label_actual']]
    frames=defaultdict(list)
    for r in rows:
        if r['frame'] in [0,64,128,192]:frames[(r['scene'],r['frame'])].append(r)
    for (s,f),rs in frames.items():
        rs.sort(key=lambda r:r['rank']);q=np.array([r['actual'] for r in rs])
        ix=np.array([i for i,r in enumerate(rs) if r['distance']<=.005],int)
        for name in ['label_best','label_actual']:
            mu=np.array([r[name] for r in rs]);score=np.where(mu>0,1.1-mu,-1.)
            order=np.lexsort((np.arange(len(rs)),-score))
            aps[group].append(dict(scene=s,role=rs[0]['role'],partition=rs[0]['partition'],frame=f,variant=name,n=len(rs),**ap_values(q[order])))
        for name in ['label_best','label_actual','actual']:
            mu=np.array([rs[i][name] for i in ix]);score=np.where(mu>0,1.1-mu,-1.)
            order=np.arange(len(rs));order[ix]=ix[np.lexsort((ix,-score))]
            aps[group].append(dict(scene=s,role=rs[0]['role'],partition=rs[0]['partition'],frame=f,variant=name+'_matched_slots',n=len(rs),**ap_values(q[order])))
def ok(q):return 0<q<=.4000001
def mean(x):return float(np.mean(x)) if len(x) else None
def obs(r):return min(r['frame'],109) if r['scene']==188 else min(r['frame'],170) if r['scene']==189 else r['frame']
def unique_mean(rows,key):
    d=defaultdict(list)
    for r in rows:d[(r['scene'],obs(r))].append(r[key])
    sc=defaultdict(list)
    for (s,o),v in d.items():sc[s].append(mean(v))
    return mean([mean(v) for v in sc.values()])
def fidelity(rows,target):
    tp=sum(ok(r['label_actual']) and ok(r[target]) for r in rows)
    fp=sum(ok(r['label_actual']) and not ok(r[target]) for r in rows)
    fn=sum(not ok(r['label_actual']) and ok(r[target]) for r in rows)
    tn=len(rows)-tp-fp-fn
    return dict(n=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,agreement=(tp+tn)/len(rows) if rows else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
summary={}; scene_ap=[]; rng=np.random.default_rng(20261018)
for key,rr in groups.items():
    pool=[r for r in rr if r['frame'] in [0,64,128,192]]
    top=[r for r in rr if r['rank']==0]
    fail=[r for r in top if not ok(r['actual'])]; ranking=[r for r in fail if r['later_valid']]
    cat=defaultdict(int)
    for r in fail:
        w=ok(r['width_only']);p=ok(r['pose_only']);j=ok(r['best_grid'])
        k='both_single_controls' if w and p else 'width_only_control' if w else 'pose_only_control' if p else 'joint_only_control' if j else 'not_repaired_in_grid'
        cat[k]+=1
    matched=[r for r in pool if r['distance']<=.005]
    quality=dict(pool_candidates=len(pool),matched_5mm=len(matched),match_fraction=len(matched)/len(pool),owner_disagreement=sum(r['owner_diff'] for r in pool),median_abs_width_delta_mm=float(np.median([abs(r['width_delta'])*1000 for r in matched])),median_rotation_error_deg=float(np.median([r['angle_error_deg'] for r in matched])),median_view_error_deg=float(np.median([r['view_error_deg'] for r in matched])),distance_sensitivity={})
    for mm in [1,2,5,10,1000]:
        sub=[r for r in pool if r['distance']<=mm/1000]
        quality['distance_sensitivity'][str(mm)]={v:fidelity(sub,v) for v in ['actual','width_only','exact_cell']}
    quality['matched_label_best_pass_actual_fail']=sum(ok(r['label_best']) and not ok(r['label_actual']) for r in matched)
    quality['matched_label_best_pass_geometry_fail']=sum(ok(r['label_best']) and not ok(r['actual']) for r in matched)
    quality['matched_geometric_potential_pass_actual_fail']=sum(ok(r['best']) and not ok(r['actual']) for r in matched)
    associations={}
    for target in ['actual','best','label_actual','label_best']:
        values=[];constant=0
        for scene,frame in sorted(set((r['scene'],r['frame']) for r in pool)):
            sub=[r for r in pool if r['scene']==scene and r['frame']==frame]
            score=np.array([r['score'] for r in sub]);mu=np.array([r[target] for r in sub]);q=np.where(mu>0,1.1-mu,-1.)
            if len(np.unique(q))<2 or len(np.unique(score))<2:constant+=1;continue
            values.append(float(spearmanr(score,q)[0]))
        associations[target]=dict(mean_frame_spearman=mean(values),nonconstant_frames=len(values),constant_frames=constant)
    quality['score_associations_frozen_pool']=associations
    st=dict(frames=len(top),top1_fail=len(fail),ranking_fail=len(ranking),top1_collision=sum(r['collision'] for r in fail),top1_empty=sum(r['empty'] for r in fail),top1_fail_matched5mm=sum(r['distance']<=.005 for r in fail),categories=dict(cat),label_optimistic_fail=sum(ok(r['label_best']) for r in fail),geometry_potential_fail=sum(ok(r['best_grid']) for r in fail),label_optimistic_and_geometry_verified=sum(ok(r['label_best']) and ok(r['best_grid']) for r in fail),label_optimistic_unverified=sum(ok(r['label_best']) and not ok(r['best_grid']) for r in fail))
    st['unique_observation_failure_rate']=unique_mean([dict(r,failure=float(not ok(r['actual']))) for r in top],'failure')
    st['matched5mm_label_optimistic_fail']=sum(r['distance']<=.005 and ok(r['label_best']) for r in fail)
    st['matched5mm_label_optimistic_verified']=sum(r['distance']<=.005 and ok(r['label_best']) and ok(r['best_grid']) for r in fail)
    st['ranking_fail_with_pose_control']=sum(ok(r['pose_only']) for r in ranking)
    st['ranking_fail_with_width_control']=sum(ok(r['width_only']) for r in ranking)
    st['ranking_fail_with_joint_control']=sum(ok(r['best_grid']) for r in ranking)
    out={}; per_scene=defaultdict(dict)
    for variant in sorted(set(r['variant'] for r in aps[key])):
        aa=[r for r in aps[key] if r['variant']==variant]
        out[variant]={m:mean([r[m] for r in aa]) for m in ['ap','ap04','ap08']}
        out[variant]['unique_observation_ap']=unique_mean(aa,'ap')
        for s in sorted(set(r['scene'] for r in aa)):
            per_scene[s][variant]=mean([r['ap'] for r in aa if r['scene']==s])
    for s,d in per_scene.items():scene_ap.append(dict(group=key,scene=s,**d))
    delta={name:out[name]['ap']-out['original']['ap'] for name in out}
    pairs={
      'actual_minus_best':out['actual']['ap']-out['best']['ap'],
      'best_ties_actual_minus_best':out['best_ties_actual']['ap']-out['best']['ap'],
      'label_actual_minus_label_best':out['label_actual']['ap']-out['label_best']['ap']}
    if key=='subset/dev':
        ds=np.array([d['actual']-d['best'] for d in per_scene.values()]); boot=ds[rng.integers(0,len(ds),(10000,len(ds)))].mean(1)
        pairs['scene_bootstrap_ci95_actual_minus_best']=np.quantile(boot,[.025,.975]).tolist()
    summary[key]=dict(quality=quality,top1=st,oracle=out,delta_from_original=delta,target_gap=pairs)
out=dict(status='COMPLETE',groups=summary,total_verified=sum(x['verified'] for x in completes),total_geometry_calls=sum(x['geometry_calls'] for x in completes),worker_wall_sum=sum(x['elapsed'] for x in completes),max_worker_rss_gib=max(x['maxrss_kib'] for x in completes)/2**20,pool_model_frames=pool_frame_count,top1_model_frames=sum(x['frames'] for x in completes),limits=['Sampled frame diagnostics: four frames per scene/model for pool oracles; 64 for Top-1. Not standard full-scene AP.','Continuous actual quality uses the official full-geometry/SDF evaluator, not a robot success measurement.','Best is a bounded grid at fixed center and approach; it is not a continuous global optimum.','Native NMS and candidate selection are frozen. No new grasps are inserted for reranking.','Oracle target gap is conditional information loss, not causal attribution or a guaranteed learnable improvement.','Two Novel scenes contain repeated observations; unique-observation sensitivity is included.','Unrestricted label rankings are nearest-cell proxies, not true oracles; matched_slots permutations preserve all unmatched candidate ranks.','Prior D5/D6 assets discovered but no new repair model was developed.'])
(N/'summary.json').write_text(json.dumps(out,indent=2))
out['additional_label_pose_collision_checks']=sum(x['verified']*48 for x in completes if x['scene']>=100)
(N/'summary.json').write_text(json.dumps(out,indent=2))
with (N/'per_scene_oracle.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(scene_ap[0]));w.writeheader();w.writerows(scene_ap)
keys=sorted(set().union(*(r.keys() for r in allrows)))
with (N/'candidate_diagnostics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
examples={}
for r in sorted(allrows,key=lambda r:(r['scene'],r['frame'],r['rank'])):
    if r['partition']!='dev' or r['rank']!=0 or ok(r['actual']):continue
    w=ok(r['width_only']);p=ok(r['pose_only']);j=ok(r['best_grid'])
    cat='both_single_controls' if w and p else 'width_only_control' if w else 'pose_only_control' if p else 'joint_only_control' if j else 'not_repaired_in_grid'
    if cat not in examples:examples[cat]=dict(category=cat,scene=r['scene'],frame=r['frame'],actual=r['actual'],width_only=r['width_only'],pose_only=r['pose_only'],joint=r['best_grid'],label_best=r['label_best'],label_actual=r['label_actual'],distance_mm=r['distance']*1000,width_delta_mm=r['width_delta']*1000,rotation_error_deg=r['angle_error_deg'],later_valid=r['later_valid'])
(N/'exploratory_cases.json').write_text(json.dumps(dict(selection='Post-hoc, first scene/frame per repair category, development scenes only. Exploratory examples, not causal evidence.',cases=list(examples.values())),indent=2))
for k,v in summary.items():
    print(k,'AP',{x:round(y['ap'],3) for x,y in v['oracle'].items()})
    print('TOP1',v['top1']);print('FIDELITY',v['quality']['match_fraction'],{x:y['agreement'] for x,y in v['quality']['distance_sensitivity']['5'].items()})
print('TOTAL',out['total_verified'],out['total_geometry_calls'])
