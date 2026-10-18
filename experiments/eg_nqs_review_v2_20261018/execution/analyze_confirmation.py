"""Frozen stratified scene inference; seed variation is not extra scene replication."""
import csv
from common import *
out=R/'results';seeds=[20261018,20261019,20261020];strata=SPLIT['official_confirmation'];scenes=sum(strata.values(),[])
duplicates=json.loads((out/'confirmation_duplicate_audit.json').read_text());keep={tuple(v) for v in duplicates['canonical_observations'].values()}
metrics=['ap','ap04','ap08','fixed_pool_ap','n','top1_valid','top1_collision','top1_empty']
allscene=[];allsummaries=[];all_decisions={}
def writecsv(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for role in ['subset','official']:
    records=[]
    for s in scenes:
        folder=R/f'evaluation/{role}/scene_{s:04d}';assert json.loads((folder/'COMPLETE.json').read_text())['status']=='PASS'
        for f in SPLIT['confirmation_frames']:
            rows=json.loads((folder/f'{f:04d}.json').read_text());assert len(rows)==20;records.extend(rows)
    assert len(records)==61440
    # Full per-frame table remains a server artifact; publish scene aggregates.
    writecsv(out/f'{role}_confirmation_per_frame.csv',records)
    names=list(dict.fromkeys(x['configuration'] for x in records));role_out={}
    for sensitivity in [False,True]:
        lookup={};scene_rows=[];empty=[]
        for split,ss in strata.items():
            for s in ss:
                for name in names:
                    rows=[r for r in records if r['scene']==s and r['configuration']==name and (not sensitivity or (s,r['frame']) in keep)]
                    if not rows:
                        if s not in empty:empty.append(s)
                        continue
                    v=dict(role=role,sensitivity='deduplicated' if sensitivity else 'native256',split=split,scene=s,configuration=name,frames=len(rows),**{m:float(np.mean([x[m] for x in rows])) for m in metrics})
                    corrs=[x['spearman'] for x in rows if x['spearman'] is not None];v['spearman_mean']=float(np.mean(corrs)) if corrs else None;v['spearman_valid_frames']=len(corrs)
                    lookup[s,name]=v;scene_rows.append(v)
        allscene.extend(scene_rows)
        if empty:
            role_out['deduplicated']=dict(status='INCOMPLETE_STRATA_AFTER_DEDUPLICATION',empty_scenes=empty);continue
        for split,ss in list(strata.items())+[('overall',scenes)]:
            for name in names:allsummaries.append(dict(role=role,sensitivity='deduplicated' if sensitivity else 'native256',split=split,configuration=name,**{m:float(np.mean([lookup[s,name][m] for s in ss])) for m in metrics}))
        def effect(left,right,suffix='',metric='ap'):
            arr=np.array([[lookup[s,f'{left}_{seed_}{suffix}'][metric]-lookup[s,(f'{right}_{seed_}' if right!='R0' else 'R0')+suffix][metric] for s in scenes] for seed_ in seeds])
            mean_scene=arr.mean(0);mean_seed=arr.mean(1);rng=np.random.default_rng(20261018)
            resampled=np.concatenate([rng.integers(i*4,i*4+4,(10000,4)) for i in range(3)],axis=1);bs=mean_scene[resampled].mean(1)
            ci95=np.quantile(bs,[.025,.975]);ci90=np.quantile(bs,[.05,.95]);mean=float(mean_scene.mean())
            return dict(mean_pp=mean,ci95_pp=ci95.tolist(),ci90_pp=ci90.tolist(),scene_mean_pp={str(s):float(v) for s,v in zip(scenes,mean_scene)},seed_mean_pp=mean_seed.tolist(),seed_mean_sd_pp=float(mean_seed.std(ddof=1)),split_mean_pp={k:float(np.mean([mean_scene[scenes.index(s)] for s in ss])) for k,ss in strata.items()},positive_scenes=int((mean_scene>0).sum()),positive_seeds=int((mean_seed>0).sum()),confirmation_success=bool(mean>=1 and ci95[0]>0 and (mean_seed>0).all() and (mean_scene>0).sum()>=8),equivalence_within_half_pp=bool(ci90[0]>-.5 and ci90[1]<.5))
        effects={}
        for suffix in ['', '_CD']:
            for l,r in [('A','B6'),('A','E'),('A','R0'),('B6','R0'),('E','R0')]:effects[l+'-'+r+suffix]=effect(l,r,suffix)
        role_out['deduplicated' if sensitivity else 'native256']=dict(status='COMPLETE',effects=effects,fixed_pool_A_minus_B6=effect('A','B6',metric='fixed_pool_ap'),primary_supported=effects['A-B6']['confirmation_success'],practical_supported=effects['A-R0_CD']['confirmation_success'])
    all_decisions[role]=role_out
writecsv(out/'confirmation_per_scene.csv',allscene);writecsv(out/'confirmation_configuration_summary.csv',allsummaries)
write_json(out/'confirmation_decision.json',dict(status='COMPLETE',backbones=all_decisions,protocol='12 selected official test scenes per backbone; full 256-frame primary; not all 90 scenes',bootstrap=dict(unit='scene',stratified=True,strata=strata,resamples=10000,seed=20261018),scenes=scenes,frames_per_backbone=3072,configurations_per_backbone=20,probe_seeds=seeds))
print(json.dumps({role:{'primary':x['native256']['effects']['A-B6'],'practical':x['native256']['effects']['A-R0_CD']} for role,x in all_decisions.items()},indent=2))
