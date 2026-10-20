import os,sys,json,time,hashlib
from pathlib import Path
R=Path(os.environ['R']);sys.path.insert(0,str(R/'code'))
from compact import *

D=R/'diagnosis_s_vs_g';L=G/'eg_lg1_local_geometry_v1_20261020';t=time.time();base=dict(np.load(RP/'prepared.npz'));mask=np.load(R/'cache/mask.npy',mmap_mode='r');tokens=np.load(R/'cache/tokens.npy',mmap_mode='r');counts=[];valid=[];spanratio=[];checks=[]
for i,(s,f) in enumerate(zip(base['scene'],base['frame'])):
 with np.load(L/f'geometry/head/{s:04d}/{f:04d}.npz') as z:raw=z['raw']
 true=np.rint(np.expm1(raw[...,:5].astype(np.float64))).astype(np.int32);observed=np.stack([mask[i,...,sum(QUOTAS[:k]):sum(QUOTAS[:k+1])].sum(-1) for k in range(5)],-1);assert np.array_equal(observed,np.minimum(true,np.array(QUOTAS)))
 with np.load(P/f'B/exports/subset/scene_{s:04d}/{f:04d}.npz') as z:pred=z['pred']
 w=np.clip(pred[base['ids'][i],1],0,.1)[:,None];yy=tokens[i,...,80:128,1]*(w[...,None]/2+.01);vm=mask[i,...,80:128];sample_span=np.where(vm,yy,-np.inf).max(-1)-np.where(vm,yy,np.inf).min(-1);truth=raw[...,21];ok=np.isfinite(truth)&(truth>1e-6)&(vm.sum(-1)>=2);ratio=np.full(truth.shape,np.nan);ratio[ok]=sample_span[ok]/truth[ok]
 counts.append(true);valid.append(observed);spanratio.append(ratio);checks.append(dict(scene=int(s),frame=int(f),count_rule_exact=True))
counts=np.stack(counts);valid=np.stack(valid);spanratio=np.stack(spanratio);rows=[]
def summary(c,v,span):
 n=v.sum(-1);total=c.sum(-1);regions=[]
 for k,name in enumerate(['left','right','palm','approach','inner']):
  a=c[...,k];b=v[...,k];regions.append(dict(region=name,quota=QUOTAS[k],mean_true=float(a.mean()),median_true=float(np.median(a)),p90_true=float(np.quantile(a,.9)),mean_sampled=float(b.mean()),empty_fraction=float((a==0).mean()),truncated_fraction=float((a>QUOTAS[k]).mean()),quota_filled_fraction=float((a>=QUOTAS[k]).mean()),pooled_retained_points_fraction=float(b.sum()/max(1,a.sum()))))
 return dict(poses=int(n.size),effective_points_mean=float(n.mean()),effective_points_median=float(np.median(n)),effective_points_p10=float(np.quantile(n,.1)),effective_points_p90=float(np.quantile(n,.9)),padding_fraction=float(1-n.mean()/128),fraction_below64=float((n<64).mean()),all_empty_fraction=float((n==0).mean()),pooled_retained_points_fraction=float(n.sum()/max(1,total.sum())),mean_true_union_points=float(total.mean()),fraction_any_region_truncated=float((c>np.array(QUOTAS)).any(-1).mean()),sample_span_ratio_median=float(np.nanmedian(span)),sample_span_under90pct_fraction=float(np.nanmean(np.where(np.isfinite(span),span<.9,np.nan))),span_eligible=int(np.isfinite(span).sum()),regions=regions)
for role in ['fit','model_val','gate_cal']:
 ix=base['role']==role
 for poses in ['all6','original']:
  c=counts[ix] if poses=='all6' else counts[ix,:,0];v=valid[ix] if poses=='all6' else valid[ix,:,0];sp=spanratio[ix] if poses=='all6' else spanratio[ix,:,0];rows.append(dict(role=role,pose_mode=poses,**summary(c,v,sp)))
np.savez_compressed(D/'results/retention_by_pose.npz',true_counts=counts,sampled_counts=valid,sampled_span_ratio=spanratio)
write(D/'results/sampling_information.json',dict(status='PASS',all196608_count_rules_exact=True,summary=rows,seconds=time.time()-t,source='cached visible-cloud G geometry + actual frozen S input tokens/masks; no new point sampling or quality labels',C4_read=False,training=0))
print('SAMPLING_DIAG_COMPLETE',time.time()-t,flush=True)
