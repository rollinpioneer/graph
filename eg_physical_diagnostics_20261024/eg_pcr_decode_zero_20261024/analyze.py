import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
CACHE=Path('C:/Users/jackx/Documents/Codex/2026-10-21/ssh-aita-xushijie-172-19-164/outputs/EG_pose_quality_20261022/data/cache')
files=[json.loads(p.read_text()) for p in sorted(ROOT.glob('grad_*.json'))]
a=pd.DataFrame([r for f in files for r in f['rows']])
d=pd.DataFrame([r for f in files for r in f['directions']])
for s in a.scene.unique():
    cache=np.load(CACHE/f'C3_{s:04d}_0000.npz');mask=a.scene==s
    a.loc[mask,'nominal_depth']=cache['pred'][a.loc[mask,'candidate_id'].to_numpy(),3]
    a.loc[mask,'nominal_angle_class']=cache['angle_index'][a.loc[mask,'candidate_id'].to_numpy()]
d=d.merge(a[['scene','candidate_id','outside','nominal_depth']],on=['scene','candidate_id'],validate='many_to_one')
d['active']=d.gradient.abs()>1e-5
d['in_depth_range']=(d.coordinate!='depth')|((d.nominal_depth-np.sign(d.gradient)*d.step>=.01-1e-8)&(d.nominal_depth-np.sign(d.gradient)*d.step<=.04+1e-8))
for mode in ['frozen','recomputed']:
    d[mode+'_decrease']=d[mode+'_descent_change'] < -1e-7
    d[mode+'_increase']=d[mode+'_descent_change'] > 1e-7

def scene_ci(frame,column):
    v=frame.groupby('scene')[column].mean();rng=np.random.default_rng(20261024)
    strata=[v.loc[[s for s in v.index if (100<=s<=129 if i==0 else 130<=s<=159 if i==1 else 160<=s<=189)]].to_numpy() for i in range(3)]
    sampled=np.concatenate([rng.choice(x,size=(10000,len(x)),replace=True) for x in strata if len(x)],axis=1).mean(1)
    return dict(mean=float(v.mean()),lo=float(np.quantile(sampled,.025)),hi=float(np.quantile(sampled,.975)),scenes=len(v))

fwd=json.loads((ROOT/'forward.json').read_text());forward=pd.DataFrame(fwd['rows'])
out=dict(n=len(a),scenes=int(a.scene.nunique()),objects=int(a.object_id.nunique()),feasible=int((a.mu<=.4).sum()),weak=int((a.mu>.4).sum()),outside=int(a.outside.sum()),
    PCR_replay_max=float(abs(a.PCR-a.previous_PCR).max()),parameter_updates=0,training=False,
    forward=dict(n_candidates=49152,frames=48,adaptation_bitwise_all=bool(forward.adapted_equal.all()),
       grad_enabled_bitwise_all=bool(forward.grad_enabled_equal.all()),gpu_history_bitwise_all=bool(forward[forward.device=='cuda'].cached_equal.all()),
       cpu_history_max=float(forward[forward.device=='cpu'].cached_max_error.max()),synthetic=fwd['synthetic']),
    gradient={},directions=[],temperature={},decision='HOLD for joint angle-depth PCR; angle-path technical feasibility supported')
for term in ['RA','RC','RS','PCR']:
    out['gradient'][term]={}
    for head in ['angle_logits','depth_logits']:
        c=term+'_'+head;a[c+'_active']=a[c].abs()>1e-7
        out['gradient'][term][head]=dict(active=int(a[c+'_active'].sum()),total=len(a),scene_ci=scene_ci(a,c+'_active'),median=float(a[c].abs().median()),max=float(a[c].abs().max()))
for temp in [.5,2.]:
    out['temperature'][str(temp)]={head:int((a[f'PCR_{head}_T{temp}'].abs()>1e-7).sum()) for head in ['angle_logits','depth_logits']}
for coord in ['angle','depth']:
    for step in sorted(d[d.coordinate==coord].step.unique()):
        for filt in ['all','inside_sdf','inside_sdf_and_depth_range']:
            z=d[(d.coordinate==coord)&(d.step==step)&(d.term=='PCR')&d.active]
            if filt!='all':z=z[~z.outside]
            if filt=='inside_sdf_and_depth_range':z=z[z.in_depth_range]
            out['directions'].append(dict(coordinate=coord,step=float(step),filter=filt,n=len(z),
                frozen_decrease=int(z.frozen_decrease.sum()),recomputed_decrease=int(z.recomputed_decrease.sum()),
                recomputed_increase=int(z.recomputed_increase.sum()),scene_ci=scene_ci(z,'recomputed_decrease'),
                improved_scenes=int((z.groupby('scene').recomputed_decrease.mean()>.5).sum()),
                median_change=float(z.recomputed_descent_change.median())))
out['weak_depth_active']=int(((a.mu>.4)&a.PCR_depth_logits_active).sum())
out['weak_angle_active']=int(((a.mu>.4)&a.PCR_angle_logits_active).sum())
out['input_hashes_unchanged']=all(next(r['cache_sha256'] for r in fwd['rows'] if r['scene']==int(x['rows'][0]['scene']) and r['frame']==0)==x['cache_sha256'] for x in files)
a.to_csv(ROOT/'candidate_gradients.csv',index=False)
d.to_csv(ROOT/'direction_checks.csv',index=False)
forward.to_csv(ROOT/'forward_invariants.csv',index=False)
scene=a.groupby('scene').agg(n=('candidate_id','size'),angle_active=('PCR_angle_logits_active','sum'),depth_active=('PCR_depth_logits_active','sum'),out_of_sdf=('outside','sum'))
for coord in ['angle','depth']:
    step=d[d.coordinate==coord].step.min();z=d[(d.coordinate==coord)&(d.step==step)&(d.term=='PCR')&d.active]
    scene[coord+'_probe_n']=z.groupby('scene').size();scene[coord+'_PCR_decrease_fraction']=z.groupby('scene').recomputed_decrease.mean()
scene.to_csv(ROOT/'scene_summary.csv')
(ROOT/'statistics.json').write_text(json.dumps(out,indent=2))
fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
terms=['RA','RC','RS','PCR'];x=np.arange(4)
for offset,head,color in [(-.18,'angle_logits','#246b9e'),(.18,'depth_logits','#dd8b32')]:
    axes[0].bar(x+offset,[out['gradient'][t][head]['active']/len(a) for t in terms],.36,label=head.split('_')[0],color=color)
axes[0].set(xticks=x,xticklabels=terms,ylim=(0,1.1),ylabel='Fraction with nonzero gradient',title='313 existing candidates / 12 scenes');axes[0].legend()
select=[v for v in out['directions'] if v['filter']=='all'];vals=[v['scene_ci']['mean'] for v in select]
axes[1].bar(range(4),vals,color=['#246b9e','#246b9e','#dd8b32','#dd8b32'])
axes[1].errorbar(range(4),vals,yerr=[[v['scene_ci']['mean']-v['scene_ci']['lo'] for v in select],[v['scene_ci']['hi']-v['scene_ci']['mean'] for v in select]],fmt='none',color='black',capsize=3)
axes[1].axhline(.5,linestyle='--',color='gray');axes[1].set(xticks=range(4),xticklabels=['0.1 deg','0.5 deg','0.05 mm','0.2 mm'],ylim=(0,1),ylabel='Scene mean fraction reducing PCR',title='Recompute SDF normals at probe pose')
axes[2].scatter(scene.angle_PCR_decrease_fraction,scene.depth_PCR_decrease_fraction,color='#326b4b')
for s,r in scene.iterrows():axes[2].annotate(str(s),(r.angle_PCR_decrease_fraction,r.depth_PCR_decrease_fraction),fontsize=7,xytext=(2,2),textcoords='offset points')
axes[2].axhline(.5,color='gray',linestyle='--');axes[2].set(xlim=(0,1.05),ylim=(0,1.05),xlabel='Angle: PCR decreases at 0.1 deg',ylabel='Depth: PCR decreases at 0.05 mm',title='Each point is one scene')
fig.savefig(ROOT/'zero_update_summary.png',dpi=180);plt.close(fig)
print(json.dumps(out,indent=2))
