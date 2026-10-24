"""Summarize deterministic implementation checks, not collision performance."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
files=sorted((R/'preflight').glob('*.json'))
frames=[json.loads(p.read_text()) for p in files]
summary=[];per=[]
fig,axes=plt.subplots(2,2,figsize=(10,8))
for i,a in enumerate(frames):
    ms=[np.array(x) for x in a['confusion_nv_rows_reference_columns']]
    rec={k:a[k] for k in ['scene','frame','poses','candidates','points','integration_seconds','query_seconds','max_same_voxel_tsdf_error']}
    for j,(name,m) in enumerate(zip(['continuous','voxel_center'],ms)):
        rec[name+'_agreement']=float(np.trace(m)/m.sum())
        rec[name+'_mismatch_points']=int(m.sum()-np.trace(m))
        rec[name+'_occupied_reference_agreement']=float(m[1,1]/m[:,1].sum())
        rec[name+'_reference_occupied_points']=int(m[:,1].sum())
        ax=axes[i,j];pct=100*m/m.sum(axis=0,keepdims=True)
        ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
        for y in range(3):
            for x in range(3):ax.text(x,y,f'{m[y,x]:,}\n({pct[y,x]:.2f}%)',ha='center',va='center',fontsize=8,color='white' if pct[y,x]>55 else 'black')
        ax.set_xticks(range(3),['Free','Occupied','Unknown']);ax.set_yticks(range(3),['Free','Occupied','Unknown'])
        ax.set_xlabel('Reference state');ax.set_ylabel('nvblox state')
        ax.set_title(f'Scene {a["scene"]:04d} frame 0000 | {name}\nagreement = {rec[name+"_agreement"]*100:.5f}%')
    summary.append(rec)
    for r in a['per_pose']:
        per.append(dict(scene=a['scene'],frame=a['frame'],candidate_id=r['candidate_id'],angle=r['angle'],points=r['points'],
          nv_free=r['nv_counts'][0],nv_occupied=r['nv_counts'][1],nv_unknown=r['nv_counts'][2],
          agreement_continuous=r['agreement_False'],agreement_voxel_center=r['agreement_True']))
fig.suptitle('N0 preflight: state agreement (not GT collision recall)',fontsize=13)
fig.tight_layout();fig.savefig(R/'preflight_confusion.png',dpi=170);plt.close(fig)
for name,data in [('frame_statistics.csv',summary),('per_pose_preflight.csv',per)]:
    with (R/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
decision=dict(status='HOLD',reason='Original continuous-projection 99% target not met on either training frame; user explicitly requested retaining that gate and archiving evidence.',
  user_decision_date='2026-10-24',completed='Environment, source provenance, synthetic plane, two training frames',
  not_executed=['48-frame calibration','OP1/OP2 threshold freezing','C3 formal comparison','dense GT collision audit','P2 panel','I1 joint angle-selection groups'],
  changed_thresholds=False,training=False,parameter_updates=0,C3_formal_runs=0,C4_C5_accessed=False,
  scene_count=2,base_candidate_count=128,angle_pose_count=1536,query_points=sum(x['points'] for x in summary),
  frames=summary,causal_or_performance_claim=False)
(R/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
for n,h in json.loads((R/'runtime.json').read_text())['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
assert all(x['continuous_agreement']<.99 and x['voxel_center_agreement']>.99 for x in summary)
assert len(per)==1536
print(json.dumps(decision,ensure_ascii=False,indent=2))
