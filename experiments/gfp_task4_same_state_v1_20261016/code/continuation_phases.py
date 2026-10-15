"""Supplementary descriptive phase intervals, using saved arrays only."""
from pathlib import Path
import json,time,resource
import numpy as np
R=Path(__file__).resolve().parents[1]; O=R/'reports/continuation'; start=time.time()
rows=json.loads((R/'reports/branches_s1.json').read_text())
selected=[r for r in rows if r['state'] in ['s1_e003_t0056','s1_e010_t0061']]
data={(r['state'],r['mode'],r['candidate'],r['repeat']):dict(np.load(r['path'])) for r in selected}
out=[]
for r in selected:
 if not ((r['state']=='s1_e003_t0056' and r['candidate']=='rebrac' and r['mode']=='short_policy' and r['repeat']==0) or (r['state']=='s1_e010_t0061' and r['candidate']=='actor_2' and r['mode']=='first_action')):continue
 z=data[(r['state'],r['mode'],r['candidate'],r['repeat'])];o=z['observations'];a=z['actions'];intervals=[]
 for label,m in [('east',o[:,39]>.1),('west',o[:,39]<-.1),('low',(o[:,23]<.6)|(o[:,38]<.3)),('stable',(o[:,23]>=1.2)&(o[:,38]>=.7))]:
  edges=np.diff(np.r_[False,m,False].astype(int));iv=zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
  intervals.append({'label':label,'runs_at_least10':[[int(s),int(e-1)] for s,e in iv if e-s>=10]})
 out.append(dict(state=r['state'],candidate=r['candidate'],mode=r['mode'],repeat=r['repeat'],intervals=intervals))
(O/'phase_intervals.json').write_text(json.dumps(out,indent=2))
same_first=[]
a=data[('s1_e003_t0056','short_policy','rebrac',0)]
for rep in range(4):
 b=data[('s1_e003_t0056','first_action','rebrac',rep)]
 row=dict(repeat=rep,same_first_action=bool(np.array_equal(a['actions'][0],b['actions'][0])),same_observation_after_first_action=bool(np.array_equal(a['observations'][1],b['observations'][1])))
 assert row['same_first_action'] and row['same_observation_after_first_action']
 for name,v,k in [('east_position_advantage20',a['observations'][:,0]-b['observations'][:,0],20),('east_velocity_advantage10',a['observations'][:,39]-b['observations'][:,39],10)]:
  m=v>=.5;ids=[i for i in range(len(m)-k+1) if m[i:i+k].all()];row[name]=ids[0] if ids else None
 same_first.append(row)
(O/'same_first_action_comparison.json').write_text(json.dumps(same_first,indent=2))
receipt=dict(status='PASS',environment_steps=0,model_inference_calls=0,optimizer_updates=0,seconds=time.time()-start,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Existing trajectories; descriptive follow-up thresholds; no new confirmatory experiment')
(O/'phase_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
