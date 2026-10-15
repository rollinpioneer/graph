"""Analyze the fixed 16 branches and reuse original endpoints; no simulation."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
assert read(R/'receipts/exit.json')['exit_code']==0
assert read(R/'receipts/completed.json')['status']=='PASS'
c=read(R/'configs/protocol.json');new=read(R/'reports/new_branches.json');rows=[];arrays={}
sources=[dict(x,length=0 if x['candidate']=='original_actor' else (1 if x['mode']=='first_action' else 200),reused=True) for x in c['references']]+[dict(x,reused=False) for x in new]
for x in sources:
 p=Path(x['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];z=dict(np.load(p));o=z['observations'];a=z['actions'];n=len(a);L=x['length'];rep=x['repeat'];arrays[(L,rep)]=z
 inside=np.all(((o[:,:2]+6)/4).astype(int)[:,::-1]==[6,5],axis=1);outside=~inside;stable=(o[:,23]>=1.2)&(o[:,38]>=.7);low=(o[:,23]<.6)|(o[:,38]<.3);dt=float(np.median(np.diff(z['time'])))
 ex=np.flatnonzero(outside);leave=int(ex[0]) if len(ex) else None;ret=np.flatnonzero(inside[leave+1:])+leave+1 if leave is not None else [];back=int(ret[0]) if len(ret) else None
 row=dict(length=L,repeat=rep,reused=x['reused'],steps=n,first_exit=leave,first_return=back,final_outside=bool(outside[-1]),outside_at_handoff=bool(outside[min(L,n)]),first_outside_seconds=None if leave is None else ((back if back is not None else n)-leave)*dt,first_outside_right_censored=leave is not None and back is None,total_outside_seconds=float(outside[1:].sum()*dt),stable_fraction=float(stable[1:].mean()),low_fraction=float(low[1:].mean()),wall_contact_fraction=float((z['wall_contacts'][1:]>0).mean()),final_x=float(o[-1,0]),final_y=float(o[-1,1]),post_handoff_dx=float(o[-1,0]-o[L,0]) if L<n else None,post_handoff_stable_fraction=float(stable[L+1:].mean()) if L<n else None,post_handoff_low_fraction=float(low[L+1:].mean()) if L<n else None,final_head=float(o[-1,23]),final_upright=float(o[-1,38]),success=float(z['success'].max()),return_=float(z['rewards'].sum()),path=str(p),sha256=x['sha256'])
 for h in [10,20,50,100,150,200]:row[f'dx_{h}']=float(o[min(h,n),0]-o[0,0])
 rows.append(row)
rows.sort(key=lambda x:(x['length'],x['repeat']))
(R/'reports/branch_metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
with (R/'reports/branch_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
summary=[]
for L in [0,1,10,50,100,150,200]:
 rr=[x for x in rows if x['length']==L]
 s=dict(length=L,branches=len(rr),independent_continuations=1 if L==200 else 4,ever_exit=sum(x['first_exit'] is not None for x in rr),final_outside=sum(x['final_outside'] for x in rr),maintained_after_handoff=sum(x['post_handoff_dx'] is not None and x['post_handoff_dx']>=0 and x['post_handoff_stable_fraction']>=.5 for x in rr),final_x=[round(x['final_x'],4) for x in rr],post_handoff_dx=[None if x['post_handoff_dx'] is None else round(x['post_handoff_dx'],4) for x in rr],first_exits=[x['first_exit'] for x in rr],first_returns=[x['first_return'] for x in rr],mean_stable=float(np.mean([x['stable_fraction'] for x in rr])),mean_low=float(np.mean([x['low_fraction'] for x in rr])))
 s['predefined_continue_signal']=L in [10,50,100,150] and s['final_outside']>=3 and s['maintained_after_handoff']>=3
 s['joint_final_outside_and_maintained']=sum(x['final_outside'] and x['post_handoff_dx'] is not None and x['post_handoff_dx']>=0 and x['post_handoff_stable_fraction']>=.5 for x in rr)
 summary.append(s)
(R/'reports/length_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
o=arrays[(200,0)]['observations']
handoff=[dict(length=L,x=float(o[L,0]),y=float(o[L,1]),vx=float(o[L,39]),vy=float(o[L,40]),head=float(o[L,23]),upright=float(o[L,38])) for L in [1,10,50,100,150,200]]
(R/'reports/handoff_states.json').write_text(json.dumps(handoff,indent=2))
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for ax,L in zip(axes.flat,[1,10,50,100,150,200]):
 for rep in range(4):ax.plot(arrays[(L,rep)]['observations'][:,0],color=colors[rep],lw=1.4,alpha=.9,label=f'continuation {rep}')
 ax.axhline(18,color='black',ls='--',lw=.8,label='east exit boundary')
 ax.axvline(L,color='gray',ls=':',lw=1)
 ax.set_title(f'ReBRAC {L} steps'+(' (reused)' if L in [1,200] else ''))
 ax.set_xlabel('Steps since frozen state');ax.set_ylabel('Root x (m)');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=7);fig.savefig(R/'reports/position_by_takeover.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
Ls=[0,1,10,50,100,150,200]
for rep in range(4):
 rr=[next(x for x in rows if x['length']==L and x['repeat']==rep) for L in Ls]
 for ax,key,label in zip(axes.flat,['dx_200','total_outside_seconds','stable_fraction','post_handoff_dx'],['East progress at h=200 (m)','Total time outside (s)','Standing fraction','East displacement after handoff (m)']):
  ax.plot(Ls,[np.nan if x[key] is None else x[key] for x in rr],'-o',color=colors[rep],ms=3,label=f'continuation {rep}');ax.set_ylabel(label);ax.set_xlabel('ReBRAC takeover steps');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.savefig(R/'reports/outcomes_by_length.png',dpi=120);plt.close(fig)
print(json.dumps(summary,indent=2))
