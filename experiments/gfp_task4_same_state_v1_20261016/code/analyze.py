from pathlib import Path
import json,hashlib,csv,collections,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(R/'reports'/n).write_text(json.dumps(x,indent=2,allow_nan=False))
states=read(R/'configs/states.json')['states'];rows=[];cost=[]
for s in [0,1]:
 assert read(R/'receipts'/f'exit_s{s}.json')['exit_code']==0
 done=read(R/'receipts'/f'completed_s{s}.json');assert done['assets_unchanged'] and done['optimizer_updates']==0;cost.append(done)
 rows+=read(R/'reports'/f'branches_s{s}.json')
assert len(rows)==len(states)*48 and sum(x['physical_environment_steps'] for x in cost)<=90000
for x in rows:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
metrics=['exit_initial_cell','east_exit','upright_east_exit','stable_fraction','stable_first50','east_progress50','wall_contact_fraction','west_wall_near_fraction','success']
agg=[]
for c in states:
 for mode in ['first_action','short_policy']:
  for label in dict.fromkeys(x['candidate'] for x in rows if x['state']==c['id'] and x['mode']==mode):
   g=[x for x in rows if (x['state'],x['mode'],x['candidate'])==(c['id'],mode,label)];assert sorted(x['repeat'] for x in g)==[0,1,2,3]
   g=sorted(g,key=lambda x:x['repeat'])
   agg.append(dict(state=c['id'],seed=c['seed'],mode=mode,candidate=label,**{n:float(np.mean([x[n] for x in g])) for n in metrics},dx50=float(np.mean([x['dx']['50'] for x in g])),dx200=float(np.mean([x['dx']['200'] for x in g])),online_q=float(np.mean([x['online_q'] for x in g])),target_q=float(np.mean([x['target_q'] for x in g])),online_members=np.mean([x['online_q'] for x in g],axis=0).tolist(),target_members=np.mean([x['target_q'] for x in g],axis=0).tolist(),paired_east_progress=[x['east_progress50'] for x in g],paired_exit=[x['east_exit'] for x in g],paired_dx50=[x['dx']['50'] for x in g],paired_dx200=[x['dx']['200'] for x in g]))
save('candidate_aggregates.json',agg)
state_rows=[]
for c in states:
 g=[x for x in agg if x['state']==c['id'] and x['mode']=='first_action'];base=next(x for x in g if x['candidate']=='original_actor');alts=[x for x in g if x is not base]
 robust=[]
 for x in alts:
  tests=[n for n in ['east_exit','east_progress50'] if x[n]>=.75 and base[n]<=.25 and x[n]-base[n]>=.5]
  if tests:robust.append(dict(candidate=x['candidate'],metrics=tests,online_q_difference=x['online_q']-base['online_q'],target_q_difference=x['target_q']-base['target_q'],member_online_differences=(np.asarray(x['online_members'])-base['online_members']).tolist(),exit_difference=x['east_exit']-base['east_exit'],east_progress_difference=x['east_progress50']-base['east_progress50']))
 top=max(g,key=lambda x:x['online_q']);best=max(g,key=lambda x:(x['east_exit'],x['east_progress50'],x['dx50']))
 short=[x for x in agg if x['state']==c['id'] and x['mode']=='short_policy']
 state_rows.append(dict(state=c['id'],seed=c['seed'],episode=c['episode'],step=c['step'],initial_xy=c['observation'][:2],base=base,best_actor=max([x for x in g if x['candidate'].startswith('actor_')],key=lambda x:(x['east_exit'],x['east_progress50'],x['dx50'])),best_vabc=max([x for x in g if x['candidate'].startswith('vabc_')],key=lambda x:(x['east_exit'],x['east_progress50'],x['dx50'])),rebrac_first=next(x for x in g if x['candidate']=='rebrac'),top_online_q=top,posthoc_best=best,robust_alternatives=robust,short_policy=short))
save('state_comparisons.json',state_rows)
with (R/'reports/branch_metrics.csv').open('w',newline='') as f:
 fields=['state','seed','mode','candidate','repeat','steps',*metrics,'dx20','dx50','dx100','dx200','online_q_mean','target_q_mean'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for x in rows:w.writerow({**{n:x[n] for n in fields if n in x},**{f'dx{h}':x['dx'][str(h)] for h in [20,50,100,200]},'online_q_mean':float(np.mean(x['online_q'])),'target_q_mean':float(np.mean(x['target_q']))})
failed_cost=[]
for p in (R/'attempts').glob('*/exit_s*.json'):
 x=read(p);failed_cost.append(dict(path=str(p),exit_code=x['exit_code'],seconds=x['end']-x['start']))
res=dict(status='COMPLETE',states=len(states),branches=len(rows),physical_environment_steps=sum(x['physical_environment_steps'] for x in cost),external_environment_steps=sum(x['external_environment_steps'] for x in cost),optimizer_updates=0,failed_attempt_costs=failed_cost,worker_costs=cost,gpu_residency_hours_including_gates_and_failed_attempts=(sum(x['elapsed_seconds'] for x in cost)+sum(x['seconds'] for x in failed_cost))/3600,robust_alternative_states=sum(bool(x['robust_alternatives']) for x in state_rows),robust_gfp_candidate_states=sum(any(y['candidate'].startswith(('actor_','vabc_')) for y in x['robust_alternatives']) for x in state_rows),source_assets_unchanged=True,all_branch_hashes_verified=True)
save('summary.json',res)
labels=['original_actor','actor_0','actor_1','actor_2','actor_3','vabc_0','vabc_1','vabc_2','vabc_3','rebrac']
fig,ax=plt.subplots(1,2,figsize=(15,7),layout='constrained')
for a,metric,title in zip(ax,['east_exit','east_progress50'],['East exit within 200 steps','East progress + stable posture at 50 steps']):
 mat=np.array([[next(v for v in agg if (v['state'],v['mode'],v['candidate'])==(c['id'],'first_action',lab))[metric] for c in states] for lab in labels]);im=a.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto');a.set_yticks(range(len(labels)),labels);a.set_xticks(range(len(states)),[c['id'] for c in states],rotation=55,ha='right');a.set_title(title)
 for i in range(len(labels)):
  for j in range(len(states)):a.text(j,i,f'{int(round(mat[i,j]*4))}/4',ha='center',va='center',color='white' if mat[i,j]>.6 else 'black')
fig.savefig(R/'reports/first_action_outcomes.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for s in [0,1]:
 x=[v for v in state_rows if v['seed']==s];xx=np.arange(len(x));width=.25
 for k,label in enumerate(['actor','vabc','rebrac']):
  yy=[v['base']['east_exit'] if label=='actor' else next(z['east_exit'] for z in v['short_policy'] if z['candidate']==label) for v in x]
  ax[s].bar(xx+(k-1)*width,yy,width,label=label)
 ax[s].set_xticks(xx,[v['state'] for v in x],rotation=35,ha='right');ax[s].set_ylim(0,1.05);ax[s].set_title(f'Seed {s}: short-policy east exit');ax[s].legend()
fig.savefig(R/'reports/short_policy_outcomes.png',dpi=140);plt.close(fig)
print(json.dumps(res));print('STATE_COMPARISONS')
for x in state_rows:print(x['state'],'original',x['base']['east_exit'],x['base']['east_progress50'],'bestActor',x['best_actor']['candidate'],x['best_actor']['east_exit'],x['best_actor']['east_progress50'],'bestVaBC',x['best_vabc']['candidate'],x['best_vabc']['east_exit'],x['best_vabc']['east_progress50'],'ReBRACfirst',x['rebrac_first']['east_exit'],'short',[(z['candidate'],z['east_exit']) for z in x['short_policy']],'robust',x['robust_alternatives'])
