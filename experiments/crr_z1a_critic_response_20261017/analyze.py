from pathlib import Path
import json,csv,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;t=time.monotonic();D=dict(np.load(R/'arrays/probes.npz'))
ids,ix=np.unique(D['trajectory'],return_inverse=True);rng=np.random.default_rng(20261731);W=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=400)[:,ix].astype(float)
masks={'start':D['start'],'nonstart':~D['start']};rows=[]
def ci(x):return [float(v) for v in np.quantile(x,[.025,.975])]
for seed in range(4):
 root=R/f'runs/critic_s{seed}';assert json.loads((root/'exit.json').read_text())['exit_code']==0;assert json.loads((root/'complete.json').read_text())['critic_only_updates']==15000
 baseline=dict(np.load(root/'probe_F_00000.npz'));g0=baseline['grad'];n0=np.linalg.norm(g0,axis=-1)
 for step in range(0,5001,1000):
  a={arm:dict(np.load(root/f'probe_{arm}_{step:05d}.npz')) for arm in ['F','P','Fprime']}
  for j in range(2):
   norms={arm:np.linalg.norm(v['grad'][j],axis=-1) for arm,v in a.items()};response=np.linalg.norm(a['P']['grad'][j]-a['F']['grad'][j],axis=-1);noise=np.linalg.norm(a['F']['grad'][j]-a['Fprime']['grad'][j],axis=-1)
   for region,m in masks.items():
    w=W*m;den=w.sum(1);assert min(den)>0;bf=lambda v:w@v/den
    scale=float(n0[j,m].mean());boot_scale=bf(n0[j]);rr=bf(response)/boot_scale;nn=bf(noise)/boot_scale
    row=dict(seed=seed,group='bad_trajectory_candidate' if seed in [0,3] else 'intended_healthy_seed_new_continuation',latent=j,region=region,updates=step,G_fork=scale,R_PminusF_vector=float(response[m].mean()/scale),N_FminusFprime_vector=float(noise[m].mean()/scale),signed_norm_response=float((norms['P'][m].mean()-norms['F'][m].mean())/scale),R_raw=float(response[m].mean()),N_raw=float(noise[m].mean()))
    for key,val in [('R',rr),('N',nn),('R_minus_N',rr-nn)]:row[key+'_lo'],row[key+'_hi']=ci(val)
    for arm in a:
     gg=norms[arm][m].mean();row[f'G_{arm}']=float(gg);row[f'G_delta_{arm}']=float(gg-scale);row[f'G_delta_normalized_{arm}']=float((gg-scale)/scale);row[f'vector_drift_{arm}']=float(np.linalg.norm(a[arm]['grad'][j]-g0[j],axis=-1)[m].mean()/scale)
     q=a[arm]['q'][j];row[f'Q_{arm}']=float(q[m].mean());row[f'Q_drift_{arm}']=float((q-baseline['q'][j])[m].mean());row[f'contrast_{arm}']=float(q[D['start']].mean()-q[~D['start']].mean())
    row['Q_abs_PminusF']=float(np.abs(a['P']['q'][j]-a['F']['q'][j])[m].mean());row['Q_abs_FminusFprime']=float(np.abs(a['F']['q'][j]-a['Fprime']['q'][j])[m].mean());rows.append(row)
with open(R/'metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
end=[x for x in rows if x['updates']==5000 and x['region']=='start'];tests=[]
for seed in [0,3]:
 for j in range(2):
  b=next(x for x in end if x['seed']==seed and x['latent']==j);h=[x for x in end if x['seed'] in [1,2] and x['latent']==j]
  tests.append(dict(seed=seed,latent=j,bad_R=b['R_PminusF_vector'],healthy_R=[v['R_PminusF_vector'] for v in h],bad_R_ci=[b['R_lo'],b['R_hi']],healthy_R_ci=[[v['R_lo'],v['R_hi']] for v in h],noise_ci=[b['N_lo'],b['N_hi']],bad_exceeds_both_healthy_and_own_noise=(b['R_lo']>max(v['R_hi'] for v in h) and b['R_lo']>b['N_hi'])))
passed={s:all(x['bad_exceeds_both_healthy_and_own_noise'] for x in tests if x['seed']==s) for s in [0,3]}
if all(passed.values()):label='LIMITED_GO_MORE_CANDIDATES_AND_LATER_FPU_ONLY'
elif any(passed.values()):label='INCONCLUSIVE_ONLY_ONE_BAD_CANDIDATE'
else:label='NO_GO_CURRENT_SHORT_RESPONSE_DISCRIMINATION'
decision=dict(status=label,endpoint=5000,tests=tests,per_bad_candidate_passed=passed,primary='vector action-gradient response, normalized by fork scale',optimizer_noise_limitation='Only one alternate minibatch stream; state-trajectory bootstrap does not estimate optimization-stream uncertainty.',healthy_control_limitation='New continuations from historically healthy seeds; not bitwise missing original checkpoints, and no new behavioral evaluation authorized.',behavioral_harm_of_individual_update='NOT_ESTABLISHED',risk_prediction='NOT_ESTABLISHED',independent_generalization='NOT_TESTED',next_experiments_started=False)
(R/'decision.json').write_text(json.dumps(decision,indent=2))
def ss(seed,j,region='start'):return [x for x in rows if x['seed']==seed and x['latent']==j and x['region']==region]
colors={0:'#bd3b34',1:'#2176a6',2:'#31966e',3:'#d38629'}
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for j,ax in enumerate(axs):
 for seed in range(4):
  s=ss(seed,j);x=[v['updates'] for v in s];ax.plot(x,[v['R_PminusF_vector'] for v in s],color=colors[seed],marker='o',ms=3,label=f's{seed} '+('bad trajectory' if seed in [0,3] else 'intended healthy'));ax.fill_between(x,[v['R_lo'] for v in s],[v['R_hi'] for v in s],color=colors[seed],alpha=.12);ax.plot(x,[v['N_FminusFprime_vector'] for v in s],color=colors[seed],ls=':',alpha=.85)
 ax.set_title(f'Fixed latent {j} | start states');ax.set_xlabel('Critic-only updates');ax.set_ylabel('Gradient vector response / fork gradient norm');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Solid: P-F candidate response | dotted: F-Fprime minibatch noise\nBands: fixed-data trajectory bootstrap, not training-seed or optimization-stream uncertainty',fontsize=10);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(R/'gradient_response.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8))
for seed,ax in enumerate(axs.flat):
 for j in range(2):
  s=ss(seed,j);x=[v['updates'] for v in s]
  for arm,ls in [('F','-'),('P','--'),('Fprime',':')]:ax.plot(x,[v[f'G_{arm}'] for v in s],ls,color=['#237aa4','#d1862a'][j],label=f'z{j} {arm}')
 ax.axhline(ss(seed,0)[0]['G_fork'],color='gray',lw=.7);ax.set_title(f's{seed}: raw online action-gradient mean norm');ax.set_xlabel('Critic-only updates');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
fig.suptitle('Absolute F/P/Fprime gradient curves | same fixed action input');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(R/'absolute_gradients.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8))
for seed,ax in enumerate(axs.flat):
 for j in range(2):
  s=ss(seed,j);x=[v['updates'] for v in s]
  for arm,ls in [('F','-'),('P','--'),('Fprime',':')]:ax.plot(x,[v[f'contrast_{arm}'] for v in s],ls,color=['#237aa4','#d1862a'][j],label=f'z{j} {arm}')
 ax.set_title(f's{seed}: Target-Q start minus nonstart');ax.set_xlabel('Critic-only updates');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
fig.suptitle('Auxiliary fixed-action Target-Q contrast | absolute drifts and differences in metrics.csv');fig.tight_layout(rect=(0,0,1,.95));fig.savefig(R/'q_response.png',dpi=140);plt.close(fig)
(R/'receipts/analysis.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-t,rows=len(rows),trajectory_clusters=len(ids),bootstrap_replicates=400,posthoc_thresholds_fitted=False),indent=2))
print('DECISION',label,flush=True)
for x in end:print({k:round(x[k],5) if isinstance(x[k],float) else x[k] for k in ['seed','latent','R_PminusF_vector','N_FminusFprime_vector','R_lo','R_hi','G_fork','G_F','G_P','G_Fprime','G_delta_F','G_delta_P']},flush=True)
