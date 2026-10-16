import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'gfp_reference_support_control_ab_v1_20261016'
def read(p):return json.loads(p.read_text())
def save(p,d):(ROOT/p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def csvfile(p,rows):
 with (ROOT/p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 train=read(ROOT/'receipts/training_complete.json');admit=read(ROOT/'receipts/admission_result.json')
 assert train['updates_total']==50000 and read(ROOT/'receipts/training_exit.json')['exit_code']==0
 events=read(ROOT/'receipts/reference_events.json');assert len(events)==500
 assert all(not e['density_model_loaded'] and not e['paired_radius'] for e in events)
 cal=read(ROOT/'receipts/calibration.json');rho=cal['rho0'];regional=np.array(cal['regional_start_outside'])
 rows=[dict(updates=e['updates'],rms=e['final']['rms'],start_rms=e['final']['regional'][0],outside_rms=e['final']['regional'][1],fraction=e['rho_fraction'],continuous=e['continuous'],alpha=e['final']['alpha'],follow_loss=e['final']['follow_loss'],reference_sha256=e['reference_sha256'],inputs_sha256=e['inputs_sha256']) for e in events]
 for e in events:
  f=e['final'];assert f['feasible'] and f['rms']<=rho and np.all(np.array(f['regional'])<=regional)
  assert e['continuous']==bool(f['rms']>=.5*rho and np.all(np.array(f['regional'])>0))
 csvfile('reports/reference_events.csv',rows)
 n=sum(e['continuous'] for e in events);continuous=n>=450
 probes=read(ROOT/'runs/A/probes.json');assert [p['updates'] for p in probes]==list(range(0,50001,5000))
 flags=[p['joint_anomaly'] for p in probes];longest=0;current=0
 for flag in flags:current=current+1 if flag else 0;longest=max(longest,current)
 evals={it:read(ROOT/f'runs/A/eval_{it}/results.json') for it in (25000,50000)}
 pairs=[];hashes={};erows=[];episodes=[];rng=np.random.default_rng(6824200)
 for it,e in evals.items():
  assert read(ROOT/f'runs/A/eval_{it}/exit.json')['exit_code']==0
  assert e['optimizer_updates']==0 and e['checkpoint_unchanged'] and len(e['episodes'])==100
  assert read(ROOT/f'runs/A/eval_{it}/cost.json')['workers_stopped']
  erows.append(dict(updates=it,success_percent=e['success_percent'],return_mean=e['return_mean'],start_stay_percent=e['start_stay_percent']))
  for ep in e['episodes']:
   if ep['episode'] in hashes:assert hashes[ep['episode']]==ep['initial_obs_sha256']
   hashes[ep['episode']]=ep['initial_obs_sha256']
   episodes.append(dict(updates=it,**{k:v for k,v in ep.items() if k!='path'}))
  for arm in ('A','B'):
   old=read(OLD/f'runs/{arm}/eval_{it}/results.json')
   a=sorted(e['episodes'],key=lambda x:x['episode']);b=sorted(old['episodes'],key=lambda x:x['episode'])
   assert all(x['initial_obs_sha256']==y['initial_obs_sha256'] for x,y in zip(a,b))
   d=np.array([x['success']-y['success'] for x,y in zip(a,b)]);boot=d[rng.integers(0,100,(10000,100))].mean(1)*100
   pairs.append(dict(updates=it,historical_arm=arm,independent_minus_historical_pp=float(d.mean()*100),paired_episode_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),training_repetitions=1,interpretation='Matched episodes, historical training run; not cross-training-seed uncertainty'))
 csvfile('reports/evaluation_episodes.csv',episodes);save('reports/evaluation_summary.json',erows);save('reports/historical_comparison.json',pairs)
 last=evals[50000];collapse=last['success_percent']<=5 and last['start_stay_percent']>=90
 strong=last['success_percent']>=25 and last['start_stay_percent']<=30 and longest<2
 decision='INCOMPLETE_AMPLITUDE_DIAGNOSTIC' if not continuous else ('LOCAL_ORDINARY_CONTROL_SUFFICIENT' if strong and not collapse else ('COLLAPSE_WITH_INDEPENDENT_UPDATES' if collapse else 'MIXED_RESULT'))
 costs={k:sum(e[k] for e in evals.values()) for k in ('external_steps','control_calls','physics_substeps')}
 assert costs['external_steps']<=400000 and costs['control_calls']<=416666 and costs['physics_substeps']<=2083330
 costs.update(gfp_updates=50000,reference_events=500,density_model_updates=0,density_model_loads=0,evaluation_episodes=200,training_wall_seconds=train['seconds']+admit['seconds'],training_cpu_seconds=train['cpu_seconds']+admit['cpu_seconds'],max_training_rss_kib=max(train['max_rss_kib'],admit['max_rss_kib']),evaluation_costs=[read(ROOT/f'runs/A/eval_{it}/cost.json') for it in evals])
 summary=dict(decision=decision,continuous_events=n,continuous_fraction=n/500,required_events=450,continuous_gate=continuous,radius_fraction_mean=float(np.mean([r['fraction'] for r in rows])),radius_fraction_quantiles=np.quantile([r['fraction'] for r in rows],[0,.05,.5,.95,1]).tolist(),zero_events=sum(r['rms']==0 for r in rows),longest_joint_anomaly_run=longest,joint_anomaly_points=[p['updates'] for p in probes if p['joint_anomaly']],behavior_collapse=collapse,strong_benefit=strong,evaluation=erows,initial_hashes_match_historical=True,source_hashes_unchanged=train['source_hashes_unchanged'],density_model_loaded=False,paired_radius=False,costs=costs)
 audit=read(ROOT/'runs/A/movement_audit.json');old_audit=read(OLD/'runs/A/support_audit.json');movement=[]
 for split in {g['split'] for g in audit[0]['groups']}:
  for region in ('start','outside'):
   own=[g for point in audit for g in point['groups'] if g['split']==split and g['region']==region]
   historical=[g for point in old_audit for g in point['groups'] if g['split']==split and g['region']==region]
   movement.append(dict(split=split,region=region,independent_coarse_path=sum(g['adjacent_checkpoint_rms'] for g in own[1:]),historical_A_coarse_path=sum(g['adjacent_checkpoint_rms'] for g in historical[1:]),independent_final_net_rms=own[-1]['net_initial_rms'],historical_A_final_net_rms=historical[-1]['net_initial_rms']))
 summary['fixed_sample_movement']=movement;summary['distinct_reference_hashes']=len({e['reference_sha256'] for e in events})
 save('reports/summary.json',summary);save('reports/costs.json',costs)
 index=[dict(path=str(p.with_suffix('.pkl')),**read(p)) for p in sorted((ROOT/'runs/A').glob('checkpoint_*.json'))]
 assert all(p['full_restore']=='PASS' for p in index);save('reports/checkpoint_index.json',index)
 oldevents=read(OLD/'receipts/reference_events.json');oldprobes=read(OLD/'runs/A/probes.json')
 summary['historical_paired_A_mean_radius_fraction']=float(np.mean([e['arms']['A']['final']['rms']/rho for e in oldevents]));save('reports/summary.json',summary)
 windows=[]
 for end in range(5000,50001,5000):
  for reg in ('start','outside'):
   own=sum(r[reg+'_rms'] for r in rows if end-5000<r['updates']<=end)
   historical=sum(e['arms']['A']['final']['regional'][0 if reg=='start' else 1] for e in oldevents if end-5000<e['updates']<=end)
   windows.append(dict(window_end=end,region=reg,independent_path=own,historical_paired_A_path=historical,ratio=own/historical if historical else None))
 csvfile('reports/regional_paths.csv',windows)
 fig,ax=plt.subplots(2,3,figsize=(14,8))
 for label,pp in [('Independent A',probes),('Historical paired A',oldprobes)]:
  xx=[p['updates'] for p in pp]
  ax[0,0].plot(xx,[p['historical_gap_increase'] for p in pp],'o-',label=label)
  ax[0,1].plot(xx,[p['historical_gradient_ratio'] for p in pp],'o-',label=label)
 ax[0,0].axhline(15,color='r',ls='--');ax[0,1].axhline(2,color='r',ls='--')
 ax[0,2].plot([25000,50000],[evals[it]['success_percent'] for it in evals],'o-',label='Independent A')
 ax[0,2].plot([25000,50000],[31,41],'o--',label='Historical paired A')
 ax[1,0].plot([r['updates'] for r in rows],[r['fraction'] for r in rows],label='Independent A')
 ax[1,0].axhline(.5,color='r',ls='--',label='minimum required/event')
 ax[1,1].plot([r['updates'] for r in rows],np.cumsum([r['continuous'] for r in rows])/np.arange(1,501),label='Independent A')
 ax[1,1].axhline(.9,color='r',ls='--',label='required final fraction')
 for reg in ('start','outside'):ax[1,2].plot([r['updates'] for r in rows],np.cumsum([r[reg+'_rms'] for r in rows]),label=reg)
 for a,title in zip(ax.flat,['Start Q gap increase vs 300k','Fixed-action gradient / 300k','Online Actor success (%)','Actual RMS / calibrated radius','Fraction of sufficient events','Regional cumulative event RMS']):
  a.set(title=title,xlabel='Additional GFP updates');a.grid(alpha=.2);a.legend(fontsize=8)
 fig.tight_layout();fig.savefig(ROOT/'reports/full_curves.png',dpi=170);plt.close(fig)
 ordered=[sorted(evals[it]['episodes'],key=lambda x:x['episode']) for it in (25000,50000)]
 a,b=ordered;d=np.array([y['success']-x['success'] for x,y in zip(a,b)]);rng=np.random.default_rng(6824201);boot=d[rng.integers(0,100,(10000,100))].mean(1)*100
 save('reports/temporal_comparison.json',dict(success_25k_to_50k_pp=float(d.mean()*100),paired_episode_ci95=np.quantile(boot,[.025,.975]).tolist(),lost_success=sum(x['success']==1 and y['success']==0 for x,y in zip(a,b)),gained_success=sum(x['success']==0 and y['success']==1 for x,y in zip(a,b)),both_success=sum(x['success']==1 and y['success']==1 for x,y in zip(a,b)),episode_bootstrap_seed=6824201,training_repetitions=1))
 print(json.dumps({k:v for k,v in summary.items() if k!='costs'}))
if __name__=='__main__':main()
