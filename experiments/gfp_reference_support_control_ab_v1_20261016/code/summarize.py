import json,csv,itertools,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):(ROOT/p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def csvfile(p,rows):
 with (ROOT/p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 training=read('receipts/training_complete.json');assert training['updates_total']==100000
 assert read('receipts/training_exit.json')['exit_code']==0
 evals={arm:{it:read(f'runs/{arm}/eval_{it}/results.json') for it in (25000,50000)} for arm in ('A','B')}
 for arm,it in itertools.product(('A','B'),(25000,50000)):
  assert read(f'runs/{arm}/eval_{it}/exit.json')['exit_code']==0
  e=evals[arm][it];assert e['optimizer_updates']==0 and e['checkpoint_unchanged'] and len(e['episodes'])==100
  assert read(f'runs/{arm}/eval_{it}/cost.json')['workers_stopped']
 hashes={}
 for arm,it in itertools.product(('A','B'),(25000,50000)):
  for e in evals[arm][it]['episodes']:
   if e['episode'] in hashes:assert hashes[e['episode']]==e['initial_obs_sha256']
   hashes[e['episode']]=e['initial_obs_sha256']
 events=read('receipts/reference_events.json');assert len(events)==500
 rows=[];paths=[]
 for e in events:
  r=dict(updates=e['updates'],input_sha256=e['inputs_sha256'],radius_fraction=e['rho_fraction'],global_match=e['global_match'],regional_match=e['regional_match'],continuous=e['continuous'],global_relative_diff=e['relative_diff_global_start_outside'][0],start_relative_diff=e['relative_diff_global_start_outside'][1],outside_relative_diff=e['relative_diff_global_start_outside'][2])
  for arm in ('A','B'):
   f=e['arms'][arm]['final'];r.update({arm+'_rms':f['rms'],arm+'_start_rms':f['regional'][0],arm+'_outside_rms':f['regional'][1],arm+'_alpha':f['alpha'],arm+'_follow_loss':f['follow_loss'],arm+'_clipped_fraction':f['clipped_fraction']})
  r['B_start_density_change'],r['B_outside_density_change']=e['arms']['B']['final']['density_change']
  r['B_projection_active']=bool(np.any(np.asarray(e['arms']['B']['projection']['multipliers'])>1e-10))
  rows.append(r)
 for end in range(5000,50001,5000):
  window=[r for r in rows if end-5000<r['updates']<=end]
  for region in ('start','outside'):
   a=sum(r['A_'+region+'_rms'] for r in window);b=sum(r['B_'+region+'_rms'] for r in window)
   paths.append(dict(window_end=end,region=region,A_path=a,B_path=b,relative_difference=abs(a-b)/max(a,b,1e-8)))
 csvfile('reports/reference_matching.csv',rows);csvfile('reports/event_path_windows.csv',paths)
 jointmatch=np.mean([r['global_match'] and r['regional_match'] for r in rows]);continuous=np.mean([r['continuous'] for r in rows]);windowpass=all(x['relative_difference']<=.1 for x in paths)
 audits={a:read(f'runs/{a}/support_audit.json') for a in ('A','B')};coarse=[]
 for split in set(g['split'] for g in audits['A'][0]['groups']):
  for region in ('start','outside'):
   totals={a:sum(g['adjacent_checkpoint_rms'] for point in audits[a][1:] for g in point['groups'] if g['split']==split and g['region']==region) for a in ('A','B')}
   coarse.append(dict(split=split,region=region,**totals,relative_difference=abs(totals['A']-totals['B'])/max(totals.values()),B_less_than_A_fraction=(totals['A']-totals['B'])/max(totals['A'],1e-8)))
 save('reports/coarse_audit_paths.json',coarse)
 testcoarse=[x for x in coarse if x['split'] in ('val','test')]
 assert testcoarse,'Holdout split labels must be explicit'
 holdoutconfound=any(x['relative_difference']>.1 for x in testcoarse)
 probes={a:read(f'runs/{a}/probes.json') for a in ('A','B')};outcomes={};pair=[]
 for arm in ('A','B'):
  flags=[p['joint_anomaly'] for p in probes[arm]];longest=0;current=0
  for flag in flags:current=current+1 if flag else 0;longest=max(longest,current)
  last=evals[arm][50000]
  outcomes[arm]=dict(joint_anomaly_points=[p['updates'] for p in probes[arm] if p['joint_anomaly']],longest_anomaly_run=longest,joint_anomaly=longest>=2,behavior_collapse=last['success_percent']<=5 and last['start_stay_percent']>=90,strong_benefit=last['success_percent']>=25 and last['start_stay_percent']<=30 and longest<2)
 rng=np.random.default_rng(6824200)
 for it in (25000,50000):
  ordered={a:sorted(evals[a][it]['episodes'],key=lambda x:x['episode']) for a in ('A','B')}
  diff=np.array([b['success']-a['success'] for a,b in zip(ordered['A'],ordered['B'])]);boot=diff[rng.integers(0,100,(10000,100))].mean(1)*100
  pair.append(dict(updates=it,B_minus_A_success_pp=float(diff.mean()*100),paired_episode_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_seed=6824200,replicates=10000,training_repetitions=1))
 summaryrows=[dict(arm=a,updates=it,success_percent=evals[a][it]['success_percent'],return_mean=evals[a][it]['return_mean'],final_start_percent=evals[a][it]['start_stay_percent'],**outcomes[a]) for a,it in itertools.product(('A','B'),(25000,50000))]
 save('reports/evaluation_summary.json',summaryrows);save('reports/paired_comparison.json',pair)
 episodes=[dict(arm=a,updates=it,**{k:v for k,v in e.items() if k!='path'}) for a,it in itertools.product(('A','B'),(25000,50000)) for e in evals[a][it]['episodes']]
 csvfile('reports/evaluation_episodes.csv',episodes)
 cost={k:sum(evals[a][it][k] for a,it in itertools.product(('A','B'),(25000,50000))) for k in ('external_steps','control_calls','physics_substeps')}
 assert cost['external_steps']<=800000 and cost['control_calls']<=833332 and cost['physics_substeps']<=4166660
 admission=read('receipts/admission_result.json');preflight=read('receipts/preflight_attempt1/execution_error.json')['time']-read('receipts/preflight_attempt1/admission_started.json')['time']
 cost.update(gfp_updates=100000,reference_attempts=1000,density_model_updates=0,evaluation_episodes=400,training_wall_seconds=admission['seconds']+training['seconds'],training_cpu_seconds=admission['cpu_seconds']+training['cpu_seconds'],extra_failed_zero_update_preflight_wall_seconds=preflight,evaluation_costs=[read(f'runs/{a}/eval_{it}/cost.json') for a,it in itertools.product(('A','B'),(25000,50000))])
 save('reports/costs.json',cost)
 matching=bool(jointmatch>=.9 and continuous>=.9 and windowpass and not holdoutconfound)
 if not matching:decision='INCONCLUSIVE_MATCHING_OR_SLOWDOWN_CONFOUND'
 elif outcomes['B']['strong_benefit'] and outcomes['A']['behavior_collapse']:decision='LOCAL_SUPPORT_PENDING_AUDIT_INTERPRETATION'
 elif pair[-1]['B_minus_A_success_pp']>=15 and pair[-1]['paired_episode_bootstrap_95'][0]>0:decision='MIXED_LOCAL_EVIDENCE'
 else:decision='NO_INCREMENTAL_BENEFIT_ESTABLISHED'
 report=dict(decision=decision,matching=dict(joint_event_match_fraction=float(jointmatch),continuous_fraction=float(continuous),all_5k_regional_path_windows_pass=windowpass,holdout_coarse_path_confound=holdoutconfound,full_gate_pass=matching),outcomes=outcomes,paired=pair,initial_hashes_all_match=True,source_hashes_unchanged=training['source_hashes_unchanged'],projection_active_fraction=float(np.mean([r['B_projection_active'] for r in rows])),costs=cost)
 save('reports/summary.json',report)
 checkpoints=[]
 for a in ('A','B'):
  for p in sorted((ROOT/f'runs/{a}').glob('checkpoint_*.json')):
   d=json.loads(p.read_text());assert d['full_restore']=='PASS';checkpoints.append(dict(arm=a,path=str(p.with_suffix('.pkl')),**d))
 save('reports/checkpoint_index.json',checkpoints)
 fig,ax=plt.subplots(2,3,figsize=(14,8))
 for a in ('A','B'):
  pp=probes[a];xx=[p['updates'] for p in pp]
  ax[0,0].plot(xx,[p['historical_gap_increase'] for p in pp],'o-',label=a)
  ax[0,1].plot(xx,[p['historical_gradient_ratio'] for p in pp],'o-',label=a)
  ax[0,2].plot([25000,50000],[evals[a][it]['success_percent'] for it in (25000,50000)],'o-',label=a)
  ax[1,0].plot([r['updates'] for r in rows],np.cumsum([r[a+'_start_rms'] for r in rows]),label=a+' start')
  for seed,style in [('1701','-'),('1702','--')]:
   group=[next(g for g in p['groups'] if g['region']=='start' and g['split'] in ('val','test')) for p in audits[a]]
   ax[1,2].plot(xx,[g[seed]['nll_mean'] for g in group],style,label=a+' MDN '+seed)
 ax[0,0].axhline(15,color='r',ls='--');ax[0,1].axhline(2,color='r',ls='--')
 ax[1,1].plot([r['updates'] for r in rows],[r['radius_fraction'] for r in rows]);ax[1,1].axhline(.5,color='r',ls='--')
 for a,title in zip(ax.flat,['Start Q gap increase vs 300k','Fixed-action gradient / 300k','Online Actor success (%)','Start-region cumulative event RMS','Common radius / initial radius','Holdout start NLL / action dimension']):
  a.set(title=title,xlabel='Additional GFP updates');a.grid(alpha=.2)
 for a in (ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,2]):a.legend()
 fig.tight_layout();fig.savefig(out/'full_curves.png',dpi=170);plt.close(fig)
 print(json.dumps({k:v for k,v in report.items() if k!='costs'}))
if __name__=='__main__':main()
