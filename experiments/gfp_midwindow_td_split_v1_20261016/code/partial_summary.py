import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xushijie3/work/gfp_midwindow_td_freeze_v1_20261016')
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=rd(ROOT/'protocol.json');a=rd(ROOT/'receipts/assets.json')
assert sha(Path(p['source']))==p['source_sha256']
assert all(sha(Path(k))==v for k,v in a['historical_sha256'].items())
result={'status':'TRAINING_COMPLETE_EVALUATION_BLOCKED','decision':'UNDETERMINED_EVALUATION_BLOCKED','source_unchanged':True,'historical_files_unchanged':True,'arms':{}}
for arm in ['L0','L1','L1a','L1b']:
 d=(OLD if arm in ['L0','L1'] else ROOT)/('runs/s3_'+arm)
 points=rd(d/'probes.json');meta=[rd(d/f'checkpoint_{i}.json') for i in range(0,50001,5000)]
 assert rd(d/'exit.json')['exit_code']==0
 assert [x['updates'] for x in points]==list(range(0,50001,5000))
 assert all(x['full_restore']=='PASS' and x['step']==350000+x['updates'] for x in meta)
 streak=longest=0;onset=None
 for x in points:
  hit=x['historical_gap_increase']>=15 and x['historical_gradient_ratio']>=2
  assert hit==x['joint_anomaly']
  streak=streak+1 if hit else 0;longest=max(longest,streak)
  if streak==2 and onset is None:onset=x['updates']-5000
 ev={str(i):rd(d/f'eval_{i}/results.json') if (d/f'eval_{i}/results.json').exists() else None for i in [25000,50000]}
 for e in ev.values():
  if e:e.pop('episodes',None)
 if arm in ['L1a','L1b']:
  assert rd(d/'admission.json')['passed'] and rd(d/'zero_update_admission.json')['passed']
  assert all(x['agent_rng']==y['agent_rng'] for x,y in zip(meta,result['arms']['L0']['checkpoint_index']))
 if arm=='L1a':assert len({x['reference_sha256'] for x in meta})==1
 if arm=='L1b':
  assert len({x['frozen_vabc_sha256'] for x in meta})==1
  assert meta[0]['reference_sha256']!=meta[-1]['reference_sha256']
 result['arms'][arm]={'historical':arm in ['L0','L1'],'probes':points,'checkpoint_index':meta,'training':rd(d/'complete.json'),'evaluation':ev,'classification':{'joint_anomaly':longest>=2,'onset_updates':onset,'longest_consecutive_points':longest,'behavior_collapse':None if ev['50000'] is None else ev['50000']['success_percent']<=5 and ev['50000']['start_stay_percent']>=90,'strong_benefit':None if ev['50000'] is None else ev['50000']['success_percent']>=25 and ev['50000']['start_stay_percent']<=30 and longest<2}}
assert sum(result['arms'][x]['training']['updates'] for x in ['L1a','L1b'])==100000
result['cost']={'new_optimizer_updates':100000,'completed_new_evaluation_episodes':0,'evaluation_external_steps':0,'failed_construction_control_calls_upper_bound':800,'failed_construction_physics_substeps_upper_bound':4000,'control_count_note':'Conservative code-path bound: 20 constructors, each one reset with 40 do_simulation calls, each 5 physics substeps. Parent blocked before all ready; no policy steps issued. No surviving exact worker counters. Do not present this bound as measured exact cost.','training_cpu_seconds':sum(result['arms'][x]['training']['cpu_seconds'] for x in ['L1a','L1b']),'evaluation_cpu_seconds':None}
result['limitations']=['One source state and one paired continuation, historical controls not bitwise replay.','Evaluation failed in native EGL rendering; success/return/start-stay and paired initial hashes unavailable.','Value anomaly alone cannot establish behavioral collapse or retained benefit.','Evaluation CPU and exact startup control counters unavailable; retained conservative control upper bound.']
(ROOT/'reports/partial_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
fig,axes=plt.subplots(2,3,figsize=(14,8))
keys=['historical_gap_increase','historical_gradient_ratio','dqda_cos_start','V_target_start','action_shift_start','dqda_norm_start']
for ax,key in zip(axes.flat,keys):
 for arm,v in result['arms'].items():
  if key not in v['probes'][0]:continue
  ax.plot([x['updates']/1000 for x in v['probes']],[x[key] for x in v['probes']],linestyle='--' if v['historical'] else '-',marker='.',label=arm+(' historical' if v['historical'] else ''))
 ax.set_title(key);ax.set_xlabel('Continuation updates (k)');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].axhline(15,color='gray',ls=':');axes[0,1].axhline(2,color='gray',ls=':')
fig.suptitle('TD split: training complete; behavior evaluation unavailable')
fig.tight_layout();fig.savefig(ROOT/'reports/partial_curves.png',dpi=160)
print(json.dumps({k:{'classification':v['classification'],'last':{q:v['probes'][-1].get(q) for q in keys}} for k,v in result['arms'].items()},indent=2))
