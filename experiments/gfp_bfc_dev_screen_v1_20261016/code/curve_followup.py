import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for seed in (0,3):
 for arm in [f'R{i}' for i in range(6)]:
  points=json.loads((root/'runs'/f's{seed}_{arm}'/'probes.json').read_text());b=points[0]
  series=[]
  for x in points:
   gap=(x['V_online_start']-x['V_online_all'])-(b['V_online_start']-b['V_online_all']);ratio=x['dqda_norm_start']/b['dqda_norm_start']
   d=x.get('critic/delta_abs');accepted=x.get('critic/accepted_abs')
   series.append(dict(updates=x['updates'],gap_increase=gap,gradient_ratio=ratio,gradient_cosine=x['dqda_cos_start'],action_shift=x['action_shift_start'],start_value=x['V_online_start'],start_target_value=x['V_target_start'],joint_anomaly=gap>=15 and ratio>=2,clip_fraction=x.get('critic/clipped_fraction'),delta_abs=d,accepted_abs=accepted,removed_abs_fraction=(1-accepted/d) if d else None))
  hits=[x['updates'] for x in series if x['joint_anomaly']]
  first_clip=next((x['updates'] for x in series if x['clip_fraction'] is not None and x['clip_fraction']>0),None)
  rows.append(dict(seed=seed,arm=arm,baseline=b,first_observed_clip=first_clip,anomaly_offsets=hits,series=series))
result=dict(scope='Existing 21 logged points per branch only; no new model inference/training/environment steps',gradient_location='Fixed fork Actor actions on fixed probe observations; Q values use CURRENT Actor actions',clipping_scope='Last minibatch of last fused4 call at each 5k snapshot; no spatial conditioning or all-update event history',ema_half_life_updates=math.log(.5)/math.log(1-.00005),rows=rows)
(root/'reports/curve_followup_evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('Analyzed',sum(len(x['series']) for x in rows),'existing records; no extra experiment')
