import pathlib,json,numpy as np,contextlib,io
from prepare import events
with contextlib.redirect_stdout(io.StringIO()):
 from audit import chain
R=pathlib.Path(__file__).resolve().parent;rows=[]
for name in ['P1_sparse8','P1_crowded8']:
 F=R/'runs'/name/'data';z=np.load(F/'trajectory.npz');cfg=json.loads((F/'environment_config.json').read_text());table=cfg['env']['tableDims'];center=cfg['env']['tablePos'];cube=cfg['env']['cubeSize'];top=center[2]+table[2]/2;lower=[center[0]-table[0]/2-cube/2,center[1]-table[1]/2-cube/2,top+.02];upper=[center[0]+table[0]/2+cube/2,center[1]+table[1]/2+cube/2,top+cube]
 for i in range(8):
  ev=events(z['states'][:,i],z['distances'][:,i],z['eef'][:,i]);t=ev['first_no_progress']
  if t is None:continue
  c=chain(z,i,t);clipped=np.clip(z['actions'][t-20:t,i],-1,1);expected=z['eef'][t-20:t,i]+clipped*.125;expected=np.clip(expected,lower,upper);actual=z['controller_goal'][t-20:t,i]
  c['scaled_clamped_goal_vs_recorded_max_error_m']=float(np.abs(expected-actual).max());c['observation']='World xyz command reaches a changing OSC target and arm response, but objects do not advance. Counterpart contact data unavailable; no unique cause assigned.'
  rows.append({'condition':name,'group':i,'first_stall_detection_step':t,'chain':c,'source':str(F/'trajectory.npz'),'role':'P1-B discovery / additional descriptive P2 evidence only, not independent mechanism validation'})
(R/'results/new_stall_chain.json').write_text(json.dumps(rows,indent=2))
print('NEW_STALL_CHAINS',len(rows));print('TRANSFORM_MAX_ERRORS',[r['chain']['scaled_clamped_goal_vs_recorded_max_error_m'] for r in rows])
