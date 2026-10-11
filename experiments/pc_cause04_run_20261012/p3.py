"""P3 first crossings/falls; pre-crossing velocity only, no contact propagation labels."""
from common import *
import time
begin=time.time();es=json.loads((R/'results/episode_index.json').read_text());rows=[];cache={};physcache={}
for e in es:
 f=e['trajectory']
 if f not in cache:cache[f]=np.load(f)
 z=cache[f];i=e['slot'];st=z['states'][:,i];T=e['steps']
 physical=pathlib.Path(f).parent/'physical.npz';meta=pathlib.Path(f).parent/'contact_source.json'
 if physical.exists() and str(physical) not in physcache:physcache[str(physical)]=np.load(physical)
 ph=physcache.get(str(physical));bodyids=[]
 if ph is not None:
  mm=json.loads(meta.read_text());bodyids=[b['index_env'] for b in mm['body_map'] if b['actor']>=3]
 for j in range(e['n']):
  cf=np.flatnonzero(bbox_margin(st[:,j])<0);fall=first_run(st[:,j,2]<.975)
  if not len(cf) and fall is None:continue
  cross=int(cf[0]) if len(cf) else None
  cut=cross if cross is not None else fall-4
  lo=max(0,cut-25);pre=st[lo:cut,j]
  action=z['actions'][lo:cut,i]
  outward=[]
  for k in range(lo,cut):
   ax=0 if .25-abs(st[k,j,0])<.30-abs(st[k,j,1]) else 1
   outward.append(float(z['actions'][k,i,ax]*np.sign(st[k,j,ax])))
  row={'episode_id':e['episode_id'],'protocol':e['protocol'],'n':e['n'],'layout':e['layout'],'slot':i,'object_id':j,
 'first_fall_detection':fall,'first_fall_threshold_run_start':fall-4 if fall is not None else None,
 'first_bbox_cross_control_step':cross,'control_cross_interval':[max(0,cross-1),cross] if cross is not None else None,
 'pre_cross_control_window':[lo,cut],'pre_cross_speed_peak_m_s':float(np.linalg.norm(pre[:,7:10],axis=-1).max()) if len(pre) else None,
 'pre_cross_velocity_xyz':pre[:,7:10].tolist(),'pre_cross_signed_bbox_margin_m':bbox_margin(pre).tolist(),
 'pre_cross_eef_xyz':z['eef'][lo:cut,i].tolist(),'pre_cross_actions':action.tolist(),'nearest_edge_outward_action_proxy':outward,
 'contact_collection':e['contact_collection'],'net_force_valid':e['contact_collection']==1,'physical_evidence':None,
 'mechanism':'unknown: no validated counterpart; direct/secondary/postcontact/mixed cannot be distinguished',
 'causal_claim':False,'image_index':image_refs(e,z,[lo,max(0,cut-10),cut,min(T,cut+10)])}
  if 'controller_goal' in z:row['pre_cross_controller_targets']=z['controller_goal'][lo:cut,i].tolist()
  if ph is not None:
   b=bodyids[j];body=ph['body_state'][:,i,b];m=bbox_margin(body);inds=np.flatnonzero(m<0)
   px=int(inds[0]) if len(inds) else None
   stop=px if px is not None else len(body);start=max(0,stop-500)
   # A physical sample after a control-level fall with no crossing is not causal evidence.
   if px is None:start=stop
   vv=body[start:stop,7:10];ff=ph['net_force'][start:stop,i,b]
   times=ph['time'];unique=np.unique(times);drops=int((np.diff(unique)>.01667*1.5).sum())
   duplicates=len(times)-len(unique)
   row['physical_evidence']={'source':str(physical),'body_index_env':b,'first_bbox_cross_sample':px,
   'first_bbox_cross_time_s':float(times[px]) if px is not None else None,
   'cross_policy_step':int(ph['policy_step'][px]) if px is not None else None,'strict_pre_cross_samples':len(vv),
   'pre_cross_speed_peak_m_s':float(np.linalg.norm(vv,axis=-1).max()) if len(vv) else None,
   'pre_cross_net_force_peak_N':float(np.linalg.norm(ff,axis=-1).max()) if len(ff) and e['contact_collection']==1 else None,
   'contact_counterpart':None,'time_source':'gym.get_sim_time at raw._refresh',
   'duplicate_time_samples':duplicates,'gaps_over_1_5_dt':drops,'coverage':'last internal substep; duplicate refresh timestamps retained; unknown transient losses'}
  rows.append(row)
dump('P3_first_crossings_and_falls.json',rows)
summary={'utc':utc(),'episodes_with_first_any_fall':sum(e['first_fall'] is not None for e in es),
'objects_with_fall':sum(r['first_fall_detection'] is not None for r in rows),
'objects_with_bbox_crossing':sum(r['first_bbox_cross_control_step'] is not None for r in rows),
'cross_without_fall_objects':sum(r['first_fall_detection'] is None for r in rows),'objects_with_physical_fields':sum(r['physical_evidence'] is not None for r in rows),
'usable_net_force_rows':sum(r['net_force_valid'] for r in rows),'validated_contact_paths':0,'interpretation':'pre-cross velocities/actions and net-force associations only; all counterpart mechanisms unknown',
'cpu_wall_seconds':time.time()-begin}
dump('P3_summary.json',summary);print(json.dumps(summary),flush=True)
