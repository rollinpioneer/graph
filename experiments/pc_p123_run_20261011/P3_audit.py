import pathlib,json,numpy as np
from prepare import events
R=pathlib.Path(__file__).resolve().parent;F=R/'runs/diagnostic_fixed5/data';z=np.load(F/'trajectory.npz');ph=np.load(F/'physical.npz');scenes=json.loads((R/'results/diagnostic_scenes.json').read_text());meta=json.loads((F/'contact_source.json').read_text());cubeids=[b['index_env'] for b in meta['body_map'] if b['actor']>=3]
rows=[]
for i,s in enumerate(scenes):
 t=events(z['states'][:,i],z['distances'][:,i],z['eef'][:,i])['first_fall']
 if t is None:continue
 j=int(np.argmin(z['states'][t,i,:,2]));lo=max(0,t-25);mask=ph['policy_step']<=t;body=ph['body_state'][mask,i,cubeids[j]];f=ph['net_force'][mask,i,cubeids[j]];tt=ph['time'][mask]
 quat=body[:,3:7];x,y,zq,w=quat.T
 rot=np.stack([1-2*(y*y+zq*zq),2*(x*y-zq*w),2*(x*zq+y*w),2*(x*y+zq*w),1-2*(x*x+zq*zq),2*(y*zq-x*w),2*(x*zq-y*w),2*(y*zq+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
 half=np.abs(rot).sum(-1)*.0175;edge=np.minimum(.25-np.abs(body[:,0])-half[:,0],.30-np.abs(body[:,1])-half[:,1]);cross=np.flatnonzero(edge<0);last=int(cross[0]) if len(cross) else len(body)-1
 pre_start=max(0,last-500);speed=np.linalg.norm(body[pre_start:last+1,7:10],axis=-1);lateral=np.linalg.norm(f[pre_start:last+1,:2],axis=-1)
 rows.append({'slot':i,'original_condition':s['condition'],'original_group':s['group'],'object_id':j,'first_fall_detection':t,'window_start_step':lo,'physical_samples':len(tt),'peak_object_speed_m_s':float(speed.max()),'mean_object_speed_m_s':float(speed.mean()),'net_contact_force_usable':False,'first_bbox_cross_physical_time':float(tt[last]),'pre_cross_samples':last-pre_start+1,'pre_cross_velocity_xyz':body[pre_start:last+1,7:10].tolist(),'signed_edge_margin_m':edge[pre_start:last+1].tolist(),'action_abs_max':float(np.abs(z['actions'][lo:t,i]).max()),'mechanism':'unknown/mixed: no contact counterpart available','evidence':'Velocity before first bounding-box crossing is measured; CC_NEVER makes net contacts unavailable. Cannot distinguish direct versus secondary contacts.'})
(R/'results/P3_diagnostic_falls.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
