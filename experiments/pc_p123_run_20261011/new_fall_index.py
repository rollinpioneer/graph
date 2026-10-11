import pathlib,json,numpy as np
from prepare import events
R=pathlib.Path(__file__).resolve().parent;rows=[]
for name in ['diagnostic_fixed5','P1_sparse8','P1_crowded8','validation_P3_8']:
 F=R/'runs'/name/'data';z=np.load(F/'trajectory.npz');cfg=json.loads((F/'environment_config.json').read_text())
 for i in range(z['states'].shape[1]):
  t=events(z['states'][:,i],z['distances'][:,i],z['eef'][:,i])['first_fall']
  if t is None:continue
  j=int(z['states'][t,i,:,2].argmin());lo=max(0,t-25);steps=z['image_steps'];before=steps[steps<=max(1,t-4)];after=steps[steps>=t]
  rows.append({'job':name,'slot':i,'object_id':j,'first_fall_detection_step':t,'first_sustained_threshold_onset_step':t-4,'fixed_window':[lo,t],'state_xyz':z['states'][lo:t+1,i,j,:3].tolist(),'velocity_xyz':z['states'][lo:t+1,i,j,7:10].tolist(),'nearest_before_stored_image_step':int(before[-1]) if len(before) else 0,'nearest_after_stored_image_step':int(after[0]) if len(after) else 200,'images_are_control_10step_samples_not_exact_onset':True,'contact_force_available':cfg['sim']['physx']['contact_collection']==1,'contact_counterparts':False,'mechanism':'unknown/mixed','trajectory':str(F/'trajectory.npz'),'physical':str(F/'physical.npz')})
(R/'results/new_first_fall_index.json').write_text(json.dumps({'episodes':rows,'unknown_fraction':1.0},indent=2));print('NEW_FIRST_FALLS',len(rows))
