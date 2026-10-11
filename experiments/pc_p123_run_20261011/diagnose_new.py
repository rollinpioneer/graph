import pathlib,json,numpy as np
from prepare import events
from audit import chain
R=pathlib.Path(__file__).resolve().parent
F=R/'runs/diagnostic_fixed5/data'
if not (F/'completed.json').exists():raise SystemExit('DIAGNOSTIC_NOT_FINISHED')
z=np.load(F/'trajectory.npz');ph=np.load(F/'physical.npz');meta=json.loads((F/'contact_source.json').read_text());scenes=json.loads((R/'results/diagnostic_scenes.json').read_text())
bmap=meta['body_map'];cubeids=[b['index_env'] for b in bmap if b['actor']>=3];fingers=[b['index_env'] for b in bmap if 'finger' in b['name']]
print('BODY_MAPPING',bmap,'CUBE_IDS',cubeids,'FINGERS',fingers)
rows=[]
for i,s in enumerate(scenes):
 ev=events(z['states'][:,i],z['distances'][:,i],z['eef'][:,i]);row={'slot':i,'original_condition':s['condition'],'original_group':s['group'],'events':ev,'contact_pairs_available':False}
 t=ev['first_no_progress']
 if t is not None:
  row['chain']=chain(z,i,t);mask=(ph['policy_step']>t-20)&(ph['policy_step']<=t);force=ph['net_force'][mask,i];body=ph['body_state'][mask,i]
  row['physical_sample_count']=int(mask.sum());row['physical_time_range']=ph['time'][mask][[0,-1]].tolist()
  row['max_net_lateral_force_N_per_cube']=np.linalg.norm(force[:,cubeids,:2],axis=-1).max(0).tolist()
  row['net_lateral_force_gt_0_1N_samples_per_cube']=(np.linalg.norm(force[:,cubeids,:2],axis=-1)>.1).sum(0).tolist()
  row['body_vertical_range_m']={b['name']:[float(body[:,b['index_env'],2].min()),float(body[:,b['index_env'],2].max())] for b in bmap if 'finger' in b['name'] or b['index_env'] in cubeids}
  row['interpretation']='Arm responded while objects failed to progress; net forces cannot identify counterparts or prove missing contact.'
 rows.append(row)
(R/'results/P2_P3_diagnostic_rerun.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
