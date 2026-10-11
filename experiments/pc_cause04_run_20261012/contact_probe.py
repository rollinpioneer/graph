"""Bounded contact-readout selfcheck; no policy, no full task trajectory."""
import os,sys,json,pathlib,time,datetime
from isaacgym import gymapi,gymtorch
import torch,numpy as np,yaml
R=pathlib.Path(__file__).resolve().parent
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
gpu=int(os.environ['STAGE5_GPU_INDEX']);mode=sys.argv[2]
cfg=yaml.safe_load((pathlib.Path(os.environ['OLD'])/'EC-Diffuser/env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml').read_text())['sim']
gym=gymapi.acquire_gym();p=gymapi.SimParams()
p.dt=cfg['dt'];p.substeps=cfg['substeps'];p.up_axis=gymapi.UP_AXIS_Z;p.gravity=gymapi.Vec3(*cfg['gravity'])
p.use_gpu_pipeline=mode=='gpu'
for k,v in cfg['physx'].items():
 if k!='contact_collection':setattr(p.physx,k,v)
p.physx.contact_collection=gymapi.ContactCollection.CC_LAST_SUBSTEP
sim=gym.create_sim(gpu,-1,gymapi.SIM_PHYSX,p)
assert sim is not None
fixed=gymapi.AssetOptions();fixed.fix_base_link=True
table=gym.create_box(sim,.5,.6,.05,fixed)
cube=gym.create_box(sim,.035,.035,.035,gymapi.AssetOptions())
envs=[]
for i in range(2):
 e=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,2),2);envs.append(e)
 pose=gymapi.Transform();pose.p=gymapi.Vec3(0,0,1)
 a=gym.create_actor(e,table,pose,'table',i,0)
 pose.p=gymapi.Vec3(0,0,1.0425 if i==0 else 1.35)
 b=gym.create_actor(e,cube,pose,'cube',i,0)
gym.prepare_sim(sim)
force=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(2,2,3)
root=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(2,2,13)
rows=[];first_fields=None
for t in range(12):
 gym.simulate(sim);gym.fetch_results(sim,True)
 gym.refresh_net_contact_force_tensor(sim);gym.refresh_actor_root_state_tensor(sim)
 f=force.cpu().numpy().copy();z=root.cpu().numpy().copy()
 pairs=[];errors=[]
 for i,e in enumerate(envs):
  try:
   contacts=gym.get_env_rigid_contacts(e)
   if first_fields is None:first_fields=list(contacts.dtype.names or [])
   pairs.append([{k:np.asarray(c[k]).tolist() for k in (contacts.dtype.names or [])} for c in contacts])
  except Exception as ex:pairs.append([]);errors.append(str(ex))
 rows.append({'step':t+1,'time_s':float(gym.get_sim_time(sim)),'force':f.tolist(),'cube_z':z[:,1,2].tolist(),'pairs':pairs,'errors':errors})
(out/'samples.json').write_text(json.dumps(rows,indent=2))
positive=max(np.linalg.norm(np.array(x['force'])[0,1]) for x in rows)
negative=max(np.linalg.norm(np.array(x['force'])[1,1]) for x in rows)
pairpositive=sum(len(x['pairs'][0]) for x in rows);pairnegative=sum(len(x['pairs'][1]) for x in rows)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':mode,'sim_config_original':cfg,
'changes':{'contact_collection':[0,1],'use_gpu_pipeline':[True,p.use_gpu_pipeline],'scene':'isolated table/cube, no robot/cameras/policy'},
'positive':'cube on known table, net reaction checked','negative':'cube bottom remains above table through 12 simulation steps',
'positive_max_cube_net_force_N':float(positive),'negative_max_cube_net_force_N':float(negative),
'negative_min_cube_bottom_m':min(x['cube_z'][1]-.0175 for x in rows),'table_top_m':1.025,
'positive_contact_records':pairpositive,'negative_contact_records':pairnegative,'contact_record_fields':first_fields,
'time_resolution_s':p.dt,'internal_substeps':p.substeps,'coverage':'last internal substep only; transient coverage incomplete',
'pair_gate_local':bool(pairpositive>0 and pairnegative==0 and negative<1e-5),
'original_backend_gate':bool(mode=='gpu' and pairpositive>0 and pairnegative==0 and negative<1e-5),
'ordering':'events within same dt cannot be ordered; no substep timestamps',
'comparability':'minimal fixture only, full policy/control/rendering not tested on CPU pipeline'}
assert positive>.01 and negative<1e-5 and receipt['negative_min_cube_bottom_m']>1.025
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));gym.destroy_sim(sim)
print(json.dumps(receipt),flush=True)
