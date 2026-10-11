import pathlib,json,hashlib,datetime,numpy as np
from prepare import metrics
R=pathlib.Path(__file__).resolve().parent
pilot=R/'runs/resource_pilot6_retry/data';done=json.loads((pilot/'completed.json').read_text());z=np.load(pilot/'physical.npz')
assert done['recorded_policy_steps']==10 and done['episodes']==6
assert np.isfinite(z['net_force']).all() and np.isfinite(z['body_state']).all()
dt=np.diff(z['time']);assert (dt>=0).all() and (dt>0).sum()>=190
scenes=json.loads((R/'results/scenes_all.json').read_text())
for s in scenes:
 s['metrics']=metrics(np.array(s['initial_xy']),np.array(s['goal_xy']))
 lo,hi=(.075,.100) if s['condition']=='6_sparse' else (.052,.068)
 for key in ['initial_nn_mean','initial_nn_min','goal_nn_mean','goal_nn_min']:assert lo<=s['metrics'][key]<=hi
(R/'results/scenes_all.json').write_text(json.dumps(scenes,indent=2))
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'P1','pairs':8,'episodes':16,'groups':list(range(8)),'policy_seed':951124,'steps':200,'parallel_per_condition':8,'conditions':['6_sparse','6_crowded'],'data_role':'independent layout replication; P1-B discovery only','geometry':'identity-wise edge and reach proxy errors <=0.030 m, identical transport vectors; density bands preserved; ring/grid mixture','pilot':'resource_pilot6_retry, success rates not inspected for allocation','pilot_setup_s':12.318902455997886,'pilot_10step_elapsed_s':done['elapsed_seconds'],'cost_projection':'8 env linear extrapolation of 6 env step cost: about 427 wall seconds, <=854 conservatively charged seconds per condition; reserve 1170 each','stage_cap_gpu_seconds':2340,'memory_stop_mib':11264,'total_cap_gpu_seconds':7200,'new_scene_seed':202611123,'frozen_controls':{'H':5,'denoising':100,'success_XY_strict_m':.04},'source_and_scenes_sha256':{}}
for n in ['evaluate.py','prepare.py','guard.py','launch_job.py','run_job.sh','results/scenes_all.json','results/geometry_gate.json','results/identity.json']:
 freeze['source_and_scenes_sha256'][n]=hashlib.sha256((R/n).read_bytes()).hexdigest()
(R/'freeze_P1.json').write_text(json.dumps(freeze,indent=2))
(R/'results/measurement_validation.json').write_text(json.dumps({'passed':True,'physical_samples':len(z['time']),'positive_time_increments':int((dt>0).sum()),'median_positive_dt':float(np.median(dt[dt>0])),'net_force_finite':True,'n_bodies':z['net_force'].shape[2],'pair_contacts_supported':False},indent=2))
print('FROZEN_P1',freeze['pairs'],freeze['cost_projection'])
