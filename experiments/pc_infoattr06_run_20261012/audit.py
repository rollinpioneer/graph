"""Meaningful recorder, sampling, physics, and frozen-code checks; CPU only."""
import os,pathlib,json,hashlib,numpy as np
R=pathlib.Path(os.environ['R']);P=pathlib.Path(os.environ['P']);checks=[]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
freeze=json.loads((R/'receipts/normal_freeze.json').read_text())
for f,h in freeze['source_sha256'].items():assert sha(R/f)==h,('frozen source changed',f)
for plan in freeze['plans']:
 d=R/'runs'/plan['name']/'data';guard=json.loads((d.parent/'guard_result.json').read_text());done=json.loads((d/'completed.json').read_text());cfg=json.loads((d/'config.json').read_text());idx=json.loads((d/'sampling_index.json').read_text());schedule=json.loads((d/'noise_schedule.json').read_text())
 assert guard['exit_code']==0 and guard['stop_reason'] is None
 assert done['policy_steps']==200 and done['full_episodes']==1 and done['arm']=='normal';assert len(idx)==200 and schedule['whole_call_seed_schedule'] is None
 assert all(x['seed'] is None and x['batch']==1 and x['slot']==0 and x['noise_draws']==101 for x in idx)
 assert len(set(x['noise_content_sha256'] for x in idx))==200
 prior=json.loads((P/'runs'/('supplement%d'%plan['n'])/'data/config.json').read_text());assert cfg['environment']==prior['environment']
 with np.load(d/'trajectory.npz') as z,np.load(d/'initial.npz') as init:
  assert len(z['states'])==201 and len(z['eef'])==201 and len(z['q'])==201 and len(z['RGB'])==200 and len(z['actions'])==200
  for k in z.files:assert np.isfinite(z[k]).all(),k
  assert np.array_equal(z['used_current_particles'][0],init['current_particles'])
  assert np.array_equal(z['used_goal_particles'][0],init['goal_particles'])
  assert np.array_equal(z['RGB'][0],init['current_image'][:,:3])
  expected=np.clip(z['eef'][:-1]+np.clip(z['actions'],-1,1)*.125,[-.2675,-.3175,1.045],[.2675,.3175,1.06]);error=float(np.abs(expected-z['controller_goal']).max());assert error<1e-6
  assert np.array_equal(z['actions'],z['generated_actions'][:,0])
  checks.append({'run':plan['name'],'complete':True,'unchanged_official_physics':True,'first_saved_input_matches':True,'200_unique_noise_streams_101draws':True,'selected_batch_slot':[1,0],'controller_target_transform_error_m':error,'final_fraction_all_falls_retained':done['goal_fraction'],'tensor_sha256':sha(d/'trajectory.npz')})
(R/'receipts/recording_audit.json').write_text(json.dumps({'checks':checks,'all_passed':True,'sources_frozen':True,'offline_additional_policy_calls':0,'training_steps':0},indent=2));print('RECORDING_AUDIT_PASS',len(checks),flush=True)
