import pathlib,json,numpy as np,hashlib,datetime
R=pathlib.Path(__file__).resolve().parent;reports=[]
for name,ep in [('resource_pilot6_retry',6),('resource_contact2',2),('diagnostic_fixed5',5),('P1_sparse8',8),('P1_crowded8',8),('validation_P3_8',8)]:
 F=R/'runs'/name/'data';g=json.loads((F.parent/'guard_result.json').read_text());done=json.loads((F/'completed.json').read_text());assert g['exit_code']==0 and g['stop_reason'] is None
 z=np.load(F/'trajectory.npz');assert z['states'].shape[1]==ep and len(z['states'])==done['recorded_policy_steps']+1
 assert np.allclose(np.linalg.norm(z['states'][...,:2]-z['goal'],axis=-1),z['distances'],atol=1e-6)
 for key in ['states','goal','actions','eef','controller_goal','q','qd']:assert np.isfinite(z[key]).all(),(name,key)
 if name in ['diagnostic_fixed5','P1_sparse8','P1_crowded8','validation_P3_8']:assert done['recorded_policy_steps']==200
 row={'job':name,'episodes':ep,'steps':done['recorded_policy_steps'],'exit_code':g['exit_code'],'finite_states_actions':True,'metric_recomputed':True,'environment_sha256':hashlib.sha256((F/'environment_config.json').read_bytes()).hexdigest()}
 reports.append(row)
for n in ['P1','validation']:
 freeze=json.loads((R/('freeze_'+n+'.json')).read_text())
 for name,digest in freeze['source_and_scenes_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,(n,name)
F=R/'runs/validation_P3_8/data';z=np.load(F/'trajectory.npz');proposal=np.load(F/'proposed_actions.npz')['actions'];assignment=json.loads((R/'results/validation_assignment.json').read_text());logs=json.loads((F/'interventions.json').read_text());assert len(logs)==8
for s in assignment:
 i=s['slot'];scale=.5 if s['arm']=='B1' else 1.;assert np.allclose(z['actions'][:,i],proposal[:,i]*scale,atol=1e-8)
# Colour identities and requested scene geometry agree in each matched pair.
a0=np.load(R/'runs/P1_sparse8/data/initial.npz');a1=np.load(R/'runs/P1_crowded8/data/initial.npz');assert np.allclose(a0['colors'],a1['colors'])
vi=np.load(F/'initial.npz')
for group in range(8,12):
    i=[s['slot'] for s in assignment if s['group']==group and s['arm']=='B0'][0];j=[s['slot'] for s in assignment if s['group']==group and s['arm']=='B1'][0]
    assert np.allclose(vi['colors'][i],vi['colors'][j])
    # Worker prereg gate is <=1mm actual-vs-requested per scene; pair bound follows triangle inequality.
    assert np.allclose(vi['goal'][i],vi['goal'][j],atol=.002)
    assert np.allclose(vi['initial'][i,:,:2],vi['initial'][j,:,:2],atol=.001)
placement=np.load(F/'placement_check.npz');assert np.max(np.abs(placement['actual_goal']-placement['requested_goal']))<.001
assert np.max(np.abs(placement['actual_initial'][...,:2]-placement['requested_initial']))<.001
pair_errors=[]
for group in range(8,12):
    i=[s['slot'] for s in assignment if s['group']==group and s['arm']=='B0'][0];j=[s['slot'] for s in assignment if s['group']==group and s['arm']=='B1'][0]
    assert np.allclose(placement['requested_goal'][i],placement['requested_goal'][j],atol=1e-12)
    pair_errors.append({'group':group,'goal_max_coordinate_difference_m':float(np.abs(vi['goal'][i]-vi['goal'][j]).max()),'initial_xy_max_coordinate_difference_m':float(np.abs(vi['initial'][i,:,:2]-vi['initial'][j,:,:2]).max())})
(R/'results/pairing_actual_errors.json').write_text(json.dumps({'per_scene_preregistered_placement_tolerance_m':.001,'paired_goal_bound_from_triangle_inequality_m':.002,'explanation':'Additional validation initially imposed an unfrozen pair 1mm bound. Corrected to verify frozen 1mm actual-vs-requested gate directly; no run or protocol tolerance was changed.','pairs':pair_errors},indent=2))
ph=np.load(F/'physical.npz');assert np.abs(ph['net_force']).max()>.05
resource=json.loads((R/'results/resource_summary.json').read_text());assert resource['gpu_hours']<2 and resource['aggregate_memory_peak_mib']<12288
P1=sum(json.loads((R/'runs'/x/'guard_result.json').read_text())['charged_gpu_seconds']+2 for x in ['P1_sparse8','P1_crowded8']);assert P1<=2340
result={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':reports,'P1_phase_gpu_seconds':P1,'actual_damping_scale_verified_all_200_steps':True,'frozen_source_hashes_match':True,'formal_complete_episodes':24,'fixed_diagnostic_complete_episodes':5,'all_formal_full_200_steps':True,'models_frozen':True,'no_policy_training':True,'limitations':['P1 and diagnostic net contact forces unavailable (CC_NEVER); no pair identities in any phase.','P3 CC_LAST_SUBSTEP readout omits first internal substep.','Identical scene geometry and policy seed do not imply restored same state.']}
(R/'results/final_validation.json').write_text(json.dumps(result,indent=2));print('FINAL_VALIDATION_PASSED')
