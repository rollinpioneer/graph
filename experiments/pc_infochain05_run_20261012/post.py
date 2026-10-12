"""Finalize descriptive stage1, full recorded execution and one preregistered contrast."""
from common import *
import os,collections,itertools,time
begin=time.time();prev=pathlib.Path(os.environ['PREV'])
def read(p):return json.loads(pathlib.Path(p).read_text())
cases=read(R/'results/representative_cases.json');diag=read(R/'results/P2_space_time_diagnosis.json')
goals=sum([read(R/'runs'/('targets%d'%n)/'data/goal_test_results.json') for n in [3,6]],[])
noise=read(R/'runs/noise_formal/data/noise_test_results.json')
table('P1_goal_response.csv',[{'n':x['n'],'case_id':x['case_id'],'episode_id':x['episode_id'],'step':x['step'],'object_id':x['object_id'],'encoding_delta':x['goal_encoding_permutation_delta_G1_G2'],'CRN_H5_median':float(np.median(x['CRN_G1_G2_H5_action_L2_mean_per_seed'])),'within_goal_sampling_median':float(np.median(x['within_goal_H5_between_seed_L2'])),'response_to_sampling_ratio':x['effect_to_sampling_median_ratio'],'RGB_change_MSE':x['goal_image_pixel_mse_G1_G2'],'scope':'generatedH5, not executed, fixed-input capabilityprobe'} for x in goals])
table('P2_ABCD.csv',[{'n':x['n'],'case_id':x['case_id'],'episode_id':x['episode_id'],'steps':str(x['steps']),'A':x['A_numerical_maxabs'],'B':x['median_B'],'C':x['median_C'],'D':x['median_D'],'C_over_B':x['median_C']/max(x['median_B'],1e-12),'input_pixel_RMS':x['input_pixel_RMS']} for x in noise])
refs=read(prev/'results/P1_success_failure_references.json');ids={x['episode']['episode_id'] for x in cases};refs=[x for x in refs if x['failure'] in ids];dump('P2_external_descriptive_references.json',refs)
facts=[]
for c in cases:
 e=c['episode'];z=np.load(e['trajectory']);i=e['slot'];st=z['states'][:,i];dd=z['distances'][:,i];ee=z['eef'][:,i];ds=diag[c['case_id']];lo,hi=ds['window'];T=e['steps'];wins={'before':[max(0,lo-20),lo],'inside':[lo,hi],'after':[hi,min(T,hi+20)],'final':[max(0,T-20),T]}
 if ds['same_episode_prior_progress_window']:wins['prior_progress']=ds['same_episode_prior_progress_window']
 for phase,(a,b) in wins.items():
  if a==b:continue
  part=ee[a:b+1];path=float(np.linalg.norm(np.diff(part,axis=0),axis=-1).sum());net=float(np.linalg.norm(part[-1]-part[0]));pending=dd[a]>=.04;dist=np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1);acts=z['actions'][a:b,i];ctrl=z['controller_goal'][a:b,i] if 'controller_goal' in z else None
  expected=np.clip(ee[a:b]+np.clip(acts,-1,1)*.125,[-.2675,-.3175,1.045],[.2675,.3175,1.06])
  facts.append({'case_id':c['case_id'],'episode_id':e['episode_id'],'n':e['n'],'phase':phase,'window':[a,b],'EEF_path_m':path,'EEF_net_m':net,'R':net/(path+1e-9),'EEF_z_minmax_m':[float(part[:,2].min()),float(part[:,2].max())],'pending_objects':np.flatnonzero(pending).tolist(),'pending_center_min_m':float(dist[a:b+1,pending].min()) if pending.any() else None,'per_object_center_approach_net_m':(dist[a]-dist[b]).tolist(),'per_object_max_movement_m':np.linalg.norm(st[a:b+1,:,:3]-st[a,:,:3],axis=-1).max(0).tolist(),'per_object_goal_error_start_m':dd[a].tolist(),'per_object_goal_error_end_m':dd[b].tolist(),'per_object_best_progress_m':(dd[a]-dd[a:b+1].min(0)).tolist(),'action_mean':acts.mean(0).tolist(),'action_clip_fraction':float((np.abs(acts)>=1).mean()),'controller_transform_maxerror_m':float(np.abs(expected-ctrl).max()) if ctrl is not None else None,'actual_EEF_endstep_delta_mean_m':float(np.linalg.norm(np.diff(part,axis=0),axis=-1).mean()),'pose_scope':'actual logged grip pose if available; missing orientation not reconstructed','contact_scope':'unknown counterparts; center approach is descriptive'})
 z.close()
dump('P2_before_inside_after_facts.json',facts)
geom=read(R/'results/P2_gripper_collision_geometry.json');lite=[{k:v for k,v in q.items() if k!='samples'} for q in geom];dump('P2_collision_geometry_summary.json',lite)
dump('stage1_summary.json',{'UTC':utc(),'index_episodes':77,'representative_episodes':12,'saved_RGB_decompositions':len(read(R/'runs/offline_formal/data/decomposition_index.json')),'valid_goal_inputs':len(goals),'valid_goal_calls':144,'ABCD_inputs':6,'ABCD_calls':204,'all_A_max':max(x['A_numerical_maxabs'] for x in noise),'C_larger_than_B_inputs':sum(x['median_C']>x['median_B'] for x in noise),'goal_ratio_range':[min(x['effect_to_sampling_median_ratio'] for x in goals),max(x['effect_to_sampling_median_ratio'] for x in goals)],'goal_complete_insensitivity':'weakened in all6 fixed-input probes, no correctness or causal scaling claim','ROI_scope':'color/projection qualifies visible regions, reconstructed colors can be weak in both successes and failures; alpha is not object identity','geometry_available_cases':sum('evaluated' in q['status'] for q in geom),'min_signed_tip_box_gaps':[q.get('min_tip_cube_SAT_gap_m') for q in geom if 'evaluated' in q['status']],'geometry_readout':'positive OBB separation at sampledtimes but within5mm contactoffset; contact/counterparts/interstep/otherlinks unknown','selected_one_validation_path':'shortterm RNG correlation, only hypothesis screening; input effects often larger'})
# Full run summaries, complete failures/stops are always represented.
summary=[];windowrows=[]
for folder in sorted((R/'runs').iterdir()):
 cp=folder/'data/completed.json'
 if not cp.exists():continue
 c=read(cp)
 if c.get('role') not in ['supplement','validation','resource_pilot']:continue
 z=np.load(folder/'data/trajectory.npz');st=z['states'];dd=z['distances'];ee=z['eef'];actions=z['actions'];xy=actions[:,:2];norm=np.linalg.norm(xy,axis=-1)
 eligible=(norm[:-1]>=.05)&(norm[1:]>=.05);cos=(xy[:-1]*xy[1:]).sum(-1)/(norm[:-1]*norm[1:]+1e-12);reverse=eligible&(cos<0)
 bad=no_progress(st,dd);returning=[];gains=[]
 for t in range(20,len(ee)):
  a=t-20;path=float(np.linalg.norm(np.diff(ee[a:t+1],axis=0),axis=-1).sum());net=float(np.linalg.norm(ee[t]-ee[a]));Rr=net/(path+1e-9);pending=dd[a]>=.04;dist=np.linalg.norm(ee[a:t+1,None,:]-st[a:t+1,:,:3],axis=-1)
  gain=float((dist[0,pending]-dist[-1,pending]).max()) if pending.any() else None
  circ=path>=.03 and Rr<.3
  returning.append(circ);gains.append(gain)
  windowrows.append({'run':folder.name,'end_step':t,'R':Rr,'EEF_path_m':path,'circuitous':circ,'strict_no_progress':t in bad,'pending_center_approach_net_gain_max_m':gain,'per_object_best_goal_improvement_m':(dd[a]-dd[a:t+1].min(0)).tolist()})
 ctrl=z['controller_goal'];expected=np.clip(ee[:-1]+np.clip(actions,-1,1)*.125,[-.2675,-.3175,1.045],[.2675,.3175,1.06])
 ix=read(folder/'data/sampling_index.json');cfg=read(folder/'data/config.json');bm={x['name']:x['index_env'] for x in cfg['body_map']};pose=z['body_state'][:,bm['panda_grip_site'],:7]
 out={k:c[k] for k in ['role','n','arm','scene_seed','policy_seed','policy_steps','full_episodes','partial_episodes','model_calls','goal_fraction','all_success','mean_goal_distance_m','trajectory_path','trajectory_sha256']}
 out.update(run=folder.name,EEF_path_m=float(np.linalg.norm(np.diff(ee,axis=0),axis=-1).sum()),EEF_net_m=float(np.linalg.norm(ee[-1]-ee[0])),strict_no_progress_window_ends=len(bad),circuitous_windows=sum(returning),eligible_action_turns=int(eligible.sum()),XY_action_reversals=int(reverse.sum()),reversal_fraction=float(reverse.sum()/max(eligible.sum(),1)),controller_transform_maxerror_m=float(np.abs(ctrl-expected).max()),controller_EEF_offset_mean_m=float(np.linalg.norm(ctrl-ee[:-1],axis=-1).mean()),first_fall=first_run((st[...,2]<.975).any(-1)),per_object_final_error_m=dd[-1].tolist(),EEF_z_minmax_m=[float(ee[:,2].min()),float(ee[:,2].max())],actual_grip_pose_range=pose[:,:3].ptp(0).tolist(),whole_noise_content_multiset=dict(collections.Counter(x['noise_content_sha256'] for x in ix)),sampling101_draws_all=all(x['noise_draws']==101 for x in ix),actual_fields='RGB/current/goal DLP/H5/RNG/pose every control, firstactiononly executed')
 summary.append(out);z.close()
dump('new_run_summaries.json',summary);dump('new_full_window_facts.json',windowrows)
pairs=[]
for n in [3,6]:
 a=[x for x in summary if x['role']=='validation' and x['n']==n and x['arm']=='permuted'];b=[x for x in summary if x['role']=='validation' and x['n']==n and x['arm']=='blocked']
 if not a or not b:continue
 a,b=a[0],b[0];pa=R/'runs'/a['run']/'data/initial.npz';pb=R/'runs'/b['run']/'data/initial.npz';za=np.load(pa);zb=np.load(pb);errs={k:float(np.abs(za[k].astype(float)-zb[k].astype(float)).max()) for k in za.files}
 ta=np.load(R/'runs'/a['run']/'data/trajectory.npz');tb=np.load(R/'runs'/b['run']/'data/trajectory.npz')
 for k in ['eef','q','qd','body_state']:errs['actual_initial_'+k]=float(np.abs(ta[k][0]-tb[k][0]).max())
 ta.close();tb.close()
 gate=all(v<=1e-5 for k,v in errs.items() if 'image' not in k) and errs['current_image']==0 and errs['goal_image']==0 and a['whole_noise_content_multiset']==b['whole_noise_content_multiset']
 reduction=1-b['reversal_fraction']/max(a['reversal_fraction'],1e-12) if a['eligible_action_turns'] else None
 pairs.append({'n':n,'arms':[a['run'],b['run']],'initial_maxabs_errors':errs,'same_noise_multiset':a['whole_noise_content_multiset']==b['whole_noise_content_multiset'],'matched_gate':gate,'reversal_fraction_A_B':[a['reversal_fraction'],b['reversal_fraction']],'relative_reversal_reduction':reduction,'goal_fraction_A_B':[a['goal_fraction'],b['goal_fraction']],'mean_goal_error_A_B':[a['mean_goal_distance_m'],b['mean_goal_distance_m']],'circuitous_windows_A_B':[a['circuitous_windows'],b['circuitous_windows']],'strict_no_progress_ends_A_B':[a['strict_no_progress_window_ends'],b['strict_no_progress_window_ends']],'first_fall_A_B':[a['first_fall'],b['first_fall']],'scope':'one matched pair per n; exact noise multiset order intervention, no population/scaling estimate; action smoothing alone not explanatory'})
dump('validation_pair_results.json',pairs)
print('POST_COMPLETE',len(summary),'new run records',len(pairs),'pairs','CPUwall',time.time()-begin,flush=True)

