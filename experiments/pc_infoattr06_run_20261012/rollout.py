"""Full closed-loop recording under unchanged official task physics."""
from runtime import *
from common import no_progress,first_run
def events(st,dd,ee):
 return {'no_progress_ends':no_progress(st,dd),'first_fall':first_run((st[...,2]<.975).any(-1)),'eef_path_m':float(np.linalg.norm(np.diff(ee,axis=0),axis=-1).sum()),'net_eef_displacement_m':float(np.linalg.norm(ee[-1]-ee[0]))}
import argparse
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--scene-seed',type=int,required=True);p.add_argument('--policy-seed',type=int,required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--role',required=True);p.add_argument('--arm',choices=['normal','permuted','blocked'],default='normal');a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();rt=Runtime(out)
seed(a.scene_seed)
cfg=yaml.safe_load((REPO/'env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml').read_text());cfg['env'].update(numEnvs=1,numObjects=a.n,episodeLength=200)
main=yaml.safe_load((REPO/'env_config/generalization_num_cubes/Config.yaml').read_text())
raw=IsaacPandaPush(cfg=cfg,rl_device=device,sim_device=device,graphics_device_id=gpu,headless=True,virtual_screen_capture=False,force_render=False)
env=IsaacPandaPushGoalSB3Wrapper(raw,obs_mode='dlp',n_views=2,latent_rep_model=rt.encoder,latent_classifier=None,reward_cfg=main['Reward']['GT'],smorl=False)
seed(a.scene_seed);ob=env.reset();goal=np.asarray(env.goal_pos)[0].copy()
def state():return np.stack([v.detach().cpu().numpy()[0].copy() for v in raw._obj_state_dict.values()])
def read():return {'states':state(),'eef':raw.states['eef_pos'].detach().cpu().numpy()[0].copy(),'body_state':raw._rigid_body_state.detach().cpu().numpy()[0].copy(),'q':raw._q.detach().cpu().numpy()[0].copy(),'qd':raw._qd.detach().cpu().numpy()[0].copy()}
hist={k:[] for k in ['states','eef','body_state','q','qd','distances','RGB','used_current_particles','used_goal_particles','generated_actions','actions','controller_goal']}
def append_state():
 z=read()
 for k,v in z.items():hist[k].append(v)
 hist['distances'].append(np.linalg.norm(z['states'][:,:2]-goal,axis=-1))
append_state();np.savez_compressed(out/'initial.npz',states=hist['states'][0],goal=goal,goal_image=env.goal_image[0],current_image=raw.image_buf.detach().cpu().numpy()[0],current_particles=ob['achieved_goal'][0],goal_particles=ob['desired_goal'][0])
bm=[]
for actor in range(raw.gym.get_actor_count(raw.envs[0])):
 for local,name in enumerate(raw.gym.get_actor_rigid_body_names(raw.envs[0],actor)):bm.append({'actor':actor,'name':name,'index_env':raw.gym.get_actor_rigid_body_index(raw.envs[0],actor,local,gymapi.DOMAIN_ENV)})
(out/'config.json').write_text(json.dumps({'args':vars(a),'environment':cfg,'body_map':bm,'task_physics':'official untouched; no isolated goal-render flags','initial_sha256':sha(out/'initial.npz')},indent=2))
schedule=None
if a.arm!='normal':
 pool=np.random.RandomState(a.policy_seed).randint(1,2**31-1,25).tolist();schedule=np.repeat(pool,8)
 if a.arm=='permuted':schedule=np.random.RandomState(a.policy_seed+1).permutation(schedule)
 assert len(schedule)==200
(out/'noise_schedule.json').write_text(json.dumps({'arm':a.arm,'whole_call_seed_schedule':schedule.tolist() if schedule is not None else None,'same_seed_multiset_across_paired_arms':a.arm!='normal'},indent=2))
seed(a.policy_seed);sampletimes=[]
for t in range(a.steps):
 cur=np.array(ob['achieved_goal'][0],copy=True);gg=np.array(ob['desired_goal'][0],copy=True)
 hist['RGB'].append(raw.image_buf.detach().cpu().numpy()[0,:,:3].copy());hist['used_current_particles'].append(cur);hist['used_goal_particles'].append(gg)
 act,receipt=rt.sample(cur,gg,'step%03d'%t,None if schedule is None else int(schedule[t]));sampletimes.append(receipt['elapsed_s'])
 h=hashlib.sha256();nz=np.load(receipt['noise_path'])
 for key in sorted(nz.files):h.update(nz[key].tobytes())
 receipt['noise_content_sha256']=h.hexdigest();receipt['generated_scope']='first XYZ action executed; remaining H5 proposals never executed';receipt['executed_first_action']=act[0].tolist()
 hist['generated_actions'].append(act);hist['actions'].append(act[0]);ob,rew,done,info=env.step(act[0][None])
 hist['controller_goal'].append(raw.controller.goal_pos.detach().cpu().numpy()[0].copy());append_state()
 dd=hist['distances'][-1]
 assert np.allclose(float((dd<.04).mean()),info[0]['goal_success_frac'])
 if (t+1)%20==0:print('CONTROL_STEPS',t+1,'wall',time.monotonic()-start,flush=True)
(out/'sampling_index.json').write_text(json.dumps(rt.calls,indent=2))
np.savez_compressed(out/'trajectory.npz',goal=goal,**{k:np.array(v) for k,v in hist.items()})
st=np.array(hist['states']);dd=np.array(hist['distances']);ee=np.array(hist['eef'])
ev=events(st,dd,ee)
(out/'events.json').write_text(json.dumps(ev,indent=2))
receipt={'utc':utc(),'role':a.role,'n':a.n,'arm':a.arm,'scene_seed':a.scene_seed,'policy_seed':a.policy_seed,'policy_steps':a.steps,'full_episodes':int(a.steps==200),'partial_episodes':int(a.steps!=200),'model_calls':len(rt.calls),'wall_s':time.monotonic()-start,'sample_wall_s':sampletimes,'actual_RGB_and_DLP_every_control':True,'full_H5_actions_and_101_noise_draws':True,'first_action_only_executed':True,'collision_pose_every_control':True,'goal_fraction':float((dd[-1]<.04).mean()),'all_success':bool((dd[-1]<.04).all()),'mean_goal_distance_m':float(dd[-1].mean()),'event_scope':'all endpoints include fallen objects; no contact counterparts','trajectory_path':str(out/'trajectory.npz'),'trajectory_sha256':sha(out/'trajectory.npz')}
(out/'completed.json').write_text(json.dumps(receipt,indent=2));raw.gym.destroy_sim(raw.sim)
print('RECORDING_COMPLETE',json.dumps({k:receipt[k] for k in ['role','n','policy_steps','wall_s','full_episodes']}),flush=True)

