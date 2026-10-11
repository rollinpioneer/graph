"""Read-only official frozen models; paired geometry and empirical snapshot gate."""
import os, sys, json, pathlib, time, argparse, random, copy
ROOT=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path('/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011')
REPO=OLD/'EC-Diffuser'; RES=OLD/'resources'
sys.path[:0]=[str(REPO),str(REPO/'diffuser'),str(ROOT)];os.chdir(REPO)
from isaacgym import gymapi,gymtorch
import numpy as np
import torch,yaml
from prepare import metrics,events
torch.set_num_threads(4)
gpu=int(os.environ['STAGE5_GPU_INDEX']);device='cuda:%d'%gpu
torch.cuda.set_device(gpu)
torch.cuda.set_per_process_memory_fraction(7.5*1024**3/torch.cuda.get_device_properties(gpu).total_memory,gpu)
import diffuser.utils as u
from diffuser.sampling import GoalConditionedPolicy
from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper
from dlp_utils import load_pretrained_rep_model
os.environ.pop('MESA_VK_DEVICE_SELECT',None)
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,required=True)
p.add_argument('--mode',choices=['pilot','calibrate','A'],required=True);p.add_argument('--condition');p.add_argument('--seed',type=int,required=True)
p.add_argument('--steps',type=int,required=True);p.add_argument('--parallel',type=int,default=4);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
def dump(name,obj): (out/name).write_text(json.dumps(obj,indent=2))
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
seed(a.seed);u.set_global_device(device);dump('config.json',vars(a))
load=RES/'pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100'
dc=u.load_config(str(load),'dataset_config.pkl');dc._dict['dataset_path']=str(RES/'push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl')
dataset=dc(seed=a.seed);normalizer=dataset.normalizer;del dataset
mc=u.load_config(str(load),'model_config.pkl');fc=u.load_config(str(load),'diffusion_config.pkl')
mc._device=device;fc._device=device;model=mc();diffusion=fc(model)
ckpt=torch.load(load/'state_1200000.pt',map_location='cpu');diffusion.load_state_dict(ckpt['ema'],strict=True);del ckpt
diffusion.eval();policy=GoalConditionedPolicy(diffusion,normalizer,[],horizon=5)
encoder=load_pretrained_rep_model(str(RES/'latent_rep_chkpts/dlp_push_6C'),pandapush=True).to(device).eval()
cfg=yaml.safe_load((REPO/'env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml').read_text())
cfg['env']['numEnvs']=a.parallel;cfg['env']['numObjects']=a.n;cfg['env']['episodeLength']=200
main=yaml.safe_load((REPO/'env_config/generalization_num_cubes/Config.yaml').read_text());dump('environment_config.json',cfg)
raw=IsaacPandaPush(cfg=cfg,rl_device=device,sim_device=device,graphics_device_id=gpu,headless=True,virtual_screen_capture=False,force_render=False)
env=IsaacPandaPushGoalSB3Wrapper(raw,obs_mode='dlp',n_views=2,latent_rep_model=encoder,latent_classifier=None,reward_cfg=main['Reward']['GT'],smorl=False)
scenes=[]
if (ROOT/'results/scenes.json').exists():
    scenes=json.loads((ROOT/'results/scenes.json').read_text())
    if a.mode=='A':scenes=[s for s in scenes if s['role']=='validation_A' and s['condition']==a.condition]
    elif a.mode=='calibrate':scenes=[s for s in scenes if s['role']=='calibration' and s['n']==a.n and s['condition']=='6_crowded']
    else:scenes=[s for s in scenes if s['role']=='calibration' and s['n']==a.n]
    if a.mode!='A' and len(scenes)!=a.parallel:scenes=[]
if scenes:
    assert len(scenes)==a.parallel
    original_reset=raw.reset
    def colored_reset(*args,**kwargs):
        obs=original_reset(*args,**kwargs)
        if raw.goal_reset:
            for i,s in enumerate(scenes):
                for j,key in enumerate(raw._obj_id_dict):
                    raw.gym.set_rigid_body_color(raw.envs[i],raw._obj_id_dict[key],0,gymapi.MESH_VISUAL,raw.object_colors[s['colors'][j]])
            raw.gym.step_graphics(raw.sim);raw.compute_observations()
            raw.obs_dict['obs']=torch.clamp(raw.obs_buf,-raw.clip_obs,raw.clip_obs).to(raw.rl_device)
            raw.obs_dict['media']=raw.image_buf.to(raw.rl_device)
            obs=raw.obs_dict
        return obs
    raw.reset=colored_reset
    initial={key:np.array([s['initial_xy'][j] for s in scenes]) for j,key in enumerate(raw._obj_state_dict)}
    desired={key:np.array([s['goal_xy'][j] for s in scenes]) for j,key in enumerate(raw._obj_state_dict)}
    seed(a.seed);obs=env.reset(set_init_states=initial,set_goal_states=desired)
else:
    assert a.mode!='A','Cannot run A without matched scenes'
    seed(a.seed);obs=env.reset()
goal=np.asarray(env.goal_pos).copy()
def states():return np.stack([v.detach().cpu().numpy().copy() for v in raw._obj_state_dict.values()],axis=1)
z=np.zeros((a.parallel,a.n,2),dtype=np.float32)
for fraction,xy in [(1.,z),(0.,z+.08),((a.n-1)/a.n,np.concatenate([z[:,:1]+.08,z[:,1:]],axis=1))]:
    got,_,_,_=env.check_success(None,None,{'position':xy,'goal_pos':z});assert np.allclose(got,fraction)
initial=states();measured=[]
np.savez_compressed(out/'placement_check.npz',actual_initial=initial,actual_goal=goal,
    requested_initial=np.array([s['initial_xy'] for s in scenes]) if scenes else np.zeros(0),
    requested_goal=np.array([s['goal_xy'] for s in scenes]) if scenes else np.zeros(0),eef=raw.states['eef_pos'].cpu().numpy().copy())
if scenes:
    for i,s in enumerate(scenes):
        initial_error=float(np.max(np.abs(initial[i,:,:2]-np.array(s['initial_xy']))))
        goal_error=float(np.max(np.abs(goal[i]-np.array(s['goal_xy']))))
        assert initial_error<.001,('placement initial error',i,initial_error,initial[i,:,:2].tolist(),s['initial_xy'])
        assert goal_error<.001,('placement goal error',i,goal_error,goal[i].tolist(),s['goal_xy'])
        measured.append({'group':s['group'],'condition':s['condition'],'actual_metrics':metrics(initial[i,:,:2],goal[i])})
dump('geometry_actual.json',measured)
colors=[]
for ep in raw.envs:
    row=[]
    for handle in raw._obj_id_dict.values():
        c=raw.gym.get_rigid_body_color(ep,handle,0,gymapi.MESH_VISUAL);row.append([c.x,c.y,c.z])
    colors.append(row)
np.savez_compressed(out/'initial.npz',initial=initial,goal=goal,colors=np.array(colors),initial_image=obs['observation'],goal_image=env.goal_image,
                    initial_latent=obs['achieved_goal'],goal_latent=obs['desired_goal'],q=raw._q.cpu().numpy(),qd=raw._qd.cpu().numpy())
dump('measurement_selfcheck.json',{'metric':'strict XY L2<0.04 identity matching','synthetic_tests':3,'threshold_m':env.dist_threshold})
history={'states':[initial],'distances':[np.linalg.norm(initial[...,:2]-goal,axis=-1)],'eef':[raw.states['eef_pos'].cpu().numpy().copy()],
         'q':[raw._q.cpu().numpy().copy()],'qd':[raw._qd.cpu().numpy().copy()],'actions':[],'controller_goal':[],'images':[],'image_steps':[]}
def infer(ob):
    with torch.no_grad():
        ac,_=policy({0:ob['achieved_goal'].reshape(a.parallel,-1),4:ob['desired_goal'].reshape(a.parallel,-1)},batch_size=1,verbose=False)
    assert np.isfinite(ac).all();return ac
def step(ac,record=True):
    ob,_,_,info=env.step(ac)
    st=states();dd=np.linalg.norm(st[...,:2]-goal,axis=-1)
    assert np.allclose((dd<.04).mean(-1),[r['goal_success_frac'] for r in info])
    assert np.allclose(dd.mean(-1),[r['avg_obj_dist'] for r in info])
    if record:
        history['states'].append(st);history['distances'].append(dd);history['eef'].append(raw.states['eef_pos'].cpu().numpy().copy())
        history['q'].append(raw._q.cpu().numpy().copy());history['qd'].append(raw._qd.cpu().numpy().copy())
        history['actions'].append(ac.copy());history['controller_goal'].append(raw.controller.goal_pos.detach().cpu().numpy().copy())
        t=len(history['actions'])
        if t==1 or t%10==0:
            history['images'].append(np.stack([r['image'] for r in info]));history['image_steps'].append(t)
    return ob,st,dd

tensor_names=['_root_state','_dof_state','_rigid_body_state','_j_eef','_mm','_pos_control','_effort_control','progress_buf','reset_buf','timeout_buf','actions']
def snapshot(ob):
    return {'raw':{k:getattr(raw,k).clone() for k in tensor_names if isinstance(getattr(raw,k,None),torch.Tensor)},
            'controller':copy.deepcopy(vars(raw.controller)),'obs':copy.deepcopy(ob),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'goal':goal.copy(),'simulation_time':raw.gym.get_sim_time(raw.sim)}
original_refresh=raw._refresh
restore_cache=[False]
def refreshed(*args,**kwargs):
    if restore_cache[0]:restore_cache[0]=False;return
    return original_refresh(*args,**kwargs)
raw._refresh=refreshed
def restore(s):
    for k,v in s['raw'].items():getattr(raw,k).copy_(v)
    vars(raw.controller).update(copy.deepcopy(s['controller']))
    raw.gym.set_actor_root_state_tensor(raw.sim,gymtorch.unwrap_tensor(raw._root_state))
    raw.gym.set_dof_state_tensor(raw.sim,gymtorch.unwrap_tensor(raw._dof_state))
    raw.gym.set_dof_position_target_tensor(raw.sim,gymtorch.unwrap_tensor(raw._pos_control))
    raw.gym.set_dof_actuation_force_tensor(raw.sim,gymtorch.unwrap_tensor(raw._effort_control))
    random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
    restore_cache[0]=True
    return copy.deepcopy(s['obs'])

print('LOADED',time.monotonic()-start,'mode',a.mode,flush=True)
snap=None
for t in range(1,a.steps+1):
    obs,_,dd=step(infer(obs))
    if a.mode=='calibrate' and t==20:
        snap=snapshot(obs);torch.save(snap,out/'snapshot.pt')
        dump('snapshot_fields.json',{'raw_tensors':list(snap['raw']),'controller_fields':list(snap['controller']),
            'rngs':['python','numpy','torch_CPU','torch_CUDA_all'], 'saved_simulation_time':snap['simulation_time'],
            'limitation':'PhysX contact warm-start and global simulation clock cannot be serialized; clock is recorded but not rewound. Gate tests short continuation empirically.'})
    if t%10==0:print(json.dumps({'t':t,'elapsed_s':time.monotonic()-start,'fraction':(dd<.04).mean(-1).tolist()}),flush=True)
    if a.mode=='A' and t in [100,200]:
        for i,s in enumerate(scenes):
            r={'group':s['group'],'condition':a.condition,'n':a.n,'seed':a.seed,'slot':i,'step':t,
               'fraction':float((dd[i]<.04).mean()),'success':bool((dd[i]<.04).all()),'mean_distance':float(dd[i].mean()),'max_distance':float(dd[i].max()),
               'common_three_fraction':float((dd[i,:3]<.04).mean())}
            with (out/'endpoints.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
if a.mode=='calibrate':
    assert a.steps==30 and snap is not None
    reference_states=np.array(history['states'][21:31]);reference_actions=np.array(history['actions'][20:30]);reference_q=np.array(history['q'][21:31])
    replay=[];replay_q=[];ob=restore(snap)
    for ac in reference_actions:
        ob,st,_=step(ac,False);replay.append(st);replay_q.append(raw._q.cpu().numpy().copy())
    closed=[];closed_q=[];closed_actions=[];ob=restore(snap)
    for t in range(10):
        ac=infer(ob);ob,st,_=step(ac,False);closed.append(st);closed_q.append(raw._q.cpu().numpy().copy());closed_actions.append(ac)
    position_replay=float(np.linalg.norm(np.array(replay)[...,:3]-reference_states[...,:3],axis=-1).max())
    position_closed=float(np.linalg.norm(np.array(closed)[...,:3]-reference_states[...,:3],axis=-1).max())
    action_error=float(np.abs(np.array(closed_actions)-reference_actions).max())
    q_error=float(np.abs(np.array(closed_q)-reference_q).max())
    gate={'passed':position_replay<=.0005 and position_closed<=.0005 and action_error<=.001 and q_error<=.005,
          'replay_max_position_error_m':position_replay,'closed_loop_max_position_error_m':position_closed,
          'closed_loop_max_action_error':action_error,'closed_loop_max_joint_error_rad':q_error,
          'frozen_acceptance':{'position_m':.0005,'action_abs':.001,'joint_rad':.005},
          'comparison_steps':10,'parallel_calibration_states':a.parallel,'model_calls':40,
          'limitation':'Empirical short-horizon gate; unobservable PhysX contacts and unreversed global clock remain limitations.'}
    dump('recovery_gate.json',gate)
    np.savez_compressed(out/'recovery_comparison.npz',reference_states=reference_states,reference_actions=reference_actions,
        replay_states=np.array(replay),closed_states=np.array(closed),closed_actions=np.array(closed_actions),reference_q=reference_q,closed_q=np.array(closed_q))
    print('RECOVERY_GATE',json.dumps(gate),flush=True)
np.savez_compressed(out/'trajectory.npz',goal=goal,**{k:np.array(v) for k,v in history.items()})
event_rows=[{'slot':i,'n':a.n,'events':events(np.array(history['states'])[:,i],np.array(history['distances'])[:,i],np.array(history['eef'])[:,i])} for i in range(a.parallel)]
dump('events.json',event_rows)
dump('completed.json',{'elapsed_seconds':time.monotonic()-start,'mode':a.mode,'episodes':a.parallel,'recorded_policy_steps':a.steps,
    'executed_control_steps_per_env':50 if a.mode=='calibrate' else a.steps,'model_calls':40 if a.mode=='calibrate' else a.steps,
    'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'model_parameters':sum(x.numel() for x in model.parameters()),
    'encoder_parameters':sum(x.numel() for x in encoder.parameters())})
print('COMPLETE',flush=True)
