"""Read-only official frozen models; paired geometry and empirical snapshot gate."""
import os, sys, json, pathlib, time, argparse, random, copy, hashlib
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
p.add_argument('--mode',choices=['pilot','diagnostic','P1','validation'],required=True);p.add_argument('--condition');p.add_argument('--seed',type=int,required=True)
p.add_argument('--steps',type=int,required=True);p.add_argument('--parallel',type=int,default=4);p.add_argument('--scene-file',default='scenes_all.json');p.add_argument('--groups',default='12,13');p.add_argument('--arms',default='B0');a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
if a.mode in ['P1','validation']:
    frozen=json.loads((ROOT/('freeze_'+a.mode+'.json')).read_text())
    assert a.steps==200
    for relative,digest in frozen['source_and_scenes_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative

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
groups=[int(x) for x in a.groups.split(',')]
scenes=[s for s in json.loads((ROOT/'results'/a.scene_file).read_text()) if s['group'] in groups and s['condition']==a.condition]
arms=a.arms.split(',')
if a.mode=='validation':
    scenes=[dict(s,arm=arm) for arm in arms for s in scenes]
assert len(scenes)==a.parallel,(len(scenes),a.parallel)
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
net=gymtorch.wrap_tensor(raw.gym.acquire_net_contact_force_tensor(raw.sim)).view(a.parallel,-1,3)
body_names=raw.gym.get_actor_rigid_body_names(raw.envs[0],raw.frankas[0]) if hasattr(raw,'frankas') else []
body_map=[]
for actor in range(raw.gym.get_actor_count(raw.envs[0])):
    for local,name in enumerate(raw.gym.get_actor_rigid_body_names(raw.envs[0],actor)):
        body_map.append({'actor':actor,'name':name,'index_env':raw.gym.get_actor_rigid_body_index(raw.envs[0],actor,local,gymapi.DOMAIN_ENV)})
contacts_available=False
try:
    contacts=raw.gym.get_env_rigid_contacts(raw.envs[0]);contacts_available=True
except Exception as exc:
    contacts_reason=str(exc)
# Unsupported GPU pipeline pair API must not be invoked during formal run.
dump('contact_source.json',{'source':'net contact force tensor','counterpart_identifiable':False,'pair_api_initial_probe':contacts_available,'pair_probe_only_not_used':True,'sampling':'every official _refresh, including each controller/physics substep','sim_dt':cfg.get('sim',{}).get('dt'),'body_map':body_map})
physical={'time':[],'policy_step':[],'body_state':[],'net_force':[],'q':[],'qd':[],'effort':[]}
recording=[False];control_index=[0];refresh_official=raw._refresh
def measured_refresh(*args,**kwargs):
    answer=refresh_official(*args,**kwargs)
    if recording[0]:
        raw.gym.refresh_net_contact_force_tensor(raw.sim)
        physical['time'].append(raw.gym.get_sim_time(raw.sim));physical['policy_step'].append(control_index[0])
        physical['body_state'].append(raw._rigid_body_state.detach().cpu().numpy().copy())
        physical['net_force'].append(net.detach().cpu().numpy().copy())
        for key,attr in [('q','_q'),('qd','_qd'),('effort','_effort_control')]:physical[key].append(getattr(raw,attr).detach().cpu().numpy().copy())
    return answer
raw._refresh=measured_refresh
original_step=raw.pre_physics_step
# Baseline pre_physics_step remains untouched; measurement wraps only reads.
triggers=[None]*a.parallel
intervention_log=[]
proposed_actions=[]
def intervene(ac,t):
    proposed_actions.append(ac.copy());ex=ac.copy()
    if a.mode=='validation':
        rule=json.loads((ROOT/'intervention.json').read_text())
        for i,scene in enumerate(scenes):
            arm=scene.get('arm','B0')
            if triggers[i] is None and len(history['states'])>=21:
                st=np.array(history['states'][-21:])[:,i];dd=np.array(history['distances'][-21:])[:,i]
                motion=np.linalg.norm(st[...,:3]-st[0,:,:3],axis=-1).max()
                improvement=dd[0].mean()-dd.mean(-1).min()
                if not (st[...,2]<.975).any() and not (dd[-1]<.04).all() and motion<.001 and improvement<.001:
                    triggers[i]=t;intervention_log.append({'slot':i,'group':scene['group'],'arm':arm,'first_action_step':t,'detected_state_step':t-1})
            if triggers[i] is not None and 0<=t-triggers[i]<rule['duration_steps']:
                if arm=='B1':ex[i]=np.array(rule['B1_action'])
                elif arm=='B2':ex[i]=np.array(rule['B2_action'])
    return ex
def infer(ob):
    with torch.no_grad():
        ac,_=policy({0:ob['achieved_goal'].reshape(a.parallel,-1),4:ob['desired_goal'].reshape(a.parallel,-1)},batch_size=1,verbose=False)
    assert np.isfinite(ac).all();return ac
def step(ac,record=True):
    control_index[0]+=1;recording[0]=True
    ob,_,_,info=env.step(ac)
    recording[0]=False
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

print('LOADED',time.monotonic()-start,'mode',a.mode,flush=True)
snap=None
for t in range(1,a.steps+1):
    proposed=infer(obs)
    executed=intervene(proposed,t)
    obs,_,dd=step(executed)
    if t%10==0:print(json.dumps({'t':t,'elapsed_s':time.monotonic()-start}),flush=True)
    if a.mode in ['P1','validation','diagnostic'] and t in [100,200]:
        for i,s in enumerate(scenes):
            r={'arm':s.get('arm','B0'),'group':s['group'],'condition':a.condition,'n':a.n,'seed':a.seed,'slot':i,'step':t,
               'fraction':float((dd[i]<.04).mean()),'success':bool((dd[i]<.04).all()),'mean_distance':float(dd[i].mean()),'max_distance':float(dd[i].max()),
               'common_three_fraction':float((dd[i,:3]<.04).mean())}
            with (out/'endpoints.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
np.savez_compressed(out/'physical.npz',**{k:np.array(v) for k,v in physical.items()})
dump('interventions.json',intervention_log)
np.savez_compressed(out/'proposed_actions.npz',actions=np.array(proposed_actions))
np.savez_compressed(out/'trajectory.npz',goal=goal,**{k:np.array(v) for k,v in history.items()})
event_rows=[{'slot':i,'n':a.n,'events':events(np.array(history['states'])[:,i],np.array(history['distances'])[:,i],np.array(history['eef'])[:,i])} for i in range(a.parallel)]
dump('events.json',event_rows)
dump('completed.json',{'elapsed_seconds':time.monotonic()-start,'mode':a.mode,'episodes':a.parallel,'recorded_policy_steps':a.steps,
    'executed_control_steps_per_env':a.steps,'model_calls':a.steps,
    'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'model_parameters':sum(x.numel() for x in model.parameters()),
    'encoder_parameters':sum(x.numel() for x in encoder.parameters())})
print('COMPLETE',flush=True)
