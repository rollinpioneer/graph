"""Frozen official runtime, exact RNG/noise capture, original goal rendering."""
import os,sys,pathlib,json,time,pickle,hashlib,random
R=pathlib.Path(__file__).resolve().parent
OLD=pathlib.Path(os.environ['OLD']);REPO=OLD/'EC-Diffuser';RES=OLD/'resources'
sys.path[:0]=[str(REPO),str(REPO/'diffuser'),str(R)];os.chdir(REPO)
from isaacgym import gymapi,gymtorch
import numpy as np,torch,yaml
from common import sha,utc
import diffuser.utils as u
from diffuser.sampling import GoalConditionedPolicy
from dlp_utils import load_pretrained_rep_model,get_dlp_rep
from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper
gpu=int(os.environ['STAGE5_GPU_INDEX']);device='cuda:%d'%gpu
torch.cuda.set_device(gpu);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(gpu).total_memory,gpu)
def seed(s):
 random.seed(s);np.random.seed(s);torch.default_generator.manual_seed(s);torch.cuda.manual_seed(s)
def rng():
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda_selected':torch.cuda.get_rng_state(gpu),'gpu_index':gpu}
def restore(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch_cpu']);torch.cuda.set_rng_state(x['torch_cuda_selected'],gpu)
class Runtime:
 def __init__(self,out):
  self.out=pathlib.Path(out);self.out.mkdir(parents=True,exist_ok=True);seed(1205000);u.set_global_device(device)
  load=RES/'pretrained_models/panda_push/diffusion/3C_adalnpintlarge_dlp_randcolor_H5_T100'
  dc=u.load_config(str(load),'dataset_config.pkl');dc._dict['dataset_path']=str(RES/'push_cubes/3C_randcolor/panda_push_replay_buffer_dlp.pkl')
  dataset=dc(seed=1205000);self.normalizer=dataset.normalizer;del dataset
  mc=u.load_config(str(load),'model_config.pkl');fc=u.load_config(str(load),'diffusion_config.pkl')
  mc._device=device;fc._device=device;model=mc();self.diffusion=fc(model)
  ckpt=torch.load(load/'state_1200000.pt',map_location='cpu');self.diffusion.load_state_dict(ckpt['ema'],strict=True);del ckpt
  self.diffusion.eval();self.policy=GoalConditionedPolicy(self.diffusion,self.normalizer,[],horizon=5)
  self.encoder=load_pretrained_rep_model(str(RES/'latent_rep_chkpts/dlp_push_6C'),pandapush=True).to(device).eval()
  for p in self.encoder.parameters():p.requires_grad_(False)
  for p in self.diffusion.parameters():p.requires_grad_(False)
  self.envs={};self.calls=[];self.count=0
 def _encode_batch(self,im,decode=False):
  a=torch.as_tensor(np.ascontiguousarray(im),dtype=torch.float32,device=device)/255
  with torch.no_grad():
   enc=self.encoder.encode_all(a,deterministic=True);rep=get_dlp_rep(enc)
   dec=self.encoder.decode_all(rep[...,:2],rep[...,5:-1],enc['z_bg'],rep[...,-1],z_depth=rep[...,4:5],z_scale=rep[...,2:4]) if decode else None
  pack={k:enc[k].detach().cpu().numpy() for k in ['z','mu_scale','mu_depth','mu_features','obj_on','z_bg']}
  if dec is not None:
   pack.update({k:dec[k].detach().cpu().numpy() for k in ['rec','dec_objects_trans','alpha_masks']})
  return rep.detach().cpu().numpy(),pack
 def encode(self,im,decode=False):
  # Official wrapper processes each view separately with its own batch.
  rep=[];packs=[]
  for v in range(len(im)):
   a,b=self._encode_batch(im[v:v+1],decode);rep.append(a[0]);packs.append(b)
  return np.array(rep),{k:np.concatenate([p[k] for p in packs],axis=0) for k in packs[0]}
 def historical_encode(self,e,t,decode=False):
  if t is None or t==0:
   z=np.load(e['initial']);images=z['goal_image' if t is None else 'initial_image']
  else:
   z=np.load(e['trajectory']);idx=int(np.flatnonzero(z['image_steps']==t)[0]);images=z['images'][idx]
  rep=[];packs=[]
  for v in range(2):
   a,b=self._encode_batch(images[:,v],decode);rep.append(a[e['slot']]);packs.append({k:q[e['slot']:e['slot']+1] for k,q in b.items()})
  z.close()
  return np.array(rep),{k:np.concatenate([p[k] for p in packs],axis=0) for k in packs[0]}
 def sample(self,current,goal,label,reset_seed=None):
  if reset_seed is not None:seed(reset_seed)
  before=rng();self.count+=1;folder=self.out/'sampling';folder.mkdir(exist_ok=True);stem='%04d_%s'%(self.count,label)
  draws=[];types=[];orig_randn=torch.randn;orig_like=torch.randn_like
  def a(*args,**kwargs):
   x=orig_randn(*args,**kwargs);draws.append(x.detach().cpu().numpy().copy());types.append('randn');return x
  def b(*args,**kwargs):
   x=orig_like(*args,**kwargs);draws.append(x.detach().cpu().numpy().copy());types.append('randn_like');return x
  torch.randn=a;torch.randn_like=b;begin=time.monotonic()
  try:
   with torch.no_grad():action,tr=self.policy({0:current.reshape(1,-1),4:goal.reshape(1,-1)},batch_size=1,verbose=False)
   torch.cuda.synchronize(gpu)
  finally:torch.randn=orig_randn;torch.randn_like=orig_like
  after=rng()
  (folder/(stem+'_rng.pkl')).write_bytes(pickle.dumps({'before':before,'after':after},protocol=4))
  np.savez_compressed(folder/(stem+'_noise.npz'),**{'draw_%03d'%k:x for k,x in enumerate(draws)})
  np.savez_compressed(folder/(stem+'_generated.npz'),actions=tr.actions,generated_particles=tr.observations)
  record={'id':self.count,'label':label,'seed':reset_seed,'batch':1,'slot':0,'elapsed_s':time.monotonic()-begin,'noise_draws':len(draws),'draw_types':types,
  'noise_sha256':sha(folder/(stem+'_noise.npz')),'rng_sha256':sha(folder/(stem+'_rng.pkl')),'rng_path':str(folder/(stem+'_rng.pkl')),
  'noise_path':str(folder/(stem+'_noise.npz')),'generated_path':str(folder/(stem+'_generated.npz')),'actions':tr.actions[0].tolist(),
  'generated_scope':'H5 output; only model proposal, none executed; t0 random draw later zeroed by official sampler'}
  self.calls.append(record);(self.out/'sampling_index.json').write_text(json.dumps(self.calls,indent=2))
  assert np.isfinite(tr.actions).all() and len(draws)==101
  return tr.actions[0].copy(),record
 def render(self,case,goals):
  n=case['n']
  if n not in self.envs:
   cfg=yaml.safe_load((REPO/'env_config/generalization_num_cubes/IsaacPandaPushConfig.yaml').read_text())
   cfg['env'].update(numEnvs=1,numObjects=n,episodeLength=200)
   main=yaml.safe_load((REPO/'env_config/generalization_num_cubes/Config.yaml').read_text())
   raw=IsaacPandaPush(cfg=cfg,rl_device=device,sim_device=device,graphics_device_id=gpu,headless=True,virtual_screen_capture=False,force_render=False)
   env=IsaacPandaPushGoalSB3Wrapper(raw,obs_mode='dlp',n_views=2,latent_rep_model=self.encoder,latent_classifier=None,reward_cfg=main['Reward']['GT'],smorl=False)
   holder={'colors':None};official=raw.reset
   def colored(*args,**kwargs):
    obs=official(*args,**kwargs)
    if raw.goal_reset:
     for j,key in enumerate(raw._obj_id_dict):
      raw.gym.set_rigid_body_color(raw.envs[0],raw._obj_id_dict[key],0,gymapi.MESH_VISUAL,raw.object_colors[holder['colors'][j]])
     raw.gym.step_graphics(raw.sim);raw.compute_observations();raw.obs_dict['media']=raw.image_buf.to(raw.rl_device);obs=raw.obs_dict
    return obs
   raw.reset=colored;self.envs[n]=(raw,env,holder,cfg)
  raw,env,holder,cfg=self.envs[n];holder['colors']=case['colors'];keys=list(raw._obj_state_dict)
  # Goal-only fixture: official cameras/API, fixed retreat robot pose, no rollout.
  # Parking scene removes object-dependent retreat collisions.
  park=np.array([[x,y] for x in [.09,.14] for y in [-.16,-.08,0.]])[:n]
  seed(1205050);env.get_set_goal({key:park[None,j] for j,key in enumerate(keys)})
  fixed_dof=raw._dof_state.clone()
  base=raw._root_state.clone();image=[];actual=[];audit=[]
  ids=raw._global_indices[torch.tensor([0],device=device),-n:].flatten()
  for xy in goals:
   xy=np.array(xy);assert np.all(np.abs(xy)<np.array([.2325,.2825]))
   assert min(np.linalg.norm(xy[i]-xy[j]) for i in range(n) for j in range(i))>.035
   raw._root_state.copy_(base);raw._dof_state.copy_(fixed_dof)
   for j,key in enumerate(keys):
    z=raw._root_state[0,raw._obj_id_dict[key]]
    z[:2]=torch.as_tensor(xy[j],device=device);z[2]=1.025+.035/2
    z[3:7]=torch.tensor([0.,0.,0.,1.],device=device);z[7:]=0
   raw.gym.set_dof_state_tensor(raw.sim,gymtorch.unwrap_tensor(raw._dof_state))
   raw.gym.set_actor_root_state_tensor_indexed(raw.sim,gymtorch.unwrap_tensor(raw._root_state),gymtorch.unwrap_tensor(ids),len(ids))
   raw.gym.step_graphics(raw.sim);raw.compute_observations()
   measured=np.stack([v.detach().cpu().numpy()[0,:2] for v in raw._obj_state_dict.values()])
   assert np.abs(measured-xy).max()<1e-6
   assert torch.equal(raw._dof_state,fixed_dof)
   image.append(raw.image_buf.detach().cpu().numpy()[0,:,:3]);actual.append(measured.tolist())
   audit.append({'max_position_error_m':float(np.abs(measured-xy).max()),'robot_dof_identical':True,'physics_steps_after_placement':0})
  (self.out/('renderer_case_'+str(case['episode_id']).replace('/','_')+'.json')).write_text(json.dumps({'method':'official camera/render API; uniform render-only goal placement after fixed parking-scene retreat; not historical simulation snapshot','audit':audit,'fixed_robot_dof':fixed_dof.detach().cpu().numpy().tolist()},indent=2))
  return np.array(image),actual,cfg
 def close(self):
  for raw,env,holder,cfg in self.envs.values():raw.gym.destroy_sim(raw.sim)
