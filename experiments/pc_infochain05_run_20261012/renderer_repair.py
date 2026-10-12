from pathlib import Path
import os,json,hashlib,datetime,shutil
r=Path(os.environ['ROUND'])
for f in ['runtime.py','probe.py']:
 shutil.copy2(r/f,r/'receipts'/('before_renderer_repair_'+f))
s=(r/'runtime.py').read_text()
start=s.index('  image=[];actual=[]\n',s.index(' def render('))
end=s.index(' def close(',start)
s=s[:start]+"""  # Goal-only fixture: official cameras/API, fixed retreat robot pose, no rollout.
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
  (self.out/('renderer_case_'+str(case['episode_id'])+'.json')).write_text(json.dumps({'method':'official camera/render API; uniform render-only goal placement after fixed parking-scene retreat; not historical simulation snapshot','audit':audit,'fixed_robot_dof':fixed_dof.detach().cpu().numpy().tolist()},indent=2))
  return np.array(image),actual,cfg
""" +s[end:]
(r/'runtime.py').write_text(s)
s=(r/'probe.py').read_text().replace("p.add_argument('--pilot',action='store_true')","p.add_argument('--pilot',action='store_true');p.add_argument('--resume-decomposition')")
start=s.index('decomp=[];raw=');end=s.index('goaltests=[]',start)
block=s[start:end]
s=s[:start]+"if a.resume_decomposition:\n decomp=json.loads(pathlib.Path(a.resume_decomposition).read_text())\n (out/'decomposition_index.json').write_text(json.dumps(decomp,indent=2))\nelse:\n"+''.join(' '+q+'\n' for q in block.splitlines())+s[end:]
(r/'probe.py').write_text(s)
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'official goal-retreat collided with cube2, displaced G0 by25mm before any policy goal samples; invalid single-target contrast','repair':'uniform official render-only root placement after common parking-scene retreat; same robot q/qd across goals, exact original goalXY/colors, only one objectXY changes; no executed trajectory intervention','pre_results':True,'resume_decomposition':str(r/'runs/offline_formal/data/decomposition_index.json'),'frozen_targets_unchanged':True,'code_sha256':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['runtime.py','probe.py']}}
(r/'receipts/renderer_repair_freeze.json').write_text(json.dumps(x,indent=2))

