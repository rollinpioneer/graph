import os,pathlib,json,hashlib,datetime,shutil
r=pathlib.Path(os.environ['ROUND'])
for f in ['runtime.py','probe.py']:shutil.copy2(r/f,r/'receipts'/('before_graphics_refresh_'+f))
s=(r/'runtime.py').read_text()
s=s.replace("raw.gym.step_graphics(raw.sim);raw.compute_observations()\n   measured=","raw.gym.simulate(raw.sim);raw.gym.fetch_results(raw.sim,True)\n   raw.gym.step_graphics(raw.sim);raw.compute_observations()\n   measured=")
s=s.replace("assert np.abs(measured-xy).max()<1e-6","assert np.abs(measured-xy).max()<1e-4")
s=s.replace("assert torch.equal(raw._dof_state,fixed_dof)","dq=float(torch.abs(raw._dof_state-fixed_dof).max());assert dq<.01\n   if not audit:common_dof=raw._dof_state.clone()\n   match=float(torch.abs(raw._dof_state-common_dof).max());assert match<1e-6")
s=s.replace("'robot_dof_identical':True,'physics_steps_after_placement':0","'robot_dof_match_between_conditions_maxabs':match,'robot_drift_from_fixture':dq,'physics_steps_after_placement':1")
s=s.replace("  return np.array(image),actual,cfg","  assert np.mean(np.square(image[1].astype(float)/255-image[2].astype(float)/255))>1e-5,'goal graphics did not change'\n  return np.array(image),actual,cfg")
(r/'runtime.py').write_text(s)
s=(r/'probe.py').read_text()
s=s.replace("p.add_argument('--resume-decomposition')","p.add_argument('--resume-decomposition');p.add_argument('--target-n',type=int);p.add_argument('--noise-only',action='store_true');p.add_argument('--renderer-only',action='store_true')")
pos=s.index('cache={};ini={}')
s=s[:pos]+"""if a.target_n:
 targets=[c for c in targets if c['n']==a.target_n];ncases=[]
if a.noise_only:targets=[]
if a.renderer_only:targets=targets[:1];ncases=[]
""" +s[pos:]
pos=s.index(' reps=[rt.encode')
s=s[:pos]+""" if a.renderer_only:
  (out/'renderer_selfcheck.json').write_text(json.dumps({'n':c['n'],'pixel_mse_G1_G2':float(np.mean(np.square(imgs[1].astype(float)/255-imgs[2].astype(float)/255))),'actual_goals':actual,'calls':0},indent=2))
  rt.close();print('RENDERER_SELFCHECK_COMPLETE',flush=True);sys.exit()
""" +s[pos:]
(r/'probe.py').write_text(s)
x={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pre_policy_results':'prior72 goal calls invalid: all3C target images identical, graphics did not update after tensor writes; excluded from inference.6C sim recreation segfault, use separate processes per n. Costs retained.','repair':'one official physics tick updates rigid-body graphics after placement; validate actualXY error<0.1mm, post-tick robotDOF identical acrossallgoals within1e-6, changedRGBMSE>1e-5 before any sample','calls_before_acceptance':0,'code_sha256':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['runtime.py','probe.py']}}
(r/'receipts/graphics_refresh_freeze.json').write_text(json.dumps(x,indent=2))

