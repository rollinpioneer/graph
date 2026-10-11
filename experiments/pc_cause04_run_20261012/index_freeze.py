"""Create immutable unified discovery index and freeze definitions before comparison."""
from common import *
import subprocess
started=utc()
sources=[]
for n in (3,6):
 for f in sorted((OLD/('runs/core%d/data'%n)).glob('trajectory_*.npz')):sources.append(('historical',f))
for f in sorted((SCALE/'runs').glob('A_*/data/trajectory.npz')):sources.append(('PC_SCALE02',f))
for run in ('P1_sparse8','P1_crowded8','diagnostic_fixed5','validation_P3_8'):
 sources.append(('PC_P12303',PRIOR/'runs'/run/'data/trajectory.npz'))
episodes=[];inventory=[];objects=[];aux=[]
for cohort,f in sources:
 z=np.load(f);cfg=json.loads((f.parent/'config.json').read_text());T,B,n,_=z['states'].shape;T-=1
 initial=f.parent/('initial_'+f.stem.split('_')[-1]+'.npz' if cohort=='historical' else 'initial.npz')
 ini=np.load(initial);envf=f.parent/'environment_config.json';ec=json.loads(envf.read_text()) if envf.exists() else {}
 meta=f.parent/'contact_source.json'
 collection=ec.get('sim',{}).get('physx',{}).get('contact_collection')
 fields={k:list(z[k].shape) for k in z.files}
 for p in [f,initial,f.parent/'config.json',envf,meta,f.parent/'physical.npz']:
  if p.exists():inventory.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'cohort':cohort})
 geom=f.parent/'geometry_actual.json';grows=json.loads(geom.read_text()) if geom.exists() else []
 scenes=[]
 if cohort=='PC_P12303':
  sf=PRIOR/'results'/cfg.get('scene_file','scenes_all.json')
  allscenes=json.loads(sf.read_text());groups=[int(g) for g in cfg['groups'].split(',')]
  scenes=[q for q in allscenes if q['group'] in groups and (q['condition']==cfg['condition'] or cfg['condition']=='diagnostic_all')]
  if cfg['mode']=='validation':
   scenes=[dict(q,arm=arm) for arm in cfg['arms'].split(',') for q in scenes]
   scenes=[scenes[k] for k in np.random.RandomState(2611124).permutation(len(scenes))]
 for i in range(B):
  st=z['states'][:,i];dd=z['distances'][:,i];np.testing.assert_allclose(np.linalg.norm(st[...,:2]-z['goal'][i],axis=-1),dd,atol=1e-6,rtol=1e-5)
  no=no_progress(st,dd);fall=first_run((st[...,2]<.975).any(-1))
  scene=scenes[i] if len(scenes)==B else {}
  condition=scene.get('condition',cfg.get('condition','historical_random'))
  arm=scene.get('arm','B0')
  protocol=cohort+('/P1' if cfg.get('mode')=='P1' else '/fixed_diagnostic' if cfg.get('mode')=='diagnostic' else '/amplitude_'+arm if cfg.get('mode')=='validation' else '')
  eid=cohort+'/'+f.parent.parent.name+'/'+f.stem+'/slot%02d'%i
  e={'episode_id':eid,'cohort':cohort,'protocol':protocol,'role':'discovery/calibration; prior validation now historical','trajectory':str(f),'initial':str(initial),'slot':i,
  'n':n,'steps':T,'policy_seed':int(f.stem.split('_')[-1]) if cohort=='historical' else cfg['seed'],'batch_size':B,'scene_seed':None,
  'scene_group':scene.get('group',i if cohort=='PC_SCALE02' else None),'layout':condition,'arm':arm,
  'scene_key':cohort+'/'+condition+'/'+str(scene.get('group',i)),'state_identity_sha256':hashlib.sha256(st[0].tobytes()+z['goal'][i].tobytes()).hexdigest(),
  'fields':fields,'image_steps':z['image_steps'].tolist(),'state_dt':'control step ~1/3 s','physical_dt_s':.01667 if (f.parent/'physical.npz').exists() else None,
  'contact_collection':collection,'contact_pair_usable':False,'normalized_policy_action':'not saved separately',
  'color_getter_unique':len(np.unique(ini['colors'][i],axis=0)),'color_identity_valid':len(np.unique(ini['colors'][i],axis=0))==n,
  'rendered_object_visibility':'unknown: no validated identity projection/segmentation','dlp_identity':'particles are not stable object identities; initial and goal only',
  'initial_pending':int((dd[0]>=.04).sum()),'initial_transport_mean_m':float(dd[0].mean()),'initial_transport_per_object_m':dd[0].tolist(),
  'first_fall':fall,'first_no_progress':no[0] if no else None,'no_progress_ends':no,
  'endpoints':{str(t):{'fraction':float((dd[t]<.04).mean()),'all_success':bool((dd[t]<.04).all()),'mean_xy_m':float(dd[t].mean()),'max_xy_m':float(dd[t].max())} for t in (100,200,T) if t<=T}}
  episodes.append(e)
  for j in range(n):
   hit=first_run(dd[:,j]<.04);leave=first_run(dd[hit+1:,j]>.05) if hit is not None else None;leave=leave+hit+1 if leave is not None else None
   rec=first_run(dd[leave+1:,j]<.04) if leave is not None else None;rec=rec+leave+1 if rec is not None else None
   closest=np.linalg.norm(z['eef'][:,i]-st[:,j,:3],axis=-1)
   near=closest<.055
   moved=np.linalg.norm(np.diff(st[:,j,:3],axis=0),axis=-1)>.001
   pre_error=st[:-1,j,:2]-z['goal'][i,j];disp=np.diff(st[:,j,:2],axis=0);dot=(disp*(-pre_error)).sum(-1)
   objects.append({'episode_id':eid,'object_id':j,'initial_goal_distance_m':float(dd[0,j]),'final_goal_distance_m':float(dd[-1,j]),
   'first_sustained_hit':hit,'departure':leave,'recovery':rec,'first_fall':first_run(st[:,j,2]<.975),'movement_steps_gt_1mm':int(moved.sum()),
   'movement_toward_goal_steps':int((moved&(dot>0)).sum()),'movement_away_goal_steps':int((moved&(dot<0)).sum()),
   'moved_after_hit_steps':int(moved[hit:].sum()) if hit is not None else 0,'near_eef_entries_proxy':int((near[1:]&~near[:-1]).sum()+near[0]),
   'never_near_eef_proxy':bool(not near.any()),'max_displacement_m':float(np.linalg.norm(st[:,j,:3]-st[0,j,:3],axis=-1).max()),
   'initial_edge_bbox_margin_m':float(bbox_margin(st[:1,j])[0]),'visibility':'unknown','contact_count':'unknown','selection':'unknown; proximity is not selection'})
# Partial probes kept outside inferential cohorts.
for base in (OLD,SCALE,PRIOR):
 for f in sorted((base/'runs').glob('*/data/trajectory*.npz')):
  if any(str(f)==str(x[1]) for x in sources):continue
  zz=np.load(f)
  aux.append({'path':str(f),'shape':list(zz['states'].shape),'role':'auxiliary pilot/recovery; not pooled with complete cohorts','sha256':sha(f)})
weights=[]
expected={'state_1200000.pt':'7116627cdd841f76035f06dcbc09c66ff84e0a58d8c4146d42264cba2d7492f5','dlp_panda_push.pth':'a8a1113048df79c0fd00cdd4539779a7b3c588cefc4e5f26eb50c0482f756236','panda_push_replay_buffer_dlp.pkl':'7abf83b82fcf2bae801ddae6fa6138d505f4957570d638e37da1e5a5290baf12'}
for p in (OLD/'resources').rglob('*'):
 if p.name in expected:
  h=sha(p);assert h==expected[p.name];weights.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h,'verified_utc':utc()})
assert len(episodes)==77 and len(weights)==3
versions=[]
for p in (R.parents[1],OLD/'EC-Diffuser',pathlib.Path('/home/xushijie3/work/graph_cp_disr')):
 def git(*args):return subprocess.check_output(['git','-C',str(p),*args],text=True)
 versions.append({'path':str(p),'head':git('rev-parse','HEAD').strip(),'branch':git('branch','--show-current').strip(),'status':git('status','--short'),'unpushed':git('log','--oneline','@{upstream}..HEAD') if p.name!='EC-Diffuser' else 'no upstream assumed'})
assert versions[1]['head']=='6b5e482fe479f647d4ca3790fac70c42f4940ab2'
dump('episode_index.json',episodes);dump('auxiliary_index.json',aux);dump('raw_inventory.json',inventory);dump('identities.json',{'weights':weights,'repositories':versions})
table('object_timeline.csv',objects)
freeze={'utc':utc(),'index_sha256':sha(R/'results/episode_index.json'),'sources':len(sources),'episodes':len(episodes),'data_role':'all existing data discovery only',
'P2':{'primary_cases':'first no-progress window per episode, not selected by outcome','candidate_controls':'every 10 steps in successful same-protocol/n/layout trajectory, start>=20, end+20<=T',
'window':20,'pre_context':20,'post_context':20,'stage_bins':[0,50,100,150,200,400,500],
'matching_tolerances':{'nearest_eef_object_m':.025,'eef_height_m':.015,'mean_pending_goal_distance_m':.025,'pending_count':0,'min_bbox_edge_m':.03,'initial_transport_mean_m':.03},
'visibility':'required validated same-object visibility at start; absent means no qualified match. Compute geometric provisional matches only, never pass gate.',
'selection':'prefer different episode, ascending sum normalized feature errors, tie episode id/start. No within-window or future outcome in matching.',
'normal_confirmation_offline':'next 20 steps after window: at least one object improves >=.01 m and no fall; confirmation after matching, never rematch according to outcome',
'independent_unit':'episode; control reuse explicitly tracked'},
'P1':{'near_proxy_m':.055,'moved_step_m':.001,'analysis_scope':'one complete audit pass; no expanding descriptive samples if G1 fails'},
'P3':{'fall':'z<.975 for5 controls; episode any rule and per-object first runs','bbox':'rotated cube .035m, true table x+/-.25 y+/-.30',
'causal_velocity':'strictly before first bbox crossing, never post-fall speed','ordering':'control samples vs physical .01667s labelled separately'},
'definitions_sha256':sha(R/'common.py'),'started_utc':started}
(R/'receipts/freeze_analysis.json').write_text(json.dumps(freeze,indent=2))
print('INDEX_FROZEN',len(episodes),'objects',len(objects),'aux_files',len(aux),flush=True)
