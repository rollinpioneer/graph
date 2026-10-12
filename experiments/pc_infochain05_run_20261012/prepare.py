"""Freeze discovery materials and P2 facts before seeing new model outputs."""
from common import *
import subprocess,time
PREV=pathlib.Path(os.environ['PREV']);begin=time.time()
es=json.loads((PREV/'results/episode_index.json').read_text());vision={v['episode_id']:v for v in json.loads((PREV/'results/P1_vision_proxy.json').read_text())}
cache={}
def load(e):
 if e['trajectory'] not in cache:
  zz=np.load(e['trajectory']);cache[e['trajectory']]={k:zz[k] for k in zz.files if k!='images'}
 return cache[e['trajectory']]
def legal(xy):
 dd=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1);np.fill_diagonal(dd,np.inf)
 return bool((np.abs(xy)<=[.15,.20]).all() and dd.min()>=.04)
def target_cases(e):
 z=load(e);i=e['slot'];ans=[]
 for t in e['image_steps']:
  t=int(t);d=z['distances'][t,i];st=z['states'][t,i]
  if (d>=.04).sum()!=1 or d.max()<.06 or (st[:,2]<.975).any():continue
  j=int(d.argmax());orig=z['goal'][i].copy();p=st[j,:2]
  for ang in [0,np.pi/2,np.pi/4,3*np.pi/4]:
   delta=.08*np.array([np.cos(ang),np.sin(ang)])
   g1=orig.copy();g2=orig.copy();g1[j]=p+delta;g2[j]=p-delta
   if legal(g1) and legal(g2):
    ans.append({'episode_id':e['episode_id'],'step':t,'object_id':j,'original_goal':orig.tolist(),'G1':g1.tolist(),'G2':g2.tolist(),
     'current_object_xyz':st[:,:3].tolist(),'goal_length_m':.08,'colors':vision[e['episode_id']]['goal_assignment_per_view'][0],
     'n':e['n'],'trajectory':e['trajectory'],'slot':i,'initial':e['initial'],'role':'fixed historical input, not trajectory reproduction'});break
 return ans
eligible={e['episode_id']:target_cases(e) for e in es}
selected=[]
for n in (3,6):
 for success in (True,False):
  pool=[e for e in es if e['n']==n and e['endpoints'][str(e['steps'])]['all_success']==success]
  pool.sort(key=lambda e:(not bool(eligible[e['episode_id']]),e['first_no_progress'] is None,e['episode_id']))
  assert len(pool)>=3
  selected+=pool[:3]
assert len(selected)==12
frames=[];diagnostics=[];traces=[]
for c,e in enumerate(selected):
 z=load(e);i=e['slot'];T=e['steps'];st=z['states'][:,i];d=z['distances'][:,i];ee=z['eef'][:,i]
 first=e['first_no_progress'];requests=[0,max(0,(first or 50)-20),first or min(50,T),min(T,(first or 50)+20),T]
 points=sorted(set(0 if t==0 else int(min(e['image_steps'],key=lambda x:abs(x-t))) for t in requests))
 frames.append({'case_id':c,'episode':e,'steps':points,'requested_steps':requests,'image_source':'exact saved policy-observation RGB, uint8 CHW; official /255 extraction; only saved times, no invented hidden history',
 'phase': 'unprogressing window and full later history' if first is not None else 'progress/success reference'})
 lo=first-20 if first else max(0,T//2-10);hi=lo+20;hi=min(hi,T);eefpart=ee[lo:hi+1]
 path=float(np.linalg.norm(np.diff(eefpart,axis=0),axis=-1).sum());net=float(np.linalg.norm(eefpart[-1]-eefpart[0]));pending=d[lo]>=.04
 dist=np.linalg.norm(ee[:,None,:]-st[...,:3],axis=-1);pending_near=dist[:,pending].min(-1) if pending.any() else np.full(T+1,np.nan)
 effective=np.flatnonzero((d[hi]-d[hi:].min(0))>=.01) # identities with eventual gain
 future=None
 for t in range(hi+1,T+1):
  if (d[hi]-d[t]).max()>=.01:future=t;break
 prior=None
 for t in range(20,lo+1):
  if (d[t-20]-d[t]).max()>=.01:prior=[t-20,t]
 cat=[]
 if path<.01:cat.append('little eef motion')
 if path>=.03 and net/(path+1e-9)<.3:cat.append('returning/circuitous path proxy')
 approach=float(pending_near[lo]-np.nanmin(pending_near[lo:hi+1])) if pending.any() else None
 if approach is not None and approach>=.01:cat.append('approaching pending-object center proxy')
 if np.linalg.norm(st[lo:hi+1,:,:3]-st[lo,:,:3],axis=-1).max()>=.001:cat.append('object motion; adjacent behavior, not strict no-progress if detector absent')
 # Physical gripper site pose/finger centers give actual geometry positions; never prove contact.
 physical=pathlib.Path(e['trajectory']).parent/'physical.npz';pose=None
 if physical.exists():
  ph=np.load(physical);meta=json.loads((physical.parent/'contact_source.json').read_text());b={q['name']:q['index_env'] for q in meta['body_map']}
  mask=(ph['policy_step']>=lo+1)&(ph['policy_step']<=hi);ids=np.flatnonzero(mask)
  if len(ids):
   body=ph['body_state'][ids,i];tips=[b[k] for k in ('panda_leftfinger_tip','panda_rightfinger_tip') if k in b]
   cubes=[q['index_env'] for q in meta['body_map'] if q['actor']>=3]
   pp=body[:,b['panda_grip_site']]
   gap=np.linalg.norm(body[:,tips,None,:3]-body[:,None,cubes,:3],axis=-1) if tips else np.zeros((0,))
   pose={'source':str(physical),'grip_site_pos_quat':pp[:,0:7].tolist(),'sample_times':ph['time'][ids].tolist(),
    'min_finger_tip_to_cube_center_m':float(gap.min()) if len(gap) else None,'finger_tip_surface_geometry':'center distance only; no mesh/SDF surface proof',
    'sampling':'dt0.01667 last-substep readout; gaps/other arm parts/inter-sample contacts remain unknown'}
  ph.close()
 diagnostics.append({'case_id':c,'episode_id':e['episode_id'],'n':e['n'],'window':[lo,hi],'original_no_progress':first is not None,
 'eef_path_m':path,'eef_start_end_displacement_m':net,'R':net/(path+1e-9),'eef_z_minmax_m':[float(eefpart[:,2].min()),float(eefpart[:,2].max())],
 'pending_start_objects':np.flatnonzero(pending).tolist(),'distance_to_pending_start_min_m':float(pending_near[lo]) if pending.any() else None,
 'pending_center_approach_gain_m':approach,'first_later_effective_progress_step':future,'time_to_progress_controls':future-hi if future else None,
 'same_episode_prior_progress_window':prior,'classification':cat or ['descriptive evidence insufficient'],'physical_pose':pose,
 'normal_reference_level':'1 facts +2 prior-progress when available; level3 imperfect external references separately indexed',
 'contact':'unknown, geometry is proxy','fall':e['first_fall'],'final_endpoint':e['endpoints'][str(T)]})
 for t in range(T+1):
  traces.append({'case_id':c,'episode_id':e['episode_id'],'step':t,'eef_xyz':ee[t].tolist(),'object_xyz':st[t,:,:3].tolist(),'distances':d[t].tolist(),
  'pending_center_distance_m':dist[t].tolist(),'action_after_unnormalize':z['actions'][t,i].tolist() if t<T else None,
  'controller_goal':z['controller_goal'][t,i].tolist() if t<T and 'controller_goal' in z else None})
targets=[]
for n in (3,6):
 pool=[q for e in selected if e['n']==n for q in eligible[e['episode_id']]]
 seen=set()
 for q in pool:
  if q['episode_id'] in seen:continue
  seen.add(q['episode_id']);targets.append(q)
  if len(seen)==3:break
 assert len(seen)>=1
# A/B/C/D uses first two saved observations in selected no-progress/repeating candidates, and success counterexamples.
noise_cases=[]
for e in selected:
 z=load(e);i=e['slot'];t=e['first_no_progress'] or min(50,e['steps']);possible=[int(x) for x in e['image_steps'] if max(1,t-20)<=x<=min(e['steps'],t+20)]
 if len(possible)<2:possible=e['image_steps'][:2]
 noise_cases.append({'episode_id':e['episode_id'],'n':e['n'],'trajectory':e['trajectory'],'initial':e['initial'],'slot':i,'steps':[int(possible[0]),int(possible[1])],
 'role':'source attribution probe; not proposed fixed-noise controller'})
dump('representative_cases.json',frames);dump('P2_space_time_diagnosis.json',diagnostics);dump('P2_full_selected_traces.json',traces);dump('target_input_cases.json',targets);dump('noise_input_cases.json',noise_cases)
freeze={'utc':utc(),'card_sha256':sha(R/'REQUESTED_CARD.md'),'branch':'codex/pc-infochain-05-20261012-card','base_commit':'c878a514710f9511189aff5bafa71fbbdd744935',
 'source_index':str(PREV/'results/episode_index.json'),'source_index_sha256':sha(PREV/'results/episode_index.json'),
 'role':'all historical discovery; new goal conditions model capability probes, not heldout outcome validation',
 'representative_episodes':12,'target_inputs':len(targets),'noise_inputs':len(noise_cases),'H':5,'T':100,'repeat_samples':8,
 'sampling':{'seeds':[1205100+k for k in range(8)],'A':'same frame,same whole RNG state repeated twice',
 'B':'same frame,8 independent whole RNG streams','C':'two actual frames, each corresponding seed restored before complete sampling',
 'D':'two actual frames, each call progresses normal independent RNG stream; complete random states/noise hashes saved',
 'batch':1,'slot':0,'full_noise': 'capture every torch.randn and randn_like result and before/after RNG states'},
 'response_measure':'full generated H5 XYZ action sequence differences vs same-goal between-seed differences; first action not direction-error criterion',
 'goal_legality':'within official samplingXY .15/.20, >=.04 goal separation, one target changes, equal .08 distance from current cube, original renderer',
 'decision_rules':{'numeric_repeat':'maxabs A <=1e-6; otherwise technical failure',
 'encoding':'goal current RGB changes and permutation-invariant foreground position/features/reconstruction; not particle-id identity',
 'target_use':'reproducible CRN G1vsG2 action changes >A numerical baseline weakens total goal-insensitivity; ratio toB is descriptive, not correctness',
 'sampling':'B vsC/D changes isolate sources; large variation alone is not harmful multimodality',
 'validation':'choose at most1 path after offline results; prereg separate matched complete runs, otherwise report not executed'},
 'budget':{'total_gpu_s':7200,'memory_mib':12288,'offline_max_s':1440,'supplement_max_s':1800,'validation_max_s':3240,'margin_s':720,
 'new_complete_supplement_cap':6,'new_complete_validation_cap':12},
 'offline_lists_sha256':{n:sha(R/'results'/n) for n in ['representative_cases.json','target_input_cases.json','noise_input_cases.json']},
 'code_sha256':sha(__file__)}
(R/'receipts/freeze_offline.json').write_text(json.dumps(freeze,indent=2))
print('MATERIALS_FROZEN',len(selected),'target probes',len(targets),'noise cases',len(noise_cases),'CPUwall',time.time()-begin,flush=True)
