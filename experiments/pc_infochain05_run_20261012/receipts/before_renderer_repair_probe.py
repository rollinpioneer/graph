"""Bounded offline DLP, goal and A/B/C/D source tests, with full random draws."""
import argparse,time
from runtime import *
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);begin=time.monotonic();rt=Runtime(out)
cases=json.loads((R/'results/representative_cases.json').read_text());targets=json.loads((R/'results/target_input_cases.json').read_text());ncases=json.loads((R/'results/noise_input_cases.json').read_text());freeze=json.loads((R/'receipts/freeze_offline.json').read_text())
cache={};ini={}
def image(e,t):
 path=e['trajectory']
 if path not in cache:
  z=np.load(path);cache[path]={k:z[k] for k in ('images','image_steps','states','distances','goal')}
 if e['initial'] not in ini:
  z=np.load(e['initial']);ini[e['initial']]={k:z[k] for k in z.files}
 if t==0:return ini[e['initial']]['initial_image'][e['slot']]
 z=cache[path];k=int(np.flatnonzero(z['image_steps']==t)[0]);return z['images'][k,e['slot']]
def goal(e):image(e,0);return ini[e['initial']]['goal_image'][e['slot']]
def input_rep(e,t):return rt.historical_encode(e,t)[0]
def goal_rep(e):image(e,0);return ini[e['initial']]['goal_latent'][e['slot']].copy()
def perm_delta(x,y):
 vals=[]
 for v in range(2):
  costs=np.linalg.norm(x[v,:,None,:]-y[v,None,:,:],axis=-1);i,j=linear_sum_assignment(costs);vals.append(float(costs[i,j].mean()))
 return float(np.mean(vals))
if a.pilot:
 e=cases[0]['episode'];cur,pack=rt.historical_encode(e,0,True);gg=goal_rep(e);image(e,0)
 err=float(np.abs(cur-ini[e['initial']]['initial_latent'][e['slot']]).max())
 v1,r1=rt.sample(cur,gg,'pilot_A1',1205100);v2,r2=rt.sample(cur,gg,'pilot_A2',1205100)
 diff=float(np.abs(v1-v2).max());assert diff<=1e-6
 result={'utc':utc(),'initial_stored_particle_maxabs_error':err,'preprocessing':'uint8 CHW /255, encode_all deterministic, get_dlp_rep; initial saved identity comparison',
 'A_repeat_maxabs':diff,'samples':2,'sample_wall_seconds':[r1['elapsed_s'],r2['elapsed_s']],'model_H':5,'diffusion_T':rt.diffusion.n_timesteps,
 'load_and_pilot_wall_s':time.monotonic()-begin,'finite_rep_and_recon':bool(np.isfinite(cur).all() and np.isfinite(pack['rec']).all()),
 'exact_hidden_state_claim':err<=1e-4,'units':'two calls, not independent episodes'}
 assert result['finite_rep_and_recon'] and result['diffusion_T']==100
 (out/'pilot_receipt.json').write_text(json.dumps(result,indent=2));rt.close();print('PILOT_COMPLETE',json.dumps(result),flush=True);sys.exit()
# Official-pipeline reencoding, saved RGB only; backgrounds are separate from foreground information.
decomp=[];raw=out/'representations';raw.mkdir(exist_ok=True)
for c in cases:
 e=c['episode']
 for t in c['steps']+[None]:
  im=goal(e) if t is None else image(e,t);rep,pack=rt.historical_encode(e,t,True)
  stem='case%02d_%s'%(c['case_id'],'goal' if t is None else 'step%03d'%t);f=raw/(stem+'.npz');np.savez_compressed(f,image=im,policy_particles=rep,**pack)
  encoded_error=float(np.abs(rep-ini[e['initial']]['initial_latent'][e['slot']]).max()) if t==0 else None
  mse=np.square(pack['rec']-im.astype(float)/255).mean(axis=(1,2,3)).tolist()
  decomp.append({'case_id':c['case_id'],'episode_id':e['episode_id'],'n':e['n'],'step':t,'source':e['initial'] if t is None or t==0 else e['trajectory'],
  'raw_path':str(f),'sha256':sha(f),'particle_position':rep[...,:2].tolist(),'scale':rep[...,2:4].tolist(),'depth':rep[...,4].tolist(),
  'appearance':rep[...,5:-1].tolist(),'transparency':rep[...,-1].tolist(),'reconstruction_full_image_mse_per_view':mse,
  'initial_saved_particle_maxabs':encoded_error,'scope':'reencoded original saved input at this step; no missing intermediate history, particles not object identities'})
 (out/'decomposition_index.json').write_text(json.dumps(decomp,indent=2))
 print('DECOMP_CASE',c['case_id'],flush=True)
goaltests=[]
for k,c in enumerate(targets):
 # Fixed historical current image; render all three goals via same official goal path.
 cur=rt.historical_encode(c,c['step'])[0];imgs,actual,cfg=rt.render(c,[c['original_goal'],c['G1'],c['G2']])
 reps=[rt.encode(im)[0] for im in imgs];saved=rt.encode(goal(c))[0]
 folder=out/'targets';folder.mkdir(exist_ok=True);f=folder/('case%02d.npz'%k)
 np.savez_compressed(f,current_image=image(c,c['step']),rendered_goal_images=imgs,goal_particles=np.array(reps),current_particles=cur,actual_goals=np.array(actual),saved_original_goal_image=goal(c))
 actions=[]
 for h,gg in enumerate(reps):
  group=[]
  for seedid in freeze['sampling']['seeds']:
   act,receipt=rt.sample(cur,gg,'G_case%02d_cond%d_seed%d'%(k,h,seedid),seedid);group.append(act)
  actions.append(group)
 acts=np.array(actions);crn=np.linalg.norm(acts[1]-acts[2],axis=-1).mean(-1)
 within=[]
 for h in range(3):
  within.append([float(np.linalg.norm(acts[h,i]-acts[h,j],axis=-1).mean()) for i in range(8) for j in range(i+1,8)])
 row={'case_id':k,**c,'raw_path':str(f),'sha256':sha(f),'actual_goals':actual,
 'goal_encoding_permutation_delta_G1_G2':perm_delta(reps[1],reps[2]),'saved_vs_rerendered_G0_particle_delta':perm_delta(saved,reps[0]),
 'goal_image_pixel_mse_G1_G2':float(np.square(imgs[1].astype(float)/255-imgs[2].astype(float)/255).mean()),
 'CRN_G1_G2_H5_action_L2_mean_per_seed':crn.tolist(),'within_goal_H5_between_seed_L2':within,
 'effect_to_sampling_median_ratio':float(np.median(crn)/(np.median(np.array(within))+1e-12)),
 'first_action_direction_not_success_criterion':True,'actions_generated_not_executed':acts.tolist()}
 goaltests.append(row);(out/'goal_test_results.json').write_text(json.dumps(goaltests,indent=2));print('TARGET_DONE',k,flush=True)
 # Avoid keeping two physics worlds resident; recreate only as needed for target n.
 if k==len(targets)-1 or targets[k+1]['n']!=c['n']:
  rt.close();rt.envs={}
noise_results=[]
for k,c in enumerate(ncases):
 r0=input_rep(c,c['steps'][0]);r1=input_rep(c,c['steps'][1]);gg=goal_rep(c)
 aa=[]
 for repeat in range(2):
  q,_=rt.sample(r0,gg,'N_case%02d_A%d'%(k,repeat),1205100);aa.append(q)
 assert np.abs(aa[0]-aa[1]).max()<=1e-6
 BB=[];CC=[];DD=[]
 for sd in freeze['sampling']['seeds']:
  b,_=rt.sample(r0,gg,'N_case%02d_B_seed%d'%(k,sd),sd);BB.append(b)
  c1,_=rt.sample(r1,gg,'N_case%02d_C_seed%d'%(k,sd),sd);CC.append(c1)
  d0,_=rt.sample(r0,gg,'N_case%02d_D0_seed%d'%(k,sd),sd+10000)
  d1,_=rt.sample(r1,gg,'N_case%02d_D1_seed%d'%(k,sd),None);DD.append([d0,d1])
 BB=np.array(BB);CC=np.array(CC);DD=np.array(DD)
 bdiff=[float(np.linalg.norm(BB[i]-BB[j],axis=-1).mean()) for i in range(8) for j in range(i+1,8)]
 cdiff=np.linalg.norm(BB-CC,axis=-1).mean(-1);ddiff=np.linalg.norm(DD[:,0]-DD[:,1],axis=-1).mean(-1)
 noise_results.append({'case_id':k,**c,'A_numerical_maxabs':float(np.abs(aa[0]-aa[1]).max()),
 'B_same_input_between_seed_H5_L2':bdiff,'C_input_changed_fixed_noise_H5_L2_per_seed':cdiff.tolist(),
 'D_input_and_normal_random_stream_changed_H5_L2_per_seed':ddiff.tolist(),
 'input_pixel_RMS':float(np.sqrt(np.mean(np.square(image(c,c['steps'][0]).astype(float)/255-image(c,c['steps'][1]).astype(float)/255)))),
 'input_particle_permutation_delta':perm_delta(r0,r1),'median_B':float(np.median(bdiff)),'median_C':float(np.median(cdiff)),'median_D':float(np.median(ddiff)),
 'interpretation':'source attribution; multimodal variation alone is not policy failure; none of generated sequences executed'})
 (out/'noise_test_results.json').write_text(json.dumps(noise_results,indent=2));print('NOISE_DONE',k,flush=True)
rt.close();(out/'completed.json').write_text(json.dumps({'utc':utc(),'wall_seconds':time.monotonic()-begin,'decompositions':len(decomp),'goal_inputs':len(goaltests),'noise_inputs':len(noise_results),'policy_calls':rt.count,'full_episodes':0},indent=2));print('OFFLINE_COMPLETE',rt.count,flush=True)
