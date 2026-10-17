"""FACB-Z0: register sources, checkpoints and samples before any scoring."""
from pathlib import Path
import csv, hashlib, json, subprocess, time, importlib.metadata as md
import numpy as np

ROOT=Path(__file__).resolve().parent
FORMAL=Path('/home/xushijie3/work/gfp_eta0_ablation_20261015')
IROOT=FORMAL/'analysis_p0p1/confirm_scratch'
SOURCE=FORMAL/'source/gfp'
DATA=(FORMAL/'home/.ogbench/data/humanoidmaze-medium-navigate-v0.npz').resolve()
REF='8c7caf61b8b91b85c7340df04a2db5c10eeb0a6f'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
 p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def git(*args):return subprocess.check_output(['git','-C',str(SOURCE),*args],text=True).strip()

t0=time.time()
assert md.version('ogbench')=='1.1.0'
# This loader only reads arrays. No make_env_and_datasets or relabel_dataset call.
import gymnasium as gym
def forbidden(*a,**kw):raise RuntimeError('FACB-Z0 prohibits environment operations')
gym.make=forbidden;gym.Env.reset=forbidden;gym.Env.step=forbidden
from ogbench.utils import load_dataset
data=load_dataset(str(DATA),add_info=True)
raw=np.load(DATA);raw_idx=np.flatnonzero(raw['terminals']==0)
obs=data['observations'];nobs=data['next_observations'];act=data['actions']
assert obs.shape[1:]==(69,) and act.shape[1:]==(21,)
assert len(obs)==len(raw_idx) and np.allclose(data['qpos'][:,:2],obs[:,:2],rtol=0,atol=1e-6)
assert np.array_equal(nobs,raw['observations'][raw_idx+1].astype(np.float32))
goal=np.array([20.,20.]);success=(np.linalg.norm(data['qpos'][:,:2]-goal,axis=1)<=.5).astype(np.float32)
rewards=success-1.;masks=1.-success
assert set(np.unique(rewards))=={-1.,0.} and np.array_equal(rewards,-masks)
finite=np.isfinite(obs).all(1)&np.isfinite(nobs).all(1)&np.isfinite(act).all(1)
bootstrap=finite&(masks==1)
xy=nobs[:,:2];start=(xy[:,0]>=-2)&(xy[:,0]<6)&(xy[:,1]>=-2)&(xy[:,1]<6)
near=np.linalg.norm(xy-goal,axis=1)<=2.
rng=np.random.default_rng(2026101700);used=np.zeros(len(obs),bool);chunks=[];labels=[]
for name,num,region in [('global',4096,np.ones(len(obs),bool)),('start',2048,start),('near_goal',1024,near)]:
 candidates=np.flatnonzero(bootstrap&region&~used)
 chosen=rng.choice(candidates,min(num,len(candidates)),replace=False)
 used[chosen]=True;chunks.append(chosen);labels.extend([name]*len(chosen))
idx=np.concatenate(chunks);labels=np.array(labels)
assert len(idx)==len(np.unique(idx))==7168,'Insufficient distinct samples: stop, do not replace'
samples=dict(observations=obs[idx],next_observations=nobs[idx],actions=np.clip(act[idx],-1+1e-5,1-1e-5),rewards=rewards[idx],masks=masks[idx],dataset_index=idx,raw_index=raw_idx[idx],pool=labels,start=start[idx],near_goal=near[idx],dataset_boundary=data['terminals'][idx].astype(bool))
(ROOT/'arrays').mkdir(exist_ok=True)
np.savez_compressed(ROOT/'arrays/samples.npz',**samples)
with (ROOT/'sample_indices.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['pool','dataset_index','raw_index','next_state_start','next_state_near_goal'])
 w.writerows(zip(labels,idx,raw_idx[idx],start[idx].astype(int),near[idx].astype(int)))

cases=[dict(seed=s,step=k,group='calibration') for s in (1,2) for k in (600000,800000,1000000)]
cases += [dict(seed=s,step=k,group='healthy_test') for s in (4,5,6,7) for k in (600000,800000,1000000)]
cases += [dict(seed=0,step=700000,group='island'),dict(seed=3,step=400000,group='island')]
cases += [dict(seed=0,step=600000,group='aux_pre_island'),dict(seed=3,step=300000,group='aux_pre_island')]
cases += [dict(seed=s,step=k,group='aux_stage_matched') for s in (1,2) for k in (400000,700000)]
anchors=[dict(seed=s,step=500000,group='proxy_anchor') for s in range(8)]
records=[];configs={}
for arm,items,base in [('G',cases,FORMAL/'runs/formal'),('I',anchors,IROOT/'runs')]:
 for row in items:
  d=base/f'{arm}_task1_s{row["seed"]}';out=list(d.glob('out/*/date_*'));assert len(out)==1
  out=out[0];config=out/'config.json';cfg=json.loads(config.read_text());a=cfg['agent']
  assert cfg['seed']==row['seed'] and cfg['online_steps']==0
  assert cfg['env_name']=='humanoidmaze-medium-navigate-singletask-task1-v0'
  assert a['discount']==.995 and a['q_agg']==a['target_agg']=='mean' and not a['target_use_next'] and a['encoder'] is None
  assert a['value_hidden_dims']==[512]*4 and a['layer_norm'] and a['flow_steps']==10
  if arm=='I':assert a['expectile']==.7
  p=out/f'params_{row["step"]}.pkl';assert p.is_file()
  key=f'{arm}_s{row["seed"]}_{row["step"]}'
  configs[key]=cfg
  records.append(dict(id=key,arm=arm,**row,path=str(p),bytes=p.stat().st_size,sha256=sha(p),config_path=str(config),config_sha256=sha(config)))
save('checkpoint_manifest.json',records);save('configs.json',configs)
source_files=[SOURCE/'agents/gfp.py',SOURCE/'agents/iql.py',SOURCE/'utils/networks.py',SOURCE/'utils/flax_utils.py',SOURCE/'envs/env_utils.py',IROOT/'code/iql_gfp.py']
dist=md.distribution('ogbench')
source_files += [dist.locate_file('ogbench/utils.py'),dist.locate_file('ogbench/relabel_utils.py'),dist.locate_file('ogbench/locomaze/maze.py')]
source_files=[p for p in source_files if p.is_file()]
sources=[];(ROOT/'sources').mkdir(exist_ok=True)
for j,p in enumerate(source_files):
 dest=ROOT/'sources'/f'{j:02d}_{p.name}';dest.write_bytes(p.read_bytes());sources.append(dict(path=str(p),sha256=sha(p),snapshot=str(dest.relative_to(ROOT))))
archived=subprocess.check_output(['git','-C',str(ROOT/'archive_worktree'),'show',REF+':experiments/gfp_confirm_scratch_20261016/code/iql_gfp.py'])
assert archived== (IROOT/'code/iql_gfp.py').read_bytes()
identity_path=Path('/home/xushijie3/work/flow_baselines_preflight_20261013/receipts/source_file_identity.json')
identities=[v for v in json.loads(identity_path.read_text()) if v['algorithm']=='gfp']
checked=[]
for v in identities:
 if v['path'] in ['agents/gfp.py','agents/iql.py','utils/networks.py','utils/flax_utils.py','envs/env_utils.py']:
  assert sha(SOURCE/v['path'])==v['sha256'];checked.append(v)
assert len(checked)==5 and len({v['commit'] for v in checked})==1
save('source_manifest.json',dict(archive_base_commit=REF,gfp_head=checked[0]['commit'],gfp_checkout_type='Immutable source snapshot without .git; verified core files against original identity receipt',source_identity_path=str(identity_path),source_identity_sha256=sha(identity_path),matched_official_files=checked,sources=sources))
save('anchor_identity.json',dict(status='SOURCE_IDENTITY_VERIFIED_PENDING_FORWARD',source_archive_commit=REF,actual_root=str(IROOT),card_path_correction='gfp_confirm_scratch_20261016 is an archive prefix, not a top-level server directory',seeds=list(range(8)),checkpoint=500000,architecture=dict(hidden_dims=[512]*4,layer_norm=True,heads=1,observations=69,actions_input=False,normalization='none'),objective='expectile 0.7 regression to own target critic at dataset actions',q_head_aggregation='mean, following GFP q_agg',update_order='V then online critic/Actor/VaBC each batch; no V-only warmup',training_steps=1000000,selected_step=500000,independent_FACB_anchor_equivalence=False,differences=['I-arm V co-trained with GFP networks rather than independent fixed IQL pretraining','I-arm mean target-Q aggregation; repository official IQL uses min(q1,q2)','Formal FACB independent-anchor implementation not supplied; its full equivalence cannot be certified'],checkpoint_ids=[x['id'] for x in records if x['arm']=='I']))
save('data_identity.json',dict(path=str(DATA),sha256=sha(DATA),bytes=DATA.stat().st_size,ogbench_version=md.version('ogbench'),observations_shape=list(obs.shape),actions_shape=list(act.shape),reward_range=[float(rewards.min()),float(rewards.max())],discount=.995,S=200.,goal=goal.tolist(),goal_tolerance=.5,reward_timing='OGBench 1.1.0 current-state qpos; no shift to next state',masks='1 - current-state task success; dataset episode boundaries do not override native bootstrap masks',raw_terminal_rows_removed=int(np.sum(raw['terminals'])),valid_transitions=len(obs),nonfinite_transitions=int((~finite).sum()),goal_terminal_mask0_count=int((masks==0).sum()),dataset_boundary_mask1_count=int(((data['terminals']>0)&(masks==1)).sum()),sample_count=len(idx),sample_sha256=sha(ROOT/'arrays/samples.npz'),index_csv_sha256=sha(ROOT/'sample_indices.csv'),pool_counts={k:int((labels==k).sum()) for k in np.unique(labels)},primary_mask1=True,environment_operations=0,preparation_wall_seconds=time.time()-t0))
save('protocol.json',dict(card='FACB-Z0',authorization='User said execute on 2026-10-17; original draft status superseded for this zero-training card only',branch='codex/facb-z0-20261017',source_archive_commit=REF,training_updates=0,environment_construct_reset_step=0,new_episodes=0,new_model_initializations=0,one_gpu_process_wall_limit_seconds=7200,checkpoint_pairings_primary=20,auxiliary_pairings=6,auxiliary_definitions='s0 600k / s3 300k pre-island; s1/s2 400k and 700k stage matched; never used to choose c',sample_seed=2026101700,sample_rule='Uniform without replacement global 4096; start 2048 excluding global; near-goal 1024 excluding both; disjoint union 7168, equal-state weighting in calibration',region_state='next_observations',start_rectangle=[-2,6,-2,6],near_goal_radius=2,goal=[20,20],main_nonterminal='native task mask==1; retain truncated data-boundary transitions as official code does',rng_bases=[2026101701,2026101702],rng_rule='For each fixed 256-state chunk fold_in(base,batch_index); then official critic_loss splits once and shares sample_rng between Actor and VaBC',batch_size=256,bandwidth='c=linear NumPy 99th percentile of pooled |qG - own-seed V500k| on all s1/s2 600k/800k/1M, both noises, disjoint union; kappa=c/S',primary_healthy_rate='pooled s4-s7 x 3 checkpoints x both noises x 7168 states; also each-noise, seed, checkpoint and region reported',thresholds=dict(healthy_upper_max=.01,island_start_upper_strict_min=.5,healthy_lower_warning=.05),rng_consistency='Require both noises separately meet healthy <=1% and each island >50%, in addition to pooled decision',q_formula='mean(target-Q heads at next Actor(next_s), target-Q heads at VaBC(current_s))',q_max_abs_tolerance=3e-5,loss_rtol=3e-5,loss_atol=3e-5,sensitivity='All eight V anchors cross-scored using already computed qG; fixed primary c, no recalibration',decision_labels=['GO_FOR_CONTROLLED_SCREEN','NO_GO_PROXY_ANCHOR_SEPARABILITY','INCONCLUSIVE'],calibration_lock_before_test=True))
print('PREPARATION_COMPLETE',len(records),'checkpoints;',len(idx),'distinct states',flush=True)
