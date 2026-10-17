"""Freeze existing assets and a bounded, zero-environment historical screen."""
from pathlib import Path
import json,hashlib,shutil,time
import numpy as np
R=Path(__file__).resolve().parent
F=Path('/home/xushijie3/work/gfp_eta0_ablation_20261015'); P=F/'analysis_p0p1'; E=P/'exp_ab'
FACB=Path('/home/xushijie3/work/facb_z0_20261017')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x): (R/n).write_text(json.dumps(x,indent=2,allow_nan=False))
t=time.monotonic(); records=[]; configs={}; history={}; source=[]
for s,base in [(0,600000),(3,300000)]:
 fork=next((F/f'runs/formal/G_task1_s{s}/out').glob(f'*/date_*/params_{base}.pkl'))
 configs[str(s)]=json.loads((fork.parent/'config.json').read_text())
 for arm in ['B0','B1a']:
  folder=E/'b_B0' if (s,arm)==(0,'B0') else E/f'c/G_task1_s{s}_{arm}'
  log=folder/f'train_{arm}.json'; hist=json.loads(log.read_text()); history[f's{s}_{arm}']=hist
  assert Path(hist['start'])==fork
  for delta in range(0,100001,10000):
   p=fork if delta==0 else folder/f'params_{base+delta}.pkl'
   assert p.is_file(),str(p)
   records.append(dict(id=f's{s}_{arm}_{delta:06d}',seed=s,arm=arm,delta=delta,step=base+delta,path=str(p),bytes=p.stat().st_size,sha256=sha(p),fork=str(fork)))
 for name,p in [(f's{s}_config.json',fork.parent/'config.json')]:
  shutil.copy2(p,R/'sources'/name)
srcs=[F/'source/gfp/agents/gfp.py',F/'source/gfp/utils/networks.py',F/'source/gfp/utils/flax_utils.py',P/'code/exp_b.py',P/'code/exp_c.py',P/'code/common.py',E/'B_EVAL_POINTS.txt',E/'c/C_EVAL_POINTS.txt',E/'c/D_window_rule.json',FACB/'data_identity.json']
for p in srcs:
 q=R/'sources'/p.name;shutil.copy2(p,q);source.append(dict(path=str(p),copy=q.name,sha256=sha(p)))
save('source_manifest.json',source);save('checkpoint_manifest.json',records);save('configs.json',configs);save('historical_probes.json',history)
# Existing behavior only. Native/formal 400k is NOT the resumed s3 B0 trajectory.
ev=[]
for p in [E/'eval_B0.json',E/'c/eval_c.json.B1a',E/'c/eval_c.json.s3B0',E/'c/eval_c.json.s3B1a',E/'c/eval_orig_s3.json']:
 d=json.loads(p.read_text())
 for key,v in d.items():
  isfork=p.name=='eval_orig_s3.json'
  if isfork and v['step']!=300000:continue
  arm='fork' if isfork else ('B0' if p.name in ['eval_B0.json','eval_c.json.s3B0'] else 'B1a')
  seed=3 if ('s3' in p.name) else 0
  step=v['step'] if isfork else int(Path(v['pkl']).stem.split('_')[-1])
  episodes=v['episodes'];n=len(episodes);assert n==100
  succ=100*sum(float(e['success']) for e in episodes)/n;assert abs(succ-v['success'])<1e-6
  start=100*sum(-2<=e['final_xy'][0]<6 and -2<=e['final_xy'][1]<6 for e in episodes)/n
  ev.append(dict(seed=seed,arm=arm,step=step,delta=step-(600000 if seed==0 else 300000),success_percent=succ,final_start_percent=start,n=n,source=str(p),source_sha256=sha(p),key=key,eval_seed=v['eval_seed'],strict_initial_state_matching_verified=False))
save('historical_evaluations.json',ev)
# Reuse one previously fixed mask1 sample pool, no new environment/data generation.
poolpath=FACB/'arrays/samples.npz';pool=dict(np.load(poolpath));a=np.flatnonzero(pool['pool']=='start');assert len(a)==2048
b=np.random.default_rng(2026101721).choice(np.flatnonzero((pool['pool']=='global')&~pool['start']),2048,replace=False)
ix=np.r_[a,b];d={k:v[ix] for k,v in pool.items()};assert d['start'][:2048].all() and not d['start'][2048:].any()
d['half']=np.tile(np.r_[np.zeros(1024,dtype=np.int8),np.ones(1024,dtype=np.int8)],2)
data=json.loads((FACB/'data_identity.json').read_text());term=np.load(data['path'])['terminals'];ends=np.flatnonzero(term);d['trajectory']=np.searchsorted(ends,d['raw_index']);assert np.max(d['trajectory'])<len(ends)
for j,seed in enumerate([2026101711,2026101712]):d[f'z{j}']=np.random.default_rng(seed).standard_normal((4096,21)).astype('float32')
np.savez_compressed(R/'arrays/samples.npz',**d)
save('sample_manifest.json',dict(source=str(poolpath),source_sha256=sha(poolpath),selection_seed=2026101721,latent_seeds=[2026101711,2026101712],states=4096,start=2048,nonstart=2048,nonstart_selection='uniform without replacement from fixed global pool excluding start',region='next_observations xy in [-2,6) x [-2,6)',sample_sha256=sha(R/'arrays/samples.npz'),trajectory_count=int(len(np.unique(d['trajectory']))),raw_data=data,half_sensitivity='two predefined 1024-per-region halves; no extra states'))
protocol=dict(card='CRR-HISTORICAL-SEPARATION-Z0',branch='codex/crr-historical-separation-z0-20261017',branch_url='https://github.com/rollinpioneer/graph/tree/codex/crr-historical-separation-z0-20261017',training_updates_allowed=0,environment_steps_allowed=0,new_model_initializations_allowed=0,seeds=[0,3],fork_steps=[600000,300000],deltas=list(range(0,100001,10000)),states=4096,latent_repeats=2,checkpoint_arm_time_pairs=44,distinct_checkpoint_files=42,
 target_score='T=.5*(ensemble_mean target_Q(next_s,actor_fork(next_s,z))+ensemble_mean target_Q(next_s,vabc_fork(current_s,z))); fixed clipped actions at all later times',contrast='mean_start T - mean_nonstart T; DeltaC=B0-B1a',Q_difference='regional absolute and signed between-branch differences plus both arms drift from fork',gradient='norm grad_a ensemble_mean ONLINE Q(next_s,a), evaluated at fixed clipped fork Actor action; no derivative through clip or Actor; no guidance min, no lambda normalization',bootstrap=dict(unit='original raw data trajectory',replicates=400,seed=2026101731,interpretation='descriptive paired cluster intervals, not training-seed significance or a fitted alarm'),
 original_alarm_delta={'0':80000,'3':40000},finer_historical_s3_joint_delta=35000,alarm_caution='s0 start V rise>=20; s3 start-minus-all contrast rise>=15. Original alarms differ. Fine 5k joint s3 alarm begins335k. Never round 5k lead up to10k.',decision='Necessary-condition screen only; no fitted numerical risk threshold, no new training/gate/critic-only experiment. A visual separation is not automatically a predictor. Missing behavioral timing => INCONCLUSIVE for prospective performance claims.',source_gfp_commit='1abbd5f0e287caf9bda70e7bc655d0a499ac8401',known_limits=['B0 uses current online Actor, not EMA','B1a checkpoints store original GFP state without frozen-generator sidecar; sufficient for Critic reads, not asserted complete B1a resume','s3 fork already success9%, no healthy pre-decline premise','existing evaluations sparse and lack verified complete initial-state hashes','two latents and two probe halves are not independent training replications','start/nonstart are selected data probes, not on-policy state distribution'])
save('protocol.json',protocol)
card='''# CRR-HISTORICAL-SEPARATION-Z0 — execution freeze\n\nAuthorized by user: execute the supplied historical necessary-condition screen.\n\nBranch: [codex/crr-historical-separation-z0-20261017](https://github.com/rollinpioneer/graph/tree/codex/crr-historical-separation-z0-20261017).\n\nOnly read existing task1 s0 600k→700k and s3 300k→400k B0/B1a weights. All 10k checkpoints exist: 42 distinct files, 44 logical arm/time models. No training, initialization, optimizer use, environment construction/reset/step, new episodes or missing-timepoint reconstruction.\n\nB0 uses the current online Actor in TD; B1a holds the fork Actor parameters fixed only for TD. Both continue online Actor/VaBC and Critic learning historically. This is not the prior EMA R3 experiment. Saved plain GFP payloads contain the Critics needed here; frozen B1a generator sidecar is not present and complete B1a resumability is not claimed.\n\n4096 fixed actual-data transitions: 2048 start and2048 nonstart from the prior FACB fixed pool. Two fixed Gaussian arrays; fork Actor(next state) and VaBC(current state) generate actions once. Target scores average Critic heads and then the two fixed action sources. Online raw action-gradient uses ensemble mean at the clipped fork Actor action, not guidance_agg=min. Region is determined by next-state xy; no changed target_use_next semantics. Two preselected balanced halves provide sensitivity without extra states.\n\nDeliver three plot groups: DeltaC, fixed-action Q differences/drift, online gradient ratio. Preserve per-arm, per-region and per-latent numbers and paired trajectory-cluster descriptive intervals. No retrospectively fitted risk threshold. Shared-data/latent repeats do not provide independent training confirmation.\n\nFreeze original historical alarm markers separately: s0+80k(start onlineV rise20), s3+40k(start-minus-all rise15). Also display finer existing5k onset, especially s3 joint alarm+35k. A+30k difference is only5k before+35k; do not claim10k warning by rounding. Historical probe actions and start-minus-all contrasts differ from the new fixed-action start-minus-nonstart score.\n\nBehavioral evaluation is reused only. s3 was already weak at fork(9%); sparse later evaluations cannot locate the onset of success decline or establish prospective prediction. NO new behavior evaluation to fill gaps.\n\nGO would permit only recommending a separately scoped Critic-only response screen; no gate training is authorized. Late/one-trajectory separation lowers priority. Temporal insufficiency remains INCONCLUSIVE rather than changing the window.\n\nModel/file identity, counts, target mapping, finite values, fork equality and zero-operation counters must pass. Record actual CPU/GPU wall cost and true exit. Full arrays/weights stay server-side; explicit lightweight archive, ordinary commit/push and SHA verification, local handoff, close this round's sole SSH session.\n'''
(R/'EXPERIMENT_CARD.md').write_text(card)
save('receipts/preparation.json',dict(status='ASSET_READY',seconds=time.monotonic()-t,training_updates=0,environment_steps=0,new_initializations=0,all_10k_weights_present=True,pre_scoring_freeze=True))
print('PREPARED',len(records),len(set(r['path'] for r in records)),len(ev),flush=True)
