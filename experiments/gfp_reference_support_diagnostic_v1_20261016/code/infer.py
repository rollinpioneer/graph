from density import *
import sys,importlib.util,gc
from scipy.stats import rankdata
def auc(real,negative):
 ranks=rankdata(np.r_[real,negative]);n,m=len(real),len(negative)
 return float((ranks[n:].sum()-m*(m+1)/2)/(n*m))
t0=time.time();norm=np.load(ROOT/'arrays/normalizer.npz');mean,std=norm['mean'],norm['std']
models={s:pickle.load(open(ROOT/'models'/f'mdn_{s}.pkl','rb'))['params'] for s in [1701,1702]}
cal=np.load(ROOT/'arrays/calibration.npz');test=np.load(ROOT/'arrays/test.npz');q=np.load(ROOT/'arrays/queries.npz');reliability=[];cuts={}
for seed,p in models.items():
 c=scores(p,cal['obs'],cal['actions'],mean,std);cuts[seed]=np.array([np.quantile(c[cal['region']==r],.95) for r in [0,1]])
 v=scores(p,test['obs'],test['actions'],mean,std);rng=np.random.default_rng(20261017)
 shuffled=scores(p,test['obs'],rng.permutation(test['actions']),mean,std)
 uniform=scores(p,test['obs'],rng.uniform(-1,1,test['actions'].shape).astype('float32'),mean,std)
 rows=[]
 for file in sorted((ROOT/'arrays').glob('positive_*.npz')):
  if '_scores_' in file.stem:continue
  d=np.load(file);values=scores(p,d['obs'],d['actions'],mean,std);ntraj=len(np.unique(d['trajectory']))
  reject=float(np.mean(values>cuts[seed][d['region']])) if len(values) else None
  rows.append(dict(group=file.stem,n=len(values),trajectories=ntraj,false_reject=reject,status='INSUFFICIENT_TRAJECTORIES' if ntraj<5 else ('PASS' if reject<=.2 else 'UNRELIABLE')))
  np.savez(ROOT/'arrays'/f'{file.stem}_scores_{seed}.npz',nll=values)
 aa,ab=auc(v,shuffled),auc(v,uniform)
 reliability.append(dict(seed=seed,auc_permuted=aa,auc_uniform=ab,actual_test_reject=float(np.mean(v>cuts[seed][test['region']])),cutoffs=cuts[seed],positive_controls=rows,basic_pass=aa>=.65 and ab>=.9))
 np.savez(ROOT/'arrays'/f'reliability_{seed}.npz',actual=v,permuted=shuffled,uniform=uniform,calibration=c)
write('reports/reliability.json',reliability);print('RELIABILITY',json.dumps(reliability,default=lambda x:x.tolist()),flush=True)
# Only frozen reference Actor forwards: no GFP losses, optimizer calls or Critic scores.
source=WORK/'gfp_bfc_dev_screen_v1_20261016'
sys.path.insert(0,str(source/'code'));sys.path.insert(0,str(source/'source/gfp'))
spec=importlib.util.spec_from_file_location('bfc_original',source/'code/common.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
ag,cfg,_=original.make_agent(3,'R3')
@jax.jit
def actor(ref,obs,z):
 params=dict(ag.network.params);params['modules_actor']=ref
 return jnp.clip(ag.network.select('actor')(obs,z,params=params),-1,1)
nquery=len(q['obs'])
noise=[np.asarray(jax.random.normal(jax.random.PRNGKey(k),(nquery,21))) for k in [72401,72402]]
np.savez(ROOT/'arrays/noise.npz',z0=noise[0],z1=noise[1])
identities=[];fixed=[];initial=None
for arm in ['L0','L1a','L1b']:
 parent=WORK/('gfp_midwindow_td_freeze_v1_20261016' if arm=='L0' else 'gfp_midwindow_td_split_v1_20261016')
 for it in range(0,50001,5000):
  path=parent/'runs'/f's3_{arm}'/f'checkpoint_{it}.pkl';meta=json.loads(path.with_suffix('.json').read_text());before=sha(path);assert before==meta['sha256']
  with path.open('rb') as f:d=pickle.load(f)
  assert d['updates']==it and int(d['agent']['network']['step'])==350001+it
  ref=d['agent']['reference'];outs=[]
  for z in noise:
   a=np.concatenate([np.asarray(actor(ref,q['obs'][i:i+512],z[i:i+512])) for i in range(0,nquery,512)])
   assert np.isfinite(a).all();outs.append(a)
  outs=np.asarray(outs)
  if it==0:
   if initial is None:initial=outs.copy()
   else:assert np.array_equal(initial,outs)
  if arm=='L1a':assert np.array_equal(initial,outs);fixed.append(it)
  values=np.stack([[scores(p,q['obs'],a,mean,std) for a in outs] for p in models.values()])
  real=np.stack([scores(p,q['obs'],q['actions'],mean,std) for p in models.values()])
  np.savez(ROOT/'arrays'/f'{arm}_{it}.npz',nll=values,real_nll=real,actions=outs)
  after=sha(path);assert before==after
  identities.append(dict(arm=arm,updates=it,path=str(path),sha256=before,unchanged=True))
  write('receipts/inference_progress.json',dict(arm=arm,updates=it,seconds=time.time()-t0,gfp_updates=0,environment_steps=0))
  print('INFER',arm,it,flush=True);del d,ref;gc.collect()
write('receipts/checkpoint_identity.json',identities)
write('receipts/inference_complete.json',dict(seconds=time.time()-t0,reference_action_evaluations=33*nquery*2,frozen_L1a_exact=True,all_initial_actions_equal=True,gfp_updates=0,environment_steps=0,noise_sha256=sha(ROOT/'arrays/noise.npz'),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
