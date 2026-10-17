from common import *
import time,resource
seed=int(sys.argv[1]);before=650000 if seed in [0,1] else 310000;after=before+10000
O=R/f'runs/critic_s{seed}';O.mkdir(exist_ok=False);t=time.monotonic()
assets=json.loads((R/'asset_manifest.json').read_text())
def path(step):
 if seed in [1,2]:
  h=R/f'runs/healthy_s{seed}';assert json.loads((h/'exit.json').read_text())['exit_code']==0;return h/f'params_{step}.pkl'
 return Path(next(x['path'] for x in assets if x['seed']==seed and x['step']==step and x['role']=='bad_trajectory_candidate'))
old=read(path(before),before);post=read(path(after),after);oldhash=sha(path(before));posthash=sha(path(after));initial=digest(post)
config=cfg(seed);g=restore(post,seed);par=g.network.params;tds={'F':old['network']['params']['modules_actor'],'P':par['modules_actor'],'Fprime':old['network']['params']['modules_actor']};vabc=par['modules_vabc'];frozen={k:digest(v) for k,v in tds.items()};frozen['vabc']=digest(vabc);frozen['online_actor']=digest(par['modules_actor'])
data=dataset();N=len(data['observations']);D=dict(np.load(R/'arrays/probes.npz'));fixed=[]
for j in range(2):
 aa=[];va=[]
 for k in range(0,4096,256):
  a,v=fixed_actions(par,D['observations'][k:k+256],D['next_observations'][k:k+256],D[f'z{j}'][k:k+256]);aa.append(np.asarray(a));va.append(np.asarray(v))
 fixed.append(np.stack([np.concatenate(aa),np.concatenate(va)]))
fixed=np.array(fixed);np.save(O/'fixed_actions.npy',fixed)
tx=optax.adam(config['lr']);cp=par['modules_critic'];tp=par['modules_target_critic'];sopt=post['network']['opt_state']
osaved={'0':{'count':sopt['0']['count'],'mu':sopt['0']['mu']['modules_critic'],'nu':sopt['0']['nu']['modules_critic']},'1':sopt['1']}
opt=flax.serialization.from_state_dict(tx.init(cp),osaved);assert digest(flax.serialization.to_state_dict(opt))==digest(osaved)
states={k:(cp,tp,opt) for k in tds};initial_critic=digest((cp,tp,opt))
def loss(cp,tp,actor,vabc,b,key):
 _,srng=jax.random.split(key);z=jax.random.normal(srng,(256,21));obs=b['observations'];nobs=b['next_observations']
 aa=jnp.clip(A.apply({'params':actor},nobs,z),-1,1);va=z
 for i in range(10):va=va+V.apply({'params':vabc},obs,va,jnp.full((256,1),i/10),is_encoded=True)/10
 va=jnp.clip(va,-1,1);qa=C.apply({'params':tp},nobs,aa).mean(0);qv=C.apply({'params':tp},nobs,va).mean(0)
 y=b['rewards']+.995*b['masks']*(qa+qv)/2;q=C.apply({'params':cp},obs,b['actions']);return jnp.square(q-y).mean()
@jax.jit
def validation(cp,tp,actor,vabc,b,key):
 pp=dict(par);pp['modules_actor']=actor;pp['modules_critic']=cp;pp['modules_target_critic']=tp
 ng=g.replace(network=g.network.replace(params=pp))
 def nloss(p):
  gradparams=dict(pp);gradparams['modules_critic']=p;return ng.critic_loss(b,gradparams,key)[0]
 l,gr=jax.value_and_grad(loss)(cp,tp,actor,vabc,b,key);nl,ngr=jax.value_and_grad(nloss)(cp)
 ge=jnp.max(jnp.stack([jnp.max(jnp.abs(x-y)) for x,y in zip(jax.tree_util.tree_leaves(gr),jax.tree_util.tree_leaves(ngr))]));return l,nl,ge
testbatch={k:v[:256] for k,v in data.items()};checks=[]
for arm in ['F','P']:
 l,nl,ge=validation(cp,tp,tds[arm],vabc,testbatch,jnp.asarray(post['rng']));l,nl,ge=float(l),float(nl),float(ge);assert abs(l-nl)<=3e-5*max(1,abs(nl)) and ge<=3e-5
 checks.append(dict(arm=arm,loss=l,native_loss=nl,critic_gradient_max_abs_error=ge))
save(O/'zero_update_admission.json',dict(status='PASS',native_loss_gradient_checks=checks,initial_critic_Adam_target_digest=initial_critic,Adam_count=after,source_old=dict(path=str(path(before)),sha256=oldhash),source_post=dict(path=str(path(after)),sha256=posthash),TD_actor_hashes=frozen,updates=0))
savedkey=jnp.asarray(post['rng'])
@jax.jit
def fused(cp,tp,opt,actor,vabc,offset,data_base,data):
 losses=[];idxes=[];keys=[]
 for k in range(4):
  step=offset+k;data_key=jax.random.fold_in(jax.random.PRNGKey(data_base),step);ix=jax.random.randint(data_key,(256,),0,N);b={key:v[ix] for key,v in data.items()}
  tdkey=jax.random.fold_in(savedkey,step);l,gr=jax.value_and_grad(loss)(cp,tp,actor,vabc,b,tdkey);updates,opt=tx.update(gr,opt,cp)
  # Preserve official GFP.target_update: old online, not the newly optimized cp.
  newtp=jax.tree_util.tree_map(lambda p,t:p*config['tau']+t*(1-config['tau']),cp,tp);cp=optax.apply_updates(cp,updates);tp=newtp
  losses.append(l);idxes.append(ix);keys.append(tdkey)
 return cp,tp,opt,jnp.stack(losses),jnp.stack(idxes),jnp.stack(keys)
def takeprobe(arm,step):
 cp,tp,opt=states[arm];qs=[];gs=[]
 for j in range(2):
  qq=[];gg=[]
  for k in range(0,4096,256):
   q,grad=probe(cp,tp,D['next_observations'][k:k+256],fixed[j,0,k:k+256],fixed[j,1,k:k+256]);qq.append(np.asarray(q));gg.append(np.asarray(grad))
  qs.append(np.concatenate(qq));gs.append(np.concatenate(gg))
 q=np.stack(qs);grad=np.stack(gs);assert np.isfinite(q).all() and np.isfinite(grad).all()
 np.savez_compressed(O/f'probe_{arm}_{step:05d}.npz',q=q,grad=grad)
def checkpoint(arm,step):
 cp,tp,opt=states[arm];p=O/f'{arm}_critic_{step:05d}.pkl';obj=dict(critic_params=cp,target_critic_params=tp,critic_opt_state=flax.serialization.to_state_dict(opt),new_updates=step,source_global_critic_Adam_count=after,logical_step=after+step,source_path=str(path(after)),source_sha256=posthash,frozen_hashes=frozen,data_rng_base=(20261720 if arm=='Fprime' else 20261710)+seed,TD_rng_base=np.asarray(savedkey),fixed_actions_sha256=sha(O/'fixed_actions.npy'))
 atomic_pickle(p,obj)
 with p.open('rb') as f:back=pickle.load(f)
 restoredopt=flax.serialization.from_state_dict(tx.init(back['critic_params']),back['critic_opt_state']);assert digest((back['critic_params'],back['target_critic_params'],restoredopt))==digest(states[arm])
 assert int(restoredopt[0].count)==after+step
 return dict(path=str(p),sha256=sha(p),updates=step,exact_custom_resume=True)
checkpoints=[]
for arm in tds:takeprobe(arm,0)
baseline=np.load(O/'probe_F_00000.npz');assert all(np.array_equal(np.load(O/f'probe_{a}_00000.npz')['grad'],baseline['grad']) for a in tds)
for offset in range(0,5000,4):
 datahash={};tdhash={}
 for arm in ['F','P','Fprime']:
  cp,tp,opt=states[arm];result=fused(cp,tp,opt,tds[arm],vabc,jnp.asarray(offset),jnp.asarray((20261720 if arm=='Fprime' else 20261710)+seed),data);cp,tp,opt,ls,ix,keys=result
  assert np.isfinite(np.asarray(ls)).all();states[arm]=(cp,tp,opt);datahash[arm]=digest(np.asarray(ix));tdhash[arm]=digest(np.asarray(keys))
  assert int(opt[0].count)==after+offset+4
  if offset==0:
   assert digest(cp)!=digest(par['modules_critic']);checkpoints.append(checkpoint(arm,4))
 assert datahash['F']==datahash['P'] and tdhash['F']==tdhash['P']==tdhash['Fprime']
 if offset==0:save(O/'first_four_updates.json',dict(status='PASS',per_arm_updates=4,counted_in_each_5000=True,full_Critic_Adam_restored=True,F_P_batches_identical=True,TD_noise_all_arms_identical=True,Fprime_changes_minibatch_only=True,custom_checkpoint_roundtrip=True))
 step=offset+4
 if step%1000==0:
  for arm in tds:
   assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(states[arm]));takeprobe(arm,step)
  save(O/'progress.json',dict(step=step,total_updates_all_three=3*step,limit=15000,seconds=time.monotonic()-t));print('CRITIC',seed,step,round(time.monotonic()-t,1),flush=True)
for arm in tds:checkpoints.append(checkpoint(arm,5000))
assert digest(post)==initial and sha(path(before))==oldhash and sha(path(after))==posthash
assert all(digest(tds[a])==frozen[a] for a in tds) and digest(vabc)==frozen['vabc'] and digest(par['modules_actor'])==frozen['online_actor']
save(O/'complete.json',dict(status='COMPLETE',seed=seed,critic_only_updates=15000,per_arm=5000,actor_updates=0,vabc_updates=0,TD_generators_frozen=True,old_source_sha256=oldhash,new_source_sha256=posthash,source_payload_unchanged=True,environment_operations=0,seconds=time.monotonic()-t,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoints=checkpoints))
print('CRITIC_COMPLETE',seed,flush=True)
