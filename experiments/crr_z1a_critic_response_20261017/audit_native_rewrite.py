from common import *
import time,resource
seed=int(sys.argv[1]);before=650000 if seed in [0,1] else 310000;after=before+10000
O=R/f'arrays/native_audit_s{seed}';O.mkdir(exist_ok=False);t=time.monotonic()
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

b={k:v[:256] for k,v in data.items()}
l,nl,ge=validation(cp,tp,tds['F'],vabc,b,jnp.asarray(post['rng']));out=dict(seed=seed,rewritten_loss=float(l),native_loss=float(nl),gradient_max_abs_error=float(ge),updates=0);save(R/f'receipts/native_rewrite_audit_s{seed}.json',out);print(out,flush=True)
