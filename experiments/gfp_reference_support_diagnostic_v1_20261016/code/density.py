from util import *
import jax,jax.numpy as jnp,optax,flax.linen as nn,pickle,resource
class MDN(nn.Module):
 @nn.compact
 def __call__(self,x):
  for _ in range(2):x=nn.relu(nn.Dense(256)(x))
  out=nn.Dense(8*(1+42))(x).reshape((-1,8,43))
  return jax.nn.log_softmax(out[:,:,0],axis=-1),jnp.tanh(out[:,:,1:22]),jnp.clip(out[:,:,22:],-4,1)
net=MDN()
@jax.jit
def nll(p,x,a):
 lw,mu,ls=net.apply({'params':p},x)
 lp=-.5*jnp.sum(((a[:,None,:]-mu)*jnp.exp(-ls))**2+2*ls+jnp.log(2*jnp.pi),axis=-1)
 return -jax.scipy.special.logsumexp(lw+lp,axis=-1)/21
def scores(p,o,a,mean,std):
 out=[]
 for i in range(0,len(o),512):out.append(np.asarray(nll(p,(o[i:i+512]-mean)/std,a[i:i+512])))
 return np.concatenate(out) if out else np.zeros(0)
def fit(seed):
 t0=time.time();o,a,t,_,_=raw('train');ids=np.flatnonzero(~t)
 norm=np.load(ROOT/'arrays/normalizer.npz');mean,std=norm['mean'],norm['std']
 cal=np.load(ROOT/'arrays/calibration.npz');p=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,69)))['params']
 tx=optax.adam(3e-4);opt=tx.init(p);rng=np.random.default_rng(seed);best=float('inf');rows=[]
 @jax.jit
 def step(p,opt,x,y):
  loss,g=jax.value_and_grad(lambda v:nll(v,x,y).mean())(p)
  up,opt=tx.update(g,opt,p);return optax.apply_updates(p,up),opt,loss
 for it in range(15001):
  if it%1000==0:
   v=scores(p,cal['obs'],cal['actions'],mean,std);val=float(v.mean());assert np.isfinite(v).all()
   rows.append(dict(step=it,validation_nll=val,seconds=time.time()-t0))
   if val<best:
    best=val
    with open(ROOT/'models'/f'mdn_{seed}.pkl','wb') as f:pickle.dump(dict(params=jax.device_get(p),best_step=it,seed=seed),f)
   write(f'receipts/fit_{seed}.json',dict(rows=rows,supervised_updates=it,gfp_updates=0,environment_steps=0,device=str(jax.devices()),pid=__import__('os').getpid()))
   print('FIT',seed,it,val,'seconds',time.time()-t0,flush=True)
  if it==15000:break
  batch=rng.choice(ids,512,replace=True);p,opt,loss=step(p,opt,(o[batch]-mean)/std,a[batch]);assert np.isfinite(float(loss))
 write(f'receipts/fit_{seed}_complete.json',dict(seconds=time.time()-t0,supervised_updates=15000,gfp_updates=0,environment_steps=0,model_sha256=sha(ROOT/'models'/f'mdn_{seed}.pkl'),peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':
 import sys
 fit(int(sys.argv[1]))
