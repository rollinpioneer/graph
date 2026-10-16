import itertools, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
from common import ROOT,SUPPORT,write,digest,sha

class MDN(nn.Module):
 @nn.compact
 def __call__(self,x):
  for _ in range(2):x=nn.relu(nn.Dense(256)(x))
  y=nn.Dense(344)(x).reshape((-1,8,43))
  return jax.nn.log_softmax(y[:,:,0],-1),jnp.tanh(y[:,:,1:22]),jnp.clip(y[:,:,22:],-4,1)
@jax.jit
def nll(p,x,a):
 w,m,l=MDN().apply({'params':p},x)
 lp=-.5*jnp.sum(((a[:,None]-m)*jnp.exp(-l))**2+2*l+jnp.log(2*jnp.pi),-1)
 return -jax.scipy.special.logsumexp(w+lp,-1)/21

def project(d,g):
 # Two-constraint Euclidean projection; Gram computations use host float64.
 dn=np.asarray(d,dtype=np.float64);gn=np.asarray(g,dtype=np.float64)
 gram=gn@gn.T;rhs=gn@dn;candidates=[]
 scale=max(float(np.max(np.abs(rhs))),1e-12)
 for count in range(3):
  for active in itertools.combinations(range(2),count):
   lam=np.zeros(2)
   if active:
    ids=list(active);lam[ids]=np.linalg.lstsq(gram[np.ix_(ids,ids)],rhs[ids],rcond=1e-12)[0]
   residual=rhs-gram@lam
   if np.min(lam)>=-1e-9 and np.max(residual)<=1e-7*scale:
    candidate=dn-lam@gn
    candidates.append((float(np.sum((candidate-dn)**2)),candidate,lam,residual))
 assert candidates,'Projection active sets inconsistent'
 _,out,lam,residual=min(candidates,key=lambda x:x[0])
 return jnp.asarray(out,dtype=d.dtype),dict(multipliers=lam.tolist(),directional_density=residual.tolist(),original_directional_density=rhs.tolist(),direction_norm=float(np.linalg.norm(out)))

class Updater:
 def __init__(self,ag,nextobs,valid):
  self.template=ag;self.flat,self.unravel=ravel_pytree(ag.reference)
  norm=np.load(SUPPORT/'arrays/normalizer.npz');self.mean=jnp.asarray(norm['mean']);self.std=jnp.asarray(norm['std'])
  with open(SUPPORT/'models/mdn_1701.pkl','rb') as f:self.mdn=jax.tree.map(jnp.asarray,pickle.load(f)['params'])
  self.obs=nextobs
  xy=(np.asarray(nextobs[:,:2])+6)/4;cell=xy.astype(int)
  start=np.isin(cell[:,0],[1,2])&np.isin(cell[:,1],[1,2])
  self.pools=[np.flatnonzero(valid&start),np.flatnonzero(valid&~start)]
  assert min(map(len,self.pools))>=2048
  self.action=jax.jit(self._action)
  self.follow=jax.jit(lambda p,o,z,c:.5*jnp.mean((self._action(p,o,z)-c)**2))
  self.follow_grad=jax.jit(jax.grad(self.follow))
  self.density=jax.jit(self._density)
  self.density_grad=jax.jit(jax.jacrev(self._density))
  self.evaluate=jax.jit(self._evaluate)
  self.velocity=jax.jit(lambda p,d,o,z:jax.jvp(lambda x:self._action(x,o,z),(p,),(d,))[1])
 def _action(self,p,o,z):
  pars=dict(self.template.network.params);pars['modules_actor']=self.unravel(p)
  return jnp.clip(self.template.network.select('actor')(o,z,params=pars),-1,1)
 def _density(self,p,o,z):
  scores=nll(self.mdn,(o-self.mean)/self.std,self._action(p,o,z))
  return scores.reshape(2,-1).mean(1)
 def _evaluate(self,p,o,z,c,old):
  a=self._action(p,o,z);delta=a-old
  rms=jnp.sqrt(jnp.mean(delta.reshape(2,-1,21)**2,axis=(1,2)))
  return jnp.sqrt(jnp.mean(delta**2)),rms,.5*jnp.mean((a-c)**2),jnp.mean(jnp.abs(a)>=1)
 def samples(self,event,cal=False):
  root=jax.random.PRNGKey(6824000 if cal else 6824100)
  key=root if cal else jax.random.fold_in(root,event)
  n=1024 if cal else 256;out=[];records={}
  chosen=[]
  for r,pool in enumerate(self.pools):
   rk=jax.random.fold_in(key,r)
   ids=np.asarray(jax.random.choice(rk,jnp.asarray(pool),(n*(1 if cal else 2),),replace=False))
   chosen.append(ids)
  for part in range(1 if cal else 2):
   ids=np.concatenate([np.tile(x[part*n:(part+1)*n],2) for x in chosen])
   z=np.concatenate([np.asarray(jax.random.normal(jax.random.fold_in(jax.random.fold_in(key,10+part),r),(2*n,21))) for r in range(2)])
   out.append((jnp.asarray(self.obs[ids]),jnp.asarray(z)))
   records['indices_'+str(part)]=ids;records['noise_'+str(part)]=z
  p=ROOT/'arrays'/('calibration_inputs.npz' if cal else f'event_{event:04d}.npz')
  p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**records)
  if not cal:assert not np.intersect1d(records['indices_0'],records['indices_1']).size
  return out,sha(p)
 def calibrate(self,ag):
  [(o,z)],ident=self.samples(0,True)
  p,_=ravel_pytree(ag.reference);online,_=ravel_pytree(ag.network.params['modules_actor'])
  cal=(1-.00005)**100*p+(1-(1-.00005)**100)*online
  base=self.action(p,o,z);r,rr,_,_=self.evaluate(cal,o,z,base,base)
  self.rho=float(r);self.regional=np.asarray(rr)
  assert self.rho>0 and np.all(self.regional>0)
  row=dict(rho0=self.rho,regional_start_outside=self.regional.tolist(),input_sha256=ident,parameter_updates=0,reference_unchanged=True)
  write(ROOT/'receipts/calibration.json',row);return row
 def propose(self,ag,fit,check,arm):
  t=time.time();p,_=ravel_pytree(ag.reference);online,_=ravel_pytree(ag.network.params['modules_actor'])
  fo,fz=fit;o,z=check;c=self.action(online,o,z);fc=self.action(online,fo,fz);old=self.action(p,o,z)
  d=-self.follow_grad(p,fo,fz,fc);projection=None
  if arm=='B':d,projection=project(d,self.density_grad(p,fo,fz))
  baseline=float(self.follow(p,o,z,c));density=np.asarray(self.density(p,o,z)) if arm=='B' else None
  velocity=self.velocity(p,d,o,z);alpha=self.rho/max(float(jnp.sqrt(jnp.mean(velocity**2))),1e-12)
  evaluations=[]
  def evaluate(step):
   x=p+step*d;r,rr,loss,clip=self.evaluate(x,o,z,c,old)
   r=float(r);rr=np.asarray(rr);loss=float(loss)
   density_now=np.asarray(self.density(x,o,z)) if arm=='B' else None
   finite=bool(np.isfinite([r,*rr,loss,float(clip)]).all() and (density_now is None or np.isfinite(density_now).all()))
   assert finite,'Nonfinite reference candidate'
   feasible=bool(r<=self.rho and np.all(rr<=self.regional) and loss<=baseline+1e-5 and (density is None or np.all(density_now-density<=1e-5)))
   row=dict(alpha=float(step),rms=r,regional=rr.tolist(),follow_loss=loss,clipped_fraction=float(clip),feasible=feasible,density_change=None if density is None else (density_now-density).tolist())
   evaluations.append(row);return row
  zero=evaluate(0.);best=zero;upper=alpha;lower=0.
  # Initial candidate, at most eight halvings, six initial interval refinements.
  for back in range(9):
   row=evaluate(alpha)
   if row['feasible']:best=row;lower=alpha;break
   upper=alpha;alpha/=2
  if lower>0 and upper>lower:
   for _ in range(6):
    mid=(lower+upper)/2;row=evaluate(mid)
    if row['feasible']:
     lower=mid
     if row['rms']>best['rms']:best=row
    else:upper=mid
  return dict(p=p,d=d,best=best,evaluate=evaluate,rows=evaluations,projection=projection,seconds=time.time()-t)
 def paired(self,agents,event):
  (fit,check),identity=self.samples(event);plans={k:self.propose(a,fit,check,k) for k,a in agents.items()}
  radius=min(self.rho,*[v['best']['rms'] for v in plans.values()]);result={};chosen={}
  for arm,plan in plans.items():
   best=plan['best'];candidates=[r for r in plan['rows'] if r['feasible']]
   low=0.;high=best['alpha']
   # Remaining six of the twelve refinements are used for paired matching.
   for _ in range(6):
    if radius==0:break
    nearest=min(candidates,key=lambda r:abs(r['rms']-radius))
    if abs(nearest['rms']-radius)<=.025*radius:break
    step=(low+high)/2;row=plan['evaluate'](step)
    if row['feasible']:candidates.append(row)
    if not row['feasible'] or row['rms']>radius:high=step
    else:low=step
   final=min(candidates,key=lambda r:abs(r['rms']-radius))
   if radius==0:final=next(r for r in candidates if r['alpha']==0)
   result[arm]=agents[arm].replace(reference=self.unravel(plan['p']+final['alpha']*plan['d']))
   chosen[arm]=dict(final=final,projection=plan['projection'],candidates=plan['rows'],seconds=plan['seconds'],reference_sha256=digest(result[arm].reference))
  a=np.array([chosen['A']['final']['rms'],*chosen['A']['final']['regional']]);b=np.array([chosen['B']['final']['rms'],*chosen['B']['final']['regional']])
  rel=np.abs(a-b)/np.maximum(np.maximum(a,b),1e-8)
  row=dict(event=event,updates=event*100,inputs_sha256=identity,rho_common=radius,rho_fraction=radius/self.rho,relative_diff_global_start_outside=rel.tolist(),global_match=bool(rel[0]<=.05),regional_match=bool(np.all(rel[1:]<=.2)),continuous=bool(radius>=.5*self.rho and np.all(a>0) and np.all(b>0)),arms=chosen)
  return result,row
