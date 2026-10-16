import time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from common import ROOT,write,digest,sha

class Updater:
 def __init__(self,ag,nextobs,valid):
  self.template=ag;self.flat,self.unravel=ravel_pytree(ag.reference)
  self.obs=nextobs
  xy=(np.asarray(nextobs[:,:2])+6)/4;cell=xy.astype(int)
  start=np.isin(cell[:,0],[1,2])&np.isin(cell[:,1],[1,2])
  self.pools=[np.flatnonzero(valid&start),np.flatnonzero(valid&~start)]
  assert min(map(len,self.pools))>=2048
  self.action=jax.jit(self._action)
  self.follow=jax.jit(lambda p,o,z,c:.5*jnp.mean((self._action(p,o,z)-c)**2))
  self.follow_grad=jax.jit(jax.grad(self.follow))
  self.evaluate=jax.jit(self._evaluate)
  self.velocity=jax.jit(lambda p,d,o,z:jax.jvp(lambda x:self._action(x,o,z),(p,),(d,))[1])
 def _action(self,p,o,z):
  pars=dict(self.template.network.params);pars['modules_actor']=self.unravel(p)
  return jnp.clip(self.template.network.select('actor')(o,z,params=pars),-1,1)
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
 def propose(self,ag,fit,check):
  t=time.time();p,_=ravel_pytree(ag.reference);online,_=ravel_pytree(ag.network.params['modules_actor'])
  fo,fz=fit;o,z=check;c=self.action(online,o,z);fc=self.action(online,fo,fz);old=self.action(p,o,z)
  d=-self.follow_grad(p,fo,fz,fc);projection=None
  baseline=float(self.follow(p,o,z,c));density=None
  velocity=self.velocity(p,d,o,z);alpha=self.rho/max(float(jnp.sqrt(jnp.mean(velocity**2))),1e-12)
  evaluations=[]
  def evaluate(step):
   x=p+step*d;r,rr,loss,clip=self.evaluate(x,o,z,c,old)
   r=float(r);rr=np.asarray(rr);loss=float(loss)
   density_now=None
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
 def independent(self,agents,event):
  assert set(agents)=={'A'}
  (fit,check),identity=self.samples(event)
  plan=self.propose(agents['A'],fit,check);final=plan['best']
  updated=agents['A'].replace(reference=self.unravel(plan['p']+final['alpha']*plan['d']))
  row=dict(event=event,updates=event*100,inputs_sha256=identity,rho_fraction=final['rms']/self.rho,
    continuous=bool(final['rms']>=.5*self.rho and np.all(np.asarray(final['regional'])>0)),
    final=final,candidates=plan['rows'],seconds=plan['seconds'],reference_sha256=digest(updated.reference),
    density_model_loaded=False,paired_radius=False)
  return {'A':updated},row
