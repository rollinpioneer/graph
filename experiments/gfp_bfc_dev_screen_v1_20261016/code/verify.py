import json,numpy as np,jax,jax.numpy as jnp
from common import ROOT,write,load_base
from bfc import BFCAgent,target

def main():
    tests={}
    r=jnp.array([-1.,0.,-1.]);m=jnp.array([1.,0.,1.]);qc=jnp.array([4.,9.,-8.]);qr=jnp.array([1.,-3.,2.]);qf=jnp.array([2.,2.,3.]);g=.995
    for e in (0.,.1,.5,float('inf')):
        y,ref,d=target(r,m,g,qc,qr,qf,e)
        assert np.allclose(y,ref+np.clip(d,-e,e));assert float(y[1])==float(r[1])
        if e==0:assert np.array_equal(y,ref)
        if np.isinf(e):assert np.allclose(y,r+g*m*(qc+qf)/2)
    assert np.all(np.asarray(jax.grad(lambda q:target(r,m,g,q,qr,qf,.1)[0].sum())(qc))==0)
    tests['target_limits_masks_bound_stop_gradient']='PASS'
    ag,cfg,path=load_base(0)
    batch={p.stem:jnp.asarray(np.load(p,mmap_mode='r')[:256]) for p in (ROOT/'data').glob('*.npy')}
    combined=BFCAgent(ag.rng,ag.network,ag.config,reference=ag.network.params['modules_actor'],epsilon=.1)
    key=jax.random.PRNGKey(19)
    old,oi=ag.critic_loss(batch,ag.network.params,key);new,ni=combined.critic_loss(batch,ag.network.params,key)
    assert np.allclose(old,new,rtol=2e-6,atol=2e-6)
    go=jax.grad(lambda p:ag.total_loss(batch,p,key)[0])(ag.network.params)
    gn=jax.grad(lambda p:combined.total_loss(batch,p,key)[0])(ag.network.params)
    diff=max(float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree_util.tree_leaves(go),jax.tree_util.tree_leaves(gn)))
    assert all(np.allclose(a,b,rtol=3e-5,atol=3e-5) for a,b in zip(jax.tree_util.tree_leaves(go),jax.tree_util.tree_leaves(gn)))
    tests['identical_actor_original_loss_gradient']=dict(passed=True,max_abs_gradient_difference=diff)
    shifted=jax.tree_util.tree_map(lambda p:p+.001,combined.reference)
    inf=combined.replace(reference=shifted,epsilon=float('inf'))
    x,_=inf.critic_loss(batch,ag.network.params,key)
    assert np.allclose(x,old,rtol=3e-5,atol=3e-5)
    tests['different_reference_infinite_bound_original_target']='PASS'
    # Reference receives no gradient through TD; online Actor/VaBC/target remain outside critic grad_params.
    gr=jax.grad(lambda ref:combined.replace(reference=ref).critic_loss(batch,ag.network.params,key)[0])(shifted)
    assert all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(gr))
    gc=jax.grad(lambda p:combined.critic_loss(batch,p,key)[0])(ag.network.params)
    for module in ('modules_actor','modules_vabc','modules_target_critic'):
        assert all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(gc[module]))
    tests['critic_gradient_routes']='PASS'
    # Four EMA applications must differ from one application to the final actor.
    old=np.array([1.,2.]);new=[np.array([3.+k,4.+k]) for k in range(4)];ref=old.copy();tau=.0005
    for a in new:ref=(1-tau)*ref+tau*a
    expected=(1-tau)**4*old+sum(tau*(1-tau)**(3-k)*a for k,a in enumerate(new))
    assert np.allclose(ref,expected) and not np.allclose(ref,(1-tau)*old+tau*new[-1])
    tests['EMA_per_logical_update_formula']='PASS'
    write(ROOT/'receipts/zero_update_admission.json',dict(passed=True,tests=tests,optimizer_updates=0,
      device=str(jax.devices()[0]),note='Actual fused4 updates and full restore tested within each allocated 100k branch, not additional updates.'))
    print('PASS',tests,flush=True)

if __name__=='__main__':main()
