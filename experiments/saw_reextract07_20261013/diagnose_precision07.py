"""Zero-update audit of native delegation versus algebra and GPU precision."""
from common07 import *
refs=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text());ref=load_ref(refs[0]['reference']);p,o=fresh(0);d=data_load();b,ix,_=Sampler(d,0).batch(1)
def native(a,pp,bb):
    full=dict(a.network.params);full.update(pp)
    aa,_=a.actor_loss(bb,full);ww,_=a.waypoint_loss(bb,full)
    return aa+ww,dict(awr=aa,waypoint=ww)
def audit(precision):
    with jax.default_matmul_precision(precision):
        (l,aux),g=jax.jit(jax.value_and_grad(native,argnums=1,has_aux=True))(ref,p,b)
        (ll,ours),gg=jax.jit(jax.value_and_grad(loss,argnums=1,has_aux=True))(ref,p,b)
        x=dict(precision=precision,official={k:float(v) for k,v in aux.items()},algebra={k:float(v) for k,v in ours.items()},loss_error=abs(float(l)-float(ll)),gradient_error=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(gg))),gradient_tolerance=1e-5+1e-5*max(float(np.max(np.abs(np.asarray(x)))) for x in jax.tree_util.tree_leaves(g)))
        jax.block_until_ready(g)
        return x
x=[audit('default'),audit('highest')]
write(ROOT/'results/P_PRECISION_DIAGNOSIS.json',dict(comparisons=x,updates=0,reference_sha256=tree_sha(ref.network.params)))
print(json.dumps(x),flush=True)
