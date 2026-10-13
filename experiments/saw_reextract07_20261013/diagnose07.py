"""P technical diagnosis, zero updates, dynamic-input official comparison."""
from common07 import *
refs=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text());ref=load_ref(refs[0]['reference']);p,o=fresh(0);d=data_load();b,ix,_=Sampler(d,0).batch(1)
def official(a,pp,bb):
    full=dict(a.network.params);full.update(pp)
    return a.actor_loss(bb,full)[0]+a.waypoint_loss(bb,full)[0]
def compare(label,fn):
    l,g=jax.jit(jax.value_and_grad(fn,argnums=1))(ref,p,b)
    (ll,_),gg=jax.jit(jax.value_and_grad(loss,argnums=1,has_aux=True))(ref,p,b)
    err=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(g),jax.tree_util.tree_leaves(gg)))
    mx=max(float(np.max(np.abs(np.asarray(x)))) for x in jax.tree_util.tree_leaves(g))
    return dict(label=label,official_loss=float(l),extractor_loss=float(ll),loss_error=abs(float(l)-float(ll)),loss_tolerance=1e-5+1e-5*abs(float(l)),gradient_max_error=err,gradient_tolerance=1e-5+1e-5*mx,gradients_finite=bool(finite((g,gg))))
def constant(a,pp,bb):return official(ref,pp,b)
x=[compare('official_closure_constant_reference_and_batch',constant),compare('official_dynamic_reference_and_batch',official)]
write(ROOT/'results/P_DIAGNOSIS.json',dict(comparisons=x,updates=0,batch_indices_sha256=tree_sha(ix),devices=[str(d) for d in jax.devices()],reference_full_state_sha256=tree_sha(flax.serialization.to_state_dict(ref))))
print(json.dumps(x),flush=True)
