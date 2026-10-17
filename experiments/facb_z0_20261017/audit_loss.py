from pathlib import Path
code=Path(__file__).with_name('infer.py').read_text().split('# First verify and evaluate')[0]
exec(compile(code,'infer_definitions','exec'))
row=next(r for r in records if r['id']=='G_s4_1000000');p,old=read(row);g=make_g(row,p)
b={k:jnp.asarray(samples[k][:256]) for k in ('observations','next_observations','actions','rewards','masks')};result=[]
for base in protocol['rng_bases']:
 key=jax.random.fold_in(jax.random.PRNGKey(base),0);native,info=native_critic_loss(g,b,key);direct=reconstructed_critic_loss(g,b,key);err=float(abs(native-direct));result.append(dict(base=base,native=float(native),reconstructed=float(direct),absolute_error=err));np.testing.assert_allclose(native,direct,rtol=3e-5,atol=3e-5)
finish(row,p,old);save('receipts/loss_boundary_audit.json',dict(results=result,passed=True,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,seconds_after_imports=time.monotonic()-t0,training_updates=0,environment_operations=0));print(result)
