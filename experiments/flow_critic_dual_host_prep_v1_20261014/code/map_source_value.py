"""Cross-environment module mapping only: original task/discount, no host hot start."""
import hashlib,json,pickle,sys
from pathlib import Path
import flax,jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspaces/gfp_floq'))
from utils.flax_utils import ModuleDict
from floq_value_core.core import network_entries,support_config,euler_integrate,aggregate_ensemble
cfg=json.loads((Path('/home/xushijie3/work/floq_min_repro_v2_20261013/runs/formal_seed0')/'actual_config.json').read_text())
cfg=support_config(cfg,cfg['r_min'],cfg['r_max'])
results=[]
for seed in [0,1]:
 for epoch in [1000000,2000000]:
    file=ROOT/'tests_zero_train'/f'floq_source_value_seed{seed}_epoch{epoch}.pkl'
    before=hashlib.sha256(file.read_bytes()).hexdigest()
    params=pickle.loads(file.read_bytes())
    with np.load(file.with_suffix('.npz')) as data:
        obs=jnp.asarray(data['obs']);actions=jnp.asarray(data['actions']);ratios=jnp.asarray(data['ratios'])
        entries=network_entries(cfg,obs,actions,guidance_target=False)
        definition=ModuleDict({k:v[0] for k,v in entries.items()})
        initialized=definition.init(jax.random.PRNGKey(14),**{k:v[1] for k,v in entries.items()})['params']
        shape_a=jax.tree_util.tree_map(lambda x:list(x.shape),initialized)
        shape_b=jax.tree_util.tree_map(lambda x:list(x.shape),params)
        assert shape_a==shape_b
        def apply(name,*a,**kw):return definition.apply({'params':params},*a,name=name,**kw)
        q=apply('critic',obs,actions)
        grad=jax.grad(lambda x:apply('critic',obs,x).mean(0).sum())(actions)
        initial=jnp.tile((cfg['noise_min']*(1-ratios)+cfg['noise_max']*ratios)[None],(2,1,1))
        flow=euler_integrate(lambda *a,**kw:apply('floq',*a,**kw),obs,actions,initial,8)
        target=aggregate_ensemble(euler_integrate(lambda *a,**kw:apply('target_floq',*a,**kw),obs,actions,initial,8),cfg['q_agg'])
        diffs={}
        for key,value in [('q',q),('grad',grad),('flow',flow),('target',target)]:
            np.testing.assert_allclose(np.asarray(value),data[key],rtol=5e-4,atol=5e-4)
            diffs[key]=float(np.max(np.abs(np.asarray(value)-data[key])))
    assert before==hashlib.sha256(file.read_bytes()).hexdigest()
    results.append(dict(seed=seed,epoch=epoch,status='PASS',all_value_keys_mapped=list(params),missing=[],
                        atol=5e-4,rtol=5e-4,max_abs_difference=diffs,source_discount=cfg['discount'],
                        purpose='original floq value module equivalence only; no host value reuse or optimizer restore'))
    print('VALUE_MAPPING_PASS',seed,epoch,diffs,flush=True)
(ROOT/'reports/source_value_mapping.json').write_text(json.dumps(dict(status='PASS',optimizer_steps=0,results=results),indent=2))
