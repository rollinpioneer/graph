from pathlib import Path
import os,json,pickle,time,resource
import numpy as np,jax,jax.numpy as jnp
from flax import linen as nn
import sys
sys.path.insert(0,'/home/xushijie3/work/gfp_eta0_ablation_20261015/source/gfp')
from utils.networks import Value
from utils.flax_utils import ModuleDict
r=Path(__file__).resolve().parent;t=time.monotonic();row=next(x for x in json.loads((r/'checkpoint_manifest.json').read_text()) if x['id']=='I_s0_500000');p=pickle.load(open(row['path'],'rb'))['agent']['value']['params'];x=np.load(r/'arrays/samples.npz')['next_observations'][:256]
definition=ModuleDict(dict(value=Value((512,)*4,layer_norm=True,num_ensembles=1)));direct=Value((512,)*4,layer_norm=True,num_ensembles=1)
f=lambda pa,o:definition.apply({'params':pa},o,name='value')
g=lambda pa,o:direct.apply({'params':pa['modules_value']},o)
a=np.asarray(jax.jit(f)(p,x));b=np.asarray(g(p,x));c=np.asarray(jax.jit(g)(p,x));d=np.asarray(f(p,x))
out=dict(jitted_module_vs_eager_direct=float(np.max(abs(a-b))),jitted_module_vs_jitted_direct=float(np.max(abs(a-c))),eager_module_vs_eager_direct=float(np.max(abs(d-b))),value_range=[float(a.min()),float(a.max())],all_finite=bool(np.isfinite([a,b,c,d]).all()),wall_seconds=time.monotonic()-t,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,training_updates=0,environment_operations=0)
(r/'receipts/compiler_boundary_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
