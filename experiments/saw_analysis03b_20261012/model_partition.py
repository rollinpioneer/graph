"""Read-only A/B module identity; separates actor changes from teacher/value changes."""
import csv,json,pickle,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=Path('/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012')
def leaves(x):
    if isinstance(x,dict):
        for k in sorted(x):yield from leaves(x[k])
    else:yield np.asarray(x)
def main():
    rows=[]
    for kind in ('navigate','stitch'):
        registered=ROOT/'results'/f'{kind}_checkpoints.json'
        if not registered.exists():continue
        bcp=json.loads(registered.read_text())
        for step in (100000,300000,500000,1000000):
            paths=[x for x in bcp if x['step']==step]
            if not paths:continue
            bpath=next((x['path'] for x in paths if Path(x['path']).name==f'params_{step}.pkl'),paths[0]['path'])
            with open(OLD/'runs'/f'formal_{kind}'/f'params_{step}.pkl','rb') as f:sa=pickle.load(f)
            with open(bpath,'rb') as f:sb=pickle.load(f)
            a,b=sa['agent']['network'],sb['agent']['network']
            numpy_equal=all(np.array_equal(x,y) for x,y in zip(sa['numpy_rng'],sb['numpy_rng']))
            jax_equal=np.array_equal(sa['agent']['rng'],sb['agent']['rng'])
            assert int(a['step'])==int(b['step'])==step+1
            assert set(a['params'])==set(b['params'])
            for module in sorted(a['params']):
                aa=list(leaves(a['params'][module]));bb=list(leaves(b['params'][module]));assert len(aa)==len(bb)
                maximum=0.;ss=0.;den=0.;count=0;exact=True
                for x,y in zip(aa,bb):
                    assert x.shape==y.shape;delta=x.astype(np.float64)-y.astype(np.float64)
                    maximum=max(maximum,float(np.max(np.abs(delta))));ss+=float(np.sum(delta**2));den+=float(np.sum(x.astype(np.float64)**2));count+=x.size;exact &= np.array_equal(x,y)
                rows.append(dict(kind=kind,checkpoint_step=step,module=module,arrays=len(aa),parameters=count,exact_equal=exact,max_abs=maximum,RMS=float(np.sqrt(ss/count)),relative_L2=float(np.sqrt(ss/(den+1e-30))),
                    checkpoint_numpy_rng_equal=numpy_equal,checkpoint_jax_rng_equal=jax_equal,
                    interpretation='module identity only; complete SGD sequence not inferred'))
    with open(ROOT/'results/MODULE_IDENTITY.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('MODULE_IDENTITY_MEASURED',len(rows),flush=True)
if __name__=='__main__':main()
