"""Measured bounded cache extraction cost, no new scoring algorithm."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time,json,resource
from pathlib import Path
import numpy as np
R=Path('/home/xushijie3/work/eg_sub25_v1_20261015')
N=Path('/home/xushijie3/work/eg_quality_validation_v1_20261018')
sys.path.insert(0,str(R/'d2d4_20261017'))
from d2_lookup import SceneLabels

def extract(lab):
    mus=[]; widths=[]; total=0
    for cid,o in sorted(lab.obj.items()):
        np_=len(o['points']); total+=np_*300
        ix=np.linspace(0,np_-1,min(64,np_),dtype=int)
        m=np.asarray(o['scores'][ix]).reshape(-1,48).copy()
        c=np.asarray(o['collision'][ix]).reshape(-1,48)
        w=np.asarray(o['offsets'][ix])[...,2].reshape(-1,48).copy()
        m[(m<=0)|c|(w>.1)]=-1
        mus.append(m); widths.append(w)
    return np.concatenate(mus),np.concatenate(widths),total

rows=[]; cache=N/'cost_cache'; cache.mkdir(exist_ok=False)
for s in [8,36]:
    t=time.perf_counter(); lab=SceneLabels(s); init=time.perf_counter()-t
    t=time.perf_counter(); mu,w,total=extract(lab); first=time.perf_counter()-t
    t=time.perf_counter(); mu2,w2,_=extract(lab); warm=time.perf_counter()-t
    assert np.array_equal(mu,mu2) and np.array_equal(w,w2)
    t=time.perf_counter(); idx=np.where(mu>0,mu,99).argmin(1)
    rr=np.arange(len(mu)); best=dict(mu=mu[rr,idx],width=w[rr,idx],angle=(idx//4).astype(np.uint8),depth=(idx%4).astype(np.uint8))
    reduce_time=time.perf_counter()-t
    rich=dict(mu=mu,width=w)
    rec=dict(scene=s,objects=len(lab.obj),sampled_pointviews=len(mu),full_scene_pointviews=total,init_s=init,first_source_read_s=first,warm_source_read_s=warm,reduce_best_s=reduce_time,variants={})
    for name,data in [('best',best),('rich48',rich)]:
        path=cache/f'{s}_{name}.npz'; t=time.perf_counter(); np.savez_compressed(path,**data); write=time.perf_counter()-t
        reads=[]
        for rep in range(3):
            t=time.perf_counter()
            with np.load(path) as z: copied={k:z[k] for k in z.files}
            reads.append(time.perf_counter()-t)
        assert all(np.array_equal(v,copied[k]) for k,v in data.items())
        rec['variants'][name]=dict(bytes=path.stat().st_size,uncompressed_bytes=sum(v.nbytes for v in data.values()),compressed_write_s=write,read_s=reads,median_read_s=float(np.median(reads)))
    b=rec['variants']['best']; h=rec['variants']['rich48'];rec['compressed_size_ratio']=h['bytes']/b['bytes'];rec['median_read_ratio']=h['median_read_s']/b['median_read_s']
    rows.append(rec); print('COST',s,json.dumps(rec),flush=True)
(N/'cost_results.json').write_text(json.dumps(dict(rows=rows,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,notes=['Source files already exist: this measures richer-label cache extraction, not first-principles force closure generation.','First read is process-first access, not guaranteed cold OS cache; caches were not dropped on the shared server.','Same sampled point/view rows in both formats; all 48 qualities and widths stored without quantization.','Compressed NPZ loads materialize arrays; actual training throughput requires a training integration study, not performed.','Sample size 64 uniformly selected points per scene object, all 300 views; not a full-scene or full-dataset measurement.']),indent=2))
