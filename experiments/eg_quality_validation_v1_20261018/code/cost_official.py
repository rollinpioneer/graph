"""Match the existing EconomicGrasp NPZ layout and quantization."""
exec(open('/home/xushijie3/work/eg_quality_validation_v1_20261018/code/cost.py').read().split('rows=[]; cache=')[0])
from scipy.spatial import cKDTree
rows=[]; cache=N/'cost_official_cache';cache.mkdir(exist_ok=False)
for s in [8,36]:
    ep=R/f'data/graspnet/economic_grasp_label_300views/scene_{s:04d}_labels.npz'
    with np.load(ep) as z:
        ix=np.linspace(0,len(z['points'])-1,256,dtype=int)
        base={k:z[k][ix] for k in z.files}
        full_points=len(z['points']); original_bytes=ep.stat().st_size
    lab=SceneLabels(s); classes=list(lab.obj)
    t=time.perf_counter(); mus=[]; widths=[]
    for k in range(len(ix)):
        ob=lab.obj[classes[int(base['pointid'][k])]]
        dist,p=cKDTree(ob['points']).query(base['points'][k]);assert dist<1e-7
        v=base['topview'][k].astype(int)
        m=np.asarray(ob['scores'][p,v]).copy();c=np.asarray(ob['collision'][p,v]);w=np.asarray(ob['offsets'][p,v])[...,2].copy()
        m[c|(m<0)]=0;m[(m<=0)|(w>.1)]=0;pos=m>0;m[pos]=1.1-m[pos]
        mus.append(m);widths.append(w)
    mu=np.stack(mus);w=np.stack(widths);extract=time.perf_counter()-t
    qm=(mu*10).astype(np.uint8);qw=(w*1000).astype(np.uint8)
    flat=mu.reshape(256,300,48);idx=flat.argmax(-1)
    rr=np.arange(256)[:,None];vv=np.arange(300)[None,:]
    assert np.array_equal(qm.reshape(256,300,48)[rr,vv,idx],base['scores'])
    assert np.array_equal(qw.reshape(256,300,48)[rr,vv,idx],base['widths'])
    assert np.array_equal(idx//4,base['rotations']) and np.array_equal(idx%4,base['depth'])
    common={k:base[k] for k in ('points','pointid','vgraspness','topview')}
    variants={'official_best':base,'rich48_uint8':dict(common,scores48=qm,widths48=qw),'rich48_float32':dict(common,scores48=mu,widths48=w)}
    rec=dict(scene=s,retained_points=256,full_retained_points=full_points,original_file_bytes=original_bytes,source_extract_s=extract,original_best_exact_match=True,variants={})
    for name,data in variants.items():
        path=cache/f'{s}_{name}.npz';t=time.perf_counter();np.savez(path,**data);write=time.perf_counter()-t
        read=[]
        for _ in range(3):
            t=time.perf_counter()
            with np.load(path) as z: copied={k:z[k] for k in z.files}
            read.append(time.perf_counter()-t)
        assert all(np.array_equal(v,copied[k]) for k,v in data.items())
        rec['variants'][name]=dict(bytes=path.stat().st_size,write_s=write,median_read_s=float(np.median(read)),all_read_s=read)
    rows.append(rec);print('OFFICIAL_COST',s,json.dumps(rec),flush=True)
(N/'cost_official_results.json').write_text(json.dumps(dict(rows=rows,notes=['Official uncompressed NPZ layout, equal retained points and view ordering; existing best score/width/angle/depth reproduced exactly.','uint8 uses the official quality and millimeter-width quantization; float32 retains source precision. Neither is a newly trained supervision scheme.','Shared metadata included. Source extraction, serialization, and warm materialized reads measured; no global OS cache flush.','Full-scene/dataset costs extrapolated only if explicitly marked.']),indent=2))
