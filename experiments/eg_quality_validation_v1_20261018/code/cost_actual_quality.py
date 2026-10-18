import time,json,os
from validate import Geometry,E,N,np
os.nice(10)
rows=[]
for s in [8,36]:
 t=time.perf_counter();ctx=Geometry(s);setup=time.perf_counter()-t
 for f in [0,64,128,192]:
  t=time.perf_counter();ctx.frame(f);frame_s=time.perf_counter()-t
  z=np.load(E/f'results/subset/dev/scene_{s:04d}/evaluated/{f:04d}.npz');g=z['grasps'];owner=ctx.owners(g)
  q=np.full(len(g),-1.);c=np.zeros(len(g),bool);t=time.perf_counter()
  for ob in np.unique(owner):
   ix=np.where(owner==ob)[0];qq,cc,_=ctx.score(g[ix],int(ob));q[ix]=qq;c[ix]=cc
  elapsed=time.perf_counter()-t
  assert np.array_equal(q,z['quality']) and np.array_equal(c,z['collision'])
  row=dict(scene=s,frame=f,n=len(g),scene_setup_s=setup,frame_geometry_s=frame_s,actual_quality_batch_s=elapsed,ms_per_candidate=elapsed/len(g)*1000);rows.append(row);print(row,flush=True)
(N/'cost_actual_quality.json').write_text(json.dumps(dict(rows=rows,notes=['Actual original candidates only; batched by native object assignment, using cached full-scene geometry and original SDF.','Four other single-thread workers ran concurrently. Timing is workload-specific; no GPU used.','This measures scoring existing candidates, not network inference or end-to-end training.','Exact native quality and collision reproduced on all measured candidates.']),indent=2))
