"""Complete physical-angle controls on the same frozen training pairs only."""
import os,sys,json,time,concurrent.futures
import numpy as np
from evaluate import ROOT,Direct,variants,save_result,sha

def run(task):
 s,f=task;p=ROOT/f'evaluation/train_{s:04d}_{f:04d}.npz'
 with np.load(p) as z:out={k:z[k] for k in z.files}
 if 'P1_aligned_mu' in out:return {'scene':s,'frame':f,'cached':True}
 with np.load(ROOT/f'cache/train_{s:04d}_{f:04d}.npz') as z:a={k:z[k] for k in z.files}
 meta=json.loads(p.with_suffix('.json').read_text());oldsha=sha(p);t=time.time();ct=time.process_time()
 direct=Direct(s);direct.frame(f);vi,full=variants(a)['P5_aligned'];original=a['pred'][vi].astype(float)
 for name,dep in [('P1_aligned',False),('P4_aligned',True)]:
  rows=original.copy();rows[:,4:13]=full[:,4:13]
  if dep:rows[:,3]=full[:,3]
  assert np.array_equal(rows[:,13:16],original[:,13:16])
  assert np.allclose(rows[:,4:13].reshape(-1,3,3)[:,:,0],original[:,4:13].reshape(-1,3,3)[:,:,0],atol=2e-6)
  r=direct.score(rows);assert np.isfinite(r['mu']).all();save_result(out,name,vi,r)
 np.savez_compressed(p,**out)
 meta['before_physical_controls_sha256']=oldsha;meta['result_sha256']=sha(p)
 meta['physical_controls_cpu_seconds']=time.process_time()-ct;meta['physical_controls_wall_seconds']=time.time()-t
 meta['physical_controls_note']='Same fixed 16 training frames; P1/P4 rotate physical representative closing axis into prediction basis. Class-index controls retained.'
 p.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('DONE',s,f,flush=True);return meta

if __name__=='__main__':
 plan=json.loads((ROOT/'study_plan.json').read_text());tasks=[(s,f) for s in plan['training_scenes'] for f in plan['training_frames']]
 shard=int(sys.argv[1]);tasks=tasks[shard::2]
 with concurrent.futures.ProcessPoolExecutor(4) as pool:
  for r in pool.map(run,tasks):print(json.dumps(r),flush=True)
 print('COMPLETE',shard,flush=True)
