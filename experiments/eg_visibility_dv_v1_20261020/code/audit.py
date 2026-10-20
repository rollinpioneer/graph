from common import *
from concurrent.futures import ProcessPoolExecutor
def job(scene):
 start=time.time();entries=[]
 for frame in range(256):
  ep=export(scene,frame);hashed=sha(ep);ref=[];qs=[]
  for root in [LG,CP,H]:
   p=root/f'evaluation/development/{scene:04d}/{frame:04d}.npz';jp=p.with_suffix('.json')
   values=json.load(open(jp));assert all(v['export_sha256']==hashed for v in values)
   z=np.load(p);qs.append((z['quality'],z['collision'],z['empty']))
   ref.append(dict(experiment=root.name,npz_sha256=sha(p),json_sha256=sha(jp)))
  for old in qs[:2]:
   m=np.isfinite(old[0]);assert np.allclose(old[0][m],qs[-1][0][m],atol=1e-7)
   assert np.array_equal(old[1][m],qs[-1][1][m]) and np.array_equal(old[2][m],qs[-1][2][m])
  z=np.load(R/f'cache/c3/{scene:04d}/{frame:04d}.npz');ids=z['ids']
  assert np.allclose(z['mu'],qs[-1][0][ids]) and np.array_equal(z['collision'],qs[-1][1][ids])
  entries.append(dict(frame=frame,export_sha256=hashed,references=ref))
 write(R/f'results/identity_C3_{scene}.json',dict(scene=scene,frames=entries,seconds=time.time()-start));return time.time()-start
if __name__=='__main__':
 os.nice(10)
 with ProcessPoolExecutor(max_workers=12) as pool:seconds=list(pool.map(job,C3))
 write(R/'results/identity_audit.json',dict(status='PASS',frames=3072,legacy_labels_exact=True,export_binding_exact=True,worker_seconds=sum(seconds)))
