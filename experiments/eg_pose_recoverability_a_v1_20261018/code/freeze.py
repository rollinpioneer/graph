from core import *
ck=OLD/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar';assert sha(ck)=='9eb793cfdcaf18cf6dfd1a41294a97db9eddf2b3e97cc208ab21009ad1d71950'
rows=[];hashes={}
for s in SCENES:
 p=PRIOR/f'decoder_control/{s:04d}.json';hashes[str(p.relative_to(G))]=sha(p);fail={x['frame']:x for x in json.loads(p.read_text())['rows']}
 for f in FRAMES:
  path=E/f'results/subset/dev/scene_{s:04d}/evaluated/{f:04d}.npz';z=np.load(path);g=z['grasps'][0];q=float(z['quality'][0]);col=bool(z['collision'][0]);good=0<q<=.4000001
  assert (f not in fail)==good
  if not good:
   v=np.asarray(fail[f]['q']);di=int(np.argmin(abs(np.arange(1,5)*.01-g[3])));assert v[di]==q
  rawpath=E/f'predictions/subset/dev/scene_{s:04d}/kinect/{f:04d}.npy';raw=np.load(rawpath);match=np.flatnonzero(np.max(np.abs(raw[:,1:]-g[None,1:]),axis=1)<1e-6);assert len(match)==1
  rows.append(dict(scene=s,frame=f,raw_index=int(match[0]),original_quality=q,original_collision=col,success=good))
  for p in [path,rawpath]+[E/f'inputs/dev/scene_{s:04d}/kinect/{j:04d}.npz' for j in range(f,f+4)]:hashes[str(p.relative_to(G))]=sha(p)
assert len(rows)==640 and sum(not x['success'] for x in rows)==187
write(R/'results/cohort.json',rows);write(R/'results/source_hashes.json',hashes);write(R/'results/freeze.json',dict(status='PASS',registered_commit='546c6d70b19e3c8b87ec4437d3ea32f779fb750e',cases=640,failures=187,successes=453,checkpoint_sha256=sha(ck),cohort_sha256=sha(R/'results/cohort.json'),source_hashes_sha256=sha(R/'results/source_hashes.json'),created=time.time()))
print('FROZEN 640 cases')
