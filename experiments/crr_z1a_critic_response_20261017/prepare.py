from common import *
import time,shutil,importlib.metadata as md
t=time.monotonic();assert md.version('ogbench')=='1.1.0'
from ogbench.utils import load_dataset
identity=json.loads(Path('/home/xushijie3/work/facb_z0_20261017/data_identity.json').read_text());p=Path(identity['path']);assert sha(p)==identity['sha256']
d=load_dataset(str(p),add_info=True);success=(np.linalg.norm(d['qpos'][:,:2]-np.array([20.,20.]),axis=1)<=.5).astype('float32')
out={k:d[k] for k in ['observations','next_observations']};out.update(actions=np.clip(d['actions'],-1+1e-5,1-1e-5),rewards=success-1.,masks=1.-success)
assert len(out['observations'])==2000000 and all(np.isfinite(v).all() for v in out.values())
for k,v in out.items():np.save(R/f'arrays/data_{k}.npy',v)
src=Path('/home/xushijie3/work/crr_historical_separation_z0_20261017/arrays/samples.npz');shutil.copy2(src,R/'arrays/probes.npz');probes=dict(np.load(src))
for k in out:np.testing.assert_array_equal(out[k][probes['dataset_index']],probes[k])
save(R/'data_manifest.json',dict(original=identity,prepared={k:dict(shape=list(v.shape),sha256=sha(R/f'arrays/data_{k}.npy')) for k,v in out.items()},probes_source=str(src),probes_sha256=sha(src),training_states=len(success),seconds=time.monotonic()-t,environment_operations=0))
sources=[]
for p in [SRC/'agents/gfp.py',SRC/'utils/networks.py',SRC/'utils/flax_utils.py',SRC/'utils/datasets.py']:
 if p.exists():
  dst=R/'sources'/p.name;shutil.copy2(p,dst);sources.append(dict(path=str(p),sha256=sha(p),copy=str(dst.relative_to(R))))
save(R/'source_manifest.json',dict(commit='1abbd5f0e287caf9bda70e7bc655d0a499ac8401',files=sources))
save(R/'receipts/preparation.json',dict(status='PASS',wall_seconds=time.monotonic()-t,dataset_matches_prior_probe_arrays=True,environment_operations=0))
print('PREPARED',flush=True)
