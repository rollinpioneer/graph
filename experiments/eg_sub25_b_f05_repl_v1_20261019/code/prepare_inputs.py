import argparse,os,time,json,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);args=p.parse_args();scene=args.scene
from repl_common import *
assert scene in sum(P['C3'].values(),[]);start=time.time();os.nice(10)
from infer import EconomicExtractor,setup_original
setup_original(CODE,DATA)
from dataset.graspnet_dataset import GraspNetDataset
ex=EconomicExtractor.__new__(EconomicExtractor);ex.dataset_root=DATA
ex.data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
folder=R/f'inputs/scene_{scene:04d}';folder.mkdir(parents=True,exist_ok=True);hashes=[]
for frame in range(256):
 target=folder/f'{frame:04d}.npz';assert not target.exists(),target
 sample=ex.sample(scene,frame);assert sample['point_clouds'].shape==(20000,3)
 assert all(np.isfinite(v).all() for v in sample.values())
 np.savez_compressed(target,**sample);hashes.append(dict(frame=frame,sha256=sha256(target)))
write(R/f'results/input_preparation_{scene:04d}.json',dict(status='PASS',scene=scene,frames=256,seconds=time.time()-start,cpu_seconds=time.process_time(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,identities=hashes,model_loaded=False,quality_or_performance_computed=False))
print('INPUT_PREPARED',scene,flush=True)
