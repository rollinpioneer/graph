"""Timing-only replay of fixed N_ref; compare outputs, never fit any parameter."""
import json,time,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.io import loadmat
from PIL import Image
from proj_sdf_ref import gripper_samples,query,classify
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text());D=R.parent/'eg_sub25_v1_20261015/data/graspnet'
def run(task):
 s,f=task;out=R/f'timing/{s:04d}_{f:04d}.json'
 if out.exists():return
 a=np.load(R/f'cpu/C3_{s:04d}_{f:04d}.npz');rows=np.concatenate([a['pred'].reshape(-1,17),a['raw']]);p=D/f'scenes/scene_{s:04d}/kinect'
 m=loadmat(p/f'meta/{f:04d}.mat',variable_names=['intrinsic_matrix','factor_depth']);K=m['intrinsic_matrix'];depth=np.array(Image.open(p/f'depth/{f:04d}.png'),np.float32)/float(m['factor_depth'].item())
 sample_seconds=0.;query_seconds=0.;occ=[];unk=[]
 for row in rows:
  t=time.perf_counter();pts=gripper_samples(row);sample_seconds+=time.perf_counter()-t
  t=time.perf_counter();phi,weight=query(pts,depth,K);st=classify(phi,weight);occ.append(np.mean(st==1));unk.append(np.mean(st==2));query_seconds+=time.perf_counter()-t
 assert np.array_equal(occ,a['ref_occ']) and np.array_equal(unk,a['ref_unk'])
 out.write_text(json.dumps(dict(scene=s,frame=f,poses=len(rows),sampling_seconds=sample_seconds,query_seconds=query_seconds,total_seconds=sample_seconds+query_seconds,GPU_MiB=0,identical_scores=True)))
if __name__=='__main__':
 (R/'timing').mkdir(exist_ok=True)
 with ProcessPoolExecutor(4) as p:list(p.map(run,[(s,f) for s in P['C3_scenes'] for f in P['C3_frames']]))
