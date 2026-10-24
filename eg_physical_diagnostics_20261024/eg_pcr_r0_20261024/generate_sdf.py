import os,sys,json,time
from pathlib import Path
ROOT=Path.home()/'work/grasp/eg_pcr_r0_20261024'
sys.path.insert(0,str(ROOT/'deps'))
os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np,trimesh
from grid_sample import grid_sample_sdf
OLD=ROOT.parent/'eg_pose_quality_diagnosis_v1_20261022';DATA=ROOT.parent/'eg_sub25_v1_20261015/data/graspnet'
plan=json.loads((OLD/'study_plan.json').read_text());objects=set()
for s in plan['C3_scenes']:
 for f in plan['C3_frames']:
  e=np.load(OLD/f'evaluation/C3_{s:04d}_{f:04d}.npz');ii=e['final_ids'];ii=ii[e['P0_mu'][ii]>0];objects.update(e['object_id'][ii].tolist())
def generate(oid):
 out=ROOT/f'sdf/{oid:03d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'grid_sampled_sdf.npz').exists():return
 np.random.seed(20261024+oid);start=time.time();mesh=trimesh.load(DATA/f'models/{oid:03d}/nontextured.ply',process=False);grid_sample_sdf(mesh,str(out));print('DONE',oid,time.time()-start,flush=True)

if __name__=='__main__':
 from concurrent.futures import ProcessPoolExecutor
 tasks=sorted(objects)[int(sys.argv[1])::3]
 with ProcessPoolExecutor(int(sys.argv[2])) as pool:list(pool.map(generate,tasks))
