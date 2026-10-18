"""D1/D3: evaluate with the official GraspNetAPI machinery, but keep the *whole* per-object top-10 pool
(the official code cuts it to the global top-50 before scoring), for the saved predictions ("base") and for the same
predictions after the official model-free collision filter ("cd", collision_thresh=0.01, as in test.py).
CPU only. Frames 0,4,8,... of each scene."""
import os, sys, json, time, inspect, argparse
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
R = Path('/home/xushijie3/work/eg_sub25_v1_20261015'); D = R / 'data_recovery'; EVAL = R / 'runs/EG_SUB25_V1/evaluation'
ap_ = argparse.ArgumentParser(); ap_.add_argument('--role'); ap_.add_argument('--partition'); ap_.add_argument('--scene', type=int)
ap_.add_argument('--stride', type=int, default=4); ap_.add_argument('--out'); a = ap_.parse_args()
os.nice(10)
cfg = json.loads((D / 'evaluation_run_config.json').read_text()); scenes = cfg['partitions'][a.partition]; sidx = scenes.index(a.scene)
sys.argv = [sys.argv[0], '--dataset_root', str(R / 'data/graspnet'), '--camera', 'kinect', '--model', 'economicgrasp', '--batch_size', '4', '--split_manifest', str(D / 'frozen_split.json')]
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector
import graspnetAPI.graspnet_eval as GE
import graspnetAPI.utils.eval_utils as EU
from graspnetAPI import GraspGroup, GraspNetEval
src = inspect.getsource(EU.eval_grasp)
assert 'remain_mask_i = pre_grasp_list[i][:,0] >= min_score' in src
src = src.replace('remain_mask_i = pre_grasp_list[i][:,0] >= min_score', 'remain_mask_i = pre_grasp_list[i][:,0] >= -1e18').replace('def eval_grasp(', 'def eval_grasp_pool(')
ns = EU.__dict__; exec(src, ns); eval_grasp_pool = ns['eval_grasp_pool']
data = GraspNetDataset(str(R / 'data/graspnet'), split=a.partition, camera='kinect', num_points=20000, voxel_size=.005, remove_outlier=True, augment=False, load_label=False)
ge = GraspNetEval(root=str(R / 'data/graspnet'), camera='kinect', split='all')
config = GE.get_config(); table = GE.create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
model_list, dexmodel_list, _ = ge.get_scene_models(a.scene, ann_id=0)
sampled = [GE.voxel_sample_points(m, 0.008) for m in model_list]
rec = {v: dict(frame=[], obj=[], conf=[], q=[], coll=[]) for v in ('base', 'cd')}; removed = []
t0 = time.time()
for f in range(0, 256, a.stride):
    _, pose_list, camera_pose, align_mat = ge.get_model_poses(a.scene, f)
    table_trans = GE.transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    pred = np.load(EVAL / f'predictions/{a.role}/{a.partition}/scene_{a.scene:04d}/kinect/{f:04d}.npy') if a.role == 'subset' or (EVAL / f'predictions/{a.role}/{a.partition}').exists() else None
    for variant in ('base', 'cd'):
        gg = GraspGroup(pred.copy())
        if variant == 'cd':
            cloud, _c = data.get_data(sidx * 256 + f, return_raw_cloud=True)
            det = ModelFreeCollisionDetector(cloud, voxel_size=0.005)
            mask = det.detect(gg, approach_dist=0.05, collision_thresh=0.01)
            removed.append(int(mask.sum())); gg = gg[~mask]
        arr = gg.grasp_group_array
        if len(arr) == 0: continue
        arr[arr[:, 1] < 0, 1] = 0; arr[arr[:, 1] > 0.1, 1] = 0.1; gg.grasp_group_array = arr
        gl, sl, cl = eval_grasp_pool(gg, sampled, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008, TOP_K=50)
        for o, (g, s, c) in enumerate(zip(gl, sl, cl)):
            if len(g) == 0: continue
            r = rec[variant]; n = len(g)
            r['frame'].append(np.full(n, f)); r['obj'].append(np.full(n, o)); r['conf'].append(g[:, 0]); r['q'].append(np.asarray(s, float)); r['coll'].append(np.asarray(c, bool))
    print('FRAME', a.role, a.partition, a.scene, f, f'{time.time()-t0:.0f}s', flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / f'{a.role}_{a.partition}_scene{a.scene:04d}.npz', cd_removed=np.array(removed),
                    **{f'{v}_{k}': (np.concatenate(x) if x else np.zeros(0)) for v in rec for k, x in rec[v].items()})
print('SCENE_DONE', a.role, a.partition, a.scene, f'{time.time()-t0:.0f}s', flush=True)
