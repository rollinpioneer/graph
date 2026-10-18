"""D5b: for colliding grasps in the evaluated top-50, apply the smallest (width, depth, in-plane angle) edit that removes
the collision (centre and approach direction fixed), re-score it with the official force-closure code, and recompute AP
with the colliding grasps replaced. Oracle-style ceiling for local geometry refinement. CPU only."""
import os, sys, json, time, argparse
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
R = Path('/home/xushijie3/work/eg_sub25_v1_20261015'); D = R / 'data_recovery'; EVAL = R / 'runs/EG_SUB25_V1/evaluation'
ap_ = argparse.ArgumentParser(); ap_.add_argument('--role'); ap_.add_argument('--partition'); ap_.add_argument('--scene', type=int)
ap_.add_argument('--stride', type=int, default=8); ap_.add_argument('--out'); a = ap_.parse_args(); os.nice(10)
import graspnetAPI.graspnet_eval as GE
from graspnetAPI import GraspNetEval
import graspnetAPI.utils.eval_utils as EU
from graspnetAPI.utils.eval_utils import get_grasp_score
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
ge = GraspNetEval(root=str(R / 'data/graspnet'), camera='kinect', split='all')
table = GE.create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
model_list, dexmodel_list, _ = ge.get_scene_models(a.scene, ann_id=0)
sampled = [GE.voxel_sample_points(m, 0.008) for m in model_list]; nm = len(sampled)
config = GE.get_config(); fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2]); fcq = {}
for v in fc_list:
    v = round(v, 2); config['metrics']['force_closure']['friction_coef'] = v; fcq[v] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
H, DB, FW = 0.02, 0.02, 0.01
WIDTHS = np.round(np.arange(0.02, 0.1001, 0.01), 3); DEPTHS = np.array([0.01, 0.02, 0.03, 0.04]); ANGLES = np.deg2rad([0, 15, -15, 30, -30])
FR = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
def masks(t, w, d):
    x, y, z = t[..., 0], t[..., 1], t[..., 2]
    m1 = (z > -H / 2) & (z < H / 2); m2 = (x > -DB) & (x < d)
    m3 = y > -(w / 2 + FW); m4 = y < -w / 2; m5 = y < (w / 2 + FW); m6 = y > w / 2; m7 = (x > -(DB + FW)) & (x < -DB)
    return ((m1 & m2 & m3 & m4) | (m1 & m2 & m5 & m6)), (m1 & m3 & m5 & m7), (m1 & m2 & (~m4) & (~m6))
def rot(Rm, ang):
    c, s = np.cos(ang), np.sin(ang); return Rm @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ap_of(q):
    acc = np.zeros((50, 6)); c = [np.cumsum((q > 0) & (q <= fr)) for fr in FR]
    for j in range(6):
        for k in range(50): acc[k, j] = (c[j][min(k, len(q) - 1)] if len(q) else 0) / (k + 1)
    return acc
rows = []; apb = []; apr = []; t0 = time.time()
for f in range(0, 256, a.stride):
    ev = np.load(EVAL / f'results/{a.role}/{a.partition}/scene_{a.scene:04d}/evaluated/{f:04d}.npz')
    g, q, coll = ev['grasps'], ev['quality'].copy(), ev['collision']
    if len(g) == 0: continue
    _, pose_list, camera_pose, align_mat = ge.get_model_poses(a.scene, f)
    table_trans = GE.transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    mt = [GE.transform_points(m, pose_list[i]) for i, m in enumerate(sampled)]
    seg = np.concatenate([i * np.ones(len(m), np.int32) for i, m in enumerate(mt)]); objpts = np.concatenate(mt)
    owner = seg[EU.compute_closest_points(g[:, 13:16], objpts)]
    allpts = np.concatenate([objpts, table_trans])
    q_new = q.copy()
    for k in np.where(coll)[0]:
        i = owner[k]; m = mt[i]; lo, hi = m.min(0) - 0.05, m.max(0) + 0.05
        P = allpts[np.all((allpts > lo) & (allpts < hi), axis=1)]
        gp, Rm, w0, d0 = g[k, 13:16], g[k, 4:13].reshape(3, 3), g[k, 1], g[k, 3]
        cand = []
        for ang in ANGLES:
            t2 = (P - gp) @ rot(Rm, ang)
            W_ = WIDTHS.reshape(-1, 1, 1); D_ = DEPTHS.reshape(1, -1, 1)
            fin, bot, inn = masks(t2[None, None], W_, D_)
            free = ~(fin | bot).any(-1) & (inn.sum(-1) >= 10)
            for wi, di in zip(*np.where(free)):
                cost = abs(np.rad2deg(ang)) / 15 + abs(WIDTHS[wi] - w0) / 0.01 + abs(DEPTHS[di] - d0) / 0.01
                cand.append((cost, ang, WIDTHS[wi], DEPTHS[di]))
        rec = dict(frame=f, rank=int(k), found=bool(cand), q_old=float(q[k]))
        if cand:
            cand.sort(key=lambda x: x[0]); best = -1.0; first = None
            for n_, (cost, ang, w, d) in enumerate(cand[:3]):
                row = g[k].copy(); row[1] = w; row[3] = d; row[4:13] = rot(Rm, ang).reshape(-1)
                gl = [np.zeros((0, 17))] * nm; gl[i] = row[None]
                cm, em, dg = EU.collision_detection(gl, mt, dexmodel_list, pose_list, np.concatenate([objpts, table_trans]), outlier=0.05, return_dexgrasps=True)
                if cm[i][0] or dg[i][0] is None: s = -1.0
                else: s = float(get_grasp_score(dg[i][0], dexmodel_list[i], fc_list, fcq))
                if n_ == 0: first = s
                if s > 0 and (best < 0 or s < best): best = s
            rec.update(q_first=first, q_best3=best, cost=float(cand[0][0]), n_free=len(cand))
            q_new[k] = best if best > 0 else (first if first is not None else -1.0)
        rows.append(rec)
    apb.append(ap_of(q)); apr.append(ap_of(q_new))
    if f % 32 == 0: print('FRAME', a.role, a.partition, a.scene, f, f'{time.time()-t0:.0f}s', flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump(dict(rows=rows, ap_base=np.stack(apb).tolist(), ap_refined=np.stack(apr).tolist()), open(out / f'{a.role}_{a.partition}_scene{a.scene:04d}.json', 'w'))
print('SCENE_DONE', a.role, a.partition, a.scene, len(rows), f'{time.time()-t0:.0f}s', flush=True)
