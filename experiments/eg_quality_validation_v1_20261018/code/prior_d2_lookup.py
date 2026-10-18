"""D2: compare the quality of the *decoded* grasp (pred angle, depth) with the best (angle, depth)
at the same (object point, view), using the dense GraspNet labels. CPU only, read-only on data.

Usage: python d2_lookup.py --partition dev --roles subset --scenes 8,36 --stride 8 --out OUTDIR
"""
import os, sys, json, struct, zipfile, argparse, time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

R = Path('/home/xushijie3/work/eg_sub25_v1_20261015')
DATA = R / 'data/graspnet'
EVAL = R / 'runs/EG_SUB25_V1/evaluation'
PI12 = np.pi / 12


def views(n=300, phi=(np.sqrt(5) - 1) / 2):
    i = np.arange(n)
    z = (2 * i + 1) / n - 1
    x = np.sqrt(1 - z ** 2) * np.cos(2 * i * np.pi * phi)
    y = np.sqrt(1 - z ** 2) * np.sin(2 * i * np.pi * phi)
    return np.stack([x, y, z], 1).astype(np.float32)


V = views()


def member_mmap(path, key):
    """Memory-map one stored (uncompressed) array of an .npz without reading it."""
    path = str(path)
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + '.npy')
        assert info.compress_type == 0, 'compressed member cannot be mmapped'
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        n, m = struct.unpack('<HH', h[26:30])
        f.seek(info.header_offset + 30 + n + m)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
        off = f.tell()
    assert not fortran
    return np.memmap(path, dtype=dtype, mode='r', offset=off, shape=shape)


def _load_small(path, key):
    with np.load(path) as z:
        return z[key]


class SceneLabels:
    def __init__(self, scene):
        self.scene = scene
        sd = DATA / 'scenes' / f'scene_{scene:04d}'
        ids = [int(x) for x in open(sd / 'object_id_list.txt').read().split()]
        m0 = sio.loadmat(sd / 'kinect/meta/0000.mat')
        cls0 = m0['cls_indexes'].ravel().astype(int)
        assert list(cls0 - 1) == ids, (cls0, ids)
        cpath = DATA / 'collision_label' / f'scene_{scene:04d}' / 'collision_labels.npz'
        self.obj = {}
        for i, cid in enumerate(cls0):
            lp = DATA / 'grasp_label' / f'{cid - 1:03d}_labels.npz'
            self.obj[cid] = dict(
                points=_load_small(lp, 'points'),
                scores=member_mmap(lp, 'scores'),
                offsets=member_mmap(lp, 'offsets'),
                collision=member_mmap(cpath, f'arr_{i}'))
            assert self.obj[cid]['collision'].shape == self.obj[cid]['scores'].shape


def lookup_frame(lab, meta, pred):
    """pred: (1024,17). Returns dict of per-candidate arrays."""
    cls = meta['cls_indexes'].ravel().astype(int)
    poses = meta['poses']
    pts_cam, owner, local = [], [], []
    for k, cid in enumerate(cls):
        Rp, tp = poses[:3, :3, k], poses[:3, 3, k]
        p = lab.obj[cid]['points'] @ Rp.T + tp
        pts_cam.append(p); owner += [k] * len(p); local.append(np.arange(len(p)))
    pts_cam = np.concatenate(pts_cam); owner = np.array(owner); local = np.concatenate(local)
    tree = cKDTree(pts_cam)
    cen = pred[:, 13:16]
    dist, nn = tree.query(cen)
    n = len(pred)
    out = dict(dist=dist.astype(np.float32), obj=np.full(n, -1), mu_exec=np.zeros(n, np.float32),
               mu_best=np.zeros(n, np.float32), a_pred=np.zeros(n, int), d_pred=np.zeros(n, int),
               a_best=np.full(n, -1), d_best=np.full(n, -1), n_good=np.zeros(n, int),
               view_dot=np.zeros(n, np.float32), a_resid=np.zeros(n, np.float32),
               mu_exec_geo=np.zeros(n, np.float32), mu_best_geo=np.zeros(n, np.float32),
               a_geo=np.full(n, -1), a_best_geo=np.full(n, -1), d_best_geo=np.full(n, -1), a_geo_resid=np.zeros(n, np.float32))
    Rm = pred[:, 4:13].reshape(-1, 3, 3)
    ax = Rm[:, :, 0]
    vdot = ax @ (-V).T
    v_scene = vdot.argmax(1)
    out['view_dot'] = vdot.max(1).astype(np.float32)
    ay = np.stack([-ax[:, 1], ax[:, 0], np.zeros(n)], 1)
    ay_n = np.linalg.norm(ay, axis=1)
    ay[ay_n == 0] = [0, 1, 0]
    ay /= np.linalg.norm(ay, axis=1, keepdims=True)
    az = np.cross(ax, ay)
    c1 = Rm[:, :, 1]
    ang = np.arctan2((c1 * az).sum(1), (c1 * ay).sum(1)) % (2 * np.pi)
    a_idx = np.rint(ang / PI12).astype(int)
    out['a_resid'] = np.abs(ang - a_idx * PI12).astype(np.float32)
    a_idx = a_idx % 24
    d_idx = np.rint(pred[:, 3] / 0.01).astype(int) - 1
    out['a_pred'], out['d_pred'] = a_idx, d_idx
    for k, cid in enumerate(cls):
        sel = np.where(owner[nn] == k)[0]
        if len(sel) == 0: continue
        Rp = poses[:3, :3, k]
        Vt = V @ Rp.T
        jmap = (V @ Vt.T).argmax(1)                      # scene view -> object view
        pt = local[nn[sel]]
        j = jmap[v_scene[sel]]
        o = lab.obj[cid]
        S = np.asarray(o['scores'][pt, j]).astype(np.float32)          # (m,12,4)
        C = np.asarray(o['collision'][pt, j])
        W = np.asarray(o['offsets'][pt, j])[..., 2]
        S = np.where(C, 0, S); S = np.where(S < 0, 0, S)
        S = np.where((S > 0) & (W <= 0.1), S, 0)
        big = np.where(S > 0, S, 9.0).reshape(len(sel), -1)
        flat_best = big.argmin(1)
        mu_best = big[np.arange(len(sel)), flat_best]
        mu_best = np.where(mu_best > 5, 0, mu_best)
        ok_a = (a_idx[sel] < 12) & (d_idx[sel] >= 0) & (d_idx[sel] < 4)
        mu_exec = np.zeros(len(sel), np.float32)
        ii = np.where(ok_a)[0]
        mu_exec[ii] = S[ii, a_idx[sel][ii], d_idx[sel][ii]]
        # geometric lookup: express the decoded rotation in the object frame and take the nearest label cell
        Ro = np.einsum('ji,njk->nik', Rp, Rm[sel])                       # Rp^T @ R_pred
        axo = Ro[:, :, 0]
        jg = (axo @ (-V).T).argmax(1)
        # label rotation is built from the *template* view; use the template's own frame as the reference
        axt = -V[jg].astype(np.float64)
        ayt = np.stack([-axt[:, 1], axt[:, 0], np.zeros(len(sel))], 1)
        ayt[np.linalg.norm(ayt, axis=1) == 0] = [0, 1, 0]
        ayt /= np.linalg.norm(ayt, axis=1, keepdims=True)
        azt = np.cross(axt, ayt)
        c1o = Ro[:, :, 1]
        ango = np.arctan2((c1o * azt).sum(1), (c1o * ayt).sum(1)) % np.pi      # 180-degree gripper symmetry
        ag = np.rint(ango / PI12).astype(int)
        out['a_geo_resid'][sel] = np.abs(ango - ag * PI12)
        ag = ag % 12
        Sg = np.asarray(o['scores'][pt, jg]).astype(np.float32)
        Cg = np.asarray(o['collision'][pt, jg])
        Wg = np.asarray(o['offsets'][pt, jg])[..., 2]
        Sg = np.where(Cg, 0, Sg); Sg = np.where(Sg < 0, 0, Sg); Sg = np.where((Sg > 0) & (Wg <= 0.1), Sg, 0)
        bigg = np.where(Sg > 0, Sg, 9.0).reshape(len(sel), -1)
        fbg = bigg.argmin(1)
        mbg = bigg[np.arange(len(sel)), fbg]; mbg = np.where(mbg > 5, 0, mbg)
        me = np.zeros(len(sel), np.float32)
        okd = (d_idx[sel] >= 0) & (d_idx[sel] < 4)
        kk = np.where(okd)[0]
        me[kk] = Sg[kk, ag[kk], d_idx[sel][kk]]
        out['mu_exec_geo'][sel] = me; out['mu_best_geo'][sel] = mbg; out['a_geo'][sel] = ag
        out['a_best_geo'][sel] = np.where(mbg > 0, fbg // 4, -1); out['d_best_geo'][sel] = np.where(mbg > 0, fbg % 4, -1)
        out['obj'][sel] = cid
        out['mu_exec'][sel] = mu_exec
        out['mu_best'][sel] = mu_best
        out['a_best'][sel] = np.where(mu_best > 0, flat_best // 4, -1)
        out['d_best'][sel] = np.where(mu_best > 0, flat_best % 4, -1)
        out['n_good'][sel] = ((S > 0) & (S <= 0.4 + 1e-6)).reshape(len(sel), -1).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition'); ap.add_argument('--roles'); ap.add_argument('--scenes')
    ap.add_argument('--stride', type=int, default=8); ap.add_argument('--out')
    ap.add_argument('--max_frames', type=int, default=256); ap.add_argument('--pred_root', default=str(EVAL / 'predictions'))
    a = ap.parse_args()
    os.nice(10)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    roles = a.roles.split(',')
    for scene in [int(s) for s in a.scenes.split(',')]:
        t0 = time.time()
        lab = SceneLabels(scene)
        recs = {r: [] for r in roles}
        for f in range(0, a.max_frames, a.stride):
            meta = sio.loadmat(DATA / f'scenes/scene_{scene:04d}/kinect/meta/{f:04d}.mat')
            for r in roles:
                pred = np.load(Path(a.pred_root) / f'{r}/{a.partition}/scene_{scene:04d}/kinect/{f:04d}.npy').astype(np.float64)
                o = lookup_frame(lab, meta, pred)
                evp = EVAL / f'results/{r}/{a.partition}/scene_{scene:04d}/evaluated/{f:04d}.npz'
                if evp.exists():
                    ev = np.load(evp)
                else:
                    ev = dict(grasps=np.zeros((0, 17)), quality=np.zeros(0), collision=np.zeros(0, bool))
                g = ev['grasps']
                idx = np.array([np.argmin(np.abs(pred[:, 0] - x[0]) + np.abs(pred[:, 13:16] - x[13:16]).sum(1)) for x in g])
                pad = 50 - len(idx)
                idx = np.concatenate([idx, np.full(pad, -1)]).astype(int)
                evq = np.concatenate([ev['quality'], np.full(pad, -2.0)])
                evc = np.concatenate([ev['collision'], np.zeros(pad, bool)])
                rec = dict(frame=f, score=pred[:, 0].astype(np.float32), width=pred[:, 1].astype(np.float32),
                           eval_idx=idx, eval_quality=evq, eval_collision=evc, **o)
                recs[r].append(rec)
        for r in roles:
            keys = recs[r][0].keys()
            np.savez_compressed(out / f'{a.partition}_{r}_scene{scene:04d}.npz',
                                **{k: np.stack([x[k] for x in recs[r]]) for k in keys})
        print('SCENE_DONE', a.partition, scene, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
