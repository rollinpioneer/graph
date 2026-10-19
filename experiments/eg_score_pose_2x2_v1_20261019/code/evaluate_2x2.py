import os, sys, json, time, hashlib, resource, inspect, argparse
from pathlib import Path
import numpy as np

R = Path(os.environ['P2_ROOT'])
V = Path(os.environ['PREV_ROOT'])
G = Path(os.environ['GRASP_ROOT'])
P = G / 'eg_pose_select_pilot_v1_20261018'
os.environ['SELECT_ROOT'] = str(P)
sys.path.insert(0, str(P / 'code'))
from core import export, Native, ap, sha, write, select, make_model, CODE, DATA, N

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=int, required=True)
parser.add_argument('--micro', action='store_true')
args = parser.parse_args()
os.nice(10)
start, cpu_start = time.time(), time.process_time()
cfg = json.loads((R / 'frozen_config.json').read_text())
locked = json.loads((V / 'results/locked_method.json').read_text())
manifest = {x['path']: x['sha256'] for x in json.loads((V / 'results/artifact_manifest.json').read_text())}
assert [h['seed'] for h in locked['heads']] == cfg['seeds']
for h in locked['heads']:
    assert h['step'] == cfg['head_step']
    assert sha(P / f'heads/{h["seed"]}/step{h["step"]}.pt') == h['sha256']
assert sha(V / 'code/evaluate_v2.py') == manifest['code/evaluate_v2.py']

sys.path[:0] = [str(CODE), str(CODE / 'libs/pointnet2')]
sys.argv = [sys.argv[0], '--dataset_root', str(DATA), '--camera', 'kinect', '--model', 'economicgrasp', '--batch_size', '1']
from graspnetAPI import GraspGroup
import graspnetAPI.utils.eval_utils as EU
from dataset.graspnet_dataset import GraspNetDataset
from utils.collision_detector import ModelFreeCollisionDetector

source = inspect.getsource(EU.eval_grasp)
source_hash = hashlib.sha256(source.encode()).hexdigest()
prefix = source[:source.index('    ## collision detection')]
ns = EU.__dict__.copy()
exec(prefix.replace('def eval_grasp(', 'def select_only(') + '\n    return grasp_list\n', ns)
select_only = ns['select_only']
native = Native(args.scene)
dataset = GraspNetDataset(str(DATA), split='train', camera='kinect', num_points=20000, voxel_size=.005, remove_outlier=True, augment=False, load_label=False)
previous_complete = json.loads((V / f'evaluation/development/{args.scene:04d}/COMPLETE.json').read_text())
assert previous_complete['native_eval_source_sha256'] == source_hash
folder = R / f'evaluation/{"micro" if args.micro else "development"}/{args.scene:04d}'
folder.mkdir(parents=True, exist_ok=True)
frames = cfg['frames'][:1] if args.micro else cfg['frames']
new_calls = cached_calls = 0
checks = []
source_rows = []

for frame in frames:
    pred, hyp, feature, actions, action_ids = export(args.scene, frame)
    export_path = N / f'exports/subset/scene_{args.scene:04d}/{frame:04d}.npz'
    export_hash = sha(export_path)
    previous_relative = f'evaluation/development/{args.scene:04d}/{frame:04d}.npz'
    previous_path = V / previous_relative
    assert sha(previous_path) == manifest[previous_relative]
    with np.load(previous_path) as old:
        assert str(old['source_export_sha256']) == export_hash
        mu = old['mu'].reshape(-1).copy()
        collision = old['collision'].reshape(-1).copy()
        empty = old['empty'].reshape(-1).copy()
        choices = {'R0': np.zeros(1024, dtype=int)}
        rows = {'R0': pred.astype(np.float64).copy()}
        for h in locked['heads']:
            seed = h['seed']
            q = old[f'Q_{seed}'].copy()
            assert q.shape == (1024, 6) and np.isfinite(q).all()
            if args.micro:
                import torch
                torch.set_num_threads(1)
                checkpoint = torch.load(P / f'heads/{seed}/step{h["step"]}.pt', map_location='cpu', weights_only=False)
                model = make_model().eval()
                model.load_state_dict(checkpoint['model'])
                x = np.concatenate([np.broadcast_to(((feature-checkpoint['mean'])/checkpoint['std'])[:, None, :], (1024, 6, 320)), actions], -1).copy()
                with torch.no_grad():
                    recomputed = model(torch.from_numpy(x)).squeeze(-1).numpy()
                assert np.array_equal(q, recomputed), 'Cached Q differs from frozen head'
            ch = select(q, action_ids, 'S4', h['tau'])
            assert np.array_equal(ch, old[f'G0_{seed}'])
            selected_pose = hyp[np.arange(1024), ch].astype(np.float64).copy()
            for method, pose, score, ids in [
                ('S', pred, q[:, 0], np.zeros(1024, dtype=int)),
                ('G0', selected_pose, pred[:, 0], ch),
                ('Gabs', selected_pose, q[np.arange(1024), ch], ch),
            ]:
                name = f'{method}_{seed}'
                rows[name] = pose.astype(np.float64).copy()
                rows[name][:, 0] = score
                choices[name] = ids.copy()
            assert np.array_equal(rows[f'S_{seed}'][:, 1:], pred[:, 1:]), 'S geometry changed'
            assert np.array_equal(rows[f'G0_{seed}'][:, 1:], rows[f'Gabs_{seed}'][:, 1:])

    native.frame(frame)
    cloud, _ = dataset.get_data(args.scene * 256 + frame, return_raw_cloud=True)
    detector = ModelFreeCollisionDetector(cloud, voxel_size=.005)
    cd_cache, selected, submitted, native_rows = {}, {}, {}, {}
    cd_metrics = {}
    for name, raw_rows in rows.items():
        ch = choices[name]
        signature = ch.tobytes()
        if signature not in cd_cache:
            cd_cache[signature] = detector.detect(GraspGroup(raw_rows.copy()), approach_dist=.05, collision_thresh=.01)
        cd = cd_cache[signature]
        clipped = raw_rows.copy()
        clipped[:, 1] = clipped[:, 1].clip(0, .1)
        lookup = {}
        for i, row in enumerate(clipped):
            lookup.setdefault(row.tobytes(), []).append(i)
        for use_cd in [False, True]:
            name_cd = name + ('_CD' if use_cd else '')
            inp = clipped[~cd] if use_cd else clipped
            if len(inp):
                groups = select_only(GraspGroup(inp.copy()), native.models, native.dex, native.poses, native.config, table=native.table_trans, voxel_size=.008, TOP_K=50)
                gs = np.concatenate(groups)
                gs = gs[np.argsort(gs[:, 0])[::-1]]
            else:
                gs = np.zeros((0, 17))
            ids = np.array([min(i for i in lookup[g.tobytes()] if not use_cd or not cd[i]) for g in gs], dtype=int)
            selected[name_cd] = (ids, ch[ids])
            submitted[name_cd], native_rows[name_cd] = inp, gs
            cd_metrics[name_cd] = dict(cd_rejected=int(cd.sum()), cd_rejection_rate=float(cd.mean()), submitted_candidates=int(len(inp)))

    requested = np.array(sorted({int(i*6+j) for ids, js in selected.values() for i, j in zip(ids, js)}), dtype=int)
    missing = requested[~np.isfinite(mu[requested])]
    cached_calls += len(requested) - len(missing)
    if len(missing):
        mu[missing], collision[missing], empty[missing], _ = native.score(hyp.reshape(-1, 17)[missing])
        new_calls += len(missing)
    assert np.isfinite(mu[requested]).all()

    if args.micro:
        for seed in cfg['seeds']:
            for suffix in ['', '_CD']:
                name = f'S_{seed}{suffix}'
                gs, quality, coll = native.official(submitted[name])
                ids, js = selected[name]
                ix = ids * 6 + js
                result = dict(configuration=name, selection=np.array_equal(gs, native_rows[name]), quality=np.array_equal(quality, mu[ix]), collision=np.array_equal(coll, collision[ix]), AP=np.array_equal(ap(quality), ap(mu[ix])))
                assert all(v for k, v in result.items() if k != 'configuration'), result
                checks.append(result)
                print('MICRO_PASS', name, flush=True)

    previous_records = {x['configuration']: x for x in json.loads((V / f'evaluation/development/{args.scene:04d}/{frame:04d}.json').read_text())}
    records = []
    for name, (ids, js) in selected.items():
        ix = ids * 6 + js
        quality, coll, emp = mu[ix], collision[ix], empty[ix]
        matrix = ap(quality)
        m = min(50, len(ix))
        original_name = name.removesuffix('_CD')
        method = original_name.split('_')[0]
        seed = int(original_name.split('_')[1]) if '_' in original_name else 0
        rec = dict(scene=args.scene, frame=frame, configuration=name, method=method, seed=seed, use_cd=name.endswith('_CD'), ap=float(matrix.mean()*100), ap04=float(matrix[:, 1].mean()*100), ap08=float(matrix[:, 3].mean()*100), top1_valid=bool(len(quality) and 0 < quality[0] <= .4000001), top1_collision=bool(len(coll) and coll[0]), top1_empty=bool(len(emp) and emp[0]), selected_top50_count=m, selected_top50_collisions=int(coll[:m].sum()), selected_top50_collision_rate=float(coll[:m].mean()) if m else None, selected_top50_valid_rate=float(((quality[:m] > 0) & (quality[:m] <= .4000001)).mean()) if m else None, modified=int((choices[original_name] != 0).sum()), candidates=1024, top_candidate=int(ids[0]) if len(ids) else None, top_action=int(js[0]) if len(js) else None, source_export_sha256=export_hash, **cd_metrics[name])
        if method != 'S':
            old = previous_records[name]
            for key in ['ap', 'ap04', 'ap08', 'top1_valid', 'top1_collision', 'top1_empty', 'modified', 'candidates']:
                assert rec[key] == old[key], (name, frame, key, rec[key], old[key])
        records.append(rec)
    write(folder / f'{frame:04d}.json', records)
    np.savez_compressed(folder / f'{frame:04d}.npz', ids=requested, mu=mu[requested], collision=collision[requested], empty=empty[requested], source_export_sha256=export_hash, **{f'index_{name}': np.stack(value, axis=1) for name, value in selected.items()})
    source_rows.append(dict(scene=args.scene, frame=frame, export_sha256=export_hash, previous_cache_sha256=manifest[previous_relative], previous_cache=previous_relative, requested=len(requested), reused=len(requested)-len(missing), new=len(missing)))
    print('FRAME', args.scene, frame, 'new', len(missing), 'reused', len(requested)-len(missing), 'wall', round(time.time()-start, 2), flush=True)

write(folder / 'COMPLETE.json', dict(status='PASS', frames=frames, configurations=20, scene=args.scene, new_native_calls=new_calls, reused_native_calls=cached_calls, seconds=time.time()-start, cpu_seconds=time.process_time()-cpu_start, rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, native_eval_source_sha256=source_hash, sources=source_rows, cached_Q_head_exact_micro=bool(args.micro), official_micro_checks=checks, repeated_V2_metrics_all_exact=True, S_geometry_all_exact=True))
