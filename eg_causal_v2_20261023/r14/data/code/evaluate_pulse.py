"""R1.4: bounded direct actual-pose evaluation and official selection replay.

No network forward, training, nearest-label substitution, or heldout test split.
An interrupted direct-evaluation reservation is intentionally not auto-retried.
"""
import argparse
import concurrent.futures
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

B = Path.home() / 'work/grasp'
ROOT = B / 'eg_causal_mechanism_v2_20261023/r14'
OLD = B / 'eg_pose_quality_diagnosis_v1_20261022'
MODULE = B / 'eg_module_controlled_diagnosis_20261023'
SRC = B / 'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
DATA = B / 'eg_sub25_v1_20261015/data/graspnet'
FOLDS = [[2, 31, 68, 76], [9, 33, 64, 82]]
ARMS = ['R0_original', 'RW_remove_q0_width', 'RW_local_keep_final_width_only',
        'Rrandom_count_matched', 'RW_actual_step_norm_matched',
        'RW_local_actual_step_norm_matched', 'Rrandom_actual_step_norm_matched']
FIELDS = ['mu', 'collision', 'empty', 'owner', 'object_id', 'unknown_reason']
MAX_JOBS = 8 * 7 * 2 * 4
JOB_CAP = 228
PREFLIGHT_CAP = 16
REGISTERED_CAP = 131072
assert MAX_JOBS * JOB_CAP + PREFLIGHT_CAP <= REGISTERED_CAP


def initialize():
    """Import the existing official evaluator only; its main/pilot is never run."""
    global Direct, postprocess, sha, GraspGroup, ModelFreeCollisionDetector
    global Image, loadmat, CameraInfo, create_point_cloud_from_depth_image
    global get_workspace_mask, EU, GE
    sys.path[:0] = [str(OLD), str(SRC)]
    previous = sys.argv
    sys.argv = ['eval', '--dataset_root', str(DATA), '--camera', 'kinect', '--batch_size', '1']
    from evaluate import Direct, postprocess, sha
    from graspnetAPI import GraspGroup
    from utils.collision_detector import ModelFreeCollisionDetector
    from PIL import Image
    from scipy.io import loadmat
    from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
    import graspnetAPI.utils.eval_utils as EU
    import graspnetAPI.graspnet_eval as GE
    sys.argv = previous


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def rawcloud(scene, frame):
    path = DATA / f'scenes/scene_{scene:04d}/kinect'
    depth = np.array(Image.open(path / f'depth/{frame:04d}.png'))
    segment = np.array(Image.open(path / f'label/{frame:04d}.png'))
    meta = loadmat(path / f'meta/{frame:04d}.mat')
    intrinsic = meta['intrinsic_matrix']
    camera = CameraInfo(1280., 720., intrinsic[0, 0], intrinsic[1, 1],
                        intrinsic[0, 2], intrinsic[1, 2], meta['factor_depth'])
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    poses = np.load(path / 'camera_poses.npy')
    align = np.load(path / 'cam0_wrt_table.npy')
    workspace = get_workspace_mask(cloud, segment, trans=align @ poses[frame],
                                   organized=True, outlier=.02)
    return cloud[(depth > 0) & workspace]


def successful(result):
    return ((result['mu'] > 0) & (result['mu'] <= .4)
            & ~result['collision'] & ~result['empty'])


def take(result, ids):
    return {key: result[key][ids] for key in FIELDS}


def known(result):
    return np.isfinite(result['mu']) & (result['unknown_reason'].astype(str) == '')


def source_hashes():
    return {fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for fn in [Direct, postprocess, EU.collision_detection, EU.get_grasp_score,
                       EU.compute_closest_points, GE.voxel_sample_points]}


@functools.lru_cache(maxsize=8)
def frozen_frame(scene):
    assert scene in FOLDS[0] + FOLDS[1]
    tag = f'train_{scene:04d}_0128'
    cache = read_npz(OLD / 'cache' / (tag + '.npz'))
    original = read_npz(OLD / 'evaluation' / (tag + '.npz'))
    mechanism = read_npz(MODULE / 'cache' / (tag + '.npz'))['mechanism_ids']
    labels = {key: original['P0_' + key] for key in FIELDS}
    assert cache['pred'].shape == (1024, 17)
    assert mechanism.shape == (128,) and len(np.unique(mechanism)) == 128
    assert known(labels).all(), ('baseline_unknown', tag)
    assert np.all((mechanism >= 0) & (mechanism < 1024))
    return tag, cache, labels, mechanism


@functools.lru_cache(maxsize=4)
def scene_context(scene):
    direct = Direct(scene)
    direct.frame(128)
    detector = ModelFreeCollisionDetector(rawcloud(scene, 128), voxel_size=.005)
    return direct, detector


def reserve(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # x mode prevents two workers or a restart from counting the same job twice.
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def preflight(scene=9, count=8):
    assert 1 <= count <= PREFLIGHT_CAP
    destination = ROOT / 'evaluation/preflight.json'
    if destination.exists():
        receipt = json.loads(destination.read_text())
        assert receipt['complete'] and receipt['source_hashes'] == source_hashes()
        print('PREFLIGHT_CACHED', json.dumps(receipt), flush=True)
        return receipt
    tag, cache, labels, ids = frozen_frame(scene)
    ids = ids[:count]
    reserve(ROOT / 'evaluation/preflight.started.json',
            {'scene': scene, 'frame': 128, 'candidate_ids': ids.tolist(),
             'reserved_calls': count, 'selection': 'first registered mechanism_ids; no outcome selection'})
    start = time.time()
    cpu = time.process_time()
    direct, detector = scene_context(scene)
    actual = direct.score(cache['pred'][ids])
    differences = {key: int(np.count_nonzero(actual[key] != labels[key][ids])) for key in FIELDS}
    assert known(actual).all(), ('preflight_unknown', actual['unknown_reason'])
    assert not any(differences.values()), ('baseline_consistency', differences)
    collision = detector.detect(GraspGroup(cache['pred'].copy()), approach_dist=.05, collision_thresh=.01)
    assert np.array_equal(collision, cache['cd_mask']), 'baseline model-free collision mismatch'
    receipt = {'complete': True, 'scene': scene, 'frame': 128, 'calls': count,
               'candidate_ids': ids.tolist(), 'differences': differences,
               'cd_mask_exact': True, 'unknowns': 0, 'cpu_seconds': time.process_time() - cpu,
               'seconds': time.time() - start, 'source_hashes': source_hashes(),
               'baseline_cache_sha256': sha(OLD / 'cache' / (tag + '.npz')),
               'baseline_labels_sha256': sha(OLD / 'evaluation' / (tag + '.npz'))}
    save_json(destination, receipt)
    print('PREFLIGHT', json.dumps(receipt), flush=True)
    return receipt


def parse_job(path):
    rel = path.relative_to(ROOT / 'predictions')
    assert len(rel.parts) == 4, rel
    source = int(re.fullmatch(r'src_(\d{4})', rel.parts[0]).group(1))
    arm = rel.parts[1]
    step = int(re.fullmatch(r'step([15])', rel.parts[2]).group(1))
    match = re.fullmatch(r'train_(\d{4})_(\d{4})\.npz', rel.parts[3])
    scene, frame = map(int, match.groups())
    assert source in FOLDS[0] + FOLDS[1] and arm in ARMS and frame == 128
    opposite = FOLDS[1] if source in FOLDS[0] else FOLDS[0]
    assert scene in opposite, ('source_holdout_fold', source, scene)
    return rel, dict(source_scene=source, target_scene=scene, target_frame=frame, arm=arm, step=step)


def job_number(path):
    meta = parse_job(path)[1]
    source = meta['source_scene']
    opposite = FOLDS[1] if source in FOLDS[0] else FOLDS[0]
    # Stable as new GPU outputs arrive, unlike position in a partial glob list.
    return (sorted(FOLDS[0] + FOLDS[1]).index(source) * 56
            + ARMS.index(meta['arm']) * 8 + [1, 5].index(meta['step']) * 4
            + opposite.index(meta['target_scene']))


def job(path):
    rel, metadata = parse_job(path)
    destination = ROOT / 'evaluation' / rel
    receipt_path = destination.with_suffix('.json')
    if destination.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['complete'] and receipt['input_sha256'] == sha(path)
        assert receipt['result_sha256'] == sha(destination)
        return dict(metadata, cached=True, calls=receipt['calls'])
    assert not destination.exists() and not receipt_path.exists(), ('partial_output', str(rel))
    assert not destination.with_suffix('.started.json').exists(), ('interrupted_reservation_stop', str(rel))
    start, cpu = time.time(), time.process_time()
    scene = metadata['target_scene']
    tag, baseline, old_labels, ids = frozen_frame(scene)
    data = read_npz(path)
    updated = data['pred']
    old = baseline['pred']
    assert updated.shape == old.shape == (1024, 17) and np.isfinite(updated).all()
    assert np.array_equal(data['candidate_id'], np.arange(1024))
    assert np.array_equal(data['mechanism_ids'], ids), ('mechanism_ids', str(rel))
    assert np.array_equal(updated[:, 13:16], old[:, 13:16]), ('position_changed', str(rel))
    assert np.max(abs(updated[:, 4:13].reshape(-1, 3, 3)[:, :, 0]
                      - old[:, 4:13].reshape(-1, 3, 3)[:, :, 0])) < 2e-6, ('approach_changed', str(rel))
    assert np.array_equal(updated[:, 2], old[:, 2]), ('height_changed', str(rel))
    gpu_base = ROOT / 'baseline' / (tag + '.npz')
    if gpu_base.exists():
        assert np.array_equal(read_npz(gpu_base)['pred'], old), ('gpu_initial_baseline', tag)
    if path.with_suffix('.json').exists():
        upstream = json.loads(path.with_suffix('.json').read_text())
        for key, value in metadata.items():
            if key in upstream:
                assert upstream[key] == value, ('metadata', key)
    direct, detector = scene_context(scene)
    owner = direct.owners(old)
    assert np.array_equal(owner, old_labels['owner'])
    new_cd = detector.detect(GraspGroup(updated.copy()), approach_dist=.05, collision_thresh=.01)
    predictions = {'G0S0': old.copy(), 'G1S1': updated.copy(),
                   'G1S0': updated.copy(), 'G0S1': old.copy()}
    predictions['G1S0'][:, 0] = old[:, 0]
    predictions['G0S1'][:, 0] = updated[:, 0]
    selections = {}
    output = {'primary_ids': ids, 'candidate_id': np.arange(1024)}
    for name, pred in predictions.items():
        cd = new_cd if name.startswith('G1') else baseline['cd_mask']
        ni, oi, fi = postprocess(pred, cd, owner)
        assert len(fi) <= 50 and len(np.unique(fi)) == len(fi)
        selections[name] = fi
        for suffix, value in [('nms_ids', ni), ('obj10_ids', oi), ('final_ids', fi), ('cd_mask', cd)]:
            output[name + '_' + suffix] = value
    new_ids = np.unique(np.concatenate([ids, selections['G1S0'], selections['G1S1']]))
    assert len(new_ids) <= JOB_CAP
    reserve(destination.with_suffix('.started.json'),
            dict(metadata, reserved_calls=len(new_ids), input_sha256=sha(path),
                 candidate_ids=new_ids.tolist(), maximum_job_calls=JOB_CAP))
    new_labels = direct.score(updated[new_ids])
    if not known(new_labels).all():
        bad = ~known(new_labels)
        save_json(destination.with_suffix('.failed.json'),
                  dict(metadata, calls=len(new_ids), complete=False,
                       failed_candidate_ids=new_ids[bad].tolist(),
                       unknown_reasons=new_labels['unknown_reason'][bad].tolist()))
        raise AssertionError(('new_unknown_stop', str(rel), new_labels['unknown_reason']))
    assert np.array_equal(new_labels['owner'], owner[new_ids])
    lookup = np.full(1024, -1, dtype=int)
    lookup[new_ids] = np.arange(len(new_ids))
    output['new_evaluated_ids'] = new_ids
    for key in FIELDS:
        output['new_' + key] = new_labels[key]
        output['primary_G0_' + key] = old_labels[key][ids]
        output['primary_G1_' + key] = new_labels[key][lookup[ids]]
    summary = dict(metadata)
    old_success = successful(take(old_labels, ids))
    new_success = successful(take(new_labels, lookup[ids]))
    summary['primary'] = {'n': len(ids), 'old_success': int(old_success.sum()),
                          'new_success': int(new_success.sum()),
                          'rescue': int((~old_success & new_success).sum()),
                          'harm': int((old_success & ~new_success).sum()), 'unknown': 0}
    summary['pipeline'] = {}
    for name, fi in selections.items():
        labels = take(new_labels, lookup[fi]) if name.startswith('G1') else take(old_labels, fi)
        assert known(labels).all()
        for key in FIELDS:
            output[name + '_' + key] = labels[key]
        successes = successful(labels)
        summary['pipeline'][name] = {'n': len(fi), 'successes': int(successes.sum()),
                                    'top1_success': bool(successes[0]) if len(fi) else False,
                                    'top1_raw_id': int(fi[0]) if len(fi) else None, 'unknown': 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **output)
    temp.replace(destination)
    receipt = dict(metadata, complete=True, calls=len(new_ids), unknowns=0,
                   cpu_seconds=time.process_time() - cpu, seconds=time.time() - start,
                   input_sha256=sha(path), result_sha256=sha(destination),
                   baseline_cache_sha256=sha(OLD / 'cache' / (tag + '.npz')),
                   baseline_labels_sha256=sha(OLD / 'evaluation' / (tag + '.npz')),
                   primary_n=len(ids), geometry_position_exact=True, approach_tolerance=2e-6,
                   max_job_calls=JOB_CAP, source_hashes=source_hashes(), summary=summary)
    save_json(receipt_path, receipt)
    print('DONE', json.dumps(dict(metadata, calls=len(new_ids), cpu_seconds=receipt['cpu_seconds'],
                                  primary=summary['primary'])), flush=True)
    return receipt


def worker(path):
    if 'Direct' not in globals():
        initialize()
    return job(path)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-scene', type=int, default=9)
    parser.add_argument('--preflight-count', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--source', type=int, nargs='*')
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    ROOT = args.root
    assert 0 <= args.shard < args.nshards and args.workers >= 1 and args.start >= 0
    initialize()
    if args.preflight:
        preflight(args.preflight_scene, args.preflight_count)
        return
    receipt = json.loads((ROOT / 'evaluation/preflight.json').read_text())
    assert receipt['complete'] and receipt['unknowns'] == 0
    assert receipt['source_hashes'] == source_hashes()
    paths = sorted((ROOT / 'predictions').glob('src_*/**/train_*_0128.npz'))
    assert len(paths) <= MAX_JOBS
    keys = [tuple(parse_job(path)[1].values()) for path in paths]
    assert len(set(keys)) == len(keys)
    tasks = [path for path in paths if job_number(path) % args.nshards == args.shard]
    if args.source is not None:
        tasks = [path for path in tasks if parse_job(path)[1]['source_scene'] in args.source]
    if args.arm is not None:
        tasks = [path for path in tasks if parse_job(path)[1]['arm'] == args.arm]
    tasks = tasks[args.start:]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    assert tasks, 'No generated prediction jobs match the requested scope'
    print('SCOPE', json.dumps({'available_jobs': len(paths), 'selected_jobs': len(tasks),
                               'max_calls': MAX_JOBS * JOB_CAP + PREFLIGHT_CAP,
                               'registered_cap': REGISTERED_CAP}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker, tasks):
            pass
    complete = {'complete': True, 'jobs': len(tasks), 'shard': args.shard, 'nshards': args.nshards,
                'scope_files': [str(path.relative_to(ROOT / 'predictions')) for path in tasks]}
    save_json(ROOT / f'evaluation_complete_shard{args.shard}_of{args.nshards}.json', complete)
    print('COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
