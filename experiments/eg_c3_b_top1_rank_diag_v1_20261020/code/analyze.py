"""Read frozen C3 caches only: no model inference, geometry evaluation or training."""
import argparse
import ast
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

SEEDS = [20261018, 20261019, 20261020]
CATEGORIES = ['valid', 'empty', 'nonempty_collision', 'weak_positive_quality',
              'no_force_closure_in_evaluated_range', 'no_candidate']
FR = np.array([.2, .4, .6, .8, 1., 1.2])


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows):
    assert rows, str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def category(i, mu, collision, empty):
    if i is None:
        return 'no_candidate'
    assert np.isfinite(mu[i])
    if empty[i]:
        return 'empty'
    if collision[i]:
        return 'nonempty_collision'
    if 0 < mu[i] <= .4000001:
        return 'valid'
    return 'weak_positive_quality' if mu[i] > 0 else 'no_force_closure_in_evaluated_range'


def utility(mu):
    return float(((mu > 0) & (mu <= FR)).mean())


def scene_job(task):
    parent, scene, split, expected = task
    start, cpu = time.time(), time.process_time()
    parent = Path(parent)
    pairs, identities = [], []
    for frame in range(256):
        base = parent / f'B/evaluation/confirmation/{scene:04d}/{frame:04d}'
        jp, zp = base.with_suffix('.json'), base.with_suffix('.npz')
        jb, zb = jp.read_bytes(), zp.read_bytes()
        jhash, zhash = sha_bytes(jb), sha_bytes(zb)
        assert jhash == expected[str(jp)] and zhash == expected[str(zp)], (scene, frame, 'source mutated')
        records = {v['configuration']: v for v in json.loads(jb)}
        assert len(records) == 8
        with np.load(zp, allow_pickle=False) as z:
            mu, collision, empty = z['quality'], z['collision'], z['empty']
            ident = json.loads(str(z['identity']))
            selected = {k.removeprefix('selected_'): z[k] for k in z.files if k.startswith('selected_')}
            qualities = {s: z[f'Q_{s}'].astype(np.float64) for s in SEEDS}
        export = parent / f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz'
        with np.load(export, allow_pickle=False) as z:
            pred = z['pred']
            assert str(z['input_sha256']) == ident['input_sha256']
            assert str(z['backbone_sha256']) == ident['backbone_sha256']
            assert str(z['raw_observation_sha256']) == ident['raw_observation_sha256']
        assert pred.shape == (1024, 17) and np.isfinite(pred).all()
        assert mu.shape == collision.shape == empty.shape == (1024,)
        geometry_hash = sha_bytes(np.ascontiguousarray(pred[:, 1:]).tobytes())
        assert geometry_hash == ident['geometry_sha256']
        for name, ids in selected.items():
            v = records[name]
            assert all(v[k] == x for k, x in ident.items())
            assert np.isfinite(mu[ids]).all()
            i = int(ids[0]) if len(ids) else None
            assert i == v['top_candidate']
            assert bool(i is not None and 0 < mu[i] <= .4000001) == v['top1_valid']
            assert bool(i is not None and collision[i]) == v['top1_collision']
            assert bool(i is not None and empty[i]) == v['top1_empty']
            assert bool(i is not None and collision[i] and not empty[i]) == v['top1_nonempty_collision']
            assert (category(i, mu, collision, empty) == 'valid') == v['top1_valid']
        identities.append(dict(scene=scene, frame=frame, json_sha256=jhash, cache_sha256=zhash,
                               export_geometry_sha256=geometry_hash))
        raw = pred[:, 0].astype(np.float64)
        zraw = (raw - raw.mean()) / raw.std(ddof=0) if raw.std(ddof=0) >= 1e-6 else np.zeros(1024)
        for seed in SEEDS:
            q = qualities[seed]
            assert q.shape == (1024,) and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
            fallback = bool(raw.std(ddof=0) < 1e-6 or q.std(ddof=0) < 1e-6)
            zq = (q - q.mean()) / q.std(ddof=0) if q.std(ddof=0) >= 1e-6 else np.zeros(1024)
            fused = raw.copy() if fallback else .5 * zraw + .5 * zq
            for cd in [False, True]:
                suffix = '_CD' if cd else ''
                old_ids, new_ids = selected['R0' + suffix], selected[f'F05_{seed}' + suffix]
                old = int(old_ids[0]) if len(old_ids) else None
                new = int(new_ids[0]) if len(new_ids) else None
                assert old is not None and new is not None, 'No selected candidate; preserve and diagnose explicitly before continuing'
                oc, nc = category(old, mu, collision, empty), category(new, mu, collision, empty)
                transition = ('success_to_failure' if oc == 'valid' and nc != 'valid' else
                              'failure_to_success' if oc != 'valid' and nc == 'valid' else
                              'both_success' if oc == nc == 'valid' else 'both_failure')
                assert records[f'F05_{seed}' + suffix]['F05_fallback'] == fallback
                assert fused[new] >= fused[old] - 1e-12, 'Final Top-1 should not lose in its own score to old Top-1'
                row = dict(split=split, scene=scene, frame=frame, seed=seed, use_cd=cd,
                           transition=transition, old_id=old, new_id=new, old_category=oc, new_category=nc,
                           old_mu=float(mu[old]), new_mu=float(mu[new]),
                           old_collision=bool(collision[old]), new_collision=bool(collision[new]),
                           old_empty=bool(empty[old]), new_empty=bool(empty[new]),
                           old_utility=utility(mu[old]), new_utility=utility(mu[new]),
                           old_EG_score=float(raw[old]), new_EG_score=float(raw[new]),
                           old_Q=float(q[old]), new_Q=float(q[new]),
                           old_F05=float(fused[old]), new_F05=float(fused[new]),
                           half_zEG_gap=float(.5 * (zraw[new] - zraw[old])),
                           half_zQ_gap=float(.5 * (zq[new] - zq[old])),
                           old_raw_EG_rank=int(1 + (raw > raw[old]).sum()),
                           new_raw_EG_rank=int(1 + (raw > raw[new]).sum()),
                           old_raw_F05_rank=int(1 + (fused > fused[old]).sum()),
                           old_in_F05_selected=bool(old in new_ids),
                           old_F05_selected_rank=int(np.flatnonzero(new_ids == old)[0] + 1) if old in new_ids else None,
                           new_in_EG_selected=bool(new in old_ids),
                           new_EG_selected_rank=int(np.flatnonzero(old_ids == new)[0] + 1) if new in old_ids else None,
                           old_width=float(pred[old, 1]), new_width=float(pred[new, 1]),
                           old_depth=float(pred[old, 3]), new_depth=float(pred[new, 3]),
                           AP_delta=float(records[f'F05_{seed}' + suffix]['ap'] - records['R0' + suffix]['ap']),
                           F05_fallback=fallback)
                pairs.append(row)
    return dict(scene=scene, pairs=pairs, identities=identities, seconds=time.time() - start,
                cpu_seconds=time.process_time() - cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def summarize(rows, **labels):
    count = collections.Counter(v['transition'] for v in rows)
    harm = [v for v in rows if v['transition'] == 'success_to_failure']
    rescue = [v for v in rows if v['transition'] == 'failure_to_success']
    out = dict(**labels, N=len(rows), **{t:count[t] for t in ['success_to_failure', 'failure_to_success', 'both_success', 'both_failure']},
               delta_top1_pp=100 * (count['failure_to_success'] - count['success_to_failure']) / len(rows))
    for cat in CATEGORIES:
        out['harm_new_' + cat] = sum(v['new_category'] == cat for v in harm)
        out['rescue_old_' + cat] = sum(v['old_category'] == cat for v in rescue)
    out.update(harm_Q_prefers_new=sum(v['new_Q'] > v['old_Q'] for v in harm),
               harm_EG_prefers_old=sum(v['old_EG_score'] > v['new_EG_score'] for v in harm),
               harm_old_retained_in_F05=sum(v['old_in_F05_selected'] for v in harm),
               harm_lower_utility=sum(v['new_utility'] < v['old_utility'] for v in harm))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--reuse-saved-pairs', action='store_true')
    a = p.parse_args()
    start = time.time()
    lock = json.loads((a.parent/'C3_method_lock.json').read_text())
    manifest = {v['path']: v['sha256'] for v in json.loads((a.parent/'results/B_artifact_manifest.json').read_text())}
    assert not (a.out/'results/summary.json').exists(), 'Do not overwrite diagnosis results'
    tasks = [(str(a.parent), s, split, {k:v for k,v in manifest.items() if f'/{s:04d}/' in k})
             for split, ids in lock['selected'].items() for s in ids]
    all_rows, identities, receipts = [], [], []
    if a.reuse_saved_pairs:
        string_keys = {'split','transition','old_category','new_category'}
        bool_keys = {'use_cd','old_collision','new_collision','old_empty','new_empty',
                     'old_in_F05_selected','new_in_EG_selected','F05_fallback'}
        int_keys = {'scene','frame','seed','old_id','new_id','old_raw_EG_rank','new_raw_EG_rank',
                    'old_raw_F05_rank','old_F05_selected_rank','new_EG_selected_rank'}
        for row in csv.DictReader((a.out/'results/all_pairs.csv').open()):
            all_rows.append({k:(v if k in string_keys else v == 'True' if k in bool_keys else
                                  None if v == '' else int(v) if k in int_keys else float(v)) for k,v in row.items()})
        identities = json.loads((a.out/'results/input_cache_identity.json').read_text())
        log = (a.out/'results/run_log_initial_failure.txt').read_text()
        receipts = [ast.literal_eval(line.split('SCENE_DONE ',1)[1]) for line in log.splitlines() if line.startswith('SCENE_DONE ')]
        assert len(receipts) == 12
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            for result in pool.map(scene_job, tasks):
                all_rows.extend(result.pop('pairs'))
                identities.extend(result.pop('identities'))
                receipts.append(result)
                print('SCENE_DONE', result, flush=True)
    all_rows.sort(key=lambda v:(v['scene'], v['frame'], v['seed'], v['use_cd']))
    assert len(all_rows) == 3072 * 3 * 2
    heads, scenes, matrices = [], [], []
    for cd in [False, True]:
        for seed in SEEDS:
            rows = [v for v in all_rows if v['seed'] == seed and v['use_cd'] == cd]
            heads.append(summarize(rows, seed=seed, use_cd=cd))
            for s in sum(lock['selected'].values(), []):
                subset = [v for v in rows if v['scene'] == s]
                scenes.append(summarize(subset, seed=seed, use_cd=cd, scene=s, split=subset[0]['split']))
            matrix = collections.Counter((v['old_category'], v['new_category']) for v in rows)
            matrices += [dict(seed=seed, use_cd=cd, old_category=x, new_category=y, count=n) for (x,y),n in sorted(matrix.items())]
    old_transitions = list(csv.DictReader((a.parent/'B/results/confirmation_top1_transitions.csv').open()))
    for old in old_transitions:
        rows = [v for v in all_rows if v['seed'] == int(old['seed']) and v['use_cd'] == (old['use_cd'] == 'True') and v['split'] == old['split']]
        assert len(rows) == int(old['N'])
        for key in ['success_to_failure','failure_to_success']:
            assert sum(v['transition'] == key for v in rows) == int(old[key])
        assert sum(v['old_id'] == v['new_id'] for v in rows) == int(old['unchanged_top_candidate'])
    harm = [v for v in all_rows if v['use_cd'] and v['transition'] == 'success_to_failure']
    frame_groups = collections.defaultdict(list)
    for v in harm:
        frame_groups[v['scene'],v['frame']].append(v)
    unique = [dict(scene=s, frame=f, split=vs[0]['split'], damage_heads=len(vs),
                   seeds=';'.join(str(v['seed']) for v in vs), new_categories=';'.join(v['new_category'] for v in vs),
                   new_candidate_ids=';'.join(str(v['new_id']) for v in vs), old_id=vs[0]['old_id'], old_mu=vs[0]['old_mu'])
              for (s,f),vs in sorted(frame_groups.items())]
    summary = dict(status='PASS',exploratory=True,training_updates=0,inference_calls=0,new_geometry_calls=0,
                   parent_commit='40d705ba6ed41032121850604a3b2f73d7425658',frames=3072,paired_rows=len(all_rows),
                   heads=heads,pooled_descriptive=[summarize([v for v in all_rows if v['use_cd']==cd],use_cd=cd) for cd in [False,True]],
                   unique_harm_frames_cd=len(unique),damage_head_agreement=dict(collections.Counter(v['damage_heads'] for v in unique)),
                   cached_top1_records_checked=3072*8,previous_transition_table_matched=True,
                   score_rank_note='Ranks across all original 1024 candidates before CD/NMS; retained rank is from cached selected list.',
                   seconds=time.time()-start,worker_wall_seconds=sum(v['seconds'] for v in receipts),workers_requested=a.workers,
                   worker_receipts=receipts,time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    assert summary['worker_wall_seconds'] <= 7200
    csv_write(a.out/'results/all_pairs.csv',all_rows)
    csv_write(a.out/'results/regressions_cd.csv',harm)
    csv_write(a.out/'results/unique_regression_frames_cd.csv',unique)
    csv_write(a.out/'results/by_head.csv',heads)
    csv_write(a.out/'results/by_scene.csv',scenes)
    csv_write(a.out/'results/category_transitions.csv',matrices)
    dump(a.out/'results/input_cache_identity.json',identities)
    examples = []
    for cat in CATEGORIES:
        found = [v for v in harm if v['new_category']==cat]
        if found:
            examples.append(found[0])
    dump(a.out/'results/representative_cases.json',dict(selection='First lexicographic damaged frame per failure class; exploratory examples, not independent evidence.',cases=examples))
    dump(a.out/'results/summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['heads','worker_receipts']}),flush=True)


if __name__ == '__main__':
    main()
