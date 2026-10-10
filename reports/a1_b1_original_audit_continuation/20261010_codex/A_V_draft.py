"""User-approved A-V symbolic feature check; no learned scoring.

Default is stdlib input validation. Feature work requires explicit --execute.
This is a prepared report artifact, not a change to the frozen repository.
"""
import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import pathlib
import resource
import time
import traceback

REPO = pathlib.Path('/home/xushijie3/work/graph_cp_disr')
V2 = REPO / 'runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z'
PREV = pathlib.Path('/home/xushijie3/work/cp_disr_a1_b1/CP_DISR_A1_B1_20261010T021755Z_cc7d79d6f')
DATA = V2 / 'training/W1/data.json'
MODEL = V2 / 'training/W1/wl_goose_w1.model.params'
EXPECTED = {
    str(DATA): '153a7fe64bdd1cbf85ed666e18485a8a0724aa28c63a34e88df6ea250eb3a60f',
    str(MODEL): '78fdaa0a46c3daa8121dca65e68d9b67950bd388d4742177640e269cc3d8fe1a',
    '/home/xushijie3/envs/goose/lib/python3.10/site-packages/_wlplan.cpython-310-x86_64-linux-gnu.so':
    '0089e22796904c5ee73d0d5c863e3ce4e82b54561393e9e86b43d37e3932587f',
    str(PREV / 'a1/pair_readout_classes.csv.gz'):
    '8f3f77a46775c691e1015ac29f8e37bdf3cd6c54926331bb1a31018c7da07fc2',
    str(PREV / 'a1/candidates.csv'):
    '931d2f95a6442fb22d77631b58e04d949c71fa24bbf7d52b0e23982e8f3e2054',
    str(PREV / 'shared/b_atoms.json.gz'):
    'e8dbe35e891be4d75f5f53fe3fed2a92359b44358df92db3521697715ddb20fb',
}

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--output', type=pathlib.Path)
    a = ap.parse_args()
    hashes = {p: sha(p) for p in EXPECTED}
    if hashes != EXPECTED:
        raise RuntimeError('Frozen input/build identity mismatch; stop')
    data = json.loads(DATA.read_text())
    model_meta = json.loads(MODEL.read_text())
    assert model_meta['iterations'] == 2 and model_meta['pruning'] == 'a-m'
    assert model_meta['graph_representation'] == 'ilg' and not model_meta['multiset_hash']
    positions = {}
    for p in data['problems']:
        assert len(p['states']) == len(p['state_masks_hex'])
        for h in p['state_masks_hex']:
            key = (p['case_id'], h)
            if key in positions:
                raise RuntimeError('Duplicate training state key')
            positions[key] = len(positions)
    assert len(data['problems']) == 96 and len(positions) == 6399
    source = PREV / 'a1/pair_readout_classes.csv.gz'
    targets, controls = {}, {}
    with gzip.open(source, 'rt') as f:
        for r in csv.DictReader(f):
            if r['model'] != 'W1' or r['pool'] != 'P1':
                continue
            key = (r['problem'], r['better_state'], r['worse_state'])
            assert (key[0], key[1]) in positions and (key[0], key[2]) in positions
            dst = targets if r['category'] == 'FEATURE_EQUAL' else controls
            dst[key] = {'kind': 'target' if dst is targets else 'control',
                        'problem': key[0], 'better_state': key[1], 'worse_state': key[2]}
    assert len(targets) == 836
    assert not (set(targets) & set(controls))
    ordered_controls = sorted(controls, key=lambda k: hashlib.sha256('|'.join(k).encode()).hexdigest())[:836]
    assert len(ordered_controls) == 836
    pairs = list(targets.values()) + [controls[k] for k in ordered_controls]
    candidates = list(csv.DictReader((PREV / 'a1/candidates.csv').open()))
    assert len(candidates) == 24
    atoms = json.load(gzip.open(PREV / 'shared/b_atoms.json.gz', 'rt'))
    assert atoms['predicates'] == data['predicates']
    extra = {}
    for r in candidates:
        for h in (r['better_state'], r['worse_state']):
            key = (r['problem'], h)
            if key not in positions:
                p = atoms['problems'][key[0]]
                extra[key] = (p, p['states'][h])
    for p in (source, PREV / 'a1/candidates.csv', PREV / 'shared/b_atoms.json.gz'):
        hashes[str(p)] = sha(p)
    preflight = {'training_states': len(positions), 'target_pairs': len(targets),
                 'control_pairs': len(ordered_controls), 'exploratory_candidates': len(candidates),
                 'extra_candidate_states': len(extra), 'input_hashes': hashes,
                 'feature_work_executed': False}
    if not a.execute:
        print(json.dumps(preflight, indent=2))
        return
    if a.output is None or a.output.exists():
        raise RuntimeError('Require a new output directory')
    if a.output.parent.resolve() != pathlib.Path('/home/xushijie3/work/cp_disr_a_v').resolve():
        raise RuntimeError('Output must be an immediate child of the dedicated A-V parent')
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    import numpy as np
    from wlplan.data import DomainDataset, ProblemDataset
    from wlplan.feature_generator import init_feature_generator, load_feature_generator
    from wlplan.planning import Atom, Domain, Predicate, Problem, State

    a.output.mkdir()
    dump(a.output / 'input_identity.json', preflight)
    preds = {n: Predicate(n, arity) for n, arity in data['predicates']}
    domain = Domain('domain', list(preds.values()), [], [], [])
    def problem(p):
        return Problem(domain, p['objects'], [Atom(preds[n], list(x)) for n, x in p['goals']], [])
    def state(s):
        return State([Atom(preds[n], list(x)) for n, x in s])
    ds = DomainDataset(domain, [ProblemDataset(problem=problem(p), states=[state(s) for s in p['states']])
                               for p in data['problems']])
    extra_keys = sorted(extra)
    ds_extra = DomainDataset(domain, [ProblemDataset(problem=problem(extra[k][0]), states=[state(extra[k][1])])
                                     for k in extra_keys]) if extra else None
    all_positions = dict(positions)
    all_positions.update({k: len(positions) + i for i, k in enumerate(extra_keys)})
    matrices, summaries = {}, []
    started = time.monotonic()
    def worker(mode, repeat, conn):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
            t = time.monotonic()
            if mode == 'frozen':
                g = load_feature_generator(str(MODEL))
            else:
                g = init_feature_generator(feature_algorithm='wl', graph_representation='ilg', domain=domain,
                                           iterations=2, pruning='a-m' if mode == 'rebuild_a-m' else 'none',
                                           multiset_hash=False)
                g.collect(ds)
            x = np.array(g.embed(ds), dtype=np.int64)
            assert x.shape[0] == len(positions)
            if ds_extra is not None:
                x = np.vstack([x, np.array(g.embed(ds_extra), dtype=np.int64)])
            canonical, groups = {}, []
            for row in x[:len(positions)]:
                value = row.tobytes()
                groups.append(canonical.setdefault(value, len(canonical)))
            summary = {'mode': mode, 'repeat': repeat, 'shape': list(x.shape),
                              'training_distinct_vectors': len(canonical),
                              'matrix_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                              'training_partition': groups, 'elapsed_seconds': time.monotonic() - t,
                              'pid': os.getpid(), 'threads_limit': 1}
            np.savez_compressed(a.output / ('matrix_' + mode + '_' + str(repeat) + '.npz'), features=x)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            summary.update(cpu_seconds=usage.ru_utime + usage.ru_stime, peak_rss_kib=usage.ru_maxrss)
            conn.send({'ok': True, 'summary': summary})
        except BaseException:
            conn.send({'ok': False, 'error': traceback.format_exc()})
        finally:
            conn.close()
    peak_total_rss = 0
    ctx = multiprocessing.get_context('fork')
    for phase in ((('frozen', 0), ('rebuild_a-m', 0)),
                  (('rebuild_none', 0), ('rebuild_none', 1)),
                  (('frozen', 1), ('rebuild_a-m', 1))):
        jobs = []
        try:
            for mode, repeat in phase:
                read_conn, write_conn = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=worker, args=(mode, repeat, write_conn))
                proc.start(); write_conn.close()
                jobs.append((proc, read_conn, mode, repeat))
            pending = set(range(len(jobs)))
            while pending:
                rss = 0
                for pid in [os.getpid()] + [j[0].pid for j in jobs]:
                    try:
                        rss += int(pathlib.Path('/proc/%s/statm' % pid).read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                    except FileNotFoundError:
                        pass
                peak_total_rss = max(peak_total_rss, rss)
                if rss > 8 * 1024**3 or time.monotonic() - started > 7200:
                    raise RuntimeError('Aggregate memory or wall budget exceeded')
                for i in list(pending):
                    proc, conn, mode, repeat = jobs[i]
                    if conn.poll():
                        result = conn.recv()
                        if not result['ok']:
                            raise RuntimeError(result['error'])
                        proc.join(); conn.close()
                        if proc.exitcode != 0:
                            raise RuntimeError('Worker failed')
                        summaries.append(result['summary'])
                        with np.load(a.output / ('matrix_' + mode + '_' + str(repeat) + '.npz')) as saved:
                            matrices[(mode, repeat)] = saved['features']
                        pending.remove(i)
                    elif not proc.is_alive():
                        raise RuntimeError('Worker exited without result')
                if sum(p.stat().st_size for p in a.output.iterdir()) > 200 * 1024**2:
                    raise RuntimeError('Output size budget exceeded')
                time.sleep(0.1)
        except BaseException:
            for proc, conn, _mode, _repeat in jobs:
                if proc.is_alive():
                    proc.terminate()
                proc.join(); conn.close()
            dump(a.output / 'failure_receipt.json', {'status':'STOPPED_FAILURE_OR_BUDGET',
                 'error':traceback.format_exc(), 'finished_pipelines':len(summaries)})
            raise
        if phase[0][0] == 'frozen':
            repeat = phase[0][1]
            groups = {s['mode']: s['training_partition'] for s in summaries if s['repeat'] == repeat}
            if groups['frozen'] != groups['rebuild_a-m']:
                dump(a.output / 'receipt.json', {'status': 'INCONCLUSIVE_PIPELINE_IDENTITY',
                     'stop': 'Frozen and rebuilt training partitions differ; no further pipelines',
                     'repeat': repeat, 'feature_pipelines_so_far': len(summaries)})
                return
    base = matrices[('frozen', 0)]
    rebuild = matrices[('rebuild_a-m', 0)]
    partitions = {(s['mode'], s['repeat']): s.pop('training_partition') for s in summaries}
    repeated = all(np.array_equal(matrices[(m, 0)], matrices[(m, 1)])
                   for m in ('frozen', 'rebuild_a-m', 'rebuild_none'))
    agreement = partitions[('frozen', 0)] == partitions[('rebuild_a-m', 0)]
    am, unpruned = partitions[('rebuild_a-m', 0)], partitions[('rebuild_none', 0)]
    am_to_none, none_to_am = {}, {}
    for ag, ng in zip(am, unpruned):
        am_to_none.setdefault(ag, set()).add(ng)
        none_to_am.setdefault(ng, set()).add(ag)
    split_am = sum(len(v) > 1 for v in am_to_none.values())
    split_none = sum(len(v) > 1 for v in none_to_am.values())
    rows = []
    for r in pairs:
        rr = dict(r)
        i, j = all_positions[(r['problem'], r['better_state'])], all_positions[(r['problem'], r['worse_state'])]
        for m in ('frozen', 'rebuild_a-m', 'rebuild_none'):
            rr[m + '_separated'] = not np.array_equal(matrices[(m, 0)][i], matrices[(m, 0)][j])
        rows.append(rr)
    with (a.output / 'pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cand_rows = []
    for r in candidates:
        rr = {k: r[k] for k in ('candidate', 'problem', 'better_state', 'worse_state')}
        rr['both_in_training'] = all((r['problem'], r[h]) in positions for h in ('better_state','worse_state'))
        for m in ('frozen', 'rebuild_a-m', 'rebuild_none'):
            i = all_positions[(r['problem'], r['better_state'])]; j = all_positions[(r['problem'], r['worse_state'])]
            rr[m + '_separated'] = not np.array_equal(matrices[(m, 0)][i], matrices[(m, 0)][j])
        cand_rows.append(rr)
    dump(a.output / 'candidates.json', cand_rows)
    restored = sum(r['rebuild_none_separated'] and not r['rebuild_a-m_separated'] for r in rows if r['kind'] == 'target')
    label = 'INCONCLUSIVE_PIPELINE_OR_DETERMINISM' if not (agreement and repeated) else (
        'FIXED_BUILD_PRUNING_DIFFERENCE_REPRODUCED' if restored else 'TARGET_DIFFERENCE_NOT_REPRODUCED')
    after = {p: sha(p) for p in hashes}
    receipt = {'status': label, 'target_pairs_separated_by_none_only': restored, 'target_denominator': len(targets),
               'loaded_rebuilt_partition_equal': agreement, 'loaded_rebuilt_matrix_equal': np.array_equal(base, rebuild),
               'partition_am_vs_none_equal': am == unpruned,
               'am_groups_split_by_none': split_am, 'none_groups_split_by_am': split_none,
               'repeats_bitwise_equal': repeated, 'input_hashes_unchanged': hashes == after,
               'pipelines': summaries, 'wall_seconds': time.monotonic() - started,
               'max_parallel_workers': 2, 'threads_per_worker': 1,
               'peak_sum_rss_bytes': peak_total_rss,
               'worker_cpu_seconds': sum(s['cpu_seconds'] for s in summaries),
               'training': 0, 'learned_fitting': 0, 'model_scoring': 0, 'neural_forward': 0,
               'search': 0, 'exact_queries': 0, 'feature_collect_calls': 4, 'feature_pipelines': 6,
               'wheel_source_byte_identity': 'NOT_PROVEN', 'search_evidence': 'E0_ONLY',
               'stop': 'NO_AUTOMATIC_EXPANSION_OR_REFIT'}
    dump(a.output / 'receipt.json', receipt)
    if hashes != after:
        raise RuntimeError('Input changed during run')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
