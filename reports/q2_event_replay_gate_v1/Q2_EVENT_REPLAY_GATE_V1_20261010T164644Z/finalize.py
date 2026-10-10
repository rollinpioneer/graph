"""Audit saved G0-G2 evidence only; no scoring, search, training, or optimality query."""
import collections
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
MODE = sys.argv[2] if len(sys.argv) > 2 else 'local'
START = time.process_time()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def records(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def score_choice(entries, vals):
    ordered = sorted((struct.unpack('<f', bytes.fromhex(vals[s]))[0], ser, s) for h, ser, s, valid in entries if valid)
    v, serial, state = ordered[0]
    return {'state': state, 'serial': serial, 'score_bits': struct.pack('<f', v).hex(),
            'minimum_ties': sum(x[0] == v for x in ordered), 'top_second_gap': ordered[1][0] - v}


def audit():
    reg, receipt, candidates, selected = [read(n) for n in ['registration.json', 'gate_receipt.json', 'candidates.json', 'selected_events.json']]
    ledger = [json.loads(s) for s in (ROOT / 'resource_ledger.jsonl').read_text().splitlines()]
    starts = [x for x in ledger if x['kind'] == 'MODEL_STATE_START']
    ends = {x['sequence']: x for x in ledger if x['kind'] == 'MODEL_STATE_END'}
    assert [x['sequence'] for x in starts] == list(range(1, len(starts) + 1))
    assert len(starts) == len(ends) == receipt['counts']['model_state_evaluations'] <= 8192
    raw = (ROOT / 'score_bytes.bin').read_bytes()
    assert len(raw) == len(starts) * 4
    maps = collections.defaultdict(set)
    for row in starts:
        end = ends[row['sequence']]
        assert row['input_graphs'] == 1 and end['status'] == 'OK'
        at = (row['sequence'] - 1) * 4
        assert end['byte_offset'] == at and raw[at:at + 4].hex() == end['value_bits_le']
        maps[row['task'], row['model'], row['state']].add(end['value_bits_le'])
    inconsistent = [list(k) for k, vs in maps.items() if len(vs) != 1]
    assert not inconsistent, inconsistent
    prefix = {cid: records(ROOT / 'decisions' / (cid + '_prefix.jsonl.gz')) for cid in reg['tasks']}
    observed = []
    for row in candidates:
        if row['origin'] == 'fresh_prefix':
            before = next(r['after'] for r in prefix[row['task']] if r['after']['expanded'] == row['k'])
        else:
            before = read(row['snapshot'])
        entries = before['heap']
        valid = [x for x in entries if x[3]]
        assert len(valid) == row['valid_OPEN']
        assert len({x[2] for x in valid}) == len(valid)
        if row['origin'] == 'fresh_prefix':
            got = []
            for model in reg['model_order']:
                vals = {s: next(iter(maps[row['task'], model, s])) for _, _, s, _ in valid}
                got.append(score_choice(entries, vals))
            assert got == row['prescreen'], (row['task'], row['k'])
        observed.append({'task': row['task'], 'k': row['k'], 'origin': row['origin'], 'full_OPEN_verified': len(valid), 'status': row['status']})
    repeat_starts = [x for x in starts if x['reason'].startswith('independent_OPEN_repeat_')]
    blocks = [(key, list(group)) for key, group in itertools.groupby(repeat_starts, key=lambda x: (x['task'], x['reason'], x['model']))]
    repeats, offset = [], 0
    for row in candidates:
        if 'independent_repetitions' not in row:
            continue
        before = read(row['snapshot']) if 'snapshot' in row else next(r['after'] for r in prefix[row['task']] if r['after']['expanded'] == row['k'])
        valid = [x for x in before['heap'] if x[3]]
        expected_states = {x[2] for x in valid}
        for rep in range(3):
            for model in reg['model_order']:
                key, calls = blocks[offset]
                offset += 1
                assert key == (row['task'], 'independent_OPEN_repeat_%d' % rep, model)
                assert len(calls) == len(valid) and {x['state'] for x in calls} == expected_states
                vals = {x['state']: ends[x['sequence']]['value_bits_le'] for x in calls}
                assert score_choice(before['heap'], vals) == row['independent_repetitions'][rep][model]
        assert canonical(before) == row['snapshot_sha256']
        if row['origin'] == 'fresh_prefix':
            baseline_before = next(r['after'] for r in prefix[row['task']] if r['after']['expanded'] == row['k'])
            assert before == baseline_before
        repeats.append({'task': row['task'], 'k': row['k'], 'origin': row['origin'],
                        'full_OPEN_three_independent_passes_verified': True,
                        'actual_invocations': 9 * len(valid), 'snapshot_file_sha256': sha(ROOT / row['snapshot']),
                        'snapshot_semantic_sha256': canonical(before)})
    assert offset == len(blocks)
    g2 = []
    for row in selected:
        snap = read(row['snapshot'])
        assert canonical(snap) == row['snapshot_sha256'] and not snap['pending_scores']
        baseline = [r for r in prefix[row['task']] if r['decision'] and r['decision']['expanded'] > row['k']]
        paths = list((ROOT / 'decisions').glob('%s_k%03d_restore*.jsonl.gz' % (row['task'], row['k'])))
        assert len(paths) == 1
        restore = records(paths[0])
        assert restore and baseline == restore
        assert baseline[0]['before_sha256'] == canonical(snap)
        for i, rec in enumerate(restore):
            before = snap if i == 0 else restore[i - 1]['after']
            assert canonical(before) == rec['before_sha256']
            closed = set(before['closed'])
            entries = [e for e in before['heap'] if e[2] not in closed]
            top = min(entries, key=lambda e: (struct.unpack('<f', bytes.fromhex(e[0]))[0], e[1], int(e[2], 16)))
            assert (top[0], top[1], top[2]) == (rec['decision']['score_bits'], rec['decision']['serial'], rec['decision']['state'])
            assert rec['after']['expanded'] == before['expanded'] + 1
            assert set(rec['after']['closed']) == closed | {top[2]}
        g2.append({'task': row['task'], 'k': row['k'], 'steps': len(restore), 'status': 'PASS',
                   'full_contents_compared': True, 'restored_first_input_matches_snapshot': True,
                   'every_next_pop_matches_actual_heap': True, 'baseline_log': str((ROOT / 'decisions' / (row['task'] + '_prefix.jsonl.gz')).name),
                   'restore_log': paths[0].name})
    # Selection uses only stable full-OPEN disagreement and G2 reliability, never intervention outcomes.
    p12 = [x for x in candidates if x['origin'] == 'fresh_prefix' and x['task'] == 'ipc_p12' and x['status'] == 'STABLE_MAIN_EVENT']
    structs = [x for x in candidates if x['origin'] == 'fresh_prefix' and x['task'].startswith('struct') and x['status'] == 'STABLE_MAIN_EVENT']
    expected = ([min(p12, key=lambda x: x['k'])] if p12 else []) + ([min(structs, key=lambda x: (reg['tasks'][x['task']]['sha256'], x['k']))] if structs else [])
    assert [(x['task'], x['k']) for x in selected] == [(x['task'], x['k']) for x in expected]
    searches = [x for x in ledger if x['kind'] == 'SEARCH_START']
    assert len(searches) == receipt['counts']['search_prefix_and_restore_calls'] <= 6
    assert all(x['intervention'] is False for x in searches)
    assert receipt['phase_C_runs'] == receipt['training'] == receipt['certification'] == receipt['graph_builds'] == receipt['GPU_device_seconds'] == 0
    grouped = collections.Counter((x['task'], x['model'], x['reason']) for x in starts)
    out = {'status': 'PASS', 'observer_full_frontiers': observed, 'repeat_checks': repeats, 'G2_independent_content_audit': g2,
           'all_observed_task_state_model_scores_bitwise_consistent': not inconsistent,
           'unique_model_state_keys': len(maps), 'successful_actual_model_calls': len(starts),
           'score_bytes_sha256': sha(ROOT / 'score_bytes.bin'), 'search_calls': searches,
           'selection_rule_verified': True, 'raw_file_bytes': len(raw),
           'call_breakdown': [{'task': a, 'model': b, 'reason': c, 'calls': n} for (a, b, c), n in sorted(grouped.items())],
           'cpu_seconds': time.process_time() - START, 'no_new_search_or_scoring': True}
    save('verification_' + MODE + '.json', out)
    return out, prefix


def server_checks(prefix):
    reg, old, assets = read('registration.json'), read('old_asset_inventory.json'), read('registration_assets.json')
    before = read('identity_before.json')
    repo = Path('/home/xushijie3/work/graph_cp_disr')
    changed = []
    for path, expected in assets['complete_project_source_hashes'].items():
        if sha(path) != expected:
            changed.append(path)
    old_changed = [x['path'] for x in old['files'].values() if sha(x['path']) != x['sha256']]
    weight_changed = [m['path'] for m in reg['models'].values() if sha(m['path']) != m['sha256_actual']]
    task_changed = []
    for task in reg['tasks'].values():
        for path, expected in [(task['file'], task['sha256']), (task['domain_file'], task['domain_sha256'])]:
            if sha(path) != expected:
                task_changed.append(path)
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain=v1', '--untracked-files=all'])
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    integrity = {'source_count': len(assets['complete_project_source_hashes']), 'source_changed': changed,
                 'old_asset_count': len(old['files']), 'old_assets_changed': old_changed, 'weight_files_changed': weight_changed,
                 'task_domain_files_changed': task_changed, 'head': head, 'branch': branch,
                 'status_sha256': hashlib.sha256(status).hexdigest(), 'status_lines': len(status.splitlines())}
    integrity['status'] = 'PASS' if not (changed or old_changed or weight_changed or task_changed) and head == before['head'] and branch == before['branch'] and integrity['status_sha256'] == before['status_sha256'] else 'FAIL'
    save('integrity_after.json', integrity)
    assert integrity['status'] == 'PASS', integrity
    sys.path.insert(0, str(repo / 'src'))
    sys.dont_write_bytecode = True
    from cp_disr.pddl import depots as DP, task as T
    plans = []
    for cid, rs in prefix.items():
        task = T.depots_task(DP.DOMAIN_IPC if cid.startswith('ipc') else DP.DOMAIN_TYPED, reg['tasks'][cid]['file'])
        assert not task.goal_satisfied(task.init_mask), 'all three tasks require a nontrivial plan'
        last = rs[-1]['after']
        if last['status'] != 'SOLVED':
            plans.append({'task': cid, 'status': 'PREFIX_TRUNCATED', 'expanded': last['expanded'], 'optimality': 'UNKNOWN'})
            continue
        by_index = {a.index: a for a in task.actions}
        state = task.init_mask
        state_path, ids = [format(state, 'x')], []
        for index in last['plan']:
            a = by_index[index]
            assert state & a.pre_mask == a.pre_mask and not (state & a.neg_mask)
            state = task.apply(state, a)
            state_path.append(format(state, 'x')); ids.append(a.aid)
        assert task.goal_satisfied(state) and task.replay(ids) == state
        plans.append({'task': cid, 'status': 'VALID_ACTION_CONTRACT_PLAN', 'expanded': last['expanded'],
                      'plan_length': len(ids), 'action_indices': last['plan'], 'action_ids': ids,
                      'states': state_path, 'final_goal_satisfied': True, 'optimality': 'UNKNOWN'})
    save('plan_validation.json', {'plans': plans, 'new_search': 0, 'new_model_calls': 0, 'optimality_queries': 0})
    contracts, correspondence = [], []
    for row in read('candidates.json'):
        if 'snapshot' not in row:
            continue
        cid = row['task']
        info = reg['tasks'][cid]
        task = T.depots_task(DP.DOMAIN_IPC if cid.startswith('ipc') else DP.DOMAIN_TYPED, info['file'])
        snapshot = read(row['snapshot'])
        assert snapshot['task_sha256'] == info['sha256'] and snapshot['domain_sha256'] == info['domain_sha256']
        assert snapshot['model_sha256'] == reg['models']['D0@820']['sha256_actual']
        assert snapshot['scoring_protocol'] == reg['scoring_protocol']['id']
        nodes = {s: (g, p, a) for s, g, p, a in snapshot['node']}
        root = format(task.init_mask, 'x')
        assert nodes[root] == (0, None, None) and set(snapshot['closed']) <= nodes.keys()
        by_index = {a.index: a for a in task.actions}
        checked = 0
        for state, (g, parent, index) in nodes.items():
            if parent is None:
                assert state == root
                continue
            a, source = by_index[index], int(parent, 16)
            assert source & a.pre_mask == a.pre_mask and not (source & a.neg_mask)
            assert task.apply(source, a) == int(state, 16) and g > nodes[parent][0]
            checked += 1
        contracts.append({'task': cid, 'k': row['k'], 'origin': row['origin'], 'snapshot': row['snapshot'],
                          'nodes': len(nodes), 'parent_action_contracts_checked': checked, 'status': 'PASS',
                          'root_task_and_protocol_bound': True, 'all_parent_chains_reach_root': True})
        if row not in read('selected_events.json'):
            continue
        choices = row['independent_repetitions'][0]
        astate, bstate = [choices[m]['state'] for m in ['D0@820', 'T1@2460']]
        ag, ap, ai = nodes[astate]; bg, bp, bi = nodes[bstate]
        feasible = next(p for p in plans if p['task'] == cid)
        plan_states = feasible.get('states', [])
        correspondence.append({'task': cid, 'k': row['k'], 'D0_state': astate, 'T1_state': bstate,
                               'D0_g': ag, 'T1_g': bg, 'D0_parent': ap, 'T1_parent': bp,
                               'D0_action_id': by_index[ai].aid, 'T1_action_id': by_index[bi].aid,
                               'same_parent_legal_siblings': ap == bp,
                               'D0_on_saved_valid_plan': astate in plan_states if plan_states else 'UNKNOWN',
                               'T1_on_saved_valid_plan': bstate in plan_states if plan_states else 'UNKNOWN',
                               'saved_plan_optimality': 'UNKNOWN',
                               'OptRank_optimal_plan_constraint_coverage': 'UNKNOWN',
                               'Chrestien_optimal_path_OPEN_constraint_coverage': 'UNKNOWN',
                               'new_ranking_relation_gap_established': False})
    save('snapshot_contracts.json', {'snapshots': contracts, 'new_model_calls': 0, 'new_search': 0})
    save('author_condition_correspondence.json', {'events': correspondence, 'optimality_queries': 0})
    packages = {}
    for name in ['torch', 'torch-geometric', 'numpy', 'pyyaml', 'scipy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'NOT_INSTALLED'
    save('runtime.json', {'python': sys.version, 'executable': sys.executable, 'package_versions': packages,
                         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
                         'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'), 'cpu_seconds': time.process_time() - START,
                         'role': 'post-run saved-evidence verification; no forward or search'})


out, prefix = audit()
if MODE == 'server':
    server_checks(prefix)
print(json.dumps({'status': out['status'], 'calls': out['successful_actual_model_calls'], 'G2': out['G2_independent_content_audit'], 'cpu_seconds': time.process_time() - START}))
