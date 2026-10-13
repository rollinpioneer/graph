"""Independent read-only audit of saved D and archived GOALPROBE05 P3 traces.

No JAX, policy, environment, simulation or optimizer is imported.  The finite
map, BFS and trace summaries below are deliberately independent of core06.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GP = Path('/home/xushijie3/work/graph_saw_goalprobe05_20261013/experiments/saw_goalprobe05_20261013')
GRID = (
    '111111111111', '100001000001', '101101010101',
    '100000010001', '101111011101', '100101000001',
    '110101010111', '100100010001', '111111111111',
)
FREE = {(i, j) for i, row in enumerate(GRID) for j, v in enumerate(row) if v == '0'}
THRESHOLDS = dict(direction_norm_gt=0.01, reversal_cosine_lt=-0.5,
                  repeat_separation_ge=3, repeat_radius_le=0.1,
                  backtrack_path_ge=1.0, backtrack_displacement_reversals_ge=3,
                  backtrack_repeat_steps_ge=2, small_command_mean_norm_le=0.25,
                  near_stationary_max_displacement_le=0.1)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    assert rows, 'EMPTY_MORPHOLOGY_OUTPUT'
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def guard():
    p = ROOT / 'receipts/RESOURCE_START.json'
    assert p.exists(), 'BUDGET_NOT_STARTED'
    start = json.loads(p.read_text())
    assert time.time() - start.get('start_epoch', start.get('epoch')) < 14400, 'CPU_WALL_CAP'
    if hasattr(os, 'sched_getaffinity'):
        assert len(os.sched_getaffinity(0)) <= 4, 'CPU_AFFINITY_CAP'


def cell(xy):
    # For all admitted maze positions this is equal to native xy_to_ij; floor
    # avoids relying on the source implementation's integer truncation.
    return (math.floor((float(xy[1]) + 6.0) / 4.0),
            math.floor((float(xy[0]) + 6.0) / 4.0))


def center(c):
    return np.array([4.0 * c[1] - 4.0, 4.0 * c[0] - 4.0])


def neighbors(c):
    return sorted(n for n in ((c[0]-1, c[1]), (c[0], c[1]-1),
                               (c[0]+1, c[1]), (c[0], c[1]+1)) if n in FREE)


def bfs(goal):
    assert goal in FREE, ('INVALID_GOAL_CELL', goal)
    dist = {goal: 0}
    todo = deque([goal])
    while todo:
        c = todo.popleft()
        for n in neighbors(c):
            if n not in dist:
                dist[n] = dist[c] + 1
                todo.append(n)
    assert len(dist) == 46
    return dist


def firsts(start, goal):
    dist = bfs(goal)
    return [n for n in neighbors(start) if dist[n] == dist[start] - 1]


def reversal_count(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) < 2:
        return 0
    norms = np.linalg.norm(vectors, axis=1)
    count = 0
    for i in range(1, len(vectors)):
        if norms[i-1] > 0.01 and norms[i] > 0.01:
            cosine = float(np.dot(vectors[i-1], vectors[i]) / (norms[i-1] * norms[i]))
            count += int(cosine < -0.5)
    return count


def repeat_count(xy):
    # Visit of saved state t: any prior saved state u <= t-3 qualifies, at
    # most once for t, including the original state at t=0.
    return sum(int(np.any(np.linalg.norm(xy[:t-2] - xy[t], axis=1) <= 0.1))
               for t in range(3, len(xy)))


def summarize(trace, old=False):
    xy = np.asarray(trace['states'], dtype=float)
    raw = np.asarray(trace['raw_actions'], dtype=float)
    actions = np.asarray(trace['actions'], dtype=float)
    qvel = np.asarray(trace['qvel'], dtype=float)
    integ = np.asarray(trace['integration'], dtype=float)
    n = len(actions)
    assert 1 <= n <= 25
    assert xy.shape == qvel.shape == (n + 1, 2)
    assert raw.shape == actions.shape == (n, 2)
    assert integ.shape[0] == n + 1 and integ.shape[1] >= 5
    assert all(np.isfinite(v).all() for v in (xy, raw, actions, qvel, integ))
    assert np.array_equal(actions, np.clip(raw, -1.0, 1.0))
    assert np.max(np.abs(integ[:, 1:3] - xy)) <= 1e-10
    assert np.max(np.abs(integ[:, 3:5] - qvel)) <= 1e-10
    initial = trace['initial']
    assert np.array_equal(xy[0], initial['qpos'])
    assert np.array_equal(qvel[0], initial['qvel'])
    assert np.array_equal(integ[0], initial['integration'])
    if old:
        goal = np.asarray(trace['goals'][int(trace['metric']['goal_index'])], dtype=float)
    else:
        goal = np.asarray(trace['goal_xy'], dtype=float)
    assert np.array_equal(goal, initial['goal'])
    assert np.array_equal(goal, initial['goal_xy'])
    start, gcell = cell(xy[0]), cell(goal)
    valid = firsts(start, gcell)
    assert valid == sorted(tuple(int(c) for c in v) for v in trace['reference_firsts'])
    assert all(cell(p) in FREE for p in xy), 'TRACE_LEFT_FREE_MAP'
    exit_items = [(t, cell(p)) for t, p in enumerate(xy[1:], 1) if cell(p) != start]
    exit_step, exit_cell = exit_items[0] if exit_items else (None, None)
    correct = int(exit_cell in valid) if exit_cell is not None else 0
    displacements = xy[1:] - xy[:-1]
    lengths = np.linalg.norm(displacements, axis=1)
    norms = np.linalg.norm(actions, axis=1)
    path = float(lengths.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    maximum = float(np.linalg.norm(xy - xy[0], axis=1).max())
    disp_reverse, action_reverse = reversal_count(displacements), reversal_count(actions)
    repeats = repeat_count(xy)
    backtrack = bool(path >= 1.0 and disp_reverse >= 3 and repeats >= 2)
    category = 'correct' if correct else ('wrong_channel' if exit_cell is not None else
                                         ('no_motion' if maximum <= 0.1 else 'not_left_region'))
    final = cell(xy[-1])
    dmap = bfs(gcell)
    ref = center(valid[0]) - center(start)
    ref /= np.linalg.norm(ref)
    terms = trace['termination_success']
    assert len(terms) == n
    assert all(len(x) == 3 for x in terms)
    assert not any(bool(t[0]) or bool(t[1]) for t in terms[:-1]), 'ADVANCED_AFTER_TERMINATION'
    assert n == 25 or bool(terms[-1][0]) or bool(terms[-1][1]), 'UNJUSTIFIED_SHORT_ROLLOUT'
    def remaining(p):
        c = cell(p)
        if c == gcell:
            return float(np.linalg.norm(p-center(gcell)))
        next_cell = firsts(c, gcell)[0]
        return float(np.linalg.norm(p-center(next_cell))+4*dmap[next_cell])
    return dict(steps=n, correct=correct, category=category,
                first_exit=None if exit_cell is None else list(exit_cell),
                first_exit_step=exit_step,
                first_exit_censored_step=25 if exit_step is None else exit_step,
                first_exit_right_censored=exit_step is None,
                first_exit_time_or_censor=25 if exit_step is None else exit_step,
                terminated=bool(terms[-1][0]), truncated=bool(terms[-1][1]),
                local_goal_reached=any(bool(t[2]) for t in terms),
                displacement=net, maximum_displacement=maximum, path_length=path,
                path_efficiency=net/(path+1e-9),
                mean_action_norm=float(norms.mean()), median_action_norm=float(np.median(norms)),
                mean_raw_action_norm=float(np.linalg.norm(raw, axis=1).mean()),
                saturation_fraction=float(np.mean(np.abs(raw) >= 1.0)),
                clipping_frequency=float(np.mean(np.any(raw != actions, axis=1))),
                small_action_fraction=float(np.mean(norms <= 0.1)),
                small_command_candidate=bool(norms.mean() <= 0.25),
                near_stationary=bool(maximum <= 0.1),
                graph_progress=int(dmap[start]-dmap[final]),
                reference_progress=remaining(xy[0])-remaining(xy[-1]),
                initial_channel_projection=float(np.dot(xy[-1]-xy[0], ref)),
                action_reversals=action_reverse, displacement_reversals=disp_reverse,
                repeat_visit_steps=repeats, backtracking_evidence=backtrack,
                start_cell=list(start), goal_cell=list(gcell), bfs_distance=int(dmap[start]),
                reference_firsts=[list(c) for c in valid])


ALIASES = {
    'CBR25': 'correct', 'cbr25': 'correct', 'actual_displacement_reversals': 'displacement_reversals',
    'actual_reversals': 'displacement_reversals', 'disp_reversals': 'displacement_reversals',
    'repeat_visits': 'repeat_visit_steps', 'repeated_visit_steps': 'repeat_visit_steps',
    'backtracking': 'backtracking_evidence', 'SMALL_COMMAND_CANDIDATE': 'small_command_candidate',
    'BACKTRACKING_EVIDENCE': 'backtracking_evidence', 'near_static': 'near_stationary',
    'clip_fraction': 'clipping_frequency', 'clipping_fraction': 'clipping_frequency',
    'graph_distance_progress': 'graph_progress', 'bfs_progress': 'graph_progress',
    'distance': 'bfs_distance', 'distance_edges': 'bfs_distance',
    'firsts': 'reference_firsts',
}


def compare(value, expected, name):
    if expected is None:
        assert value in ('', None, 'None', 'null', 'nan'), (name, value, expected)
        return 0.0
    if isinstance(expected, list):
        actual = json.loads(value) if isinstance(value, str) else value
        assert actual == expected, (name, actual, expected)
        return 0.0
    if isinstance(expected, bool):
        if isinstance(value, str):
            assert value.lower() in ('true', 'false', '1', '0'), (name, value, expected)
        actual = value.lower() in ('true', '1') if isinstance(value, str) else bool(value)
        assert actual == expected, (name, actual, expected)
        return 0.0
    if isinstance(expected, (int, float)):
        error = abs(float(value) - expected)
        assert error <= 1e-10, (name, value, expected, error)
        return error
    assert value == expected, (name, value, expected)
    return 0.0


def audit_record(row, trace, old=False):
    result = summarize(trace, old)
    errors = []
    checked = []
    for field, value in row.items():
        metric_name = ALIASES.get(field, field)
        if metric_name in result:
            errors.append(compare(value, result[metric_name], field))
            checked.append(field)
    for field, value in trace['metric'].items():
        metric_name = ALIASES.get(field, field)
        if metric_name in result:
            errors.append(compare(value, result[metric_name], 'trace.metric.'+field))
    assert {'correct', 'steps', 'path_length', 'displacement', 'mean_action_norm'} <= set(checked), 'MISSING_BASIC_METRICS'
    return result, checked, max(errors, default=0.0)


def morphology_row(row, result, source):
    identity = {k: row[k] for k in ('seed', 'variant', 'condition_id', 'state_id', 'goal_index',
                                  'bank', 'role', 'partition', 'distance', 'distance_edges', 'region', 'task') if k in row}
    return dict(source_round=source, measurement='SAVED_TRAJECTORY_REANALYSIS',
                status='AVAILABLE_AND_IDENTITY_VERIFIED', **identity, **{
                    k: json.dumps(v, separators=(',', ':')) if isinstance(v, list) else v
                    for k, v in result.items()}, trace_path=row['trace_path'], trace_sha256=row['trace_sha256'])


def self_check():
    assert len(FREE) == 46
    assert bfs((1, 1))[(7, 10)] == 15
    assert len(firsts((1, 1), (7, 10))) == 2
    assert reversal_count([[0.01, 0], [-1, 0]]) == 0  # strict norm threshold
    assert reversal_count([[1, 0], [-1, 0], [1, 0], [-1, 0]]) == 3
    assert reversal_count([[1, 0], [-0.49, 0.88]]) == 0
    assert repeat_count(np.array([[0., 0.], [1, 0], [0, 0], [0, 0], [0, 0]])) == 2
    assert repeat_count(np.array([[0., 0.], [1, 0], [0, 0]])) == 0
    print('AUDIT06_SELF_CHECK_PASS', flush=True)


def main():
    self_check()
    guard()
    assert not (ROOT/'AUDIT06.json').exists(), 'NO_AUDIT_OVERWRITE'
    new_rows = read_csv(ROOT/'BEHAVIOR_ROLLOUTS.csv')
    assert new_rows and len(new_rows) <= 768
    manifest = json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    bank_path = ROOT/'runs/SCENE_GOAL_BANK.pkl'
    assert sha(bank_path) == manifest['bank_sha256'], 'FROZEN_BANK_IDENTITY_MISMATCH'
    with open(bank_path, 'rb') as f:
        bank_list = pickle.load(f)
    bank = {x['condition_id']: x for x in bank_list}
    assert len(bank) == len(bank_list) <= 128
    expected_conditions = {x['condition_id'] for x in bank_list if x['admitted']}
    expected_models = {(str(x['variant']), str(x['seed'])): x for x in manifest['models']}
    assert set(expected_models) == {(v, str(s)) for v in ('A', 'B') for s in (0, 1, 2)}
    seen, out, audited_fields, physical_steps = set(), [], set(), 0
    maximum_error = 0.0
    for row in new_rows:
        guard()
        key = (row['variant'], row['seed'], row['condition_id'])
        assert key not in seen, ('DUPLICATE_D_CONDITION', key)
        seen.add(key)
        path = Path(row['trace_path'])
        assert path.is_relative_to(ROOT/'runs'), 'D_TRACE_OUTSIDE_ROUND'
        assert sha(path) == row['trace_sha256']
        with open(path, 'rb') as f:
            trace = pickle.load(f)
        assert str(trace['metric']['condition_id']) == row['condition_id']
        frozen = bank[row['condition_id']]
        assert frozen['admitted']
        for k in ('integration', 'qpos', 'qvel'):
            assert np.max(np.abs(np.asarray(trace['initial'][k])-np.asarray(frozen['scene'][k]))) <= 1e-10, ('INITIAL_PHYSICS_MISMATCH', k)
        for k in ('env_rng', 'action_rng', 'numpy_rng', 'python_rng', 'wrappers', 'task', 'reset_seed'):
            assert equal(trace['initial'][k], frozen['scene'][k]), ('INITIAL_AUX_STATE_MISMATCH', k)
        assert np.array_equal(trace['goal_xy'], frozen['goal_xy'])
        assert np.array_equal(trace['goals'][0], frozen['goal_xy'])
        assert np.array_equal(trace['initial']['goal'], frozen['goal_xy'])
        assert np.array_equal(trace['initial']['goal_xy'], frozen['goal_xy'])
        for k in ('bank', 'role', 'region', 'source'):
            assert row[k] == trace['metric'][k] == frozen[k], ('BANK_METADATA_MISMATCH', k)
        model = expected_models[(row['variant'], row['seed'])]
        assert row['checkpoint_sha256'] == trace['metric']['checkpoint_sha256'] == model['sha256']
        assert str(trace['metric']['seed']) == row['seed'] and trace['metric']['variant'] == row['variant']
        assert trace['metric']['state_id'] == row['state_id'] == row['condition_id']
        result, fields, error = audit_record(row, trace)
        assert result['start_cell'] == list(frozen['start_cell'])
        assert result['goal_cell'] == list(frozen['goal_cell'])
        assert result['bfs_distance'] == frozen['distance']
        restore_error = max(float(np.max(np.abs(np.asarray(trace['initial'][k])-np.asarray(frozen['scene'][k])))) for k in ('integration', 'qpos', 'qvel'))
        error = max(error, compare(row['restore_error'], restore_error, 'restore_error'))
        assert result['reference_firsts'] == [list(c) for c in frozen['firsts']]
        audited_fields.update(fields)
        maximum_error = max(maximum_error, error)
        physical_steps += result['steps']
        out.append(morphology_row(row, result, 'SAW-DISTANCE-06_D'))
    groups = {s: {r['variant'] for r in new_rows if int(r['seed']) == s} for s in (0, 1, 2)}
    assert groups == {0: {'A', 'B'}, 1: {'A', 'B'}, 2: {'A', 'B'}}
    banks = [{r['condition_id'] for r in new_rows if r['variant'] == v and int(r['seed']) == s}
             for s in (0, 1, 2) for v in ('A', 'B')]
    assert all(b == banks[0] for b in banks), 'UNMATCHED_MODEL_CONDITIONS'
    assert banks[0] == expected_conditions, 'INCOMPLETE_ADMITTED_CONDITIONS'
    old_rows, unavailable = [], []
    index_path = GP/'RAW_ARTIFACT_INDEX.json'
    old_csv = GP/'BRANCH_ROLLOUTS.csv'
    if index_path.exists() and old_csv.exists():
        index = {x['path']: x for x in json.loads(index_path.read_text())['files']}
        old_rows = [r for r in read_csv(old_csv) if r['group'] == 'P3']
        assert len(old_rows) == 192
        for row in old_rows:
            guard()
            path = Path(row['trace_path'])
            identity = index.get(str(path))
            if not path.is_file():
                unavailable.append(str(path))
                out.append(dict(source_round='SAW-GOALPROBE-05_P3', measurement='SAVED_TRAJECTORY_REANALYSIS',
                                status='RAW_TRAJECTORY_UNAVAILABLE', seed=row['seed'], variant=row['variant'],
                                state_id=row['state_id'], goal_index=row['goal_index'], trace_path=str(path),
                                trace_sha256=row['trace_sha256']))
                continue
            assert identity is not None and identity['sha256'] == row['trace_sha256']
            assert path.stat().st_size == identity['bytes']
            assert sha(path) == row['trace_sha256']
            with open(path, 'rb') as f:
                trace = pickle.load(f)
            assert trace['metric']['group'] == 'P3'
            result, fields, error = audit_record(row, trace, old=True)
            maximum_error = max(maximum_error, error)
            out.append(morphology_row(row, result, 'SAW-GOALPROBE-05_P3'))
    else:
        unavailable = ['P3_SOURCE_INDEX_OR_CSV_UNAVAILABLE']
        out.append(dict(source_round='SAW-GOALPROBE-05_P3', measurement='SAVED_TRAJECTORY_REANALYSIS',
                        status='RAW_TRAJECTORY_UNAVAILABLE'))
    budget = json.loads((ROOT/'receipts/CONTROL_BUDGET.json').read_text())
    assert physical_steps <= 19200 and budget['used'] <= 25000
    formal_key = next((k for k in ('formal_D', 'formal', 'D') if k in budget['by_phase']), None)
    assert formal_key is not None, ('FORMAL_BUDGET_KEY_MISSING', budget['by_phase'])
    assert physical_steps == budget['by_phase'][formal_key]
    assert sum(budget['by_phase'].values()) == budget['used']
    verified_inputs = []
    for identity in list(manifest['models']) + list(manifest['inputs']):
        guard()
        assert sha(identity['path']) == identity['sha256'], ('FROZEN_INPUT_CHANGED', identity['path'])
        verified_inputs.append(identity['path'])
    write_csv(ROOT/'TRAJECTORY_MORPHOLOGY.csv', out)
    strict_json(ROOT/'AUDIT06.json', dict(
        status='PASS', epoch=time.time(), new_D_traces_recomputed=len(new_rows),
        old_P3_traces_recomputed=len(old_rows)-len(unavailable) if old_rows else 0,
        old_P3_status='RAW_TRAJECTORY_UNAVAILABLE' if unavailable else 'PASS',
        unavailable_old_P3_paths=unavailable, total_morphology_rows=len(out),
        new_D_metric_fields_recomputed=sorted(audited_fields),
        maximum_numeric_error=maximum_error, original_trajectories_unchanged=True,
        integration_qpos_qvel_alignment=True, all_shortest_first_channels_accepted=True,
        complete_initial_physics_rng_wrappers_match_frozen_bank=True,
        input_and_environment_goals_match_frozen_bank=True,
        weights_datasets_configs_indices_verified_unchanged=True,
        verified_read_only_input_paths=verified_inputs,
        formal_D_physical_steps=physical_steps, total_new_control_steps=budget['used'],
        no_new_model_forward=True, no_new_simulation=True, no_new_parameter_update=True,
        independent_source_sha256=sha(__file__), thresholds=THRESHOLDS,
        morphology_sha256=sha(ROOT/'TRAJECTORY_MORPHOLOGY.csv')))
    print('AUDIT06_PASS', len(new_rows), len(old_rows)-len(unavailable) if old_rows else 0, flush=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--self-check-only', action='store_true')
    opt = args.parse_args()
    self_check() if opt.self_check_only else main()
