"""Freeze native HGCDataset F/R tuples and fixed-goal conditional neighborhoods.

This module does not load checkpoints, query a policy, or advance physics.
"""
from core import *
from utils.datasets import Dataset, HGCDataset
from scipy.spatial import cKDTree

SAMPLE_SEED = 26061301
BINS = ('0-1', '2-3', '4-5', '6-7', '8-9', '10-11', '12-13', '14-15', '16+')


class Trace(HGCDataset):
    def get_observations(self, idx):
        self.indices.append(np.asarray(idx).copy())
        return super().get_observations(idx)

    def observed(self, indices):
        self.indices = []
        batch = self.sample(len(indices), indices, evaluation=True)
        assert len(self.indices) == 4
        return batch, self.indices


def bin_name(distance):
    if distance < 0:
        return 'GEOMETRY_UNAVAILABLE'
    return BINS[min(distance // 2, 8)]


def geometry_arrays(geom, s, n, w, g):
    cells = [np.asarray([geom.cell(p) for p in points], np.int16) for points in (s, n, w, g)]
    def distances(left, right):
        return np.asarray([geom.distances.get(tuple(b), {}).get(tuple(a), -1)
                           for a, b in zip(left, right)], np.int16)
    ds, dn, dw = [distances(c, cells[3]) for c in cells[:3]]
    valid = (ds >= 0) & (dn >= 0) & (dw >= 0)
    return dict(state_cell=cells[0], next_cell=cells[1], waypoint_cell=cells[2], goal_cell=cells[3],
                graph_distance=ds, next_graph_distance=dn, waypoint_graph_distance=dw,
                delta_w=np.where(valid, ds-dw, -32768).astype(np.int16),
                delta_next=np.where(valid, ds-dn, -32768).astype(np.int16), geometry_valid=valid)


def choose_groups(bank):
    pools = {}
    for item in bank:
        if item['bank'] != 'main' or not item['admitted']:
            continue
        d = int(item['distance'])
        band = 'near' if d in (2, 4) else 'far' if d in (8, 10, 12, 14) else None
        if band is None:
            continue
        key = (tuple(item['start_cell']), band, str(item['region']))
        pools.setdefault(key, []).append(item)
    # One condition per geometric start/band/region stratum, before any result.
    picked = []
    for key in sorted(pools, key=lambda k: (k[0], 0 if k[1] == 'near' else 1, k[2])):
        item = min(pools[key], key=lambda x: (tuple(x['goal_cell']), int(x['distance']), x['condition_id']))
        picked.append(item)
    if len(picked) > 32:
        # Round robin over start locations keeps the cap from excluding late starts.
        starts = sorted(set(tuple(x['start_cell']) for x in picked))
        by_start = {s: [x for x in picked if tuple(x['start_cell']) == s] for s in starts}
        picked = [by_start[s][j] for j in range(max(map(len, by_start.values())))
                  for s in starts if j < len(by_start[s])][:32]
    return picked


def save_npz(path, values):
    assert not path.exists(), f'EXISTS:{path}'
    np.savez_compressed(path, **values)


def main():
    guard()
    started = time.time()
    out = ROOT / 'runs'
    result = ROOT / 'results'
    assert not (result/'SIGNAL_SAMPLE_FREEZE.json').exists()
    bank_path = out/'SCENE_GOAL_BANK.pkl'
    with bank_path.open('rb') as f:
        bank = pickle.load(f)
    conf = get_config().to_dict()
    assert conf['frame_stack'] is None and conf['subgoal_steps'] == 25
    env, loaded, validation = ogbench.make_env_and_datasets(
        'pointmaze-large-stitch-v0', dataset_dir=str(OLD/'vendor/data'), compact_dataset=True)
    dataset = Dataset.create(**loaded)
    raw_path = OLD/'vendor/data/pointmaze-large-stitch-v0.npz'
    with np.load(raw_path) as raw:
        obs = raw['observations'].copy()
        raw_vel = raw['qvel'].copy()
        terminals = np.flatnonzero(raw['terminals'])
    assert np.array_equal(dataset['observations'], obs[:dataset.size]), 'COMPACT_RAW_OBS_MAPPING'
    assert terminals[-1] == len(obs)-1
    geom = Geometry(env)
    valid = dataset.valid_idxs
    np.random.seed(SAMPLE_SEED)
    indices = dataset.get_random_idxs(8192)
    pair_rng = np.random.get_state()
    F = Trace(dataset, dict(conf, actor_p_trajgoal=1., actor_p_randomgoal=0.))
    R = Trace(dataset, dict(conf, actor_p_trajgoal=0., actor_p_randomgoal=1.))
    fb, fi = F.observed(indices)
    after_f = np.random.get_state()
    np.random.set_state(pair_rng)
    rb, ri = R.observed(indices)
    assert equal(after_f, np.random.get_state()), 'PAIRED_RANDOM_CONSUMPTION'
    for key in ('observations', 'actions', 'next_observations', 'value_goals', 'low_actor_goals', 'rewards', 'masks'):
        assert np.array_equal(fb[key], rb[key]), key
    end = F.terminal_locs[np.searchsorted(F.terminal_locs, indices)]
    assert np.all(dataset['valids'][indices] > 0)
    assert np.array_equal(fi[3], np.minimum(indices+25, fi[2]))
    assert np.array_equal(ri[3], np.minimum(indices+25, end))
    assert np.all((fi[2] >= indices) & (fi[2] <= end))
    assert np.all(dataset['valids'][ri[2]] > 0)
    assert np.array_equal(fb['rewards'], (indices == fi[0]).astype(float)-1.)
    assert np.array_equal(fb['masks'], (indices != fi[0]).astype(float))
    tuple_rows = []
    source_paths = []
    for branch, batch, sampled in [('F', fb, fi), ('R', rb, ri)]:
        values = dict(batch)
        values.update(state_idx=indices, next_idx=np.minimum(indices+1, dataset.size-1),
                      value_goal_idx=sampled[0], low_goal_idx=sampled[1], goal_idx=sampled[2],
                      waypoint_idx=sampled[3], compact_end=end,
                      original_trajectory=np.searchsorted(terminals, indices),
                      goal_original_trajectory=np.searchsorted(terminals, sampled[2]))
        values.update(geometry_arrays(geom, batch['observations'], batch['next_observations'],
                                      batch['high_actor_targets'], batch['high_actor_goals']))
        path = out/f'S1_{branch}_TUPLES.npz'
        save_npz(path, values)
        source_paths.append(dict(branch=branch, path=str(path), sha256=sha(path), count=len(indices)))
        for j, i in enumerate(indices):
            d = int(values['graph_distance'][j])
            tuple_rows.append(dict(tuple_id=f'S1_{branch}_{j:05d}', pair_id=j, branch=branch,
                state_idx=int(i), next_idx=int(values['next_idx'][j]),
                value_goal_idx=int(sampled[0][j]), low_goal_idx=int(sampled[1][j]),
                goal_idx=int(sampled[2][j]), waypoint_idx=int(sampled[3][j]), compact_end=int(end[j]),
                original_trajectory=int(values['original_trajectory'][j]),
                goal_original_trajectory=int(values['goal_original_trajectory'][j]),
                actually_cross_trajectory=int(values['original_trajectory'][j] != values['goal_original_trajectory'][j]),
                reward=float(batch['rewards'][j]), mask=float(batch['masks'][j]),
                graph_distance=d, distance_bin=bin_name(d), delta_w=int(values['delta_w'][j]),
                delta_next=int(values['delta_next'][j]), geometry_valid=int(values['geometry_valid'][j])))
    rows(result/'SIGNAL_TUPLES.csv', tuple_rows)
    tree = cKDTree(obs[valid])
    groups = []
    s2_arrays = {key: [] for key in ('observations', 'next_observations', 'actions', 'high_actor_goals',
                                   'high_actor_targets', 'state_idx', 'goal_idx', 'waypoint_idx',
                                   'original_trajectory', 'group_index', 'state_position_error', 'velocity_error')}
    for group_index, item in enumerate(choose_groups(bank)):
        guard()
        ref = np.asarray(item['scene']['qpos'], float)
        goal = np.asarray(item['goal_xy'], float)
        candidates = valid[np.asarray(tree.query_ball_point(goal, .5), int)]
        distances = np.linalg.norm(obs[candidates]-goal, axis=1)
        keep = distances <= .5
        candidates, distances = candidates[keep], distances[keep]
        base = dict(group_id=f'S2_{group_index:02d}', group_index=group_index,
                    condition_id=item['condition_id'], start_cell=item['start_cell'], goal_cell=item['goal_cell'],
                    reference_position=ref.tolist(), reference_velocity=np.asarray(item['scene']['qvel']).tolist(),
                    reference_goal=goal.tolist(), distance=int(item['distance']), region=str(item['region']))
        if len(candidates) == 0:
            groups.append(dict(**base, status='NO_GOAL_RECORD_WITHIN_0_5', n=0, original_trajectories=0))
            continue
        chosen_goal = int(candidates[np.lexsort((candidates, distances))[0]])
        candidate_idx = valid[np.asarray(tree.query_ball_point(ref, .15), int)]
        errors = np.linalg.norm(obs[candidate_idx]-ref, axis=1)
        keep = errors <= .15
        candidate_idx, errors = candidate_idx[keep], errors[keep]
        traj = np.searchsorted(terminals, candidate_idx)
        order = np.lexsort((candidate_idx, traj, errors))
        ordered = candidate_idx[order]
        ordered_traj = traj[order]
        first = []; seen = set()
        for k, t in zip(ordered, ordered_traj):
            if int(t) not in seen:
                first.append(int(k)); seen.add(int(t))
        chosen = first[:64]
        used = set(chosen)
        chosen += [int(k) for k in ordered if int(k) not in used][:64-len(chosen)]
        idx = np.asarray(chosen, int)
        if len(idx):
            # Preserve the registered distance, original trajectory, then index ordering in storage.
            err = np.linalg.norm(obs[idx]-ref, axis=1)
            ot = np.searchsorted(terminals, idx)
            idx = idx[np.lexsort((idx, ot, err))]
        n = len(idx)
        count_traj = len(set(np.searchsorted(terminals, idx).tolist()))
        status = 'ELIGIBLE' if n >= 16 and count_traj >= 4 else 'INSUFFICIENT_CONDITIONAL_SUPPORT'
        goal_record = obs[chosen_goal]
        group = dict(**base, status=status, n=n, original_trajectories=count_traj,
                     fixed_goal_idx=chosen_goal, fixed_goal=goal_record.tolist(),
                     goal_record_distance=float(np.linalg.norm(goal_record-goal)),
                     state_indices=idx.tolist(), natural_frequency_interpretation=False)
        groups.append(group)
        if not n:
            continue
        ends = R.terminal_locs[np.searchsorted(R.terminal_locs, idx)]
        wp = np.minimum(idx+25, ends)
        b = dataset.sample(n, idx)
        values = dict(observations=b['observations'], next_observations=b['next_observations'], actions=b['actions'],
                      high_actor_goals=np.repeat(goal_record[None], n, axis=0), high_actor_targets=obs[wp],
                      state_idx=idx, goal_idx=np.full(n, chosen_goal, int), waypoint_idx=wp,
                      original_trajectory=np.searchsorted(terminals, idx), group_index=np.full(n, group_index, int),
                      state_position_error=np.linalg.norm(obs[idx]-ref, axis=1),
                      velocity_error=np.linalg.norm(raw_vel[idx]-np.asarray(item['scene']['qvel']), axis=1))
        for key in s2_arrays:
            s2_arrays[key].append(values[key])
    shaped = {'observations', 'next_observations', 'actions', 'high_actor_goals', 'high_actor_targets'}
    s2 = {key: np.concatenate(chunks) if chunks else np.empty((0,2) if key in shaped else (0,), dtype=float)
          for key, chunks in s2_arrays.items()}
    s2.update(geometry_arrays(geom, s2['observations'], s2['next_observations'], s2['high_actor_targets'], s2['high_actor_goals']))
    assert len(s2['state_idx']) <= 2048
    path = out/'S2_TUPLES.npz'
    save_npz(path, s2)
    write(result/'S2_GROUPS.json', groups)
    assert 16384 + len(s2['state_idx']) + 6*64 + 6*len(groups) <= 32768
    write(result/'SIGNAL_SAMPLE_FREEZE.json', dict(status='FROZEN_BEFORE_ANY_CURRENT_MODEL_QUERY',
        sample_seed=SAMPLE_SEED, bootstrap_seed=26061303, bootstrap_repetitions=500,
        bootstrap_unit='original_trajectory; identical F/R base indices retained',
        S1_pairs=8192, S1_tuple_count=16384, S1_unique_base_indices=len(np.unique(indices)),
        S2_groups=len(groups), S2_eligible_groups=sum(x['status']=='ELIGIBLE' for x in groups),
        S2_tuple_count=len(s2['state_idx']), probe_cap=32768,
        worst_case_count_including_six_model_consistency_and_ref_queries=16384+len(s2['state_idx'])+6*64+6*len(groups),
        source_dataset=dict(path=str(raw_path), sha256=sha(raw_path), bytes=raw_path.stat().st_size),
        frozen_bank_sha256=sha(bank_path), native_sampler='HGCDataset; card HGCDDataset is a spelling variation',
        S1=source_paths, S2=dict(path=str(path), sha256=sha(path)),
        S2_group_selection='one lexicographic condition per start x near/far x region; round robin start if >32',
        S2_neighborhood_selection='nearest representative per original trajectory, then nearest remaining; <=64; final distance/trajectory/index sort',
        S2_high_weight_direction_audit='top quartile of each loss coefficient; continuous minimum cosine/max angle, no multimodality verdict',
        paired_checks=dict(shared_rng_consumption=True, shared_reward_masks_and_base=True,
                           future_waypoint=True, random_waypoint=True, random_goals_valid=True),
        seconds=time.time()-started, parameter_updates=0, physical_control_steps=0))
    env.close()
    print('SIGNAL_PREPARE_FROZEN', len(groups), len(s2['state_idx']), flush=True)


if __name__ == '__main__':
    main()
