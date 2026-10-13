"""SAW-DISTANCE-06: frozen behavior summaries, paired contrasts and full-data figures.

This module only reads archived measurements. It never loads a policy, advances
physics, computes a gradient, or selects new scientific conditions. Main contrasts
average targets within distance, distances within band, channels within start,
then starts. Confidence intervals resample entire starting-cell clusters.
"""
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core import ROOT, guard, rows, sha, write

NEAR = (2, 4)
FAR = (8, 10, 12, 14)
MAIN_DISTANCES = (2, 4, 6, 8, 10, 12, 14)
STAT_SEED = 26061303
N_BOOT = 5000
VARIANTS = ('A', 'B')
SEEDS = (0, 1, 2)


def clean(value):
    """Convert pandas/NumPy values to strict finite JSON-compatible scalars."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def cell_string(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError('CELL_SHAPE')
    return json.dumps([int(value[0]), int(value[1])], separators=(',', ':'))


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        if value not in (0, 1):
            raise ValueError('BOOLEAN_OUT_OF_RANGE')
        return bool(value)
    if str(value).lower() in ('1', 'true'):
        return True
    if str(value).lower() in ('0', 'false'):
        return False
    raise ValueError(f'INVALID_BOOLEAN: {value!r}')


def records(frame):
    return [clean(row) for row in frame.to_dict('records')]


def save_table(path, values, fields=None):
    """Derived exports are created once; raw input tables are never rewritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'DERIVED_EXPORT_ALREADY_EXISTS: {path}')
    if isinstance(values, pd.DataFrame):
        fields = list(values.columns)
        values = records(values)
    values = [clean(value) for value in values]
    if fields is None:
        fields = list(dict.fromkeys(key for row in values for key in row))
    fields = fields or ['status']
    rows(path, [{key: row.get(key) for key in fields} for row in values], fields)


def load_measurements(path):
    data = pd.read_csv(path)
    required = {
        'condition_id', 'bank', 'start_cell', 'goal_cell', 'region',
        'distance', 'firsts', 'variant', 'seed', 'correct', 'category',
        'first_exit_step', 'steps', 'mean_action_norm', 'median_action_norm',
        'saturation_fraction', 'displacement', 'maximum_displacement',
        'path_length', 'graph_progress', 'initial_channel_projection',
        'displacement_reversals', 'action_reversals', 'repeated_visit_steps',
        'small_command_candidate', 'near_stationary', 'backtracking_evidence',
    }
    if not required.issubset(data.columns):
        raise ValueError(f'MISSING_COLUMNS: {sorted(required - set(data.columns))}')
    if data.empty:
        raise ValueError('NO_BEHAVIOR_MEASUREMENTS')
    if data.duplicated(['condition_id', 'variant', 'seed']).any():
        raise ValueError('DUPLICATED_CONDITION_MODEL')
    for column in ('correct', 'small_command_candidate', 'near_stationary', 'backtracking_evidence'):
        data[column] = data[column].map(as_bool).astype(int)
    for column in ('seed', 'distance', 'steps', 'displacement_reversals', 'action_reversals', 'repeated_visit_steps'):
        data[column] = pd.to_numeric(data[column], errors='raise').astype(int)
    if not set(data['seed']).issubset(SEEDS) or not set(data['variant']).issubset(VARIANTS):
        raise ValueError('UNREGISTERED_MODEL')
    if (data['steps'] > 25).any() or (data['steps'] < 0).any():
        raise ValueError('LOCAL_HORIZON_VIOLATION')
    for column in ('start_cell', 'goal_cell'):
        data[column] = data[column].map(cell_string)
    parsed_firsts = data['firsts'].map(lambda value: json.loads(value) if isinstance(value, str) else value)
    data['first_count'] = parsed_firsts.map(len)
    data['firsts'] = parsed_firsts.map(lambda value: json.dumps(
        [[int(x) for x in v] for v in value], separators=(',', ':')))
    data['channel'] = parsed_firsts.map(lambda value: cell_string(value[0]) if len(value) == 1 else None)
    data['is_main'] = data['bank'].astype(str).str.lower().str.contains('main', regex=False)
    if not data['is_main'].any():
        raise ValueError('MAIN_BANK_LABEL_MUST_CONTAIN_MAIN')
    if (data.loc[data['is_main'], 'first_count'] != 1).any():
        raise ValueError('MAIN_BANK_NONUNIQUE_FIRST_CHANNEL')
    if not set(data.loc[data['is_main'], 'distance']).issubset(MAIN_DISTANCES):
        raise ValueError('NONREGISTERED_MAIN_DISTANCE')
    for _, group in data.groupby('condition_id', sort=True):
        for field in ('bank', 'start_cell', 'goal_cell', 'region', 'distance', 'firsts'):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(f'CONDITION_METADATA_CHANGED: {field}')
    # Incomplete conditions remain in the complete-data exports, but contribute
    # to neither paired estimates nor summaries presented as six-model controls.
    counts = data.groupby('condition_id').size()
    pairs = {condition: set(zip(frame['variant'], frame['seed']))
             for condition, frame in data.groupby('condition_id', sort=True)}
    complete = {condition for condition, count in counts.items()
                if count == 6 and pairs[condition] == set((v, s) for v in VARIANTS for s in SEEDS)}
    data['model_complete'] = data['condition_id'].isin(complete)
    data['distance_band'] = np.select(
        [data['distance'].isin(NEAR), data['distance'].isin(FAR), data['distance'] == 6],
        ['near', 'far', 'middle'], default='auxiliary')
    return data


def get_common_blocks(main):
    """Common support is derived from admitted measured conditions, never candidates."""
    bank = main.drop_duplicates('condition_id')
    support = []
    for (start, channel), group in bank.groupby(['start_cell', 'channel'], sort=True):
        distances = sorted(int(v) for v in group['distance'].unique())
        near = [v for v in distances if v in NEAR]
        far = [v for v in distances if v in FAR]
        support.append(dict(
            start_cell=start, channel=channel, near_distances=json.dumps(near),
            far_distances=json.dumps(far), all_distances=json.dumps(distances),
            common_support=bool(near and far), conditions=len(group),
            near_regions=json.dumps(sorted(group.loc[group['distance'].isin(NEAR), 'region'].astype(str).unique())),
            far_regions=json.dumps(sorted(group.loc[group['distance'].isin(FAR), 'region'].astype(str).unique())),
        ))
    supported = {(row['start_cell'], row['channel']) for row in support if row['common_support']}
    mask = [(start, channel) in supported for start, channel in zip(main['start_cell'], main['channel'])]
    return main.loc[mask].copy(), support


def start_estimands(common):
    output = []
    if common.empty:
        return pd.DataFrame(columns=['seed', 'start_cell', 'channels', 'A_near', 'B_near', 'A_far', 'B_far', 'gap_near', 'gap_far', 'distance_interaction'])
    goal_means = common.groupby(['seed', 'start_cell', 'channel', 'distance', 'variant'], sort=True)['correct'].mean()
    channel_values = []
    for (seed, start, channel), frame in goal_means.reset_index().groupby(['seed', 'start_cell', 'channel'], sort=True):
        item = dict(seed=int(seed), start_cell=start, channel=channel)
        for variant in VARIANTS:
            for band, distances in (('near', NEAR), ('far', FAR)):
                selected = frame.loc[(frame['variant'] == variant) & frame['distance'].isin(distances), 'correct']
                if selected.empty:
                    raise ValueError('INCOMPLETE_PAIRED_COMMON_BLOCK')
                item[f'{variant}_{band}'] = float(selected.mean())
        channel_values.append(item)
    channel_values = pd.DataFrame(channel_values)
    for (seed, start), group in channel_values.groupby(['seed', 'start_cell'], sort=True):
        item = dict(seed=int(seed), start_cell=start, channels=len(group))
        for field in ('A_near', 'B_near', 'A_far', 'B_far'):
            item[field] = float(group[field].mean())
        item['gap_near'] = item['A_near'] - item['B_near']
        item['gap_far'] = item['A_far'] - item['B_far']
        item['distance_interaction'] = item['gap_far'] - item['gap_near']
        output.append(item)
    return pd.DataFrame(output)


def estimate_contrasts(common, support):
    clusters = start_estimands(common)
    starts = sorted(common['start_cell'].unique())
    distances = sorted(int(v) for v in common['distance'].unique())
    far_regions = sorted(common.loc[common['distance'].isin(FAR), 'region'].astype(str).unique())
    support_ok = bool(len(starts) >= 4 and len(distances) >= 3
                      and any(v in NEAR for v in distances) and any(v in FAR for v in distances))
    support_meta = dict(
        starting_clusters=len(starts), starts=starts, distances=distances,
        common_blocks=sum(bool(row['common_support']) for row in support),
        far_regions=far_regions, geometry_support_pass=support_ok,
        far_region_support_pass=len(far_regions) > 1,
        common_condition_count=int(common['condition_id'].nunique()),
    )
    rng = np.random.default_rng(STAT_SEED)
    draw = rng.integers(0, len(starts), size=(N_BOOT, len(starts))) if starts else None
    metrics = ('A_near', 'B_near', 'A_far', 'B_far', 'gap_near', 'gap_far', 'distance_interaction')
    contrasts = []
    for seed in SEEDS:
        group = clusters.loc[clusters['seed'] == seed].set_index('start_cell').reindex(starts)
        item = dict(seed=seed, historical_discovery_model=seed == 0,
                    n_start_clusters=len(starts), n_common_blocks=support_meta['common_blocks'],
                    n_common_conditions=support_meta['common_condition_count'],
                    common_distances=json.dumps(distances), far_regions=json.dumps(far_regions),
                    support_pass=support_ok, far_region_support_pass=len(far_regions) > 1,
                    bootstrap_replicates=N_BOOT, bootstrap_seed=STAT_SEED,
                    bootstrap_unit='starting_cell', confidence_scope='fixed_models_starting_position_uncertainty')
        if starts:
            if group[list(metrics)].isna().any().any():
                raise ValueError('MODEL_SPECIFIC_COMMON_SUPPORT')
            matrix = group[list(metrics)].to_numpy(dtype=float)
            point = matrix.mean(axis=0)
            bootstrap = matrix[draw].mean(axis=1)
            low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
            for index, metric in enumerate(metrics):
                scale = 100 if metric.startswith('gap_') or metric == 'distance_interaction' else 1
                name = f'{metric}_pp' if scale == 100 else f'CBR_{metric}'
                item[name] = float(point[index] * scale)
                item[f'{name}_ci_low'] = float(low[index] * scale)
                item[f'{name}_ci_high'] = float(high[index] * scale)
            item['gate_far_gap'] = bool(item['gap_far_pp'] >= 30)
            item['gate_near_gap'] = bool(abs(item['gap_near_pp']) < 10)
            item['gate_B_near'] = bool(item['CBR_B_near'] >= .80)
            item['gate_interaction_size'] = bool(item['distance_interaction_pp'] >= 20)
            item['gate_interaction_interval'] = bool(item['distance_interaction_pp_ci_low'] > 0)
            item['operational_gate_pass'] = all(bool(item[key]) for key in (
                'support_pass', 'far_region_support_pass', 'gate_far_gap',
                'gate_near_gap', 'gate_B_near', 'gate_interaction_size',
                'gate_interaction_interval'))
            item['verdict'] = ('DISTANCE_PATTERN_SUPPORTED_CONDITIONALLY' if item['operational_gate_pass']
                               else ('DISTANCE_PATTERN_NOT_SUPPORTED_BY_REGISTERED_GATE' if support_ok
                                     else 'INSUFFICIENT_COMMON_SUPPORT'))
        else:
            item['operational_gate_pass'] = False
            item['verdict'] = 'INSUFFICIENT_COMMON_SUPPORT'
        contrasts.append(item)
    confirmed = all(row['operational_gate_pass'] for row in contrasts if row['seed'] in (1, 2))
    verdict = ('DISTANCE_PATTERN_SUPPORTED_CONDITIONALLY' if confirmed else
               ('DISTANCE_PATTERN_NOT_SUPPORTED_BY_REGISTERED_GATE' if support_ok else 'INSUFFICIENT_COMMON_SUPPORT'))
    return contrasts, clusters, dict(**support_meta, verdict=verdict, per_seed=contrasts,
                                    estimator_order='target_mean_then_distance_mean_then_channel_mean_then_start_mean',
                                    bootstrap='5000 entire-start clusters with replacement; common draw matrix across seeds',
                                    bootstrap_seed=STAT_SEED,
                                    seed_scope='three previously observed training seeds; seed0 historical discovery')


def distance_table(data):
    measured = data.loc[data['model_complete']].copy()
    measured['series'] = np.select(
        [measured['is_main'], measured['first_count'] > 1],
        ['main_unique', 'auxiliary_tie'], default='auxiliary_unique_anchor')
    output = []
    for (seed, variant, series, distance), group in measured.groupby(['seed', 'variant', 'series', 'distance'], sort=True):
        if series == 'main_unique':
            level = group.groupby(['start_cell', 'channel'], sort=True)['correct'].mean().reset_index()
            start_values = level.groupby('start_cell')['correct'].mean()
        else:
            start_values = group.groupby('start_cell')['correct'].mean()
        output.append(dict(seed=int(seed), variant=variant, series=series, distance=int(distance),
                           CBR25=float(start_values.mean()), starting_cells=len(start_values),
                           conditions=group['condition_id'].nunique(), regions=group['region'].nunique(),
                           weighting='targets_then_channels_then_starts' if series == 'main_unique' else 'targets_then_starts',
                           caution='coverage may change across distances; main decision uses paired common support'))
    return pd.DataFrame(output)


def region_and_tie_controls(main, auxiliary):
    controls = []
    # Exact same start, unique first channel, distance and training seed. Region
    # means use all available targets; this is conditional association, not cause.
    for (start, channel, distance, seed), group in main.groupby(['start_cell', 'channel', 'distance', 'seed'], sort=True):
        regions = sorted(group['region'].astype(str).unique())
        if len(regions) < 2:
            controls.append(dict(control_type='cross_region_support', start_cell=start,
                                 channel=channel, distance=int(distance), seed=int(seed),
                                 status='NO_CROSS_REGION_SUPPORT', region_1=regions[0] if regions else None))
        else:
            for first, second in combinations(regions, 2):
                item = dict(control_type='same_start_channel_distance_cross_region',
                            start_cell=start, channel=channel, distance=int(distance), seed=int(seed),
                            region_1=first, region_2=second, status='MEASURED_DESCRIPTIVE_REGION_CONTRAST')
                for variant in VARIANTS:
                    for number, region in ((1, first), (2, second)):
                        frame = group.loc[(group['variant'] == variant) & (group['region'].astype(str) == region)]
                        item[f'{variant}_CBR_region_{number}'] = float(frame['correct'].mean())
                        item[f'{variant}_conditions_region_{number}'] = len(frame)
                item['gap_region_1_pp'] = 100 * (item['A_CBR_region_1'] - item['B_CBR_region_1'])
                item['gap_region_2_pp'] = 100 * (item['A_CBR_region_2'] - item['B_CBR_region_2'])
                item['gap_region_2_minus_1_pp'] = item['gap_region_2_pp'] - item['gap_region_1_pp']
                controls.append(item)
    # Auxiliary matches are fixed by geometry: same starting cell and distance,
    # same region preferred, then condition-id lexical order. Preserve all anchors.
    for condition, group in auxiliary.groupby('condition_id', sort=True):
        metadata = group.iloc[0]
        candidates = main.loc[(main['start_cell'] == metadata['start_cell']) & (main['distance'] == metadata['distance'])]
        candidate_metadata = candidates.drop_duplicates('condition_id').copy()
        if not candidate_metadata.empty:
            candidate_metadata['_region_match'] = candidate_metadata['region'].astype(str) == str(metadata['region'])
            candidate_metadata = candidate_metadata.sort_values(['_region_match', 'condition_id'], ascending=[False, True])
            match = candidate_metadata.iloc[0]
            matched = main.loc[main['condition_id'] == match['condition_id']]
        else:
            match = None
            matched = main.iloc[:0]
        for _, row in group.sort_values(['seed', 'variant']).iterrows():
            item = dict(control_type='tie_to_unique' if row['first_count'] > 1 else 'official_unique_anchor',
                        auxiliary_condition_id=condition, auxiliary_bank=row['bank'], start_cell=row['start_cell'],
                        distance=int(row['distance']), region_1=str(row['region']), seed=int(row['seed']), variant=row['variant'],
                        auxiliary_CBR25=int(row['correct']), auxiliary_displacement_reversals=int(row['displacement_reversals']),
                        auxiliary_backtracking_evidence=int(row['backtracking_evidence']),
                        auxiliary_firsts=row['firsts'], status='UNMATCHED_AUXILIARY_ANCHOR')
            if match is not None:
                reference = matched.loc[(matched['seed'] == row['seed']) & (matched['variant'] == row['variant'])]
                if len(reference) != 1:
                    raise ValueError('AUXILIARY_MATCH_MODEL_INCOMPLETE')
                reference = reference.iloc[0]
                item.update(main_condition_id=reference['condition_id'], channel=reference['channel'],
                            region_2=str(reference['region']), same_region=str(row['region']) == str(reference['region']),
                            main_CBR25=int(reference['correct']),
                            auxiliary_minus_main_CBR_pp=100 * (int(row['correct']) - int(reference['correct'])),
                            main_displacement_reversals=int(reference['displacement_reversals']),
                            auxiliary_minus_main_displacement_reversals=int(row['displacement_reversals']) - int(reference['displacement_reversals']),
                            status='MATCHED_START_DISTANCE_DESCRIPTIVE',
                            remaining_confounding='different_exact_goal; different_first-channel_set; region_if_unmatched')
            controls.append(item)
    return controls


def region_trends(main):
    region_distances = []
    region_contrasts = []
    supported_regions = []
    for region, group in main.groupby('region', sort=True):
        common, support = get_common_blocks(group)
        if not common.empty:
            supported_regions.append(str(region))
        contrasts, starts, meta = estimate_contrasts(common, support)
        for item in contrasts:
            item = dict(item, region=str(region),
                        interpretation='within-region descriptive near-far association; limited start clusters',
                        region_common_support=not common.empty)
            # The global operational gate is not a separately registered
            # hypothesis test for every region. Preserve estimates, not a regional
            # declaration of H-DIST significance.
            item.pop('operational_gate_pass', None)
            item['verdict'] = 'DESCRIPTIVE_REGION_STRATUM' if not common.empty else 'NO_NEAR_FAR_REGION_COMMON_SUPPORT'
            region_contrasts.append(item)
        for (seed, variant, distance), frame in group.groupby(['seed', 'variant', 'distance'], sort=True):
            level = frame.groupby(['start_cell', 'channel'])['correct'].mean().reset_index()
            values = level.groupby('start_cell')['correct'].mean()
            region_distances.append(dict(region=str(region), seed=int(seed), variant=variant, distance=int(distance),
                                         CBR25=float(values.mean()), conditions=len(frame), starts=len(values),
                                         channels=len(level), near_far_common_support=not common.empty))
    return region_distances, region_contrasts, supported_regions


def same_goal_controls(main):
    output = []
    for (goal, seed, variant), group in main.groupby(['goal_cell', 'seed', 'variant'], sort=True):
        conditions = group.drop_duplicates('condition_id').sort_values(['distance', 'start_cell', 'condition_id'])
        for first, second in combinations(list(conditions.to_dict('records')), 2):
            if first['start_cell'] == second['start_cell'] or first['distance'] == second['distance']:
                continue
            output.append(dict(goal_cell=goal, region=str(first['region']), seed=int(seed), variant=variant,
                               near_condition_id=first['condition_id'], farther_condition_id=second['condition_id'],
                               nearer_start=first['start_cell'], farther_start=second['start_cell'],
                               nearer_distance=int(first['distance']), farther_distance=int(second['distance']),
                               nearer_channel=first['channel'], farther_channel=second['channel'],
                               nearer_CBR25=int(first['correct']), farther_CBR25=int(second['correct']),
                               farther_minus_nearer_CBR_pp=100 * (int(second['correct']) - int(first['correct'])),
                               remaining_confounding='starting physical state and first channel differ; not a causal distance contrast'))
    return output


def counterexamples(data):
    output = data.copy()
    counterpart = {(row.condition_id, int(row.seed), row.variant): int(row.correct)
                   for row in data.itertuples()}
    flags = []
    for row in data.itertuples():
        reasons = []
        if not row.correct:
            reasons.append('LOCAL_CHANNEL_FAILURE_RETAINED')
        if row.variant == 'B' and row.distance in NEAR and not row.correct:
            reasons.append('B_NEAR_FAILURE')
        if row.variant == 'B' and row.distance in FAR and row.correct:
            reasons.append('B_FAR_SUCCESS')
        if row.variant == 'A' and not row.correct:
            reasons.append('A_FAILURE_CHALLENGES_B_SPECIFICITY')
        if row.variant == 'B' and row.correct and counterpart.get((row.condition_id, int(row.seed), 'A')) == 0:
            reasons.append('B_SUCCESS_A_FAILURE')
        if not row.model_complete:
            reasons.append('INCOMPLETE_MODEL_COMPARISON')
        flags.append(';'.join(reasons))
    output['counterexample_flags'] = flags
    # Keep every condition, including rows with no flagged exception. This table
    # cannot become a curated subset that hides successful far targets.
    return output


def plot_distance(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    colors = {'A': '#1764AB', 'B': '#D14B38'}
    for seed, axis in zip(SEEDS, axes):
        frame = summary.loc[summary['seed'] == seed]
        for variant in VARIANTS:
            for series, style, marker in (('main_unique', '-', 'o'), ('auxiliary_tie', ':', 's'), ('auxiliary_unique_anchor', '', '^')):
                group = frame.loc[(frame['variant'] == variant) & (frame['series'] == series)].sort_values('distance')
                if group.empty:
                    continue
                axis.plot(group['distance'], group['CBR25'], linestyle=style, marker=marker,
                          color=colors[variant], alpha=1 if series == 'main_unique' else .65,
                          label=f'{variant}: {series.replace("_", " ")}', markersize=5)
        axis.set_title(f'Training seed {seed}' + (' (historical discovery)' if seed == 0 else ''))
        axis.set_xlabel('BFS distance (edges)')
        axis.set_xticks(sorted(summary['distance'].unique()))
        axis.set_ylim(-.03, 1.03)
        axis.grid(alpha=.2)
    axes[0].set_ylabel('CBR25 (first exit only)')
    handles, labels = [], []
    for axis in axes:
        hs, ls = axis.get_legend_handles_labels()
        for handle, label in zip(hs, ls):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, -.04))
    fig.suptitle('All frozen conditions; main curves use equal start/channel weights\nCoverage changes across distance; formal decision uses paired common support', fontsize=11)
    fig.tight_layout(rect=(0, .08, 1, .91))
    fig.savefig(output / 'distance_CBR_by_seed.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_regions(main, output):
    regions = sorted(main['region'].astype(str).unique())
    channels = sorted(set(zip(main['start_cell'], main['channel'])))
    labels = [f'{start} -> {channel}' for start, channel in channels]
    for seed in SEEDS:
        fig, axes = plt.subplots(2, 2, figsize=(15, max(9, len(channels) * .43 + 3)), sharex=True, sharey=True)
        for index, axis in enumerate(axes.flat):
            if index >= len(regions):
                axis.set_visible(False)
                continue
            region = regions[index]
            subset = main.loc[(main['seed'] == seed) & (main['region'].astype(str) == region)]
            grid = np.full((len(channels), len(MAIN_DISTANCES)), np.nan)
            counts = np.zeros_like(grid, dtype=int)
            for (start, channel, distance), group in subset.groupby(['start_cell', 'channel', 'distance']):
                ri = channels.index((start, channel))
                ci = MAIN_DISTANCES.index(int(distance))
                a = group.loc[group['variant'] == 'A', 'correct']
                b = group.loc[group['variant'] == 'B', 'correct']
                grid[ri, ci] = float(a.mean() - b.mean())
                counts[ri, ci] = int(group['condition_id'].nunique())
            colormap = plt.get_cmap('RdBu_r').copy()
            colormap.set_bad('#F3F3F3')
            im = axis.imshow(np.ma.masked_invalid(grid), vmin=-1, vmax=1, cmap=colormap, aspect='auto')
            for ri, ci in np.argwhere(np.isfinite(grid)):
                axis.text(ci, ri, f'{grid[ri, ci]:+.2f}\nn={counts[ri, ci]}', ha='center', va='center',
                          fontsize=7, color='white' if abs(grid[ri, ci]) > .6 else 'black')
            axis.set_title(f'Region {region}')
            axis.set_xticks(range(len(MAIN_DISTANCES)), MAIN_DISTANCES)
            axis.set_yticks(range(len(channels)), labels, fontsize=8)
            axis.set_xlabel('BFS distance (edges)')
            axis.set_ylabel('Starting cell -> unique first channel')
        fig.suptitle(f'Seed {seed}: A minus B CBR25 by region, start and first channel\nAll measured main-bank targets; n is target count, gray is missing support', fontsize=11)
        fig.subplots_adjust(left=.16, right=.89, top=.91, bottom=.06, wspace=.3, hspace=.25)
        caxis = fig.add_axes([.92, .18, .015, .65])
        fig.colorbar(im, cax=caxis, label='CBR25 gap (A - B)')
        fig.savefig(output / f'region_channel_seed{seed}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_morphology(data, output):
    colors = {'near': '#2573AF', 'middle': '#8E6BA1', 'far': '#D4552A', 'auxiliary': '#444444'}
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10), sharex=True, sharey=True)
    for ri, seed in enumerate(SEEDS):
        for ci, variant in enumerate(VARIANTS):
            axis = axes[ri, ci]
            frame = data.loc[(data['seed'] == seed) & (data['variant'] == variant)]
            for band, color in colors.items():
                selected = frame.loc[frame['distance_band'] == band]
                if not selected.empty:
                    axis.scatter(selected['mean_action_norm'], selected['displacement_reversals'],
                                 c=color, s=26, alpha=.6, label=f'{band} (n={len(selected)})')
            flagged = frame.loc[frame['backtracking_evidence'] == 1]
            if not flagged.empty:
                axis.scatter(flagged['mean_action_norm'], flagged['displacement_reversals'],
                             facecolors='none', edgecolors='black', marker='s', s=78,
                             label=f'Backtracking gate (n={len(flagged)})')
            axis.axvline(.25, color='gray', linestyle=':', linewidth=1)
            axis.axhline(3, color='gray', linestyle=':', linewidth=1)
            axis.set_title(f'{variant}, training seed {seed}')
            axis.grid(alpha=.2)
            axis.legend(fontsize=7, loc='upper right')
            axis.set_xlabel('Mean clipped action norm')
            axis.set_ylabel('Actual displacement reversals')
    fig.suptitle('Every measured condition: command size and actual motion reversal\nBacktracking additionally requires path length >=1 and >=2 repeated visits', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output / 'action_norm_and_actual_reversals.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    guard()
    result = ROOT / 'results'
    figures = ROOT / 'figures'
    result.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    input_path = result / 'BEHAVIOR_ROLLOUTS.csv'
    if not input_path.exists():
        input_path = ROOT / 'BEHAVIOR_ROLLOUTS.csv'
    bank_path = ROOT / 'runs' / 'SCENE_GOAL_BANK.pkl'
    if not bank_path.exists():
        raise FileNotFoundError('FROZEN_SCENE_GOAL_BANK_MISSING')
    # The bank is a trusted locally produced artifact, not an untrusted download.
    # Read its structure and retain identity without assumptions about key names.
    with bank_path.open('rb') as file:
        bank = pickle.load(file)
    if not isinstance(bank, list) or not all(isinstance(item, dict) for item in bank):
        raise ValueError('FROZEN_BANK_STRUCTURE')
    bank_length = len(bank)
    data = load_measurements(input_path)
    frozen = {item['condition_id']: item for item in bank}
    if len(frozen) != len(bank):
        raise ValueError('FROZEN_BANK_DUPLICATE_CONDITION_ID')
    manifest = json.loads((ROOT / 'FROZEN_MANIFEST.json').read_text())
    if sha(bank_path) != manifest['bank_sha256']:
        raise ValueError('FROZEN_BANK_IDENTITY_CHANGED')
    for condition, group in data.groupby('condition_id', sort=True):
        if condition not in frozen or not frozen[condition]['admitted']:
            raise ValueError(f'UNREGISTERED_OR_NOT_ADMITTED_CONDITION: {condition}')
        item = frozen[condition]
        for field in ('bank', 'distance', 'region'):
            if str(group.iloc[0][field]) != str(item[field]):
                raise ValueError(f'FROZEN_METADATA_MISMATCH: {condition}: {field}')
        for field in ('start_cell', 'goal_cell'):
            if group.iloc[0][field] != cell_string(item[field]):
                raise ValueError(f'FROZEN_CELL_MISMATCH: {condition}: {field}')
        if json.loads(group.iloc[0]['firsts']) != [[int(v) for v in cell] for cell in item['firsts']]:
            raise ValueError(f'FROZEN_FIRST_CHANNEL_MISMATCH: {condition}')
        if 'role' in group and group.iloc[0]['role'] != item['role']:
            raise ValueError(f'FROZEN_ROLE_MISMATCH: {condition}')
    admitted_conditions = {condition for condition, item in frozen.items() if item['admitted']}
    main_data = data.loc[data['is_main'] & data['model_complete']].copy()
    auxiliary = data.loc[~data['is_main'] & data['model_complete']].copy()
    common, support = get_common_blocks(main_data)
    contrast, clusters, verdict = estimate_contrasts(common, support)
    summary = distance_table(data)
    controls = region_and_tie_controls(main_data, auxiliary)
    region_distance, region_contrast, regions_with_common = region_trends(main_data)
    shared_goal = same_goal_controls(main_data)
    guard()
    save_table(ROOT / 'DISTANCE_CONTRASTS.csv', contrast)
    save_table(result / 'DISTANCE_START_CLUSTERS.csv', clusters)
    save_table(result / 'DISTANCE_COMMON_SUPPORT.csv', support)
    save_table(result / 'DISTANCE_CURVES.csv', summary)
    save_table(ROOT / 'REGION_TIE_CONTROLS.csv', controls)
    save_table(result / 'REGION_DISTANCE_CURVES.csv', region_distance)
    save_table(result / 'REGION_NEAR_FAR_CONTRASTS.csv', region_contrast)
    save_table(result / 'SAME_GOAL_DIFFERENT_DISTANCE.csv', shared_goal)
    save_table(result / 'BEHAVIOR_COUNTEREXAMPLES.csv', counterexamples(data))
    stratification = main_data.groupby(
        ['start_cell', 'channel', 'distance', 'region', 'seed', 'variant'], sort=True).agg(
        CBR25=('correct', 'mean'), conditions=('condition_id', 'nunique'),
        mean_action_norm=('mean_action_norm', 'mean'),
        actual_reversals=('displacement_reversals', 'mean'),
        backtracking_fraction=('backtracking_evidence', 'mean')).reset_index()
    save_table(result / 'FULL_REGION_CHANNEL_STRATA.csv', stratification)
    if not summary.empty:
        plot_distance(summary, figures)
    if not main_data.empty:
        plot_regions(main_data, figures)
    plot_morphology(data, figures)
    guard()
    verdict.update(
        input_path=str(input_path), input_sha256=sha(input_path),
        scene_bank_path=str(bank_path), scene_bank_sha256=sha(bank_path), bank_top_level_length=bank_length,
        measured_rows=len(data), measured_conditions=int(data['condition_id'].nunique()),
        admitted_frozen_conditions=len(admitted_conditions),
        unmeasured_admitted_conditions=sorted(admitted_conditions - set(data['condition_id'])),
        model_incomplete_conditions=sorted(data.loc[~data['model_complete'], 'condition_id'].unique()),
        main_measured_conditions=int(main_data['condition_id'].nunique()),
        auxiliary_measured_conditions=int(auxiliary['condition_id'].nunique()),
        regions_with_within_region_near_far_common_support=regions_with_common,
        at_least_two_region_trends_identifiable=len(regions_with_common) >= 2,
        interpretation_limits=[
            'Fixed previously observed models and at most eight starting-cell clusters.',
            'BFS distance, region and exact goal identity are constrained by map support.',
            'Same-goal controls change starting physical states; tie controls change exact goals.',
            'No formal regression coefficient or regional causal mechanism is inferred.',
            'All near failures, far successes, A failures and model-incomplete conditions remain in exports.',
        ])
    write(ROOT / 'receipts' / 'BEHAVIOR_ANALYSIS.json', clean(verdict))
    print(json.dumps(clean(dict(status='BEHAVIOR_ANALYSIS_COMPLETE', verdict=verdict['verdict'],
                               measured_rows=len(data), common_starts=verdict['starting_clusters'],
                               seed_effects=[dict(seed=x['seed'], gap_near_pp=x.get('gap_near_pp'),
                                                 gap_far_pp=x.get('gap_far_pp'),
                                                 interaction_pp=x.get('distance_interaction_pp'),
                                                 operational_gate_pass=x['operational_gate_pass']) for x in contrast])),
                     allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
