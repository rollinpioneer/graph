"""Deterministic canonical-chain linearization and reward helpers."""
import numpy as np

def rank_map(sequence):
    return {node: i for i, node in enumerate(sequence)}

def progress(node, sequence):
    ranks = rank_map(sequence)
    return ranks.get(node, 0) / max(1, len(sequence) - 1)

def _node_at(annotation, step):
    for interval in annotation.get('node_intervals', []):
        if interval['start_step'] <= step <= interval['end_step']:
            return interval
    return annotation.get('node_intervals', [{}])[-1]

def compute_within_node_phi(step, node_interval, progress_anchors=None):
    start, end = int(node_interval['start_step']), int(node_interval['end_step'])
    if progress_anchors:
        for anchor in progress_anchors:
            if anchor.get('node_id') == node_interval.get('node_id') and int(anchor.get('step', -1)) == int(step):
                return float(anchor.get('value', 0.0))
    return float(np.clip((int(step) - start) / max(1, end - start), 0.0, 1.0))

def project_to_linear_progress(node_id, phi, chain, state=None):
    """Project an on-chain node, or retain the last canonical rank for off-chain nodes."""
    ranks = rank_map(chain)
    if node_id in ranks:
        return (ranks[node_id] + float(np.clip(phi, 0.0, 1.0))) / max(1, len(chain) - 1)
    last = 0 if state is None else int(state.get('last_valid_rank', 0))
    return last / max(1, len(chain) - 1)

def build_stepwise_runtime_labels(annotation, runtime_patch=None, chain=None):
    """Return one causal label row per annotated step."""
    if chain is None:
        if isinstance(runtime_patch, (list, tuple)): chain = list(runtime_patch)
        elif isinstance(runtime_patch, dict): chain = runtime_patch.get('canonical_nodes') or runtime_patch.get('chain') or []
        else: chain = []
    rows, last_rank = [], 0
    intervals = annotation.get('node_intervals', [])
    max_step = max((int(x['end_step']) for x in intervals), default=-1)
    for step in range(max_step + 1):
        it = _node_at(annotation, step); node = it.get('node_id')
        if node in chain: last_rank = rank_map(chain)[node]
        phi = compute_within_node_phi(step, it, annotation.get('progress_anchors'))
        p = project_to_linear_progress(node, phi, chain, {'last_valid_rank': last_rank}) if chain else phi
        rows.append({'step': step, 'node_id_runtime': node, 'within_node_phi': phi,
                     'stage_rank': last_rank, 'progress': float(p)})
    return rows

def compute_reward_delta(progress_values):
    p = np.asarray(progress_values, dtype=np.float32)
    return np.diff(p, prepend=p[:1])
