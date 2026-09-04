"""Data loading and leakage-safe feature/window helpers for Stage 3."""
import csv, json
from typing import Iterable
import numpy as np
from pathlib import Path

def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

FIELDS = ('eef_pos', 'object_pos', 'target_pos', 'gripper_state', 'action')

def load_episode(episode_row):
    """Load one raw JSON episode using the resolved source path in its index row."""
    path = episode_row.get('resolved_source_path') or episode_row.get('source_path')
    if not path:
        raise ValueError('episode row has no source path')
    return json.loads(Path(path).read_text())

def _state_vector(state, fields: Iterable[str] = FIELDS):
    values = []
    for key in fields:
        value = state.get(key, [])
        values.extend(np.asarray(value, dtype=np.float32).reshape(-1).tolist())
    info = state.get('info') or {}
    values.extend([float(bool(info.get('subgoal_A_done', False))),
                   float(bool(info.get('subgoal_B_done', False)))])
    return values

def build_feature_matrix(episode, forbidden_fields=None):
    """Return float32 current-state features; outcome/IDs are never read as features."""
    forbidden_fields = set(forbidden_fields or ())
    if forbidden_fields & {'outcome', 'success', 'info.success', 'scenario',
                           'controller_source', 'episode_id'}:
        # These are metadata names, not feature names; accepting them documents the
        # caller's exclusion contract without allowing them into the vector.
        pass
    return np.asarray([_state_vector(s) for s in episode['states']], dtype=np.float32)

def load_runtime_annotation(episode_id, annotation_path=None):
    """Load a runtime annotation by episode id from a JSONL annotation file."""
    if annotation_path is None:
        raise ValueError('annotation_path is required')
    for row in read_jsonl(annotation_path):
        if row.get('episode_id') == episode_id:
            return row
    raise KeyError(episode_id)

def window_sequence(features, history_steps):
    """Build left-padded, causal windows (no future frames)."""
    x = np.asarray(features, dtype=np.float32)
    h = int(history_steps)
    if x.ndim != 2 or h < 1:
        raise ValueError('features must be [steps, dim] and history_steps >= 1')
    out = np.empty((len(x), h, x.shape[1]), dtype=np.float32)
    for i in range(len(x)):
        start = max(0, i - h + 1)
        chunk = x[start:i + 1]
        out[i] = np.pad(chunk, ((h - len(chunk), 0), (0, 0)), mode='constant')
    return out
