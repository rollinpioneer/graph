"""Frozen original-action quality scoring. No evaluation truth is an input."""
from pathlib import Path
import hashlib
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fused_score(original_score, quality):
    s = np.asarray(original_score, dtype=np.float64)
    q = np.asarray(quality, dtype=np.float64)
    if s.shape != (1024,) or q.shape != s.shape:
        raise ValueError('Expected all original 1024 candidates before filtering')
    if not (np.isfinite(s).all() and np.isfinite(q).all()):
        raise ValueError('Nonfinite score is a technical error')
    ss, qs = s.std(ddof=0), q.std(ddof=0)
    fallback = bool(ss < 1e-6 or qs < 1e-6)
    score = s.copy() if fallback else .5*(s-s.mean())/ss + .5*(q-q.mean())/qs
    return score, dict(fallback=fallback, original_std=float(ss), quality_std=float(qs))


def replace_score(original, new_score):
    original = np.asarray(original)
    if original.shape != (1024, 17) or not np.isfinite(original).all():
        raise ValueError('Expected original finite 1024x17 grasps')
    score = np.asarray(new_score)
    if score.shape != (1024,) or not np.isfinite(score).all():
        raise ValueError('Invalid replacement score')
    rows = original.astype(np.float64).copy()
    rows[:, 0] = score
    assert np.array_equal(rows[:, 1:], original[:, 1:])
    return rows


def cache_inputs(path, layout='nqs_v1'):
    """Read only prediction/features, never labels or object identities."""
    with np.load(path, allow_pickle=False) as cache:
        pred = (cache['original_pred'] if 'original_pred' in cache else cache['pred']).copy()
        if layout == 'lqr_v3':
            feature = cache['feature320'].copy()
            action = cache['original_action'].copy()
        elif layout == 'nqs_v1':
            old = cache['feature']
            feature = np.concatenate([old[:, 64:320], old[:, :64]], axis=1)
            action = old[:, 320:324].copy()
        else:
            raise ValueError(layout)
        identity = str(cache['input_sha256']) if 'input_sha256' in cache else None
    return pred, feature, action, identity


class FrozenScorer:
    def __init__(self, checkpoint_path, expected_sha256=None):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.source_sha256 = sha256(checkpoint_path)
        if expected_sha256 and self.source_sha256 != expected_sha256:
            raise ValueError('Frozen head SHA256 mismatch')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model = torch.nn.Sequential(torch.nn.Linear(324, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1), torch.nn.Sigmoid()).eval()
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.requires_grad_(False)
        self.mean = np.asarray(checkpoint['mean']).astype(np.float32, copy=True)
        self.std = np.asarray(checkpoint['std']).astype(np.float32, copy=True)
        assert self.mean.shape == (320,) and self.std.shape == (320,)
        assert np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std > 0).all()
        assert sum(p.numel() for p in self.model.parameters()) == 41729

    def quality(self, feature320, original_action):
        feature = np.asarray(feature320, dtype=np.float32)
        action = np.asarray(original_action, dtype=np.float32)
        if feature.shape != (1024, 320) or action.shape != (1024, 4):
            raise ValueError('Feature order must be group256, score64, original-action4')
        if not (np.isfinite(feature).all() and np.isfinite(action).all()):
            raise ValueError('Nonfinite input')
        x = np.concatenate([(feature-self.mean)/self.std, action], axis=1).copy()
        with self.torch.no_grad():
            q = self.model(self.torch.from_numpy(x)).squeeze(-1).numpy()
        assert q.shape == (1024,) and np.isfinite(q).all()
        return q

    def apply(self, original, feature320, original_action, family='S'):
        q = self.quality(feature320, original_action)
        info = dict(family=family, head_sha256=self.source_sha256)
        if family == 'R0':
            score = original[:, 0]
        elif family == 'S':
            score = q
        elif family == 'F05':
            score, details = fused_score(original[:, 0], q)
            info.update(details)
        else:
            raise ValueError(family)
        return replace_score(original, score), q, info

    def save(self, path):
        """Portable bundle preserves learned parameters and fit statistics."""
        self.torch.save(dict(model=self.model.state_dict(), mean=self.mean, std=self.std, source_head_sha256=self.source_sha256, feature_order='group256_score64_action4', format='EG_LQR_V3'), path)
