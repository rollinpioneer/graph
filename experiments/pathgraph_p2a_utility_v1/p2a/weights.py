"""One declared training-only weight transform for every reward method."""
from __future__ import annotations
import numpy as np

METHODS=('BC_UNIFORM','SPARSE_TERMINAL','LINEAR_A_FIRST_R1','LINEAR_B_FIRST_R1',
         'UNORDERED_VALID_COUNT','VALID_COUNT_PLUS_MATCHED_EVENTS_V1',
         'V6_CAP_COST_ONLY','V6_CAP_POTENTIAL','V6_WEIGHT_SHUFFLED')

def make_weights(rewards, *, floor=.2, uniform=False):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('finite nonempty reward vector required')
    if not 0<floor<=1: raise ValueError('floor must be in (0,1]')
    nonzero=np.abs(r[np.abs(r)>1e-12])
    scale=float(np.quantile(nonzero,.95)) if len(nonzero) else 1.
    raw=np.ones_like(r) if uniform else floor+(1-floor)*np.clip(np.maximum(r,0)/scale,0,1)
    mean=float(raw.mean()); w=raw/mean
    report=dict(scale=scale,raw_mean=mean,floor=floor,mean=float(w.mean()),
        min=float(w.min()),max=float(w.max()),ess=float(w.sum()**2/(w@w)),
        signed_return=float(r.sum()),positive_mass=float(np.maximum(r,0).sum()),
        all_zero_signal=bool(not len(nonzero)),uniform=uniform)
    return w.astype(np.float32),report

def shuffled_weights(weights, seed=20260915):
    w=np.asarray(weights,dtype=np.float32)
    return w[np.random.default_rng(seed).permutation(len(w))].copy()
