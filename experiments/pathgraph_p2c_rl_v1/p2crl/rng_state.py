"""RNG fingerprints for validation isolation."""
from __future__ import annotations
import hashlib, random
import numpy as np

def _digest(label, blob):
    h = hashlib.sha256()
    h.update(label.encode())
    h.update(blob)
    return h.hexdigest()

def python_rng_digest():
    return _digest("py", repr(random.getstate()).encode())

def numpy_rng_digest():
    return _digest("np", repr(np.random.get_state()).encode())

def torch_rng_digest():
    try:
        import torch
        return _digest("th", repr(tuple(int(x) for x in torch.random.get_rng_state().tolist()[:16])).encode())
    except Exception:
        return "torch_unavailable"

def capture(model=None, extra=None):
    rec = {
        "python": python_rng_digest(),
        "numpy": numpy_rng_digest(),
        "torch": torch_rng_digest(),
        "num_timesteps": int(getattr(model, "num_timesteps", 0) or 0),
    }
    if extra:
        rec.update(extra)
    return rec
