"""Checkpoint hashing. Does not load untrusted pickles."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .io_utils import sha256_file, write_new

def state_dict_digest(model):
    import numpy as np
    h = hashlib.sha256()
    sd = model.policy.state_dict()
    for k, v in sorted(sd.items()):
        if hasattr(v, 'detach'):
            a = v.detach().cpu().contiguous().numpy()
        else:
            a = np.asarray(v)
        h.update(str(k).encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save_and_hash(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    model.save(str(path))
    digest = sha256_file(path)
    write_new(path.with_suffix(path.suffix + ".sha256.json"), {"path": str(path), "sha256": digest})
    return digest

def save_milestone(model, ckpt_dir, step, post_update=True):
    ckpt_dir = Path(ckpt_dir)
    path = ckpt_dir / f'policy_{int(step)}.zip'
    digest = save_and_hash(model, path)
    rec = {
        'step': int(step),
        'post_update': bool(post_update),
        'path': str(path),
        'zip_sha256': digest,
        'policy_state_dict_sha256': state_dict_digest(model),
        'num_timesteps': int(getattr(model, 'num_timesteps', 0) or 0),
        'optimizer_update_count': int(getattr(model, 'optimizer_updates', getattr(model, 'gradient_updates', 0)) or 0),
    }
    write_new(path.with_name(path.name + '.meta.json'), rec)
    if not rec['post_update'] and int(step) != 0:
        raise RuntimeError('checkpoint is not post-update')
    return rec
