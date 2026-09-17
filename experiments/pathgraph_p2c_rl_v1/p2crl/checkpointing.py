"""Checkpoint hashing. Does not load untrusted pickles."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .io_utils import sha256_file, write_new

def state_dict_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.tobytes())
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
