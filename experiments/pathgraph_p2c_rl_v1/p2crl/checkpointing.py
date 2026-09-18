"""Checkpoint hashing. Zip is created only after requested==actual timesteps."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .constants_b import GRADIENT_UPDATES_SEMANTICS
from .errors import CheckpointStepMismatch
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


def _accounting(model):
    opt = int(getattr(model, "_p2crl_optimizer_step_count", getattr(model, "optimizer_updates", getattr(model, "gradient_updates", 0))) or 0)
    return {
        "rollout_quantum": getattr(model, "_p2crl_rollout_quantum", None),
        "rollout_iteration_count": int(getattr(model, "_p2crl_rollout_iteration_count", 0) or 0),
        "ppo_epoch_count": int(getattr(model, "_p2crl_ppo_epoch_count", 0) or 0),
        "optimizer_step_count": opt,
        "gradient_updates": opt,
        "gradient_updates_semantics": GRADIENT_UPDATES_SEMANTICS,
    }


def save_milestone(model, ckpt_dir, step, post_update=True):
    ckpt_dir = Path(ckpt_dir)
    requested = int(step)
    actual = int(getattr(model, "num_timesteps", 0) or 0)
    if actual != requested:
        raise CheckpointStepMismatch(f"requested={requested} actual={actual}")
    if requested == 0:
        phase = "INITIALIZATION"
        post_update = False
    else:
        phase = "POST_UPDATE"
        if not post_update:
            raise RuntimeError("checkpoint is not post-update")
        post_update = True
    path = ckpt_dir / f"policy_{requested}.zip"
    digest = save_and_hash(model, path)
    acc = _accounting(model)
    rec = {
        "schema": "P2CRL_CHECKPOINT_META_V2",
        "step": requested,
        "requested_step": requested,
        "actual_num_timesteps": actual,
        "checkpoint_phase": phase,
        "post_update": bool(post_update),
        "path": str(path),
        "zip_sha256": digest,
        "policy_state_dict_sha256": state_dict_digest(model),
        "num_timesteps": actual,
        "optimizer_update_count": acc["optimizer_step_count"],
        **acc,
    }
    write_new(path.with_name(path.name + ".meta.json"), rec)
    return rec