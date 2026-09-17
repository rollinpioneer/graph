"""Load registered contracts. Training/validation paths never open test payloads."""
from __future__ import annotations
from pathlib import Path
from .io_utils import load_json
from .release import assert_no_test_payload, ReleaseRejected

def load_families(data_root):
    return load_json(Path(data_root) / "dataset" / "families.json")["families"]

def contracts_for_split(data_root, split, draw=None):
    if split == "test":
        raise ReleaseRejected("TEST_CONTAMINATION_PROTOCOL_VIOLATION", "split=test")
    from .qualify import load_family_contracts
    wanted = []
    for fam in load_families(data_root):
        if fam["split"] != split:
            continue
        assert_no_test_payload(fam)
        left, right = load_family_contracts(Path(data_root) / "dataset", fam)
        assert_no_test_payload(left)
        assert_no_test_payload(right)
        wanted.extend([left, right])
    if split.startswith("train") and draw is not None:
        # train_A/B/C already selected by split name
        pass
    if not wanted:
        raise RuntimeError(f"no contracts for {split}")
    return wanted

def train_split_name(draw):
    return f"train_{draw}"

def load_test_contracts(data_root, *, allow=False):
    if not allow:
        raise ReleaseRejected("TEST_CONTAMINATION_PROTOCOL_VIOLATION", "test load blocked")
    from .qualify import load_family_contracts
    out = []
    for fam in load_families(data_root):
        if fam["split"] != "test":
            continue
        left, right = load_family_contracts(Path(data_root) / "dataset", fam)
        out.extend([left, right])
    return out

def load_job_contracts(data_root, job):
    if job.get('kind') == 'smoke':
        return contracts_for_split(data_root, 'integration')
    draw = job.get('draw')
    if not draw:
        raise RuntimeError('formal job missing draw')
    return contracts_for_split(data_root, train_split_name(draw))
