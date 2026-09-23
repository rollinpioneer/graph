"""Identity-safe empty-patch accounting for contract nominal patches."""
from __future__ import annotations

from .common import DataIntegrityError
from .graph import four_views


def contract_by_id(snapshot, cid):
    return next(c for c in snapshot.template.contracts if c.id == cid)


def nominal_nonempty(snapshot, cid, edges=None):
    contract = contract_by_id(snapshot, cid)
    prior = snapshot.prior_edges if edges is None else edges
    k, h, ki, hi = four_views(snapshot.template, snapshot.facts.values, prior, contract)
    return ki.values != k.values


def legal_nonempty_indices(snapshot):
    legal = [i for i, m in enumerate(snapshot.mask) if m]
    nonempty = [i for i in legal if nominal_nonempty(snapshot, snapshot.candidate_ids[i])]
    nonempty_ids = {snapshot.candidate_ids[i] for i in nonempty}
    if nonempty_ids != {snapshot.candidate_ids[i] for i in nonempty}:
        raise DataIntegrityError("nonempty index/id set mismatch")
    return legal, nonempty, nonempty_ids


def selected_action_empty_patch(cid, idx, snapshot, nonempty_idx, nonempty_ids=None):
    if cid is None or idx is None:
        return None
    if not (0 <= idx < len(snapshot.candidate_ids)):
        raise DataIntegrityError("selected index out of range")
    if snapshot.candidate_ids[idx] != cid:
        raise DataIntegrityError("selected cid/index mismatch: cid=%s idx=%s got=%s" % (cid, idx, snapshot.candidate_ids[idx]))
    ids = nonempty_ids if nonempty_ids is not None else {snapshot.candidate_ids[i] for i in nonempty_idx}
    empty_by_idx = idx not in nonempty_idx
    empty_by_id = cid not in ids
    if empty_by_idx != empty_by_id:
        raise DataIntegrityError("empty-patch identity mismatch cid=%s idx=%s" % (cid, idx))
    return bool(empty_by_idx)
