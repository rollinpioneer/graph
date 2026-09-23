"""Production empty-patch identity regressions for Stage 0D accounting."""
from __future__ import annotations

from pathlib import Path

import pytest

from cp_disr.fixtures import build_fixture, snapshot
from cp_disr.patch_identity import (
    legal_nonempty_indices,
    nominal_nonempty,
    selected_action_empty_patch,
)


@pytest.fixture
def fx():
    return build_fixture(Path(__file__).parent / "fixtures")


@pytest.mark.pure
def test_no_prior_can_have_nonempty_contract_patch(fx):
    snap = snapshot(fx, prior=False)
    legal, nonempty, ids = legal_nonempty_indices(snap)
    assert legal
    assert nonempty, "empty prior must not imply empty contract patch"
    for i in nonempty:
        assert nominal_nonempty(snap, snap.candidate_ids[i], edges=())


@pytest.mark.pure
def test_prior_present_empty_contract_patch_is_distinct(fx):
    snap = snapshot(fx, prior=True)
    legal, nonempty, ids = legal_nonempty_indices(snap)
    empty_idx = [i for i in legal if i not in nonempty]
    assert snap.prior_edges, "this case requires a nonempty source prior"
    assert empty_idx or nonempty
    # Empty patch and empty prior are different flags.
    assert (len(snap.prior_edges) == 0) is False
    if empty_idx:
        cid = snap.candidate_ids[empty_idx[0]]
        assert selected_action_empty_patch(cid, empty_idx[0], snap, nonempty, ids) is True
        assert not nominal_nonempty(snap, cid)


@pytest.mark.pure
def test_no_prior_empty_contract_patch_is_not_missing_cache(fx):
    snap = snapshot(fx, prior=False)
    legal, nonempty, ids = legal_nonempty_indices(snap)
    empty_idx = [i for i in legal if i not in nonempty]
    assert snap.prior_edges == ()
    if empty_idx:
        cid = snap.candidate_ids[empty_idx[0]]
        assert selected_action_empty_patch(cid, empty_idx[0], snap, nonempty, ids) is True
    # Missing cache is not represented by prior_edges==(); callers must audit source files.


@pytest.mark.pure
def test_permutation_index_and_canonical_id_agree(fx):
    snap = snapshot(fx, prior=False)
    legal, nonempty, ids = legal_nonempty_indices(snap)
    order = list(range(len(snap.candidate_ids)))[::-1]
    from dataclasses import replace
    perm_ids = tuple(snap.candidate_ids[i] for i in order)
    perm_mask = tuple(snap.mask[i] for i in order)
    perm_feat = tuple(snap.candidate_features[i] for i in order)
    perm = replace(snap, candidate_ids=perm_ids, mask=perm_mask, candidate_features=perm_feat)
    legal2, nonempty2, ids2 = legal_nonempty_indices(perm)
    assert ids2 == ids
    for cid in perm.candidate_ids:
        idx = perm.candidate_ids.index(cid)
        empty = selected_action_empty_patch(cid, idx, perm, nonempty2, ids2)
        orig_idx = snap.candidate_ids.index(cid)
        orig_empty = selected_action_empty_patch(cid, orig_idx, snap, nonempty, ids)
        assert empty is orig_empty
    # String cid must never be compared against integer index lists.
    cid0 = snap.candidate_ids[0]
    assert (cid0 not in nonempty) in (True, False)
    # The buggy comparison is always True for a string vs int list.
    assert (cid0 not in nonempty) is True
    assert selected_action_empty_patch(cid0, 0, snap, nonempty, ids) is (0 not in nonempty)
