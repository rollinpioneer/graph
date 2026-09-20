import pytest
from dataclasses import replace
from cp_disr.rl import remap_stored

@pytest.mark.torch_runtime
def test_T10_mask_identity(snap,policy):
    import torch
    a=policy(snap);perm=tuple(reversed(range(len(snap.mask))))
    changed=replace(snap,candidate_ids=tuple(snap.candidate_ids[i] for i in perm),mask=tuple(snap.mask[i] for i in perm),candidate_features=tuple(snap.candidate_features[i] for i in perm))
    b=policy(changed);remap_stored(snap,changed.candidate_ids,changed.mask)
    for cid in snap.candidate_ids:
        assert torch.allclose(a.distribution.log_prob(torch.tensor(a.candidate_ids.index(cid))),b.distribution.log_prob(torch.tensor(b.candidate_ids.index(cid))),atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError):remap_stored(snap,snap.candidate_ids,(False,)+snap.mask[1:])
