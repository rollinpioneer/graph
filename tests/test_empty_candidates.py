import pytest
from dataclasses import replace
from cp_disr.contracts import Effects,nominal_overlay

@pytest.mark.pure
def test_T24_wait_empty_patch(fixture):
    c=next(c for c in fixture['contracts'] if c.name=='OPEN');wait=replace(c,name='WAIT',pre_pos=(),effects=Effects(),timeout_seconds=1.)
    assert dict(nominal_overlay(wait,fixture['facts'].values))==dict(fixture['facts'].values)
    assert wait.timeout_seconds>0

@pytest.mark.torch_runtime
def test_T24_empty_mask_and_only_wait(snap,policy):
    import torch
    out=policy(replace(snap,mask=tuple(False for _ in snap.mask)))
    assert out.distribution is None and out.ended_reason=='NO_SAFE_CANDIDATES'
    with pytest.raises(ValueError):out.select()
    # A registered bounded WAIT uses a no-effect contract, not a fabricated reward transition.
    c=replace(snap.template.contracts[0],name='WAIT',pre_pos=(),pre_neg=(),effects=Effects(),timeout_seconds=1.)
    old=snap.template.contracts[0]
    # Keep schema embedding identity in this fixture as an equivalent registered no-effect skill.
    c=replace(c,name=old.name)
    template=replace(snap.template,contracts=(c,)+snap.template.contracts[1:])
    out=policy(replace(snap,template=template,mask=(True,)+(False,)*(len(snap.mask)-1)))
    assert out.distribution.probs[0]==1 and all(torch.count_nonzero(d.dk)==0 for d in out.diagnostics['differences'].values())
