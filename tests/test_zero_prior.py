from dataclasses import replace
import pytest

@pytest.mark.torch_runtime
def test_T01_dp_zero(fixture,policy):
    import torch
    from cp_disr.graph import four_views
    from cp_disr.neural import differences
    assert 6<=len(fixture['template'].nodes)<=10
    for c in fixture['contracts'][:2]:
        graphs=four_views(fixture['template'],fixture['facts'].values,(),c);d=differences(policy.encoder,graphs)
        assert d.encodings[0] is d.encodings[1] and d.encodings[2] is d.encodings[3]
        assert torch.equal(d.dp,torch.zeros_like(d.dp)) and d.dp.dtype==torch.float32
        assert d.dk.shape==d.dh.shape==d.dp.shape

@pytest.mark.torch_runtime
def test_T02_residual_zero(snap,policy):
    import torch
    from cp_disr.common import digest
    out=policy(replace(snap,prior_edges=(),prior_hash=digest(())))
    assert all(torch.equal(x,torch.zeros_like(x)) for x in out.diagnostics['up'].values())
    assert all(x.item()==0 for x in out.diagnostics['delta'].values())
    policy.method='B2';control=policy(replace(snap,prior_edges=(),prior_hash=digest(())))
    assert torch.allclose(out.distribution.probs,control.distribution.probs,atol=1e-6,rtol=1e-5)
