import pytest
from dataclasses import replace

@pytest.mark.torch_runtime
def test_T25_ablation_switches(snap,policy):
    import torch
    from cp_disr.common import digest
    full=policy(snap);policy.method='A_Q';aq=policy(snap)
    assert policy.q_coefficient==0 and torch.equal(full.logits,aq.logits) and torch.equal(full.q,aq.q)
    policy.method='A_DD';dd=policy(snap)
    for cid,d in dd.diagnostics['differences'].items():assert torch.equal(dd.diagnostics['prior_inputs'][cid],d.dh)
    no=policy(replace(snap,prior_edges=(),prior_hash=digest(())))
    assert all(torch.count_nonzero(v)==0 for v in no.diagnostics['prior_inputs'].values())
    policy.method='A_B';ab=policy(snap)
    for cid in ab.diagnostics['delta']:
        s=policy.prior.final(ab.diagnostics['up'][cid]).squeeze()
        assert torch.allclose(ab.diagnostics['delta'][cid],policy.B*s)
        assert torch.allclose(full.diagnostics['delta'][cid],policy.B*torch.tanh(s))
    for method in ['B0','B1','B2']:
        policy.method=method;before=policy.encoder.forward_calls;out=policy(snap)
        assert torch.isfinite(out.value) and torch.isfinite(out.logits[out.mask]).all()
        if method=='B0':assert policy.encoder.forward_calls==before
        if method=='B1':assert out.diagnostics['differences']=={}
