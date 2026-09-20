import pytest
from dataclasses import replace

@pytest.mark.torch_runtime
def test_T18_current_parameter_recompute(snap,policy,fixture,monkeypatch):
    import torch
    from cp_disr.rl import Transition
    from cp_disr.torch_rl import recompute_transition
    from cp_disr.prior import PriorSampler
    from cp_disr.common import digest
    import socket
    def prohibited_network(*args,**kwargs):raise AssertionError('API/network during PPO recomputation')
    monkeypatch.setattr(socket,'create_connection',prohibited_network)
    n=replace(snap,decision_id=1);t=Transition(snap,n,snap.candidate_ids[0],-.5,0.,0.,0.,1.,1.,False,False,'synthetic',())
    sampler=PriorSampler(0);draws=sampler.draw_count;before_calls=policy.encoder.forward_calls;prior=digest(snap.prior_edges)
    outputs=[];opt=torch.optim.Adam(policy.parameters(),lr=1e-4)
    for epoch in range(4):
        out=recompute_transition(policy,t);outputs.append(out.value.detach().clone())
        if epoch==0:
            opt.zero_grad();(out.value-1).square().backward();opt.step()
    assert policy.encoder.forward_calls>=before_calls+4
    assert not torch.equal(outputs[0],outputs[1])
    assert sampler.draw_count==draws==0 and digest(snap.prior_edges)==prior
    assert 'dk' not in t.__dict__ and 'dp' not in t.__dict__
