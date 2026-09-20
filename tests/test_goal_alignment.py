import pytest
from dataclasses import replace

@pytest.mark.torch_runtime
def test_T05_goals_and_padding(fixture,policy):
    import torch
    from cp_disr.graph import view
    template=fixture['template'];g=view(template,fixture['facts'].values);z=policy.encoder(g)
    perm=(2,0,1);other=replace(template,goals=tuple(template.goals[i] for i in perm));zp=policy.encoder(view(other,fixture['facts'].values))
    assert len(template.goals)==3 and tuple(g.sign for g in template.goals)==(1,1,-1)
    assert torch.allclose(z[0],zp[0],atol=1e-6,rtol=1e-5)
    assert torch.allclose(z[1:][list(perm)],zp[1:],atol=1e-6,rtol=1e-5)
    context=torch.arange(384,dtype=torch.float32)/384
    padded=torch.cat((z,torch.ones(2,128)*999));mask=torch.tensor([True]*len(z)+[False]*2)
    assert torch.allclose(policy.contract(context,z),policy.contract(context,padded,mask),atol=1e-6,rtol=1e-5)
