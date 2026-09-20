import pytest

@pytest.mark.torch_runtime
def test_T11_executed_only():
    import torch
    from cp_disr.torch_rl import q_loss
    q=torch.zeros(2,3,requires_grad=True);q_loss(q,torch.tensor([0,2]),torch.tensor([.5,1.])).backward()
    assert q.grad[0,0]!=0 and q.grad[1,2]!=0
    assert torch.equal(q.grad[torch.tensor([[False,True,True],[True,True,False]])],torch.zeros(4))
