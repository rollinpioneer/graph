import pytest

@pytest.mark.torch_runtime
def test_T12_stop_gradient():
    import torch
    from cp_disr.torch_rl import q_targets,value_targets
    old=torch.tensor([2.],requires_grad=True);r=torch.tensor([.3]);g=torch.tensor([.9]);term=torch.tensor([False])
    y=q_targets(r,g,term,old);v=value_targets(old,torch.tensor([.2],requires_grad=True))
    assert not y.requires_grad and y.grad_fn is None and not v.requires_grad
    q=torch.tensor([0.],requires_grad=True);(q-y).square().sum().backward();assert old.grad is None and q.grad is not None
