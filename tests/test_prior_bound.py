import pytest,math

@pytest.mark.torch_runtime
def test_T21_bound_probability_ratio():
    import torch
    from cp_disr.neural import AnchoredPrior
    torch.manual_seed(11);prior=AnchoredPrior();c=torch.randn(512);dp=torch.randn(4,128)
    with torch.no_grad():prior.final.weight.fill_(100.)
    deltas=torch.stack([prior(c,x*dp,.5)[1] for x in [0.,1.,-1.,100.]])
    assert torch.max(torch.abs(deltas))<=.5+1e-7 and deltas[0]==0
    b=torch.tensor([-2.,-.1,.4,1.]);ratio=torch.softmax(b+deltas,0)/torch.softmax(b,0)
    assert torch.all(ratio>=math.exp(-1)-1e-6) and torch.all(ratio<=math.exp(1)+1e-6)
